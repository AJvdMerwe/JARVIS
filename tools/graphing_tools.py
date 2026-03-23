"""
tools/graphing_tools.py
─────────────────────────
Rich data-visualisation tools for the GraphingAgent and other specialist agents.

All charts are saved to data/uploads/generated/charts/ as PNG files.
Every tool returns the saved file path so agents can cite it in responses
and the UI can render it inline.

Tools
─────
  QuickChartTool        — one-shot chart from a description + data snippet
  TimeSeriesChartTool   — line/area charts for temporal data
  BarChartTool          — grouped/stacked/horizontal bar charts
  ScatterPlotTool       — scatter plots with optional colour/size encoding
  DistributionChartTool — histogram, KDE, box plot, violin plot
  CorrelationHeatmapTool — correlation matrix heatmap
  CompositeChartTool    — multi-panel / dashboard layout (2–4 charts)
  StyleConfigTool       — set global theme, palette, figure size
  ListChartsTool        — list recently generated chart files

Design principles
─────────────────
  • All tools use matplotlib with Agg backend (no display required).
  • seaborn is used where it adds value (statistical plots, heatmaps).
  • Every chart uses a dark theme consistent with the Jarvis UI.
  • Titles, axis labels, and legends are always set from tool inputs.
  • Charts are always 120 dpi, tight_layout, saved as PNG.
  • Generated filenames are sanitised and counter-deduplicated.
"""
from __future__ import annotations

import io
import logging
import re
import textwrap
import traceback
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings
from core.llm_manager import get_llm

logger = logging.getLogger(__name__)

# ── Output directory ──────────────────────────────────────────────────────────
CHART_DIR = settings.uploads_path / "generated" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style state ────────────────────────────────────────────────────────
_STYLE: dict = {
    "theme":   "dark",
    "palette": "jarvis",   # custom dark-mode palette
    "figsize": (10, 6),
    "dpi":     120,
}

# Jarvis brand palette — consistent with the UI CSS variables
_PALETTES: dict[str, list[str]] = {
    "jarvis":   ["#6c63ff", "#22c55e", "#f59e0b", "#38bdf8",
                 "#ef4444", "#a78bfa", "#fb923c", "#34d399",
                 "#f472b6", "#60a5fa"],
    "cool":     ["#38bdf8", "#6c63ff", "#a78bfa", "#34d399", "#60a5fa"],
    "warm":     ["#f59e0b", "#fb923c", "#ef4444", "#f472b6", "#fbbf24"],
    "neutral":  ["#94a3b8", "#64748b", "#475569", "#334155", "#1e293b"],
    "green":    ["#22c55e", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5"],
    "seaborn":  None,   # delegate entirely to seaborn
}

_DARK_BG   = "#0d0f14"
_DARK_BG2  = "#13161f"
_DARK_TEXT = "#e2e8f0"
_DARK_GRID = "#2a2f4a"


def _apply_dark_style() -> None:
    """Apply Jarvis dark theme to the current matplotlib figure."""
    plt.rcParams.update({
        "figure.facecolor":  _DARK_BG,
        "axes.facecolor":    _DARK_BG2,
        "axes.edgecolor":    _DARK_GRID,
        "axes.labelcolor":   _DARK_TEXT,
        "axes.grid":         True,
        "grid.color":        _DARK_GRID,
        "grid.linewidth":    0.6,
        "grid.alpha":        0.5,
        "xtick.color":       _DARK_TEXT,
        "ytick.color":       _DARK_TEXT,
        "text.color":        _DARK_TEXT,
        "legend.facecolor":  _DARK_BG2,
        "legend.edgecolor":  _DARK_GRID,
        "legend.labelcolor": _DARK_TEXT,
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titlepad":     10,
        "axes.titleweight":  "bold",
    })


def _colors(n: int = 10) -> list[str]:
    """Return n colors from the active palette."""
    pal = _PALETTES.get(_STYLE["palette"])
    if pal is None:
        import seaborn as sns
        return [c for c in sns.color_palette("tab10", n)]
    return (pal * ((n // len(pal)) + 1))[:n]


def _safe_name(name: str, ext: str = ".png") -> Path:
    stem    = re.sub(r"[^\w\-]", "_", Path(name).stem.strip())[:60] or "chart"
    path    = CHART_DIR / f"{stem}{ext}"
    counter = 1
    while path.exists():
        path = CHART_DIR / f"{stem}_{counter}{ext}"
        counter += 1
    return path


def _save(fig: plt.Figure, filename: str) -> str:
    """Save figure and return a human-readable result string."""
    path = _safe_name(filename)
    fig.savefig(path, dpi=_STYLE["dpi"], bbox_inches="tight",
                facecolor=_DARK_BG, edgecolor="none")
    plt.close(fig)
    size_kb = path.stat().st_size / 1024
    logger.info("Chart saved: %s (%.1f KB)", path, size_kb)
    return f"Chart saved: {path}\n({size_kb:.1f} KB — {path.name})"


def _parse_csv_data(csv_text: str) -> "pd.DataFrame":
    """Parse a small CSV string into a DataFrame."""
    import pandas as pd
    return pd.read_csv(io.StringIO(csv_text.strip()))


def _df_from_registry(name: str) -> "Optional[pd.DataFrame]":
    """Retrieve a loaded DataFrame from the DataAnalysisAgent registry."""
    try:
        from tools.data_analysis_tools import _FRAMES
        return _FRAMES.get(name)   # returns None if not found
    except Exception:
        return None


def _load_data(source: str) -> "pd.DataFrame":
    """
    Load data from: (a) registry name, (b) CSV string.
    Always returns a DataFrame or raises ValueError.
    """
    import pandas as pd
    df = _df_from_registry(source)
    if df is not None:
        return df
    # Try parsing as CSV text
    return _parse_csv_data(source)


# =============================================================================
#  Schemas
# =============================================================================

class QuickChartInput(BaseModel):
    description: str = Field(
        ...,
        description=(
            "Natural-language description of the chart you want. "
            "Example: 'Bar chart of quarterly revenue: Q1=1.2M, Q2=1.5M, Q3=1.8M, Q4=2.1M'"
        ),
    )
    chart_type: str = Field(
        "auto",
        description="Chart type hint: auto | bar | line | scatter | pie | histogram | heatmap",
    )
    title:    str = Field("",    description="Chart title. Auto-generated if blank.")
    filename: str = Field("chart.png", description="Output filename.")


class TimeSeriesInput(BaseModel):
    data_source: str = Field(
        ...,
        description=(
            "Either: (a) a CSV string with date and value columns, "
            "(b) a registered dataset name (from load_data), "
            "(c) a JSON list of {date, value} objects."
        ),
    )
    date_col:  str  = Field("date",  description="Name of the date/time column.")
    value_cols: str = Field("value", description="Comma-separated value column names to plot.")
    title:     str  = Field("Time Series", description="Chart title.")
    y_label:   str  = Field("", description="Y-axis label.")
    area:      bool = Field(False, description="Fill area under lines.")
    filename:  str  = Field("time_series.png")


class BarChartInput(BaseModel):
    categories: str = Field(..., description="Comma-separated category labels.")
    values:     str = Field(..., description="Comma-separated numeric values (or multiple series as JSON).")
    title:      str = Field("Bar Chart")
    x_label:    str = Field("")
    y_label:    str = Field("")
    horizontal: bool = Field(False, description="Horizontal bar chart.")
    stacked:    bool = Field(False, description="Stacked bar chart (when multiple series).")
    filename:   str  = Field("bar_chart.png")


class ScatterInput(BaseModel):
    data_source: str = Field(..., description="CSV string or registered dataset name.")
    x_col:  str = Field(..., description="X-axis column name.")
    y_col:  str = Field(..., description="Y-axis column name.")
    color_col:  str = Field("", description="Optional column for colour encoding.")
    size_col:   str = Field("", description="Optional column for point-size encoding.")
    title:  str = Field("Scatter Plot")
    trend:  bool = Field(False, description="Add a linear trend line.")
    filename: str = Field("scatter.png")


class DistributionInput(BaseModel):
    data_source: str = Field(..., description="CSV string or registered dataset name.")
    column:   str  = Field(..., description="Column to visualise.")
    plot_type: str = Field(
        "histogram",
        description="histogram | kde | box | violin | combined (hist+kde)",
    )
    group_col: str  = Field("", description="Optional grouping column.")
    title:     str  = Field("")
    filename:  str  = Field("distribution.png")


class HeatmapInput(BaseModel):
    data_source: str = Field(..., description="CSV string or registered dataset name.")
    columns:  str = Field("", description="Comma-separated columns to include. Blank = all numeric.")
    title:    str = Field("Correlation Heatmap")
    annotate: bool = Field(True, description="Show correlation values in cells.")
    filename: str  = Field("heatmap.png")


class CompositeInput(BaseModel):
    charts: str = Field(
        ...,
        description=(
            "JSON list of chart specs. Each spec is a dict with keys: "
            "type (bar|line|scatter|histogram|pie), title, data (CSV or dataset name), "
            "x (col), y (col). Example: "
            "[{\"type\":\"bar\",\"title\":\"Revenue\",\"data\":\"sales\",\"x\":\"quarter\",\"y\":\"revenue\"}]"
        ),
    )
    layout_title: str = Field("Dashboard")
    cols:         int = Field(2, ge=1, le=4, description="Number of columns in the grid.")
    filename:     str = Field("dashboard.png")


class StyleInput(BaseModel):
    theme:   str = Field("dark", description="dark | light")
    palette: str = Field("jarvis", description="jarvis | cool | warm | neutral | green | seaborn")
    figsize: str = Field("10x6", description="Width x Height in inches, e.g. '12x8'.")


# =============================================================================
#  Tools
# =============================================================================

class QuickChartTool(BaseTool):
    """
    Generate a chart from a natural-language description or inline data.

    The LLM describes what it wants — 'bar chart of Q1–Q4 revenue' — and
    this tool parses the description, extracts the data, and renders the
    appropriate chart type automatically.

    For more control use the specialised tools (BarChartTool, TimeSeriesChartTool, etc.).
    """
    name: str        = "quick_chart"
    description: str = (
        "Generate a chart from a natural-language description of the data. "
        "Great for quick one-shot visualisations without needing a loaded dataset. "
        "Example: 'Bar chart of quarterly revenue Q1=1.2M Q2=1.5M Q3=1.8M Q4=2.1M'. "
        "Inputs: description (str), chart_type (auto|bar|line|scatter|pie|histogram), "
        "title (str), filename (str)."
    )
    args_schema: type[BaseModel] = QuickChartInput

    def _run(
        self,
        description: str,
        chart_type:  str = "auto",
        title:       str = "",
        filename:    str = "chart.png",
        **kw,
    ) -> str:
        # Ask the LLM to extract structured data + chart spec from the description
        prompt = (
            "Extract chart data from the following description. "
            "Return ONLY a JSON object with keys: "
            "labels (list of strings), values (list of numbers), "
            "chart_type (bar|line|scatter|pie|histogram), title (string), "
            "x_label (string), y_label (string). "
            "If multiple series, 'values' should be a list of {name, data} objects. "
            "Example output: {\"labels\":[\"Q1\",\"Q2\"],\"values\":[1.2,1.5],"
            "\"chart_type\":\"bar\",\"title\":\"Revenue\",\"x_label\":\"Quarter\",\"y_label\":\"Revenue ($M)\"}\n\n"
            f"Description: {description}"
        )
        try:
            llm    = get_llm()
            result = llm.invoke(prompt)
            raw    = str(result.content).strip()
            raw    = re.sub(r"^```json?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            import json
            spec = json.loads(raw)
        except Exception as exc:
            return f"Could not parse chart description: {exc}"

        if chart_type != "auto":
            spec["chart_type"] = chart_type
        if title:
            spec["title"] = title

        return _render_quick_chart(spec, filename)


def _render_quick_chart(spec: dict, filename: str) -> str:
    """Render a chart from a parsed spec dict."""
    _apply_dark_style()
    ctype  = spec.get("chart_type", "bar")
    labels = spec.get("labels", [])
    vals   = spec.get("values", [])
    title  = spec.get("title", "Chart")
    xl     = spec.get("x_label", "")
    yl     = spec.get("y_label", "")
    colors = _colors(max(len(labels), 10))

    fig, ax = plt.subplots(figsize=_STYLE["figsize"])
    fig.patch.set_facecolor(_DARK_BG)

    if ctype == "pie":
        ax.pie(vals, labels=labels, colors=colors[:len(labels)],
               autopct="%1.1f%%", startangle=90,
               textprops={"color": _DARK_TEXT})
        ax.set_facecolor(_DARK_BG)
    elif ctype == "line":
        ax.plot(labels, vals, color=colors[0], linewidth=2.5, marker="o", markersize=5)
        ax.fill_between(range(len(labels)), vals, alpha=0.12, color=colors[0])
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right")
    elif ctype == "histogram":
        ax.hist(vals, bins="auto", color=colors[0], edgecolor=_DARK_BG, alpha=0.85)
    else:  # bar (default)
        if vals and isinstance(vals[0], dict):
            # Multiple series
            x   = np.arange(len(labels))
            w   = 0.8 / len(vals)
            for i, series in enumerate(vals):
                ax.bar(x + i * w, series["data"], width=w,
                       label=series["name"], color=colors[i], alpha=0.88)
            ax.set_xticks(x + w * (len(vals) - 1) / 2)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.legend()
        else:
            bars = ax.bar(labels, vals, color=colors[:len(labels)], alpha=0.88, width=0.65)
            ax.bar_label(bars, fmt="%.2g", padding=3, color=_DARK_TEXT, fontsize=9)
            ax.set_xticklabels(labels, rotation=20, ha="right")

    ax.set_title(title, fontweight="bold", color=_DARK_TEXT)
    if xl: ax.set_xlabel(xl)
    if yl: ax.set_ylabel(yl)
    plt.tight_layout()
    return _save(fig, filename)


class TimeSeriesChartTool(BaseTool):
    """
    Plot one or more variables over time.

    Handles line charts, area charts, and multi-line comparisons.
    Auto-detects date formats and sorts chronologically.
    """
    name: str        = "time_series_chart"
    description: str = (
        "Create a time-series line or area chart. "
        "Inputs: data_source (CSV string or registered dataset name), "
        "date_col (str), value_cols (comma-separated column names), "
        "title (str), y_label (str), area (bool), filename (str)."
    )
    args_schema: type[BaseModel] = TimeSeriesInput

    def _run(
        self,
        data_source: str,
        date_col:    str  = "date",
        value_cols:  str  = "value",
        title:       str  = "Time Series",
        y_label:     str  = "",
        area:        bool = False,
        filename:    str  = "time_series.png",
        **kw,
    ) -> str:
        import pandas as pd
        try:
            df = _load_data(data_source)
        except Exception as exc:
            return f"Could not load data: {exc}"

        if date_col not in df.columns:
            return f"Column '{date_col}' not found. Available: {list(df.columns)}"

        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        except Exception:
            pass  # non-date x-axis is fine

        cols   = [c.strip() for c in value_cols.split(",") if c.strip() in df.columns]
        if not cols:
            return f"None of {value_cols!r} found. Available: {list(df.columns)}"

        _apply_dark_style()
        colors = _colors(len(cols))
        fig, ax = plt.subplots(figsize=_STYLE["figsize"])
        fig.patch.set_facecolor(_DARK_BG)

        for i, col in enumerate(cols):
            ax.plot(df[date_col], df[col], label=col,
                    color=colors[i], linewidth=2.2, marker="o",
                    markersize=4 if len(df) <= 50 else 0)
            if area:
                ax.fill_between(df[date_col], df[col], alpha=0.12, color=colors[i])

        ax.set_title(title, fontweight="bold", color=_DARK_TEXT)
        ax.set_xlabel(date_col.replace("_", " ").title())
        ax.set_ylabel(y_label or (cols[0] if len(cols) == 1 else "Value"))
        if len(cols) > 1:
            ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()
        return _save(fig, filename)


class BarChartTool(BaseTool):
    """
    Create grouped, stacked, or horizontal bar charts.

    Pass categories and values as comma-separated strings, or pass a JSON
    array of {name, data} series for grouped/stacked charts.
    """
    name: str        = "bar_chart"
    description: str = (
        "Create a bar chart (vertical, horizontal, grouped, or stacked). "
        "Inputs: categories (comma-separated labels), "
        "values (comma-separated numbers OR JSON array of {name,data} for multi-series), "
        "title, x_label, y_label, horizontal (bool), stacked (bool), filename."
    )
    args_schema: type[BaseModel] = BarChartInput

    def _run(
        self,
        categories: str,
        values:     str,
        title:      str  = "Bar Chart",
        x_label:    str  = "",
        y_label:    str  = "",
        horizontal: bool = False,
        stacked:    bool = False,
        filename:   str  = "bar_chart.png",
        **kw,
    ) -> str:
        import json as _json
        labels = [c.strip() for c in categories.split(",")]
        colors = _colors(20)

        # Detect multi-series JSON
        multi_series = None
        try:
            parsed = _json.loads(values)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                multi_series = parsed
        except Exception:
            pass

        _apply_dark_style()
        fig, ax = plt.subplots(figsize=_STYLE["figsize"])
        fig.patch.set_facecolor(_DARK_BG)

        if multi_series:
            x      = np.arange(len(labels))
            n_ser  = len(multi_series)
            w      = 0.75 / n_ser if not stacked else 0.65
            bottom = np.zeros(len(labels))
            for i, series in enumerate(multi_series):
                data = np.array(series["data"], dtype=float)
                if stacked:
                    if horizontal:
                        ax.barh(labels, data, label=series["name"],
                                color=colors[i], alpha=0.88, left=bottom)
                        bottom += data
                    else:
                        ax.bar(labels, data, label=series["name"],
                               color=colors[i], alpha=0.88, bottom=bottom)
                        bottom += data
                else:
                    offset = x + i * w - (n_ser - 1) * w / 2
                    if horizontal:
                        ax.barh(offset, data, w * 0.9,
                                label=series["name"], color=colors[i], alpha=0.88)
                    else:
                        ax.bar(offset, data, w * 0.9,
                               label=series["name"], color=colors[i], alpha=0.88)
            if not stacked:
                if horizontal:
                    ax.set_yticks(x); ax.set_yticklabels(labels)
                else:
                    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.legend()
        else:
            nums = np.array([float(v.strip()) for v in values.split(",")])
            if horizontal:
                bars = ax.barh(labels, nums, color=colors[:len(labels)], alpha=0.88)
                ax.bar_label(bars, fmt="%.3g", padding=3, color=_DARK_TEXT, fontsize=9)
            else:
                bars = ax.bar(labels, nums, color=colors[:len(labels)], alpha=0.88, width=0.65)
                ax.bar_label(bars, fmt="%.3g", padding=3, color=_DARK_TEXT, fontsize=9)
                ax.set_xticklabels(labels, rotation=20, ha="right")

        ax.set_title(title, fontweight="bold", color=_DARK_TEXT)
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
        plt.tight_layout()
        return _save(fig, filename)


class ScatterPlotTool(BaseTool):
    """
    Plot two continuous variables against each other.

    Optionally encode a third variable as colour or point size.
    Adds a linear regression trend line if requested.
    """
    name: str        = "scatter_plot"
    description: str = (
        "Create a scatter plot from a loaded dataset or CSV string. "
        "Inputs: data_source (dataset name or CSV), x_col, y_col, "
        "color_col (optional), size_col (optional), title, trend (bool), filename."
    )
    args_schema: type[BaseModel] = ScatterInput

    def _run(
        self,
        data_source: str,
        x_col:    str,
        y_col:    str,
        color_col: str  = "",
        size_col:  str  = "",
        title:    str  = "Scatter Plot",
        trend:    bool = False,
        filename: str  = "scatter.png",
        **kw,
    ) -> str:
        import pandas as pd
        try:
            df = _load_data(data_source)
        except Exception as exc:
            return f"Could not load data: {exc}"

        for col in [x_col, y_col]:
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {list(df.columns)}"

        _apply_dark_style()
        colors = _colors(20)
        fig, ax = plt.subplots(figsize=_STYLE["figsize"])
        fig.patch.set_facecolor(_DARK_BG)

        c_vals = None
        if color_col and color_col in df.columns:
            cats   = df[color_col].astype("category")
            c_vals = [colors[i % len(colors)] for i in cats.cat.codes]

        s_vals = 40
        if size_col and size_col in df.columns:
            s_raw  = df[size_col].astype(float)
            s_vals = ((s_raw - s_raw.min()) / (s_raw.max() - s_raw.min() + 1e-9) * 200 + 20)

        sc = ax.scatter(
            df[x_col], df[y_col],
            c=c_vals or colors[0], s=s_vals,
            alpha=0.72, edgecolors="none",
        )

        if color_col and color_col in df.columns:
            cats = df[color_col].astype("category")
            from matplotlib.lines import Line2D
            handles = [Line2D([0],[0], marker="o", color="w",
                              markerfacecolor=colors[i % len(colors)],
                              markersize=8, label=str(cat))
                       for i, cat in enumerate(cats.cat.categories)]
            ax.legend(handles=handles, title=color_col)

        if trend:
            xs = df[x_col].astype(float).values
            ys = df[y_col].astype(float).values
            mask = ~(np.isnan(xs) | np.isnan(ys))
            if mask.sum() >= 2:
                m, b = np.polyfit(xs[mask], ys[mask], 1)
                xr   = np.linspace(xs[mask].min(), xs[mask].max(), 100)
                ax.plot(xr, m * xr + b, "--", color="#ef4444",
                        linewidth=1.5, label=f"Trend  y={m:.3g}x+{b:.3g}")
                ax.legend()

        ax.set_title(title, fontweight="bold", color=_DARK_TEXT)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        plt.tight_layout()
        return _save(fig, filename)


class DistributionChartTool(BaseTool):
    """
    Visualise the distribution of a single variable.

    Supports histogram, KDE curve, box plot, violin plot,
    and a combined histogram+KDE overlay.
    Optional group_col splits the distribution by category.
    """
    name: str        = "distribution_chart"
    description: str = (
        "Visualise the distribution of a column. "
        "Inputs: data_source (dataset name or CSV), column (str), "
        "plot_type (histogram|kde|box|violin|combined), "
        "group_col (optional), title (str), filename."
    )
    args_schema: type[BaseModel] = DistributionInput

    def _run(
        self,
        data_source: str,
        column:     str,
        plot_type:  str = "histogram",
        group_col:  str = "",
        title:      str = "",
        filename:   str = "distribution.png",
        **kw,
    ) -> str:
        import seaborn as sns
        import pandas as pd
        try:
            df = _load_data(data_source)
        except Exception as exc:
            return f"Could not load data: {exc}"
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        _apply_dark_style()
        colors  = _colors(10)
        palette = {cat: colors[i % len(colors)]
                   for i, cat in enumerate(df[group_col].unique())} if group_col else None
        fig, ax = plt.subplots(figsize=_STYLE["figsize"])
        fig.patch.set_facecolor(_DARK_BG)

        hue = group_col if group_col and group_col in df.columns else None

        if plot_type == "histogram":
            sns.histplot(df, x=column, hue=hue, palette=palette,
                         ax=ax, color=colors[0], edgecolor=_DARK_BG, alpha=0.85)
        elif plot_type == "kde":
            sns.kdeplot(df, x=column, hue=hue, palette=palette,
                        ax=ax, color=colors[0], fill=True, alpha=0.4)
        elif plot_type == "combined":
            sns.histplot(df, x=column, hue=hue, palette=palette,
                         ax=ax, color=colors[0], edgecolor=_DARK_BG, alpha=0.6, stat="density")
            sns.kdeplot(df, x=column, hue=hue, palette=palette,
                        ax=ax, color=colors[1], linewidth=2.5)
        elif plot_type == "box":
            y_col = column; x_col = hue
            if x_col:
                sns.boxplot(df, x=x_col, y=y_col, palette=palette, ax=ax,
                            linecolor=_DARK_TEXT, linewidth=1.2)
            else:
                sns.boxplot(data=df[column], ax=ax, color=colors[0],
                            linecolor=_DARK_TEXT, linewidth=1.2)
        elif plot_type == "violin":
            if hue:
                sns.violinplot(df, x=hue, y=column, palette=palette,
                               ax=ax, linecolor=_DARK_TEXT, linewidth=1)
            else:
                sns.violinplot(data=df[column], ax=ax, color=colors[0],
                               linecolor=_DARK_TEXT, linewidth=1)

        ax.set_title(title or f"Distribution of {column}", fontweight="bold", color=_DARK_TEXT)
        plt.tight_layout()
        return _save(fig, filename)


class CorrelationHeatmapTool(BaseTool):
    """
    Compute and visualise the Pearson correlation matrix for numeric columns.

    Colour scale runs from deep red (−1) through black (0) to vivid purple (+1).
    Cell annotations show the correlation coefficient to 2 decimal places.
    """
    name: str        = "correlation_heatmap"
    description: str = (
        "Compute and plot a correlation heatmap for numeric columns. "
        "Inputs: data_source (dataset name or CSV), "
        "columns (comma-separated subset, blank = all numeric), "
        "title, annotate (bool), filename."
    )
    args_schema: type[BaseModel] = HeatmapInput

    def _run(
        self,
        data_source: str,
        columns:  str  = "",
        title:    str  = "Correlation Heatmap",
        annotate: bool = True,
        filename: str  = "heatmap.png",
        **kw,
    ) -> str:
        import seaborn as sns
        import pandas as pd
        try:
            df = _load_data(data_source)
        except Exception as exc:
            return f"Could not load data: {exc}"

        if columns.strip():
            cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
            if not cols:
                return f"None of {columns!r} found. Available: {list(df.columns)}"
            df = df[cols]

        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return "Need at least 2 numeric columns for a correlation heatmap."

        corr = num_df.corr()
        n    = corr.shape[0]
        size = max(6, min(14, n * 1.1))

        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        fig.patch.set_facecolor(_DARK_BG)

        # Custom diverging palette that fits the dark theme
        cmap = sns.diverging_palette(0, 270, s=85, l=40, as_cmap=True)
        sns.heatmap(
            corr, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=annotate, fmt=".2f", annot_kws={"size": 9, "color": _DARK_TEXT},
            linewidths=0.4, linecolor=_DARK_GRID,
            square=True, cbar_kws={"shrink": 0.8},
        )
        ax.tick_params(colors=_DARK_TEXT, labelsize=9)
        ax.set_title(title, fontweight="bold", color=_DARK_TEXT, pad=12)
        plt.tight_layout()
        return _save(fig, filename)


class CompositeChartTool(BaseTool):
    """
    Create a multi-panel dashboard combining 2–4 different charts in a grid.

    Each panel is specified as a JSON object with chart type and data.
    Useful for executive dashboards, comparison views, and report inserts.
    """
    name: str        = "composite_chart"
    description: str = (
        "Create a multi-panel dashboard with 2–4 charts in a grid. "
        "Inputs: charts (JSON list of chart specs — each with type, title, data, x, y), "
        "layout_title (str), cols (1–4), filename."
    )
    args_schema: type[BaseModel] = CompositeInput

    def _run(
        self,
        charts:       str,
        layout_title: str = "Dashboard",
        cols:         int = 2,
        filename:     str = "dashboard.png",
        **kw,
    ) -> str:
        import json, pandas as pd
        try:
            specs = json.loads(charts)
        except Exception:
            return "Could not parse charts JSON."
        if not specs or not isinstance(specs, list):
            return "charts must be a non-empty JSON list."

        n    = len(specs)
        rows = (n + cols - 1) // cols
        fw   = _STYLE["figsize"][0] * min(cols, n)
        fh   = _STYLE["figsize"][1] * rows * 0.7

        _apply_dark_style()
        colors  = _colors(10)
        fig, axes = plt.subplots(rows, cols, figsize=(fw, fh), squeeze=False)
        fig.patch.set_facecolor(_DARK_BG)

        for idx, (spec, ax) in enumerate(zip(specs, [axes[r][c] for r in range(rows) for c in range(cols)])):
            ax.set_facecolor(_DARK_BG2)
            _render_panel(ax, spec, colors)

        # Hide unused panels
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(layout_title, fontsize=14, fontweight="bold",
                     color=_DARK_TEXT, y=1.01)
        plt.tight_layout()
        return _save(fig, filename)


def _render_panel(ax: plt.Axes, spec: dict, colors: list) -> None:
    """Render a single panel inside a composite chart."""
    import pandas as pd
    ctype = spec.get("type", "bar")
    title = spec.get("title", "")
    data  = spec.get("data", "")
    x_col = spec.get("x", "")
    y_col = spec.get("y", "")

    df = _df_from_registry(data)
    if df is None:
        try:
            df = _parse_csv_data(data)
        except Exception:
            ax.text(0.5, 0.5, f"No data:\n{data[:40]}", ha="center", va="center",
                    color=_DARK_TEXT, transform=ax.transAxes)
            ax.set_title(title, color=_DARK_TEXT)
            return

    try:
        if ctype in ("bar", "column"):
            if x_col in df.columns and y_col in df.columns:
                ax.bar(df[x_col], df[y_col], color=colors[0], alpha=0.85, width=0.65)
                ax.set_xticklabels(df[x_col], rotation=30, ha="right", fontsize=8)
        elif ctype == "line":
            if x_col in df.columns and y_col in df.columns:
                ax.plot(df[x_col], df[y_col], color=colors[1], linewidth=2, marker="o", markersize=3)
                ax.tick_params(axis="x", rotation=30, labelsize=8)
        elif ctype == "scatter":
            if x_col in df.columns and y_col in df.columns:
                ax.scatter(df[x_col], df[y_col], color=colors[2], alpha=0.65, s=25, edgecolors="none")
        elif ctype == "histogram":
            col = y_col or x_col
            if col in df.columns:
                ax.hist(df[col], bins="auto", color=colors[3], edgecolor=_DARK_BG, alpha=0.85)
        elif ctype == "pie":
            if x_col in df.columns and y_col in df.columns:
                ax.pie(df[y_col], labels=df[x_col], colors=colors[:len(df)],
                       autopct="%1.1f%%", textprops={"color": _DARK_TEXT, "fontsize": 8})
    except Exception as exc:
        ax.text(0.5, 0.5, f"Render error:\n{exc}", ha="center", va="center",
                color="#ef4444", transform=ax.transAxes, fontsize=8)

    ax.set_title(title, fontweight="bold", color=_DARK_TEXT, fontsize=10)
    ax.tick_params(colors=_DARK_TEXT, labelsize=8)
    ax.set_facecolor(_DARK_BG2)


class StyleConfigTool(BaseTool):
    """
    Configure global chart style (theme, colour palette, figure size).

    Changes apply to all subsequent charts generated in this session.
    """
    name: str        = "configure_chart_style"
    description: str = (
        "Set global chart styling for all subsequent charts. "
        "Inputs: theme (dark|light), "
        "palette (jarvis|cool|warm|neutral|green|seaborn), "
        "figsize (e.g. '12x8')."
    )
    args_schema: type[BaseModel] = StyleInput

    def _run(
        self,
        theme:   str = "dark",
        palette: str = "jarvis",
        figsize: str = "10x6",
        **kw,
    ) -> str:
        _STYLE["theme"]   = theme
        _STYLE["palette"] = palette if palette in _PALETTES else "jarvis"
        try:
            w, h = [float(x) for x in figsize.lower().replace("x", " ").split()]
            _STYLE["figsize"] = (w, h)
        except Exception:
            pass
        return (
            f"Style updated: theme={_STYLE['theme']}, "
            f"palette={_STYLE['palette']}, "
            f"figsize={_STYLE['figsize'][0]}×{_STYLE['figsize'][1]}"
        )


class ListChartsTool(BaseTool):
    """List recently generated chart files with sizes and paths."""
    name: str        = "list_charts"
    description: str = "List recently generated chart files. No inputs required."

    def _run(self, **kw) -> str:
        files = sorted(CHART_DIR.glob("*.png"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return "No charts generated yet."
        lines = [f"Generated charts ({len(files)} total):"]
        for f in files[:15]:
            lines.append(f"  {f.name}  ({f.stat().st_size/1024:.1f} KB)  →  {f}")
        return "\n".join(lines)


# ── Public factory ────────────────────────────────────────────────────────────

def get_graphing_tools() -> list[BaseTool]:
    """Return all graphing/visualisation tools."""
    return [
        QuickChartTool(),
        TimeSeriesChartTool(),
        BarChartTool(),
        ScatterPlotTool(),
        DistributionChartTool(),
        CorrelationHeatmapTool(),
        CompositeChartTool(),
        StyleConfigTool(),
        ListChartsTool(),
    ]
