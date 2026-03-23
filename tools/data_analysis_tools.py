"""
tools/data_analysis_tools.py
──────────────────────────────
Tools for the DataAnalysisAgent.

  LoadDataTool          — load CSV / TSV / Excel into an in-memory store
  InspectDataTool       — shape, dtypes, head, describe, missing values
  RunPandasTool         — execute arbitrary pandas code against the loaded frame
  PlotDataTool          — generate matplotlib chart saved as PNG
  ExportDataTool        — save the (possibly transformed) frame to CSV / Excel
  ListLoadedDataTool    — show what datasets are currently in memory
"""
from __future__ import annotations

import io
import logging
import textwrap
import traceback
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# ── Shared in-process dataframe registry ─────────────────────────────────────
# Maps dataset name → pandas DataFrame.  Lives for the process lifetime so
# multiple tool calls within one agent turn can share state.
_FRAMES: dict[str, "pd.DataFrame"] = {}   # noqa: F821

_CHART_DIR = settings.uploads_path / "generated" / "charts"
_CHART_DIR.mkdir(parents=True, exist_ok=True)

_MAX_OUTPUT = 4_000   # chars returned to the agent per tool call


# =============================================================================
#  Schemas
# =============================================================================

class LoadDataInput(BaseModel):
    file_path: str = Field(..., description="Path to the CSV, TSV, or Excel file.")
    name:      str = Field("df", description="Short name to register the dataset as (e.g. 'sales').")
    encoding:  str = Field("utf-8", description="File encoding (default utf-8).")


class InspectDataInput(BaseModel):
    name: str = Field("df", description="Dataset name (from load_data).")
    rows: int = Field(5, ge=1, le=50, description="Number of rows to show in head().")


class RunPandasInput(BaseModel):
    code: str  = Field(
        ...,
        description=(
            "Python code using pandas. The loaded dataframe is available as `df` "
            "(or by its registered name). Write results to `result` variable or print them. "
            "Example: result = df.groupby('category')['revenue'].sum()"
        ),
    )
    name: str  = Field("df", description="Dataset name to expose as `df` in the code.")


class PlotDataInput(BaseModel):
    code:     str  = Field(
        ...,
        description=(
            "Matplotlib code to generate a chart. `df` is the loaded dataframe. "
            "Do NOT call plt.show() — the tool saves the figure automatically. "
            "Example: df['revenue'].plot(kind='bar', title='Revenue'); plt.tight_layout()"
        ),
    )
    name:     str  = Field("df", description="Dataset name to expose as `df`.")
    filename: str  = Field("chart.png", description="Output filename for the chart image.")


class ExportDataInput(BaseModel):
    name:     str  = Field("df", description="Dataset name to export.")
    filename: str  = Field("output.csv", description="Output filename (.csv or .xlsx).")
    code:     str  = Field(
        "",
        description="Optional pandas code to transform `df` before exporting. Leave blank to export as-is.",
    )


# =============================================================================
#  Tools
# =============================================================================

class LoadDataTool(BaseTool):
    """
    Load a CSV, TSV, or Excel file into an in-memory pandas DataFrame.

    Use this first before any analysis. The dataset is registered under a
    short name (default 'df') and can be referenced by subsequent tools.
    """
    name: str        = "load_data"
    description: str = (
        "Load a CSV, TSV, or Excel file into memory for analysis. "
        "Call this before inspect_data, run_pandas, or plot_data. "
        "Inputs: file_path (str), name (str, default 'df'), encoding (str, default 'utf-8')."
    )
    args_schema: type[BaseModel] = LoadDataInput

    def _run(self, file_path: str, name: str = "df", encoding: str = "utf-8", **kw) -> str:
        import pandas as pd
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        try:
            ext = path.suffix.lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            elif ext == ".tsv":
                df = pd.read_csv(path, sep="\t", encoding=encoding, encoding_errors="replace")
            else:
                df = pd.read_csv(path, encoding=encoding, encoding_errors="replace")

            _FRAMES[name] = df
            rows, cols = df.shape
            col_summary = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns[:10])
            if len(df.columns) > 10:
                col_summary += f" … (+{len(df.columns)-10} more)"
            logger.info("Loaded dataset '%s': %d rows × %d cols from %s", name, rows, cols, path.name)
            return (
                f"Loaded '{name}' from {path.name}\n"
                f"  Shape : {rows:,} rows × {cols} columns\n"
                f"  Columns: {col_summary}\n"
                f"  Memory : {df.memory_usage(deep=True).sum() / 1024:.1f} KB"
            )
        except Exception as exc:
            return f"Failed to load {file_path}: {exc}"

    async def _arun(self, **kw) -> str:
        raise NotImplementedError


class InspectDataTool(BaseTool):
    """
    Show shape, column dtypes, head, descriptive statistics, and missing-value counts
    for a loaded dataset.
    """
    name: str        = "inspect_data"
    description: str = (
        "Inspect a loaded dataset: show shape, dtypes, head, describe(), and missing values. "
        "Inputs: name (str, default 'df'), rows (int, default 5)."
    )
    args_schema: type[BaseModel] = InspectDataInput

    def _run(self, name: str = "df", rows: int = 5, **kw) -> str:
        import pandas as pd
        df = _FRAMES.get(name)
        if df is None:
            return f"Dataset '{name}' not loaded. Call load_data first."
        try:
            buf = io.StringIO()
            buf.write(f"Dataset: '{name}'  shape={df.shape}\n\n")

            buf.write("--- dtypes ---\n")
            buf.write(str(df.dtypes) + "\n\n")

            buf.write(f"--- head({rows}) ---\n")
            buf.write(df.head(rows).to_string(max_cols=12) + "\n\n")

            buf.write("--- describe() ---\n")
            buf.write(df.describe(include="all").to_string(max_cols=12) + "\n\n")

            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing):
                buf.write("--- missing values ---\n")
                buf.write(str(missing) + "\n")
            else:
                buf.write("No missing values.\n")

            result = buf.getvalue()
            if len(result) > _MAX_OUTPUT:
                result = result[:_MAX_OUTPUT] + "\n… [truncated]"
            return result
        except Exception as exc:
            return f"Inspection failed: {exc}"

    async def _arun(self, **kw) -> str:
        raise NotImplementedError


class RunPandasTool(BaseTool):
    """
    Execute arbitrary pandas code against a loaded dataset and return the output.

    The named dataset is available as both `df` (always) and its registered name.
    Write your answer to a variable called `result` OR use `print()`.
    """
    name: str        = "run_pandas"
    description: str = (
        "Execute pandas Python code on a loaded dataset and return the result. "
        "The dataframe is available as `df`. Write answers to `result` or use print(). "
        "Inputs: code (str), name (str, default 'df')."
    )
    args_schema: type[BaseModel] = RunPandasInput

    # Blocked operations — prevents destructive or network-accessing code
    _BLOCKED = frozenset([
        "os.remove", "shutil", "subprocess", "socket", "open('/",
        "os.system", "__import__", "eval(", "exec(",
    ])

    def _safety_check(self, code: str) -> Optional[str]:
        for blocked in self._BLOCKED:
            if blocked in code:
                return f"Blocked: code contains forbidden pattern '{blocked}'."
        return None

    def _run(self, code: str, name: str = "df", **kw) -> str:
        import pandas as pd
        import numpy as np

        df = _FRAMES.get(name)
        if df is None:
            return f"Dataset '{name}' not loaded. Call load_data first."

        violation = self._safety_check(code)
        if violation:
            return violation

        local_ns: dict = {
            "df": df, name: df,
            "pd": pd, "np": np,
            "result": None,
        }
        captured = io.StringIO()

        try:
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured
            exec(textwrap.dedent(code), local_ns)   # noqa: S102
            sys.stdout = old_stdout
        except Exception:
            sys.stdout = old_stdout
            return f"Code error:\n{traceback.format_exc(limit=5)}"

        # If the user mutated df in-place, update the registry
        if "df" in local_ns and local_ns["df"] is not df:
            _FRAMES[name] = local_ns["df"]

        parts: list[str] = []
        printed = captured.getvalue().strip()
        if printed:
            parts.append(printed)

        result = local_ns.get("result")
        if result is not None:
            result_str = (
                result.to_string(max_rows=50)
                if hasattr(result, "to_string")
                else str(result)
            )
            parts.append(result_str)

        output = "\n\n".join(parts) if parts else "(No output — assign to `result` or use print())"
        if len(output) > _MAX_OUTPUT:
            output = output[:_MAX_OUTPUT] + "\n… [truncated]"
        return output

    async def _arun(self, **kw) -> str:
        raise NotImplementedError


class PlotDataTool(BaseTool):
    """
    Generate a matplotlib chart from a loaded dataset and save it as a PNG.
    Returns the file path so the agent can cite it in the response.
    """
    name: str        = "plot_data"
    description: str = (
        "Generate a matplotlib chart from a loaded dataset. "
        "Do NOT call plt.show() — the tool saves the figure automatically. "
        "Inputs: code (str, matplotlib code using `df`), "
        "name (str, dataset name, default 'df'), filename (str, e.g. 'revenue_chart.png')."
    )
    args_schema: type[BaseModel] = PlotDataInput

    def _run(self, code: str, name: str = "df", filename: str = "chart.png", **kw) -> str:
        import pandas as pd
        import numpy as np

        df = _FRAMES.get(name)
        if df is None:
            return f"Dataset '{name}' not loaded. Call load_data first."

        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend, must be before pyplot
            import matplotlib.pyplot as plt

            local_ns = {
                "df": df, name: df,
                "pd": pd, "np": np,
                "plt": plt,
            }

            fig, ax = plt.subplots(figsize=(10, 6))
            local_ns["fig"] = fig
            local_ns["ax"]  = ax

            exec(textwrap.dedent(code), local_ns)   # noqa: S102

            # Sanitise filename
            import re as _re
            safe = _re.sub(r"[^\w\-.]", "_", Path(filename).name)[:60]
            if not safe.endswith(".png"):
                safe = safe.rsplit(".", 1)[0] + ".png"

            out_path = _CHART_DIR / safe
            counter  = 1
            stem     = out_path.stem
            while out_path.exists():
                out_path = _CHART_DIR / f"{stem}_{counter}.png"
                counter += 1

            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close("all")
            logger.info("Chart saved: %s", out_path)
            return f"Chart saved: {out_path}\n({out_path.stat().st_size:,} bytes)"

        except Exception:
            return f"Chart generation failed:\n{traceback.format_exc(limit=5)}"

    async def _arun(self, **kw) -> str:
        raise NotImplementedError


class ExportDataTool(BaseTool):
    """
    Save the current (or transformed) dataset to a CSV or Excel file.
    Optionally apply pandas code to transform the frame before exporting.
    """
    name: str        = "export_data"
    description: str = (
        "Export a loaded dataset to a CSV or Excel file. "
        "Optionally apply transformation code before saving. "
        "Inputs: name (str), filename (str, .csv or .xlsx), code (str, optional transform)."
    )
    args_schema: type[BaseModel] = ExportDataInput

    def _run(self, name: str = "df", filename: str = "output.csv", code: str = "", **kw) -> str:
        import pandas as pd
        import numpy as np

        df = _FRAMES.get(name)
        if df is None:
            return f"Dataset '{name}' not loaded. Call load_data first."

        if code.strip():
            local_ns = {"df": df.copy(), name: df.copy(), "pd": pd, "np": np, "result": None}
            try:
                exec(textwrap.dedent(code), local_ns)  # noqa: S102
                df = local_ns.get("result") or local_ns.get("df") or df
            except Exception:
                return f"Transform code failed:\n{traceback.format_exc(limit=5)}"

        out_dir = settings.uploads_path / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        import re as _re
        safe = _re.sub(r"[^\w\-.]", "_", Path(filename).name)[:60]
        out_path = out_dir / safe

        try:
            if safe.endswith(".xlsx"):
                df.to_excel(out_path, index=False)
            else:
                df.to_csv(out_path, index=False, encoding="utf-8")
            rows, cols = df.shape
            return f"Exported: {out_path}\n({rows:,} rows × {cols} cols, {out_path.stat().st_size:,} bytes)"
        except Exception as exc:
            return f"Export failed: {exc}"

    async def _arun(self, **kw) -> str:
        raise NotImplementedError


class ListLoadedDataTool(BaseTool):
    """List all datasets currently loaded in memory."""
    name: str        = "list_loaded_data"
    description: str = "List all datasets currently loaded in memory with their shapes. No inputs."

    def _run(self, **kw) -> str:
        if not _FRAMES:
            return "No datasets loaded. Use load_data to load a CSV, TSV, or Excel file."
        lines = ["Loaded datasets:"]
        for dname, df in _FRAMES.items():
            lines.append(f"  '{dname}'  — {df.shape[0]:,} rows × {df.shape[1]} cols")
        return "\n".join(lines)

    async def _arun(self, **kw) -> str:
        raise NotImplementedError


def get_data_analysis_tools() -> list[BaseTool]:
    """Return all data-analysis tools."""
    return [
        LoadDataTool(),
        InspectDataTool(),
        RunPandasTool(),
        PlotDataTool(),
        ExportDataTool(),
        ListLoadedDataTool(),
    ]
