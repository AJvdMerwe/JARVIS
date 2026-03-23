"""
agents/graphing_agent.py
─────────────────────────
GraphingAgent — the dedicated data-visualisation agent.

Intent: GRAPH

Routing triggers
────────────────
Explicit chart requests:
  "plot", "chart", "graph", "visualise", "visualize", "draw a chart"
  "bar chart", "line chart", "scatter plot", "histogram", "heatmap"
  "show me a chart of", "create a dashboard"

Also invoked by other agents when they need a chart:
  • DataAnalysisAgent — delegates chart requests here via tool calls
  • FinancialAgent    — price history charts, comparison charts
  • WritingAssistantAgent — embeds chart paths in reports

Workflow
────────
1. The ReAct executor chooses the most appropriate graphing tool.
2. The tool generates the chart PNG and returns the file path.
3. The agent summarises what was plotted and recommends follow-up charts.
4. The agent response includes the file path in references so the UI
   can render the chart inline.

Conversation context
────────────────────
_augment_query() is called so "make it a bar chart instead" and
"add a trend line" correctly reference the prior exchange.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agents.base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a data visualisation specialist. Your job is to create clear,
informative, and visually appealing charts and graphs using the available tools.

Workflow for every visualisation request:
1. Understand what the user wants to see and why.
2. Choose the most appropriate chart type:
   - Comparisons across categories → bar_chart
   - Trends over time → time_series_chart
   - Relationships between two variables → scatter_plot
   - Data distribution → distribution_chart
   - Correlations across many variables → correlation_heatmap
   - Multiple charts at once → composite_chart
   - Quick one-shot from a description → quick_chart
3. Generate the chart with meaningful title, axis labels, and legend.
4. Briefly interpret what the chart shows — don't just say "chart saved".
5. Suggest 1–2 follow-up charts that would add insight.

Chart quality standards:
- Always set a descriptive title and axis labels.
- Use the configured style (dark theme by default) for consistency.
- For financial data: use time_series_chart with line/area mode.
- For distributions: prefer 'combined' (histogram + KDE) over plain histogram.
- For correlations: always annotate the heatmap cells.
- Keep composite dashboards to 4 panels maximum.

When data is provided as text or CSV, parse it directly.
When a dataset name is mentioned, check if it is loaded in the registry.
"""


class GraphingAgent(BaseAgent):
    """
    Dedicated data-visualisation agent.

    Has access to all 9 graphing tools plus the data-analysis registry
    (so it can chart any loaded DataFrame by name).
    Also gets file_output tools so it can save chart summaries as Markdown.
    """

    def __init__(
        self,
        llm:     Optional[BaseChatModel] = None,
        memory:  Optional[AssistantMemory] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._executor = self._build_react_agent(system_prompt=_SYSTEM_PROMPT)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "graphing_agent"

    @property
    def description(self) -> str:
        return (
            "Data visualisation agent. Creates bar charts, line charts, "
            "scatter plots, histograms, correlation heatmaps, and multi-panel "
            "dashboards. Works with loaded datasets or inline data."
        )

    def get_tools(self) -> list[BaseTool]:
        from tools.graphing_tools import get_graphing_tools
        from tools.data_analysis_tools import (
            LoadDataTool, ListLoadedDataTool, RunPandasTool,
        )
        from tools.file_output_tools import SaveTextFileTool
        return (
            get_graphing_tools()
            + [LoadDataTool(), ListLoadedDataTool(), RunPandasTool()]
            + [SaveTextFileTool()]
        )

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a visualisation request.

        Extra kwargs
        ------------
        dataset_name : str
            If provided, mentions the dataset in the augmented query so the
            executor knows which registered frame to use.
        file_path : str
            If provided, loads the file automatically before charting.
        chart_type : str
            Optional hint: bar | line | scatter | histogram | heatmap | composite
        """
        query        = self._augment_query(query)
        dataset_name = kwargs.get("dataset_name", "")
        file_path    = kwargs.get("file_path", "")
        chart_type   = kwargs.get("chart_type", "")

        self._logger.info("GraphingAgent handling: %s", query[:80])

        # Auto-load file if provided
        if file_path:
            from tools.data_analysis_tools import _FRAMES, LoadDataTool
            name = dataset_name or "df"
            if name not in _FRAMES:
                LoadDataTool()._run(file_path=file_path, name=name)

        # Augment the query with context hints
        extra = ""
        if dataset_name:
            extra += f"\n[Dataset registered as '{dataset_name}']"
        if chart_type:
            extra += f"\n[Preferred chart type: {chart_type}]"

        full_query = query + extra if extra else query

        try:
            result     = self._executor.invoke({"input": full_query})
            output     = result.get("output", "No visualisation generated.")
            tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
            references = self._extract_chart_paths(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
                metadata={
                    "dataset_name": dataset_name,
                    "chart_type":   chart_type,
                    "charts_saved": len(references),
                },
            )
        except Exception as exc:
            self._logger.error("GraphingAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"Visualisation failed: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_chart_paths(self, tool_calls: list[tuple]) -> list[str]:
        """Extract saved chart file paths from tool observations."""
        refs:     list[str] = []
        _PATH_RE = re.compile(r"(?:Chart saved|Saved):\s*(.+?)(?:\n|$)")
        for _, _, obs in tool_calls:
            for m in _PATH_RE.finditer(str(obs)):
                path = m.group(1).strip()
                if path and path not in refs:
                    refs.append(path)
        return refs
