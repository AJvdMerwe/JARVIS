"""
agents/data_analysis_agent.py
──────────────────────────────
DataAnalysisAgent — answers questions about structured data.

Routing triggers (Intent.DATA)
──────────────────────────────
  Explicit file references: "analyse this CSV", "load the spreadsheet",
  "plot the data", "what does the Excel file show"
  Task keywords: "correlation", "distribution", "trend", "outlier",
  "pivot", "group by", "average / mean / median / sum by", "visualise"

Workflow
────────
  1. Auto-ingest — if the query or kwargs contain a `file_path`, call
     load_data before routing to the ReAct executor.
  2. ReAct loop — the executor has access to:
       load_data · inspect_data · run_pandas · plot_data
       export_data · list_loaded_data
       save_text_file (for saving written summaries)
  3. Post-process — extract any chart / export file paths from the
     intermediate steps and attach them as references.

Conversation context
────────────────────
  Inherits BaseAgent._augment_query() so follow-up questions like
  "now show it as a bar chart" or "filter to 2024 only" work correctly.
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

_SYSTEM_PROMPT = """You are a data analysis assistant with expertise in Python, pandas, and matplotlib.

Your workflow for every data request:
1. If a file path is provided and the data isn't loaded yet, call load_data first.
2. Use inspect_data to understand the dataset structure.
3. Use run_pandas for calculations, aggregations, filtering, and transformations.
4. Use plot_data to generate charts when visualisation is requested or helpful.
5. Use export_data to save transformed datasets when the user wants to keep them.
6. Provide a clear written interpretation of your findings — numbers alone aren't enough.

Guidelines:
- Always describe what you found, not just the code output.
- Suggest follow-up analyses the user might find interesting.
- When generating charts, choose the right chart type (bar for comparison,
  line for trends, scatter for correlation, histogram for distribution).
- Handle missing values and data quality issues gracefully.
- For large datasets, sample or aggregate before plotting.
"""


class DataAnalysisAgent(BaseAgent):
    """
    Analyse structured data (CSV / TSV / Excel) using pandas and matplotlib.

    Exposes six tools: load_data · inspect_data · run_pandas · plot_data ·
    export_data · list_loaded_data.  The agent runs a ReAct loop to
    orchestrate multi-step analysis workflows in response to natural-language
    questions about the data.
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
        return "data_analysis_agent"

    @property
    def description(self) -> str:
        return (
            "Analyse structured data files (CSV, TSV, Excel). "
            "Use for: loading datasets, computing statistics, generating charts, "
            "running pandas queries, finding correlations, trends, and outliers."
        )

    def get_tools(self) -> list[BaseTool]:
        from tools.data_analysis_tools import get_data_analysis_tools
        from tools.file_output_tools import SaveTextFileTool
        return get_data_analysis_tools() + [SaveTextFileTool()]

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Analyse data in response to a natural-language query.

        Extra kwargs
        ------------
        file_path : str, optional
            If provided, the file is loaded automatically before any other
            tool is called.  The agent skips re-loading if the dataset is
            already in memory.
        dataset_name : str, optional
            Short name to register the dataset under (default "df").
        """
        query     = self._augment_query(query)
        file_path = kwargs.get("file_path", "")
        ds_name   = kwargs.get("dataset_name", "df")

        self._logger.info("DataAnalysisAgent handling: %s", query[:80])

        # ── Auto-ingest ───────────────────────────────────────────────────
        if file_path:
            from tools.data_analysis_tools import _FRAMES, LoadDataTool
            if ds_name not in _FRAMES:
                load_result = LoadDataTool()._run(
                    file_path=file_path, name=ds_name
                )
                self._logger.debug("Auto-load: %s", load_result[:100])

        # ── If no file provided, check if a path is in the query itself ───
        if not file_path:
            path_match = re.search(
                r"(?:file|dataset|csv|excel|spreadsheet)[:\s]+([^\s,\"']+\.(csv|tsv|xlsx|xls))",
                query, re.IGNORECASE,
            )
            if path_match:
                from tools.data_analysis_tools import _FRAMES, LoadDataTool
                found_path = path_match.group(1)
                if ds_name not in _FRAMES:
                    LoadDataTool()._run(file_path=found_path, name=ds_name)

        # ── ReAct executor ────────────────────────────────────────────────
        try:
            result     = self._executor.invoke({"input": query})
            output     = result.get("output", "No analysis generated.")
            tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
            references = self._extract_file_references(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
                metadata={
                    "file_path":    file_path,
                    "dataset_name": ds_name,
                },
            )
        except Exception as exc:
            self._logger.error("DataAnalysisAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"Analysis failed: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_file_references(
        self, tool_calls: list[tuple]
    ) -> list[str]:
        """
        Scan tool outputs for saved file paths (charts, exports) and return
        them as references so the REPL can display them.
        """
        refs: list[str] = []
        _PATH_RE = re.compile(r"(?:Saved|Chart saved|Exported):\s*(.+?)(?:\n|$)")
        for _, _, observation in tool_calls:
            for m in _PATH_RE.finditer(str(observation)):
                path = m.group(1).strip()
                if path and path not in refs:
                    refs.append(path)
        return refs
