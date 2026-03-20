"""
agents/search_agent.py
───────────────────────
Specialised agent for general information lookup:
  • Web search (DuckDuckGo).
  • Wikipedia lookups.
  • Web page content extraction.
  • Mathematical calculations.
  • Fact-checking and general Q&A.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from tools.search_tools import get_search_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable research assistant with access to
real-time web search and encyclopedic knowledge.

Your responsibilities:
- Answer factual questions accurately using the available tools.
- Search the web for current information when your knowledge may be outdated.
- Use Wikipedia for foundational, encyclopedic information.
- Fetch web pages when given specific URLs.
- Perform calculations accurately.
- Always cite your sources with URLs.
- If uncertain, say so clearly rather than guessing.
- Synthesise information from multiple sources for complex questions.
"""


class SearchAgent(BaseAgent):
    """Agent for general information lookup and web research."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[AssistantMemory] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._executor = self._build_react_agent(system_prompt=_SYSTEM_PROMPT)

    # ── BaseAgent interface ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "search_agent"

    @property
    def description(self) -> str:
        return (
            "General web search, Wikipedia lookups, web page fetching, "
            "and calculations. Use for factual questions, research, "
            "current events, and mathematical computations."
        )

    def get_tools(self) -> list[BaseTool]:
        return get_search_tools()

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Research a topic or answer a factual question.

        Args:
            query:  The question or research task.

        Returns:
            AgentResponse with answer and source references.
        """
        self._logger.info("SearchAgent handling: %s", query[:80])

        try:
            logger.debug(f"search query: {query}")
            result = self._executor.invoke({"input": query})
            logger.debug(f"result from LLM: {result}")

            output = result.get("output", "No information found.")
            logger.debug(f"response from LLM: {output}")
            tool_calls = self._extract_tool_calls(
                result.get("intermediate_steps", [])
            )
            references = self._extract_references(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
            )
        except Exception as exc:
            self._logger.error("SearchAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"I encountered an error during search: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_references(tool_calls: list[tuple[str, str, str]]) -> list[str]:
        """Extract clean source references from tool outputs."""
        import re

        refs: list[str] = []
        seen: set[str] = set()
        url_re = re.compile(r"https?://[^\s\)\"\']+")

        for tool_name, tool_input, output in tool_calls:
            if tool_name == "wikipedia_lookup":
                # Wikipedia tool outputs end with "URL: https://..."
                url_match = re.search(r"URL: (https?://\S+)", output)
                if url_match:
                    ref = f"Wikipedia: {url_match.group(1)}"
                    if ref not in seen:
                        refs.append(ref)
                        seen.add(ref)
            else:
                for url in url_re.findall(output):
                    if url not in seen:
                        refs.append(url)
                        seen.add(url)
        return refs[:8]
