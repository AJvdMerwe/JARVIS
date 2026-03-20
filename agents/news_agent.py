"""
agents/news_agent.py
─────────────────────
Specialised agent for news aggregation and summarisation:
  • Fetch top headlines across multiple RSS sources.
  • Search news by topic or keyword.
  • Summarise a news story from a URL.
  • Provide a daily briefing.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from tools.news_tools import get_news_tools
from tools.search_tools import WebFetchTool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable news assistant and journalist.
Your job is to:
- Aggregate and present the latest news in a clear, balanced way.
- Summarise stories concisely with key facts.
- Provide context and background where helpful.
- Always cite sources (publication name and URL).
- Remain neutral and factual; present multiple perspectives on contentious topics.
- Format responses with clear headings and bullet points when listing multiple stories.

Never fabricate news or events. Only report what the tools return.
"""


class NewsAgent(BaseAgent):
    """Agent that aggregates, searches, and summarises news."""

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
        return "news_agent"

    @property
    def description(self) -> str:
        return (
            "Fetch, search, and summarise news. "
            "Use for: latest headlines, news about a specific topic, "
            "daily briefings, and summarising news articles from URLs."
        )

    def get_tools(self) -> list[BaseTool]:
        return [*get_news_tools(), WebFetchTool()]

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a news-related request.

        Args:
            query:    Request such as "What's happening in tech today?" or
                      "Give me a briefing on the latest AI news."
            topic:    Optional explicit topic filter.

        Returns:
            AgentResponse with formatted news output.
        """
        topic = kwargs.get("topic", "")
        full_query = query
        if topic and topic.lower() not in query.lower():
            full_query = f"{query} [Focus topic: {topic}]"

        self._logger.info("NewsAgent handling: %s", query[:80])

        try:
            result = self._executor.invoke({"input": full_query})
            logger.debug(f"results from search agent: {result}")
            output = result.get("output", "Could not retrieve news.")
            tool_calls = self._extract_tool_calls(
                result.get("intermediate_steps", [])
            )
            # Extract URLs from tool outputs as lightweight references
            references = self._extract_urls(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
            )
        except Exception as exc:
            self._logger.error("NewsAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"I encountered an error fetching news: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_urls(
        tool_calls: list[tuple[str, str, str]],
    ) -> list[str]:
        """Pull URLs out of tool outputs to use as references."""
        import re

        url_pattern = re.compile(r"https?://[^\s)\"\']+")
        urls: list[str] = []
        seen: set[str] = set()
        for _, _, output in tool_calls:
            for url in url_pattern.findall(output):
                if url not in seen:
                    urls.append(url)
                    seen.add(url)
        return urls[:10]
