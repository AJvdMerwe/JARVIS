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
import re
from datetime import datetime, timezone
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
        Handle a news-related request by selecting and invoking the most
        appropriate combination of tools directly, then synthesising results.

        Routing strategy
        ----------------
        1. **RSS feed URL in query** — the URL is passed straight to
           ``fetch_rss_feed`` to pull the feed, then the LLM summarises it.

        2. **Article URL in query** — ``fetch_webpage`` retrieves the full
           article text; the LLM extracts the key news points.

        3. **Topic-specific query** — ``search_topic_news`` fetches recent
           DuckDuckGo news results for the extracted topic. When the query
           also signals breadth (e.g. "everything about X", "briefing on X"),
           ``get_headlines`` is also called with the topic as a category filter
           so RSS sources are included alongside real-time search results.

        4. **General headlines / daily briefing** — ``get_headlines`` fetches
           across all configured RSS feeds. When a category keyword is detected
           (tech, finance, sports, science, …) it is passed as a filter. For
           explicit briefing requests, ``search_topic_news`` is also called for
           the top three news categories to round out the picture.

        5. **ReAct fallback** — if direct tool calls return empty or error
           output, the full ReAct executor is invoked so the LLM can iterate.

        Args:
            query:  The news request (e.g. "What's happening in AI today?",
                    "Give me a tech briefing", "Summarise https://…").
            topic:  Optional explicit topic override — appended to the query
                    when not already present.
            max_articles: Override the default article count (default 10).

        Returns:
            AgentResponse with formatted Markdown news output and source URLs
            as references.
        """
        # ── Resolve kwargs ────────────────────────────────────────────────────
        topic        = kwargs.get("topic", "")
        max_articles = int(kwargs.get("max_articles", 10))

        # Merge explicit topic into the query when absent
        if topic and topic.lower() not in query.lower():
            query = f"{query} — topic: {topic}"

        self._logger.info("NewsAgent handling: %s", query[:80])

        # ── Instantiate tools once ────────────────────────────────────────────
        tools         = {t.name: t for t in self.get_tools()}
        headlines     = tools.get("get_headlines")
        topic_search  = tools.get("search_topic_news")
        rss_fetch     = tools.get("fetch_rss_feed")
        web_fetch     = tools.get("fetch_webpage")

        tool_calls: list[tuple[str, str, str]] = []
        evidence:   list[str]                  = []

        try:
            # ─────────────────────────────────────────────────────────────────
            # Route 1 — RSS feed URL detected in query
            # ─────────────────────────────────────────────────────────────────
            rss_url = self._extract_rss_url(query) if rss_fetch else None
            if rss_url:
                raw = rss_fetch._run(url=rss_url, max_items=max_articles)
                tool_calls.append(("fetch_rss_feed", rss_url, raw))
                self._logger.debug("Fetched RSS feed: %s", rss_url)
                evidence.append(raw)

            # ─────────────────────────────────────────────────────────────────
            # Route 2 — Article URL detected in query → fetch full text
            # ─────────────────────────────────────────────────────────────────
            elif self._has_article_url(query) and web_fetch:
                article_url = self._extract_article_url(query)
                raw = web_fetch._run(url=article_url, max_chars=5000)
                tool_calls.append(("fetch_webpage", article_url, raw))
                self._logger.debug("Fetched article: %s", article_url)
                if "Could not fetch" not in raw:
                    evidence.append(raw)
                else:
                    # Fallback: search for the URL's domain as a topic
                    fallback_topic = self._domain_from_url(article_url)
                    if topic_search and fallback_topic:
                        raw2 = topic_search._run(topic=fallback_topic, max_results=5)
                        tool_calls.append(("search_topic_news", fallback_topic, raw2))
                        evidence.append(raw2)

            # ─────────────────────────────────────────────────────────────────
            # Route 3 — Topic-specific query
            # ─────────────────────────────────────────────────────────────────
            elif self._is_topic_query(query):
                detected_topic = topic or self._extract_topic(query)
                self._logger.debug("Topic query detected: '%s'", detected_topic)

                # 3a. Real-time DuckDuckGo news search for the topic.
                if topic_search and detected_topic:
                    raw = topic_search._run(
                        topic=detected_topic,
                        max_results=min(max_articles, 8),
                    )
                    tool_calls.append(("search_topic_news", detected_topic, raw))
                    self._logger.debug(
                        "Topic news for '%s': %d chars", detected_topic, len(raw)
                    )
                    if "No news found" not in raw:
                        evidence.append(raw)

                # 3b. Supplement with RSS headlines filtered to the same topic
                #     when the query requests breadth ("everything about X",
                #     "briefing on X", "all news on X").
                if headlines and detected_topic and self._wants_breadth(query):
                    raw_hl = headlines._run(
                        max_articles=max_articles,
                        category=detected_topic,
                    )
                    tool_calls.append(("get_headlines", detected_topic, raw_hl))
                    self._logger.debug("RSS headlines for category '%s'", detected_topic)
                    if "No articles found" not in raw_hl:
                        evidence.append(raw_hl)

            # ─────────────────────────────────────────────────────────────────
            # Route 4 — General headlines / daily briefing
            # ─────────────────────────────────────────────────────────────────
            else:
                category = self._extract_category(query)
                self._logger.debug(
                    "General headlines request (category=%s)", category or "all"
                )

                # 4a. Pull RSS headlines (with optional category filter).
                if headlines:
                    raw_hl = headlines._run(
                        max_articles=max_articles,
                        category=category,
                    )
                    tool_calls.append(("get_headlines", category or "all", raw_hl))
                    if "No articles found" not in raw_hl:
                        evidence.append(raw_hl)

                # 4b. For explicit briefing requests, augment with real-time
                #     search results across the top news categories.
                if self._is_briefing_request(query) and topic_search:
                    briefing_topics = self._briefing_topics(category)
                    for bt in briefing_topics:
                        raw_bt = topic_search._run(topic=bt, max_results=3)
                        tool_calls.append(("search_topic_news", bt, raw_bt))
                        self._logger.debug("Briefing topic '%s'", bt)
                        if "No news found" not in raw_bt:
                            evidence.append(raw_bt)

            # ─────────────────────────────────────────────────────────────────
            # Fallback — no useful content gathered → full ReAct loop
            # ─────────────────────────────────────────────────────────────────
            if not evidence or all(
                    not e.strip()
                    or "No articles found" in e
                    or "No news found" in e
                    or "Could not" in e
                    for e in evidence
            ):
                self._logger.info(
                    "Direct tools yielded no news; falling back to ReAct executor."
                )
                return self._run_react(query)

            # ─────────────────────────────────────────────────────────────────
            # Synthesise all gathered evidence into a final formatted response
            # ─────────────────────────────────────────────────────────────────
            combined = "\n\n---\n\n".join(evidence)
            output   = self._synthesise(query, combined)
            refs     = self._extract_urls(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=refs,
            )

        except Exception as exc:
            self._logger.error("NewsAgent error: %s", exc, exc_info=True)
            try:
                self._logger.info("Attempting ReAct fallback after error.")
                return self._run_react(query)
            except Exception:
                return AgentResponse(
                    output=f"I encountered an error fetching news: {exc}",
                    agent_name=self.name,
                    error=str(exc),
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_react(self, query: str) -> AgentResponse:
        """Full ReAct executor fallback."""
        result     = self._executor.invoke({"input": query})
        output     = result.get("output", "Could not retrieve news.")
        tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
        refs       = self._extract_urls(tool_calls)
        return AgentResponse(
            output=output,
            agent_name=self.name,
            tool_calls=tool_calls,
            references=refs,
        )

    def _synthesise(self, query: str, evidence: str) -> str:
        """
        Ask the LLM to format the raw tool output into a clean news response.
        The evidence is wrapped in a clear block so the model doesn't confuse
        source material with conversation history.
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        prompt = (
            f"News request: {query}\n"
            f"Timestamp: {ts}\n\n"
            f"Raw news data from tools:\n"
            f"{'─' * 60}\n"
            f"{evidence[:8000]}\n"
            f"{'─' * 60}\n\n"
            "Instructions:\n"
            "- Present the news clearly with Markdown headings.\n"
            "- Summarise each story in 1–3 sentences.\n"
            "- Always cite the source name and URL for each item.\n"
            "- Group related stories under subheadings where appropriate.\n"
            "- Remain factual and neutral.\n"
            "- Never invent or embellish details not present in the data.\n\n"
            "Formatted news response:"
        )
        response = self._llm.invoke(prompt)
        return str(response.content).strip()

    # ── Intent detectors ──────────────────────────────────────────────────────

    # Patterns that signal a general headlines / daily-briefing request
    _BRIEFING_PATTERNS = re.compile(
        r"\b(briefing|daily (news|brief|roundup|digest)|"
        r"what'?s (happening|going on|in the news)|"
        r"top (stories|headlines|news)|"
        r"latest (news|headlines|stories|updates)|"
        r"news (today|this morning|tonight|this week)|"
        r"give me (the news|headlines|a summary of today))\b",
        re.IGNORECASE,
    )

    # Patterns that indicate the user wants news about a specific topic
    _TOPIC_PATTERNS = re.compile(
        r"\b(news (about|on|regarding|covering)|"
        r"(latest|recent|new|current) .{3,40} (news|stories|updates|developments)|"
        r"what'?s (happening|going on) (with|in) |"
        r"(headlines|updates|coverage) (about|on)|"
        r"tell me about .{3,30} news|"
        r"any news (on|about))\b",
        re.IGNORECASE,
    )

    # Signals that the user wants broad coverage, not just search results
    _BREADTH_PATTERNS = re.compile(
        r"\b(everything|comprehensive|all (news|stories|coverage)|"
        r"full briefing|roundup|digest|deep.?dive|in depth|overview)\b",
        re.IGNORECASE,
    )

    # Category keywords mapped to clean category strings for the headline filter
    _CATEGORY_KEYWORDS: dict[str, str] = {
        "tech": "technology",          "technology": "technology",
        "ai": "artificial intelligence", "artificial intelligence": "artificial intelligence",
        "finance": "finance",           "financial": "finance",
        "business": "business",         "economy": "economy",
        "sport": "sports",              "sports": "sports",
        "science": "science",
        "health": "health",             "medical": "health",
        "politic": "politics",          "government": "politics",
        "world": "world",               "international": "world",
        "climate": "climate",           "environment": "environment",
        "entertainment": "entertainment", "celebrity": "entertainment",
        "crypto": "cryptocurrency",     "bitcoin": "cryptocurrency",
    }

    # RSS feed URL pattern (ends with /feed, /rss, .xml, /atom, etc.)
    _RSS_URL_RE = re.compile(
        r"https?://[^\s]+(?:/feed|/rss|/atom|\.xml|/rss\.xml|/feed\.xml)[^\s]*",
        re.IGNORECASE,
    )

    # Generic article URL pattern
    _ARTICLE_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)

    @classmethod
    def _is_briefing_request(cls, query: str) -> bool:
        return bool(cls._BRIEFING_PATTERNS.search(query))

    @classmethod
    def _is_topic_query(cls, query: str) -> bool:
        return bool(cls._TOPIC_PATTERNS.search(query))

    @classmethod
    def _wants_breadth(cls, query: str) -> bool:
        return bool(cls._BREADTH_PATTERNS.search(query))

    @classmethod
    def _extract_rss_url(cls, query: str) -> Optional[str]:
        """Return the first RSS/Atom feed URL found in the query."""
        m = cls._RSS_URL_RE.search(query)
        return m.group(0) if m else None

    @classmethod
    def _has_article_url(cls, query: str) -> bool:
        return bool(cls._ARTICLE_URL_RE.search(query))

    @classmethod
    def _extract_article_url(cls, query: str) -> str:
        m = cls._ARTICLE_URL_RE.search(query)
        return m.group(0) if m else ""

    @staticmethod
    def _domain_from_url(url: str) -> str:
        """Extract a readable domain name for use as a fallback topic."""
        try:
            domain = url.split("/")[2]
            # Strip www., TLD, and punctuation to get a bare name
            bare = re.sub(r"^www\.", "", domain).split(".")[0]
            return bare
        except IndexError:
            return ""

    @classmethod
    def _extract_category(cls, query: str) -> Optional[str]:
        """
        Detect a news category keyword in the query and return the normalised
        category string, or None for general/uncategorised requests.
        """
        ql = query.lower()
        for kw, category in cls._CATEGORY_KEYWORDS.items():
            if kw in ql:
                return category
        return None

    @staticmethod
    def _extract_topic(query: str) -> str:
        """
        Pull a clean topic keyword out of a topic-specific news query by
        stripping question wrappers and filler phrases.
        """
        cleaned = re.sub(
            r"(?i)(what'?s (happening|going on) (with|in|around)?|"
            r"news (about|on|regarding|covering)|"
            r"(latest|recent|any|current|new) (news|stories|updates|developments)?\s*(about|on|for)?|"
            r"tell me about\s+|"
            r"give me\s+|"
            r"(headlines|coverage|updates) (about|on)\s*)",
            "",
            query,
        ).strip().strip("?.")
        # Trim to a sensible search length
        return cleaned[:80] if cleaned else query[:80]

    @staticmethod
    def _briefing_topics(category: Optional[str] = None) -> list[str]:
        """
        Return the topic list to search when building a daily briefing.
        If a category is given, return focused sub-topics; otherwise use
        a broad cross-section of news areas.
        """
        if category == "technology":
            return ["technology news", "AI developments", "cybersecurity"]
        if category == "finance":
            return ["financial markets", "economy", "business news"]
        if category == "sports":
            return ["sports results", "football", "athletics"]
        # Default: broad cross-section for a general daily briefing
        return ["world news", "technology", "business"]

    # ── Reference extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_urls(
            tool_calls: list[tuple[str, str, str]],
    ) -> list[str]:
        """Pull URLs out of tool outputs to use as source references."""
        url_pattern = re.compile(r"https?://[^\s\)\"\']+")
        urls: list[str] = []
        seen: set[str] = set()
        for _, _, output in tool_calls:
            for url in url_pattern.findall(output):
                if url not in seen:
                    urls.append(url)
                    seen.add(url)
        return urls[:10]