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
import re
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

        Rather than blindly handing every query to the ReAct executor, this
        method actively selects the most appropriate tool(s) for each query
        type, then synthesises the gathered evidence into a final answer.

        Routing strategy
        ----------------
        1. **Calculator** — detected by arithmetic operators or math keywords.
           The expression is evaluated directly; the LLM formats the answer.

        2. **Web fetch** — detected when the query contains a bare URL.
           The page is fetched first; the LLM summarises the content.

        3. **Wikipedia** — detected for definitional / encyclopedic queries
           (keywords: "what is", "who is", "define", "history of", etc.).
           Wikipedia is tried first; DuckDuckGo fills in if it fails or the
           result is thin.

        4. **Multi-source research** — all other queries.
           DuckDuckGo runs first for broad coverage; top result URLs are
           fetched for depth; Wikipedia adds encyclopedic context when the
           topic is identifiable. The LLM synthesises all sources.

        5. **ReAct fallback** — if the direct tool approach yields no useful
           content (empty output, all errors), the full ReAct executor is
           invoked so the LLM can iterate freely.

        Args:
            query:  The factual question or research task.

        Returns:
            AgentResponse with synthesised answer, tool call log,
            and deduplicated source references.
        """
        self._logger.info("SearchAgent handling: %s", query[:80])

        # Instantiate tools once and reuse across this call.
        tools        = {t.name: t for t in self.get_tools()}
        web_search   = tools.get("web_search")
        wikipedia    = tools.get("wikipedia_lookup")
        web_fetch    = tools.get("fetch_webpage")
        calculator   = tools.get("calculator")

        tool_calls: list[tuple[str, str, str]] = []
        evidence:   list[str]                  = []

        try:
            # ─────────────────────────────────────────────────────────────────
            # Route 1 — Pure arithmetic / calculation
            # ─────────────────────────────────────────────────────────────────
            if calculator and self._looks_like_math(query):
                expr   = self._extract_expression(query)
                result = calculator._run(expr)
                tool_calls.append(("calculator", expr, result))
                self._logger.debug("Calculator: %s → %s", expr, result)

                if "=" in result and "Could not" not in result:
                    answer = self._synthesise(
                        query,
                        f"Calculator result:\n{result}",
                        instruction="Present this calculation result clearly.",
                    )
                    return AgentResponse(
                        output=answer,
                        agent_name=self.name,
                        tool_calls=tool_calls,
                    )

            # ─────────────────────────────────────────────────────────────────
            # Route 2 — URL in query → fetch the page directly
            # ─────────────────────────────────────────────────────────────────
            url = self._extract_url(query) if web_fetch else None
            if url:
                page_content = web_fetch._run(url, max_chars=4000)
                tool_calls.append(("fetch_webpage", url, page_content))
                self._logger.debug("Fetched URL: %s", url)
                evidence.append(page_content)

            # ─────────────────────────────────────────────────────────────────
            # Route 3 — Encyclopedic / definitional query → Wikipedia first
            # ─────────────────────────────────────────────────────────────────
            if not url and wikipedia and self._looks_encyclopedic(query):
                topic      = self._clean_topic(query)
                wiki_result = wikipedia._run(topic, sentences=6)
                tool_calls.append(("wikipedia_lookup", topic, wiki_result))
                self._logger.debug("Wikipedia lookup: %s", topic)

                if "Could not retrieve" not in wiki_result:
                    evidence.append(wiki_result)
                    # For a clean encyclopedic hit, return without web search.
                    if self._is_sufficient(wiki_result) and not self._needs_current_data(query):
                        answer = self._synthesise(
                            query,
                            "\n\n---\n\n".join(evidence),
                            instruction=(
                                "Answer the question using the Wikipedia information. "
                                "Cite the Wikipedia source."
                            ),
                        )
                        refs = self._extract_references(tool_calls)
                        return AgentResponse(
                            output=answer,
                            agent_name=self.name,
                            tool_calls=tool_calls,
                            references=refs,
                        )
                else:
                    # Wikipedia failed — fall through to web search below.
                    self._logger.debug(
                        "Wikipedia lookup failed for '%s', trying web search.", topic
                    )

            # ─────────────────────────────────────────────────────────────────
            # Route 4 — Multi-source research (DuckDuckGo + fetch + Wikipedia)
            # ─────────────────────────────────────────────────────────────────
            if web_search and not url:
                # 4a. Broad web search.
                search_result = web_search._run(query, max_results=5)
                tool_calls.append(("web_search", query, search_result))
                self._logger.debug("Web search returned %d chars", len(search_result))
                evidence.append(search_result)

                # 4b. Fetch the top result URL for deeper content.
                top_url = self._pick_best_url(search_result)
                if top_url and web_fetch:
                    page = web_fetch._run(top_url, max_chars=3000)
                    tool_calls.append(("fetch_webpage", top_url, page))
                    self._logger.debug("Fetched top result: %s", top_url)
                    if "Could not fetch" not in page:
                        evidence.append(page)

                # 4c. Supplement with Wikipedia for definitional context.
                if wikipedia and not self._looks_encyclopedic(query):
                    topic = self._clean_topic(query)
                    wiki  = wikipedia._run(topic, sentences=4)
                    tool_calls.append(("wikipedia_lookup", topic, wiki))
                    if "Could not retrieve" not in wiki:
                        evidence.append(wiki)

            # ─────────────────────────────────────────────────────────────────
            # Fallback — no useful evidence gathered → full ReAct loop
            # ─────────────────────────────────────────────────────────────────
            if not evidence or all(
                    not e.strip() or "No results found" in e or "Could not" in e
                    for e in evidence
            ):
                self._logger.info(
                    "Direct tool calls yielded no evidence; "
                    "falling back to ReAct executor."
                )
                return self._run_react(query)

            # ─────────────────────────────────────────────────────────────────
            # Synthesise all gathered evidence into a final answer
            # ─────────────────────────────────────────────────────────────────
            combined = "\n\n---\n\n".join(evidence)
            answer   = self._synthesise(
                query,
                combined,
                instruction=(
                    "Answer the question using only the evidence provided. "
                    "Cite sources with URLs. If the evidence is insufficient, "
                    "say so clearly."
                ),
            )
            references = self._extract_references(tool_calls)

            return AgentResponse(
                output=answer,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
            )

        except Exception as exc:
            self._logger.error("SearchAgent error: %s", exc, exc_info=True)
            # Attempt ReAct fallback before giving up entirely.
            try:
                self._logger.info("Attempting ReAct fallback after error.")
                return self._run_react(query)
            except Exception:
                return AgentResponse(
                    output=f"I encountered an error while researching your question: {exc}",
                    agent_name=self.name,
                    error=str(exc),
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_react(self, query: str) -> AgentResponse:
        """
        Full ReAct executor fallback — lets the LLM freely decide which
        tools to call and in what order. Used when direct routing fails.
        """
        result     = self._executor.invoke({"input": query})
        output     = result.get("output", "No information found.")
        tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
        references = self._extract_references(tool_calls)
        return AgentResponse(
            output=output,
            agent_name=self.name,
            tool_calls=tool_calls,
            references=references,
        )

    def _synthesise(self, query: str, evidence: str, instruction: str = "") -> str:
        """
        Ask the LLM to produce a final answer given the raw tool evidence.

        Keeps the evidence in a structured block so the model doesn't
        confuse source material with the conversation history.
        """
        prompt = (
            f"Question: {query}\n\n"
            f"Evidence gathered from search tools:\n"
            f"{'─' * 60}\n"
            f"{evidence[:6000]}\n"
            f"{'─' * 60}\n\n"
        )
        if instruction:
            prompt += f"{instruction}\n\n"
        prompt += "Answer:"

        response = self._llm.invoke(prompt)
        return str(response.content).strip()

    # ── Intent detectors ──────────────────────────────────────────────────────

    _MATH_PATTERNS = re.compile(
        r"(\d+\s*[\+\-\*\/\%\^]+\s*\d+"          # 3 + 4, 10 * 5
        r"|\b(sqrt|log|sin|cos|tan|pi|abs)\s*\("  # sqrt(9)
        r"|\d+\s*\*\*\s*\d+"                       # 2**8
        r"|\d+\s*%\s*(of\s*)?\d*"                  # 15% of 200
        r"|\b(calculate|compute|evaluate|"
        r"what is \d|how much is \d|"
        r"\d+\s+percent\s+of)\b)",
        re.IGNORECASE,
    )

    _ENCYCLOPEDIC_PATTERNS = re.compile(
        r"\b(what is|what are|who is|who was|who were|"
        r"define|definition of|meaning of|"
        r"history of|origin of|biography|"
        r"explain what|tell me about|"
        r"invented by|discovered by|founded by|"
        r"when was|where is|capital of|"
        r"how does .+ work)\b",
        re.IGNORECASE,
    )

    _CURRENT_DATA_PATTERNS = re.compile(
        r"\b(latest|current|today|now|recent|"
        r"this year|this month|this week|"
        r"stock price|weather|news|score|standings)\b",
        re.IGNORECASE,
    )

    _URL_PATTERN = re.compile(r"https?://[^\s]+")

    @classmethod
    def _looks_like_math(cls, query: str) -> bool:
        return bool(cls._MATH_PATTERNS.search(query))

    @classmethod
    def _looks_encyclopedic(cls, query: str) -> bool:
        return bool(cls._ENCYCLOPEDIC_PATTERNS.search(query))

    @classmethod
    def _needs_current_data(cls, query: str) -> bool:
        return bool(cls._CURRENT_DATA_PATTERNS.search(query))

    @classmethod
    def _extract_url(cls, query: str) -> Optional[str]:
        """Return the first URL found in the query, or None."""
        match = cls._URL_PATTERN.search(query)
        return match.group(0) if match else None

    @staticmethod
    def _extract_expression(query: str) -> str:
        """
        Pull the arithmetic expression out of a natural-language query.
        Falls back to the full query if no clean expression is found.
        """
        # Strip common question wrappers
        cleaned = re.sub(
            r"(?i)(what is|calculate|compute|evaluate|how much is|"
            r"what'?s|find|solve)\s*",
            "",
            query,
        ).strip().rstrip("?.")
        return cleaned or query

    @staticmethod
    def _clean_topic(query: str) -> str:
        """
        Extract a clean topic keyword for Wikipedia lookups by removing
        question words, articles, and filler phrases.
        """
        topic = re.sub(
            r"(?i)(what is|what are|who is|who was|who were|"
            r"tell me about|explain|define|the history of|"
            r"a brief|an introduction to|"
            r"\?|please|can you|could you)\s*",
            "",
            query,
        ).strip().rstrip("?.")
        # Capitalise for Wikipedia's title matching
        return topic[:100]

    @staticmethod
    def _is_sufficient(text: str, min_chars: int = 200) -> bool:
        """Return True if the text has enough content to be a real answer."""
        return len(text.strip()) >= min_chars

    @staticmethod
    def _pick_best_url(search_output: str) -> Optional[str]:
        """
        Parse the DuckDuckGo search output and return the first result URL
        that looks like a real article (skips social media, ads, etc.).
        """
        _SKIP_DOMAINS = {
            "twitter.com", "x.com", "facebook.com", "instagram.com",
            "reddit.com", "tiktok.com", "youtube.com", "pinterest.com",
        }
        url_re = re.compile(r"URL:\s*(https?://[^\s\n]+)")
        for match in url_re.finditer(search_output):
            url = match.group(1).strip()
            try:
                hostname = url.split("/")[2].lower()
                # Remove port, then strip www. subdomains for matching
                bare = re.sub(r"^www\.", "", hostname.split(":")[0])
                # Check both hostname and bare domain against skip list
                if bare not in _SKIP_DOMAINS and hostname not in _SKIP_DOMAINS:
                    return url
            except IndexError:
                continue
        return None

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