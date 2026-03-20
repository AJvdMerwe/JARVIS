"""
tools/search_tools.py
──────────────────────
LangChain tools for the Search Agent:
  • DuckDuckGoSearchTool  – general web search (no API key).
  • WikipediaTool         – look up Wikipedia articles.
  • WebFetchTool          – fetch and extract text from a URL.
  • CalculatorTool        – safe arithmetic evaluation.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Type
import builtins
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  DuckDuckGo Web Search
# ─────────────────────────────────────────────────────────────────────────────

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., description="The search query")
    max_results: int = Field(default=5, ge=1, le=20)


class DuckDuckGoSearchTool(BaseTool):
    """Search the web using DuckDuckGo (no API key required)."""

    name: str = "web_search"
    description: str = (
        "Search the internet for up-to-date information on any topic. "
        "Returns titles, snippets, and URLs from web search results."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            # from duckduckgo_search import DDGS  # type: ignore
            from ddgs import DDGS

            logger.debug(f"now searching the web")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                logger.debug(f"\nweb search results: {results}\n")

            if not results:
                return f"No results found for: '{query}'"

            lines = [f"## Web search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. **{r.get('title', 'No title')}**\n"
                    f"   {r.get('body', '')[:300]}\n"
                    f"   URL: {r.get('href', '')}\n"
                )
            return "\n".join(lines)

        except Exception as exc:
            logger.error("DuckDuckGo search failed: %s", exc)
            return f"Search failed: {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  Wikipedia
# ─────────────────────────────────────────────────────────────────────────────

class WikipediaInput(BaseModel):
    query: str = Field(..., description="The topic to look up on Wikipedia")
    sentences: int = Field(default=5, ge=1, le=20, description="Number of summary sentences")


class WikipediaTool(BaseTool):
    """Look up factual information on Wikipedia."""

    name: str = "wikipedia_lookup"
    description: str = (
        "Look up factual, encyclopedic information on Wikipedia. "
        "Best for definitions, historical facts, biographies, and general knowledge."
    )
    args_schema: Type[BaseModel] = WikipediaInput

    def _run(self, query: str, sentences: int = 5) -> str:
        try:
            import wikipedia  # type: ignore

            wikipedia.set_lang("en")
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
            page = wikipedia.page(query, auto_suggest=True)
            return (
                f"## Wikipedia: {page.title}\n\n"
                f"{summary}\n\n"
                f"URL: {page.url}"
            )
        except Exception as exc:
            logger.warning("Wikipedia lookup failed for '%s': %s", query, exc)
            # Fallback: return what we can
            return f"Could not retrieve Wikipedia article for '{query}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  Web page fetcher
# ─────────────────────────────────────────────────────────────────────────────

class WebFetchInput(BaseModel):
    url: str = Field(..., description="The URL to fetch content from")
    max_chars: int = Field(default=3000, ge=100, le=10000)


class WebFetchTool(BaseTool):
    """Fetch and extract readable text from a web page URL."""

    name: str = "fetch_webpage"
    description: str = (
        "Fetch and extract the main text content from a web page. "
        "Use when you have a specific URL to read."
    )
    args_schema: Type[BaseModel] = WebFetchInput

    def _run(self, url: str, max_chars: int = 3000) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup  # type: ignore

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; VirtualAssistant/1.0)"
                )
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            # Remove boilerplate tags
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            # Collapse multiple blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text[:max_chars]

            return f"## Content from: {url}\n\n{text}"

        except Exception as exc:
            logger.error("WebFetch failed for '%s': %s", url, exc)
            return f"Could not fetch '{url}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  Calculator (safe eval)
# ─────────────────────────────────────────────────────────────────────────────

class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="A mathematical expression to evaluate, e.g. '2 ** 10 + sqrt(144)'",
    )


class CalculatorTool(BaseTool):
    """Safely evaluate a mathematical expression."""

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression. Supports basic arithmetic, "
        "exponentiation (**), sqrt, floor, ceil, abs, round, log, sin, cos, tan, pi, e."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    _ALLOWED_NAMES = {
        k: v
        for k, v in vars(builtins).items()  # type: ignore[arg-type]
        if k in {"abs", "round", "min", "max", "sum", "pow"}
    } if isinstance(vars(builtins), dict) else {}

    def _run(self, expression: str) -> str:
        import math

        allowed = {
            **self._ALLOWED_NAMES,
            "sqrt": math.sqrt,
            "floor": math.floor,
            "ceil": math.ceil,
            "log": math.log,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

        # Reject anything that looks dangerous
        if any(kw in expression for kw in ["import", "exec", "eval", "__", "open"]):
            return "Expression blocked: contains forbidden keywords."

        try:
            result = eval(expression, {"builtins": {}}, allowed)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Could not evaluate '{expression}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


def get_search_tools() -> list[BaseTool]:
    """Return all search and information tools."""
    return [
        DuckDuckGoSearchTool(),
        WikipediaTool(),
        WebFetchTool(),
        CalculatorTool(),
    ]
