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

from langchain_core.tools import BaseTool

# Lazy imports used in tool methods (placed here for patchability in tests)
try:
    from playwright.async_api import async_playwright  # type: ignore
except ImportError:
    async_playwright = None  # type: ignore

try:
    from core.llm_manager import get_llm as _get_llm_for_rerank
except ImportError:
    _get_llm_for_rerank = None  # type: ignore
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
            from duckduckgo_search import DDGS  # type: ignore

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

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


def _safe_builtins_subset() -> dict:
    """Extract a safe subset of builtins, handling both dict and module forms."""
    import builtins as _builtins_mod
    _allowed_names = {"abs", "round", "min", "max", "sum", "pow"}
    return {k: getattr(_builtins_mod, k) for k in _allowed_names if hasattr(_builtins_mod, k)}


class CalculatorTool(BaseTool):
    """Safely evaluate a mathematical expression."""

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression. Supports basic arithmetic, "
        "exponentiation (**), sqrt, floor, ceil, abs, round, log, sin, cos, tan, pi, e."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    _ALLOWED_NAMES: dict = {}  # populated at first use via _safe_builtins_subset()

    def _run(self, expression: str) -> str:
        import math

        allowed = {
            **_safe_builtins_subset(),
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
            result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
            return f"{expression} = {result}"
        except Exception as exc:
            return f"Could not evaluate '{expression}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  Re-ranked Web Search
# ─────────────────────────────────────────────────────────────────────────────

class RankedSearchInput(BaseModel):
    query:       str = Field(..., description="The search query")
    max_results: int = Field(default=8, ge=2, le=20,
                             description="Number of results to fetch before re-ranking")
    top_k:       int = Field(default=3, ge=1, le=10,
                             description="Number of top results to return after re-ranking")


class RankedWebSearchTool(BaseTool):
    """
    Web search with LLM-based relevance re-ranking.

    Fetches up to ``max_results`` raw results from DuckDuckGo, then uses the
    LLM to score each result's relevance to the query and returns only the
    top ``top_k`` most relevant results.

    Use this instead of ``web_search`` for research queries where quality
    matters more than speed, or when the raw results often contain irrelevant
    content.
    """
    name: str        = "ranked_web_search"
    description: str = (
        "Web search with re-ranking: fetches multiple results then uses AI to "
        "select only the most relevant ones. Use for research or when you need "
        "high-quality, on-topic results. "
        "Inputs: query (str), max_results (int, default 8), top_k (int, default 3)."
    )
    args_schema: type[BaseModel] = RankedSearchInput

    def _run(
        self,
        query:       str,
        max_results: int = 8,
        top_k:       int = 3,
        **kwargs,
    ) -> str:
        # ── Step 1: fetch raw results ──────────────────────────────────────
        try:
            from duckduckgo_search import DDGS  # type: ignore
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
        except Exception as exc:
            logger.error("DuckDuckGo fetch failed: %s", exc)
            return f"Search failed: {exc}"

        if not raw:
            return f"No results found for: '{query}'"

        if len(raw) <= top_k:
            # Not enough results to bother re-ranking
            return _format_search_results(query, raw)

        # ── Step 2: LLM re-ranking ────────────────────────────────────────
        try:
            ranked = _llm_rerank(query, raw, top_k=top_k)
        except Exception as exc:
            logger.warning("Re-ranking failed (%s); returning raw top-%d.", exc, top_k)
            ranked = raw[:top_k]

        return _format_search_results(query, ranked, note=f"(re-ranked from {len(raw)} results)")

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


def _llm_rerank(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    """
    Use the LLM to score and re-rank search results by relevance to the query.

    Sends a compact prompt with result titles + snippets and asks the model to
    return the indices of the most relevant results in order.

    Parameters
    ----------
    query   : str
    results : list[dict]  Raw DuckDuckGo result dicts.
    top_k   : int         How many top results to return.

    Returns
    -------
    list[dict]  Re-ordered subset of *results*.
    """
    get_llm = _get_llm_for_rerank
    import re as _re

    # Build compact representation
    items = []
    for i, r in enumerate(results):
        title   = r.get("title", "No title")[:80]
        snippet = r.get("body", "")[:150]
        items.append(f"[{i}] {title} — {snippet}")

    prompt = (
        f"Query: {query}\n\n"
        "Rate these search results by relevance to the query. "
        f"Return ONLY the indices of the top {top_k} most relevant results, "
        "separated by commas, most relevant first. "
        "Example: 2,0,5\n\n"
        + "\n".join(items) + "\n\nTop indices:"
    )

    llm    = get_llm()
    result = llm.invoke(prompt)
    raw    = str(result.content).strip()

    # Parse indices
    indices = [int(x.strip()) for x in _re.findall(r"\d+", raw)
               if int(x.strip()) < len(results)]

    if not indices:
        return results[:top_k]

    seen   = set()
    ranked = []
    for idx in indices:
        if idx not in seen:
            ranked.append(results[idx])
            seen.add(idx)
        if len(ranked) >= top_k:
            break

    # Pad with unreranked results if needed
    for r in results:
        if len(ranked) >= top_k:
            break
        if r not in ranked:
            ranked.append(r)

    return ranked


def _format_search_results(query: str, results: list[dict], note: str = "") -> str:
    heading = f"## Web search results for: {query}"
    if note:
        heading += f"  {note}"
    lines = [heading + "\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. **{r.get('title', 'No title')}**\n"
            f"   {r.get('body', '')[:300]}\n"
            f"   URL: {r.get('href', '')}\n"
        )
    return "\n".join(lines)



# ─────────────────────────────────────────────────────────────────────────────
#  Playwright web fetcher (handles JavaScript-rendered pages)
# ─────────────────────────────────────────────────────────────────────────────

class JSFetchInput(BaseModel):
    url:       str = Field(..., description="The URL to fetch (supports JS-heavy pages)")
    wait_for:  str = Field("domcontentloaded",
                            description="Wait condition: load | domcontentloaded | networkidle")
    max_chars: int = Field(default=4000, ge=500, le=20000)
    click_selector: str = Field(
        default="",
        description="Optional CSS selector to click before extracting (e.g. to dismiss cookie banners)"
    )


class PlaywrightFetchTool(BaseTool):
    """
    Fetch a web page with full JavaScript execution using a headless Chromium browser.

    Use this instead of ``fetch_webpage`` when:
    - The page is a single-page application (React/Vue/Angular)
    - Content is loaded dynamically after the initial HTML
    - The simple HTTP fetch returns empty or incomplete content
    - You need to interact with the page before reading it (e.g. dismiss modals)

    This tool is slower than ``fetch_webpage`` (~3-8 seconds per page) but handles
    virtually any modern web page.
    """
    name: str        = "fetch_webpage_js"
    description: str = (
        "Fetch a web page with full JavaScript execution (handles SPAs and dynamic content). "
        "Slower than fetch_webpage but works on any modern website. "
        "Inputs: url (str), wait_for (str, default 'domcontentloaded'), "
        "max_chars (int, default 4000), click_selector (str, optional CSS selector to click)."
    )
    args_schema: type[BaseModel] = JSFetchInput

    def _run(
        self,
        url:            str,
        wait_for:       str  = "domcontentloaded",
        max_chars:      int  = 4000,
        click_selector: str  = "",
        **kwargs,
    ) -> str:
        import asyncio
        try:
            return asyncio.run(
                self._fetch_async(url, wait_for, max_chars, click_selector)
            )
        except Exception as exc:
            logger.error("Playwright fetch failed for '%s': %s", url, exc)
            return f"Could not fetch '{url}' with JS rendering: {exc}"

    async def _fetch_async(
        self,
        url:            str,
        wait_for:       str,
        max_chars:      int,
        click_selector: str,
    ) -> str:
        import re as _re
        _ap = async_playwright

        async with _ap() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                await page.set_extra_http_headers({
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                })

                await page.goto(url, wait_until=wait_for, timeout=20000)

                # Optional: click a selector (e.g. cookie dismiss, load-more)
                if click_selector.strip():
                    try:
                        await page.click(click_selector, timeout=3000)
                        await page.wait_for_timeout(500)
                    except Exception:
                        pass  # selector not found — continue anyway

                # Extract visible text, stripping script/style noise
                text = await page.evaluate("""() => {
                    // Remove scripts, styles, nav, footer
                    const remove = document.querySelectorAll(
                        'script,style,nav,footer,header,[aria-hidden=true]'
                    );
                    remove.forEach(el => el.remove());
                    return document.body ? document.body.innerText : '';
                }""")

                # Clean whitespace
                text = _re.sub(r"\n{3,}", "\n\n", text.strip())
                text = _re.sub(r" {2,}", " ", text)

                if not text.strip():
                    # Fallback: get all text content
                    text = await page.inner_text("body")

                if len(text) > max_chars:
                    text = text[:max_chars].rstrip() + "\n\n[content truncated]"

                title = await page.title()
                return f"# {title}\n\nURL: {url}\n\n{text}"

            finally:
                await browser.close()

    async def _arun(self, **kwargs) -> str:
        return await self._fetch_async(
            kwargs.get("url", ""),
            kwargs.get("wait_for", "domcontentloaded"),
            kwargs.get("max_chars", 4000),
            kwargs.get("click_selector", ""),
        )


def get_search_tools() -> list[BaseTool]:
    """Return all search and information tools."""
    return [
        DuckDuckGoSearchTool(),
        RankedWebSearchTool(),
        WikipediaTool(),
        WebFetchTool(),
        PlaywrightFetchTool(),
        CalculatorTool(),
    ]
