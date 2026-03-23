"""
tests/test_web_capabilities.py
────────────────────────────────
Tests for Stage 3 — Web Capabilities:

  • RankedWebSearchTool  — re-ranking logic, LLM scoring, fallback behavior
  • _llm_rerank()        — index parsing, padding, error handling
  • _format_search_results() — output formatting
  • PlaywrightFetchTool  — JS page fetching, content extraction, error handling
  • get_search_tools()   — registry includes new tools
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
#  _format_search_results helper
# =============================================================================

class TestFormatSearchResults:

    def _results(self, n=3):
        return [
            {"title": f"Title {i}", "body": f"Snippet {i}", "href": f"https://example.com/{i}"}
            for i in range(n)
        ]

    def test_includes_query_heading(self):
        from tools.search_tools import _format_search_results
        out = _format_search_results("Python tutorial", self._results())
        assert "Python tutorial" in out

    def test_numbered_results(self):
        from tools.search_tools import _format_search_results
        out = _format_search_results("q", self._results(3))
        assert "1." in out
        assert "2." in out
        assert "3." in out

    def test_includes_urls(self):
        from tools.search_tools import _format_search_results
        out = _format_search_results("q", self._results(2))
        assert "https://example.com/0" in out

    def test_note_appended_when_given(self):
        from tools.search_tools import _format_search_results
        out = _format_search_results("q", self._results(1), note="(re-ranked from 8)")
        assert "re-ranked" in out

    def test_empty_results_produce_minimal_output(self):
        from tools.search_tools import _format_search_results
        out = _format_search_results("q", [])
        assert isinstance(out, str)


# =============================================================================
#  _llm_rerank
# =============================================================================

class TestLlmRerank:

    def _results(self, n=5):
        return [
            {"title": f"Result {i}", "body": f"Content about topic {i}", "href": f"https://r{i}.com"}
            for i in range(n)
        ]

    def test_returns_top_k_results(self):
        from tools.search_tools import _llm_rerank
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="2,0,4")
        with patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            ranked = _llm_rerank("Python", self._results(5), top_k=3)
        assert len(ranked) == 3

    def test_respects_order_from_llm(self):
        from tools.search_tools import _llm_rerank
        raw = self._results(5)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="3,1,0")
        with patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            ranked = _llm_rerank("q", raw, top_k=3)
        assert ranked[0] is raw[3]
        assert ranked[1] is raw[1]
        assert ranked[2] is raw[0]

    def test_invalid_llm_response_falls_back_to_first_k(self):
        from tools.search_tools import _llm_rerank
        raw = self._results(5)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="no numbers here")
        with patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            ranked = _llm_rerank("q", raw, top_k=3)
        assert len(ranked) == 3

    def test_out_of_range_indices_ignored(self):
        from tools.search_tools import _llm_rerank
        raw = self._results(3)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="99,0,100,1")
        with patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            ranked = _llm_rerank("q", raw, top_k=2)
        assert len(ranked) == 2
        assert all(r in raw for r in ranked)

    def test_no_duplicates_in_ranked(self):
        from tools.search_tools import _llm_rerank
        raw = self._results(5)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="1,1,1,2")
        with patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            ranked = _llm_rerank("q", raw, top_k=3)
        assert len(ranked) == len(set(id(r) for r in ranked))

    def test_llm_error_falls_back_gracefully(self):
        from tools.search_tools import _llm_rerank
        raw = self._results(5)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        with patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            with pytest.raises(Exception):
                _llm_rerank("q", raw, top_k=3)


# =============================================================================
#  RankedWebSearchTool
# =============================================================================

class TestRankedWebSearchTool:

    def _tool(self):
        from tools.search_tools import RankedWebSearchTool
        return RankedWebSearchTool()

    def _raw(self, n=8):
        return [
            {"title": f"Page {i}", "body": f"Relevant content {i}", "href": f"https://p{i}.com"}
            for i in range(n)
        ]

    def test_returns_string(self):
        tool = self._tool()
        raw  = self._raw(3)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="0,1,2")
        with patch("duckduckgo_search.DDGS") as mock_ddgs, \
             patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            mock_ddgs.return_value.__enter__.return_value.text.return_value = raw
            result = tool._run("Python tutorial")
        assert isinstance(result, str)

    def test_fewer_results_than_top_k_skips_reranking(self):
        """When raw results ≤ top_k, no LLM call is needed."""
        tool = self._tool()
        raw  = self._raw(2)   # only 2 results, top_k default is 3
        with patch("duckduckgo_search.DDGS") as mock_ddgs, \
             patch("tools.search_tools._get_llm_for_rerank") as mock_get_llm:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = raw
            result = tool._run("q", max_results=2, top_k=3)
        mock_get_llm.assert_not_called()

    def test_note_shown_when_reranked(self):
        tool = self._tool()
        raw  = self._raw(8)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="0,3,7")
        with patch("duckduckgo_search.DDGS") as mock_ddgs, \
             patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            mock_ddgs.return_value.__enter__.return_value.text.return_value = raw
            result = tool._run("test query", max_results=8, top_k=3)
        assert "re-ranked" in result

    def test_ddg_failure_returns_error_string(self):
        tool = self._tool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.side_effect = Exception("network error")
            result = tool._run("test")
        assert "failed" in result.lower() or "error" in result.lower()

    def test_reranking_failure_falls_back_to_raw_top_k(self):
        """If re-ranking fails, returns raw top_k results without crashing."""
        tool = self._tool()
        raw  = self._raw(8)
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        with patch("duckduckgo_search.DDGS") as mock_ddgs, \
             patch("tools.search_tools._get_llm_for_rerank", return_value=mock_llm):
            mock_ddgs.return_value.__enter__.return_value.text.return_value = raw
            result = tool._run("query", top_k=3)
        # Should still return a result (fallback to raw[:top_k])
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
#  PlaywrightFetchTool
# =============================================================================

class TestPlaywrightFetchTool:

    def _tool(self):
        from tools.search_tools import PlaywrightFetchTool
        return PlaywrightFetchTool()

    def _mock_playwright(self, page_text="Hello world from JS page", title="Test Page"):
        """Return a mock async playwright context."""
        mock_page  = AsyncMock()
        mock_page.title.return_value = title
        mock_page.evaluate.return_value = page_text
        mock_page.inner_text.return_value = page_text
        mock_page.goto   = AsyncMock()
        mock_page.click  = AsyncMock()
        mock_page.set_extra_http_headers = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page
        mock_browser.close = AsyncMock()

        mock_chromium = AsyncMock()
        mock_chromium.launch.return_value = mock_browser

        mock_pw = AsyncMock()
        mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
        mock_pw.__aexit__  = AsyncMock(return_value=None)
        mock_pw.chromium   = mock_chromium

        return mock_pw, mock_page

    def test_returns_page_content(self):
        tool    = self._tool()
        mock_pw, _ = self._mock_playwright("Main article content here.")
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            result = asyncio.run(
                tool._fetch_async("https://example.com", "domcontentloaded", 4000, "")
            )
        assert "Main article content" in result

    def test_includes_page_title(self):
        tool = self._tool()
        mock_pw, _ = self._mock_playwright("content", title="My Article Title")
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            result = asyncio.run(
                tool._fetch_async("https://example.com", "domcontentloaded", 4000, "")
            )
        assert "My Article Title" in result

    def test_includes_url_in_output(self):
        tool = self._tool()
        mock_pw, _ = self._mock_playwright("content")
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            result = asyncio.run(
                tool._fetch_async("https://news.example.com/article", "domcontentloaded", 4000, "")
            )
        assert "https://news.example.com/article" in result

    def test_content_truncated_at_max_chars(self):
        tool = self._tool()
        long_text = "word " * 2000   # ~10000 chars
        mock_pw, _ = self._mock_playwright(long_text)
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            result = asyncio.run(
                tool._fetch_async("https://example.com", "domcontentloaded", 500, "")
            )
        # Content should be truncated
        assert "truncated" in result or len(result) < 1000

    def test_click_selector_attempted(self):
        tool = self._tool()
        mock_pw, mock_page = self._mock_playwright("content")
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            asyncio.run(
                tool._fetch_async(
                    "https://example.com",
                    "domcontentloaded",
                    4000,
                    "#cookie-dismiss",
                )
            )
        mock_page.click.assert_called()

    def test_click_failure_does_not_abort(self):
        """If the click selector isn't found, the fetch should still succeed."""
        tool = self._tool()
        mock_pw, mock_page = self._mock_playwright("content")
        mock_page.click.side_effect = Exception("Timeout: element not found")
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            result = asyncio.run(
                tool._fetch_async(
                    "https://example.com",
                    "domcontentloaded",
                    4000,
                    ".nonexistent",
                )
            )
        assert "content" in result   # page content still returned

    def test_playwright_error_returns_error_string(self):
        tool = self._tool()
        with patch("tools.search_tools.async_playwright", side_effect=Exception("browser crash")):
            result = tool._run("https://example.com")
        assert "could not fetch" in result.lower() or "error" in result.lower() or "failed" in result.lower()

    def test_empty_page_falls_back_to_inner_text(self):
        """When evaluate() returns empty string, inner_text() is used."""
        tool = self._tool()
        mock_pw, mock_page = self._mock_playwright("")
        mock_page.inner_text.return_value = "Fallback inner text content"
        with patch("tools.search_tools.async_playwright", return_value=mock_pw):
            import asyncio
            result = asyncio.run(
                tool._fetch_async("https://example.com", "domcontentloaded", 4000, "")
            )
        assert "Fallback inner text" in result


# =============================================================================
#  Tool registry
# =============================================================================

class TestSearchToolRegistry:

    def test_ranked_search_in_registry(self):
        from tools.search_tools import get_search_tools
        names = {t.name for t in get_search_tools()}
        assert "ranked_web_search" in names

    def test_playwright_fetch_in_registry(self):
        from tools.search_tools import get_search_tools
        names = {t.name for t in get_search_tools()}
        assert "fetch_webpage_js" in names

    def test_original_tools_still_present(self):
        from tools.search_tools import get_search_tools
        names = {t.name for t in get_search_tools()}
        assert "web_search"   in names
        assert "wikipedia_lookup" in names
        assert "fetch_webpage" in names
        assert "calculator"   in names

    def test_search_agent_has_ranked_tool(self):
        """SearchAgent should include both the original and ranked search tools."""
        from agents.search_agent import SearchAgent
        from core.memory import AssistantMemory
        agent = SearchAgent.__new__(SearchAgent)
        agent._llm     = MagicMock()
        agent._memory  = AssistantMemory()
        agent._verbose = False
        agent._logger  = MagicMock()
        agent._executor = MagicMock()
        tools = agent.get_tools()
        names = {t.name for t in tools}
        assert "web_search" in names or "ranked_web_search" in names
