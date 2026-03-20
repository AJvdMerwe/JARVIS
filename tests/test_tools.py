"""
tests/test_tools.py
────────────────────
Unit tests for all LangChain tools (network and LLM calls mocked).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Code tools
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeExecutorTool:
    def test_simple_execution(self):
        from tools.code_tools import CodeExecutorTool
        tool = CodeExecutorTool()
        result = tool._run("print('hello from test')")
        assert "hello from test" in result
        assert "Return code: 0" in result

    def test_execution_with_error(self):
        from tools.code_tools import CodeExecutorTool
        tool = CodeExecutorTool()
        result = tool._run("raise ValueError('intentional error')")
        assert "Return code: 1" in result
        assert "STDERR" in result

    def test_timeout(self):
        from tools.code_tools import CodeExecutorTool
        tool = CodeExecutorTool()
        result = tool._run("import time; time.sleep(10)", timeout=1)
        assert "timed out" in result.lower()

    def test_blocked_pattern(self):
        from tools.code_tools import CodeExecutorTool
        tool = CodeExecutorTool()
        result = tool._run("import os; os.remove('/tmp/test')")
        assert "blocked" in result.lower()

    def test_math_execution(self):
        from tools.code_tools import CodeExecutorTool
        tool = CodeExecutorTool()
        result = tool._run("print(2 ** 10)")
        assert "1024" in result

    def test_multiline_code(self):
        from tools.code_tools import CodeExecutorTool
        tool = CodeExecutorTool()
        code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

print(fib(10))
"""
        result = tool._run(code)
        assert "55" in result


class TestCalculatorTool:
    @pytest.mark.parametrize("expr,expected", [
        ("2 + 2",             "4"),
        ("2 ** 10",           "1024"),
        ("sqrt(144)",         "12.0"),
        ("round(3.14159, 2)", "3.14"),
        ("abs(-42)",          "42"),
    ])
    def test_expressions(self, expr: str, expected: str):
        from tools.search_tools import CalculatorTool
        tool = CalculatorTool()
        result = tool._run(expr)
        assert expected in result

    def test_blocked_import(self):
        from tools.search_tools import CalculatorTool
        tool = CalculatorTool()
        result = tool._run("import os")
        assert "blocked" in result.lower() or "forbidden" in result.lower()

    def test_invalid_expression(self):
        from tools.search_tools import CalculatorTool
        tool = CalculatorTool()
        result = tool._run("not_a_function()")
        assert "Could not evaluate" in result


# ─────────────────────────────────────────────────────────────────────────────
#  News tools
# ─────────────────────────────────────────────────────────────────────────────

class TestRSSFetcher:
    def test_format_articles_empty(self):
        from tools.news_tools import _format_articles
        result = _format_articles([])
        assert "No articles" in result

    def test_format_articles(self):
        from tools.news_tools import _format_articles
        articles = [
            {
                "title": "Test Headline",
                "summary": "A test article summary.",
                "link": "https://example.com/1",
                "published": "2024-01-01",
                "source": "Test News",
            }
        ]
        result = _format_articles(articles)
        assert "Test Headline" in result
        assert "Test News" in result
        assert "https://example.com/1" in result

    def test_fetch_rss_handles_timeout(self):
        from tools.news_tools import _fetch_rss
        with patch("requests.get", side_effect=Exception("timeout")):
            result = _fetch_rss("https://invalid.example.com/feed.xml")
        assert result == []


class TestTopicNewsTool:
    def test_topic_search_success(self):
        from tools.news_tools import TopicNewsTool

        mock_results = [
            {
                "title": "AI breakthrough announced",
                "body": "Scientists made a breakthrough in AI research.",
                "url": "https://example.com/ai",
                "date": "2024-01-15",
                "source": "Tech News",
            }
        ]
        tool = TopicNewsTool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.news.return_value = mock_results
            result = tool._run("AI research")

        assert "AI breakthrough" in result
        assert "https://example.com/ai" in result

    def test_topic_search_no_results(self):
        from tools.news_tools import TopicNewsTool

        tool = TopicNewsTool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.news.return_value = []
            result = tool._run("extremely obscure topic xyz123")

        assert "No news found" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Search tools
# ─────────────────────────────────────────────────────────────────────────────

class TestDuckDuckGoTool:
    def test_search_success(self):
        from tools.search_tools import DuckDuckGoSearchTool

        mock_results = [
            {"title": "Python Tutorial", "body": "Learn Python.", "href": "https://python.org"}
        ]
        tool = DuckDuckGoSearchTool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = mock_results
            result = tool._run("Python programming")

        assert "Python Tutorial" in result
        assert "https://python.org" in result

    def test_search_empty_results(self):
        from tools.search_tools import DuckDuckGoSearchTool

        tool = DuckDuckGoSearchTool()
        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = []
            result = tool._run("zzz_no_results_xyz")

        assert "No results" in result


class TestWikipediaTool:
    def test_wikipedia_lookup(self):
        from tools.search_tools import WikipediaTool

        mock_page = MagicMock()
        mock_page.title = "Python (programming language)"
        mock_page.url = "https://en.wikipedia.org/wiki/Python"

        tool = WikipediaTool()
        with (
            patch("wikipedia.summary", return_value="Python is a high-level language."),
            patch("wikipedia.page", return_value=mock_page),
        ):
            result = tool._run("Python programming")

        assert "Python" in result
        assert "https://en.wikipedia.org" in result

    def test_wikipedia_not_found(self):
        from tools.search_tools import WikipediaTool
        import wikipedia

        tool = WikipediaTool()
        with patch("wikipedia.summary", side_effect=wikipedia.DisambiguationError("x", ["a", "b"])):
            result = tool._run("zzzinvalidpagexxx")

        assert "Could not retrieve" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Document tools
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentTools:
    def test_search_empty_kb(self):
        from tools.document_tools import SearchDocumentsTool

        mock_dm = MagicMock()
        mock_dm.total_chunks = 0

        tool = SearchDocumentsTool()
        with patch("tools.document_tools._get_doc_manager", return_value=mock_dm):
            result = tool._run("find revenue")

        assert "empty" in result.lower()

    def test_search_returns_results(self):
        from tools.document_tools import SearchDocumentsTool
        from document_processing import SearchResult, DocumentChunk

        mock_chunk = DocumentChunk(
            chunk_id="c1",
            text="Revenue was $10M in Q3.",
            doc_path="/docs/report.pdf",
            doc_title="Q3 Report",
            page_number=4,
            section_path=["Financials"],
        )
        mock_result = SearchResult(chunk=mock_chunk, score=0.92)

        mock_dm = MagicMock()
        mock_dm.total_chunks = 10
        mock_dm.search.return_value = [mock_result]
        mock_dm.format_search_results.return_value = "**[1] Q3 Report › Page 4 › Financials** (0.92)\nRevenue was $10M in Q3."

        tool = SearchDocumentsTool()
        with patch("tools.document_tools._get_doc_manager", return_value=mock_dm):
            result = tool._run("revenue Q3")

        assert "Q3 Report" in result
        assert "Revenue" in result

    def test_list_documents_empty(self):
        from tools.document_tools import ListDocumentsTool

        mock_dm = MagicMock()
        mock_dm.list_documents.return_value = []

        tool = ListDocumentsTool()
        with patch("tools.document_tools._get_doc_manager", return_value=mock_dm):
            result = tool._run()

        assert "No documents" in result

    def test_ingest_success(self):
        from tools.document_tools import IngestDocumentTool

        mock_dm = MagicMock()
        mock_dm.ingest.return_value = 42

        tool = IngestDocumentTool()
        with patch("tools.document_tools._get_doc_manager", return_value=mock_dm):
            result = tool._run("/docs/report.pdf")

        assert "42" in result
        assert "success" in result.lower() or "Ingested" in result

    def test_ingest_file_not_found(self):
        from tools.document_tools import IngestDocumentTool

        mock_dm = MagicMock()
        mock_dm.ingest.side_effect = FileNotFoundError("not found")

        tool = IngestDocumentTool()
        with patch("tools.document_tools._get_doc_manager", return_value=mock_dm):
            result = tool._run("/nonexistent.pdf")

        assert "not found" in result.lower()
