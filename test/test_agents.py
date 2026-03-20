"""
tests/test_agents.py
─────────────────────
Unit tests for:
  • BaseAgent            (AgentResponse model, helpers)
  • Orchestrator routing (keyword + LLM routing logic)
  • Each specialist agent's run() (mocked LLM + tools)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import AgentResponse
from agents.orchestrator import Intent, _keyword_route, _llm_route, Orchestrator


# ─────────────────────────────────────────────────────────────────────────────
#  AgentResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentResponse:
    def test_str(self):
        r = AgentResponse(output="Hello world", agent_name="test_agent")
        assert str(r) == "Hello world"

    def test_has_references_false(self):
        r = AgentResponse(output="out", agent_name="x")
        assert not r.has_references

    def test_has_references_true(self):
        r = AgentResponse(output="out", agent_name="x", references=["doc.pdf › Page 1"])
        assert r.has_references

    def test_format_references(self):
        r = AgentResponse(
            output="Here is your answer.",
            agent_name="doc_agent",
            references=["Report.pdf › Page 3 › Revenue", "Plan.docx › Page 1"],
        )
        formatted = r.format_references()
        assert "Sources" in formatted
        assert "Report.pdf" in formatted
        assert "Plan.docx" in formatted

    def test_full_response_no_refs(self):
        r = AgentResponse(output="Answer.", agent_name="x")
        assert r.full_response() == "Answer."

    def test_full_response_with_refs(self):
        r = AgentResponse(output="Answer.", agent_name="x", references=["doc.pdf"])
        full = r.full_response()
        assert "Answer." in full
        assert "doc.pdf" in full


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator routing
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordRouter:
    @pytest.mark.parametrize("query,expected", [
        ("write a python function to sort a list", Intent.CODE),
        ("debug this code: def foo(): pass", Intent.CODE),
        ("implement a REST API endpoint", Intent.CODE),
        ("what are today's top headlines?", Intent.NEWS),
        ("latest news about AI research", Intent.NEWS),
        ("give me a daily news briefing", Intent.NEWS),
        ("what is in the uploaded PDF?", Intent.DOCUMENT),
        ("search my documents for revenue figures", Intent.DOCUMENT),
        ("ingest this report.pdf", Intent.DOCUMENT),
        ("what is the capital of France?", Intent.SEARCH),
        ("calculate 15% of 2500", Intent.SEARCH),
    ])
    def test_routing(self, query: str, expected: Intent):
        result = _keyword_route(query)
        assert result == expected, f"Expected {expected} for: {query!r}, got {result}"


class TestLLMRouter:
    def test_llm_route_returns_valid_intent(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="code")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_route("something ambiguous")
        assert result in list(Intent)

    def test_llm_route_fallback_on_bad_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="INVALID_GIBBERISH")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_route("something ambiguous")
        assert result == Intent.SEARCH


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator.run()
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:

    def _make_orchestrator(self) -> Orchestrator:
        """Return an orchestrator with all LLM calls mocked out."""
        with patch("agents.orchestrator.PersistentMemory"):
            orch = Orchestrator.__new__(Orchestrator)
            orch._session_id = "test"

            mock_memory = MagicMock()
            mock_memory.messages = []
            orch._memory = mock_memory

            expected_response = AgentResponse(
                output="Mocked response.", agent_name="mock_agent"
            )
            mock_agent = MagicMock()
            mock_agent.run.return_value = expected_response

            orch._agents = {intent: mock_agent for intent in Intent}
        return orch

    def test_run_returns_agent_response(self):
        orch = self._make_orchestrator()
        with patch.object(orch, "_route", return_value=Intent.SEARCH):
            resp = orch.run("What is quantum entanglement?")
        assert isinstance(resp, AgentResponse)
        assert resp.output == "Mocked response."

    def test_forced_intent_bypasses_routing(self):
        orch = self._make_orchestrator()
        route_spy = MagicMock(return_value=Intent.SEARCH)
        with patch.object(orch, "_route", route_spy):
            orch.run("Write a quicksort", intent="code")
        # _route should NOT be called when intent is forced
        route_spy.assert_not_called()

    def test_memory_saved_after_run(self):
        orch = self._make_orchestrator()
        with patch.object(orch, "_route", return_value=Intent.SEARCH):
            orch.run("Test query")
        orch._memory.save_context.assert_called_once()

    def test_clear_memory(self):
        orch = self._make_orchestrator()
        orch.clear_memory()
        orch._memory.clear.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
#  Individual agent run()
# ─────────────────────────────────────────────────────────────────────────────

def _mock_executor_result(output: str = "Test output.") -> MagicMock:
    return MagicMock(
        invoke=MagicMock(
            return_value={
                "output": output,
                "intermediate_steps": [
                    (MagicMock(tool="web_search", tool_input="query"), "search result"),
                ],
            }
        )
    )


class TestCodeAgent:
    def test_run_success(self):
        from agents.code_agent import CodeAgent
        from core.memory import AssistantMemory

        agent = CodeAgent.__new__(CodeAgent)
        agent._memory = AssistantMemory()
        agent._verbose = False
        agent._logger = MagicMock()
        agent._executor = _mock_executor_result("```python\nprint('hello')\n```")

        resp = agent.run("Write hello world")
        assert isinstance(resp, AgentResponse)
        assert resp.agent_name == "code_agent"
        assert "print" in resp.output or "hello" in resp.output

    def test_run_error_handled(self):
        from agents.code_agent import CodeAgent
        from core.memory import AssistantMemory

        agent = CodeAgent.__new__(CodeAgent)
        agent._memory = AssistantMemory()
        agent._verbose = False
        agent._logger = MagicMock()
        agent._executor = MagicMock(invoke=MagicMock(side_effect=RuntimeError("boom")))

        resp = agent.run("Write something")
        assert resp.error == "boom"
        assert "error" in resp.output.lower()


class TestNewsAgent:
    def test_extract_urls(self):
        from agents.news_agent import NewsAgent

        tool_calls = [
            ("get_headlines", "", "Check this URL: https://bbc.co.uk/news/123\nAnd also https://nytimes.com/456"),
            ("search_topic_news", "AI", "Source: https://techcrunch.com/ai-story"),
        ]
        urls = NewsAgent._extract_urls(tool_calls)
        assert "https://bbc.co.uk/news/123" in urls
        assert "https://nytimes.com/456" in urls
        assert "https://techcrunch.com/ai-story" in urls

    def test_no_duplicate_urls(self):
        from agents.news_agent import NewsAgent

        tool_calls = [
            ("tool1", "", "https://example.com/story"),
            ("tool2", "", "https://example.com/story"),  # duplicate
        ]
        urls = NewsAgent._extract_urls(tool_calls)
        assert len(urls) == 1


class TestDocumentAgent:
    def test_extract_doc_references(self):
        from agents.document_agent import DocumentAgent

        tool_calls = [
            (
                "search_documents",
                "revenue",
                "**[1] Report.pdf › Page 3 › Revenue** (0.92)\nSome text here.\n"
                "**[2] Plan.docx › Page 1** (0.85)\nMore text.",
            )
        ]
        refs = DocumentAgent._extract_doc_references(tool_calls)
        assert "Report.pdf › Page 3 › Revenue" in refs
        assert "Plan.docx › Page 1" in refs

    def test_no_refs_from_non_search_tool(self):
        from agents.document_agent import DocumentAgent

        tool_calls = [
            ("list_documents", "", "**[1] Doc Title**"),
        ]
        refs = DocumentAgent._extract_doc_references(tool_calls)
        assert refs == []
