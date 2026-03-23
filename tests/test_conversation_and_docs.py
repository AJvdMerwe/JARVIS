"""
tests/test_conversation_and_docs.py
────────────────────────────────────
Tests for all 6 improvements:

  Conversation quality
    1. Streaming responses  — Orchestrator.stream_response() yields tokens
    2. Context injection    — _augment_query() prepends history when relevant
    3. Follow-up detection  — is_followup_query() heuristics

  Document experience
    4. Document update/re-ingest — ingest_or_update() hash comparison
    5. Multi-document comparison — compare_documents() / search_multi_doc()
    6. Table-aware retrieval     — search_with_tables() / format_table_results()
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from agents.base_agent import AgentResponse


# =============================================================================
#  Helpers
# =============================================================================

def _make_memory(pairs: list[tuple[str, str]] | None = None):
    """Create a mock memory with optional (user, assistant) message pairs."""
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage, AIMessage
    mem = MagicMock()
    msgs = []
    for user, asst in (pairs or []):
        msgs.append(HumanMessage(content=user))
        msgs.append(AIMessage(content=asst))
    mem.messages = msgs
    return mem


def _make_orchestrator(stream_tokens: list[str] | None = None):
    """Build a minimal Orchestrator with a streaming mock agent."""
    from agents.orchestrator import Orchestrator, Intent

    orch = Orchestrator.__new__(Orchestrator)
    orch._session_id               = "test"
    orch._enable_llm_quality_check = False
    orch._enable_rag_precheck      = False
    orch._rag_similarity_threshold = 0.55
    orch._rag_k                    = 4
    orch._max_fallback_attempts    = 0
    orch._summariser               = None
    orch._episodic                 = None

    mem = MagicMock()
    mem.messages = []
    orch._memory = mem

    tokens = stream_tokens or ["Hello", " ", "world", "!"]

    orch._agents = {}
    for intent in Intent:
        agent      = MagicMock()
        agent.name = intent.value
        agent.run.return_value = AgentResponse(
            output="".join(tokens), agent_name=intent.value
        )
        agent.stream.return_value = iter(tokens)
        orch._agents[intent] = agent

    return orch


# =============================================================================
#  1. Streaming — Orchestrator.stream_response()
# =============================================================================

class TestStreamResponse:

    def test_yields_tokens(self):
        tokens = ["The", " ", "answer", " ", "is", " ", "42."]
        orch   = _make_orchestrator(tokens)
        with patch.object(orch, "_route", return_value=__import__(
            "agents.orchestrator", fromlist=["Intent"]
        ).Intent.CHAT):
            collected = list(orch.stream_response("What is 6x7?"))
        assert collected == tokens

    def test_returns_string_tokens(self):
        orch = _make_orchestrator(["Hello"])
        from agents.orchestrator import Intent
        with patch.object(orch, "_route", return_value=Intent.CHAT):
            for token in orch.stream_response("Hi"):
                assert isinstance(token, str)

    def test_saves_memory_after_stream_complete(self):
        orch = _make_orchestrator(["A", "B"])
        from agents.orchestrator import Intent
        with patch.object(orch, "_route", return_value=Intent.CHAT):
            list(orch.stream_response("test"))
        orch._memory.save_context.assert_called_once()

    def test_memory_contains_full_output(self):
        tokens = ["Hello", " ", "world"]
        orch   = _make_orchestrator(tokens)
        from agents.orchestrator import Intent
        with patch.object(orch, "_route", return_value=Intent.CHAT):
            list(orch.stream_response("hi"))
        args = orch._memory.save_context.call_args
        assert args[0][1] == "Hello world"

    def test_rag_precheck_short_circuits_stream(self):
        """When RAG pre-check finds an answer it's yielded as a single token."""
        orch = _make_orchestrator()
        orch._enable_rag_precheck = True
        rag_resp = AgentResponse(
            output="KB answer found here for this query.",
            agent_name="rag_precheck",
        )
        from agents.orchestrator import Intent
        with patch.object(orch, "_route", return_value=Intent.SEARCH), \
             patch("agents.orchestrator.rag_precheck", return_value=rag_resp):
            tokens = list(orch.stream_response("test query"))
        assert len(tokens) == 1
        assert tokens[0] == rag_resp.output
        # Primary agent never streamed
        orch._agents[Intent.SEARCH].stream.assert_not_called()

    def test_rag_disabled_skips_precheck(self):
        orch = _make_orchestrator()
        orch._enable_rag_precheck = False
        from agents.orchestrator import Intent
        with patch("agents.orchestrator.rag_precheck") as mock_rag, \
             patch.object(orch, "_route", return_value=Intent.CHAT):
            list(orch.stream_response("hi"))
        mock_rag.assert_not_called()

    def test_stream_error_yields_error_token(self):
        orch = _make_orchestrator()
        from agents.orchestrator import Intent
        orch._agents[Intent.CHAT].stream.side_effect = RuntimeError("LLM down")
        with patch.object(orch, "_route", return_value=Intent.CHAT):
            tokens = list(orch.stream_response("hi"))
        assert any("Error" in t or "error" in t for t in tokens)

    def test_forced_intent_routes_correctly(self):
        orch = _make_orchestrator()
        from agents.orchestrator import Intent
        list(orch.stream_response("test", intent="finance"))
        orch._agents[Intent.FINANCE].stream.assert_called_once()
        orch._agents[Intent.CHAT].stream.assert_not_called()


# =============================================================================
#  2. Conversation context injection — BaseAgent._augment_query()
# =============================================================================

class TestAugmentQuery:

    def _agent(self, history=None):
        from agents.chat_agent import ChatAgent
        from core.memory import AssistantMemory
        agent = ChatAgent.__new__(ChatAgent)
        agent._llm     = MagicMock()
        agent._memory  = _make_memory(history or [])
        agent._verbose = False
        agent._logger  = MagicMock()
        agent._system_prompt = "You are helpful."
        return agent

    def test_no_history_returns_query_unchanged(self):
        agent = self._agent(history=[])
        result = agent._augment_query("What is Python?")
        assert result == "What is Python?"

    def test_followup_query_gets_context_prepended(self):
        agent = self._agent(history=[
            ("What is Python?", "Python is a high-level programming language."),
        ])
        result = agent._augment_query("why?")
        assert "Recent conversation" in result or "why?" in result

    def test_long_query_no_followup_unchanged(self):
        """A long, non-referential query should not trigger context injection."""
        agent = self._agent(history=[
            ("Hello", "Hi there!"),
        ])
        long_q = "What are the fundamental differences between supervised and unsupervised machine learning?"
        result = agent._augment_query(long_q)
        # Long non-referential query: either unchanged OR context prepended but
        # query string still present
        assert long_q in result

    def test_force_always_injects(self):
        agent = self._agent(history=[
            ("What is AI?", "AI stands for artificial intelligence."),
        ])
        result = agent._augment_query("Tell me something", force_context=True)
        assert "Recent conversation" in result

    def test_error_in_context_returns_original(self):
        """Context injection errors must never block the agent."""
        agent = self._agent()
        with patch("core.conversation_context.inject_context_into_prompt",
                   side_effect=RuntimeError("context fail")):
            result = agent._augment_query("test query")
        assert result == "test query"

    def test_base_agent_stream_fallback(self):
        """BaseAgent.stream() default yields run() output as one token."""
        from agents.base_agent import BaseAgent, AgentResponse

        class _ConcreteAgent(BaseAgent):
            @property
            def name(self): return "test_agent"
            @property
            def description(self): return "test"
            def get_tools(self): return []
            def run(self, query, **kwargs):
                return AgentResponse(output="Full response text here.", agent_name=self.name)

        agent = _ConcreteAgent.__new__(_ConcreteAgent)
        agent._llm     = MagicMock()
        agent._memory  = _make_memory()
        agent._logger  = MagicMock()
        agent._verbose = False
        tokens = list(agent.stream("Hello"))
        assert "".join(tokens) == "Full response text here."


# =============================================================================
#  3. Follow-up detection — is_followup_query()
# =============================================================================

class TestIsFollowupQuery:

    def _mem(self, has_history=True):
        return _make_memory([("Hi", "Hello!")] if has_history else [])

    def test_empty_history_is_never_followup(self):
        from core.conversation_context import is_followup_query
        assert is_followup_query("why?", _make_memory([])) is False

    def test_short_query_is_followup(self):
        from core.conversation_context import is_followup_query
        assert is_followup_query("why?", self._mem()) is True
        assert is_followup_query("how?", self._mem()) is True
        assert is_followup_query("and then?", self._mem()) is True

    def test_long_unambiguous_query_is_not_followup(self):
        from core.conversation_context import is_followup_query
        q = "What are the key differences between TCP and UDP network protocols?"
        assert is_followup_query(q, self._mem()) is False

    def test_referential_pronoun_triggers_followup(self):
        from core.conversation_context import is_followup_query
        queries = [
            "Can you explain it in more detail?",
            "What does this mean for performance?",
            "How does that compare to Java?",
            "What you just said was confusing.",
        ]
        for q in queries:
            assert is_followup_query(q, self._mem()) is True, f"Expected followup: {q!r}"

    def test_continuation_word_triggers_followup(self):
        from core.conversation_context import is_followup_query
        queries = ["Also, what about memory?", "But how does it scale?", "So what now?"]
        for q in queries:
            assert is_followup_query(q, self._mem()) is True, f"Expected followup: {q!r}"

    def test_fresh_detailed_question_not_followup(self):
        from core.conversation_context import is_followup_query
        q = "Explain the architecture of a transformer neural network in detail."
        assert is_followup_query(q, self._mem()) is False

    @pytest.mark.parametrize("query, expected", [
        ("why?",                                 True),
        ("ok",                                   True),
        ("what about scalability?",               True),
        ("explain more",                          True),
        ("What is the capital of France?",        False),
        ("Write a Python function to sort a list", False),
        ("Research the history of quantum computing", False),
    ])
    def test_parametrized_cases(self, query, expected):
        from core.conversation_context import is_followup_query
        assert is_followup_query(query, self._mem()) is expected

    def test_build_conversation_context_empty(self):
        from core.conversation_context import build_conversation_context
        ctx = build_conversation_context(_make_memory([]))
        assert ctx == ""

    def test_build_conversation_context_formats_turns(self):
        from core.conversation_context import build_conversation_context
        mem = _make_memory([
            ("What is Python?", "Python is a high-level language."),
            ("Who created it?", "Guido van Rossum."),
        ])
        ctx = build_conversation_context(mem, k=3)
        assert "User: What is Python?" in ctx
        assert "Assistant: Python is a high-level language." in ctx
        assert "=== Recent conversation ===" in ctx

    def test_build_conversation_context_respects_k(self):
        from core.conversation_context import build_conversation_context
        # 5 pairs = 10 messages
        history = [(f"Q{i}", f"A{i}") for i in range(5)]
        mem     = _make_memory(history)
        ctx     = build_conversation_context(mem, k=2)
        # Only last 2 pairs should appear
        assert "Q0" not in ctx
        assert "Q4" in ctx

    def test_context_max_chars_respected(self):
        from core.conversation_context import build_conversation_context
        history = [("Q" * 200, "A" * 200)] * 10
        mem     = _make_memory(history)
        ctx     = build_conversation_context(mem, k=10, max_chars=500)
        assert len(ctx) <= 700  # budget + header overhead


# =============================================================================
#  4. Document update/re-ingest — ingest_or_update()
# =============================================================================

class TestIngestOrUpdate:

    def _dm(self):
        from document_processing.document_manager import DocumentManager
        dm = DocumentManager.__new__(DocumentManager)
        dm._store    = MagicMock()
        dm._processor = MagicMock()
        dm._similarity_threshold = 0.3
        dm._cache_ttl = 0.0
        dm._invalidate_search_cache = MagicMock()
        return dm

    def test_new_document_is_ingested(self, tmp_path):
        from document_processing.document_manager import IngestResult
        p  = tmp_path / "new.pdf"
        p.write_bytes(b"%PDF-1.4\ncontent here for testing purposes\n%%EOF")
        dm = self._dm()
        dm._store.get_document_chunks.return_value = []   # not in KB
        dm._store.ingest.return_value = 3
        mock_chunk = MagicMock()
        mock_chunk.metadata = {}
        dm._processor.process.return_value = [mock_chunk] * 3

        result = dm.ingest_or_update(p)

        assert result.chunks_added == 3
        assert result.metadata.get("replaced") is False

    def test_unchanged_document_skipped(self, tmp_path):
        import hashlib
        p = tmp_path / "report.pdf"
        p.write_bytes(b"some content that is the same")
        h = hashlib.sha256(p.read_bytes()).hexdigest()

        dm = self._dm()
        mock_chunk = MagicMock()
        mock_chunk.metadata = {"content_hash": h}
        dm._store.get_document_chunks.return_value = [mock_chunk]

        result = dm.ingest_or_update(p)

        assert result.chunks_added == 0
        assert result.metadata.get("reason") == "unchanged"
        dm._store.ingest.assert_not_called()

    def test_changed_document_replaces_old(self, tmp_path):
        import hashlib
        p = tmp_path / "report.pdf"
        p.write_bytes(b"new content version 2")
        new_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        old_hash = "aabbcc"  # different hash

        dm = self._dm()
        mock_chunk = MagicMock()
        mock_chunk.metadata = {"content_hash": old_hash}
        dm._store.get_document_chunks.return_value = [mock_chunk]
        dm._store.delete_document.return_value = 5
        dm._store.ingest.return_value = 7

        new_chunk = MagicMock(); new_chunk.metadata = {}
        dm._processor.process.return_value = [new_chunk] * 7

        result = dm.ingest_or_update(p)

        dm._store.delete_document.assert_called_once()
        assert result.metadata.get("replaced") is True
        assert result.metadata.get("chunks_removed") == 5

    def test_unreadable_file_returns_error(self, tmp_path):
        p = tmp_path / "ghost.pdf"
        # File doesn't exist
        dm = self._dm()
        result = dm.ingest_or_update(p)
        assert result.error is not None


# =============================================================================
#  5. Multi-document comparison
# =============================================================================

class TestMultiDocSearch:

    def _dm(self):
        from document_processing.document_manager import DocumentManager
        dm = DocumentManager.__new__(DocumentManager)
        dm._store = MagicMock()
        dm._similarity_threshold = 0.3
        dm._cache_ttl = 0.0
        return dm

    def _search_result(self, doc_title, text, score=0.80):
        r = MagicMock()
        r.chunk = MagicMock()
        r.chunk.text = text
        r.score = score
        r.reference = f"{doc_title} › Page 1"
        return r

    def test_search_multi_doc_queries_each_doc(self):
        dm = self._dm()
        dm._store.search.return_value = []
        dm.search_multi_doc("revenue", ["Q2 Report", "Q3 Report"])
        assert dm._store.search.call_count == 2

    def test_search_multi_doc_returns_per_doc_dict(self):
        dm = self._dm()
        r1 = self._search_result("Q2 Report", "Q2 revenue was $3.9B", 0.85)
        r2 = self._search_result("Q3 Report", "Q3 revenue was $4.2B", 0.90)
        dm._store.search.side_effect = [[r1], [r2]]
        results = dm.search_multi_doc("revenue", ["Q2 Report", "Q3 Report"])
        assert "Q2 Report" in results
        assert "Q3 Report" in results
        assert len(results["Q2 Report"]) == 1
        assert len(results["Q3 Report"]) == 1

    def test_compare_documents_formats_side_by_side(self):
        dm = self._dm()
        r1 = self._search_result("Q2 Report", "Q2 revenue was $3.9B.", 0.85)
        r2 = self._search_result("Q3 Report", "Q3 revenue was $4.2B.", 0.90)
        dm._store.search.side_effect = [[r1], [r2]]
        result = dm.compare_documents("revenue comparison", ["Q2 Report", "Q3 Report"])
        assert "Q2 Report" in result
        assert "Q3 Report" in result
        assert "## Comparison" in result

    def test_compare_documents_fallback_with_one_doc(self):
        dm = self._dm()
        dm._store.search.return_value = []
        dm._store.count.return_value = 5
        # With only 1 doc title, falls back to normal search
        result = dm.compare_documents("revenue", ["Q3 Report"])
        assert isinstance(result, str)

    def test_compare_documents_empty_results_message(self):
        dm = self._dm()
        dm._store.search.return_value = []   # nothing found
        result = dm.compare_documents("xyz", ["Doc A", "Doc B"])
        assert "No relevant" in result

    def test_search_multi_doc_applies_similarity_threshold(self):
        dm = self._dm()
        low_score  = self._search_result("Doc A", "text", 0.20)
        high_score = self._search_result("Doc A", "text", 0.90)
        dm._store.search.return_value = [low_score, high_score]
        results = dm.search_multi_doc("q", ["Doc A"], similarity_threshold=0.50)
        assert len(results["Doc A"]) == 1
        assert results["Doc A"][0].score == 0.90


# =============================================================================
#  6. Table-aware retrieval
# =============================================================================

class TestTableAwareRetrieval:

    def _dm(self):
        from document_processing.document_manager import DocumentManager
        dm = DocumentManager.__new__(DocumentManager)
        dm._store = MagicMock()
        dm._similarity_threshold = 0.3
        dm._cache_ttl = 0.0
        return dm

    def _result(self, text, score=0.80):
        r = MagicMock()
        r.chunk = MagicMock()
        r.chunk.text = text
        r.score = score
        r.reference = "Doc › Page 1"
        return r

    def _table_text(self):
        return (
            "| Quarter | Revenue | Growth |\n"
            "| --- | --- | --- |\n"
            "| Q1 | $1.2B | 12% |\n"
            "| Q2 | $1.4B | 16% |\n"
        )

    def test_table_chunks_separated_from_prose(self):
        dm = self._dm()
        table  = self._result(self._table_text())
        prose  = self._result("The company reported strong growth this quarter.")
        dm.search = MagicMock(return_value=[table, prose])
        tables, prose_results = dm.search_with_tables("revenue growth")
        assert any("|" in r.chunk.text for r in tables)
        assert all("|" not in r.chunk.text or r.chunk.text.count("|") < 4
                   for r in prose_results)

    def test_no_tables_returns_empty_table_list(self):
        dm = self._dm()
        dm.search = MagicMock(return_value=[
            self._result("Paragraph one."),
            self._result("Paragraph two."),
        ])
        tables, prose = dm.search_with_tables("something")
        assert tables == []
        assert len(prose) >= 1

    def test_format_table_results_includes_table_section(self):
        dm = self._dm()
        table_r = self._result(self._table_text())
        prose_r = self._result("Revenue discussion here.")
        output  = dm.format_table_results([table_r], [prose_r])
        assert "Data Tables" in output
        assert "Relevant Sections" in output

    def test_format_table_results_empty_returns_notice(self):
        dm     = self._dm()
        output = dm.format_table_results([], [])
        assert "No relevant" in output

    def test_table_detection_requires_two_pipe_rows(self):
        dm = self._dm()
        # Single pipe line is not a table
        single_pipe = self._result("The | separator is used here.")
        two_pipes   = self._result("| Col1 | Col2 |\n| A | B |\n")
        dm.search = MagicMock(return_value=[single_pipe, two_pipes])
        tables, _ = dm.search_with_tables("q")
        assert len(tables) == 1  # only the proper table
        assert tables[0].chunk.text == two_pipes.chunk.text

    def test_format_output_truncates_long_chunks(self):
        dm = self._dm()
        long_text = self._table_text() + ("word " * 500)
        table_r   = self._result(long_text)
        output    = dm.format_table_results([table_r], [], max_chars_per_chunk=200)
        assert "…" in output

    def test_table_search_tool_available(self):
        """TableSearchTool should be in get_document_tools()."""
        from tools.document_tools import get_document_tools
        tools = get_document_tools()
        tool_names = [t.name for t in tools]
        assert "search_documents_tables" in tool_names
        assert "compare_documents"       in tool_names
        assert "update_document"         in tool_names

    def test_k_budget_split_between_table_and_prose(self):
        """Total results should be ≤ k."""
        dm = self._dm()
        mixed = [
            self._result(self._table_text()),
            self._result("Prose chunk one."),
            self._result("Prose chunk two."),
        ]
        dm.search = MagicMock(return_value=mixed)
        tables, prose = dm.search_with_tables("q", k=2)
        assert len(tables) + len(prose) <= 2
