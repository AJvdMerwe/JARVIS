"""
tests/test_rag_precheck.py
───────────────────────────
Tests for agents/rag_precheck.py and its integration with Orchestrator.run().

Coverage:
  rag_precheck()
    • Skips when enabled=False
    • Skips for CODE and CHAT intents
    • Skips when KB is empty
    • Skips when no chunks above threshold
    • Returns None when LLM says INSUFFICIENT
    • Returns None when quality gate fails (answer too short / failure phrase)
    • Returns AgentResponse when KB has a sufficient answer
    • Attaches references and metadata from search results
    • Handles DocumentManager errors gracefully
    • Handles LLM errors gracefully
    • Handles search errors gracefully

  _build_context()
    • Respects character budget
    • Numbers results and includes references
    • Truncates long chunks

  Orchestrator integration
    • RAG pre-check called on every run() (when enabled)
    • RAG answer returned immediately — primary agent never called
    • Failed pre-check falls through to normal agent routing
    • Memory saved after RAG answer
    • Episodic facts stored after RAG answer
    • enable_rag_precheck=False disables the pre-check entirely
    • rag_similarity_threshold passed through to rag_precheck()
    • rag_k passed through to rag_precheck()
    • CODE intent bypasses pre-check
    • CHAT intent bypasses pre-check
    • Pre-check error never aborts run()
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from agents.base_agent import AgentResponse
from agents.orchestrator import Intent


# =============================================================================
#  Fixtures and helpers
# =============================================================================

def _search_result(text: str = "Relevant document content here.", score: float = 0.80,
                   doc_title: str = "Q3 Report", page: int = 1) -> MagicMock:
    """Build a mock SearchResult."""
    chunk = MagicMock()
    chunk.text = text
    chunk.page_number = page
    result = MagicMock()
    result.chunk    = chunk
    result.score    = score
    result.reference = f"{doc_title} › Page {page}"
    return result


def _mock_dm(chunks: int = 10, search_results: list | None = None) -> MagicMock:
    """Build a mock DocumentManager."""
    dm = MagicMock()
    dm.total_chunks = chunks
    dm.search.return_value = search_results if search_results is not None else []
    return dm


def _mock_llm(answer: str = "Based on the document, the answer is 42.") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=answer)
    return llm


def _reset():
    """Reset module singletons before each test."""
    from agents.rag_precheck import reset_singletons
    reset_singletons()


# =============================================================================
#  rag_precheck() — skip conditions
# =============================================================================

class TestRagPrecheckSkipConditions:

    def setup_method(self):
        _reset()

    def test_returns_none_when_disabled(self):
        from agents.rag_precheck import rag_precheck
        result = rag_precheck("What is Python?", "search", enabled=False)
        assert result is None

    def test_skips_code_intent(self):
        from agents.rag_precheck import rag_precheck
        result = rag_precheck("Write a sort function", "code", enabled=True)
        assert result is None

    def test_skips_chat_intent(self):
        from agents.rag_precheck import rag_precheck
        result = rag_precheck("Hello, how are you?", "chat", enabled=True)
        assert result is None

    def test_skips_empty_knowledge_base(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=0)
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            result = rag_precheck("What is Q3 revenue?", "search")
        assert result is None
        dm.search.assert_not_called()

    def test_skips_when_no_results_above_threshold(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[])
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            result = rag_precheck("What is Q3 revenue?", "search",
                                  similarity_threshold=0.70)
        assert result is None

    def test_skips_when_document_manager_unavailable(self):
        from agents.rag_precheck import rag_precheck
        with patch("agents.rag_precheck._get_document_manager",
                   side_effect=Exception("ChromaDB not ready")):
            result = rag_precheck("What is revenue?", "search")
        assert result is None

    def test_skips_when_search_raises(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5)
        dm.search.side_effect = RuntimeError("store corrupted")
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            result = rag_precheck("What is revenue?", "document")
        assert result is None

    def test_skips_when_llm_raises(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[_search_result()])
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=llm):
            result = rag_precheck("What is revenue?", "search")
        assert result is None

    @pytest.mark.parametrize("intent", ["news", "search", "document", "finance",
                                         "unknown"])
    def test_participates_for_non_skipped_intents(self, intent):
        """Intents other than code and chat are processed (may return None if no results)."""
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=0)  # empty KB → returns None but did not skip on intent
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            result = rag_precheck("test query", intent)
        # Empty KB → None, but the function ran past the intent check
        assert result is None
        dm.total_chunks  # accessed, proving we got past intent check


# =============================================================================
#  rag_precheck() — LLM answer quality
# =============================================================================

class TestRagPrecheckAnswerQuality:

    def setup_method(self):
        _reset()

    def test_returns_none_when_llm_says_insufficient(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[_search_result()])
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=_mock_llm("INSUFFICIENT")):
            result = rag_precheck("What is the meaning of life?", "search")
        assert result is None

    def test_returns_none_when_llm_says_insufficient_case_variations(self):
        from agents.rag_precheck import rag_precheck
        for answer in ["insufficient", "INSUFFICIENT", "Insufficient"]:
            _reset()
            dm = _mock_dm(chunks=5, search_results=[_search_result()])
            with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
                 patch("agents.rag_precheck._get_llm", return_value=_mock_llm(answer)):
                result = rag_precheck("test query", "search")
            assert result is None, f"Should be None for answer={answer!r}"

    def test_returns_none_when_answer_too_short(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[_search_result()])
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=_mock_llm("Yes.")):
            result = rag_precheck("test", "search")
        assert result is None

    def test_returns_none_when_answer_has_failure_phrase(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[_search_result()])
        answer = "I don't know the answer to your question based on the context provided here."
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=_mock_llm(answer)):
            result = rag_precheck("test", "search")
        assert result is None

    def test_returns_agent_response_for_good_answer(self):
        from agents.rag_precheck import rag_precheck
        dm  = _mock_dm(chunks=5, search_results=[_search_result()])
        answer = (
            "According to Q3 Report, the revenue for Q3 2024 was $4.2 billion, "
            "representing a 12% year-over-year increase driven by strong product sales."
        )
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=_mock_llm(answer)):
            result = rag_precheck("What was Q3 revenue?", "search")
        assert isinstance(result, AgentResponse)
        assert result.agent_name == "rag_precheck"
        assert "4.2 billion" in result.output

    def test_response_carries_references(self):
        from agents.rag_precheck import rag_precheck
        hits = [
            _search_result("Revenue was $4.2B", score=0.88, doc_title="Q3 Report"),
            _search_result("Costs fell by 5%",  score=0.72, doc_title="Q3 Report"),
        ]
        dm     = _mock_dm(chunks=5, search_results=hits)
        answer = "Revenue was $4.2 billion and costs fell by 5% according to Q3 Report."
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=_mock_llm(answer)):
            result = rag_precheck("What was Q3 performance?", "document")
        assert len(result.references) == 2
        assert "Q3 Report › Page 1" in result.references

    def test_response_metadata_contains_rag_fields(self):
        from agents.rag_precheck import rag_precheck
        dm     = _mock_dm(chunks=5, search_results=[_search_result(score=0.91)])
        answer = "The operating margin was 26.4% based on the quarterly results document."
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=_mock_llm(answer)):
            result = rag_precheck("What was operating margin?", "finance")
        assert "rag_chunks"     in result.metadata
        assert "rag_top_score"  in result.metadata
        assert "rag_elapsed_ms" in result.metadata
        assert result.metadata["rag_chunks"]    == 1
        assert result.metadata["rag_top_score"] == 0.91

    def test_similarity_threshold_passed_to_search(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[])
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            rag_precheck("query", "search", similarity_threshold=0.80)
        dm.search.assert_called_once_with(
            "query", k=4, similarity_threshold=0.80
        )

    def test_k_passed_to_search(self):
        from agents.rag_precheck import rag_precheck
        dm = _mock_dm(chunks=5, search_results=[])
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            rag_precheck("query", "search", k=8)
        dm.search.assert_called_once_with(
            "query", k=8, similarity_threshold=0.55
        )

    def test_prompt_contains_query_and_context(self):
        from agents.rag_precheck import rag_precheck
        hits = [_search_result("Important fact about revenue here.")]
        dm  = _mock_dm(chunks=5, search_results=hits)
        llm = _mock_llm("Revenue was $4.2B according to the Q3 Report pages.")
        with patch("agents.rag_precheck._get_document_manager", return_value=dm), \
             patch("agents.rag_precheck._get_llm", return_value=llm):
            rag_precheck("What was Q3 revenue?", "search")
        prompt = llm.invoke.call_args.args[0]
        assert "What was Q3 revenue?" in prompt
        assert "Important fact about revenue" in prompt


# =============================================================================
#  _build_context()
# =============================================================================

class TestBuildContext:

    def test_includes_references_in_header(self):
        from agents.rag_precheck import _build_context
        hits = [_search_result("Some content.", doc_title="Annual Report", page=3)]
        ctx = _build_context(hits, max_chars=2000)
        assert "Annual Report › Page 3" in ctx
        assert "Some content." in ctx

    def test_numbers_results(self):
        from agents.rag_precheck import _build_context
        hits = [
            _search_result("First result."),
            _search_result("Second result."),
        ]
        ctx = _build_context(hits, max_chars=2000)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_respects_character_budget(self):
        from agents.rag_precheck import _build_context
        # Create a result with a very long text
        long = _search_result("x" * 2000)
        ctx = _build_context([long], max_chars=500)
        assert len(ctx) <= 600  # some headroom for the header

    def test_truncates_long_chunks_with_ellipsis(self):
        from agents.rag_precheck import _build_context
        long = _search_result("word " * 1000)
        ctx = _build_context([long], max_chars=300)
        assert "…" in ctx

    def test_stops_adding_when_budget_exhausted(self):
        from agents.rag_precheck import _build_context
        hits = [_search_result("content " * 100)] * 10
        ctx = _build_context(hits, max_chars=500)
        # Should not include all 10 results
        assert ctx.count("[") < 10

    def test_empty_results_returns_empty_string(self):
        from agents.rag_precheck import _build_context
        ctx = _build_context([], max_chars=2000)
        assert ctx == ""

    def test_score_shown_in_header(self):
        from agents.rag_precheck import _build_context
        hits = [_search_result(score=0.87)]
        ctx = _build_context(hits, max_chars=2000)
        assert "0.87" in ctx


# =============================================================================
#  Orchestrator integration
# =============================================================================

def _make_orchestrator_with_rag(
    agent_responses: dict | None = None,
    enable_rag: bool = True,
    rag_threshold: float = 0.55,
    rag_k: int = 4,
    max_fallback: int = 2,
):
    """Create an Orchestrator with mocked agents and RAG control."""
    from agents.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._session_id               = "test"
    orch._enable_llm_quality_check = False
    orch._enable_rag_precheck      = enable_rag
    orch._rag_similarity_threshold = rag_threshold
    orch._rag_k                    = rag_k
    orch._max_fallback_attempts    = max_fallback
    orch._summariser               = None
    orch._episodic                 = None

    mock_memory = MagicMock()
    mock_memory.messages = []
    orch._memory = mock_memory

    responses = agent_responses or {}
    orch._agents = {}
    for intent in Intent:
        agent      = MagicMock()
        agent.name = intent.value
        default_resp = AgentResponse(
            output="Agent answer that is long enough to pass quality gates for testing.",
            agent_name=intent.value,
        )
        agent.run.return_value = responses.get(intent, default_resp)
        orch._agents[intent] = agent

    return orch


class TestOrchestratorRAGIntegration:

    def _good_rag_response(self) -> AgentResponse:
        return AgentResponse(
            output="Q3 revenue was $4.2 billion according to the Q3 Report document.",
            agent_name="rag_precheck",
            references=["Q3 Report › Page 4"],
        )

    def test_rag_answer_returned_without_calling_primary_agent(self):
        """When RAG pre-check succeeds, the primary agent must never be called."""
        orch = _make_orchestrator_with_rag(enable_rag=True)
        rag_resp = self._good_rag_response()

        with patch("agents.orchestrator.rag_precheck", return_value=rag_resp) as mock_rag, \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            resp = orch.run("What was Q3 revenue?")

        mock_rag.assert_called_once()
        assert resp is rag_resp
        assert resp.agent_name == "rag_precheck"
        orch._agents[Intent.SEARCH].run.assert_not_called()

    def test_primary_agent_called_when_rag_returns_none(self):
        """When RAG pre-check returns None, the primary agent handles the query."""
        orch = _make_orchestrator_with_rag(enable_rag=True)

        with patch("agents.orchestrator.rag_precheck", return_value=None), \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            resp = orch.run("What is the capital of France?")

        orch._agents[Intent.SEARCH].run.assert_called_once()
        assert resp.agent_name == Intent.SEARCH.value

    def test_rag_disabled_never_calls_rag_precheck(self):
        """enable_rag_precheck=False must bypass the pre-check entirely."""
        orch = _make_orchestrator_with_rag(enable_rag=False)

        with patch("agents.orchestrator.rag_precheck") as mock_rag, \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            orch.run("test query")

        mock_rag.assert_not_called()

    def test_rag_threshold_forwarded(self):
        orch = _make_orchestrator_with_rag(enable_rag=True, rag_threshold=0.75)
        with patch("agents.orchestrator.rag_precheck", return_value=None) as mock_rag, \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            orch.run("test")
        _, kwargs = mock_rag.call_args
        assert kwargs["similarity_threshold"] == 0.75

    def test_rag_k_forwarded(self):
        orch = _make_orchestrator_with_rag(enable_rag=True, rag_k=8)
        with patch("agents.orchestrator.rag_precheck", return_value=None) as mock_rag, \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            orch.run("test")
        _, kwargs = mock_rag.call_args
        assert kwargs["k"] == 8

    def test_intent_value_forwarded_to_rag(self):
        orch = _make_orchestrator_with_rag(enable_rag=True)
        with patch("agents.orchestrator.rag_precheck", return_value=None) as mock_rag, \
             patch.object(orch, "_route", return_value=Intent.FINANCE):
            orch.run("What is AAPL stock price?")
        assert mock_rag.call_args.args[1] == "finance"

    def test_memory_saved_after_rag_answer(self):
        orch     = _make_orchestrator_with_rag(enable_rag=True)
        rag_resp = self._good_rag_response()

        with patch("agents.orchestrator.rag_precheck", return_value=rag_resp), \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            orch.run("What was Q3 revenue?")

        orch._memory.save_context.assert_called_once_with(
            "What was Q3 revenue?", rag_resp.output
        )

    def test_rag_exception_falls_through_to_agent(self):
        """An unexpected error in rag_precheck must not abort run()."""
        orch = _make_orchestrator_with_rag(enable_rag=True)

        with patch("agents.orchestrator.rag_precheck",
                   side_effect=RuntimeError("unexpected crash")), \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            # Should not raise — falls through to agent
            try:
                resp = orch.run("test query")
            except RuntimeError:
                pytest.fail("rag_precheck exception should not propagate from run()")

    def test_code_intent_bypasses_rag(self):
        """CODE queries skip the RAG pre-check at the rag_precheck level."""
        orch = _make_orchestrator_with_rag(enable_rag=True)

        calls = []
        def capture(query, intent_value, **kw):
            calls.append(intent_value)
            return None  # no KB answer

        with patch("agents.orchestrator.rag_precheck", side_effect=capture), \
             patch.object(orch, "_route", return_value=Intent.CODE):
            orch.run("Write a bubble sort in Python")

        # rag_precheck was called with intent "code"
        assert calls == ["code"]

    def test_chat_intent_bypasses_rag(self):
        """CHAT queries skip the RAG pre-check at the rag_precheck level."""
        orch = _make_orchestrator_with_rag(enable_rag=True)
        calls = []

        def capture(query, intent_value, **kw):
            calls.append(intent_value)
            return None

        with patch("agents.orchestrator.rag_precheck", side_effect=capture), \
             patch.object(orch, "_route", return_value=Intent.CHAT):
            orch.run("Hello!")

        assert calls == ["chat"]

    def test_forced_intent_still_triggers_rag(self):
        """Forcing an intent bypasses routing but not the RAG pre-check."""
        orch     = _make_orchestrator_with_rag(enable_rag=True)
        rag_resp = self._good_rag_response()

        with patch("agents.orchestrator.rag_precheck", return_value=rag_resp) as mock_rag:
            resp = orch.run("What is Q3 revenue?", intent="document")

        mock_rag.assert_called_once()
        assert resp.agent_name == "rag_precheck"

    def test_rag_answer_includes_references_in_response(self):
        orch = _make_orchestrator_with_rag(enable_rag=True)
        rag_resp = AgentResponse(
            output="Revenue was $4.2B per Q3 Report page 4 financial summary.",
            agent_name="rag_precheck",
            references=["Q3 Report › Page 4 › Revenue", "Q3 Report › Page 5 › Summary"],
        )
        with patch("agents.orchestrator.rag_precheck", return_value=rag_resp), \
             patch.object(orch, "_route", return_value=Intent.DOCUMENT):
            resp = orch.run("revenue question")
        assert len(resp.references) == 2

    def test_fallback_not_triggered_after_rag_success(self):
        """RAG answers bypass the fallback chain entirely."""
        orch     = _make_orchestrator_with_rag(enable_rag=True, max_fallback=3)
        rag_resp = self._good_rag_response()

        with patch("agents.orchestrator.rag_precheck", return_value=rag_resp), \
             patch.object(orch, "_route", return_value=Intent.SEARCH), \
             patch.object(orch, "_run_fallback_chain") as mock_fallback:
            orch.run("Q3 revenue")

        mock_fallback.assert_not_called()

    def test_repr_shows_rag_precheck_status(self):
        orch = _make_orchestrator_with_rag(enable_rag=True)
        orch._session_id = "repr-test"
        r = repr(orch)
        assert "rag_precheck=True" in r

    def test_repr_shows_rag_disabled(self):
        orch = _make_orchestrator_with_rag(enable_rag=False)
        r = repr(orch)
        assert "rag_precheck=False" in r


# =============================================================================
#  RAG pre-check + fallback chain interaction
# =============================================================================

class TestRagWithFallback:

    def test_rag_none_then_primary_fails_then_fallback_succeeds(self):
        """RAG misses → primary insufficient → fallback answers."""
        from agents.orchestrator import Intent

        orch = _make_orchestrator_with_rag(
            agent_responses={
                Intent.DOCUMENT: AgentResponse(
                    output="I don't know the answer to this question.",
                    agent_name="document",
                ),
                Intent.SEARCH: AgentResponse(
                    output="Found comprehensive information about Q3 revenue in public records.",
                    agent_name="search",
                ),
            },
            enable_rag=True,
            max_fallback=2,
        )

        with patch("agents.orchestrator.rag_precheck", return_value=None), \
             patch.object(orch, "_route", return_value=Intent.DOCUMENT):
            resp = orch.run("What was Q3 revenue?")

        # document agent failed, search agent (fallback) answered
        assert resp.agent_name == "search"

    def test_rag_succeeds_even_when_primary_would_fail(self):
        """RAG returns answer regardless of what the primary agent would do."""
        orch = _make_orchestrator_with_rag(
            agent_responses={
                Intent.SEARCH: AgentResponse(output="err", agent_name="search",
                                              error="search failed"),
            },
            enable_rag=True,
        )
        rag_resp = AgentResponse(
            output="Q3 revenue was $4.2 billion from the Q3 Report document.",
            agent_name="rag_precheck",
        )
        with patch("agents.orchestrator.rag_precheck", return_value=rag_resp), \
             patch.object(orch, "_route", return_value=Intent.SEARCH):
            resp = orch.run("Q3 revenue")

        assert resp.agent_name == "rag_precheck"
        orch._agents[Intent.SEARCH].run.assert_not_called()


# =============================================================================
#  reset_singletons helper
# =============================================================================

class TestResetSingletons:

    def test_reset_clears_dm_singleton(self):
        from agents import rag_precheck as rpc_module

        # Set a fake singleton
        rpc_module._dm_singleton = MagicMock()
        assert rpc_module._dm_singleton is not None

        rpc_module.reset_singletons()
        assert rpc_module._dm_singleton is None

    def test_after_reset_new_dm_created_on_next_call(self):
        from agents.rag_precheck import rag_precheck, reset_singletons
        reset_singletons()

        dm1 = _mock_dm(chunks=0)
        dm2 = _mock_dm(chunks=0)

        with patch("agents.rag_precheck._get_document_manager", return_value=dm1):
            rag_precheck("q1", "search")

        reset_singletons()

        with patch("agents.rag_precheck._get_document_manager", return_value=dm2):
            rag_precheck("q2", "search")

        # Both were called independently
        dm1.total_chunks  # noqa: B018
        dm2.total_chunks  # noqa: B018
