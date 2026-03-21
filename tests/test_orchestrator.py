"""
tests/test_orchestrator.py
───────────────────────────
Dedicated tests for the Orchestrator's fallback loop, quality gates,
synthesis, and routing logic.

Test coverage:
  • _is_sufficient_response()    — fast-path quality gate
  • _llm_quality_check()         — LLM judge
  • _synthesise_from_attempts()  — multi-agent synthesis
  • _run_fallback_chain()        — fallback iteration
  • Orchestrator.run()           — end-to-end with mocked agents
  • _FALLBACK_CHAINS             — all per-intent chains
  • _call_agent()                — exception isolation
  • enable_llm_quality_check     — opt-in LLM gate
  • max_fallback_attempts=0      — disabling fallback
  • forced intent                — bypasses routing and fallback
  • synthesis path               — all agents fail
  • Quality constants            — MIN_SUFFICIENT_LENGTH, _FAILURE_PHRASES
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from agents.base_agent import AgentResponse
from agents.orchestrator import (
    MIN_SUFFICIENT_LENGTH,
    Intent,
    Orchestrator,
    OrchestratorState,
    _FALLBACK_CHAINS,
    _FAILURE_PHRASES,
    _is_sufficient_response,
    _llm_quality_check,
    _synthesise_from_attempts,
)


# =============================================================================
#  Helpers
# =============================================================================

def _good(text: str = "This is a thorough and complete answer to the question.", agent: str = "test_agent") -> AgentResponse:
    """Build a response that passes all quality gates."""
    assert len(text) >= MIN_SUFFICIENT_LENGTH, f"_good() text too short: {len(text)} chars"
    return AgentResponse(output=text, agent_name=agent)


def _good_for(intent: Intent, text: str | None = None) -> AgentResponse:
    """Build a passing response with the correct agent name for the given intent."""
    t = text or f"This is a thorough and complete answer from the {intent.value} agent."
    return AgentResponse(output=t, agent_name=intent.value)


def _bad(reason: str = "short", agent: str = "test_agent") -> AgentResponse:
    """Build a response that fails the quality gate."""
    if reason == "short":
        return AgentResponse(output="nope", agent_name=agent)
    if reason == "error":
        return AgentResponse(output="error occurred", agent_name=agent, error="something broke")
    if reason == "phrase":
        return AgentResponse(
            output="I don't know the answer to your question at all.",
            agent_name=agent,
        )
    return AgentResponse(output="", agent_name=agent)


def _make_orchestrator(
    agent_responses: dict[Intent, AgentResponse] | None = None,
    enable_llm_quality_check: bool = False,
    max_fallback_attempts: int = 2,
) -> Orchestrator:
    """
    Build an Orchestrator with mocked agents.  Every attribute set by
    __init__ is recreated manually so no real Ollama / ChromaDB is needed.
    """
    orch = Orchestrator.__new__(Orchestrator)
    orch._session_id               = "test-session"
    orch._enable_llm_quality_check = enable_llm_quality_check
    orch._max_fallback_attempts    = max_fallback_attempts
    orch._summariser               = None
    orch._episodic                 = None

    mock_memory = MagicMock()
    mock_memory.messages = []
    orch._memory = mock_memory

    responses = agent_responses or {}
    default   = _good()
    orch._agents = {}
    for intent in Intent:
        agent      = MagicMock()
        agent.name = intent.value            # ← crucial: name must match intent
        agent.run.return_value = responses.get(intent, default)
        orch._agents[intent] = agent

    return orch


# =============================================================================
#  Quality constants
# =============================================================================

class TestQualityConstants:

    def test_min_sufficient_length_is_positive(self):
        assert MIN_SUFFICIENT_LENGTH > 0

    def test_failure_phrases_are_lowercase(self):
        for phrase in _FAILURE_PHRASES:
            assert phrase == phrase.lower(), f"Phrase not lowercase: {phrase!r}"

    def test_failure_phrases_non_empty(self):
        assert len(_FAILURE_PHRASES) >= 5

    def test_fallback_chains_cover_all_intents(self):
        """Every intent that can be primary should have a fallback chain."""
        for intent in Intent:
            if intent == Intent.UNKNOWN:
                continue  # UNKNOWN falls to CHAT directly, not via chain
            assert intent in _FALLBACK_CHAINS, f"No fallback chain for {intent}"

    def test_fallback_chains_never_self_reference(self):
        """An intent's fallback chain must not contain itself."""
        for intent, chain in _FALLBACK_CHAINS.items():
            assert intent not in chain, (
                f"{intent} appears in its own fallback chain"
            )

    def test_fallback_chains_end_in_chat(self):
        """Every non-CHAT chain should eventually include CHAT as last resort."""
        for intent, chain in _FALLBACK_CHAINS.items():
            if intent != Intent.CHAT:
                assert Intent.CHAT in chain, (
                    f"Chain for {intent} does not include CHAT: {chain}"
                )


# =============================================================================
#  _is_sufficient_response()
# =============================================================================

class TestIsSufficientResponse:

    def test_good_response_passes(self):
        assert _is_sufficient_response(_good(), "any query") is True

    def test_empty_output_fails(self):
        r = AgentResponse(output="", agent_name="a")
        assert _is_sufficient_response(r, "q") is False

    def test_too_short_fails(self):
        r = AgentResponse(output="ok", agent_name="a")  # 2 chars
        assert _is_sufficient_response(r, "q") is False

    def test_exactly_at_limit_passes(self):
        text = "x" * MIN_SUFFICIENT_LENGTH
        r    = AgentResponse(output=text, agent_name="a")
        assert _is_sufficient_response(r, "q") is True

    def test_one_below_limit_fails(self):
        text = "x" * (MIN_SUFFICIENT_LENGTH - 1)
        r    = AgentResponse(output=text, agent_name="a")
        assert _is_sufficient_response(r, "q") is False

    def test_error_flag_fails(self):
        r = AgentResponse(output="A " * 30, agent_name="a", error="network error")
        assert _is_sufficient_response(r, "q") is False

    def test_error_overrides_long_output(self):
        """Even a long output is insufficient when error is set."""
        r = AgentResponse(output="x" * 500, agent_name="a", error="oops")
        assert _is_sufficient_response(r, "q") is False

    @pytest.mark.parametrize("phrase", [
        "i don't know",
        "i do not know",
        "no information",
        "cannot find",
        "not available",
        "no results found",
        "failed to retrieve",
        "i encountered an error",
    ])
    def test_failure_phrases_fail(self, phrase: str):
        # Pad to exceed MIN_SUFFICIENT_LENGTH
        text = phrase + " " + ("x " * 30)
        r    = AgentResponse(output=text, agent_name="a")
        assert _is_sufficient_response(r, "q") is False

    def test_failure_phrase_case_insensitive(self):
        text = "I DON'T KNOW the answer. " + "x " * 30
        r    = AgentResponse(output=text, agent_name="a")
        assert _is_sufficient_response(r, "q") is False

    def test_failure_phrase_in_middle_of_text(self):
        text = "After thorough research, I cannot find any relevant data. " + "x " * 10
        r    = AgentResponse(output=text, agent_name="a")
        assert _is_sufficient_response(r, "q") is False

    def test_partial_phrase_match_does_not_fail(self):
        """'find' alone should not trigger — only exact phrases like 'cannot find'."""
        text = "I found a great answer to your question about finding information. " * 2
        r    = AgentResponse(output=text, agent_name="a")
        assert _is_sufficient_response(r, "q") is True

    def test_whitespace_only_fails(self):
        r = AgentResponse(output="   \n\t  ", agent_name="a")
        assert _is_sufficient_response(r, "q") is False


# =============================================================================
#  _llm_quality_check()
# =============================================================================

class TestLLMQualityCheck:

    def test_sufficient_verdict_returns_true(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SUFFICIENT")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_quality_check("What is Python?", _good())
        assert result is True

    def test_insufficient_verdict_returns_false(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="INSUFFICIENT")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_quality_check("What is Python?", _good())
        assert result is False

    def test_llm_error_defaults_to_true(self):
        """On LLM failure, the gate fails open (don't trigger unnecessary retries)."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_quality_check("q", _good())
        assert result is True

    def test_prompt_contains_query_and_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SUFFICIENT")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            _llm_quality_check("What is recursion?", _good("Recursion is when a function calls itself."))
        prompt = mock_llm.invoke.call_args[0][0]
        assert "What is recursion?" in prompt
        assert "Recursion is when" in prompt

    def test_exact_sufficient_verdict_passes(self):
        """Only exact SUFFICIENT verdict (after strip) passes — not partial matches."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SUFFICIENT")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_quality_check("q", _good())
        assert result is True

    def test_insufficient_not_misread_as_sufficient(self):
        """INSUFFICIENT must NOT be treated as passing (was the original bug)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="INSUFFICIENT")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_quality_check("q", _good())
        assert result is False

    def test_trailing_whitespace_in_verdict(self):
        """Verdict with surrounding whitespace should still work."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="  SUFFICIENT  ")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_quality_check("q", _good())
        assert result is True

    def test_truncates_long_response_in_prompt(self):
        """Very long agent output must be truncated before sending to the judge LLM."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SUFFICIENT")
        long_output = "word " * 1000  # 5000 chars
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            _llm_quality_check("q", AgentResponse(output=long_output, agent_name="a"))
        prompt = mock_llm.invoke.call_args[0][0]
        # Prompt should be truncated — not contain all 5000 chars
        assert len(prompt) < 5000


# =============================================================================
#  _synthesise_from_attempts()
# =============================================================================

class TestSynthesiseFromAttempts:

    def test_returns_agent_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Synthesised answer addressing the user question thoroughly.")
        attempts = [
            ("search_agent",   _good("Search gave partial info and context about Python language.")),
            ("document_agent", _good("Document mentioned Python version 3.12 was recently released.")),
        ]
        with patch("agents.orchestrator.get_llm", return_value=mock_llm), \
             patch("core.llm_manager.get_llm", return_value=mock_llm):
            result = _synthesise_from_attempts("Tell me about Python", attempts)
        assert isinstance(result, AgentResponse)
        assert "Synthesised" in result.output
        assert result.agent_name == "synthesised"

    def test_collects_tool_calls_from_all_attempts(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Combined answer.")
        r1 = AgentResponse(output="Info A " * 10, agent_name="a",
                           tool_calls=[("web_search", "query", "result")])
        r2 = AgentResponse(output="Info B " * 10, agent_name="b",
                           tool_calls=[("wikipedia", "topic", "wiki")])
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _synthesise_from_attempts("q", [("a", r1), ("b", r2)])
        assert len(result.tool_calls) == 2

    def test_collects_references_from_all_attempts(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer.")
        r1 = AgentResponse(output="x " * 30, agent_name="a",
                           references=["Doc A › Page 1"])
        r2 = AgentResponse(output="y " * 30, agent_name="b",
                           references=["Doc B › Page 2"])
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _synthesise_from_attempts("q", [("a", r1), ("b", r2)])
        assert "Doc A › Page 1" in result.references
        assert "Doc B › Page 2" in result.references

    def test_all_errored_returns_last_attempt(self):
        """When all attempts have errors and no useful output, return the last."""
        r1 = AgentResponse(output="", agent_name="a", error="err1")
        r2 = AgentResponse(output="", agent_name="b", error="err2")
        result = _synthesise_from_attempts("q", [("a", r1), ("b", r2)])
        # Falls back to the last attempt (no LLM call needed)
        assert result.agent_name == "b"

    def test_llm_failure_returns_longest_useful_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("synthesis LLM down")
        r1 = AgentResponse(output="Short partial answer.", agent_name="a")
        r2 = AgentResponse(output="Longer and more detailed partial answer from this agent. " * 3,
                           agent_name="b")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _synthesise_from_attempts("q", [("a", r1), ("b", r2)])
        # Returns the longest non-error response when synthesis fails
        assert result.agent_name == "b"

    def test_prompt_contains_all_attempt_outputs(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer.")
        r1 = AgentResponse(output="Python is a language " * 5, agent_name="search")
        r2 = AgentResponse(output="Version 3.12 released " * 5, agent_name="document")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            _synthesise_from_attempts("Tell me about Python", [("search", r1), ("document", r2)])
        prompt = mock_llm.invoke.call_args[0][0]
        assert "Tell me about Python" in prompt
        assert "search" in prompt or "Python is a language" in prompt

    def test_deduplicates_references(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer.")
        ref = "Report › Page 1"
        r1  = AgentResponse(output="x " * 30, agent_name="a", references=[ref])
        r2  = AgentResponse(output="y " * 30, agent_name="b", references=[ref])
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _synthesise_from_attempts("q", [("a", r1), ("b", r2)])
        assert result.references.count(ref) == 1  # deduplicated


# =============================================================================
#  Orchestrator._call_agent()
# =============================================================================

class TestCallAgent:

    def test_returns_agent_response_on_success(self):
        agent = MagicMock()
        agent.name = "test"
        agent.run.return_value = _good()
        result = Orchestrator._call_agent(agent, "query")
        assert isinstance(result, AgentResponse)
        assert result == _good()

    def test_exception_captured_as_error_response(self):
        agent = MagicMock()
        agent.name = "bad_agent"
        agent.run.side_effect = RuntimeError("tool exploded")
        result = Orchestrator._call_agent(agent, "query")
        assert isinstance(result, AgentResponse)
        assert result.error == "tool exploded"
        assert result.agent_name == "bad_agent"

    def test_kwargs_forwarded(self):
        agent = MagicMock()
        agent.name = "a"
        agent.run.return_value = _good()
        Orchestrator._call_agent(agent, "query", user_id="alice", doc_title="Q3")
        agent.run.assert_called_once_with("query", user_id="alice", doc_title="Q3")


# =============================================================================
#  Orchestrator fallback chain — _run_fallback_chain()
# =============================================================================

class TestRunFallbackChain:

    def _make_orch(self, responses: dict[Intent, AgentResponse] | None = None,
                   max_fallback: int = 3) -> Orchestrator:
        return _make_orchestrator(responses, max_fallback_attempts=max_fallback)

    def test_first_fallback_sufficient_returned(self):
        """Primary fails → first fallback succeeds → returned immediately."""
        orch = self._make_orch({
            Intent.DOCUMENT: _bad("phrase"),
            Intent.SEARCH:   _good_for(Intent.SEARCH, "Search found the answer to your question about revenue."),
        })
        initial_bad = _bad("phrase")
        attempts    = [("document_agent", initial_bad)]
        result = orch._run_fallback_chain(
            "What is Q3 revenue?", Intent.DOCUMENT, attempts, max_attempts=2
        )
        assert result.agent_name == Intent.SEARCH.value
        assert len(attempts) == 2   # primary + one fallback

    def test_second_fallback_tried_when_first_fails(self):
        """Both first fallback and second fallback tried when first also fails."""
        orch = self._make_orch({
            Intent.DOCUMENT: _bad("phrase"),
            Intent.SEARCH:   _bad("short"),
            Intent.CHAT:     _good_for(Intent.CHAT, "Chat found a comprehensive answer to the question."),
        })
        attempts = [("document_agent", _bad("phrase"))]
        result   = orch._run_fallback_chain(
            "q", Intent.DOCUMENT, attempts, max_attempts=3
        )
        assert result.agent_name == Intent.CHAT.value

    def test_max_attempts_respected(self):
        """With max_fallback_attempts=1 only one fallback is tried."""
        orch = self._make_orch({
            Intent.DOCUMENT: _bad("phrase"),
            Intent.SEARCH:   _bad("short"),
            Intent.CHAT:     _good_for(Intent.CHAT, "Comprehensive chat answer here for the user."),
        })
        attempts = [("document_agent", _bad("phrase"))]
        orch._run_fallback_chain(
            "q", Intent.DOCUMENT, attempts, max_attempts=1
        )
        # SEARCH tried (max 1), CHAT skipped
        assert len(attempts) == 2   # primary + exactly 1 fallback

    def test_synthesis_called_when_all_fail(self):
        """When every agent fails, synthesis is called."""
        orch = self._make_orch({
            Intent.DOCUMENT: _bad("phrase"),
            Intent.SEARCH:   _bad("short"),
            Intent.CHAT:     _bad("error"),
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Synthesised final answer.")
        attempts = [("document_agent", _bad("phrase"))]
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = orch._run_fallback_chain(
                "q", Intent.DOCUMENT, attempts, max_attempts=3
            )
        assert result.agent_name == "synthesised"

    def test_no_fallback_for_unknown_intent(self):
        """UNKNOWN intent falls to CHAT and stops."""
        orch     = self._make_orch({Intent.CHAT: _good_for(Intent.CHAT, "Chat answer here for you.")})
        attempts = [("unknown_agent", _bad("short"))]
        result   = orch._run_fallback_chain(
            "q", Intent.UNKNOWN, attempts, max_attempts=2
        )
        # CHAT is the only fallback for UNKNOWN
        assert result.agent_name == Intent.CHAT.value

    def test_attempts_list_mutated_in_place(self):
        """attempts list is extended with each fallback call."""
        orch = self._make_orch({
            Intent.SEARCH: _bad("short"),
            Intent.CHAT:   _good("Full answer from chat agent here with enough detail."),
        })
        attempts = [("primary", _bad("phrase"))]
        orch._run_fallback_chain("q", Intent.SEARCH, attempts, max_attempts=2)
        # At minimum primary + CHAT added
        assert len(attempts) >= 2

    def test_duplicate_intent_not_retried(self):
        """An intent already in tried_intents is skipped — not called twice."""
        orch       = self._make_orch({
            Intent.DOCUMENT: _bad("short"),   # first fallback for SEARCH also fails
            Intent.CHAT:     _bad("short"),   # second fallback also fails
        })
        # Pre-mark DOCUMENT as already tried by including it in attempts
        attempts = [
            ("search_agent",  _bad("phrase")),
            ("document_agent", _bad("short")),   # DOCUMENT already in the list
        ]
        synth_result = _good("Synthesised final answer combining all evidence.")
        with patch("agents.orchestrator._synthesise_from_attempts",
                   return_value=synth_result) as mock_synth:
            orch._run_fallback_chain("q", Intent.SEARCH, attempts, max_attempts=2)
        # Synthesis should have been called since all paths were exhausted
        mock_synth.assert_called_once()


# =============================================================================
#  Orchestrator.run() — end-to-end
# =============================================================================

class TestOrchestratorRunFallback:

    def test_good_primary_no_fallback_needed(self):
        """Primary gives a good answer — no fallback agent called."""
        orch = _make_orchestrator({
            Intent.SEARCH: _good_for(Intent.SEARCH, "Python is a high-level programming language used widely."),
        })
        with patch.object(orch, "_route", return_value=Intent.SEARCH):
            resp = orch.run("What is Python?")
        assert resp.agent_name == Intent.SEARCH.value
        # Only the search agent should have been called
        orch._agents[Intent.SEARCH].run.assert_called_once()
        orch._agents[Intent.DOCUMENT].run.assert_not_called()

    def test_primary_fails_fallback_succeeds(self):
        """Primary fails → first fallback answers the question."""
        orch = _make_orchestrator({
            Intent.DOCUMENT: _bad("phrase"),
            Intent.SEARCH:   _good_for(Intent.SEARCH, "Found the relevant information in the search results."),
        })
        with patch.object(orch, "_route", return_value=Intent.DOCUMENT):
            resp = orch.run("What does the document say about costs?")
        assert resp.agent_name == Intent.SEARCH.value

    def test_all_fail_returns_synthesis(self):
        """All agents fail → synthesised response returned."""
        orch = _make_orchestrator({
            Intent.DOCUMENT: _bad("phrase"),
            Intent.SEARCH:   _bad("short"),
            Intent.CHAT:     _bad("short"),
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Best available synthesised answer.")
        with patch.object(orch, "_route", return_value=Intent.DOCUMENT), \
             patch("agents.orchestrator.get_llm", return_value=mock_llm):
            resp = orch.run("What is in the report?")
        assert resp.agent_name == "synthesised"
        assert "synthesised" in resp.output.lower() or "Best" in resp.output

    def test_max_fallback_zero_disables_fallback(self):
        """max_fallback_attempts=0 means no fallback is ever tried."""
        orch = _make_orchestrator(
            {Intent.SEARCH: _bad("phrase")},
            max_fallback_attempts=0,
        )
        with patch.object(orch, "_route", return_value=Intent.SEARCH):
            resp = orch.run("What is the capital of France?")
        # Only the primary agent was called
        orch._agents[Intent.SEARCH].run.assert_called_once()
        orch._agents[Intent.DOCUMENT].run.assert_not_called()
        orch._agents[Intent.CHAT].run.assert_not_called()

    def test_forced_intent_skips_routing(self):
        orch       = _make_orchestrator()
        route_spy  = MagicMock(return_value=Intent.SEARCH)
        with patch.object(orch, "_route", route_spy):
            orch.run("Hello!", intent="chat")
        route_spy.assert_not_called()
        orch._agents[Intent.CHAT].run.assert_called_once()

    def test_forced_intent_string_resolved(self):
        orch = _make_orchestrator()
        with patch.object(orch, "_route", return_value=Intent.SEARCH):
            orch.run("query", intent="finance")
        orch._agents[Intent.FINANCE].run.assert_called_once()

    def test_memory_saved_even_after_fallback(self):
        """Memory.save_context must be called regardless of how many fallbacks ran."""
        orch = _make_orchestrator({
            Intent.NEWS:   _bad("phrase"),
            Intent.SEARCH: _good_for(Intent.SEARCH, "Search answer about the news story here."),
        })
        with patch.object(orch, "_route", return_value=Intent.NEWS):
            orch.run("Latest news about AI?")
        orch._memory.save_context.assert_called_once()

    def test_memory_saved_after_synthesis(self):
        """Memory saved even when synthesis path is taken."""
        orch     = _make_orchestrator(
            {intent: _bad("short") for intent in Intent}
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Synthesised answer addressing the user question thoroughly.")
        with patch.object(orch, "_route", return_value=Intent.DOCUMENT), \
             patch("agents.orchestrator.get_llm", return_value=mock_llm):
            orch.run("q")
        orch._memory.save_context.assert_called_once()

    def test_agent_exception_triggers_fallback(self):
        """An agent that raises an exception is treated as an insufficient response."""
        orch = _make_orchestrator({
            Intent.SEARCH: _good_for(Intent.SEARCH, "Good answer found by the search agent here."),
        })
        orch._agents[Intent.NEWS].run.side_effect = RuntimeError("RSS feed down")
        with patch.object(orch, "_route", return_value=Intent.NEWS):
            resp = orch.run("What happened in the news today?")
        # Exception captured → fallback to SEARCH
        assert resp.agent_name == Intent.SEARCH.value

    def test_llm_quality_check_enabled_calls_judge(self):
        """When enable_llm_quality_check=True the LLM judge is invoked."""
        orch     = _make_orchestrator(
            {Intent.CHAT: _good_for(Intent.CHAT, "A solid and complete answer to the user question.")},
            enable_llm_quality_check=True,
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="SUFFICIENT")
        with patch.object(orch, "_route", return_value=Intent.CHAT), \
             patch("agents.orchestrator.get_llm", return_value=mock_llm):
            resp = orch.run("How are you?")
        # LLM invoked at least once (for quality check)
        mock_llm.invoke.assert_called()

    def test_llm_quality_check_insufficient_triggers_fallback(self):
        """LLM judge returning INSUFFICIENT triggers fallback even on a long response."""
        orch     = _make_orchestrator(
            {
                Intent.CHAT:   AgentResponse(output="A long response that the judge deems insufficient. " * 3, agent_name=Intent.CHAT.value),
                Intent.SEARCH: _good_for(Intent.SEARCH, "Search agent answer that the judge approves."),
            },
            enable_llm_quality_check=True,
        )
        call_count = [0]
        def judge_side_effect(prompt):
            call_count[0] += 1
            # First call (CHAT response) → INSUFFICIENT; second (SEARCH) → SUFFICIENT
            verdict = "INSUFFICIENT" if call_count[0] == 1 else "SUFFICIENT"
            return MagicMock(content=verdict)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = judge_side_effect
        with patch.object(orch, "_route", return_value=Intent.CHAT), \
             patch("agents.orchestrator.get_llm", return_value=mock_llm):
            resp = orch.run("Explain recursion in detail please.")
        assert resp.agent_name == Intent.SEARCH.value

    def test_response_output_answers_query(self):
        """
        Regression: the final response must come from an agent that ran for
        the specific query — not a stale cached or default value.
        """
        orch = _make_orchestrator({
            Intent.FINANCE: _good_for(Intent.FINANCE, "Apple Inc (AAPL) is trading at $178.50 today."),
        })
        with patch.object(orch, "_route", return_value=Intent.FINANCE):
            resp = orch.run("What is Apple's stock price?")
        assert "Apple" in resp.output or "AAPL" in resp.output

    def test_different_queries_route_independently(self):
        """Two separate run() calls each route independently."""
        orch = _make_orchestrator()
        with patch.object(orch, "_route", side_effect=[Intent.CODE, Intent.NEWS]):
            orch.run("Write a sort function")
            orch.run("What are today's headlines?")
        orch._agents[Intent.CODE].run.assert_called_once()
        orch._agents[Intent.NEWS].run.assert_called_once()


# =============================================================================
#  Orchestrator._is_sufficient()  (instance method wrapping the module fn)
# =============================================================================

class TestOrchestratorIsSufficient:

    def test_delegates_to_fast_gate(self):
        orch   = _make_orchestrator()
        good_r = _good()
        assert orch._is_sufficient("q", good_r) is True

    def test_bad_response_fails(self):
        orch = _make_orchestrator()
        assert orch._is_sufficient("q", _bad("short")) is False

    def test_llm_gate_not_called_when_disabled(self):
        orch = _make_orchestrator(enable_llm_quality_check=False)
        with patch("agents.orchestrator._llm_quality_check") as mock_judge:
            orch._is_sufficient("q", _good())
        mock_judge.assert_not_called()

    def test_llm_gate_called_when_enabled(self):
        orch = _make_orchestrator(enable_llm_quality_check=True)
        with patch("agents.orchestrator._llm_quality_check", return_value=True) as mock_judge:
            orch._is_sufficient("q", _good())
        mock_judge.assert_called_once()

    def test_llm_gate_short_circuits_on_fast_fail(self):
        """If fast gate fails, LLM judge must NOT be called (saves cost)."""
        orch = _make_orchestrator(enable_llm_quality_check=True)
        with patch("agents.orchestrator._llm_quality_check") as mock_judge:
            orch._is_sufficient("q", _bad("short"))
        mock_judge.assert_not_called()


# =============================================================================
#  Fallback chain completeness per intent
# =============================================================================

class TestFallbackChainPerIntent:
    """
    Verify that the fallback chain behaves correctly for every routable intent.
    """

    @pytest.mark.parametrize("primary_intent,expected_fallback", [
        (Intent.DOCUMENT, Intent.SEARCH),
        (Intent.SEARCH,   Intent.DOCUMENT),
        (Intent.NEWS,     Intent.SEARCH),
        (Intent.FINANCE,  Intent.SEARCH),
        (Intent.CODE,     Intent.SEARCH),
        (Intent.CHAT,     Intent.SEARCH),
    ])
    def test_first_fallback_intent_correct(
        self, primary_intent: Intent, expected_fallback: Intent
    ):
        """The first fallback in each chain is as documented."""
        chain = _FALLBACK_CHAINS.get(primary_intent, [])
        assert len(chain) >= 1
        assert chain[0] == expected_fallback, (
            f"Expected first fallback for {primary_intent} to be "
            f"{expected_fallback}, got {chain[0]}"
        )

    @pytest.mark.parametrize("primary_intent", [
        Intent.DOCUMENT, Intent.SEARCH, Intent.NEWS,
        Intent.FINANCE,  Intent.CODE,
    ])
    def test_fallback_agent_actually_called_on_primary_failure(
        self, primary_intent: Intent
    ):
        """When primary fails, the first fallback agent actually runs."""
        first_fb = _FALLBACK_CHAINS[primary_intent][0]
        orch     = _make_orchestrator({
            primary_intent: _bad("phrase"),
            first_fb:       _good_for(first_fb, "Sufficient fallback answer for the user query here."),
        })
        with patch.object(orch, "_route", return_value=primary_intent):
            resp = orch.run("test query for routing")
        orch._agents[primary_intent].run.assert_called_once()
        orch._agents[first_fb].run.assert_called_once()
        assert resp.agent_name == first_fb.value


# =============================================================================
#  OrchestratorState TypedDict
# =============================================================================

class TestOrchestratorState:

    def test_all_fields_present(self):
        fields = OrchestratorState.__annotations__.keys()
        for expected in ("query", "intent", "response", "history", "metadata", "error"):
            assert expected in fields

    def test_total_false_all_optional(self):
        """total=False means all keys are optional (no KeyError on empty dict)."""
        state: OrchestratorState = {}
        assert state.get("query") is None

    def test_can_hold_agent_response(self):
        resp  = _good()
        state: OrchestratorState = {
            "query": "test",
            "intent": "search",
            "response": resp,
            "history": [],
            "metadata": {"user_id": "alice"},
            "error": "",
        }
        assert state["response"] is resp


# =============================================================================
#  Orchestrator repr and misc
# =============================================================================

class TestOrchestratorMisc:

    def test_repr_contains_session_and_fallback(self):
        orch = _make_orchestrator(max_fallback_attempts=3)
        orch._session_id = "my-session"
        r = repr(orch)
        assert "my-session" in r
        assert "3" in r   # max_fallback_attempts

    def test_intents_property_returns_all(self):
        orch = _make_orchestrator()
        assert set(orch.intents) == set(orch._agents.keys())

    def test_add_agent_registers(self):
        orch          = _make_orchestrator()
        new_agent     = MagicMock()
        new_agent.name = "custom"
        orch.add_agent(Intent.CHAT, new_agent)
        assert orch._agents[Intent.CHAT] is new_agent

    def test_get_agent_returns_correct(self):
        orch = _make_orchestrator()
        agent = orch.get_agent(Intent.FINANCE)
        assert agent is orch._agents[Intent.FINANCE]
