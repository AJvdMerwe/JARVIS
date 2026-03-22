"""
tests/test_deep_research_agent.py
───────────────────────────────────
Tests for:
  • DeepResearchAgent — plan, gather, evaluate, synthesise, fallback
  • ResearchPlan, EvidenceItem, ResearchState data models
  • _strip_thinking() helper
  • _parse_search_results() helper
  • _format_evidence_summary() and _format_evidence_for_synthesis()
  • Orchestrator intent routing for research queries
  • _RESEARCH_KEYWORDS pattern
  • Intent.RESEARCH in fallback chains
  • DeepResearchAgent registered in Orchestrator
  • RAG pre-check skip for RESEARCH intent
  • get_reasoning_llm() in llm_manager
  • New settings fields
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from agents.base_agent import AgentResponse


# =============================================================================
#  Helpers
# =============================================================================

def _make_agent(
    plan_output: str | None = None,
    evaluate_output: str = "SUFFICIENT",
    synthesise_output: str = "# Research Report\n\n## Executive Summary\n\nThis is a comprehensive research report about Python programming with detailed citations [1]. The language was created in 1991.\n\n## References\n\n1. https://python.org",
    search_output: str = "Python is a programming language.\nhttps://python.org\nCreated by Guido van Rossum.",
    wiki_output: str  = "Python (programming language): High-level, interpreted programming language. Created in 1991 by Guido van Rossum.",
    max_iterations: int = 3,
    max_sources: int    = 6,
):
    """Build a DeepResearchAgent with all LLM/tool calls mocked."""
    from agents.deep_research_agent import DeepResearchAgent
    from core.memory import AssistantMemory

    mock_chat_llm = MagicMock()
    mock_chat_llm.invoke.return_value = MagicMock(content="SUFFICIENT")

    if plan_output is None:
        plan_output = json.dumps({
            "topic": "Python programming",
            "sub_questions": [
                "What is Python's origin and history?",
                "What are Python's main use cases?",
                "What are Python's strengths and weaknesses?",
            ],
            "search_queries": [
                "Python programming language history",
                "Python use cases applications",
            ],
        })

    mock_reasoning_llm = MagicMock()
    # First call → plan, subsequent calls → evaluate or synthesise
    call_responses = [
        MagicMock(content=plan_output),
        MagicMock(content=evaluate_output),
        MagicMock(content=synthesise_output),
    ]
    mock_reasoning_llm.invoke.side_effect = call_responses + [
        MagicMock(content=synthesise_output)
    ] * 10  # extra calls in case of retries

    with patch("agents.deep_research_agent.DeepResearchAgent._load_reasoning_llm",
               return_value=mock_reasoning_llm):
        agent = DeepResearchAgent(
            llm=mock_chat_llm,
            reasoning_llm=mock_reasoning_llm,
            memory=AssistantMemory(),
            max_iterations=max_iterations,
            max_sources=max_sources,
        )

    # Mock search tools
    mock_web  = MagicMock(); mock_web.name = "web_search"
    mock_web._run.return_value = search_output
    mock_wiki = MagicMock(); mock_wiki.name = "wikipedia_lookup"
    mock_wiki._run.return_value = wiki_output

    agent._search_tools = {
        "web_search":       mock_web,
        "wikipedia_lookup": mock_wiki,
    }
    agent._reasoning_llm = mock_reasoning_llm

    return agent, mock_reasoning_llm, mock_web, mock_wiki


def _make_orchestrator(enable_rag: bool = False) -> "Orchestrator":
    """Build a test Orchestrator with all agents mocked."""
    from agents.orchestrator import Orchestrator, Intent

    orch = Orchestrator.__new__(Orchestrator)
    orch._session_id               = "test"
    orch._enable_llm_quality_check = False
    orch._enable_rag_precheck      = enable_rag
    orch._rag_similarity_threshold = 0.55
    orch._rag_k                    = 4
    orch._max_fallback_attempts    = 0
    orch._summariser               = None
    orch._episodic                 = None

    mock_memory = MagicMock()
    mock_memory.messages = []
    orch._memory = mock_memory

    orch._agents = {}
    for intent in Intent:
        agent      = MagicMock()
        agent.name = intent.value
        agent.run.return_value = AgentResponse(
            output=f"Response from {intent.value} agent with enough detail.",
            agent_name=intent.value,
        )
        orch._agents[intent] = agent

    return orch


# =============================================================================
#  Data models
# =============================================================================

class TestResearchPlan:

    def test_str_representation(self):
        from agents.deep_research_agent import ResearchPlan
        plan = ResearchPlan(
            topic="Python",
            sub_questions=["What is Python?", "Who created it?"],
            search_queries=["Python language", "Guido van Rossum"],
        )
        s = str(plan)
        assert "Python" in s
        assert "What is Python?" in s

    def test_all_fields_accessible(self):
        from agents.deep_research_agent import ResearchPlan
        plan = ResearchPlan(
            topic="AI", sub_questions=["q1"], search_queries=["s1"]
        )
        assert plan.topic == "AI"
        assert plan.sub_questions == ["q1"]
        assert plan.search_queries == ["s1"]


class TestEvidenceItem:

    def test_defaults(self):
        from agents.deep_research_agent import EvidenceItem
        item = EvidenceItem(source="https://example.com", content="text", query="q")
        assert item.score == 1.0

    def test_custom_score(self):
        from agents.deep_research_agent import EvidenceItem
        item = EvidenceItem(source="s", content="c", query="q", score=0.75)
        assert item.score == 0.75


class TestResearchState:

    def test_initial_state(self):
        from agents.deep_research_agent import ResearchState
        state = ResearchState(query="test query")
        assert state.query == "test query"
        assert state.plan is None
        assert state.evidence == []
        assert state.iterations == 0
        assert state.done is False
        assert state.searches_run == []


# =============================================================================
#  _strip_thinking helper
# =============================================================================

class TestStripThinking:

    def test_removes_think_block(self):
        from agents.deep_research_agent import DeepResearchAgent
        raw = "<think>Let me reason about this carefully.</think>The answer is 42."
        assert DeepResearchAgent._strip_thinking(raw) == "The answer is 42."

    def test_removes_multiline_think_block(self):
        from agents.deep_research_agent import DeepResearchAgent
        raw = "<think>\nStep 1: ...\nStep 2: ...\n</think>\nFinal answer."
        assert DeepResearchAgent._strip_thinking(raw) == "Final answer."

    def test_handles_pipe_variant(self):
        from agents.deep_research_agent import DeepResearchAgent
        raw = "<|think|>reasoning here</|think|>Result"
        result = DeepResearchAgent._strip_thinking(raw)
        assert "reasoning here" not in result
        assert "Result" in result

    def test_no_think_block_unchanged(self):
        from agents.deep_research_agent import DeepResearchAgent
        text = "Plain answer without any thinking block."
        assert DeepResearchAgent._strip_thinking(text) == text

    def test_empty_string(self):
        from agents.deep_research_agent import DeepResearchAgent
        assert DeepResearchAgent._strip_thinking("") == ""

    def test_only_think_block_returns_empty(self):
        from agents.deep_research_agent import DeepResearchAgent
        raw = "<think>internal reasoning only</think>"
        assert DeepResearchAgent._strip_thinking(raw) == ""


# =============================================================================
#  _parse_search_results helper
# =============================================================================

class TestParseSearchResults:

    def test_extracts_url_as_source(self):
        from agents.deep_research_agent import DeepResearchAgent
        from core.memory import AssistantMemory
        with patch("agents.deep_research_agent.DeepResearchAgent._load_reasoning_llm",
                   return_value=MagicMock()):
            agent = DeepResearchAgent.__new__(DeepResearchAgent)
        # DuckDuckGo output separates results with double newlines
        raw = (
            "Python programming language overview.\n"
            "https://python.org\n"
            "Python is a high-level language created by Guido van Rossum.\n\n"
            "History of Python.\n"
            "https://en.wikipedia.org/wiki/Python_(programming_language)\n"
            "Python was first released in 1991."
        )
        items = agent._parse_search_results(raw, "Python history")
        assert len(items) >= 1
        assert any("python.org" in i.source or "wikipedia" in i.source
                   for i in items)

    def test_content_populated(self):
        from agents.deep_research_agent import DeepResearchAgent
        agent = DeepResearchAgent.__new__(DeepResearchAgent)
        raw = "Python history information here.\nhttps://example.com\nMore detail about Python language and its history."
        items = agent._parse_search_results(raw, "python")
        assert all(len(i.content) > 10 for i in items)

    def test_empty_raw_returns_empty_list(self):
        from agents.deep_research_agent import DeepResearchAgent
        agent = DeepResearchAgent.__new__(DeepResearchAgent)
        items = agent._parse_search_results("", "query")
        assert items == []

    def test_query_recorded_on_items(self):
        from agents.deep_research_agent import DeepResearchAgent
        agent = DeepResearchAgent.__new__(DeepResearchAgent)
        raw = "https://example.com\nSufficiently long content here about the topic in question."
        items = agent._parse_search_results(raw, "my_query")
        assert all(i.query == "my_query" for i in items)


# =============================================================================
#  _plan step
# =============================================================================

class TestDeepResearchAgentPlan:

    def test_plan_returns_research_plan(self):
        from agents.deep_research_agent import ResearchPlan
        agent, reasoning_llm, _, _ = _make_agent()
        plan = agent._plan("What is Python?")
        assert isinstance(plan, ResearchPlan)
        assert len(plan.sub_questions) >= 1
        assert len(plan.search_queries) >= 1

    def test_plan_calls_reasoning_llm(self):
        agent, reasoning_llm, _, _ = _make_agent()
        agent._plan("What is Python?")
        reasoning_llm.invoke.assert_called()

    def test_plan_prompt_contains_query(self):
        agent, reasoning_llm, _, _ = _make_agent()
        agent._plan("Deep research on quantum computing")
        prompt = reasoning_llm.invoke.call_args.args[0]
        assert "quantum computing" in prompt

    def test_plan_graceful_fallback_on_invalid_json(self):
        from agents.deep_research_agent import DeepResearchAgent, ResearchPlan
        from core.memory import AssistantMemory
        mock_reasoning = MagicMock()
        mock_reasoning.invoke.return_value = MagicMock(content="not valid json at all")
        with patch("agents.deep_research_agent.DeepResearchAgent._load_reasoning_llm",
                   return_value=mock_reasoning):
            agent = DeepResearchAgent(
                llm=MagicMock(), reasoning_llm=mock_reasoning,
                memory=AssistantMemory()
            )
        plan = agent._plan("test query")
        assert isinstance(plan, ResearchPlan)
        assert plan.topic != ""

    def test_plan_strips_thinking_blocks(self):
        from agents.deep_research_agent import ResearchPlan
        plan_json = json.dumps({
            "topic": "test", "sub_questions": ["q1"], "search_queries": ["s1"]
        })
        thinking_wrapped = f"<think>I will plan this carefully.</think>{plan_json}"
        agent, reasoning_llm, _, _ = _make_agent(plan_output=thinking_wrapped)
        plan = agent._plan("test")
        assert isinstance(plan, ResearchPlan)
        assert plan.topic == "test"


# =============================================================================
#  _gather_for_query step
# =============================================================================

class TestDeepResearchAgentGather:

    def test_gather_adds_evidence_from_web_search(self):
        from agents.deep_research_agent import ResearchState
        agent, _, mock_web, _ = _make_agent(
            search_output="https://python.org\nPython is a language used widely.\n\nMore Python details here."
        )
        state = ResearchState(query="Python")
        agent._gather_for_query("Python programming", state)
        assert len(state.evidence) >= 1

    def test_gather_adds_evidence_from_wikipedia(self):
        from agents.deep_research_agent import ResearchState
        agent, _, mock_web, mock_wiki = _make_agent(search_output="")
        mock_web._run.return_value = ""   # web search returns nothing
        state = ResearchState(query="Python")
        agent._gather_for_query("Python programming", state)
        assert any("Wikipedia" in e.source for e in state.evidence)

    def test_gather_records_query_in_searches_run(self):
        from agents.deep_research_agent import ResearchState
        agent, _, _, _ = _make_agent()
        state = ResearchState(query="test")
        agent._gather_for_query("unique search query here", state)
        assert "unique search query here" in state.searches_run

    def test_gather_skips_duplicate_query(self):
        from agents.deep_research_agent import ResearchState
        agent, _, mock_web, _ = _make_agent()
        state = ResearchState(query="test", searches_run=["duplicate query"])
        agent._gather_for_query("duplicate query", state)
        mock_web._run.assert_not_called()

    def test_gather_respects_max_sources(self):
        from agents.deep_research_agent import ResearchState, EvidenceItem
        agent, _, _, _ = _make_agent(max_sources=3)
        state = ResearchState(query="test")
        # Pre-fill to limit
        state.evidence = [
            EvidenceItem(source="s1", content="c" * 100, query="q")
            for _ in range(3)
        ]
        agent._gather_for_query("extra query", state)
        assert len(state.evidence) == 3  # not exceeded

    def test_gather_tool_failure_does_not_raise(self):
        from agents.deep_research_agent import ResearchState
        agent, _, mock_web, _ = _make_agent()
        mock_web._run.side_effect = RuntimeError("search API down")
        state = ResearchState(query="test")
        # Should not raise
        agent._gather_for_query("any query", state)


# =============================================================================
#  _evaluate step
# =============================================================================

class TestDeepResearchAgentEvaluate:

    def test_returns_sufficient_when_llm_says_so(self):
        from agents.deep_research_agent import ResearchState, ResearchPlan, EvidenceItem
        agent, reasoning_llm, _, _ = _make_agent(evaluate_output="SUFFICIENT")
        # Override subsequent LLM calls
        reasoning_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({
                "topic": "t", "sub_questions": ["q1"], "search_queries": ["s1"]
            })),
            MagicMock(content="SUFFICIENT"),
        ]
        state = ResearchState(query="test")
        state.plan = ResearchPlan("t", ["q1"], ["s1"])
        state.evidence = [EvidenceItem("src", "content text here " * 5, "q")]
        result = agent._evaluate(state)
        assert result == "SUFFICIENT"

    def test_returns_need_more_with_query(self):
        from agents.deep_research_agent import ResearchState, ResearchPlan, EvidenceItem
        agent, reasoning_llm, _, _ = _make_agent()
        reasoning_llm.invoke.side_effect = [
            MagicMock(content="NEED_MORE: Python performance benchmarks 2024"),
        ] * 10
        state = ResearchState(query="Python")
        state.plan = ResearchPlan("Python", ["perf?"], ["python perf"])
        state.evidence = [EvidenceItem("src", "x " * 50, "q")]
        result = agent._evaluate(state)
        assert result.startswith("NEED_MORE")
        assert "Python performance" in result

    def test_evaluate_fails_open_on_llm_error(self):
        from agents.deep_research_agent import ResearchState, ResearchPlan, EvidenceItem
        agent, reasoning_llm, _, _ = _make_agent()
        reasoning_llm.invoke.side_effect = RuntimeError("LLM down")
        state = ResearchState(query="test")
        state.plan = ResearchPlan("t", ["q"], ["s"])
        state.evidence = [EvidenceItem("s", "c " * 10, "q")]
        # Should not raise and should return SUFFICIENT (fail-open)
        result = agent._evaluate(state)
        assert result == "SUFFICIENT"

    def test_no_plan_returns_sufficient(self):
        from agents.deep_research_agent import ResearchState
        agent, _, _, _ = _make_agent()
        state = ResearchState(query="test")
        result = agent._evaluate(state)
        assert result == "SUFFICIENT"


# =============================================================================
#  Full run() integration
# =============================================================================

class TestDeepResearchAgentRun:

    def test_run_returns_agent_response(self):
        agent, _, _, _ = _make_agent()
        resp = agent.run("What is Python programming?")
        assert isinstance(resp, AgentResponse)
        assert resp.agent_name == "deep_research_agent"
        assert resp.error is None

    def test_run_output_is_non_empty(self):
        agent, _, _, _ = _make_agent()
        resp = agent.run("Research Python programming language")
        assert len(resp.output) > 50

    def test_run_calls_plan_once(self):
        agent, reasoning_llm, _, _ = _make_agent()
        agent.run("Research Python")
        # First call to reasoning_llm is always the plan
        first_prompt = reasoning_llm.invoke.call_args_list[0].args[0]
        assert "Python" in first_prompt

    def test_run_calls_search_tools(self):
        agent, _, mock_web, mock_wiki = _make_agent()
        agent.run("Research Python programming language")
        assert mock_web._run.called or mock_wiki._run.called

    def test_run_populates_references(self):
        agent, _, _, _ = _make_agent()
        resp = agent.run("Research Python programming")
        # References should be populated from gathered evidence
        assert isinstance(resp.references, list)

    def test_run_populates_metadata(self):
        agent, _, _, _ = _make_agent()
        resp = agent.run("Research Python")
        assert "iterations"    in resp.metadata
        assert "sources_found" in resp.metadata
        assert "elapsed_s"     in resp.metadata
        assert "model"         in resp.metadata

    def test_run_respects_max_iterations(self):
        """Agent should never exceed max_iterations evaluate cycles."""
        agent, reasoning_llm, _, _ = _make_agent(
            evaluate_output="NEED_MORE: another search query needed",
            max_iterations=2,
        )
        # Always return NEED_MORE so the loop terminates via max_iterations
        plan_json = json.dumps({
            "topic": "t", "sub_questions": ["q"], "search_queries": ["s"]
        })
        reasoning_llm.invoke.side_effect = (
            [MagicMock(content=plan_json)] +
            [MagicMock(content="NEED_MORE: query " + str(i)) for i in range(20)] +
            [MagicMock(content="# Final report\n\n## Summary\n\nContent here [1].\n\n## References\n\n1. https://src")]
        )
        resp = agent.run("research topic")
        assert resp.metadata["iterations"] <= 2

    def test_run_error_returns_error_response(self):
        from agents.deep_research_agent import DeepResearchAgent
        from core.memory import AssistantMemory
        mock_reasoning = MagicMock()
        mock_reasoning.invoke.side_effect = Exception("catastrophic failure")
        with patch("agents.deep_research_agent.DeepResearchAgent._load_reasoning_llm",
                   return_value=mock_reasoning):
            agent = DeepResearchAgent(
                llm=MagicMock(), reasoning_llm=mock_reasoning,
                memory=AssistantMemory(), max_iterations=1
            )
        resp = agent.run("research topic")
        assert resp.error is not None or "error" in resp.output.lower()

    def test_run_loop_stops_early_on_sufficient(self):
        """When LLM says SUFFICIENT, no more searches should happen."""
        agent, _, mock_web, _ = _make_agent(evaluate_output="SUFFICIENT")
        calls_before = mock_web._run.call_count
        agent.run("test research topic")
        # The loop terminated — web search called for initial queries only
        total_calls = mock_web._run.call_count - calls_before
        assert total_calls <= 5   # initial queries only, no extra iterations

    def test_run_tool_calls_populated(self):
        agent, _, _, _ = _make_agent()
        resp = agent.run("Research Python")
        assert isinstance(resp.tool_calls, list)


# =============================================================================
#  Format helpers
# =============================================================================

class TestFormatHelpers:

    def _agent(self):
        from agents.deep_research_agent import DeepResearchAgent
        from core.memory import AssistantMemory
        with patch("agents.deep_research_agent.DeepResearchAgent._load_reasoning_llm",
                   return_value=MagicMock()):
            return DeepResearchAgent(
                llm=MagicMock(),
                reasoning_llm=MagicMock(),
                memory=AssistantMemory(),
            )

    def _items(self, n: int = 3):
        from agents.deep_research_agent import EvidenceItem
        return [
            EvidenceItem(f"https://source{i}.com", "content " * 50, f"query {i}")
            for i in range(n)
        ]

    def test_evidence_summary_truncates_to_max_chars(self):
        agent = self._agent()
        items = self._items(10)
        summary = agent._format_evidence_summary(items, max_chars=500)
        assert len(summary) <= 600   # some slack for header overhead

    def test_evidence_summary_includes_source_refs(self):
        agent = self._agent()
        items = self._items(2)
        summary = agent._format_evidence_summary(items)
        assert "[1]" in summary
        assert "[2]" in summary

    def test_evidence_for_synthesis_respects_chunk_budget(self):
        agent = self._agent()
        agent._chunk_budget = 1000
        items = self._items(10)
        evidence = agent._format_evidence_for_synthesis(items)
        assert len(evidence) <= 1200   # budget + small overhead

    def test_evidence_for_synthesis_numbers_sources(self):
        agent = self._agent()
        items = self._items(3)
        evidence = agent._format_evidence_for_synthesis(items)
        assert "[1]" in evidence
        assert "[2]" in evidence

    def test_fallback_report_always_returns_string(self):
        from agents.deep_research_agent import ResearchState, ResearchPlan, EvidenceItem
        agent = self._agent()
        state = ResearchState(query="test topic")
        state.plan = ResearchPlan("test", ["q1"], ["s1"])
        state.evidence = [
            EvidenceItem("https://src.com", "Evidence content here", "query")
        ]
        report = agent._fallback_report(state)
        assert isinstance(report, str)
        assert "test topic" in report or "Research Report" in report


# =============================================================================
#  Orchestrator routing — RESEARCH intent
# =============================================================================

class TestResearchRouting:

    @pytest.mark.parametrize("query", [
        "Deep research on climate change",
        "Comprehensive report on quantum computing",
        "Thoroughly research machine learning",
        "Research the history of the internet",
        "Deep dive into blockchain technology",
        "Literature review on CRISPR gene editing technology",
        "In-depth analysis of renewable energy",
        "Research into artificial intelligence safety",
        "Tell me everything about the Roman Empire",
    ])
    def test_research_queries_route_to_research(self, query: str):
        from agents.orchestrator import _keyword_route, Intent
        result = _keyword_route(query)
        assert result == Intent.RESEARCH, (
            f"Expected RESEARCH for {query!r}, got {result}"
        )

    def test_plain_search_not_routed_to_research(self):
        from agents.orchestrator import _keyword_route, Intent
        result = _keyword_route("What is the capital of France?")
        assert result != Intent.RESEARCH

    def test_chat_not_routed_to_research(self):
        from agents.orchestrator import _keyword_route, Intent
        result = _keyword_route("Hello, how are you?")
        assert result != Intent.RESEARCH

    def test_research_intent_in_enum(self):
        from agents.orchestrator import Intent
        assert Intent.RESEARCH in Intent
        assert Intent.RESEARCH.value == "research"

    def test_research_in_fallback_chains(self):
        from agents.orchestrator import _FALLBACK_CHAINS, Intent
        assert Intent.RESEARCH in _FALLBACK_CHAINS
        chain = _FALLBACK_CHAINS[Intent.RESEARCH]
        assert Intent.SEARCH in chain
        assert Intent.CHAT in chain

    def test_deep_research_agent_registered(self):
        from agents.orchestrator import Orchestrator, Intent
        with patch("agents.orchestrator.PersistentMemory"), \
             patch("agents.orchestrator.ChatAgent"), \
             patch("agents.orchestrator.CodeAgent"), \
             patch("agents.orchestrator.NewsAgent"), \
             patch("agents.orchestrator.SearchAgent"), \
             patch("agents.orchestrator.DocumentAgent"), \
             patch("agents.orchestrator.FinancialAgent"), \
             patch("agents.orchestrator.DeepResearchAgent") as mock_dr:
            orch = Orchestrator(session_id="test-research")
        mock_dr.assert_called_once()
        assert Intent.RESEARCH in orch._agents

    def test_orchestrator_routes_research_query(self):
        orch = _make_orchestrator()
        with patch.object(orch, "_route", return_value=__import__(
                "agents.orchestrator", fromlist=["Intent"]).Intent.RESEARCH
        ):
            resp = orch.run("Research the history of Python")
        from agents.orchestrator import Intent
        orch._agents[Intent.RESEARCH].run.assert_called_once()

    def test_rag_precheck_skips_research_intent(self):
        """RESEARCH intent must bypass the RAG pre-check."""
        from agents.rag_precheck import rag_precheck, reset_singletons
        reset_singletons()
        # Even with a populated KB, research queries skip the pre-check
        dm = MagicMock()
        dm.total_chunks = 50
        dm.search.return_value = []
        with patch("agents.rag_precheck._get_document_manager", return_value=dm):
            result = rag_precheck("Research the history of AI", "research")
        assert result is None
        dm.search.assert_not_called()   # skipped before even searching


# =============================================================================
#  get_reasoning_llm() in llm_manager
# =============================================================================

class TestGetReasoningLlm:

    def setup_method(self):
        from core.llm_manager import get_reasoning_llm
        get_reasoning_llm.cache_clear()

    def teardown_method(self):
        from core.llm_manager import get_reasoning_llm
        get_reasoning_llm.cache_clear()

    def test_returns_chat_ollama_instance(self):
        from core.llm_manager import get_reasoning_llm
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.ChatOllama = mock_cls

        with patch.object(settings, "ollama_reasoning_model", "deepseek-r1:7b"), \
             patch.object(settings, "ollama_base_url", "http://localhost:11434"), \
             patch.dict("sys.modules", {"langchain_ollama": mock_pkg}):
            llm = get_reasoning_llm()

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "deepseek-r1:7b"

    def test_uses_configured_reasoning_model(self):
        from core.llm_manager import get_reasoning_llm
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.ChatOllama = mock_cls

        with patch.object(settings, "ollama_reasoning_model", "qwen3:14b"), \
             patch.object(settings, "ollama_base_url", "http://myserver:11434"), \
             patch.dict("sys.modules", {"langchain_ollama": mock_pkg}):
            get_reasoning_llm()

        kwargs = mock_cls.call_args.kwargs
        assert kwargs["model"] == "qwen3:14b"
        assert kwargs["base_url"] == "http://myserver:11434"

    def test_is_cached(self):
        from core.llm_manager import get_reasoning_llm
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.ChatOllama = mock_cls

        with patch.object(settings, "ollama_reasoning_model", "deepseek-r1:7b"), \
             patch.dict("sys.modules", {"langchain_ollama": mock_pkg}):
            r1 = get_reasoning_llm()
            r2 = get_reasoning_llm()

        assert r1 is r2
        mock_cls.assert_called_once()

    def test_separate_from_get_llm_cache(self):
        """get_reasoning_llm and get_llm are independent singletons."""
        from core.llm_manager import get_reasoning_llm, get_llm
        # They must be different @lru_cache objects
        assert get_reasoning_llm is not get_llm


# =============================================================================
#  New Settings fields
# =============================================================================

class TestResearchSettings:

    def test_default_reasoning_model(self):
        from config.settings import Settings
        s = Settings()
        assert s.ollama_reasoning_model == "deepseek-r1:7b"

    def test_default_max_iterations(self):
        from config.settings import Settings
        s = Settings()
        assert s.research_max_iterations == 5

    def test_default_max_sources(self):
        from config.settings import Settings
        s = Settings()
        assert s.research_max_sources == 8

    def test_default_chunk_budget(self):
        from config.settings import Settings
        s = Settings()
        assert s.research_chunk_budget == 6000

    def test_custom_reasoning_model(self):
        from config.settings import Settings
        s = Settings(ollama_reasoning_model="qwen3:14b")
        assert s.ollama_reasoning_model == "qwen3:14b"

    def test_iteration_limit_validated(self):
        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            Settings(research_max_iterations=0)

    def test_env_reasoning_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_REASONING_MODEL", "deepseek-r1:14b")
        from config.settings import Settings
        s = Settings()
        assert s.ollama_reasoning_model == "deepseek-r1:14b"
