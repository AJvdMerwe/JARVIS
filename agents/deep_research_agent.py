"""
agents/deep_research_agent.py
──────────────────────────────
Multi-turn deep research agent powered by a reasoning model (e.g. deepseek-r1).

Architecture
────────────

  User query
      │
      ▼
  ① Plan       — reasoning model breaks the topic into sub-questions
      │
      ▼
  ② Gather     — for each sub-question: web search + Wikipedia + fetch
      │            (up to research_max_iterations iterations)
      │
      ▼
  ③ Evaluate   — reasoning model scores evidence coverage:
      │            "Is there enough to write a thorough report?"
      │            YES → move to Synthesise
      │            NO  → generate one more targeted search query
      │
      ▼
  ④ Synthesise — reasoning model writes a structured Markdown report
                  citing every source by number

Key properties
──────────────
• Uses the *reasoning model* (OLLAMA_REASONING_MODEL, default deepseek-r1:7b)
  for planning, evaluation and synthesis — deliberate, step-by-step output.
• Uses the *chat model* (OLLAMA_MODEL) only as a cheap classifier to decide
  whether another search iteration is needed.
• Multi-turn: each gather→evaluate cycle is a separate LLM call with the
  accumulated evidence in context.
• Bounded: respects ``research_max_iterations`` and ``research_max_sources``
  from settings to stay within VRAM and time budgets.
• Falls back gracefully: any tool failure is logged and skipped; research
  continues with whatever evidence was gathered.

Routing
───────
The orchestrator routes to this agent when ``Intent.RESEARCH`` is detected.
Typical trigger queries:
  "Research …", "Deep dive into …", "Comprehensive report on …",
  "Thoroughly analyse …", "What does the literature say about …"
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
#  Internal data models
# =============================================================================

@dataclass
class ResearchPlan:
    """The structured plan produced by the reasoning model in Step ①."""
    topic:         str
    sub_questions: list[str]
    search_queries: list[str]     # initial queries derived from sub-questions

    def __str__(self) -> str:
        lines = [f"Topic: {self.topic}", "Sub-questions:"]
        for i, sq in enumerate(self.sub_questions, 1):
            lines.append(f"  {i}. {sq}")
        return "\n".join(lines)


@dataclass
class EvidenceItem:
    """A single gathered piece of evidence from any tool."""
    source:  str         # URL or "Wikipedia: <title>"
    content: str         # raw text (may be truncated)
    query:   str         # the search query that retrieved this
    score:   float = 1.0 # relevance score placeholder


@dataclass
class ResearchState:
    """Mutable state threaded through the gather→evaluate loop."""
    query:       str
    plan:        Optional[ResearchPlan] = None
    evidence:    list[EvidenceItem]     = field(default_factory=list)
    iterations:  int                    = 0
    searches_run: list[str]             = field(default_factory=list)
    done:        bool                   = False


# =============================================================================
#  Prompts
# =============================================================================

_PLAN_PROMPT = """You are a rigorous research planner. Given a research question,
produce a structured research plan.

Research question: {query}

Output a JSON object with exactly these keys:
{{
  "topic": "<concise topic label>",
  "sub_questions": ["<sub-question 1>", "<sub-question 2>", ...],
  "search_queries": ["<initial search query 1>", "<initial search query 2>", ...]
}}

Rules:
- 3–6 sub-questions that together cover the full topic
- 3–5 distinct search queries (concrete, specific, ready to paste into a search engine)
- Return ONLY the JSON — no markdown fences, no explanation

JSON:"""

_EVALUATE_PROMPT = """You are evaluating whether collected research evidence is
sufficient to write a thorough, well-cited report.

Research question: {query}

Evidence collected so far ({n_items} items from {n_sources} unique sources):
{evidence_summary}

Sub-questions still to answer:
{unanswered}

Decide:
- SUFFICIENT: the evidence covers the topic well enough for a thorough report
- NEED_MORE: important sub-questions lack evidence; output one targeted search query

Reply with exactly one of:
  SUFFICIENT
  NEED_MORE: <single targeted search query>

Decision:"""

_SYNTHESISE_PROMPT = """You are a professional research analyst. Write a comprehensive,
well-structured research report answering the question below.

Research question: {query}

Research plan:
{plan}

Evidence ({n_items} items):
{evidence}

Instructions:
- Use Markdown with clear headings (##, ###)
- Cite sources inline as [1], [2], etc. matching the evidence numbers
- Include an Executive Summary at the top
- Cover all sub-questions from the plan
- End with a numbered References section listing every source URL
- Be factual, balanced, and acknowledge uncertainty where present
- Aim for depth and thoroughness — this is a research report, not a summary
- If evidence is conflicting, discuss both sides

Report:"""


# =============================================================================
#  DeepResearchAgent
# =============================================================================

class DeepResearchAgent(BaseAgent):
    """
    Multi-turn research agent that uses a reasoning model to plan, gather,
    evaluate, and synthesise information into a comprehensive report.

    Parameters
    ----------
    llm : BaseChatModel, optional
        Chat LLM for cheap classification/routing.  Defaults to ``get_llm()``.
    reasoning_llm : BaseChatModel, optional
        Reasoning model for plan generation, evaluation, and synthesis.
        Defaults to ``get_reasoning_llm()``.
    memory : AssistantMemory, optional
        Shared session memory.
    max_iterations : int, optional
        Maximum gather→evaluate cycles.  Defaults to ``settings.research_max_iterations``.
    max_sources : int, optional
        Maximum evidence items to collect.  Defaults to ``settings.research_max_sources``.
    chunk_budget : int, optional
        Maximum characters of evidence to include in the synthesis prompt.
        Defaults to ``settings.research_chunk_budget``.
    """

    def __init__(
        self,
        llm:              Optional[BaseChatModel] = None,
        reasoning_llm:    Optional[BaseChatModel] = None,
        memory:           Optional[AssistantMemory] = None,
        max_iterations:   int  = settings.research_max_iterations,
        max_sources:      int  = settings.research_max_sources,
        chunk_budget:     int  = settings.research_chunk_budget,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=False)
        self._reasoning_llm = reasoning_llm or self._load_reasoning_llm()
        self._max_iterations = max_iterations
        self._max_sources    = max_sources
        self._chunk_budget   = chunk_budget

        # Lazily initialised tools
        self._search_tools: Optional[dict[str, BaseTool]] = None

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "deep_research_agent"

    @property
    def description(self) -> str:
        return (
            "Multi-turn deep research agent. Produces a comprehensive, cited "
            "Markdown report by planning, iteratively gathering evidence from "
            "web search and Wikipedia, and synthesising via a reasoning model. "
            "Use for: 'Research X', 'Deep dive into Y', 'Comprehensive report on Z'."
        )

    def get_tools(self) -> list[BaseTool]:
        return list(self._get_search_tools().values())

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Execute a multi-turn research loop and return a comprehensive report.

        Steps
        -----
        1. Plan: reasoning model generates sub-questions and initial search queries.
        2. Gather: run initial searches and collect evidence.
        3. Evaluate → continue loop until evidence is SUFFICIENT or limit reached.
        4. Synthesise: reasoning model writes the final Markdown report.

        Parameters
        ----------
        query : str
            The research question or topic.

        Returns
        -------
        AgentResponse
            Comprehensive Markdown report with citations.
        """
        t0 = time.monotonic()
        self._logger.info("DeepResearchAgent starting: %s", query[:80])

        state = ResearchState(query=query)

        try:
            # ── Step 1: Plan ──────────────────────────────────────────────────
            state.plan = self._plan(query)
            self._logger.info(
                "Research plan: %d sub-questions, %d initial queries",
                len(state.plan.sub_questions),
                len(state.plan.search_queries),
            )

            # ── Step 2 + 3: Gather → Evaluate loop ───────────────────────────
            # Run initial queries from the plan
            for q in state.plan.search_queries:
                self._gather_for_query(q, state)
                if len(state.evidence) >= self._max_sources:
                    break

            # Iterative refinement
            while (
                not state.done
                and state.iterations < self._max_iterations
                and len(state.evidence) < self._max_sources
            ):
                state.iterations += 1
                decision = self._evaluate(state)
                if decision == "SUFFICIENT":
                    state.done = True
                    self._logger.info(
                        "Evidence sufficient after %d iteration(s).", state.iterations
                    )
                else:
                    new_query = decision.removeprefix("NEED_MORE:").strip()
                    if not new_query or new_query in state.searches_run:
                        state.done = True   # no new angle found
                    else:
                        self._logger.info(
                            "Gathering additional evidence: %s", new_query[:60]
                        )
                        self._gather_for_query(new_query, state)

            # ── Step 4: Synthesise ────────────────────────────────────────────
            report, tool_calls = self._synthesise(state)

            elapsed = time.monotonic() - t0
            self._logger.info(
                "Deep research complete: %d sources, %d iterations, %.1fs",
                len(state.evidence), state.iterations, elapsed,
            )

            return AgentResponse(
                output=report,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=[e.source for e in state.evidence],
                metadata={
                    "iterations":    state.iterations,
                    "sources_found": len(state.evidence),
                    "elapsed_s":     round(elapsed, 1),
                    "model":         settings.ollama_reasoning_model,
                },
            )

        except Exception as exc:
            self._logger.error("DeepResearchAgent failed: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"Research encountered an error: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Step implementations ──────────────────────────────────────────────────

    def _plan(self, query: str) -> ResearchPlan:
        """Use the reasoning model to decompose the query into a research plan."""
        import json

        prompt = _PLAN_PROMPT.format(query=query)
        result = self._reasoning_llm.invoke(prompt)
        raw    = str(result.content).strip()

        # Strip common reasoning-model preambles (<think>...</think>, etc.)
        raw = self._strip_thinking(raw)

        # Parse JSON
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Graceful degradation: extract any JSON block from the output
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                obj = json.loads(match.group())
            else:
                # Final fallback: build a minimal plan from the raw text
                logger.warning("Could not parse research plan JSON; using fallback plan.")
                obj = {
                    "topic": query[:60],
                    "sub_questions": [query],
                    "search_queries": [query],
                }

        return ResearchPlan(
            topic=obj.get("topic", query[:60]),
            sub_questions=obj.get("sub_questions", [query]),
            search_queries=obj.get("search_queries", [query]),
        )

    def _gather_for_query(self, query: str, state: ResearchState) -> None:
        """
        Run web search + Wikipedia for a single query and add results to state.

        Tries DuckDuckGo first, then Wikipedia.  Any tool failure is caught and
        logged — research continues with whatever evidence was gathered.
        """
        if query in state.searches_run:
            return
        state.searches_run.append(query)

        tools = self._get_search_tools()
        tool_budget = self._max_sources - len(state.evidence)
        if tool_budget <= 0:
            return

        # Web search
        if "web_search" in tools:
            try:
                raw = tools["web_search"]._run(query=query)
                if raw and len(raw.strip()) > 50:
                    items = self._parse_search_results(raw, query)
                    for item in items[:min(3, tool_budget)]:
                        state.evidence.append(item)
                        if len(state.evidence) >= self._max_sources:
                            return
            except Exception as exc:
                logger.debug("web_search failed for '%s': %s", query[:40], exc)

        # Wikipedia
        if "wikipedia_lookup" in tools and len(state.evidence) < self._max_sources:
            try:
                raw = tools["wikipedia_lookup"]._run(query=query)
                if raw and len(raw.strip()) > 100:
                    # Truncate to 1500 chars — Wikipedia entries can be huge
                    content = raw[:1500].rstrip() + ("…" if len(raw) > 1500 else "")
                    state.evidence.append(EvidenceItem(
                        source=f"Wikipedia: {query}",
                        content=content,
                        query=query,
                    ))
            except Exception as exc:
                logger.debug("wikipedia_lookup failed for '%s': %s", query[:40], exc)

    def _evaluate(self, state: ResearchState) -> str:
        """
        Ask the reasoning model whether evidence is sufficient or more is needed.

        Returns
        -------
        str
            ``"SUFFICIENT"`` or ``"NEED_MORE: <new query>"``.
        """
        if not state.plan:
            return "SUFFICIENT"

        evidence_summary = self._format_evidence_summary(state.evidence, max_chars=2000)
        answered   = {e.query for e in state.evidence}
        unanswered = [sq for sq in state.plan.sub_questions
                      if not any(sq.lower()[:20] in q.lower() for q in answered)]

        prompt = _EVALUATE_PROMPT.format(
            query=state.query,
            n_items=len(state.evidence),
            n_sources=len({e.source for e in state.evidence}),
            evidence_summary=evidence_summary,
            unanswered="\n".join(f"- {sq}" for sq in unanswered) or "(all covered)",
        )

        try:
            result  = self._reasoning_llm.invoke(prompt)
            decision = self._strip_thinking(str(result.content)).strip()
            # Normalise
            if decision.upper().startswith("SUFFICIENT"):
                return "SUFFICIENT"
            if decision.upper().startswith("NEED_MORE"):
                return decision
            # Fallback: if we can't parse, declare sufficient to avoid infinite loop
            return "SUFFICIENT"
        except Exception as exc:
            logger.debug("Evaluation LLM call failed: %s", exc)
            return "SUFFICIENT"

    def _synthesise(
        self, state: ResearchState
    ) -> tuple[str, list[tuple[str, str, str]]]:
        """
        Use the reasoning model to write the final structured report.

        Returns
        -------
        tuple[str, list]
            (report_markdown, tool_calls_list)
        """
        plan_str  = str(state.plan) if state.plan else state.query
        evidence  = self._format_evidence_for_synthesis(state.evidence)
        tool_calls: list[tuple[str, str, str]] = [
            ("web_search", e.query, e.content[:200]) for e in state.evidence
        ]

        prompt = _SYNTHESISE_PROMPT.format(
            query=state.query,
            plan=plan_str,
            n_items=len(state.evidence),
            evidence=evidence,
        )

        try:
            result = self._reasoning_llm.invoke(prompt)
            report = self._strip_thinking(str(result.content)).strip()
        except Exception as exc:
            logger.error("Synthesis LLM call failed: %s", exc)
            # Emergency fallback: compile evidence into a basic report
            report = self._fallback_report(state)

        return report, tool_calls

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _format_evidence_summary(self, items: list[EvidenceItem],
                                  max_chars: int = 2000) -> str:
        """Short summary of evidence for the evaluation prompt."""
        parts: list[str] = []
        used = 0
        for i, item in enumerate(items, 1):
            snippet  = item.content[:200].replace("\n", " ")
            line     = f"[{i}] {item.source}\n    {snippet}"
            if used + len(line) > max_chars:
                break
            parts.append(line)
            used += len(line) + 1
        return "\n".join(parts) if parts else "(no evidence collected)"

    def _format_evidence_for_synthesis(self, items: list[EvidenceItem]) -> str:
        """Full evidence block for the synthesis prompt, respecting char budget."""
        parts: list[str] = []
        used  = 0
        per_item_budget = max(400, self._chunk_budget // max(len(items), 1))

        for i, item in enumerate(items, 1):
            text    = item.content[:per_item_budget]
            header  = f"[{i}] SOURCE: {item.source}\n"
            block   = header + text
            if used + len(block) > self._chunk_budget:
                # Add a truncated stub so the model knows there's more
                stub = f"[{i}] SOURCE: {item.source}\n(content truncated — budget exhausted)"
                parts.append(stub)
                break
            parts.append(block)
            used += len(block) + 2

        return "\n\n---\n\n".join(parts) if parts else "(no evidence collected)"

    def _parse_search_results(
        self, raw: str, query: str
    ) -> list[EvidenceItem]:
        """
        Parse raw DuckDuckGo search output into EvidenceItem objects.
        DuckDuckGoSearchTool returns a text blob; we split on double-newlines.
        """
        items: list[EvidenceItem] = []
        # Each result is separated by blank lines; extract URL + snippet
        import re
        # Try to find URL lines
        url_pattern = re.compile(r"https?://\S+")
        chunks  = [c.strip() for c in raw.split("\n\n") if c.strip()]
        for chunk in chunks[:4]:
            urls = url_pattern.findall(chunk)
            source = urls[0] if urls else f"web_search:{query[:30]}"
            # Use the full chunk as content (capped to 600 chars)
            content = chunk[:600]
            if len(content) > 80:   # skip tiny fragments
                items.append(EvidenceItem(source=source, content=content, query=query))
        return items

    def _fallback_report(self, state: ResearchState) -> str:
        """Emergency plain-text report when the synthesis LLM call fails."""
        lines = [
            f"# Research Report: {state.query}",
            "",
            "## Evidence Collected",
            "",
        ]
        for i, ev in enumerate(state.evidence, 1):
            lines += [f"### [{i}] {ev.source}", ev.content[:500], ""]
        lines += [
            "## References",
            *[f"{i}. {ev.source}" for i, ev in enumerate(state.evidence, 1)],
        ]
        return "\n".join(lines)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Remove <think>…</think> blocks that reasoning models (e.g. deepseek-r1)
        emit before their final answer.  Also handles ``<|think|>`` variants.
        """
        import re
        # Remove <think>...</think> and <|think|>...</|think|>
        text = re.sub(r"<\|?think\|?>.*?</\|?think\|?>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_search_tools(self) -> dict[str, BaseTool]:
        if self._search_tools is None:
            from tools.search_tools import get_search_tools
            self._search_tools = {t.name: t for t in get_search_tools()}
        return self._search_tools

    @staticmethod
    def _load_reasoning_llm() -> BaseChatModel:
        from core.llm_manager import get_reasoning_llm
        return get_reasoning_llm()
