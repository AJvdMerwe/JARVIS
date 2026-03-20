"""
agents/orchestrator.py
───────────────────────
LangGraph-based multi-agent orchestrator.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │                    Orchestrator                      │
  │                                                      │
  │  User Input ──► Intent Router ──► Agent Selector     │
  │                      │                               │
  │           ┌──────────┼──────────┐                   │
  │           ▼          ▼          ▼          ▼         │
  │       CodeAgent  NewsAgent  SearchAgent  DocAgent    │
  │           │          │          │          │         │
  │           └──────────┴──────────┴──────────┘         │
  │                       │                              │
  │                 AgentResponse                        │
  │                       │                              │
  │             Response Synthesiser                     │
  └──────────────────────────────────────────────────────┘

The router uses keyword heuristics first (fast, no LLM call), then
falls back to an LLM classification call for ambiguous inputs.

State is a TypedDict compatible with LangGraph's StateGraph.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional, TypedDict

from .base_agent import AgentResponse, BaseAgent
from .code_agent import CodeAgent
from .document_agent import DocumentAgent
from .news_agent import NewsAgent
from .search_agent import SearchAgent
from core.memory import PersistentMemory
from core.llm_manager import get_llm
from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Intent enum
# ─────────────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    CODE     = "code"
    NEWS     = "news"
    SEARCH   = "search"
    DOCUMENT = "document"
    UNKNOWN  = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraph state
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict, total=False):
    query:    str
    intent:   str
    response: AgentResponse
    history:  list[dict[str, str]]
    metadata: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
#  Keyword-based fast router
# ─────────────────────────────────────────────────────────────────────────────

_CODE_KEYWORDS = re.compile(
    r"\b(code|write|program|function|class|algorithm|debug|bug|error|"
    r"script|python|javascript|typescript|java|c\+\+|rust|sql|html|css|"
    r"implement|refactor|review|execute|run|compile|test|unit test|"
    r"data structure|recursion|regex|api|endpoint)\b",
    re.IGNORECASE,
)

_NEWS_KEYWORDS = re.compile(
    r"\b(news|headline|article|story|report|latest|today|breaking|"
    r"current events|what'?s happening|briefing|rss|feed|journalist|"
    r"coverage|press|media|broadcast)\b",
    re.IGNORECASE,
)

_DOCUMENT_KEYWORDS = re.compile(
    r"\b(document|pdf|docx|excel|spreadsheet|ppt|powerpoint|file|"
    r"upload|ingest|knowledge base|my docs?|this report|the report|"
    r"the paper|the document|the file|from the|according to|"
    r"what does the|summarise the|summarize the|in the doc)\b",
    re.IGNORECASE,
)


def _keyword_route(query: str) -> Optional[Intent]:
    """Fast keyword-based intent detection. Returns None if ambiguous."""
    code_score = len(_CODE_KEYWORDS.findall(query))
    news_score = len(_NEWS_KEYWORDS.findall(query))
    doc_score  = len(_DOCUMENT_KEYWORDS.findall(query))

    scores = {Intent.CODE: code_score, Intent.NEWS: news_score, Intent.DOCUMENT: doc_score}
    best_intent, best_score = max(scores.items(), key=lambda x: x[1])

    # Require a clear winner (score ≥ 1 and at least 1 more than second-best)
    second_best = sorted(scores.values(), reverse=True)[1]
    if best_score >= 1 and best_score > second_best:
        return best_intent

    # Default non-matched queries to search
    if best_score == 0 and news_score == 0 and doc_score == 0:
        return Intent.SEARCH

    return None


def _llm_route(query: str) -> Intent:
    """LLM-based intent classification for ambiguous queries."""
    llm = get_llm()
    prompt = (
        "Classify the following user query into exactly one category.\n\n"
        "Categories:\n"
        "- code: writing, debugging, reviewing, or executing code / programs\n"
        "- news: fetching or summarising news articles and current events\n"
        "- document: questions about uploaded files or the knowledge base\n"
        "- search: general knowledge questions and web research\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY the category name (lowercase, no punctuation)."
    )
    try:
        result = llm.invoke(prompt)
        label = str(result.content).strip().lower()
        return Intent(label) if label in Intent._value2member_map_ else Intent.SEARCH
    except Exception:
        return Intent.SEARCH


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Routes user queries to the most appropriate specialist agent.

    Usage::

        orch = Orchestrator()
        response = orch.run("Write a bubble sort in Python")
        print(response.full_response())
    """

    def __init__(self, session_id: str = "default") -> None:
        self._session_id = session_id
        # Shared persistent memory across all agents in this session
        self._memory = PersistentMemory(session_id=session_id)

        # Instantiate agents (lazy: tools only load when agent is first used)
        self._agents: dict[Intent, BaseAgent] = {
            Intent.CODE:     CodeAgent(memory=self._memory),
            Intent.NEWS:     NewsAgent(memory=self._memory),
            Intent.SEARCH:   SearchAgent(memory=self._memory),
            Intent.DOCUMENT: DocumentAgent(memory=self._memory),
        }

        logger.info("Orchestrator initialised for session '%s'.", session_id)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Route a query to the best agent and return its response.

        Args:
            query:      The user's input text.
            intent:     Optional forced Intent to skip routing.
            **kwargs:   Forwarded to the selected agent's run().

        Returns:
            AgentResponse from the selected agent.
        """
        forced = kwargs.pop("intent", None)

        if forced:
            intent = Intent(forced) if isinstance(forced, str) else forced
        else:
            intent = self._route(query)

        logger.info("Routing '%s…' → %s", query[:60], intent.value)

        agent = self._agents[intent]
        response = agent.run(query, **kwargs)

        # Save to shared memory
        self._memory.save_context(query, response.output)

        return response

    def route_only(self, query: str) -> Intent:
        """Return the detected intent without running the agent."""
        return self._route(query)

    def ingest_document(self, file_path: str) -> str:
        """Convenience: ingest a document directly without a conversational query."""
        doc_agent: DocumentAgent = self._agents[Intent.DOCUMENT]  # type: ignore[assignment]
        return doc_agent.ingest(file_path)

    def list_documents(self) -> list[dict]:
        """Return all documents in the knowledge base."""
        doc_agent: DocumentAgent = self._agents[Intent.DOCUMENT]  # type: ignore[assignment]
        return doc_agent.list_documents()

    def clear_memory(self) -> None:
        """Clear all conversation history for this session."""
        self._memory.clear()
        logger.info("Cleared memory for session '%s'.", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def memory(self) -> PersistentMemory:
        return self._memory

    def get_agent(self, intent: Intent) -> BaseAgent:
        return self._agents[intent]

    # ── Internal routing ─────────────────────────────────────────────────────

    def _route(self, query: str) -> Intent:
        """Two-stage routing: fast keyword match, then LLM fallback."""
        intent = _keyword_route(query)
        if intent is not None:
            return intent
        return _llm_route(query)

    # ── LangGraph integration ─────────────────────────────────────────────────

    def build_graph(self):
        """
        Build and return a compiled LangGraph StateGraph.
        Useful for visualisation, async execution, or custom pipelines.
        """
        try:
            from langgraph.graph import StateGraph, END  # type: ignore
        except ImportError as exc:
            raise ImportError("langgraph is not installed. Run: pip install langgraph") from exc

        graph = StateGraph(OrchestratorState)

        # ── Node definitions ─────────────────────────────────────────────────

        def route_node(state: OrchestratorState) -> OrchestratorState:
            intent = self._route(state["query"])
            return {**state, "intent": intent.value}

        def code_node(state: OrchestratorState) -> OrchestratorState:
            resp = self._agents[Intent.CODE].run(state["query"])
            return {**state, "response": resp}

        def news_node(state: OrchestratorState) -> OrchestratorState:
            resp = self._agents[Intent.NEWS].run(state["query"])
            return {**state, "response": resp}

        def search_node(state: OrchestratorState) -> OrchestratorState:
            resp = self._agents[Intent.SEARCH].run(state["query"])
            return {**state, "response": resp}

        def document_node(state: OrchestratorState) -> OrchestratorState:
            resp = self._agents[Intent.DOCUMENT].run(state["query"])
            return {**state, "response": resp}

        def dispatch(state: OrchestratorState) -> str:
            return state.get("intent", Intent.SEARCH.value)

        # ── Wire the graph ────────────────────────────────────────────────────

        graph.add_node("router",   route_node)
        graph.add_node("code",     code_node)
        graph.add_node("news",     news_node)
        graph.add_node("search",   search_node)
        graph.add_node("document", document_node)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            dispatch,
            {
                Intent.CODE.value:     "code",
                Intent.NEWS.value:     "news",
                Intent.SEARCH.value:   "search",
                Intent.DOCUMENT.value: "document",
                Intent.UNKNOWN.value:  "search",
            },
        )
        for node in ("code", "news", "search", "document"):
            graph.add_edge(node, END)

        return graph.compile()
