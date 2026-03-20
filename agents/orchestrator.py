"""
agents/orchestrator.py
───────────────────────
LangGraph-based multi-agent orchestrator — the single entry point for all
user interactions.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                           Orchestrator                                  │
  │                                                                         │
  │  User Input                                                             │
  │      │                                                                  │
  │      ▼                                                                  │
  │  Intent Router  ──► keyword heuristics (fast, no LLM)                  │
  │      │          └──► LLM classifier    (fallback for ambiguous queries) │
  │      │                                                                  │
  │      ▼                                                                  │
  │  ┌──────┬──────┬──────┬──────────┬──────────┐                          │
  │  │ Chat │ Code │ News │  Search  │ Document │                          │
  │  └──┬───┴──┬───┴──┬───┴──────┬───┴──────┬───┘                          │
  │     └──────┴──────┴──────────┴──────────┘                              │
  │                         │                                               │
  │                   AgentResponse                                         │
  │                         │                                               │
  │  ┌──────────────────────▼──────────────────────────────────────────┐   │
  │  │                   Post-processing                               │   │
  │  │  PersistentMemory save  •  EpisodicMemory extract               │   │
  │  │  ConversationSummariser •  Tracer record                        │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────────┘

Routing (two-stage):
  Stage 1 — _keyword_route(): regex patterns, < 1 ms, no LLM call.
  Stage 2 — _llm_route():     LLM classification, only for ambiguous queries.

  Intent priority: CODE > NEWS > DOCUMENT > CHAT > SEARCH > (LLM fallback)

Extensibility:
  • Subclass BaseAgent (~50 lines) and add_agent(intent, agent).
  • Or drop a plugin in plugins/*.py → register_agents().
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional, TypedDict

from .base_agent import AgentResponse, BaseAgent
from .chat_agent import ChatAgent
from .code_agent import CodeAgent
from .document_agent import DocumentAgent
from .news_agent import NewsAgent
from .search_agent import SearchAgent
from core.memory import PersistentMemory
from core.llm_manager import get_llm
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
#  Intent enum
# =============================================================================

class Intent(str, Enum):
    CHAT     = "chat"      # General conversation, creative, explanatory
    CODE     = "code"      # Write / debug / execute code
    NEWS     = "news"      # Fetch / summarise news
    SEARCH   = "search"    # Web search, Wikipedia, calculations
    DOCUMENT = "document"  # KB / document Q&A
    UNKNOWN  = "unknown"   # Unclassifiable — falls back to CHAT


# =============================================================================
#  LangGraph state
# =============================================================================

class OrchestratorState(TypedDict, total=False):
    """
    Shared state dict threaded through every node in the LangGraph pipeline.

    Fields
    ------
    query:    str
        Original user query string.
    intent:   str
        Detected or forced intent value (Intent enum .value).
    response: AgentResponse
        Produced by the chosen agent node.
    history:  list[dict[str, str]]
        Snapshot of recent conversation turns (for logging / debugging).
    metadata: dict[str, Any]
        Arbitrary extra context — user_id, doc_title, request_id, etc.
    error:    str
        Set if a node raises an exception; empty string otherwise.
    """
    query:    str
    intent:   str
    response: AgentResponse
    history:  list[dict[str, str]]
    metadata: dict[str, Any]
    error:    str


# =============================================================================
#  Keyword-based fast router — Stage 1
# =============================================================================

# Chat: greetings, creative writing, conversational openers, explanations.
# Creative "write X" patterns are explicit here so they score above CODE's
# plain "write" match when the noun is non-code (poem, story, essay …).
_CHAT_KEYWORDS = re.compile(
    r"\b(hi|hello|hey|how are you|what do you think|tell me about yourself|"
    r"who are you|let'?s talk|can you help|i need help|chat|convers|talk to me|"
    r"what is your|do you know|explain|describe|what does|how does|why does|"
    r"compare|difference between|pros and cons|give me an example|"
    r"brainstorm|write (a|an|me) (poem|story|essay|letter|haiku|song|joke|"
    r"paragraph|blog|speech|narrative|tale|summary)|"
    r"tell me (a|an)|create (a|an) (poem|story|essay)|draft (a|an)|"
    r"poem|haiku|creative writing|opinion|advice|recommend|suggest|"
    r"what would you|translate|rewrite|improve|paraphrase|"
    r"summarise|summarize|like i'?m (five|5)|eli5)\b",
    re.IGNORECASE,
)

# Code: programming-specific nouns.
# "write" only matches when followed by a code noun (function, class, script…)
# to avoid routing "write a poem" or "write a story" to CODE.
_CODE_KEYWORDS = re.compile(
    r"\b(code|program|function|class|algorithm|debug|bug|error|"
    r"script|python|javascript|typescript|java|c\+\+|rust|sql|html|css|"
    r"implement|refactor|execute|compile|unit test|"
    r"data structure|recursion|regex|api|endpoint|"
    r"write (a |an )?(function|class|script|program|algorithm|module|"
    r"method|loop|test|decorator|interface|enum|query|snippet|macro)|"
    r"review (this |the )?(code|function|class|script|program))\b",
    re.IGNORECASE,
)

# News: clearly news-specific terms.
# "story", "article", "today", "report" deliberately omitted — too ambiguous.
_NEWS_KEYWORDS = re.compile(
    r"\b(news|headlines?|breaking|current events|what'?s happening|briefing|"
    r"rss|feed|journalist|coverage|press|broadcast|"
    r"latest (news|headlines?|stories|developments|updates)|"
    r"news (about|on|from)|top (stories|headlines?)|daily news)\b",
    re.IGNORECASE,
)

# Document: knowledge-base and file-specific terms.
_DOCUMENT_KEYWORDS = re.compile(
    r"\b(document|pdf|docx|excel|spreadsheet|ppt|powerpoint|file|"
    r"upload|ingest|knowledge base|my docs?|this report|the report|"
    r"the paper|the document|the file|from the|according to|"
    r"what does the|summarise the|summarize the|in the doc)\b",
    re.IGNORECASE,
)


def _keyword_route(query: str) -> Optional[Intent]:
    """
    Stage-1 intent detector using pre-compiled regex patterns.

    Returns an Intent or None (None triggers Stage-2 LLM routing).
    """
    code_score = len(_CODE_KEYWORDS.findall(query))
    news_score = len(_NEWS_KEYWORDS.findall(query))
    doc_score  = len(_DOCUMENT_KEYWORDS.findall(query))
    chat_score = len(_CHAT_KEYWORDS.findall(query))

    specialist_scores: dict[Intent, int] = {
        Intent.CODE:     code_score,
        Intent.NEWS:     news_score,
        Intent.DOCUMENT: doc_score,
    }
    best_specialist, best_score = max(specialist_scores.items(), key=lambda x: x[1])
    second_best = sorted(specialist_scores.values(), reverse=True)[1]

    # Specialist wins only if it has a clear lead AND beats CHAT signals.
    if best_score >= 1 and best_score > second_best and chat_score < best_score:
        return best_specialist

    # CHAT wins when it has signals matching or exceeding all specialists.
    if chat_score >= 1 and (best_score == 0 or chat_score >= best_score):
        return Intent.CHAT

    # Specialist still wins with a strong numerical lead over CHAT.
    if best_score >= 2 and best_score > chat_score:
        return best_specialist

    # SEARCH for purely factual queries with no other signals.
    if best_score == 0 and chat_score == 0:
        return Intent.SEARCH

    return None   # ambiguous — delegate to LLM


# =============================================================================
#  LLM-based router — Stage 2
# =============================================================================

def _llm_route(query: str) -> Intent:
    """
    Stage-2 classifier: calls the LLM for ambiguous queries.
    Falls back to Intent.CHAT on any failure.
    """
    llm = get_llm()
    prompt = (
        "Classify the following user query into exactly one category.\n\n"
        "Categories:\n"
        "- chat: general conversation, creative writing, brainstorming, opinions,\n"
        "  advice, explanations, follow-up questions, or anything conversational\n"
        "- code: writing, debugging, reviewing, or executing code / programs\n"
        "- news: fetching or summarising news articles and current events\n"
        "- document: questions about uploaded files or the knowledge base\n"
        "- search: factual questions that need real-time web lookup or Wikipedia\n\n"
        f"Query: {query}\n\n"
        "Respond with ONLY the category name (lowercase, no punctuation)."
    )
    try:
        result = llm.invoke(prompt)
        label  = str(result.content).strip().lower()
        return Intent(label) if label in Intent._value2member_map_ else Intent.CHAT
    except Exception:
        return Intent.CHAT


# =============================================================================
#  Orchestrator
# =============================================================================

class Orchestrator:
    """
    Routes user queries to the most appropriate specialist agent and
    coordinates all post-processing (memory, tracing, episodic facts,
    context compression, plugins).

    Parameters
    ----------
    session_id : str
        Identifier for persistent memory, tracing, and episodic recall.
    enable_episodic : bool
        Store and recall facts from past sessions (EpisodicMemory).
        Off by default — requires ChromaDB + embedding model.
    enable_summariser : bool
        Compress old conversation turns automatically (ConversationSummariser).
        Off by default — suitable for long-running sessions.
    enable_plugins : bool
        Auto-discover and load agents/tools from plugins/ at startup.
    enable_scheduler : bool
        Start the background TaskScheduler (cache purge, trace rotation…).
    summarise_after : int
        Total message count that triggers history compression.
    recent_k : int
        Number of most-recent turns to keep verbatim after compression.
    """

    def __init__(
        self,
        session_id:        str  = "default",
        enable_episodic:   bool = False,
        enable_summariser: bool = False,
        enable_plugins:    bool = True,
        enable_scheduler:  bool = False,
        summarise_after:   int  = 40,
        recent_k:          int  = 10,
    ) -> None:
        self._session_id = session_id

        # Persistent conversation memory (shared by all agents).
        self._memory = PersistentMemory(session_id=session_id)

        # Specialist agents — all share the same memory instance.
        self._agents: dict[Intent, BaseAgent] = {
            Intent.CHAT:     ChatAgent(memory=self._memory),
            Intent.CODE:     CodeAgent(memory=self._memory),
            Intent.NEWS:     NewsAgent(memory=self._memory),
            Intent.SEARCH:   SearchAgent(memory=self._memory),
            Intent.DOCUMENT: DocumentAgent(memory=self._memory),
        }

        # Optional: conversation history summariser.
        self._summariser = None
        if enable_summariser:
            try:
                from core.summariser import ConversationSummariser
                self._summariser = ConversationSummariser(
                    summarise_after=summarise_after,
                    recent_k=recent_k,
                )
                logger.info(
                    "ConversationSummariser enabled (after=%d, k=%d).",
                    summarise_after, recent_k,
                )
            except Exception as exc:
                logger.warning("ConversationSummariser init failed: %s", exc)

        # Optional: long-term episodic memory.
        self._episodic = None
        if enable_episodic:
            try:
                from core.long_term_memory import EpisodicMemory
                self._episodic = EpisodicMemory()
                logger.info("EpisodicMemory enabled.")
            except Exception as exc:
                logger.warning("EpisodicMemory init failed: %s", exc)

        # Optional: plugin discovery.
        if enable_plugins:
            self._load_plugins()

        # Optional: background scheduler.
        if enable_scheduler:
            self.start_scheduler()

        logger.info(
            "Orchestrator ready | session='%s' | agents=%d | "
            "episodic=%s | summariser=%s",
            session_id,
            len(self._agents),
            bool(self._episodic),
            bool(self._summariser),
        )

    # -------------------------------------------------------------------------
    #  Core run
    # -------------------------------------------------------------------------

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Route a query to the best agent and return its response.

        Cross-cutting concerns handled automatically:
          * Intent routing (keyword → LLM fallback)
          * Episodic context injection when available
          * Agent execution
          * Persistent memory save
          * Episodic fact extraction
          * History compression trigger
          * Request tracing

        Parameters
        ----------
        query : str
            The user input text.
        intent : str | Intent, optional
            Force a specific intent, bypassing routing entirely.
        user_id : str, optional
            Load UserPreferences for this user and pass to the agent.
        doc_title : str, optional
            Restrict DocumentAgent search to a specific document.
        **kwargs
            All other kwargs are forwarded to the selected agent's run().

        Returns
        -------
        AgentResponse
            Contains .output, .tool_calls, .references, .error.
        """
        forced  = kwargs.pop("intent",  None)
        user_id = kwargs.get("user_id", None)

        # Stage 1 / 2 routing.
        if forced:
            intent = Intent(forced) if isinstance(forced, str) else forced
        else:
            intent = self._route(query)

        logger.info("session='%s' intent=%s query='%s…'",
                    self._session_id, intent.value, query[:60])

        # Open a trace span (best-effort — never blocks the call).
        trace_obj = self._open_trace(query)

        # Inject episodic context as a prompt prefix (if available).
        if getattr(self, "_episodic", None):
            self._inject_episodic_context(query, kwargs)

        # Apply user preferences (if user_id provided).
        if user_id:
            kwargs.setdefault("user_id", user_id)

        # Execute the agent.
        agent    = self._agents.get(intent) or self._agents[Intent.CHAT]
        response = agent.run(query, **kwargs)

        # Close the trace span.
        self._close_trace(trace_obj, response)

        # Persist conversation turn.
        self._memory.save_context(query, response.output)

        # Extract episodic facts from this exchange (non-blocking, swallows errors).
        if getattr(self, "_episodic", None) and response.output and not response.error:
            self._store_episodic_facts(query, response.output)

        # Maybe compress conversation history.
        if getattr(self, "_summariser", None):
            try:
                self._summariser.maybe_summarise(self._memory)
            except Exception as exc:
                logger.debug("Summariser error: %s", exc)

        return response

    # -------------------------------------------------------------------------
    #  Routing helpers
    # -------------------------------------------------------------------------

    def route_only(self, query: str) -> Intent:
        """
        Classify a query without running an agent.
        Useful for UI previews (e.g. "code agent will handle this").
        """
        return self._route(query)

    # -------------------------------------------------------------------------
    #  Document convenience helpers
    # -------------------------------------------------------------------------

    def ingest_document(self, file_path: str) -> str:
        """
        Ingest a document directly into the knowledge base.

        Parameters
        ----------
        file_path : str
            Path to PDF, DOCX, XLSX, or PPTX.

        Returns
        -------
        str
            Human-readable status message.
        """
        doc_agent: DocumentAgent = self._agents[Intent.DOCUMENT]  # type: ignore[assignment]
        return doc_agent.ingest(file_path)

    def list_documents(self) -> list[dict]:
        """Return all documents currently in the knowledge base."""
        doc_agent: DocumentAgent = self._agents[Intent.DOCUMENT]  # type: ignore[assignment]
        return doc_agent.list_documents()

    # -------------------------------------------------------------------------
    #  Memory
    # -------------------------------------------------------------------------

    def clear_memory(self) -> None:
        """Clear all conversation history for this session."""
        self._memory.clear()
        logger.info("Cleared memory for session '%s'.", self._session_id)

    # -------------------------------------------------------------------------
    #  Agent registry
    # -------------------------------------------------------------------------

    def get_agent(self, intent: Intent) -> BaseAgent:
        """
        Retrieve an agent by intent.

        Raises
        ------
        KeyError
            If no agent is registered for the given intent.
        """
        return self._agents[intent]

    def add_agent(self, intent: Intent, agent: BaseAgent) -> None:
        """
        Register or replace an agent at runtime.

        Used by the plugin system and for testing / extension.
        """
        self._agents[intent] = agent
        logger.info("Agent '%s' registered for intent '%s'.",
                    agent.name, intent.value)

    # -------------------------------------------------------------------------
    #  Scheduler
    # -------------------------------------------------------------------------

    def start_scheduler(self) -> None:
        """Start the background TaskScheduler if it is not already running."""
        try:
            from core.scheduler import get_scheduler
            sched = get_scheduler()
            if not sched.is_running:
                sched.start()
                logger.info("Background scheduler started.")
        except Exception as exc:
            logger.warning("Scheduler start failed: %s", exc)

    def stop_scheduler(self) -> None:
        """Gracefully stop the background scheduler."""
        try:
            from core.scheduler import get_scheduler
            get_scheduler().stop()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    #  Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Session identifier passed at construction."""
        return self._session_id

    @property
    def memory(self) -> PersistentMemory:
        """Shared persistent memory for this session."""
        return self._memory

    @property
    def intents(self) -> list[Intent]:
        """All intents that currently have a registered agent."""
        return list(self._agents.keys())

    # -------------------------------------------------------------------------
    #  Async variants
    # -------------------------------------------------------------------------

    async def run_async(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Non-blocking async variant of run().

        Wraps the synchronous agent call in asyncio.to_thread() so it
        does not block the FastAPI event loop.
        """
        import asyncio
        return await asyncio.to_thread(self.run, query, **kwargs)

    async def fanout(
        self,
        query:   str,
        intents: list[Intent],
        **kwargs: Any,
    ) -> list[AgentResponse]:
        """
        Run multiple agents in parallel for the same query.

        Parameters
        ----------
        query : str
            Query sent to all agents simultaneously.
        intents : list[Intent]
            Which agents to invoke.
        **kwargs
            Forwarded to each agent's run().

        Returns
        -------
        list[AgentResponse]
            One response per intent, in the same order.
        """
        from core.async_runner import AsyncAgentRunner
        runner  = AsyncAgentRunner()
        agents  = [self._agents[i] for i in intents if i in self._agents]
        result  = await runner.fanout(query, agents, **kwargs)
        return result.responses

    # -------------------------------------------------------------------------
    #  LangGraph pipeline
    # -------------------------------------------------------------------------

    def build_graph(self):
        """
        Build and return a compiled LangGraph StateGraph.

        Node layout:
          router → conditional → [chat | code | news | search | document] → END

        The ``dispatch`` function reads ``state["intent"]`` after the
        router node writes it, routing execution to the correct agent node.

        Returns
        -------
        CompiledGraph
            Ready for .invoke({"query": "…"}) or async .astream().

        Raises
        ------
        ImportError
            If langgraph is not installed.
        """
        try:
            from langgraph.graph import StateGraph, END  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langgraph is not installed. Run: pip install langgraph"
            ) from exc

        graph = StateGraph(OrchestratorState)

        # Router node — detects intent and writes to state["intent"].
        def route_node(state: OrchestratorState) -> OrchestratorState:
            intent = self._route(state["query"])
            return {**state, "intent": intent.value}

        # Agent node factory — one closure per intent.
        def _agent_node(target: Intent):
            def _node(state: OrchestratorState) -> OrchestratorState:
                try:
                    resp = self._agents[target].run(state["query"])
                    return {**state, "response": resp}
                except Exception as exc:
                    err_resp = AgentResponse(
                        output=f"Agent error: {exc}",
                        agent_name=target.value,
                        error=str(exc),
                    )
                    return {**state, "response": err_resp, "error": str(exc)}
            _node.__name__ = f"{target.value}_node"
            return _node

        # Dispatch function — reads state["intent"] → returns node name.
        def dispatch(state: OrchestratorState) -> str:
            return state.get("intent", Intent.CHAT.value)

        # Register nodes.
        graph.add_node("router",   route_node)
        graph.add_node("chat",     _agent_node(Intent.CHAT))
        graph.add_node("code",     _agent_node(Intent.CODE))
        graph.add_node("news",     _agent_node(Intent.NEWS))
        graph.add_node("search",   _agent_node(Intent.SEARCH))
        graph.add_node("document", _agent_node(Intent.DOCUMENT))

        # Entry + conditional edges.
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            dispatch,
            {
                Intent.CHAT.value:     "chat",
                Intent.CODE.value:     "code",
                Intent.NEWS.value:     "news",
                Intent.SEARCH.value:   "search",
                Intent.DOCUMENT.value: "document",
                Intent.UNKNOWN.value:  "chat",
            },
        )
        for node_name in ("chat", "code", "news", "search", "document"):
            graph.add_edge(node_name, END)

        return graph.compile()

    # -------------------------------------------------------------------------
    #  Internal helpers
    # -------------------------------------------------------------------------

    def _route(self, query: str) -> Intent:
        intent = _keyword_route(query)
        return intent if intent is not None else _llm_route(query)

    def _load_plugins(self) -> None:
        """Discover and inject plugins from plugins/ and entry-points."""
        try:
            from plugins import load_all_plugins
            agent_classes, _ = load_all_plugins()
            for name, cls in agent_classes.items():
                try:
                    self._agents[name] = cls(memory=self._memory)  # type: ignore[index]
                    logger.info("Plugin agent '%s' registered.", name)
                except Exception as exc:
                    logger.warning("Plugin '%s' failed: %s", name, exc)
        except Exception as exc:
            logger.debug("Plugin loading skipped: %s", exc)

    def _open_trace(self, query: str):
        """Open a trace context manager; returns None on failure."""
        try:
            from core.tracing import get_tracer
            # Return the internal Trace object directly (not the context manager)
            # to avoid __enter__/__exit__ complexity in the run() path.
            import time as _time
            from core.tracing.tracer import Trace
            tracer = get_tracer()
            trace = Trace(
                trace_id=tracer._next_id(),
                session_id=self._session_id,
                query=query,
            )
            return (tracer, trace)
        except Exception:
            return None

    def _close_trace(self, trace_ctx, response: AgentResponse) -> None:
        """Finalise and record the trace."""
        if trace_ctx is None:
            return
        try:
            tracer, trace = trace_ctx
            trace.set_outcome(
                response.output,
                error=response.error,
                agent_name=response.agent_name,
            )
            tracer.store.record(trace)
        except Exception as exc:
            logger.debug("Trace record failed: %s", exc)

    def _inject_episodic_context(self, query: str, kwargs: dict) -> None:
        """Retrieve relevant past facts and add them to kwargs."""
        try:
            context = self._episodic.recall_as_context(query)  # type: ignore[union-attr]
            if context:
                kwargs.setdefault("episodic_context", context)
        except Exception as exc:
            logger.debug("Episodic recall failed: %s", exc)

    def _store_episodic_facts(self, query: str, output: str) -> None:
        """Extract memorable facts from the agent response (non-blocking)."""
        try:
            self._episodic.extract_and_store(  # type: ignore[union-attr]
                session_id=self._session_id,
                user_query=query,
                agent_response=output,
            )
        except Exception as exc:
            logger.debug("Episodic store failed: %s", exc)

    def __repr__(self) -> str:
        agents_str = ", ".join(i.value if isinstance(i, Intent) else str(i)
                               for i in self._agents)
        return (
            f"<Orchestrator session='{self._session_id}' "
            f"agents=[{agents_str}]>"
        )
