"""
agents/base_agent.py
─────────────────────
Abstract base class for all agents.

Design principles applied:
  • Single Responsibility – each agent does one thing.
  • Open/Closed          – extend BaseAgent, don't modify it.
  • Liskov Substitution  – all agents are interchangeable via BaseAgent.
  • Interface Segregation– minimal required interface (run / arun).
  • Dependency Inversion – agents depend on abstractions (BaseChatModel, BaseTool).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from core.llm_manager import get_llm
from core.memory import AssistantMemory
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
#  Compatibility shim — replaces the removed AgentExecutor
# =============================================================================

class _ExecutorCompat:
    """
    Drop-in replacement for the old ``langchain.agents.AgentExecutor``.

    Wraps the new ``langchain.agents.create_agent`` compiled graph and
    exposes the same ``.invoke({"input": query})`` interface that all
    specialist agents use, returning a dict with ``"output"`` and
    ``"intermediate_steps"`` keys.

    LangChain 1.x Migration Note
    ─────────────────────────────
    ``AgentExecutor`` and ``create_react_agent`` were removed in LangChain
    1.0.  The replacement is ``langchain.agents.create_agent`` (which
    internally wraps LangGraph's ``prebuilt.create_react_agent``).  The
    new graph takes ``{"messages": [...]}`` and returns an ``AgentState``
    with a ``messages`` list.

    This shim:
      1. Converts ``{"input": query}`` → ``{"messages": [HumanMessage(query)]}``
      2. Invokes the graph (respecting ``max_iterations`` via a timeout guard)
      3. Extracts the final assistant message as ``"output"``
      4. Reconstructs ``"intermediate_steps"`` from tool-call messages so
         ``_extract_tool_calls()`` continues to work unchanged
    """

    def __init__(self, graph, max_iterations: int = 15) -> None:
        self._graph          = graph
        self._max_iterations = max_iterations

    def invoke(self, inputs: dict, **kwargs) -> dict:
        """
        Execute the agent graph and return a legacy-compatible result dict.

        Parameters
        ----------
        inputs : dict
            Must contain ``"input"`` (str) — the user query.

        Returns
        -------
        dict
            ``{"output": str, "intermediate_steps": list[tuple]}``
        """
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        query    = inputs.get("input", "")
        messages = [HumanMessage(content=query)]

        try:
            state = self._graph.invoke(
                {"messages": messages},
                config={"recursion_limit": self._max_iterations * 3},
            )
        except Exception as exc:
            return {"output": f"Agent error: {exc}", "intermediate_steps": []}

        all_messages = getattr(state, "messages", [])

        # ── Extract final output ──────────────────────────────────────────────
        output = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content
                if isinstance(content, list):
                    # Some models return content as list of blocks
                    parts = [
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    ]
                    content = "".join(parts)
                if content.strip():
                    output = str(content).strip()
                    break

        # ── Reconstruct intermediate_steps ────────────────────────────────────
        # Format: list of (AgentAction-like, observation) pairs
        # We approximate this from ToolMessage pairs in the message list
        intermediate_steps: list[tuple] = []
        pending_tool_call: dict | None  = None

        for msg in all_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    pending_tool_call = tc
            elif isinstance(msg, ToolMessage) and pending_tool_call:
                # Reconstruct a minimal action-like namespace
                class _Action:
                    def __init__(self, name, inp):
                        self.tool       = name
                        self.tool_input = inp
                        self.log        = f"Calling {name}"

                action = _Action(
                    pending_tool_call.get("name", "tool"),
                    pending_tool_call.get("args", {}),
                )
                intermediate_steps.append((action, str(msg.content)[:500]))
                pending_tool_call = None

        return {
            "output":             output or "No answer generated.",
            "intermediate_steps": intermediate_steps,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Agent response model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """
    Structured response returned by every agent.

    Attributes:
        output:        The agent's final answer.
        agent_name:    Which agent produced this.
        tool_calls:    List of (tool_name, tool_input, tool_output) tuples.
        references:    Source document references (used by DocumentAgent).
        metadata:      Arbitrary extra data.
        error:         Non-None if the agent encountered an error.
    """
    output: str
    agent_name: str
    tool_calls: list[tuple[str, str, str]] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __str__(self) -> str:
        return self.output

    @property
    def has_references(self) -> bool:
        return bool(self.references)

    def format_references(self) -> str:
        if not self.references:
            return ""
        lines = ["\n\n---\n**Sources:**"]
        for i, ref in enumerate(self.references, 1):
            lines.append(f"  [{i}] {ref}")
        return "\n".join(lines)

    def full_response(self) -> str:
        """Output + appended references block."""
        return self.output + self.format_references()


# ─────────────────────────────────────────────────────────────────────────────
#  Base agent
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all assistant agents.

    Subclasses must implement:
        • ``name``       – unique identifier string.
        • ``description``– one-line summary of the agent's purpose.
        • ``get_tools()``– return the list of LangChain tools this agent uses.
        • ``run()``      – execute a query and return an AgentResponse.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[AssistantMemory] = None,
        verbose: bool = settings.agent_verbose,
    ) -> None:
        self._llm = llm or get_llm()
        self._memory = memory or AssistantMemory()
        self._verbose = verbose
        self._logger = logging.getLogger(f"agents.{self.name}")

    # ── Abstract interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique short identifier, e.g. 'code_agent'."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description used by the orchestrator for routing."""

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return the tools available to this agent."""

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Execute a query.

        Args:
            query:  The user's input / task description.
            **kwargs: Agent-specific extra parameters.

        Returns:
            An ``AgentResponse`` with output, tool calls, and optional references.
        """

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _build_react_agent(self, system_prompt: Optional[str] = None):
        """
        Build a tool-calling agent compatible with LangChain 1.x.

        LangChain 1.x removed ``AgentExecutor`` and ``create_react_agent``.
        This method builds an equivalent agent using the new
        ``langchain.agents.create_agent`` API (which wraps LangGraph's
        ``create_react_agent`` under the hood) and returns a thin
        ``_ExecutorCompat`` wrapper that preserves the original
        ``.invoke({"input": query})`` call-site interface used by every
        specialist agent.

        The returned object satisfies::

            result = executor.invoke({"input": query})
            output       = result["output"]           # str
            tool_calls   = result["intermediate_steps"]  # list[tuple]

        Parameters
        ----------
        system_prompt : str, optional
            Custom system message.  Defaults to a generic description-based prompt.

        Returns
        -------
        _ExecutorCompat
            Drop-in replacement for the old AgentExecutor.
        """
        from langchain.agents import create_agent

        tools = self.get_tools()
        prompt = system_prompt or (
            f"You are a helpful assistant specialised in: {self.description}.\n"
            "Use the available tools to answer the user's question thoroughly."
        )

        graph = create_agent(
            self._llm,
            tools=tools,
            system_prompt=prompt,
        )
        return _ExecutorCompat(graph, max_iterations=settings.agent_max_iterations)

    def _extract_tool_calls(
        self, intermediate_steps: list
    ) -> list[tuple[str, str, str]]:
        """Parse LangChain intermediate_steps into our cleaner tuple format."""
        calls = []
        for action, observation in intermediate_steps:
            calls.append((
                getattr(action, "tool", "unknown"),
                str(getattr(action, "tool_input", "")),
                str(observation)[:500],
            ))
        return calls



    def stream(self, query: str, **kwargs):
        """
        Yield response tokens one by one.

        Default implementation falls back to ``run()`` and yields the full
        output as a single token.  Subclasses override this to provide true
        token-by-token streaming (ChatAgent already does).

        Yields
        ------
        str  Token string fragments.
        """
        response = self.run(query, **kwargs)
        if response.output:
            yield response.output

    def _augment_query(self, query: str, force_context: bool = False) -> str:
        """
        Prepend recent conversation context to *query* when appropriate.

        Called by specialist agents (search, code, finance, news) at the
        start of their ``run()`` methods so every agent benefits from
        conversation history — not just ChatAgent.

        Context is injected when:
          • ``force_context=True``, OR
          • the query is detected as a follow-up (short / referential / pronoun)

        See ``core.conversation_context`` for detection heuristics.
        """
        try:
            from core.conversation_context import inject_context_into_prompt
            return inject_context_into_prompt(
                query, self._memory, query, force=force_context
            )
        except Exception:
            return query   # never block the agent

    @property
    def memory(self) -> AssistantMemory:
        return self._memory

    def clear_memory(self) -> None:
        self._memory.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
