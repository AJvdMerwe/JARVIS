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
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate

from core.llm_manager import get_llm
from core.memory import AssistantMemory
from config import settings

logger = logging.getLogger(__name__)


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

        # def _build_react_agent(self, system_prompt: Optional[str] = None):
        #     """
        #     Build a LangChain ReAct agent executor with this agent's tools.
        #     Shared utility so subclasses don't repeat boilerplate.
        #     """
        #     from langchain.agents import AgentExecutor, create_react_agent
        #     from langchain_core.prompts import PromptTemplate
        #
        #     tools = self.get_tools()
        #
        #     # Build tool descriptions string for the prompt
        #     tool_descriptions = "\n".join(
        #         f"- {t.name}: {t.description}" for t in tools
        #     )
        #     tool_names = ", ".join(t.name for t in tools)
        #
        #     base_prompt = system_prompt or (
        #         f"You are a helpful assistant specialised in: {self.description}.\n"
        #         "Answer the human's question as best you can using the available tools."
        #     )
        #
        #     template = (
        #         f"{base_prompt}\n\n"
        #         "You have access to the following tools:\n"
        #         "{tools}\n\n"
        #         "Use the following format:\n"
        #         "Question: the input question you must answer\n"
        #         "Thought: you should always think about what to do\n"
        #         "Action: the action to take, should be one of [{tool_names}]\n"
        #         "Action Input: the input to the action\n"
        #         "Observation: the result of the action\n"
        #         "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        #         "Thought: I now know the final answer\n"
        #         "Final Answer: the final answer to the original input question\n\n"
        #         "Begin!\n\n"
        #         "Question: {input}\n"
        #         "Thought: {agent_scratchpad}"
        #     )
        #
        #     prompt = PromptTemplate.from_template(template).partial(
        #         tools=tool_descriptions,
        #         tool_names=tool_names,
        #     )
        #
        #     agent = create_react_agent(self._llm, tools, prompt)
        #     return AgentExecutor(
        #         agent=agent,
        #         tools=tools,
        #         memory=self._memory.lc_memory,
        #         max_iterations=settings.agent_max_iterations,
        #         verbose=self._verbose,
        #         handle_parsing_errors=True,
        #         return_intermediate_steps=True,
        #     )


    def _build_react_agent(self, system_prompt: Optional[str] = None):
        """
        Build a modern LangChain agent.
        In v1.2+, create_agent returns a compiled graph that handles
        execution, so AgentExecutor is no longer required.
        """
        tools = self.get_tools()

        # Build tool descriptions string for the prompt
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        tool_names = ", ".join(t.name for t in tools)

        base_prompt = system_prompt or (
            f"You are a helpful assistant specialised in: {self.description}.\n"
            "Answer the human's question as best you can using the available tools."
        )

        template = (
            f"{base_prompt}\n\n"
            "You have access to the following tools:\n"
            "{tools}\n\n"
            "Use the following format:\n"
            "Question: the input question you must answer\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, should be one of [{tool_names}]\n"
            "Action Input: the input to the action\n"
            "Observation: the result of the action\n"
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question\n\n"
            "Begin!\n\n"
            "Question: {input}\n"
            "Thought: {agent_scratchpad}"
        )

        prompt = PromptTemplate.from_template(template).partial(
            tools=tool_descriptions,
            tool_names=tool_names,
        )

        prompt = system_prompt or (
            f"You are a helpful assistant specialised in: {self.description}."
        )

        # create_agent returns an executable 'agent' object (a compiled graph)
        agent = create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=prompt,
            # 'checkpointer' replaces manual memory management
            # 'max_steps' replaces max_iterations
            # max_steps=getattr(settings, "agent_max_iterations", 10),
            # handle_tool_errors=True,
            # stream_runnables=True
        )

        return agent

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

    @property
    def memory(self) -> AssistantMemory:
        return self._memory

    def clear_memory(self) -> None:
        self._memory.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"