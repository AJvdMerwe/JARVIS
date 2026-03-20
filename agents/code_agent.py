"""
agents/code_agent.py
─────────────────────
Specialised agent for all code-related tasks:
  • Write code from natural language descriptions.
  • Review, explain, or debug existing code.
  • Execute Python snippets and return output.
  • Suggest architecture and design patterns.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from tools.code_tools import get_code_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert software engineer and code assistant.
You follow SOLID principles, write clean and well-documented code, and always
consider edge cases, error handling, and security implications.

When writing code:
- Choose the most appropriate language unless specified.
- Add docstrings and inline comments.
- Handle errors gracefully.
- Suggest tests where applicable.

When reviewing code:
- Identify bugs, security issues, and performance problems.
- Suggest concrete improvements with reasoning.

Always use the available tools rather than generating code from memory alone.
"""


class CodeAgent(BaseAgent):
    """Agent that writes, reviews, and executes code."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[AssistantMemory] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._executor = self._build_react_agent(system_prompt=_SYSTEM_PROMPT)

    # ── BaseAgent interface ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "code_agent"

    @property
    def description(self) -> str:
        return (
            "Write, review, debug, and execute code. "
            "Use for programming tasks, code generation, code explanation, "
            "algorithm design, and running Python snippets."
        )

    def get_tools(self) -> list[BaseTool]:
        return get_code_tools()

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a code-related request.

        Args:
            query:    Natural language task (e.g. "write a binary search in Python").
            language: Optional hint, e.g. "typescript".

        Returns:
            AgentResponse with code output and tool call log.
        """
        language_hint = kwargs.get("language", "")
        full_query = query
        if language_hint:
            full_query = f"[Language preference: {language_hint}] {query}"

        self._logger.info("CodeAgent handling: %s", query[:80])

        try:
            result = self._executor.invoke({"input": full_query})
            output = result.get("output", "No output generated.")
            tool_calls = self._extract_tool_calls(
                result.get("intermediate_steps", [])
            )
            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
            )
        except Exception as exc:
            self._logger.error("CodeAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"I encountered an error while processing your code request: {exc}",
                agent_name=self.name,
                error=str(exc),
            )
