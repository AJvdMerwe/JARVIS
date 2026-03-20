"""
agents/chat_agent.py
─────────────────────
General-purpose conversational agent.

Unlike the specialist agents (code, news, search, document), ChatAgent
talks directly to the LLM without invoking any external tools. It is the
right choice for:

  • Open-ended conversation ("how are you", "let's brainstorm…")
  • Reasoning and explanation ("explain X to me like I'm 5")
  • Creative writing (stories, poems, emails, essays)
  • Opinion / advice ("what do you think about…")
  • Follow-up questions on previous agent responses
  • Math / logic reasoning that doesn't require a calculator tool
  • Any query the LLM can answer well from its training alone

The agent injects the full conversation history and (optionally) the
user's style preferences so every response feels personalised and
context-aware.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = """You are a knowledgeable, thoughtful, and engaging personal assistant.
You have a broad knowledge base covering science, technology, history, culture, \
mathematics, philosophy, and the arts.

Your communication style:
- Be conversational and natural, not robotic or overly formal.
- Match the complexity of your answer to the complexity of the question.
- For simple questions, give brief direct answers.
- For complex topics, structure your response clearly.
- Be honest about uncertainty — say "I'm not sure" rather than guessing.
- Use examples and analogies to make abstract ideas concrete.
- Remember and build on earlier parts of the conversation.
"""


class ChatAgent(BaseAgent):
    """
    Direct conversational LLM agent — no tools, just the model and memory.

    Sends the full conversation history along with every request so the
    model can follow multi-turn dialogue naturally. The system prompt is
    augmented with the user's style preferences when available.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[AssistantMemory] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM

    # ── BaseAgent interface ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "chat_agent"

    @property
    def description(self) -> str:
        return (
            "General conversational chat with the LLM. "
            "Use for: open-ended conversation, creative writing, brainstorming, "
            "reasoning, explanation, opinions, follow-up questions, "
            "and anything the model can answer from its own knowledge."
        )

    def get_tools(self):
        # ChatAgent is intentionally tool-free — it talks directly to the LLM.
        return []

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Send a conversational query to the LLM with full history context.

        Args:
            query:       The user's message.
            user_id:     Optional user ID for loading style preferences.
            system_override: Optional custom system prompt for this call only.

        Returns:
            AgentResponse with the model's reply.
        """
        user_id: Optional[str]  = kwargs.get("user_id")
        sys_override: Optional[str] = kwargs.get("system_override")

        self._logger.info("ChatAgent handling: %s", query[:80])

        try:
            system_content = self._build_system(user_id=user_id, override=sys_override)
            messages = self._build_messages(system_content, query)
            response = self._llm.invoke(messages)
            output = str(response.content).strip()

            return AgentResponse(
                output=output,
                agent_name=self.name,
            )

        except Exception as exc:
            self._logger.error("ChatAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"I encountered an error: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_system(
        self,
        user_id: Optional[str] = None,
        override: Optional[str] = None,
    ) -> str:
        """
        Compose the system prompt from the base prompt + user style preferences.
        """
        if override:
            return override

        base = self._system_prompt

        # Inject user preferences if available
        if user_id:
            try:
                from core.user_prefs import get_preferences
                prefs = get_preferences(user_id)
                style = prefs.agent_style_prompt()
                if style:
                    base = f"{base}\n\nUser preferences: {style}"
            except Exception:
                pass   # preferences are optional — never block chat

        return base

    def _build_messages(self, system_content: str, query: str) -> list:
        """
        Build the message list: [system] + [conversation history] + [new user turn].
        """
        msgs: list = [SystemMessage(content=system_content)]

        # Append the recent conversation history (last k*2 messages)
        for msg in self._memory.messages:
            msgs.append(msg)

        # Append the new user query
        msgs.append(HumanMessage(content=query))
        return msgs

    # ── Streaming variant ────────────────────────────────────────────────────

    def stream(self, query: str, **kwargs: Any):
        """
        Yield response tokens one by one (generator, not async).
        Useful for the REPL's live-typing effect without asyncio.
        """
        user_id    = kwargs.get("user_id")
        sys_override = kwargs.get("system_override")
        system_content = self._build_system(user_id=user_id, override=sys_override)
        messages = self._build_messages(system_content, query)

        try:
            for chunk in self._llm.stream(messages):
                token = getattr(chunk, "content", "")
                if token:
                    yield token
        except Exception as exc:
            self._logger.error("ChatAgent stream error: %s", exc)
            yield f"\n[Error: {exc}]"
