"""
core/summariser/conversation_summariser.py
───────────────────────────────────────────
Compresses old conversation turns into a rolling summary so that the
LLM's context window is never exhausted during long sessions.

Strategy:
  • Keep the last ``recent_k`` turns verbatim (recency is most important).
  • Summarise everything older into a single paragraph using the LLM.
  • Replace the older turns with a synthetic ``SystemMessage`` containing
    the summary — the LLM sees it as prior context, not raw history.
  • Re-summarise periodically as the rolling summary itself grows.

Usage::

    from core.summariser import ConversationSummariser
    from core.memory import AssistantMemory

    mem = AssistantMemory(k=40)
    summariser = ConversationSummariser(recent_k=10, summarise_after=30)

    # Call after each agent turn:
    summariser.maybe_summarise(mem)
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class ConversationSummariser:
    """
    Rolling conversation summariser.

    Args:
        recent_k:         Number of recent turns to keep verbatim.
        summarise_after:  Total message count that triggers compression.
        max_summary_len:  Max chars for the summary prompt injection.
    """

    def __init__(
        self,
        recent_k: int = 10,
        summarise_after: int = 30,
        max_summary_len: int = 800,
    ) -> None:
        self.recent_k         = recent_k
        self.summarise_after  = summarise_after
        self.max_summary_len  = max_summary_len
        self._last_summary: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def maybe_summarise(self, memory) -> bool:
        """
        Compress old turns in *memory* if the total message count exceeds
        ``summarise_after``.

        Args:
            memory: An ``AssistantMemory`` or ``PersistentMemory`` instance.

        Returns:
            True if compression happened, False if not needed yet.
        """
        messages = memory.messages
        if len(messages) < self.summarise_after:
            return False

        # Split: old turns to compress | recent turns to keep
        keep_count  = self.recent_k * 2          # 2 messages per turn (human + AI)
        old_messages = messages[:-keep_count] if keep_count else messages
        recent_messages = messages[-keep_count:] if keep_count else []

        if not old_messages:
            return False

        summary = self._summarise(old_messages)
        if not summary:
            return False

        # Replace all messages with: [summary system msg] + [recent turns]
        memory.lc_memory.chat_memory.clear()

        # Re-inject the rolling summary as context
        summary_msg = f"[Conversation summary — earlier turns compressed]\n{summary}"
        memory.lc_memory.chat_memory.add_message(SystemMessage(content=summary_msg))

        # Re-inject recent turns
        for msg in recent_messages:
            memory.lc_memory.chat_memory.add_message(msg)

        self._last_summary = summary
        logger.info(
            "Compressed %d old messages → %d-char summary. %d recent turns kept.",
            len(old_messages), len(summary), len(recent_messages) // 2,
        )
        return True

    def force_summarise(self, memory) -> str:
        """
        Force a summary regardless of message count. Returns the summary text.
        """
        messages = memory.messages
        if not messages:
            return ""
        summary = self._summarise(messages)
        return summary or ""

    @property
    def last_summary(self) -> Optional[str]:
        """The most recent generated summary, or None if none yet."""
        return self._last_summary

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _summarise(self, messages: list[BaseMessage]) -> Optional[str]:
        """Ask the LLM to summarise a list of messages."""
        from core.llm_manager import get_llm

        if not messages:
            return None

        # Build a readable transcript
        transcript_lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                transcript_lines.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                transcript_lines.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                transcript_lines.append(f"[Context: {msg.content[:200]}]")

        transcript = "\n".join(transcript_lines)
        # Truncate very long transcripts before sending
        if len(transcript) > 6000:
            transcript = transcript[:6000] + "\n[... truncated]"

        prompt = (
            "Summarise the following conversation in 2–4 sentences, capturing "
            "the main topics discussed, any decisions made, and key information "
            "exchanged. Be concise and factual.\n\n"
            f"Conversation:\n{transcript}\n\n"
            "Summary:"
        )

        try:
            llm = get_llm()
            result = llm.invoke(prompt)
            summary = str(result.content).strip()
            # Enforce max length
            if len(summary) > self.max_summary_len:
                summary = summary[: self.max_summary_len].rsplit(".", 1)[0] + "."
            return summary
        except Exception as exc:
            logger.warning("Summarisation failed: %s", exc)
            return None
