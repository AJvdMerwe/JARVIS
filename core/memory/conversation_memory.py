"""
core/memory/conversation_memory.py
────────────────────────────────────
Sliding-window conversation memory using the modern langchain-core API
(compatible with LangChain ≥ 0.2 / 1.x — no legacy langchain.memory).

Provides:
  • ``AssistantMemory``   – in-process memory for a single session.
  • ``PersistentMemory``  – JSON-backed memory that survives restarts.

The ``lc_memory`` attribute exposes a dict-based memory shim that agents
can inject into ReAct executors via the ``memory`` parameter.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight window-memory shim compatible with LangChain agent executors
# ─────────────────────────────────────────────────────────────────────────────

class _WindowMemoryShim:
    """
    Minimal drop-in replacement for ConversationBufferWindowMemory.
    Keeps the last *k* human/AI pairs and exposes the standard
    ``chat_memory``, ``load_memory_variables()``, and ``save_context()`` API.
    """

    def __init__(self, k: int = 20) -> None:
        self.k = k
        self.chat_memory = InMemoryChatMessageHistory()
        self.return_messages = True
        self.memory_key = "chat_history"
        self.input_key  = "input"
        self.output_key = "output"

    def load_memory_variables(self, inputs: dict) -> dict:
        msgs = self.chat_memory.messages
        # Return only the last k*2 messages (k pairs)
        return {self.memory_key: msgs[-(self.k * 2):]}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        human_text = inputs.get(self.input_key, "")
        ai_text    = outputs.get(self.output_key, "")
        if human_text:
            self.chat_memory.add_user_message(human_text)
        if ai_text:
            self.chat_memory.add_ai_message(ai_text)

    def clear(self) -> None:
        self.chat_memory.clear()

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]


# ─────────────────────────────────────────────────────────────────────────────
#  AssistantMemory
# ─────────────────────────────────────────────────────────────────────────────

class AssistantMemory:
    """
    Conversation memory for a single assistant session.

    Attributes:
        lc_memory: Window-memory shim; agents bind this directly.
    """

    def __init__(self, k: int = 20) -> None:
        self.lc_memory = _WindowMemoryShim(k=k)

    # ── Convenience wrappers ─────────────────────────────────────────────────

    def add_user_message(self, text: str) -> None:
        self.lc_memory.chat_memory.add_user_message(text)

    def add_ai_message(self, text: str) -> None:
        self.lc_memory.chat_memory.add_ai_message(text)

    def save_context(self, user_input: str, ai_output: str) -> None:
        self.lc_memory.save_context(
            {self.lc_memory.input_key:  user_input},
            {self.lc_memory.output_key: ai_output},
        )

    @property
    def messages(self) -> list[BaseMessage]:
        return self.lc_memory.chat_memory.messages

    @property
    def history_str(self) -> str:
        lines = []
        for msg in self.messages:
            role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.lc_memory.clear()
        logger.debug("Memory cleared.")


# ─────────────────────────────────────────────────────────────────────────────
#  PersistentMemory
# ─────────────────────────────────────────────────────────────────────────────

class PersistentMemory(AssistantMemory):
    """
    AssistantMemory whose messages are persisted to a JSON file so that
    conversation context survives application restarts.
    """

    def __init__(self, session_id: str, k: int = 20) -> None:
        super().__init__(k=k)
        self._session_id = session_id
        self._path = (
            Path(settings.log_path).parent / "sessions" / f"{session_id}.json"
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data: list[dict[str, Any]] = json.loads(self._path.read_text())
            for entry in data:
                if entry["role"] == "human":
                    self.lc_memory.chat_memory.add_user_message(entry["content"])
                else:
                    self.lc_memory.chat_memory.add_ai_message(entry["content"])
            logger.info("Loaded %d messages for session '%s'.", len(data), self._session_id)
        except Exception as exc:
            logger.warning("Could not load memory from %s: %s", self._path, exc)

    def _save(self) -> None:
        records = [
            {
                "role": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            for msg in self.messages
        ]
        self._path.write_text(json.dumps(records, indent=2))

    # ── Override mutating methods to trigger save ────────────────────────────

    def save_context(self, user_input: str, ai_output: str) -> None:
        super().save_context(user_input, ai_output)
        self._save()

    def add_user_message(self, text: str) -> None:
        super().add_user_message(text)
        self._save()

    def add_ai_message(self, text: str) -> None:
        super().add_ai_message(text)
        self._save()

    def clear(self) -> None:
        super().clear()
        if self._path.exists():
            self._path.unlink()
        logger.info("Cleared persistent memory for session '%s'.", self._session_id)
