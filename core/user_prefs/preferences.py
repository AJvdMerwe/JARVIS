"""
core/user_prefs/preferences.py
────────────────────────────────
Per-user preference store backed by a JSON file.

Supported preferences:
  • preferred_agent     — default agent for new sessions
  • response_style      — "concise" | "detailed" | "technical" | "friendly"
  • language            — ISO 639-1 language code (default "en")
  • voice_enabled       — bool
  • whisper_model       — override global Whisper model size
  • timezone            — IANA timezone name
  • max_results         — default k for document search
  • news_topics         — list of preferred news topics / keywords
  • show_tool_calls     — bool — show tool call details in REPL

Preferences are validated with Pydantic and serialised atomically to disk.
A UserPreferences singleton is associated with a user_id (defaults to "default").

Usage::

    prefs = UserPreferences.load("alice")
    prefs.response_style = "technical"
    prefs.news_topics = ["AI", "climate"]
    prefs.save()

    # In an agent:
    style = prefs.response_style
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from config import settings

logger = logging.getLogger(__name__)

_PREFS_DIR = settings.log_path.parent / "user_prefs"
_PREFS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class UserPreferences(BaseModel):
    """
    All user-configurable preferences with sane defaults.
    Persisted as JSON; validated on load and save.
    """

    user_id: str = "default"

    # Agent behaviour
    preferred_agent: Literal["auto", "code", "news", "search", "document"] = "auto"
    response_style:  Literal["concise", "detailed", "technical", "friendly"] = "concise"
    language:        str = Field(default="en", min_length=2, max_length=10)

    # Voice
    voice_enabled:  bool = False
    whisper_model:  Literal["tiny", "base", "small", "medium", "large"] = "base"

    # Localisation
    timezone: str = "UTC"

    # Document search
    max_results: int = Field(default=5, ge=1, le=20)

    # News
    news_topics: list[str] = Field(default_factory=list)

    # UI / REPL
    show_tool_calls: bool = False

    # Custom system prompt suffix (appended to every agent system prompt)
    custom_instructions: str = ""

    @field_validator("news_topics", mode="before")
    @classmethod
    def _clean_topics(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return [str(t).strip() for t in (v or [])]

    @field_validator("language", mode="before")
    @classmethod
    def _lower_lang(cls, v):
        return str(v).lower()[:10]

    # ── Persistence ──────────────────────────────────────────────────────────

    @classmethod
    def _path(cls, user_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in user_id)
        return _PREFS_DIR / f"{safe_id}.json"

    @classmethod
    def load(cls, user_id: str = "default") -> "UserPreferences":
        """Load preferences from disk, or return defaults if not found."""
        path = cls._path(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["user_id"] = user_id
                return cls(**data)
            except Exception as exc:
                logger.warning("Could not load prefs for '%s': %s — using defaults.", user_id, exc)
        return cls(user_id=user_id)

    def save(self) -> None:
        """Atomically persist preferences to disk."""
        path = self._path(self.user_id)
        tmp  = path.with_suffix(".tmp")
        tmp.write_text(
            self.model_dump_json(indent=2, exclude={"user_id"}),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.debug("Saved preferences for user '%s'.", self.user_id)

    def reset(self) -> None:
        """Reset to defaults and remove the persisted file."""
        path = self._path(self.user_id)
        if path.exists():
            path.unlink()
        # Reset all fields to defaults
        defaults = UserPreferences(user_id=self.user_id)
        for field_name in self.model_fields:
            if field_name != "user_id":
                setattr(self, field_name, getattr(defaults, field_name))
        logger.info("Reset preferences for user '%s'.", self.user_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def agent_style_prompt(self) -> str:
        """
        Returns a style instruction string to append to agent system prompts,
        respecting the user's preferences.
        """
        style_map = {
            "concise":   "Be brief and direct. Aim for 2–4 sentences unless more detail is needed.",
            "detailed":  "Provide comprehensive, thorough answers with examples.",
            "technical": "Use precise technical language. Include code where helpful.",
            "friendly":  "Use a warm, conversational tone. Avoid jargon unless necessary.",
        }
        parts = [style_map[self.response_style]]

        if self.language != "en":
            parts.append(f"Respond in language: {self.language}.")
        if self.custom_instructions:
            parts.append(self.custom_instructions.strip())
        if self.news_topics:
            parts.append(
                f"When discussing news, prioritise topics: {', '.join(self.news_topics)}."
            )
        return " ".join(parts)

    def summary(self) -> str:
        return (
            f"User: {self.user_id} | Style: {self.response_style} | "
            f"Agent: {self.preferred_agent} | Lang: {self.language} | "
            f"TZ: {self.timezone}"
        )


# ── In-memory registry (avoids re-reading disk on every request) ─────────────

_prefs_cache: dict[str, UserPreferences] = {}
_prefs_lock = threading.Lock()


def get_preferences(user_id: str = "default") -> UserPreferences:
    """Return cached preferences for a user, loading from disk if needed."""
    with _prefs_lock:
        if user_id not in _prefs_cache:
            _prefs_cache[user_id] = UserPreferences.load(user_id)
        return _prefs_cache[user_id]


def invalidate_cache(user_id: str) -> None:
    """Force reload from disk on next get_preferences() call."""
    with _prefs_lock:
        _prefs_cache.pop(user_id, None)
