"""
core/profile_extractor.py
───────────────────────────
Learns user preferences automatically from conversation and updates the
UserPreferences store, and manages session recall for returning users.

Two public functions:

    extract_and_update_profile(session_id, user_id, query, response)
        Runs asynchronously after each agent turn. Extracts name, interests,
        and preferred agent type from the exchange and persists to the prefs
        store. Never blocks the main response path — all errors are silently
        absorbed.

    build_session_recall(user_id, session_id, episodic_memory)
        Returns a greeting string for returning users that summarises what
        was discussed last time. Returns empty string for first-time users
        or when there are no stored facts.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum gap before a session is considered a "return" worth greeting
_RETURN_GAP_HOURS = 4.0

# How many episodic facts to include in the session recap
_RECAP_FACTS = 5


# =============================================================================
#  Profile extraction
# =============================================================================

# Simple heuristics — extracted before calling the LLM to catch easy cases cheaply
_NAME_PATTERNS = [
    re.compile(r"\bmy name is ([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    re.compile(r"\bi'?m ([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Z][a-z]{1,20})\b", re.IGNORECASE),
    re.compile(r"\bthis is ([A-Z][a-z]{1,20})\b", re.IGNORECASE),
]


def extract_and_update_profile(
    session_id: str,
    user_id:    str,
    query:      str,
    response:   str,
    agent_name: str = "",
) -> None:
    """
    Extract learnable signals from one conversation turn and persist them
    to the user's preference store.

    Extracted signals
    -----------------
    • User's first name (from "my name is X", "I'm X", "call me X")
    • Topics of interest (from what the user keeps asking about)
    • Preferred agent (inferred from which agent answered well)

    This function is designed to be called from the Orchestrator's
    post-processing step.  It catches all exceptions and never raises.

    Parameters
    ----------
    session_id : str
    user_id    : str
    query      : str   The user's message.
    response   : str   The agent's response.
    agent_name : str   Which agent produced the response.
    """
    try:
        from core.user_prefs import get_preferences

        prefs = get_preferences(user_id)
        changed = False

        # ── Extract name ──────────────────────────────────────────────────────
        if not prefs.name:
            for pattern in _NAME_PATTERNS:
                m = pattern.search(query)
                if m:
                    prefs.name = m.group(1).strip().title()
                    logger.info(
                        "Profile: learned name '%s' for user '%s'.",
                        prefs.name, user_id,
                    )
                    changed = True
                    break

        # ── Update session tracking ───────────────────────────────────────────
        prefs.last_seen        = time.time()
        prefs.last_session_id  = session_id
        prefs.total_sessions   = max(prefs.total_sessions, 1)
        changed = True

        # ── Track preferred agent (simple frequency) ──────────────────────────
        _SPECIALIST_AGENTS = {"code_agent", "document_agent", "financial_agent",
                               "deep_research_agent", "news_agent"}
        if agent_name.replace("_agent", "") in {
            "code", "document", "financial", "deep_research", "news"
        }:
            # Infer preferred_agent from consistent use
            clean = agent_name.replace("_agent", "")
            if clean == "deep_research":
                clean = "research"
            if prefs.preferred_agent == "auto":
                # Don't auto-switch on the first use — wait for LLM extraction
                pass

        if changed:
            prefs.save()

    except Exception as exc:
        logger.debug("Profile extraction failed (non-fatal): %s", exc)


def extract_and_update_profile_llm(
    session_id: str,
    user_id:    str,
    query:      str,
    response:   str,
) -> None:
    """
    Use the LLM to extract richer profile signals: topics of interest,
    explicit preference statements ("I prefer technical answers"), etc.

    Runs only every N turns to avoid per-turn LLM overhead.
    Called from the orchestrator on a background thread.
    """
    try:
        from core.user_prefs import get_preferences
        from core.llm_manager import get_llm

        prefs = get_preferences(user_id)

        prompt = (
            "Extract user profile signals from this conversation turn.\n\n"
            f"User said: {query}\n"
            f"Assistant replied: {response[:500]}\n\n"
            "Output a JSON object with these optional keys "
            "(omit keys where nothing was learned):\n"
            '  "name": string (user\'s first name, only if explicitly stated)\n'
            '  "interests": list of strings (topics the user cares about, 1-3 words each)\n'
            '  "style_preference": "concise" | "detailed" | "technical" | "friendly"\n'
            '  "preferred_agent": "code" | "news" | "search" | "document" | "finance" | "research"\n\n'
            "Return {} if nothing was learned. Return ONLY valid JSON."
        )

        llm    = get_llm()
        result = llm.invoke(prompt)
        raw    = str(result.content).strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```json?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        if not raw or raw == "{}":
            return

        import json
        data = json.loads(raw)

        changed = False

        if "name" in data and data["name"] and not prefs.name:
            prefs.name = str(data["name"]).strip()[:30]
            changed = True

        if "interests" in data:
            new_interests = [
                i.strip().lower() for i in data["interests"]
                if isinstance(i, str) and i.strip()
            ]
            for interest in new_interests:
                if interest not in prefs.interests:
                    prefs.interests.append(interest)
                    changed = True
            # Keep top 20 interests
            prefs.interests = prefs.interests[:20]

        if "style_preference" in data:
            valid_styles = ("concise", "detailed", "technical", "friendly")
            if data["style_preference"] in valid_styles:
                prefs.response_style = data["style_preference"]
                changed = True

        if "preferred_agent" in data:
            valid_agents = ("chat", "code", "news", "search", "document", "finance", "research")
            if data["preferred_agent"] in valid_agents:
                prefs.preferred_agent = data["preferred_agent"]
                changed = True

        if changed:
            prefs.save()
            logger.debug(
                "Profile updated for user '%s': %s", user_id,
                {k: v for k, v in data.items() if k in ("name", "interests",
                                                          "style_preference",
                                                          "preferred_agent")}
            )

    except Exception as exc:
        logger.debug("LLM profile extraction failed (non-fatal): %s", exc)


# =============================================================================
#  Session recall
# =============================================================================

def build_session_recall(
    user_id:        str,
    session_id:     str,
    episodic_memory,
    max_facts:      int = _RECAP_FACTS,
) -> str:
    """
    Build a "welcome back" recap for a returning user.

    Returns empty string when:
    - This is the user's first session
    - The user was last seen < _RETURN_GAP_HOURS ago
    - There are no stored episodic facts
    - Any error occurs

    Parameters
    ----------
    user_id        : str
    session_id     : str   The new/current session ID.
    episodic_memory: EpisodicMemory (or None)
    max_facts      : int   Max facts to include in the recap.

    Returns
    -------
    str  Greeting + brief summary, or empty string.
    """
    if episodic_memory is None:
        return ""

    try:
        from core.user_prefs import get_preferences

        prefs   = get_preferences(user_id)
        name    = prefs.greeting_name()
        last    = prefs.last_seen
        n_sess  = prefs.total_sessions

        # First-time user or recent session — no recap needed
        if last == 0.0 or n_sess <= 1:
            return ""
        gap_hours = (time.time() - last) / 3600.0
        if gap_hours < _RETURN_GAP_HOURS:
            return ""

        # Retrieve recent facts
        facts = episodic_memory.list_all()[:max_facts]
        if not facts:
            return ""

        # Format recap
        greeting = f"Welcome back{', ' + name if name else ''}! "
        fact_lines = "\n".join(f"  • {f.text}" for f in facts[:3])

        import math
        if gap_hours < 24:
            when = f"{int(gap_hours)} hour{'s' if gap_hours >= 2 else ''} ago"
        elif gap_hours < 48:
            when = "yesterday"
        else:
            when = f"{int(gap_hours / 24)} days ago"

        recap = (
            f"{greeting}Last time ({when}) we covered:\n"
            f"{fact_lines}\n"
            "Let me know how I can help you today."
        )
        return recap

    except Exception as exc:
        logger.debug("Session recall failed (non-fatal): %s", exc)
        return ""


def update_session_count(user_id: str, session_id: str) -> None:
    """
    Increment the session counter and update last_seen for this user.
    Called at the start of each new REPL session.
    """
    try:
        from core.user_prefs import get_preferences
        prefs                  = get_preferences(user_id)
        prefs.total_sessions   += 1
        prefs.last_seen        = time.time()
        prefs.last_session_id  = session_id
        prefs.save()
    except Exception as exc:
        logger.debug("update_session_count failed: %s", exc)
