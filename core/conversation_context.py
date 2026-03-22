"""
core/conversation_context.py
─────────────────────────────
Utilities for injecting conversation history into any agent, and for
detecting follow-up queries that only make sense with prior context.

Two public functions:

    build_conversation_context(memory, k)
        Returns a formatted string of the last *k* conversation turns,
        suitable for appending to any agent's system/user prompt.

    is_followup_query(query, memory)
        Returns True when the query is short, referential, or otherwise
        likely to be a follow-up that needs prior context to answer well.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory import AssistantMemory

# ── Follow-up detection heuristics ────────────────────────────────────────────

# Short queries that are almost always follow-ups
_FOLLOWUP_MIN_WORDS = 4          # queries with ≤ this many words are candidates

# Referential pronouns / determiners that point to prior context
_REFERENTIAL = re.compile(
    r"\b(it|this|that|these|those|they|them|their|its|the (above|below|previous|last|"
    r"same|following|mentioned|said|above.mentioned|aforementioned)|"
    r"what you (just|said|mentioned|described|showed|wrote|generated)|"
    r"the (code|answer|result|output|response|example|function|class|query|"
    r"report|document|analysis|summary) (above|below|you (wrote|generated|showed))|"
    r"as (mentioned|described|shown|said|above|noted)|"
    r"from (the|that|this) (above|last|previous)|"
    r"(same|that) (approach|method|idea|concept|code|result)|"
    r"(go on|continue|more detail|elaborate|explain (more|further|that)|"
    r"tell me more|and what about|what about|how about|why (is|was|did|does|do|"
    r"would)|can you|could you|please))\b",
    re.IGNORECASE,
)

# Explicit continuation phrases
_CONTINUATION = re.compile(
    r"^(why|and|but|so|also|additionally|furthermore|however|then|now|next|"
    r"okay|ok|yes|no|really|seriously|interesting|got it|i see|understood|"
    r"makes sense|right|correct|exactly|actually|well|sure|fine|great|"
    r"thanks|thank you)\b",
    re.IGNORECASE,
)


def is_followup_query(query: str, memory: "AssistantMemory") -> bool:
    """
    Determine whether a query is a follow-up that needs conversation context.

    A query is treated as a follow-up when ALL of:
      • There is at least one prior turn in memory (otherwise nothing to follow up on)
      • Any of:
          – The query is ≤ _FOLLOWUP_MIN_WORDS words
          – The query contains referential pronouns / phrases ("it", "this", "that")
          – The query starts with a continuation phrase ("why", "how", "and", "but")

    Parameters
    ----------
    query  : str
        The user's raw query text.
    memory : AssistantMemory
        The session memory containing prior conversation turns.

    Returns
    -------
    bool
    """
    if not memory.messages:
        return False

    stripped = query.strip()
    words    = stripped.split()

    if len(words) <= _FOLLOWUP_MIN_WORDS:
        return True
    if _REFERENTIAL.search(stripped):
        return True
    if _CONTINUATION.match(stripped):
        return True

    return False


def build_conversation_context(
    memory:    "AssistantMemory",
    k:         int = 6,
    max_chars: int = 2000,
) -> str:
    """
    Format the last *k* conversation turns as a string for prompt injection.

    The context block looks like::

        === Recent conversation ===
        User: What is the current Apple stock price?
        Assistant: Apple Inc (AAPL) is trading at $178.50…

        User: What about Microsoft?
        Assistant: Microsoft (MSFT) is at $415.20…
        ===========================

    Parameters
    ----------
    memory    : AssistantMemory
        The session memory.
    k         : int
        Maximum number of message pairs (user + assistant) to include.
        Defaults to 6 (= 12 messages).
    max_chars : int
        Total character budget for the context block.  Older turns are
        truncated first if the budget is exceeded.

    Returns
    -------
    str
        Formatted context block, or empty string when there is no history.
    """
    from langchain_core.messages import HumanMessage

    messages = list(memory.messages)
    if not messages:
        return ""

    # Take the last k*2 messages
    recent = messages[-(k * 2):]

    lines: list[str] = []
    used  = 0

    for msg in recent:
        role    = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = str(msg.content).strip()

        # Truncate very long individual messages
        if len(content) > 600:
            content = content[:597] + "…"

        line = f"{role}: {content}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line) + 1

    if not lines:
        return ""

    body = "\n\n".join(lines)
    return f"=== Recent conversation ===\n{body}\n==========================="


def inject_context_into_prompt(
    prompt:  str,
    memory:  "AssistantMemory",
    query:   str,
    *,
    k:       int  = 6,
    force:   bool = False,
) -> str:
    """
    Prepend conversation context to *prompt* when appropriate.

    Context is injected when ``force=True`` **or** when
    :func:`is_followup_query` returns True.

    Parameters
    ----------
    prompt  : str
        The user-facing part of the prompt to augment.
    memory  : AssistantMemory
    query   : str
        The user's raw query (used for follow-up detection).
    k       : int
        Number of prior turns to include.
    force   : bool
        Always inject context regardless of query type.

    Returns
    -------
    str
        The (possibly augmented) prompt string.
    """
    if not force and not is_followup_query(query, memory):
        return prompt

    context = build_conversation_context(memory, k=k)
    if not context:
        return prompt

    return f"{context}\n\n{prompt}"
