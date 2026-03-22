"""
agents/rag_precheck.py
───────────────────────
Lightweight RAG pre-check that runs after intent detection and BEFORE
the primary agent is invoked.

Design goals
────────────
• Fast  — a single vector-store query + one LLM call; typically < 200 ms.
• Non-blocking — any exception falls through silently to the normal agent.
• Transparent — every decision is logged at DEBUG level.
• Configurable — threshold, result count, and enabled/disabled via the
  Orchestrator constructor or env vars.

How it works
────────────
After the intent is detected the pre-check:

  1. Checks whether the knowledge base has any content at all.
     If empty → skip immediately (no latency penalty).

  2. Runs a similarity search for the user query.
     If no results exceed ``rag_similarity_threshold`` → skip.

  3. Asks the LLM to answer the query using only the retrieved chunks.
     Prompt is strict: answer only from context, say "INSUFFICIENT" if
     the context does not contain the answer.

  4. Evaluates the LLM's answer with the same ``_is_sufficient_response``
     gate used by the fallback loop.

     SUFFICIENT  → return the RAG answer immediately; agents are never called.
     INSUFFICIENT → return None; caller proceeds to the normal agent routing.

The caller (Orchestrator.run) checks the return value:
  • AgentResponse  → use it, skip agents
  • None           → proceed as normal

Intents that bypass the pre-check
──────────────────────────────────
• CODE   — code generation / debugging is not a retrieval task.
• CHAT   — open-ended conversation does not benefit from KB pre-checks.

All other intents (NEWS, SEARCH, DOCUMENT, FINANCE, UNKNOWN) participate.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from agents.base_agent import AgentResponse

logger = logging.getLogger(__name__)

# Intents that always skip the RAG pre-check.
# Import here to avoid a circular import; the str values are checked directly.
_SKIP_INTENTS: frozenset[str] = frozenset({"code", "chat"})

# Prompt template used for the RAG answer generation step.
_RAG_PROMPT_TEMPLATE = """\
You are answering a user question using ONLY the document excerpts below.

Rules:
- If the excerpts contain a clear, direct answer: provide it concisely.
- If the excerpts are partially relevant: use what is available and note limitations.
- If the excerpts do NOT contain enough information to answer: reply with exactly
  the single word INSUFFICIENT and nothing else.
- Never fabricate information not present in the excerpts.
- Always cite the source reference (document name / page) when answering.

User question: {query}

Document excerpts:
{context}

Answer:"""


def rag_precheck(
    query: str,
    intent_value: str,
    *,
    similarity_threshold: float = 0.55,
    k: int = 4,
    max_context_chars: int = 3000,
    enabled: bool = True,
) -> Optional[AgentResponse]:
    """
    Run the RAG pre-check and return an AgentResponse when the KB can answer
    the query, or None when the pre-check should be skipped or failed.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    intent_value : str
        The detected intent's string value (e.g. ``"search"``, ``"finance"``).
    similarity_threshold : float
        Minimum cosine similarity for a chunk to be considered relevant.
        Higher values (0.65+) are more selective; lower values (0.45) are
        more permissive.  Default 0.55 is a good balance.
    k : int
        Maximum number of chunks to retrieve and include in the context.
    max_context_chars : int
        Total character budget for all context chunks combined.
    enabled : bool
        Master switch.  When False the function immediately returns None.

    Returns
    -------
    AgentResponse or None
        ``AgentResponse`` (with ``agent_name="rag_precheck"``) when the KB
        produced a sufficient answer.
        ``None`` when the pre-check was skipped, found nothing relevant, or
        the generated answer was insufficient.
    """
    if not enabled:
        return None

    if intent_value in _SKIP_INTENTS:
        logger.debug("RAG pre-check: skipping for intent '%s'.", intent_value)
        return None

    t0 = time.monotonic()

    # ── Step 1: obtain the DocumentManager (lazy, singleton-safe) ────────────
    try:
        dm = _get_document_manager()
    except Exception as exc:
        logger.debug("RAG pre-check: DocumentManager unavailable (%s).", exc)
        return None

    # ── Step 2: bail early if KB is empty ────────────────────────────────────
    try:
        if dm.total_chunks == 0:
            logger.debug("RAG pre-check: knowledge base is empty — skipping.")
            return None
    except Exception:
        return None

    # ── Step 3: semantic search ───────────────────────────────────────────────
    try:
        results = dm.search(
            query,
            k=k,
            similarity_threshold=similarity_threshold,
        )
    except Exception as exc:
        logger.debug("RAG pre-check: search failed (%s).", exc)
        return None

    if not results:
        logger.debug(
            "RAG pre-check: no chunks above threshold=%.2f — skipping.",
            similarity_threshold,
        )
        return None

    top_score = results[0].score
    logger.debug(
        "RAG pre-check: %d chunk(s) found, top score=%.3f, threshold=%.2f.",
        len(results), top_score, similarity_threshold,
    )

    # ── Step 4: build context block ──────────────────────────────────────────
    context = _build_context(results, max_context_chars)

    # ── Step 5: ask the LLM to answer from context ───────────────────────────
    try:
        llm    = _get_llm()
        prompt = _RAG_PROMPT_TEMPLATE.format(query=query, context=context)
        result = llm.invoke(prompt)
        answer = str(result.content).strip()
    except Exception as exc:
        logger.debug("RAG pre-check: LLM call failed (%s).", exc)
        return None

    elapsed_ms = (time.monotonic() - t0) * 1000

    # ── Step 6: quality gate ─────────────────────────────────────────────────
    if answer.upper() == "INSUFFICIENT" or not answer:
        logger.debug(
            "RAG pre-check: LLM said INSUFFICIENT (%.0fms).", elapsed_ms
        )
        return None

    from agents.orchestrator import _is_sufficient_response

    candidate = AgentResponse(
        output=answer,
        agent_name="rag_precheck",
        references=[r.reference for r in results],
        metadata={
            "rag_chunks": len(results),
            "rag_top_score": round(top_score, 3),
            "rag_elapsed_ms": round(elapsed_ms, 1),
        },
    )

    if not _is_sufficient_response(candidate, query):
        logger.debug(
            "RAG pre-check: answer failed quality gate (%.0fms).", elapsed_ms
        )
        return None

    logger.info(
        "RAG pre-check: KB answered query in %.0fms "
        "(intent=%s, chunks=%d, top_score=%.3f).",
        elapsed_ms, intent_value, len(results), top_score,
    )
    return candidate


# =============================================================================
#  Context assembly
# =============================================================================

def _build_context(results, max_chars: int) -> str:
    """
    Concatenate search results into a numbered context block, respecting the
    character budget.  Results are already ordered by descending similarity.
    """
    from document_processing.vector_store import SearchResult

    parts: list[str] = []
    used  = 0

    for i, r in enumerate(results, start=1):
        text      = r.chunk.text
        ref       = r.reference
        header    = f"[{i}] {ref} (score: {r.score:.2f})\n"
        available = max_chars - used - len(header) - 2   # 2 for \n\n

        if available <= 40:
            break  # no budget left

        if len(text) > available:
            text = text[:available].rstrip() + "…"

        part  = header + text
        parts.append(part)
        used += len(part) + 2

    return "\n\n".join(parts)


# =============================================================================
#  Lazy singletons (avoid circular imports and heavy startup cost)
# =============================================================================

_dm_singleton = None
_dm_lock      = None


def _get_document_manager():
    """Return the shared DocumentManager, creating it lazily on first call."""
    global _dm_singleton, _dm_lock
    import threading
    if _dm_lock is None:
        _dm_lock = threading.Lock()
    with _dm_lock:
        if _dm_singleton is None:
            from document_processing import DocumentManager
            _dm_singleton = DocumentManager()
        return _dm_singleton


def _get_llm():
    """Return the shared LLM instance."""
    from core.llm_manager import get_llm
    return get_llm()


def reset_singletons() -> None:
    """
    Clear the module-level singletons.  Intended for tests only — allows
    each test to inject its own DocumentManager without state leaking.
    """
    global _dm_singleton
    _dm_singleton = None
