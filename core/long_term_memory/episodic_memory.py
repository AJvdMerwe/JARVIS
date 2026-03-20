"""
core/long_term_memory/episodic_memory.py
──────────────────────────────────────────
Long-term episodic memory: stores and retrieves semantically relevant
past conversation turns across sessions.

Unlike ``PersistentMemory`` (which replays recent turns verbatim),
episodic memory stores *facts and summaries* from past interactions and
surfaces the most relevant ones as context when answering new queries.

Architecture:
  • Each ``EpisodicFact`` captures a key fact, its source session, and a
    timestamp.
  • Facts are embedded and stored in a dedicated ChromaDB collection
    (``episodic_memory``).
  • On ``recall(query)``, the top-k most semantically similar facts are
    returned and can be injected into the agent's context.
  • Facts are extracted automatically from agent responses via
    ``extract_and_store()``.

Usage::

    mem = EpisodicMemory()

    # After an agent turn:
    mem.extract_and_store(
        session_id="abc",
        user_query="What is our Q3 revenue?",
        agent_response="Q3 revenue was $15.2M, up 8% YoY.",
    )

    # Before the next turn:
    context = mem.recall("revenue this quarter")
    # context → "In session abc: Q3 revenue was $15.2M, up 8% YoY."
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodicFact:
    """
    A single memorable fact extracted from a conversation.

    Attributes:
        fact_id:     Stable SHA-256 ID.
        text:        The fact text (usually 1–3 sentences).
        session_id:  Which session this fact came from.
        query:       The user query that generated this fact.
        timestamp:   Unix timestamp when the fact was stored.
        importance:  0.0–1.0 subjective importance score (higher = recalled first).
    """
    fact_id:    str
    text:       str
    session_id: str
    query:      str
    timestamp:  float = field(default_factory=time.time)
    importance: float = 0.5

    @property
    def age_days(self) -> float:
        return (time.time() - self.timestamp) / 86400

    @property
    def formatted(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return f"[{dt.strftime('%Y-%m-%d')} | session:{self.session_id}] {self.text}"


# ─────────────────────────────────────────────────────────────────────────────
#  EpisodicMemory
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Semantic long-term memory backed by a dedicated ChromaDB collection.

    Args:
        persist_directory: Where to store the ChromaDB files.
        collection_name:   ChromaDB collection name.
        max_facts:         Maximum facts to store before pruning oldest.
        recall_k:          How many facts to return per recall query.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "episodic_memory",
        max_facts: int = 2000,
        recall_k: int = 5,
    ) -> None:
        self._persist_dir   = persist_directory or (settings.vector_store_path.parent / "episodic")
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._max_facts     = max_facts
        self._recall_k      = recall_k
        self._collection    = None
        self._embeddings    = None

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb is required for EpisodicMemory.") from exc

        from core.llm_manager import get_embeddings
        self._embeddings = get_embeddings()
        client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "EpisodicMemory ready | %d facts stored.", self._collection.count()
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def store(self, fact: EpisodicFact) -> None:
        """Store a single fact. Idempotent — duplicate fact_ids are skipped."""
        self._ensure_ready()

        # Check for duplicates
        try:
            existing = self._collection.get(ids=[fact.fact_id])
            if existing["ids"]:
                return
        except Exception:
            pass

        embedding = self._embeddings.embed_query(fact.text)
        self._collection.add(
            ids=[fact.fact_id],
            embeddings=[embedding],
            documents=[fact.text],
            metadatas=[{
                "session_id": fact.session_id,
                "query":      fact.query[:200],
                "timestamp":  fact.timestamp,
                "importance": fact.importance,
                "formatted":  fact.formatted,
            }],
        )

        # Prune if over limit
        count = self._collection.count()
        if count > self._max_facts:
            self._prune_oldest(count - self._max_facts)

    def recall(self, query: str, k: Optional[int] = None) -> list[EpisodicFact]:
        """
        Retrieve the most semantically relevant facts for a query.

        Args:
            query: The current user query.
            k:     Max facts to return (defaults to self.recall_k).

        Returns:
            List of ``EpisodicFact`` sorted by relevance.
        """
        self._ensure_ready()
        effective_k = k or self._recall_k

        if self._collection.count() == 0:
            return []

        embedding = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(effective_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        facts: list[EpisodicFact] = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - float(distance)
            if score < 0.3:    # skip very low-relevance facts
                continue
            facts.append(EpisodicFact(
                fact_id    = hashlib.sha256(text.encode()).hexdigest()[:16],
                text       = text,
                session_id = meta.get("session_id", ""),
                query      = meta.get("query", ""),
                timestamp  = float(meta.get("timestamp", 0)),
                importance = float(meta.get("importance", 0.5)),
            ))

        return facts

    def recall_as_context(self, query: str, k: Optional[int] = None) -> str:
        """
        Recall facts and format them as a context string for injection
        into an agent prompt.

        Returns empty string if no relevant facts exist.
        """
        facts = self.recall(query, k=k)
        if not facts:
            return ""
        lines = ["Relevant memories from past conversations:"]
        for fact in facts:
            lines.append(f"  • {fact.formatted}")
        return "\n".join(lines)

    # ── Auto-extraction from agent responses ──────────────────────────────────

    def extract_and_store(
        self,
        session_id: str,
        user_query: str,
        agent_response: str,
        importance: float = 0.5,
    ) -> int:
        """
        Use the LLM to extract memorable facts from an agent response
        and store them automatically.

        Args:
            session_id:     The current session.
            user_query:     What the user asked.
            agent_response: The agent's answer.
            importance:     Importance score for these facts.

        Returns:
            Number of facts stored.
        """
        from core.llm_manager import get_llm

        prompt = (
            "Extract 0–3 concise, self-contained facts worth remembering "
            "from this conversation exchange. Only extract genuinely useful "
            "facts (names, numbers, decisions, preferences). Skip facts that "
            "are obvious or temporary.\n\n"
            f"User asked: {user_query}\n"
            f"Assistant answered: {agent_response[:1000]}\n\n"
            "Output one fact per line. If nothing is worth remembering, "
            "output the single word: NONE"
        )

        try:
            llm = get_llm()
            result = llm.invoke(prompt)
            raw = str(result.content).strip()

            if raw.upper() == "NONE" or not raw:
                return 0

            stored = 0
            for line in raw.splitlines():
                line = line.strip().lstrip("•-–*123456789. ")
                if not line or len(line) < 10:
                    continue
                fact_id = hashlib.sha256(
                    f"{session_id}:{line}".encode()
                ).hexdigest()[:16]
                self.store(EpisodicFact(
                    fact_id    = fact_id,
                    text       = line,
                    session_id = session_id,
                    query      = user_query,
                    importance = importance,
                ))
                stored += 1

            if stored:
                logger.debug("Stored %d episodic facts from session '%s'.", stored, session_id)
            return stored

        except Exception as exc:
            logger.warning("Fact extraction failed: %s", exc)
            return 0

    # ── Management ────────────────────────────────────────────────────────────

    def list_all(self) -> list[EpisodicFact]:
        """Return all stored facts (for inspection/debugging)."""
        self._ensure_ready()
        results = self._collection.get(include=["documents", "metadatas"])
        facts = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            facts.append(EpisodicFact(
                fact_id    = hashlib.sha256(text.encode()).hexdigest()[:16],
                text       = text,
                session_id = meta.get("session_id", ""),
                query      = meta.get("query", ""),
                timestamp  = float(meta.get("timestamp", 0)),
                importance = float(meta.get("importance", 0.5)),
            ))
        return sorted(facts, key=lambda f: f.timestamp, reverse=True)

    def clear_session(self, session_id: str) -> int:
        """Remove all facts from a specific session."""
        self._ensure_ready()
        results = self._collection.get(
            where={"session_id": {"$eq": session_id}},
            include=[],
        )
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Cleared %d episodic facts for session '%s'.", len(ids), session_id)
        return len(ids)

    def clear_all(self) -> int:
        """Remove all facts. Returns count removed."""
        self._ensure_ready()
        n = self._collection.count()
        # ChromaDB: delete all by getting all ids first
        if n > 0:
            all_ids = self._collection.get(include=[])["ids"]
            self._collection.delete(ids=all_ids)
        return n

    @property
    def count(self) -> int:
        self._ensure_ready()
        return self._collection.count()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prune_oldest(self, n: int) -> None:
        """Remove the n oldest facts by timestamp."""
        results = self._collection.get(include=["metadatas"])
        pairs = list(zip(results["ids"], results["metadatas"]))
        pairs.sort(key=lambda x: float(x[1].get("timestamp", 0)))
        to_delete = [pid for pid, _ in pairs[:n]]
        if to_delete:
            self._collection.delete(ids=to_delete)
            logger.debug("Pruned %d oldest episodic facts.", len(to_delete))
