"""
document_processing/vector_store.py
─────────────────────────────────────
ChromaDB-backed vector store that keeps rich source references.

Features:
  • Ingest ``DocumentChunk`` objects into a persistent Chroma collection.
  • Similarity search returning chunks AND human-readable references.
  • "Cite-by-document" — retrieve all chunks from a specific document.
  • Deduplication by ``chunk_id`` so re-ingesting a file is idempotent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import settings
from .docling_processor import DocumentChunk

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Return types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search hit from the vector store."""

    chunk: DocumentChunk
    score: float                # cosine similarity 0..1 (higher = more similar)

    @property
    def reference(self) -> str:
        return self.chunk.reference

    def __str__(self) -> str:
        return (
            f"[{self.score:.3f}] {self.reference}\n"
            f"{self.chunk.text[:200]}…"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  VectorStore
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Persistent ChromaDB-backed vector store.

    All mutating operations are synchronous and safe to call from any thread
    (Chroma's sqlite backend serialises writes).
    """

    def __init__(
        self,
        persist_directory: str | Path = settings.vector_store_path,
        collection_name: str = settings.vector_store_collection,
    ) -> None:
        self._persist_dir = Path(persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._embeddings = None

    # ── Lazy initialisation ─────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        if self._collection is not None:
            return
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError("chromadb is not installed. Run: pip install chromadb") from exc

        from core.llm_manager import get_embeddings

        self._embeddings = get_embeddings()
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready | collection='%s' | path=%s | docs=%d",
            self._collection_name,
            self._persist_dir,
            self._collection.count(),
        )

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, chunks: list[DocumentChunk], batch_size: int | None = None) -> int:
        """
        Add chunks to the collection. Skips chunks already present
        (idempotent based on ``chunk_id``).

        Parameters
        ----------
        chunks : list[DocumentChunk]
        batch_size : int, optional
            Override the default ``settings.embedding_batch_size``.
            Smaller values reduce peak memory; larger values improve throughput.

        Returns
        -------
        int
            Number of *new* chunks actually added.
        """
        from config import settings as _s
        self._ensure_ready()

        effective_batch = batch_size if batch_size is not None else _s.embedding_batch_size

        # Find which chunk_ids are new
        existing_ids: set[str] = set()
        try:
            existing = self._collection.get(ids=[c.chunk_id for c in chunks])
            existing_ids = set(existing["ids"])
        except Exception:
            pass

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All %d chunks already in store. Nothing to add.", len(chunks))
            return 0

        added = 0
        for start in range(0, len(new_chunks), effective_batch):
            batch = new_chunks[start : start + effective_batch]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metas = [
                {
                    "doc_path": c.doc_path,
                    "doc_title": c.doc_title,
                    "page_number": c.page_number or 0,
                    "section_path": " › ".join(c.section_path),
                    "reference": c.reference,
                    "char_offset": c.char_offset,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in batch
            ]
            embeddings = self._embeddings.embed_documents(texts)
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metas,
            )
            added += len(batch)

        logger.info("Ingested %d new chunks (skipped %d duplicates).", added, len(chunks) - added)
        return added

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        doc_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic similarity search.

        Args:
            query:      Natural-language query string.
            k:          Number of results to return.
            doc_filter: If given, restrict results to this doc_title.

        Returns:
            List of ``SearchResult`` sorted by descending similarity.
        """
        self._ensure_ready()

        where: Optional[dict] = None
        if doc_filter:
            where = {"doc_title": {"$eq": doc_filter}}

        query_embedding = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, max(1, self._collection.count())),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - float(distance)  # cosine distance → similarity
            chunk = DocumentChunk(
                chunk_id=meta.get("chunk_id", ""),
                text=text,
                doc_path=meta.get("doc_path", ""),
                doc_title=meta.get("doc_title", ""),
                page_number=int(meta.get("page_number", 0)) or None,
                section_path=(
                    meta.get("section_path", "").split(" › ")
                    if meta.get("section_path")
                    else []
                ),
                metadata={k: v for k, v in meta.items()
                          if k not in {"doc_path", "doc_title", "page_number",
                                       "section_path", "reference", "char_offset"}},
            )
            search_results.append(SearchResult(chunk=chunk, score=score))

        return sorted(search_results, key=lambda r: r.score, reverse=True)

    # ── Document-level helpers ───────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """Return a deduplicated list of ingested documents with metadata."""
        self._ensure_ready()
        all_metas = self._collection.get(include=["metadatas"])["metadatas"]
        seen: dict[str, dict] = {}
        for m in all_metas:
            title = m.get("doc_title", "")
            if title and title not in seen:
                seen[title] = {
                    "doc_title": title,
                    "doc_path": m.get("doc_path", ""),
                    "doctype": m.get("doctype", ""),
                }
        return list(seen.values())

    def get_document_chunks(self, doc_title: str) -> list[DocumentChunk]:
        """Retrieve ALL chunks belonging to a specific document."""
        self._ensure_ready()
        results = self._collection.get(
            where={"doc_title": {"$eq": doc_title}},
            include=["documents", "metadatas"],
        )
        chunks = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            chunks.append(
                DocumentChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=text,
                    doc_path=meta.get("doc_path", ""),
                    doc_title=meta.get("doc_title", ""),
                    page_number=int(meta.get("page_number", 0)) or None,
                    section_path=(
                        meta.get("section_path", "").split(" › ")
                        if meta.get("section_path")
                        else []
                    ),
                    char_offset=int(meta.get("char_offset", 0)),
                )
            )
        return sorted(chunks, key=lambda c: (c.page_number or 0, c.char_offset))

    def delete_document(self, doc_title: str) -> int:
        """Remove all chunks belonging to a document. Returns count removed."""
        self._ensure_ready()
        existing = self._collection.get(
            where={"doc_title": {"$eq": doc_title}},
            include=[],
        )
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Deleted %d chunks for document '%s'.", len(ids), doc_title)
        return len(ids)

    @property
    def count(self) -> int:
        self._ensure_ready()
        return self._collection.count()

    # ── LangChain retriever ──────────────────────────────────────────────────

    def as_langchain_retriever(self, k: int = 5, doc_filter: Optional[str] = None):
        """
        Return a LangChain-compatible retriever wrapping this store.
        Agents can plug this directly into RetrievalQA chains.
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

        store_ref = self
        _k = k
        _doc_filter = doc_filter

        class _StoreRetriever(BaseRetriever):
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,  # noqa: ARG002
            ) -> list[Document]:
                results = store_ref.search(query, k=_k, doc_filter=_doc_filter)
                return [r.chunk.to_langchain_doc() for r in results]

        return _StoreRetriever()
