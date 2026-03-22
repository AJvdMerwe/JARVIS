"""
document_processing/document_manager.py
─────────────────────────────────────────
High-level façade that orchestrates the full document lifecycle:

  1. Parsing    — DoclingProcessor converts files to DocumentChunks with
                  page numbers and section breadcrumbs.
  2. Storage    — VectorStore (ChromaDB) holds embeddings + metadata.
  3. Retrieval  — semantic search, document-level fetch, reference formatting.
  4. Management — list, delete, stats, export.

This is the single entry point that agents, tools, and the API should use
for all document-related operations.

Cross-cutting concerns wired in:
  • Structured logging — every operation emits a JSON-compatible log record.
  • Tool cache         — search results cached for CACHE_TTL seconds.
  • Graceful degradation — public methods return safe fallbacks rather than
                           propagating low-level exceptions to callers.

Usage::

    dm = DocumentManager()

    # Ingest
    dm.ingest("reports/Q3.pdf")
    dm.ingest_directory("./documents/")

    # Search
    hits = dm.search("What was Q3 revenue?", k=5)
    hits = dm.search("costs", k=3, doc_title="Q3 Report")
    print(dm.format_search_results(hits))

    # Document-level access
    docs   = dm.list_documents()
    text   = dm.get_full_document_text("Q3 Report")
    chunks = dm.get_document("Q3 Report")

    # Stats
    print(dm.stats())
    print(dm.format_stats())

    # LangChain integration
    retriever = dm.as_retriever(k=5)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .docling_processor import DoclingProcessor, DocumentChunk, SUPPORTED_SUFFIXES
from .vector_store import SearchResult, VectorStore
from config import settings

logger = logging.getLogger(__name__)

# Default cache TTL for search results (seconds). 0 = disabled.
_SEARCH_CACHE_TTL: float = 60.0

# Minimum cosine similarity for a result to be returned (0.0 = include all).
_DEFAULT_SIMILARITY_THRESHOLD: float = 0.0


# =============================================================================
#  Result models
# =============================================================================

@dataclass
class IngestResult:
    """
    Structured outcome of ingesting a single document.

    Attributes
    ----------
    filename : str
        Original filename (basename only, not full path).
    doc_title : str
        Human-readable title stored in the vector store.
    chunks_added : int
        New chunks written to the store (0 if already indexed).
    chunks_total : int
        Total chunks for this document after ingestion.
    elapsed_ms : float
        Wall-clock time for the parse + embed + store pipeline.
    error : str
        Non-empty if ingestion failed; empty string on success.
    """
    filename:     str
    doc_title:    str
    chunks_added: int   = 0
    chunks_total: int   = 0
    elapsed_ms:   float = 0.0
    error:        str   = ""
    metadata:     dict  = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True when the operation completed without error."""
        return not self.error

    @property
    def was_skipped(self) -> bool:
        """True when the document was already indexed (no new chunks added)."""
        return self.success and self.chunks_added == 0

    def __str__(self) -> str:
        if self.error:
            return f"FAILED  {self.filename}: {self.error}"
        if self.was_skipped:
            return f"SKIPPED {self.filename} (already indexed, {self.chunks_total} chunks)"
        return (
            f"OK      {self.filename} "
            f"(+{self.chunks_added} new, {self.chunks_total} total, "
            f"{self.elapsed_ms:.0f}ms)"
        )


@dataclass
class KnowledgeBaseStats:
    """
    Snapshot of the knowledge-base state at a point in time.

    Attributes
    ----------
    document_count : int
        Number of distinct documents in the store.
    total_chunks : int
        Total indexed text chunks across all documents.
    store_path : str
        Filesystem path of the ChromaDB persistence directory.
    documents : list[dict]
        Per-document metadata: doc_title, doc_path, doctype.
    """
    document_count: int
    total_chunks:   int
    store_path:     str
    documents:      list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.document_count} document(s), "
            f"{self.total_chunks} chunks | {self.store_path}"
        )


# =============================================================================
#  DocumentManager
# =============================================================================

class DocumentManager:
    """
    Manages the full lifecycle of documents in the assistant.

    Parameters
    ----------
    chunk_size : int
        Maximum character length of each text chunk.
        Defaults to ``settings.chunk_size`` (512).
    chunk_overlap : int
        Overlap in characters between adjacent chunks.
        Defaults to ``settings.chunk_overlap`` (64).
    similarity_threshold : float
        Minimum cosine similarity (0–1) for search results.
        Results below this score are silently filtered out.
    cache_ttl : float
        Seconds to cache identical search queries. Pass 0.0 to disable.
    """

    def __init__(
        self,
        chunk_size:           int   = settings.chunk_size,
        chunk_overlap:        int   = settings.chunk_overlap,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        cache_ttl:            float = _SEARCH_CACHE_TTL,
    ) -> None:
        self._processor            = DoclingProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._store                = VectorStore()
        self._similarity_threshold = similarity_threshold
        self._cache_ttl            = cache_ttl

    # =========================================================================
    #  Ingestion
    # =========================================================================

    def ingest(self, path: str | Path) -> int:
        """
        Parse a document and store its chunks in the knowledge base.

        The operation is idempotent — chunks that already exist (matched by
        SHA-256 ``chunk_id``) are skipped, so re-ingesting an unchanged file
        adds zero chunks and returns 0.

        Parameters
        ----------
        path : str | Path
            Path to the document. Supported:
            ``.pdf``, ``.docx``, ``.xlsx``, ``.xls``, ``.pptx``, ``.ppt``

        Returns
        -------
        int
            Number of *new* chunks added to the vector store.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist on disk.
        ValueError
            If the file extension is not in SUPPORTED_SUFFIXES.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )

        logger.info("Ingesting '%s'…", path.name)
        t0 = time.monotonic()

        chunks = self._processor.process(path)
        added  = self._store.ingest(chunks)

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "Ingested '%s': %d new chunk(s) in %.0fms (doc total: %d).",
            path.name, added, elapsed, len(chunks),
        )
        return added

    def ingest_with_result(self, path: str | Path) -> IngestResult:
        """
        Like :meth:`ingest` but returns an :class:`IngestResult` instead of
        raising on failure — ideal for batch workflows.

        Parameters
        ----------
        path : str | Path
            Path to the document.

        Returns
        -------
        IngestResult
            Structured outcome with success flag, chunk counts, and timing.
        """
        path     = Path(path)
        filename = path.name
        t0       = time.monotonic()

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"Unsupported type: {path.suffix}")

            chunks    = self._processor.process(path)
            added     = self._store.ingest(chunks)
            elapsed   = (time.monotonic() - t0) * 1000
            doc_title = chunks[0].doc_title if chunks else path.stem.title()

            return IngestResult(
                filename=filename,
                doc_title=doc_title,
                chunks_added=added,
                chunks_total=len(chunks),
                elapsed_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            logger.error("Ingest failed for '%s': %s", filename, exc)
            return IngestResult(
                filename=filename,
                doc_title=path.stem.title(),
                elapsed_ms=elapsed,
                error=str(exc),
            )

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> dict[str, IngestResult]:
        """
        Ingest all supported documents found in a directory.

        Parameters
        ----------
        directory : str | Path
            Root directory to scan.
        recursive : bool
            Descend into subdirectories when True (default).

        Returns
        -------
        dict[str, IngestResult]
            Maps filename → :class:`IngestResult` for every file found.
            Returns an empty dict if no supported files are discovered.

        Raises
        ------
        NotADirectoryError
            If ``directory`` is not a directory.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        glob  = directory.rglob("*") if recursive else directory.glob("*")
        files = sorted(f for f in glob if f.suffix.lower() in SUPPORTED_SUFFIXES)

        if not files:
            logger.info("No supported documents found in '%s'.", directory)
            return {}

        logger.info(
            "Batch ingesting %d file(s) from '%s'…", len(files), directory
        )
        results: dict[str, IngestResult] = {}
        for f in files:
            result         = self.ingest_with_result(f)
            results[f.name] = result
            logger.debug("%s", result)

        ok      = sum(1 for r in results.values() if r.success)
        skipped = sum(1 for r in results.values() if r.was_skipped)
        failed  = sum(1 for r in results.values() if not r.success)
        logger.info(
            "Batch complete: %d OK (%d already indexed), %d failed.",
            ok, skipped, failed,
        )
        return results

    # =========================================================================
    #  Search
    # =========================================================================

    def search(
        self,
        query: str,
        k: int = 5,
        doc_title: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[SearchResult]:
        """
        Semantic search across the knowledge base.

        Identical queries are served from an in-process cache (configurable
        TTL) to avoid repeated ChromaDB round-trips within a session.

        Parameters
        ----------
        query : str
            Natural-language question or search phrase.
        k : int
            Maximum number of results to return.
        doc_title : str, optional
            Restrict the search to a single document by its exact title.
        similarity_threshold : float, optional
            Per-call similarity floor, overriding the instance default.

        Returns
        -------
        list[SearchResult]
            Ordered by descending cosine similarity. Empty when the KB is
            empty or no chunks meet the similarity threshold.
        """
        if not query.strip():
            return []

        if self.total_chunks == 0:
            logger.debug("Knowledge base is empty — skipping search.")
            return []

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )

        # ── Cache lookup ──────────────────────────────────────────────────────
        cache     = self._get_cache()
        cache_key = None
        if cache and self._cache_ttl > 0:
            cache_key = cache.make_key(
                "dm_search", query, k, doc_title or "", threshold
            )
            cached = cache.get(cache_key, ttl=self._cache_ttl)
            if cached is not None:
                import json
                try:
                    results = self._deserialise_results(json.loads(cached))
                    logger.debug(
                        "Search cache HIT  '%s…' (%d results).",
                        query[:40], len(results),
                    )
                    return results
                except Exception:
                    pass  # fall through to live search

        # ── Live vector search ────────────────────────────────────────────────
        t0      = time.monotonic()
        results = self._store.search(query, k=k, doc_filter=doc_title)
        elapsed = (time.monotonic() - t0) * 1000

        # Apply similarity threshold.
        if threshold > 0.0:
            before   = len(results)
            results  = [r for r in results if r.score >= threshold]
            filtered = before - len(results)
            if filtered:
                logger.debug(
                    "Filtered %d result(s) below threshold %.2f.",
                    filtered, threshold,
                )

        logger.debug(
            "Search '%s…' → %d result(s) in %.0fms%s.",
            query[:40], len(results), elapsed,
            f" (doc='{doc_title}')" if doc_title else "",
        )

        # ── Cache store ───────────────────────────────────────────────────────
        if cache and cache_key and self._cache_ttl > 0 and results:
            import json
            try:
                cache.set(cache_key, json.dumps(self._serialise_results(results)))
            except Exception:
                pass

        return results

    def format_search_results(
        self,
        results: list[SearchResult],
        max_chars_per_chunk: int = 500,
    ) -> str:
        """
        Format search results as Markdown suitable for LLM prompt injection.

        Parameters
        ----------
        results : list[SearchResult]
            Ordered search hits, typically from :meth:`search`.
        max_chars_per_chunk : int
            Truncate individual chunk text at this length.

        Returns
        -------
        str
            Formatted Markdown string, or a "no results" notice.
        """
        if not results:
            return "No relevant document sections found."

        lines = ["### Relevant Document Sections\n"]
        for i, result in enumerate(results, start=1):
            text = result.chunk.text
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk].rstrip() + "…"

            lines.append(
                f"**[{i}] {result.reference}** "
                f"(relevance: {result.score:.2f})\n"
                f"{text}\n"
            )
        return "\n".join(lines)

    # =========================================================================
    #  Document-level operations
    # =========================================================================

    def list_documents(self) -> list[dict]:
        """
        Return metadata for every document in the knowledge base.

        Returns
        -------
        list[dict]
            Each dict contains: ``doc_title``, ``doc_path``, ``doctype``.
        """
        return self._store.list_documents()

    def get_document(self, doc_title: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks of a document in reading order.

        Chunks are sorted by page number then character offset so the
        concatenated result reads naturally.

        Parameters
        ----------
        doc_title : str
            Exact document title (case-sensitive), as from :meth:`list_documents`.

        Returns
        -------
        list[DocumentChunk]
            Ordered chunks, or an empty list if the document is not found.
        """
        return self._store.get_document_chunks(doc_title)

    def get_full_document_text(self, doc_title: str) -> str:
        """
        Reconstruct the full text of a document by joining chunks in order.

        Parameters
        ----------
        doc_title : str
            Exact document title.

        Returns
        -------
        str
            Concatenated text separated by double newlines, or an informative
            message if the document is not found.
        """
        chunks = self.get_document(doc_title)
        if not chunks:
            available = ", ".join(
                f"'{d['doc_title']}'" for d in self.list_documents()
            ) or "none"
            return (
                f"Document '{doc_title}' was not found in the knowledge base. "
                f"Available documents: {available}."
            )
        return "\n\n".join(c.text for c in chunks)

    def document_exists(self, doc_title: str) -> bool:
        """
        Return True if the document is in the knowledge base.

        Parameters
        ----------
        doc_title : str
            Exact document title (case-sensitive).
        """
        return any(d["doc_title"] == doc_title for d in self.list_documents())


    def ingest_or_update(self, path: str | Path) -> IngestResult:
        """
        Ingest a document or replace it if a newer version is detected.

        Unlike :meth:`ingest` (which is idempotent and skips existing chunks),
        this method checks whether the file content has changed by comparing
        its SHA-256 hash against the hash stored in the existing chunks'
        metadata.

        Behaviour
        ---------
        • If the document is **not** in the KB → ingest normally.
        • If the document **is** in the KB with the **same** content hash
          → skip (no change).
        • If the document **is** in the KB with a **different** content hash
          → delete all old chunks, then ingest the new version.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        IngestResult
            ``replaced=True`` in ``metadata`` when an old version was deleted.
        """
        import hashlib
        path = Path(path)

        # Compute hash of the new file
        try:
            raw  = path.read_bytes()
            new_hash = hashlib.sha256(raw).hexdigest()
        except Exception as exc:
            return IngestResult(
                filename=path.name,
                doc_title=path.stem.title(),
                error=f"Cannot read file: {exc}",
            )

        # Look up any existing chunks for this doc title
        doc_title = path.stem.replace("_", " ").replace("-", " ").title()
        existing  = self._store.get_document_chunks(doc_title)

        if existing:
            old_hash = existing[0].metadata.get("content_hash", "")
            if old_hash and old_hash == new_hash:
                logger.info(
                    "ingest_or_update: '%s' unchanged (hash match) — skipping.",
                    path.name,
                )
                return IngestResult(
                    filename=path.name,
                    doc_title=doc_title,
                    chunks_added=0,
                    chunks_total=len(existing),
                    metadata={"replaced": False, "reason": "unchanged"},
                )

            # Content changed — replace
            removed = self.delete_document(doc_title)
            logger.info(
                "ingest_or_update: '%s' changed — removed %d old chunk(s), re-ingesting.",
                path.name, removed,
            )
            result = self.ingest_with_result(path)
            result.metadata = result.metadata or {}
            result.metadata["replaced"]      = True
            result.metadata["chunks_removed"] = removed
            result.metadata["content_hash"]   = new_hash
            return result

        # Not in KB yet — fresh ingest, store hash in chunk metadata
        chunks = self._processor.process(path)
        for chunk in chunks:
            chunk.metadata["content_hash"] = new_hash

        if not chunks:
            return IngestResult(filename=path.name, doc_title=doc_title)

        added = self._store.ingest(chunks)
        self._invalidate_search_cache()
        logger.info(
            "ingest_or_update: new document '%s' — %d chunks added.", path.name, added
        )
        return IngestResult(
            filename=path.name,
            doc_title=doc_title,
            chunks_added=added,
            chunks_total=len(chunks),
            metadata={"replaced": False, "content_hash": new_hash},
        )


    def search_multi_doc(
        self,
        query:       str,
        doc_titles:  list[str],
        k_per_doc:   int   = 3,
        similarity_threshold: float | None = None,
    ) -> dict[str, list]:
        """
        Search across multiple documents simultaneously, returning results
        grouped by document title.

        Useful for comparison queries like "compare the Q2 and Q3 reports".
        Each document contributes up to *k_per_doc* chunks; results within
        each document are ranked by cosine similarity.

        Parameters
        ----------
        query       : str
            Natural-language question or search phrase.
        doc_titles  : list[str]
            Exact document titles to query.
        k_per_doc   : int
            Maximum results per document (default 3).
        similarity_threshold : float, optional
            Per-call similarity floor.

        Returns
        -------
        dict[str, list[SearchResult]]
            ``{doc_title: [SearchResult, …]}`` in the order *doc_titles* was given.
            Documents with no relevant results have an empty list.
        """
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )
        results: dict[str, list] = {}
        for title in doc_titles:
            doc_results = self._store.search(query, k=k_per_doc, doc_filter=title)
            if threshold > 0.0:
                doc_results = [r for r in doc_results if r.score >= threshold]
            results[title] = doc_results
        return results

    def compare_documents(
        self,
        query:      str,
        doc_titles: list[str],
        k_per_doc:  int = 3,
    ) -> str:
        """
        Retrieve relevant chunks from each document and format them as a
        side-by-side comparison block for LLM prompt injection.

        Produces::

            ## Comparison: Q2 Report vs Q3 Report

            ### Q2 Report
            [1] Q2 Report › Page 4 › Revenue (score: 0.88)
            Revenue for Q2 was $3.9B…

            ### Q3 Report
            [1] Q3 Report › Page 4 › Revenue (score: 0.91)
            Revenue for Q3 was $4.2B…

        Parameters
        ----------
        query      : str
        doc_titles : list[str]
        k_per_doc  : int

        Returns
        -------
        str
            Formatted Markdown comparison block, or a "not found" notice.
        """
        if len(doc_titles) < 2:
            return self.format_search_results(
                self.search(query, k=k_per_doc * 2)
            )

        multi = self.search_multi_doc(query, doc_titles, k_per_doc=k_per_doc)
        title_str = " vs ".join(doc_titles)
        lines     = [f"## Comparison: {title_str}\n"]

        any_results = False
        for doc_title, results in multi.items():
            lines.append(f"### {doc_title}")
            if not results:
                lines.append("_(No relevant sections found.)_\n")
                continue
            any_results = True
            for i, r in enumerate(results, 1):
                text = r.chunk.text[:500]
                if len(r.chunk.text) > 500:
                    text += "…"
                lines.append(
                    f"**[{i}] {r.reference}** (relevance: {r.score:.2f})\n{text}\n"
                )

        if not any_results:
            return f"No relevant sections found in: {', '.join(doc_titles)}"
        return "\n".join(lines)


    def search_with_tables(
        self,
        query:               str,
        k:                   int   = 5,
        doc_title:           str   | None = None,
        similarity_threshold: float | None = None,
    ) -> tuple[list, list]:
        """
        Semantic search that separates table chunks from prose chunks.

        Since chunks from the Markdown pipeline that contain pipe-table
        syntax (``|``) carry structured data, they are returned in a
        dedicated list so callers can format them differently (e.g. render
        as a proper table rather than flowing text).

        Parameters
        ----------
        query               : str
        k                   : int
        doc_title           : str, optional
        similarity_threshold : float, optional

        Returns
        -------
        tuple[list[SearchResult], list[SearchResult]]
            (table_results, prose_results)  — both ordered by cosine similarity.
        """
        all_results = self.search(
            query, k=k * 2,   # fetch extra to compensate for split
            doc_title=doc_title,
            similarity_threshold=similarity_threshold,
        )

        table_results = []
        prose_results = []
        for r in all_results:
            text = r.chunk.text
            # A chunk is "table-like" if it contains at least two pipe-table rows
            pipe_lines = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
            if len(pipe_lines) >= 2:
                table_results.append(r)
            else:
                prose_results.append(r)

        # Trim to k total, preferring table results
        return table_results[:k], prose_results[:max(0, k - len(table_results[:k]))]

    def format_table_results(
        self,
        table_results: list,
        prose_results: list,
        max_chars_per_chunk: int = 800,
    ) -> str:
        """
        Format results from :meth:`search_with_tables` as Markdown.

        Table chunks are placed in a dedicated "Data Tables" section;
        prose chunks follow in the standard "Document Sections" section.

        Parameters
        ----------
        table_results : list[SearchResult]
        prose_results : list[SearchResult]
        max_chars_per_chunk : int

        Returns
        -------
        str  Formatted Markdown string.
        """
        lines: list[str] = []

        if table_results:
            lines.append("### 📊 Data Tables\n")
            for i, r in enumerate(table_results, 1):
                text = r.chunk.text
                if len(text) > max_chars_per_chunk:
                    text = text[:max_chars_per_chunk].rstrip() + "…"
                lines.append(
                    f"**[T{i}] {r.reference}** "
                    f"(relevance: {r.score:.2f})\n\n{text}\n"
                )

        if prose_results:
            lines.append("### 📄 Relevant Sections\n")
            for i, r in enumerate(prose_results, 1):
                text = r.chunk.text
                if len(text) > max_chars_per_chunk:
                    text = text[:max_chars_per_chunk].rstrip() + "…"
                lines.append(
                    f"**[{i}] {r.reference}** "
                    f"(relevance: {r.score:.2f})\n\n{text}\n"
                )

        if not lines:
            return "No relevant document sections found."
        return "\n".join(lines)

    def delete_document(self, doc_title: str) -> int:
        """
        Remove a document and all its chunks from the knowledge base.

        Parameters
        ----------
        doc_title : str
            Exact document title to remove.

        Returns
        -------
        int
            Number of chunks removed. Returns 0 if the document was not found.
        """
        removed = self._store.delete_document(doc_title)
        if removed:
            logger.info(
                "Deleted '%s' (%d chunk(s) removed).", doc_title, removed
            )
            self._invalidate_search_cache()
        else:
            logger.info("Delete: '%s' not found.", doc_title)
        return removed

    # =========================================================================
    #  Stats and observability
    # =========================================================================

    def stats(self) -> KnowledgeBaseStats:
        """
        Return a :class:`KnowledgeBaseStats` snapshot.

        Returns
        -------
        KnowledgeBaseStats
            Document count, total chunks, store path, and document list.
        """
        docs = self.list_documents()
        return KnowledgeBaseStats(
            document_count=len(docs),
            total_chunks=self.total_chunks,
            store_path=str(self._store._persist_dir),
            documents=docs,
        )

    def format_stats(self) -> str:
        """Return a human-readable multi-line stats string."""
        s = self.stats()
        if s.document_count == 0:
            return "Knowledge base is empty."
        lines = [
            f"Knowledge Base: {s.document_count} document(s), "
            f"{s.total_chunks} total chunks",
            f"Store: {s.store_path}",
            "",
            "Documents:",
        ]
        for doc in s.documents:
            lines.append(
                f"  • {doc['doc_title']}  "
                f"({doc.get('doctype', '?').upper()})  "
                f"{doc.get('doc_path', '')}"
            )
        return "\n".join(lines)

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks across all documents."""
        return self._store.count

    # =========================================================================
    #  LangChain integration
    # =========================================================================

    def as_retriever(
        self,
        k: int = 5,
        doc_title: Optional[str] = None,
    ):
        """
        Return a LangChain ``BaseRetriever`` backed by this vector store.

        Plugs directly into RetrievalQA chains, LCEL pipelines, and any
        other LangChain construct that accepts a retriever.

        Parameters
        ----------
        k : int
            Maximum documents to retrieve per query.
        doc_title : str, optional
            Scope retrieval to a single document.

        Returns
        -------
        BaseRetriever
        """
        return self._store.as_langchain_retriever(k=k, doc_filter=doc_title)

    # =========================================================================
    #  Internal helpers
    # =========================================================================

    @staticmethod
    def _get_cache():
        """Return the shared tool cache, or None if unavailable."""
        try:
            from core.cache import get_cache
            return get_cache()
        except Exception:
            return None

    def _invalidate_search_cache(self) -> None:
        """Evict expired cache entries (conservative purge after deletes)."""
        cache = self._get_cache()
        if cache:
            try:
                cache.purge_expired()
            except Exception:
                pass

    @staticmethod
    def _serialise_results(results: list[SearchResult]) -> list[dict]:
        """Convert SearchResult objects to JSON-safe dicts for cache storage."""
        return [
            {
                "score":        r.score,
                "chunk_id":     r.chunk.chunk_id,
                "text":         r.chunk.text,
                "doc_path":     r.chunk.doc_path,
                "doc_title":    r.chunk.doc_title,
                "page_number":  r.chunk.page_number,
                "section_path": r.chunk.section_path,
                "char_offset":  r.chunk.char_offset,
                "metadata":     r.chunk.metadata,
            }
            for r in results
        ]

    @staticmethod
    def _deserialise_results(raw: list[dict]) -> list[SearchResult]:
        """Restore SearchResult objects from their cached JSON representation."""
        results = []
        for d in raw:
            chunk = DocumentChunk(
                chunk_id    =d["chunk_id"],
                text        =d["text"],
                doc_path    =d["doc_path"],
                doc_title   =d["doc_title"],
                page_number =d.get("page_number"),
                section_path=d.get("section_path", []),
                char_offset =d.get("char_offset", 0),
                metadata    =d.get("metadata", {}),
            )
            results.append(SearchResult(chunk=chunk, score=d["score"]))
        return results
