"""
document_processing/document_manager.py
─────────────────────────────────────────
High-level façade that orchestrates:
  1. Parsing  (DoclingProcessor)
  2. Storage  (VectorStore)
  3. Querying (semantic search + full-document retrieval)

This is the single entry point that agents should use for all
document-related operations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .docling_processor import DoclingProcessor, DocumentChunk, SUPPORTED_SUFFIXES
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Manages the full lifecycle of documents in the assistant.

    Usage::

        dm = DocumentManager()
        dm.ingest("reports/Q3.pdf")
        results = dm.search("What was the revenue in Q3?")
        for r in results:
            print(r.reference)
            print(r.chunk.text)
    """

    def __init__(self) -> None:
        self._processor = DoclingProcessor()
        self._store = VectorStore()

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, path: str | Path) -> int:
        """
        Parse and store a document.

        Args:
            path: Path to the document (PDF, DOCX, XLSX, PPTX).

        Returns:
            Number of new chunks added to the vector store.

        Raises:
            FileNotFoundError: if the file doesn't exist.
            ValueError: if the file type is unsupported.
        """
        path = Path(path)
        logger.info("Ingesting document: %s", path.name)
        chunks = self._processor.process(path)
        added = self._store.ingest(chunks)
        logger.info("Document '%s' ingested. Chunks added: %d", path.name, added)
        return added

    def ingest_directory(self, directory: str | Path) -> dict[str, int]:
        """
        Recursively ingest all supported documents in a directory.

        Returns:
            Dict mapping filename → number of new chunks added.
        """
        directory = Path(directory)
        results: dict[str, int] = {}
        for file in sorted(directory.rglob("*")):
            if file.suffix.lower() in SUPPORTED_SUFFIXES:
                try:
                    results[file.name] = self.ingest(file)
                except Exception as exc:
                    logger.error("Failed to ingest '%s': %s", file.name, exc)
                    results[file.name] = -1
        return results

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        doc_title: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search across all ingested documents.

        Args:
            query:     Natural-language query.
            k:         Maximum number of results.
            doc_title: Restrict search to a single document (by title).

        Returns:
            List of ``SearchResult`` objects sorted by relevance.
        """
        return self._store.search(query, k=k, doc_filter=doc_title)

    def format_search_results(self, results: list[SearchResult]) -> str:
        """
        Format search results as a readable string suitable for injection
        into an LLM prompt.
        """
        if not results:
            return "No relevant document sections found."

        lines = ["### Relevant Document Sections\n"]
        for i, result in enumerate(results, start=1):
            lines.append(
                f"**[{i}] {result.reference}** (relevance: {result.score:.2f})\n"
                f"{result.chunk.text}\n"
            )
        return "\n".join(lines)

    # ── Document-level operations ────────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        """Return all ingested documents with their metadata."""
        return self._store.list_documents()

    def get_document(self, doc_title: str) -> list[DocumentChunk]:
        """Retrieve all chunks of a specific document in reading order."""
        return self._store.get_document_chunks(doc_title)

    def get_full_document_text(self, doc_title: str) -> str:
        """Return the full reconstructed text of a document."""
        chunks = self.get_document(doc_title)
        if not chunks:
            return f"Document '{doc_title}' not found in the knowledge base."
        return "\n\n".join(c.text for c in chunks)

    def delete_document(self, doc_title: str) -> int:
        """Remove a document from the knowledge base."""
        removed = self._store.delete_document(doc_title)
        logger.info("Removed document '%s' (%d chunks).", doc_title, removed)
        return removed

    # ── LangChain integration ────────────────────────────────────────────────

    def as_retriever(self, k: int = 5, doc_title: Optional[str] = None):
        """Return a LangChain retriever for use in chains/agents."""
        return self._store.as_langchain_retriever(k=k, doc_filter=doc_title)

    @property
    def total_chunks(self) -> int:
        return self._store.count
