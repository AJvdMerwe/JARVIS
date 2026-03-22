"""
document_processing/mass_uploader.py
──────────────────────────────────────
Mass document upload and RAG ingestion pipeline.

Orchestrates bulk ingestion of documents from files or directories,
applying type-appropriate extraction, chunking, and embedding, then
storing results in the vector store for retrieval-augmented generation.

Concurrency model — split-lane processing
──────────────────────────────────────────
  DOCLING / FALLBACK lane  (PDF, DOCX, XLSX, PPTX)
    Processed **sequentially** in document order.

    Why: pypdfium2 (used by Docling for PDF rendering) is NOT thread-safe.
    Its global C library state is destroyed when the first worker thread
    exits, causing the error:
        "Cannot close object; pdfium library is destroyed.
         This may cause a memory leak."
    Running these files one at a time eliminates the race condition while
    keeping the pipeline deterministic and leak-free.

    Each file gets its own fresh DoclingProcessor / DocumentConverter
    instance that is fully garbage-collected before the next file starts,
    ensuring complete pdfium resource cleanup.

  TEXT / STRUCTURED lane  (TXT, MD, CSV, HTML, JSON, XML)
    Processed **concurrently** via a ThreadPoolExecutor.

    These formats use only pure-Python readers (no pdfium, no C extensions),
    so parallel execution is safe and significantly speeds up large batches.

Capacity planning
─────────────────
  max_workers is applied to the text/structured lane only.
  The Docling lane always uses 1 worker (sequential).

  Recommended settings:
    max_workers=1  — minimal VRAM / RAM (default safe mode)
    max_workers=4  — good for CPU-bound text batches on 4+ cores
    max_workers=8  — high-throughput text-only ingestion

Other features
──────────────
  • Auto-detects document type and selects the optimal extraction strategy.
  • Content-hash (SHA-256) deduplication — re-uploading the same file is a no-op.
  • Per-file and batch-level progress callbacks (hook-friendly for UIs).
  • Rich UploadReport with per-file outcomes, timing, and summary stats.
  • Graceful error isolation — one bad file never aborts the whole batch.
  • Optional dry-run mode — detects and plans without writing to the store.

Usage::

    from document_processing.mass_uploader import MassUploader

    uploader = MassUploader(max_workers=4)

    # Ingest a whole directory
    report = uploader.upload_directory("./documents/")
    print(report.summary())

    # Ingest a specific list of files
    report = uploader.upload_files([
        "reports/Q3.pdf",
        "data/customers.xlsx",
        "notes/README.md",
    ])
    for outcome in report.outcomes:
        print(outcome)

    # RAG: search after ingestion
    results = uploader.search("What was Q3 revenue?", k=5)
    for r in results:
        print(r.reference, r.score)

    # Dry run — inspect without writing
    report = uploader.upload_directory("./docs/", dry_run=True)
    print(report.summary())
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .document_manager import DocumentManager, IngestResult
from .docling_processor import DoclingProcessor as _DoclingProcessor
from .type_detector import (
    DocumentType,
    DocumentTypeInfo,
    ExtractionStrategy,
    TypeDetector,
)
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


# =============================================================================
#  Result types
# =============================================================================

@dataclass
class FileOutcome:
    """
    Result of processing one file in a batch upload.

    Attributes
    ----------
    path : Path
        Absolute path to the file.
    doc_type : DocumentType
        Detected type.
    strategy : ExtractionStrategy
        Extraction strategy that was applied.
    status : str
        One of: ``ok`` · ``skipped`` · ``duplicate`` · ``unsupported`` · ``error``
    chunks_added : int
        New chunks written to the vector store (0 for skipped/duplicate).
    chunks_total : int
        Total chunks for this document in the store after processing.
    elapsed_ms : float
        Wall-clock time for the full process (detect + parse + embed).
    doc_title : str
        Human-readable title stored in the vector store.
    content_hash : str
        SHA-256 of the file content (for deduplication audit).
    error : str
        Non-empty when status == ``error``.
    warnings : list[str]
        Non-fatal warnings from detection (e.g. "needs OCR").
    """
    path:         Path
    doc_type:     DocumentType               = DocumentType.UNKNOWN
    strategy:     ExtractionStrategy         = ExtractionStrategy.SKIP
    status:       str                        = "error"
    chunks_added: int                        = 0
    chunks_total: int                        = 0
    elapsed_ms:   float                      = 0.0
    doc_title:    str                        = ""
    content_hash: str                        = ""
    error:        str                        = ""
    warnings:     list[str]                  = field(default_factory=list)

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def succeeded(self) -> bool:
        return self.status in ("ok", "skipped", "duplicate")

    def __str__(self) -> str:
        icons = {"ok": "✓", "skipped": "↷", "duplicate": "≡",
                 "unsupported": "✗", "error": "✗"}
        icon = icons.get(self.status, "?")
        detail = f"+{self.chunks_added} chunks" if self.status == "ok" else self.status
        if self.error:
            detail = self.error[:60]
        return (
            f"{icon}  {self.filename:<45} "
            f"{self.doc_type.value:<8} "
            f"{detail:<25} "
            f"{self.elapsed_ms:>7.0f}ms"
        )


@dataclass
class UploadReport:
    """
    Full report for a batch upload operation.

    Attributes
    ----------
    outcomes : list[FileOutcome]
        One entry per file processed.
    total_elapsed_ms : float
        Wall-clock time for the entire batch.
    is_dry_run : bool
        True when no data was written to the vector store.
    """
    outcomes:         list[FileOutcome] = field(default_factory=list)
    total_elapsed_ms: float             = 0.0
    is_dry_run:       bool              = False

    # ── Aggregate counts ──────────────────────────────────────────────────────

    @property
    def total_files(self) -> int:
        return len(self.outcomes)

    @property
    def ok_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "ok")

    @property
    def skipped_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped")

    @property
    def duplicate_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "duplicate")

    @property
    def error_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "error")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "unsupported")

    @property
    def total_chunks_added(self) -> int:
        return sum(o.chunks_added for o in self.outcomes)

    @property
    def failed_outcomes(self) -> list[FileOutcome]:
        return [o for o in self.outcomes if o.status == "error"]

    @property
    def successful_outcomes(self) -> list[FileOutcome]:
        return [o for o in self.outcomes if o.succeeded]

    def summary(self) -> str:
        """Human-readable one-page summary of the batch upload."""
        dry = "  [DRY RUN — nothing was written]\n" if self.is_dry_run else ""
        elapsed_s = self.total_elapsed_ms / 1000
        lines = [
            "═" * 70,
            f"  Mass Upload Report{' (DRY RUN)' if self.is_dry_run else ''}",
            "─" * 70,
            dry,
            f"  Total files processed : {self.total_files}",
            f"  Successfully ingested : {self.ok_count}",
            f"  Already indexed       : {self.duplicate_count}",
            f"  Empty (skipped)       : {self.skipped_count}",
            f"  Unsupported type      : {self.unsupported_count}",
            f"  Errors                : {self.error_count}",
            f"  New chunks added      : {self.total_chunks_added}",
            f"  Total time            : {elapsed_s:.1f}s",
        ]
        if self.ok_count and elapsed_s > 0:
            lines.append(
                f"  Throughput            : "
                f"{self.ok_count / elapsed_s:.1f} docs/s  |  "
                f"{self.total_chunks_added / elapsed_s:.0f} chunks/s"
            )
        if self.error_count:
            lines += ["", "  Errors:"]
            for o in self.failed_outcomes:
                lines.append(f"    ✗ {o.filename}: {o.error[:70]}")
        lines += ["", "  Per-file breakdown:"]
        for o in self.outcomes:
            lines.append(f"    {o}")
        lines.append("═" * 70)
        return "\n".join(line for line in lines if line is not None)


# =============================================================================
#  MassUploader
# =============================================================================

class MassUploader:
    """
    Concurrent mass document upload and RAG ingestion pipeline.

    Parameters
    ----------
    document_manager : DocumentManager, optional
        Shared DocumentManager instance. A new one is created if not provided.
    max_workers : int
        Maximum concurrent worker threads for parsing + embedding.
        Default 4 — good balance for I/O-bound document processing.
    chunk_size : int
        Text chunk size (chars) passed to the processor.
    chunk_overlap : int
        Overlap between adjacent chunks.
    dedup_hashes : set[str], optional
        Pre-populated set of known content hashes to skip.
        Pass an empty set to disable session-level deduplication.
        Pass None (default) to use an instance-level dedup cache.
    on_progress : callable, optional
        Called after each file with ``(outcome: FileOutcome, done: int, total: int)``.
        Useful for progress bars / streaming status updates.
    """

    def __init__(
        self,
        document_manager: Optional[DocumentManager] = None,
        max_workers:      int                       = 4,
        chunk_size:       int                       = 512,
        chunk_overlap:    int                       = 64,
        dedup_hashes:     Optional[set[str]]        = None,
        on_progress:      Optional[Callable]        = None,
    ) -> None:
        self._dm          = document_manager or DocumentManager(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._detector    = TypeDetector(
            chunk_size=chunk_size,
            use_docling=self._docling_available(),
        )
        self._max_workers = max_workers
        self._on_progress = on_progress

        # Session-level deduplication: maps content_hash → doc_title
        self._seen_hashes: set[str] = dedup_hashes if dedup_hashes is not None else set()
        self._dedup_lock  = threading.Lock()   # serialise concurrent hash checks

    # =========================================================================
    #  Primary upload methods
    # =========================================================================

    def upload_files(
        self,
        paths:     list[str | Path],
        dry_run:   bool = False,
    ) -> UploadReport:
        """
        Ingest a list of files into the vector store.

        Parameters
        ----------
        paths : list
            File paths to process.
        dry_run : bool
            If True, detect and plan but do not write to the store.

        Returns
        -------
        UploadReport
        """
        t0      = time.monotonic()
        paths_p = [Path(p).resolve() for p in paths]

        logger.info(
            "MassUploader: starting batch of %d file(s)%s.",
            len(paths_p),
            " [DRY RUN]" if dry_run else "",
        )

        outcomes = self._process_batch(paths_p, dry_run=dry_run)

        elapsed_ms = (time.monotonic() - t0) * 1000
        report = UploadReport(
            outcomes=outcomes,
            total_elapsed_ms=elapsed_ms,
            is_dry_run=dry_run,
        )
        logger.info(
            "MassUploader: batch complete — %d ok, %d errors, %d duplicates "
            "in %.1fs.",
            report.ok_count, report.error_count,
            report.duplicate_count, elapsed_ms / 1000,
        )
        return report

    def upload_directory(
        self,
        directory:  str | Path,
        recursive:  bool        = True,
        dry_run:    bool        = False,
        glob_pattern: Optional[str] = None,
    ) -> UploadReport:
        """
        Recursively scan a directory and ingest all supported documents.

        Parameters
        ----------
        directory : str | Path
            Root directory to scan.
        recursive : bool
            Descend into sub-directories.
        dry_run : bool
            Detect and plan only; do not write to the store.
        glob_pattern : str, optional
            Optional glob pattern to filter files, e.g. ``"*.pdf"``.

        Returns
        -------
        UploadReport
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        if glob_pattern:
            glob_fn = directory.rglob if recursive else directory.glob
            paths   = sorted(glob_fn(glob_pattern))
        else:
            glob_fn = directory.rglob if recursive else directory.glob
            paths   = sorted(f for f in glob_fn("*") if f.is_file())

        logger.info(
            "MassUploader: scanning '%s' — %d file(s) found.",
            directory.name, len(paths),
        )
        return self.upload_files(paths, dry_run=dry_run)

    # =========================================================================
    #  RAG search
    # =========================================================================

    def search(
        self,
        query:       str,
        k:           int            = 5,
        doc_title:   Optional[str]  = None,
        threshold:   float          = 0.0,
    ) -> list[SearchResult]:
        """
        Semantic search across all ingested documents.

        Parameters
        ----------
        query : str
        k : int
            Maximum results.
        doc_title : str, optional
            Scope search to one document.
        threshold : float
            Minimum similarity score.

        Returns
        -------
        list[SearchResult]
        """
        return self._dm.search(
            query,
            k=k,
            doc_title=doc_title,
            similarity_threshold=threshold,
        )

    def format_rag_context(
        self,
        query:     str,
        k:         int   = 5,
        max_chars: int   = 500,
    ) -> str:
        """
        Retrieve relevant chunks and format them as a RAG context block
        ready for injection into an LLM prompt.

        Parameters
        ----------
        query : str
            The user's question.
        k : int
            Number of chunks to retrieve.
        max_chars : int
            Truncate each chunk at this length.

        Returns
        -------
        str
            Formatted context with numbered references.
        """
        results = self.search(query, k=k)
        return self._dm.format_search_results(results, max_chars_per_chunk=max_chars)

    # =========================================================================
    #  Inspection helpers
    # =========================================================================

    def inspect_directory(
        self,
        directory:  str | Path,
        recursive:  bool = True,
    ) -> list[DocumentTypeInfo]:
        """
        Scan a directory and return type information without ingesting.
        Useful for previewing what a batch upload would process.
        """
        return self._detector.scan_directory(
            directory, recursive=recursive, skip_unsupported=False
        )

    def inspect_files(
        self, paths: list[str | Path]
    ) -> list[DocumentTypeInfo]:
        """Return type information for a list of files without ingesting."""
        return [self._detector.detect(p) for p in paths]

    @property
    def knowledge_base_stats(self):
        """Return KnowledgeBaseStats from the underlying DocumentManager."""
        return self._dm.stats()

    @property
    def document_manager(self) -> DocumentManager:
        return self._dm

    # =========================================================================
    #  Internal processing
    # =========================================================================

    # ── Strategies that MUST be processed sequentially ─────────────────────────
    # pdfium (used by Docling for PDF rendering) is NOT thread-safe.
    # Running DOCLING/FALLBACK files in parallel threads causes:
    #   "Cannot close object; pdfium library is destroyed."
    # These strategies are always processed one-at-a-time.
    _SEQUENTIAL_STRATEGIES: frozenset = frozenset({
        ExtractionStrategy.DOCLING,
        ExtractionStrategy.FALLBACK,
    })

    def _process_batch(
        self,
        paths:   list[Path],
        dry_run: bool,
    ) -> list[FileOutcome]:
        """
        Split-lane batch processor.

        Files are classified by strategy before any work begins:

          Sequential lane  (DOCLING / FALLBACK — PDF, DOCX, XLSX, PPTX)
            Processed one at a time to avoid pdfium thread-safety issues.
            A fresh DoclingProcessor / DocumentConverter is created per file
            and fully garbage-collected before the next file starts.

          Parallel lane  (TEXT / STRUCTURED — TXT, MD, CSV, HTML, JSON, XML)
            Processed concurrently using a ThreadPoolExecutor with
            ``self._max_workers`` workers.  These use only pure-Python
            readers with no shared C-library state.

        The final output list preserves the original input order regardless
        of which lane processed each file.

        Parameters
        ----------
        paths : list[Path]
        dry_run : bool

        Returns
        -------
        list[FileOutcome]
            One entry per input path, in the same order.
        """
        if not paths:
            return []

        total    = len(paths)
        outcomes: list[FileOutcome | None] = [None] * total
        done     = 0

        def _record(idx: int, outcome: FileOutcome) -> None:
            nonlocal done
            outcomes[idx] = outcome
            done += 1
            if self._on_progress:
                try:
                    self._on_progress(outcome, done, total)
                except Exception:
                    pass

        # ── Phase 1: quick type detection for all paths ───────────────────────
        # Only reads file headers — fast and safe to run up-front.
        detections: list[tuple[int, Path, DocumentTypeInfo | None]] = []
        for idx, path in enumerate(paths):
            try:
                info = self._detector.detect(path)
            except Exception as exc:
                outcome = FileOutcome(
                    path=path,
                    status="error",
                    elapsed_ms=0.0,
                    error=f"Detection failed: {exc}",
                )
                _record(idx, outcome)
                info = None
            detections.append((idx, path, info))

        # Separate into sequential and parallel groups
        seq_items:  list[tuple[int, Path, DocumentTypeInfo]] = []
        par_items:  list[tuple[int, Path, DocumentTypeInfo]] = []

        for idx, path, info in detections:
            if info is None:
                continue   # already recorded as error
            if info.strategy in self._SEQUENTIAL_STRATEGIES:
                seq_items.append((idx, path, info))
            else:
                par_items.append((idx, path, info))

        logger.info(
            "MassUploader split: %d sequential (Docling/pdfium) "
            "+ %d parallel (text/structured).",
            len(seq_items), len(par_items),
        )

        # ── Phase 2: sequential lane (pdfium-safe) ────────────────────────────
        for idx, path, info in seq_items:
            outcome = self._process_one(path, dry_run, pre_detected=info)
            _record(idx, outcome)

        # ── Phase 3: parallel lane (text/structured) ─────────────────────────
        if par_items:
            workers = min(self._max_workers, len(par_items))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(self._process_one, path, dry_run, info): idx
                    for idx, path, info in par_items
                }
                for future in as_completed(future_to_idx):
                    idx     = future_to_idx[future]
                    outcome = future.result()   # never raises
                    _record(idx, outcome)

        return [o for o in outcomes if o is not None]

    def _process_one(
        self,
        path:         Path,
        dry_run:      bool,
        pre_detected: "DocumentTypeInfo | None" = None,
    ) -> FileOutcome:
        """
        Full pipeline for a single file: detect → deduplicate → ingest.

        Parameters
        ----------
        path : Path
        dry_run : bool
        pre_detected : DocumentTypeInfo, optional
            If the caller already ran type detection (e.g. _process_batch
            pre-detects to partition into lanes), pass the result here to
            skip a second detection call.

        All exceptions are caught; errors are captured in FileOutcome.error
        rather than propagated.
        """
        t0 = time.monotonic()

        # ── Step 1: Detect type ───────────────────────────────────────────────
        if pre_detected is not None:
            info = pre_detected
        else:
            try:
                info = self._detector.detect(path)
            except Exception as exc:
                return FileOutcome(
                    path=path, elapsed_ms=(time.monotonic() - t0) * 1000,
                    error=f"Detection failed: {exc}", status="error",
                )

        # ── Step 2: Skip unsupported files ────────────────────────────────────
        if not info.is_supported:
            logger.debug("Skipping unsupported file: %s", path.name)
            return FileOutcome(
                path=path,
                doc_type=info.doc_type,
                strategy=info.strategy,
                status="unsupported",
                content_hash=info.content_hash,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                warnings=info.warnings,
            )

        # ── Step 3: Deduplicate by content hash (atomic check-and-reserve) ─────
        if info.content_hash:
            with self._dedup_lock:
                if info.content_hash in self._seen_hashes:
                    logger.debug("Duplicate (hash match): %s", path.name)
                    return FileOutcome(
                        path=path,
                        doc_type=info.doc_type,
                        strategy=info.strategy,
                        status="duplicate",
                        content_hash=info.content_hash,
                        elapsed_ms=(time.monotonic() - t0) * 1000,
                        warnings=info.warnings,
                    )
                # Reserve the hash so a concurrent worker doesn't also ingest this content
                self._seen_hashes.add(info.content_hash)

        # ── Step 4: Dry-run exit ──────────────────────────────────────────────
        if dry_run:
            return FileOutcome(
                path=path,
                doc_type=info.doc_type,
                strategy=info.strategy,
                status="skipped",
                content_hash=info.content_hash,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                warnings=info.warnings,
            )

        # ── Step 5: Ingest via appropriate strategy ───────────────────────────
        try:
            ingest_result = self._ingest_by_strategy(path, info)
        except Exception as exc:
            logger.error("Ingest failed for '%s': %s", path.name, exc)
            return FileOutcome(
                path=path,
                doc_type=info.doc_type,
                strategy=info.strategy,
                status="error",
                content_hash=info.content_hash,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                warnings=info.warnings,
                error=str(exc),
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        status     = "ok" if ingest_result.chunks_added > 0 else "skipped"

        logger.info(
            "%s '%s' — strategy=%s chunks_added=%d  %.0fms",
            "Ingested" if status == "ok" else "Skipped (already indexed)",
            path.name, info.strategy.value,
            ingest_result.chunks_added, elapsed_ms,
        )
        return FileOutcome(
            path=path,
            doc_type=info.doc_type,
            strategy=info.strategy,
            status=status,
            chunks_added=ingest_result.chunks_added,
            chunks_total=ingest_result.chunks_total,
            doc_title=ingest_result.doc_title,
            content_hash=info.content_hash,
            elapsed_ms=elapsed_ms,
            warnings=info.warnings,
            error=ingest_result.error,
        )

    def _ingest_by_strategy(
        self,
        path: Path,
        info: DocumentTypeInfo,
    ) -> IngestResult:
        """
        Route to the correct ingestion path based on the detected strategy.

        DOCLING / FALLBACK
            Creates a **fresh** ``DoclingProcessor`` for each file and
            explicitly deletes it after ingestion.  This ensures pdfium
            objects are fully released before the next document is processed,
            eliminating the "pdfium library is destroyed" memory leak.

        TEXT       → _ingest_text_file()
        STRUCTURED → _ingest_structured_file()
        """
        if info.strategy in (ExtractionStrategy.DOCLING,
                              ExtractionStrategy.FALLBACK):
            return self._ingest_docling_file(path)

        if info.strategy == ExtractionStrategy.TEXT:
            return self._ingest_text_file(path, info)

        if info.strategy == ExtractionStrategy.STRUCTURED:
            return self._ingest_structured_file(path, info)

        return IngestResult(
            filename=path.name,
            doc_title=path.stem.title(),
            error=f"No handler for strategy '{info.strategy.value}'",
        )

    def _ingest_docling_file(self, path: Path) -> IngestResult:
        """
        Ingest a PDF/DOCX/XLSX/PPTX file using a short-lived DoclingProcessor.

        A fresh processor is instantiated for EACH file so that pdfium's
        global C-library state is fully initialised and torn down within
        the scope of a single call.  Explicit ``del`` + ``gc.collect()``
        after processing guarantees pdfium objects are destroyed before
        control returns to the caller (and before the next file starts on
        the sequential lane).

        This eliminates the race condition that produces:
            "Cannot close object; pdfium library is destroyed."
        """
        import gc

        t0        = time.monotonic()
        processor = None
        chunks    = []

        try:
            processor = _DoclingProcessor(
                chunk_size=self._dm._processor.chunk_size,
                chunk_overlap=self._dm._processor.chunk_overlap,
            )
            chunks = processor.process(path)
        except Exception as exc:
            logger.error("Docling processing failed for '%s': %s", path.name, exc)
            return IngestResult(
                filename=path.name,
                doc_title=path.stem.title(),
                error=str(exc),
            )
        finally:
            # Explicit cleanup: release pdfium resources before returning
            del processor
            gc.collect()

        if not chunks:
            return IngestResult(
                filename=path.name,
                doc_title=path.stem.title(),
                chunks_added=0,
                chunks_total=0,
            )

        doc_title = chunks[0].doc_title if chunks else path.stem.title()

        try:
            added = self._dm._store.ingest(chunks)
        except Exception as exc:
            logger.error("Vector store ingest failed for '%s': %s", path.name, exc)
            return IngestResult(
                filename=path.name,
                doc_title=doc_title,
                error=str(exc),
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Docling ingest '%s': %d new chunks in %.0fms.",
            path.name, added, elapsed_ms,
        )
        return IngestResult(
            filename=path.name,
            doc_title=doc_title,
            chunks_added=added,
            chunks_total=len(chunks),
            elapsed_ms=elapsed_ms,
        )

    def _ingest_text_file(
        self, path: Path, info: DocumentTypeInfo
    ) -> IngestResult:
        """
        Ingest a plain-text file (TXT, MD, CSV, HTML) by reading its content
        and building DocumentChunks manually.
        """
        from .docling_processor import DocumentChunk
        import hashlib

        t0       = time.monotonic()
        encoding = info.encoding or "utf-8"

        try:
            text = path.read_text(encoding=encoding, errors="replace")
        except Exception as exc:
            return IngestResult(
                filename=path.name,
                doc_title=path.stem.title(),
                error=f"Could not read file: {exc}",
            )

        # For HTML strip tags
        if info.doc_type == DocumentType.HTML:
            text = self._strip_html(text)

        # For CSV preserve structure but add a header describing the file
        if info.doc_type == DocumentType.CSV:
            lines  = text.splitlines()
            header = lines[0] if lines else ""
            text   = f"CSV file: {path.name}\nColumns: {header}\n\n" + text

        if not text.strip():
            return IngestResult(
                filename=path.name,
                doc_title=path.stem.title(),
                chunks_added=0,
                chunks_total=0,
            )

        doc_title = path.stem.replace("_", " ").replace("-", " ").title()
        doc_path  = str(path)

        # Split into overlapping chunks using the processor's splitter
        chunks: list[DocumentChunk] = []
        step = max(1, self._dm._processor.chunk_size - self._dm._processor.chunk_overlap)
        for i, start in enumerate(range(0, len(text), step)):
            sub = text[start: start + self._dm._processor.chunk_size].strip()
            if not sub:
                continue
            raw_id   = f"{doc_path}:{i}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=sub,
                doc_path=doc_path,
                doc_title=doc_title,
                page_number=None,
                section_path=[],
                char_offset=start,
                metadata={"doctype": info.doc_type.value, "encoding": encoding},
            ))

        if not chunks:
            return IngestResult(filename=path.name, doc_title=doc_title,
                                chunks_added=0, chunks_total=0)

        added = self._dm._store.ingest(chunks)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "Text ingest '%s': %d new chunks in %.0fms.", path.name, added, elapsed
        )
        return IngestResult(
            filename=path.name,
            doc_title=doc_title,
            chunks_added=added,
            chunks_total=len(chunks),
            elapsed_ms=elapsed,
        )

    def _ingest_structured_file(
        self, path: Path, info: DocumentTypeInfo
    ) -> IngestResult:
        """
        Ingest a JSON or XML file by flattening its structure into text chunks.
        """
        from .docling_processor import DocumentChunk
        import hashlib, json

        t0        = time.monotonic()
        doc_title = path.stem.replace("_", " ").replace("-", " ").title()
        doc_path  = str(path)

        try:
            if info.doc_type == DocumentType.JSON:
                raw  = path.read_text(encoding="utf-8", errors="replace")
                # Try to pretty-print for better chunking
                try:
                    obj  = json.loads(raw)
                    text = json.dumps(obj, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    text = raw   # malformed JSON → treat as raw text
            else:
                # XML → read as text (preserves structure as readable string)
                text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return IngestResult(
                filename=path.name, doc_title=doc_title,
                error=f"Could not read file: {exc}",
            )

        if not text.strip():
            return IngestResult(filename=path.name, doc_title=doc_title,
                                chunks_added=0, chunks_total=0)

        chunks: list[DocumentChunk] = []
        step = max(1, self._dm._processor.chunk_size - self._dm._processor.chunk_overlap)
        for i, start in enumerate(range(0, len(text), step)):
            sub = text[start: start + self._dm._processor.chunk_size].strip()
            if not sub:
                continue
            chunk_id = hashlib.sha256(f"{doc_path}:{i}".encode()).hexdigest()[:16]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=sub,
                doc_path=doc_path,
                doc_title=doc_title,
                page_number=None,
                section_path=[],
                char_offset=start,
                metadata={"doctype": info.doc_type.value},
            ))

        if not chunks:
            return IngestResult(filename=path.name, doc_title=doc_title,
                                chunks_added=0, chunks_total=0)

        added = self._dm._store.ingest(chunks)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "Structured ingest '%s': %d new chunks in %.0fms.",
            path.name, added, elapsed,
        )
        return IngestResult(
            filename=path.name,
            doc_title=doc_title,
            chunks_added=added,
            chunks_total=len(chunks),
            elapsed_ms=elapsed,
        )

    # =========================================================================
    #  Utility
    # =========================================================================

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and normalise whitespace."""
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;",  "&", text)
        text = re.sub(r"&lt;",   "<", text)
        text = re.sub(r"&gt;",   ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _docling_available() -> bool:
        """Return True if Docling can be imported."""
        try:
            import docling  # noqa: F401
            return True
        except ImportError:
            return False
