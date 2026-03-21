"""
document_processing/type_detector.py
─────────────────────────────────────
Identifies the type, encoding, and optimal extraction strategy for any
document file before it enters the ingestion pipeline.

This sits upstream of DoclingProcessor and DocumentManager, so the
mass uploader can make informed decisions about:
  • Which parser to use (Docling vs pure-Python fallback)
  • Whether the document needs OCR
  • Appropriate chunk sizes for the content type
  • Whether to skip or warn about unsupported files
  • Deduplication using content hashing

Supported detection:
  Extension-based (fast, always runs):
    .pdf .docx .xlsx .xls .pptx .ppt
    .txt .md .csv .json .html .htm .xml

  Content-based sniffing (runs when extension is ambiguous or missing):
    Magic-byte detection for PDF, ZIP-based Office formats, plain text

  Metadata extraction:
    File size, estimated page count, word count estimate, encoding detection

Usage::

    from document_processing.type_detector import TypeDetector, DocumentType

    detector = TypeDetector()
    info     = detector.detect("reports/Q3.pdf")

    print(info.doc_type)           # DocumentType.PDF
    print(info.strategy)           # ExtractionStrategy.DOCLING
    print(info.estimated_chunks)   # 42
    print(info.is_supported)       # True
    print(info.content_hash)       # "a3f1..."  (for deduplication)
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
#  Enumerations
# =============================================================================

class DocumentType(str, Enum):
    """All document types the system can handle."""
    # Natively supported by the DoclingProcessor / fallback parsers
    PDF    = "pdf"
    DOCX   = "docx"
    XLSX   = "xlsx"
    XLS    = "xls"
    PPTX   = "pptx"
    PPT    = "ppt"
    # Supported via plain-text extraction
    TXT    = "txt"
    MD     = "md"
    CSV    = "csv"
    JSON   = "json"
    HTML   = "html"
    XML    = "xml"
    # Not supported
    UNKNOWN = "unknown"


class ExtractionStrategy(str, Enum):
    """
    Recommended extraction pipeline for a given document.

    DOCLING       — Full Docling pipeline (OCR, table extraction, headings).
                   Best for PDFs and Office formats.
    FALLBACK      — Pure-Python parsers (python-docx, openpyxl, python-pptx).
                   Used when Docling is unavailable or the file is simple.
    TEXT          — Direct text reading (plain text, Markdown, CSV).
    STRUCTURED    — JSON/XML — parse structure, then flatten to chunks.
    SKIP          — Unsupported or unreadable file; skip with a warning.
    """
    DOCLING    = "docling"
    FALLBACK   = "fallback"
    TEXT       = "text"
    STRUCTURED = "structured"
    SKIP       = "skip"


# =============================================================================
#  Detection result
# =============================================================================

@dataclass
class DocumentTypeInfo:
    """
    Everything the ingestion pipeline needs to know about a file before
    actually parsing it.

    Attributes
    ----------
    path : Path
        Absolute path to the document.
    doc_type : DocumentType
        Detected type.
    strategy : ExtractionStrategy
        Recommended extraction strategy.
    is_supported : bool
        True when the file can be ingested.
    file_size_bytes : int
        Raw file size.
    content_hash : str
        SHA-256 hex digest of the file content (for deduplication).
    estimated_chunks : int
        Rough estimate based on file size and type (used for progress bars).
    encoding : str
        Detected text encoding (meaningful for TXT/CSV/HTML only).
    needs_ocr : bool
        True if the PDF appears to be a scanned/image-only document.
    warnings : list[str]
        Non-fatal issues discovered during detection.
    metadata : dict
        Raw extra metadata (page count for PDF, sheet names for XLSX, etc.)
    """
    path:             Path
    doc_type:         DocumentType
    strategy:         ExtractionStrategy
    is_supported:     bool
    file_size_bytes:  int         = 0
    content_hash:     str         = ""
    estimated_chunks: int         = 1
    encoding:         str         = "utf-8"
    needs_ocr:        bool        = False
    warnings:         list[str]   = field(default_factory=list)
    metadata:         dict        = field(default_factory=dict)

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)

    @property
    def friendly_size(self) -> str:
        if self.file_size_bytes >= 1_048_576:
            return f"{self.file_size_mb:.1f} MB"
        if self.file_size_bytes >= 1_024:
            return f"{self.file_size_bytes / 1024:.1f} KB"
        return f"{self.file_size_bytes} B"

    def __str__(self) -> str:
        supported = "✓" if self.is_supported else "✗"
        ocr_note  = "  [needs OCR]" if self.needs_ocr else ""
        warn_note = f"  [{len(self.warnings)} warning(s)]" if self.warnings else ""
        return (
            f"{supported} {self.path.name:<40} "
            f"{self.doc_type.value:<8} "
            f"{self.strategy.value:<12} "
            f"{self.friendly_size:>10} "
            f"~{self.estimated_chunks} chunks"
            f"{ocr_note}{warn_note}"
        )


# =============================================================================
#  TypeDetector
# =============================================================================

class TypeDetector:
    """
    Identifies the type and extraction strategy for document files.

    Design principles:
    - Extension-first: fast and covers 99% of cases.
    - Content-sniff fallback: handles files with wrong/missing extensions.
    - Non-destructive: never modifies files, reads only enough to detect type.
    - Safe: all exceptions are caught; unknown files get SKIP strategy.

    Parameters
    ----------
    chunk_size : int
        Used to estimate chunk counts from file size.
    use_docling : bool
        If False, always recommend FALLBACK strategy regardless of type.
        Useful when Docling is not installed.
    max_sniff_bytes : int
        How many bytes to read for magic-byte / content detection.
    """

    # Extension → (DocumentType, ExtractionStrategy)
    _EXT_MAP: dict[str, tuple[DocumentType, ExtractionStrategy]] = {
        ".pdf":  (DocumentType.PDF,  ExtractionStrategy.DOCLING),
        ".docx": (DocumentType.DOCX, ExtractionStrategy.DOCLING),
        ".doc":  (DocumentType.DOCX, ExtractionStrategy.FALLBACK),   # legacy
        ".xlsx": (DocumentType.XLSX, ExtractionStrategy.DOCLING),
        ".xls":  (DocumentType.XLS,  ExtractionStrategy.FALLBACK),
        ".pptx": (DocumentType.PPTX, ExtractionStrategy.DOCLING),
        ".ppt":  (DocumentType.PPT,  ExtractionStrategy.FALLBACK),
        ".txt":  (DocumentType.TXT,  ExtractionStrategy.TEXT),
        ".md":   (DocumentType.MD,   ExtractionStrategy.TEXT),
        ".markdown": (DocumentType.MD, ExtractionStrategy.TEXT),
        ".csv":  (DocumentType.CSV,  ExtractionStrategy.TEXT),
        ".tsv":  (DocumentType.CSV,  ExtractionStrategy.TEXT),
        ".json": (DocumentType.JSON, ExtractionStrategy.STRUCTURED),
        ".jsonl":(DocumentType.JSON, ExtractionStrategy.STRUCTURED),
        ".html": (DocumentType.HTML, ExtractionStrategy.TEXT),
        ".htm":  (DocumentType.HTML, ExtractionStrategy.TEXT),
        ".xml":  (DocumentType.XML,  ExtractionStrategy.STRUCTURED),
    }

    # Magic bytes → DocumentType
    _MAGIC_MAP: list[tuple[bytes, DocumentType]] = [
        (b"%PDF",         DocumentType.PDF),
        (b"PK\x03\x04",  DocumentType.DOCX),   # ZIP-based Office (DOCX/XLSX/PPTX)
        (b"\xd0\xcf\x11\xe0", DocumentType.XLS),  # Compound Document (old Office)
    ]

    # Average chars-per-chunk denominator used for chunk count estimates
    _AVG_CHARS_PER_CHUNK = 400

    def __init__(
            self,
            chunk_size:     int  = 512,
            use_docling:    bool = True,
            max_sniff_bytes:int  = 8192,
    ) -> None:
        self._chunk_size      = chunk_size
        self._use_docling     = use_docling
        self._max_sniff_bytes = max_sniff_bytes

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, path: str | Path) -> DocumentTypeInfo:
        """
        Analyse a file and return full type/strategy information.

        Never raises — all errors are captured in DocumentTypeInfo.warnings
        and the strategy is set to SKIP for unreadable files.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        DocumentTypeInfo
        """
        path = Path(path).resolve()

        # ── Basic existence checks ────────────────────────────────────────────
        if not path.exists():
            return self._unsupported(path, f"File not found: {path}")
        if not path.is_file():
            return self._unsupported(path, f"Not a regular file: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            return self._unsupported(path, "File is empty (0 bytes)", size=0)

        # ── Content hash (always compute — used for deduplication) ────────────
        content_hash = self._hash_file(path)

        # ── Extension-based detection (fast path) ────────────────────────────
        ext = path.suffix.lower()
        doc_type, strategy = self._EXT_MAP.get(ext, (DocumentType.UNKNOWN, ExtractionStrategy.SKIP))

        # ── Content sniff (when extension is unknown or suspicious) ──────────
        warnings: list[str] = []
        if doc_type == DocumentType.UNKNOWN or ext == "":
            sniffed = self._sniff_type(path)
            if sniffed:
                doc_type, strategy = sniffed
                warnings.append(
                    f"Extension '{ext}' is unusual; type detected from file content."
                )
            else:
                return DocumentTypeInfo(
                    path=path,
                    doc_type=DocumentType.UNKNOWN,
                    strategy=ExtractionStrategy.SKIP,
                    is_supported=False,
                    file_size_bytes=file_size,
                    content_hash=content_hash,
                    warnings=[f"Unrecognised file type (extension: '{ext}')"],
                )

        # ── Adjust strategy when Docling is not available ─────────────────────
        if not self._use_docling and strategy == ExtractionStrategy.DOCLING:
            strategy = ExtractionStrategy.FALLBACK
            warnings.append("Docling unavailable; using pure-Python fallback parser.")

        # ── Type-specific metadata ────────────────────────────────────────────
        metadata: dict = {"extension": ext, "file_size_bytes": file_size}
        needs_ocr = False
        encoding  = "utf-8"

        if doc_type == DocumentType.PDF:
            pdf_meta  = self._inspect_pdf(path)
            metadata.update(pdf_meta)
            needs_ocr = pdf_meta.get("needs_ocr", False)
            if needs_ocr:
                warnings.append(
                    "PDF appears to be a scanned/image document. "
                    "Docling will attempt OCR but accuracy may be lower."
                )

        elif doc_type in (DocumentType.XLSX, DocumentType.XLS):
            xl_meta = self._inspect_xlsx(path)
            metadata.update(xl_meta)

        elif doc_type in (DocumentType.TXT, DocumentType.MD,
                          DocumentType.CSV, DocumentType.HTML):
            encoding = self._detect_encoding(path)
            metadata["encoding"] = encoding

        # ── Size warnings ─────────────────────────────────────────────────────
        if file_size > 100 * 1024 * 1024:  # > 100 MB
            warnings.append(
                f"Large file ({file_size / 1_048_576:.0f} MB). "
                "Ingestion may take several minutes."
            )

        # ── Chunk count estimate ──────────────────────────────────────────────
        estimated_chunks = self._estimate_chunks(file_size, doc_type)

        is_supported = strategy != ExtractionStrategy.SKIP

        return DocumentTypeInfo(
            path=path,
            doc_type=doc_type,
            strategy=strategy,
            is_supported=is_supported,
            file_size_bytes=file_size,
            content_hash=content_hash,
            estimated_chunks=estimated_chunks,
            encoding=encoding,
            needs_ocr=needs_ocr,
            warnings=warnings,
            metadata=metadata,
        )

    def detect_many(
            self,
            paths: list[str | Path],
            skip_unsupported: bool = True,
    ) -> list[DocumentTypeInfo]:
        """
        Detect types for a list of files.

        Parameters
        ----------
        paths : list
            File paths to inspect.
        skip_unsupported : bool
            If True, unsupported files are excluded from the result.

        Returns
        -------
        list[DocumentTypeInfo]
        """
        results = [self.detect(p) for p in paths]
        if skip_unsupported:
            results = [r for r in results if r.is_supported]
        return results

    def scan_directory(
            self,
            directory:        str | Path,
            recursive:        bool = True,
            skip_unsupported: bool = True,
    ) -> list[DocumentTypeInfo]:
        """
        Scan a directory and return type info for every file found.

        Parameters
        ----------
        directory : str | Path
        recursive : bool
            Descend into sub-directories.
        skip_unsupported : bool
            Exclude unrecognised files from the result.

        Returns
        -------
        list[DocumentTypeInfo]
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        glob    = directory.rglob("*") if recursive else directory.glob("*")
        files   = sorted(f for f in glob if f.is_file())
        results = self.detect_many(files, skip_unsupported=skip_unsupported)
        logger.info(
            "Scanned '%s': %d file(s), %d supported.",
            directory.name, len(files), sum(1 for r in results if r.is_supported),
        )
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 of the whole file content — used for deduplication."""
        h = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
        except OSError:
            return ""
        return h.hexdigest()

    def _sniff_type(
            self, path: Path
    ) -> Optional[tuple[DocumentType, ExtractionStrategy]]:
        """Read up to max_sniff_bytes and try to identify the type."""
        try:
            with path.open("rb") as f:
                header = f.read(self._max_sniff_bytes)
        except OSError:
            return None

        for magic, doc_type in self._MAGIC_MAP:
            if header.startswith(magic):
                # ZIP-based: refine to DOCX / XLSX / PPTX by inspecting ZIP
                if magic == b"PK\x03\x04":
                    refined = self._refine_zip_type(path)
                    if refined:
                        return refined
                    # Generic ZIP Office
                    return (doc_type, ExtractionStrategy.FALLBACK)
                strategy = (ExtractionStrategy.DOCLING if self._use_docling
                            else ExtractionStrategy.FALLBACK)
                return (doc_type, strategy)

        # Plain text heuristic: high fraction of printable ASCII
        printable = sum(1 for b in header[:512] if 32 <= b < 127 or b in (9, 10, 13))
        if len(header) > 0 and printable / min(len(header), 512) > 0.85:
            return (DocumentType.TXT, ExtractionStrategy.TEXT)

        return None

    @staticmethod
    def _refine_zip_type(
            path: Path,
    ) -> Optional[tuple[DocumentType, ExtractionStrategy]]:
        """Open a ZIP and check internal paths to distinguish DOCX/XLSX/PPTX."""
        try:
            import zipfile
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
            if any("word/" in n for n in names):
                return (DocumentType.DOCX, ExtractionStrategy.DOCLING)
            if any("xl/" in n for n in names):
                return (DocumentType.XLSX, ExtractionStrategy.DOCLING)
            if any("ppt/" in n for n in names):
                return (DocumentType.PPTX, ExtractionStrategy.DOCLING)
        except Exception:
            pass
        return None

    @staticmethod
    def _inspect_pdf(path: Path) -> dict:
        """Extract page count and detect scanned PDFs."""
        meta: dict = {"page_count": None, "needs_ocr": False}
        try:
            with path.open("rb") as f:
                content = f.read()
            # Count pages via /Type /Page objects
            import re
            pages = len(re.findall(rb"/Type\s*/Page\b", content))
            meta["page_count"] = pages or None
            # Heuristic: if there are very few /Font resources relative to pages,
            # the PDF is likely image-only (scanned)
            fonts = len(re.findall(rb"/Font\b", content))
            if pages and pages > 0 and fonts == 0:
                meta["needs_ocr"] = True
        except Exception:
            pass
        return meta

    @staticmethod
    def _inspect_xlsx(path: Path) -> dict:
        """Extract sheet names from XLSX without loading data."""
        meta: dict = {"sheet_names": [], "sheet_count": 0}
        try:
            import zipfile, re
            with zipfile.ZipFile(path, "r") as zf:
                if "xl/workbook.xml" in zf.namelist():
                    xml = zf.read("xl/workbook.xml").decode("utf-8", errors="replace")
                    sheets = re.findall(r'name="([^"]+)"', xml)
                    meta["sheet_names"] = sheets
                    meta["sheet_count"] = len(sheets)
        except Exception:
            pass
        return meta

    @staticmethod
    def _detect_encoding(path: Path, sample_bytes: int = 4096) -> str:
        """Detect text encoding via BOM or chardet (if available)."""
        try:
            with path.open("rb") as f:
                raw = f.read(sample_bytes)
            # BOM checks
            if raw.startswith(b"\xff\xfe"):   return "utf-16-le"
            if raw.startswith(b"\xfe\xff"):   return "utf-16-be"
            if raw.startswith(b"\xef\xbb\xbf"):return "utf-8-sig"
            # Try chardet
            try:
                import chardet
                result = chardet.detect(raw)
                if result and result.get("confidence", 0) > 0.7:
                    return result["encoding"] or "utf-8"
            except ImportError:
                pass
        except OSError:
            pass
        return "utf-8"

    def _estimate_chunks(self, file_size: int, doc_type: DocumentType) -> int:
        """
        Rough estimate of how many chunks a file will produce.

        Different file types have different text density per byte
        (PDFs are denser than XLSX; plain text is very dense).
        """
        bytes_per_char: dict[DocumentType, float] = {
            DocumentType.PDF:  1.5,   # PDF has binary overhead
            DocumentType.DOCX: 2.0,   # DOCX has XML overhead
            DocumentType.XLSX: 3.0,   # XLSX is very sparse
            DocumentType.XLS:  3.0,
            DocumentType.PPTX: 2.5,
            DocumentType.PPT:  2.5,
            DocumentType.TXT:  1.0,
            DocumentType.MD:   1.0,
            DocumentType.CSV:  1.2,
            DocumentType.HTML: 3.0,   # lots of markup
            DocumentType.JSON: 2.0,
            DocumentType.XML:  2.5,
        }
        bpc         = bytes_per_char.get(doc_type, 2.0)
        approx_chars= file_size / bpc
        n_chunks    = max(1, int(approx_chars / self._AVG_CHARS_PER_CHUNK))
        return n_chunks

    @staticmethod
    def _unsupported(
            path: Path,
            reason: str,
            size: int = -1,
    ) -> DocumentTypeInfo:
        if size < 0:
            try:
                size = path.stat().st_size if path.exists() else 0
            except OSError:
                size = 0
        return DocumentTypeInfo(
            path=path,
            doc_type=DocumentType.UNKNOWN,
            strategy=ExtractionStrategy.SKIP,
            is_supported=False,
            file_size_bytes=size,
            warnings=[reason],
        )