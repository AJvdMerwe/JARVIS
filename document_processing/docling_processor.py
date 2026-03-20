"""
document_processing/docling_processor.py
──────────────────────────────────────────
Wraps Docling to convert documents (PDF, DOCX, XLSX, PPTX) into structured
text chunks with rich metadata for vector-store ingestion.

Each ``DocumentChunk`` carries:
  • The text content
  • Source document path & friendly title
  • Page / slide / sheet reference
  • Section heading path (breadcrumb)
  • Unique chunk ID for deep-linking back to the source
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".ppt"}


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single text chunk extracted from a document.

    Attributes:
        chunk_id:       Stable SHA-256 derived from source + page + offset.
        text:           The chunk's textual content.
        doc_path:       Absolute path to the source document.
        doc_title:      Human-readable document name.
        page_number:    1-based page/slide/sheet number (None if unavailable).
        section_path:   Breadcrumb of heading levels, e.g. ["Chapter 1", "Intro"].
        char_offset:    Character offset within the page/element (for ordering).
        metadata:       Extra key-value metadata (doctype, etc.).
    """

    chunk_id: str
    text: str
    doc_path: str
    doc_title: str
    page_number: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    char_offset: int = 0
    metadata: dict = field(default_factory=dict)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def reference(self) -> str:
        """
        Short human-readable reference, e.g.:
            "Q3_Report.pdf › Page 4 › Revenue Analysis"
        """
        parts = [self.doc_title]
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        if self.section_path:
            parts.append(" › ".join(self.section_path))
        return " › ".join(parts)

    def to_langchain_doc(self):
        """Convert to a LangChain Document for vector-store ingestion."""
        from langchain_core.documents import Document

        return Document(
            page_content=self.text,
            metadata={
                "chunk_id": self.chunk_id,
                "doc_path": self.doc_path,
                "doc_title": self.doc_title,
                "page_number": self.page_number,
                "section_path": " › ".join(self.section_path),
                "reference": self.reference,
                **self.metadata,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Docling processor
# ─────────────────────────────────────────────────────────────────────────────

class DoclingProcessor:
    """
    Uses Docling to parse documents and yields ``DocumentChunk`` objects.

    Docling handles the heavy lifting of OCR, table extraction, and heading
    detection for all supported file types.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _make_chunk_id(doc_path: str, page: Optional[int], offset: int) -> str:
        raw = f"{doc_path}:{page}:{offset}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _split_text(self, text: str) -> list[str]:
        """Naive overlapping-window splitter (no external dependency)."""
        chunks: list[str] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # ── Docling-specific parsing ─────────────────────────────────────────────

    def _parse_with_docling(self, path: Path) -> list[DocumentChunk]:
        """
        Primary parsing path: use Docling's DocumentConverter.
        Returns a list of DocumentChunk objects.
        """
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
            from docling.datamodel.base_models import InputFormat  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "docling is not installed. Run: pip install docling"
            ) from exc

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document

        doc_title = path.stem.replace("_", " ").replace("-", " ").title()
        doc_path_str = str(path.resolve())

        chunks: list[DocumentChunk] = []
        current_headings: list[str] = []

        # Iterate over Docling's document body items
        for item in doc.iterate_items():
            label = getattr(item, "label", "")
            text_content = ""

            # Extract text from the item
            if hasattr(item, "text"):
                text_content = str(item.text).strip()
            elif hasattr(item, "export_to_markdown"):
                text_content = item.export_to_markdown().strip()

            if not text_content:
                continue

            # Track heading hierarchy for section breadcrumbs
            if "heading" in str(label).lower() or "title" in str(label).lower():
                level = 1
                for lvl_name in ["heading1", "heading2", "heading3"]:
                    if lvl_name in str(label).lower():
                        level = int(lvl_name[-1])
                        break
                current_headings = current_headings[: level - 1]
                current_headings.append(text_content[:80])

            # Get page reference if available
            page_num: Optional[int] = None
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "page_no"):
                    page_num = prov.page_no

            # Split long text into overlapping chunks
            for offset, sub_text in enumerate(self._split_text(text_content)):
                chunk_id = self._make_chunk_id(doc_path_str, page_num, offset)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=sub_text,
                        doc_path=doc_path_str,
                        doc_title=doc_title,
                        page_number=page_num,
                        section_path=list(current_headings),
                        char_offset=offset * max(1, self.chunk_size - self.chunk_overlap),
                        metadata={"doctype": path.suffix.lstrip("."), "label": str(label)},
                    )
                )

        logger.info("Docling parsed '%s' → %d chunks.", path.name, len(chunks))
        return chunks

    # ── Fallback parsers ─────────────────────────────────────────────────────

    def _parse_fallback(self, path: Path) -> list[DocumentChunk]:
        """
        Pure-Python fallback parsers for when Docling is unavailable.
        Supports DOCX, XLSX, PPTX via python-docx / openpyxl / python-pptx.
        """
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return self._parse_docx(path)
        if suffix in (".xlsx", ".xls"):
            return self._parse_xlsx(path)
        if suffix in (".pptx", ".ppt"):
            return self._parse_pptx(path)
        # Last resort: treat as plain text
        return self._parse_plaintext(path)

    def _parse_docx(self, path: Path) -> list[DocumentChunk]:
        import docx  # type: ignore

        document = docx.Document(str(path))
        doc_title = path.stem.title()
        chunks, headings = [], []

        for para in document.paragraphs:
            if not para.text.strip():
                continue
            style = para.style.name if para.style else ""
            if style.startswith("Heading"):
                try:
                    lvl = int(style.split()[-1])
                except ValueError:
                    lvl = 1
                headings = headings[: lvl - 1] + [para.text.strip()[:80]]
            for sub_text in self._split_text(para.text.strip()):
                cid = self._make_chunk_id(str(path), None, len(chunks))
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        text=sub_text,
                        doc_path=str(path.resolve()),
                        doc_title=doc_title,
                        section_path=list(headings),
                        metadata={"doctype": "docx"},
                    )
                )
        return chunks

    def _parse_xlsx(self, path: Path) -> list[DocumentChunk]:
        import openpyxl  # type: ignore

        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        doc_title = path.stem.title()
        chunks = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_text = []
            for row in ws.iter_rows(values_only=True):
                row_str = "\t".join(str(c) for c in row if c is not None)
                if row_str.strip():
                    rows_text.append(row_str)

            full_text = "\n".join(rows_text)
            for offset, sub_text in enumerate(self._split_text(full_text)):
                cid = self._make_chunk_id(str(path), None, offset)
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        text=sub_text,
                        doc_path=str(path.resolve()),
                        doc_title=doc_title,
                        section_path=[sheet_name],
                        metadata={"doctype": "xlsx", "sheet": sheet_name},
                    )
                )
        wb.close()
        return chunks

    def _parse_pptx(self, path: Path) -> list[DocumentChunk]:
        from pptx import Presentation  # type: ignore

        prs = Presentation(str(path))
        doc_title = path.stem.title()
        chunks = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            texts.append(t)
            full_text = "\n".join(texts)
            for offset, sub_text in enumerate(self._split_text(full_text)):
                cid = self._make_chunk_id(str(path), slide_num, offset)
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        text=sub_text,
                        doc_path=str(path.resolve()),
                        doc_title=doc_title,
                        page_number=slide_num,
                        metadata={"doctype": "pptx"},
                    )
                )
        return chunks

    def _parse_plaintext(self, path: Path) -> list[DocumentChunk]:
        text = path.read_text(errors="replace")
        doc_title = path.stem.title()
        return [
            DocumentChunk(
                chunk_id=self._make_chunk_id(str(path), None, i),
                text=chunk,
                doc_path=str(path.resolve()),
                doc_title=doc_title,
                metadata={"doctype": path.suffix.lstrip(".")},
            )
            for i, chunk in enumerate(self._split_text(text))
        ]

    # ── Public API ──────────────────────────────────────────────────────────

    def process(self, path: str | Path) -> list[DocumentChunk]:
        """
        Parse a document and return its chunks.

        Tries Docling first; falls back to pure-Python parsers if Docling
        is unavailable or raises an exception.

        Args:
            path: Path to the document file.

        Returns:
            List of ``DocumentChunk`` objects.

        Raises:
            ValueError: If the file type is not supported.
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Allowed: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )

        try:
            return self._parse_with_docling(path)
        except ImportError:
            logger.warning(
                "Docling not available – falling back to pure-Python parsers for '%s'.",
                path.name,
            )
            return self._parse_fallback(path)
        except Exception as exc:
            logger.warning(
                "Docling failed for '%s' (%s) – falling back to pure-Python parsers.",
                path.name,
                exc,
            )
            return self._parse_fallback(path)

    def process_many(self, paths: Iterable[str | Path]) -> list[DocumentChunk]:
        """Process multiple documents and return all chunks combined."""
        all_chunks: list[DocumentChunk] = []
        for p in paths:
            try:
                all_chunks.extend(self.process(p))
            except Exception as exc:
                logger.error("Failed to process '%s': %s", p, exc)
        return all_chunks
