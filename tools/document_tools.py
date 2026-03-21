"""
tools/document_tools.py
────────────────────────
LangChain tools that wrap the DocumentManager for agent use:
  • IngestDocumentTool       – parse and store a document file.
  • SearchDocumentsTool      – semantic search across the knowledge base.
  • ListDocumentsTool        – list all ingested documents.
  • GetFullDocumentTool      – retrieve a full document's text.
  • DeleteDocumentTool       – remove a document from the store.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Shared singleton DocumentManager ────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_doc_manager():
    from document_processing import DocumentManager
    return DocumentManager()


# ── Schemas ──────────────────────────────────────────────────────────────────

class IngestInput(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the document to ingest")


class SearchInput(BaseModel):
    query: str = Field(..., description="Natural language query to search for")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    doc_title: Optional[str] = Field(None, description="Restrict search to a specific document title")


class GetDocInput(BaseModel):
    doc_title: str = Field(..., description="Exact title of the document to retrieve")


class DeleteDocInput(BaseModel):
    doc_title: str = Field(..., description="Exact title of the document to delete")


class BulkIngestDirInput(BaseModel):
    directory: str = Field(...,
                           description="Path to a directory. All supported files are ingested recursively.")
    recursive: bool = Field(default=True,
                            description="Descend into sub-directories (default: True).")
    dry_run: bool = Field(default=False,
                          description="If True, detect file types and plan without writing to the store.")


class BulkIngestFilesInput(BaseModel):
    paths: str = Field(...,
                       description="Comma-separated list of file paths to ingest.")
    dry_run: bool = Field(default=False,
                          description="If True, detect and plan without writing to the store.")


# ── Tools ────────────────────────────────────────────────────────────────────

class IngestDocumentTool(BaseTool):
    """Parse and index a document file (PDF, DOCX, XLSX, PPTX)."""

    name: str = "ingest_document"
    description: str = (
        "Parse and index a document (PDF, DOCX, XLSX, PPTX) into the knowledge base. "
        "Provide the file path. After ingestion, the document is searchable."
    )
    args_schema: Type[BaseModel] = IngestInput

    def _run(self, path: str) -> str:
        dm = _get_doc_manager()
        try:
            added = dm.ingest(path)
            if added == 0:
                return f"Document '{path}' was already in the knowledge base (no new chunks added)."
            return f"Successfully ingested '{path}'. Added {added} new text chunks to the knowledge base."
        except FileNotFoundError:
            return f"File not found: '{path}'. Please check the path and try again."
        except ValueError as exc:
            return f"Unsupported file type: {exc}"
        except Exception as exc:
            return f"Failed to ingest document: {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class SearchDocumentsTool(BaseTool):
    """Semantic search across all ingested documents."""

    name: str = "search_documents"
    description: str = (
        "Search the knowledge base for relevant document sections using natural language. "
        "Returns the most relevant passages with references (document name, page, section). "
        "Use this to answer questions from documents or find specific information."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, k: int = 5, doc_title: Optional[str] = None) -> str:
        dm = _get_doc_manager()

        if dm.total_chunks == 0:
            return (
                "The knowledge base is empty. Please ingest documents first "
                "using the ingest_document tool."
            )

        results = dm.search(query, k=k, doc_title=doc_title)
        if not results:
            scope = f" in '{doc_title}'" if doc_title else ""
            return f"No relevant sections found{scope} for: '{query}'"

        return dm.format_search_results(results)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class ListDocumentsTool(BaseTool):
    """List all documents currently in the knowledge base."""

    name: str = "list_documents"
    description: str = (
        "List all documents that have been ingested into the knowledge base. "
        "Returns document titles, file types, and paths."
    )

    def _run(self, tool_input: str = "") -> str:  # noqa: ARG002
        dm = _get_doc_manager()
        docs = dm.list_documents()
        if not docs:
            return "No documents in the knowledge base yet."

        lines = [f"## Knowledge Base ({len(docs)} documents, {dm.total_chunks} total chunks)\n"]
        for doc in docs:
            lines.append(
                f"- **{doc['doc_title']}**  ({doc.get('doctype', '?').upper()})  "
                f"`{doc['doc_path']}`"
            )
        return "\n".join(lines)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class GetFullDocumentTool(BaseTool):
    """Retrieve the full reconstructed text of a document."""

    name: str = "get_full_document"
    description: str = (
        "Retrieve the full text of a specific document from the knowledge base. "
        "Use the exact document title as shown by list_documents."
    )
    args_schema: Type[BaseModel] = GetDocInput

    def _run(self, doc_title: str) -> str:
        dm = _get_doc_manager()
        text = dm.get_full_document_text(doc_title)
        # Truncate to avoid token explosion in the agent context
        if len(text) > 8000:
            text = text[:8000] + f"\n\n[... truncated — {len(text) - 8000} more characters]"
        return text

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class DeleteDocumentTool(BaseTool):
    """Remove a document from the knowledge base."""

    name: str = "delete_document"
    description: str = (
        "Remove a document and all its chunks from the knowledge base. "
        "Use the exact document title as shown by list_documents."
    )
    args_schema: Type[BaseModel] = DeleteDocInput

    def _run(self, doc_title: str) -> str:
        dm = _get_doc_manager()
        removed = dm.delete_document(doc_title)
        if removed == 0:
            return f"Document '{doc_title}' was not found in the knowledge base."
        return f"Removed '{doc_title}' ({removed} chunks deleted)."

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


def get_document_tools() -> list[BaseTool]:
    """Return all document-related tools."""
    return [
        IngestDocumentTool(),
        SearchDocumentsTool(),
        ListDocumentsTool(),
        GetFullDocumentTool(),
        DeleteDocumentTool(),
        BulkIngestDirectoryTool(),
        BulkIngestFilesTool(),
    ]


# ── Bulk ingestion tools ──────────────────────────────────────────────────────

class BulkIngestDirectoryTool(BaseTool):
    """
    Ingest an entire directory of documents into the knowledge base.

    Applies type-appropriate extraction per file:
      • PDF / DOCX / XLSX / PPTX  → Docling (OCR, tables, headings)
      • TXT / MD / CSV / HTML     → text splitter
      • JSON / XML                → structured flattener
      • Unknown / empty           → skipped with a warning

    Content-hash deduplication prevents re-ingesting unchanged files.
    Returns a structured report with per-file outcomes and summary stats.
    """

    name: str        = "bulk_ingest_directory"
    description: str = (
        "Ingest all documents in a directory into the knowledge base. "
        "Automatically detects file type (PDF, DOCX, XLSX, PPTX, TXT, MD, "
        "CSV, HTML, JSON, XML) and applies the optimal extraction strategy. "
        "Supports dry-run mode to preview without writing. "
        "Use for: 'Index all files in ./reports/', 'Upload the documents folder'."
    )
    args_schema: Type[BaseModel] = BulkIngestDirInput

    def _run(self, directory: str, recursive: bool = True,
             dry_run: bool = False) -> str:
        from document_processing.mass_uploader import MassUploader
        try:
            uploader = MassUploader()
            report   = uploader.upload_directory(
                directory, recursive=recursive, dry_run=dry_run
            )
            return report.summary()
        except NotADirectoryError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            logger.error("BulkIngestDirectoryTool failed: %s", exc)
            return f"Bulk ingest failed: {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class BulkIngestFilesTool(BaseTool):
    """
    Ingest a specific list of document files into the knowledge base.

    Accepts a comma-separated list of paths.  Each file is type-detected
    independently and processed with the optimal extraction strategy.
    Identical files (same content hash) are skipped automatically.
    """

    name: str        = "bulk_ingest_files"
    description: str = (
        "Ingest a list of specific document files into the knowledge base. "
        "Provide a comma-separated list of file paths. "
        "Each file type is auto-detected (PDF, DOCX, TXT, CSV, JSON, etc.). "
        "Use for: 'Ingest these files: report.pdf, data.xlsx, notes.md'."
    )
    args_schema: Type[BaseModel] = BulkIngestFilesInput

    def _run(self, paths: str, dry_run: bool = False) -> str:
        from document_processing.mass_uploader import MassUploader
        path_list = [p.strip() for p in paths.split(",") if p.strip()]
        if not path_list:
            return "No file paths provided."
        try:
            uploader = MassUploader()
            report   = uploader.upload_files(path_list, dry_run=dry_run)
            return report.summary()
        except Exception as exc:
            logger.error("BulkIngestFilesTool failed: %s", exc)
            return f"Bulk ingest failed: {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError