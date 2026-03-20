"""
agents/document_agent.py
─────────────────────────
Specialised agent for document Q&A and knowledge base management:
  • Ingest documents (PDF, DOCX, XLSX, PPTX) via Docling.
  • Semantic search with precise chunk-level references.
  • Chat with a specific document (scoped retrieval).
  • List and manage the knowledge base.
  • Always surfaces references: document title, page, and section heading.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from tools.document_tools import _get_doc_manager, get_document_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise document analysis assistant with access
to a knowledge base of ingested documents.

Your core behaviours:
1. ALWAYS include references when answering from documents.
   Format: [Source: <document title> › Page <n> › <section>]
2. When searching, use multiple queries if the first yields insufficient results.
3. If the user asks about a specific document, use doc_title filter in search.
4. For broad questions, synthesise across multiple documents and cite each one.
5. If a document has not been ingested, tell the user and offer to ingest it.
6. Never hallucinate document content — only use what the search tools return.
7. When listing documents, always show the full inventory first.

Reference format examples:
  [Source: Q3_Report › Page 4 › Revenue Analysis]
  [Source: Employee_Handbook › Page 12 › Benefits]
"""


class DocumentAgent(BaseAgent):
    """Agent for document ingestion, Q&A, and knowledge base management."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[AssistantMemory] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._executor = self._build_react_agent(system_prompt=_SYSTEM_PROMPT)

    # ── BaseAgent interface ──────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "document_agent"

    @property
    def description(self) -> str:
        return (
            "Ingest, search, and chat with documents (PDF, DOCX, XLSX, PPTX). "
            "Use for: answering questions from uploaded files, summarising documents, "
            "finding specific information in the knowledge base, document management."
        )

    def get_tools(self) -> list[BaseTool]:
        return get_document_tools()

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a document-related request.

        Args:
            query:      The user's question or command.
            doc_title:  Optional — restrict context to one document.
            file_path:  Optional — auto-ingest before answering.

        Returns:
            AgentResponse with answer and rich document references.
        """
        doc_title = kwargs.get("doc_title")
        file_path = kwargs.get("file_path")

        full_query = query
        if doc_title:
            full_query = f"[Focus on document: '{doc_title}'] {query}"
        if file_path:
            full_query = (
                f"First ingest the document at '{file_path}', "
                f"then answer: {query}"
            )

        self._logger.info("DocumentAgent handling: %s", query[:80])

        try:
            result = self._executor.invoke({"input": full_query})
            output = result.get("output", "No answer generated.")
            tool_calls = self._extract_tool_calls(
                result.get("intermediate_steps", [])
            )
            references = self._extract_doc_references(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
                metadata={"doc_title": doc_title, "file_path": file_path},
            )
        except Exception as exc:
            self._logger.error("DocumentAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"Error processing document request: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Convenience methods ──────────────────────────────────────────────────

    def ingest(self, file_path: str) -> str:
        """Directly ingest a document (no LLM round-trip needed)."""
        dm = _get_doc_manager()
        added = dm.ingest(file_path)
        return f"Ingested '{file_path}': {added} new chunks added."

    def list_documents(self) -> list[dict]:
        """Return the knowledge base inventory."""
        return _get_doc_manager().list_documents()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_doc_references(
        tool_calls: list[tuple[str, str, str]],
    ) -> list[str]:
        """
        Pull rich document references from search_documents tool outputs.
        Looks for the '**[N] <reference>**' pattern the DocumentManager emits.
        """
        import re

        ref_re = re.compile(r"\*\*\[\d+\]\s+(.+?)\*\*")
        refs: list[str] = []
        seen: set[str] = set()

        for tool_name, _, output in tool_calls:
            if tool_name == "search_documents":
                for match in ref_re.finditer(output):
                    ref = match.group(1).strip()
                    if ref not in seen:
                        refs.append(ref)
                        seen.add(ref)

        return refs
