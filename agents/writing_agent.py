"""
agents/writing_agent.py
────────────────────────
WritingAssistantAgent — long-form writing partner.

Routing triggers (Intent.WRITING)
──────────────────────────────────
  "write an article about", "draft a report on", "help me write",
  "create a blog post", "write an essay", "compose a letter",
  "write a technical document", "draft an email", "outline a"

Workflow
────────
  Standard path (ReAct loop with all writing tools):
    create_outline → draft_section (×N) → assemble_draft → edit_draft
    → export_markdown / export_docx

  Fast path (short requests — poem, email, one-shot content):
    Direct LLM generation without the outline/section pipeline.
    Threshold: target_words ≤ _FAST_PATH_THRESHOLD or specific keywords.

  The agent detects which path is appropriate from the query and kwargs.

Conversation context
────────────────────
  Inherits _augment_query() so "make the intro shorter", "add a section on
  pricing", and "change the tone to formal" all reference the previous draft.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agents.base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory

logger = logging.getLogger(__name__)

# Queries under this estimated word count bypass the outline/section pipeline
_FAST_PATH_THRESHOLD = 300

# Short-form types that always use the fast path
_FAST_PATH_TYPES = frozenset({
    "email", "e-mail", "poem", "haiku", "tweet", "caption",
    "tagline", "slogan", "summary", "blurb", "bio", "headline",
})

_SYSTEM_PROMPT = """You are an expert writing assistant specialised in long-form content creation.

You follow a structured workflow for substantial documents:
1. create_outline — establish structure, headings, and section briefs
2. draft_section  — write each section one at a time with consistent voice
3. assemble_draft — combine sections into a cohesive document
4. edit_draft     — refine for clarity, style, grammar, and flow
5. export_docx or export_markdown — deliver the final file

Your writing principles:
- Match tone to document type (formal for reports, conversational for blogs).
- Write in flowing prose — avoid bullet-point-heavy body sections unless appropriate.
- Open with a strong hook; close with a clear takeaway.
- Use concrete examples, data, and specifics rather than vague generalities.
- Vary sentence length: short for emphasis, longer for explanation.
- After completing a draft, always offer to: adjust tone, expand sections,
  add citations, or export to a different format.
"""

_FAST_PATH_PROMPT = """You are an expert writer. Write the following in a polished, ready-to-use style.
Match the requested format and length precisely. Return ONLY the written content — no preamble, no meta-commentary."""


class WritingAssistantAgent(BaseAgent):
    """
    Long-form writing assistant: outline → draft → edit → export.

    Exposes seven tools: create_outline · draft_section · assemble_draft ·
    edit_draft · export_markdown · export_docx · list_drafts.

    Short requests (emails, poems, short summaries) bypass the outline
    pipeline and use a direct LLM call for speed.
    """

    def __init__(
        self,
        llm:     Optional[BaseChatModel] = None,
        memory:  Optional[AssistantMemory] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(llm=llm, memory=memory, verbose=verbose)
        self._executor = self._build_react_agent(system_prompt=_SYSTEM_PROMPT)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "writing_agent"

    @property
    def description(self) -> str:
        return (
            "Long-form writing assistant for articles, essays, reports, blog posts, "
            "technical documents, emails, and letters. Produces structured outlines, "
            "drafts sections, edits for clarity and style, and exports to DOCX or Markdown."
        )

    def get_tools(self) -> list[BaseTool]:
        from tools.writing_tools import get_writing_tools
        from tools.file_output_tools import SaveTextFileTool, SaveCodeFileTool
        return get_writing_tools() + [SaveTextFileTool(), SaveCodeFileTool()]

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a writing request.

        Extra kwargs
        ------------
        doc_type      : str  — article | essay | report | blog_post | email | letter | …
        target_words  : int  — approximate word count target
        draft_id      : str  — ID to store/retrieve draft under
        fast_path     : bool — force fast (direct LLM) path
        """
        query        = self._augment_query(query)
        doc_type     = kwargs.get("doc_type", self._infer_doc_type(query))
        target_words = int(kwargs.get("target_words", self._infer_word_count(query)))
        draft_id     = kwargs.get("draft_id", "draft")
        force_fast   = bool(kwargs.get("fast_path", False))

        self._logger.info("WritingAssistantAgent handling: %s", query[:80])

        # ── Fast path for short / simple content ─────────────────────────
        if force_fast or self._is_fast_path(query, doc_type, target_words):
            return self._fast_generate(query, doc_type, target_words)

        # ── Full pipeline via ReAct executor ──────────────────────────────
        enriched = (
            f"{query}\n\n"
            f"[Document type: {doc_type} | "
            f"Target: {target_words} words | "
            f"Draft ID: {draft_id}]"
        )

        try:
            result     = self._executor.invoke({"input": enriched})
            output     = result.get("output", "No output generated.")
            tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
            references = self._extract_file_references(tool_calls)

            return AgentResponse(
                output=output,
                agent_name=self.name,
                tool_calls=tool_calls,
                references=references,
                metadata={
                    "doc_type":    doc_type,
                    "target_words": target_words,
                    "draft_id":    draft_id,
                },
            )
        except Exception as exc:
            self._logger.error("WritingAssistantAgent error: %s", exc, exc_info=True)
            return AgentResponse(
                output=f"Writing failed: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_fast_path(self, query: str, doc_type: str, target_words: int) -> bool:
        """
        True when the request is short-form enough to bypass the
        outline/section pipeline.
        """
        if doc_type in _FAST_PATH_TYPES:
            return True
        if target_words <= _FAST_PATH_THRESHOLD:
            return True
        # Regex check for explicit short-form requests
        _SHORT_RE = re.compile(
            r"\b(quick|brief|short|one.?paragraph|one.?line|concise|"
            r"summarise|summarize|abstract|tldr|short email)\b",
            re.IGNORECASE,
        )
        return bool(_SHORT_RE.search(query))

    def _fast_generate(
        self, query: str, doc_type: str, target_words: int
    ) -> AgentResponse:
        """Direct LLM generation for short-form content."""
        prompt = (
            f"{_FAST_PATH_PROMPT}\n\n"
            f"Request: {query}\n"
            f"Type: {doc_type}\n"
            f"Length: approximately {target_words} words"
        )
        try:
            result = self._llm.invoke(prompt)
            output = str(result.content).strip()
            return AgentResponse(
                output=output,
                agent_name=self.name,
                metadata={"doc_type": doc_type, "fast_path": True},
            )
        except Exception as exc:
            return AgentResponse(
                output=f"Writing failed: {exc}",
                agent_name=self.name,
                error=str(exc),
            )

    @staticmethod
    def _infer_doc_type(query: str) -> str:
        """Detect the document type from the query string."""
        mapping = [
            (r"\b(blog post|blog)\b",                   "blog_post"),
            (r"\b(email|e-mail|message)\b",             "email"),
            (r"\b(essay)\b",                            "essay"),
            (r"\b(report|analysis)\b",                  "report"),
            (r"\b(letter)\b",                           "letter"),
            (r"\b(technical|tech doc|documentation)\b", "technical_doc"),
            (r"\b(poem|poetry|haiku)\b",                "poem"),
            (r"\b(article|piece)\b",                    "article"),
        ]
        q_lower = query.lower()
        for pattern, dtype in mapping:
            if re.search(pattern, q_lower):
                return dtype
        return "article"

    @staticmethod
    def _infer_word_count(query: str) -> int:
        """Extract an approximate word count from the query if mentioned."""
        m = re.search(
            r"(\d+)\s*(?:word|words|w\b)",
            query, re.IGNORECASE,
        )
        if m:
            return int(m.group(1))
        # Keyword-based defaults
        kw_map = [
            (r"\b(short|brief|quick)\b",   200),
            (r"\b(medium|moderate)\b",     600),
            (r"\b(long|detailed|in.depth|comprehensive)\b", 1500),
            (r"\b(email|poem|haiku)\b",     150),
        ]
        for pattern, words in kw_map:
            if re.search(pattern, query, re.IGNORECASE):
                return words
        return 800   # default

    def _extract_file_references(
        self, tool_calls: list[tuple]
    ) -> list[str]:
        """Extract saved file paths from tool observations."""
        refs: list[str] = []
        _PATH_RE = re.compile(r"(?:Saved|Exported):\s*(.+?)(?:\n|$)")
        for _, _, obs in tool_calls:
            for m in _PATH_RE.finditer(str(obs)):
                path = m.group(1).strip()
                if path and path not in refs:
                    refs.append(path)
        return refs
