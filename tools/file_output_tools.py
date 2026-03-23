"""
tools/file_output_tools.py
───────────────────────────
Tools for saving agent output as files users can download or reference later.

Supports: plain text, Markdown, Python/code files, JSON, CSV.

Every file is written to the configured UPLOADS_PATH under a ``generated/``
sub-directory so it stays separate from ingested knowledge-base documents.
The tool returns the file path so the agent can cite it in its response.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

# Output directory for all agent-generated files
_OUTPUT_DIR = settings.uploads_path / "generated"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_SAFE_NAME_RE = re.compile(r"[^\w\-]")  # no dots to prevent path traversal via ".."


def _safe_filename(name: str, ext: str) -> str:
    """Sanitise a user-supplied filename and ensure it has the right extension."""
    # Strip directory traversal: take basename only
    name = Path(name).name
    # Split stem from any existing extension to avoid double-extension
    p     = Path(name)
    stem  = p.stem if p.suffix else name
    # Sanitise: keep word chars, hyphens, underscores; no dots in stem
    _STEM_RE = re.compile(r"[^\w\-]")
    stem  = _STEM_RE.sub("_", stem.strip())[:60] or "output"
    return f"{stem}{ext}"


# =============================================================================
#  Schemas
# =============================================================================

class SaveTextInput(BaseModel):
    content:  str = Field(..., description="Text content to save.")
    filename: str = Field("output.txt", description="Desired filename (e.g. report.md).")
    format:   str = Field("txt", description="File format: txt | md | py | js | ts | json | csv | html")


class SaveCodeInput(BaseModel):
    code:     str = Field(..., description="Code content to save.")
    filename: str = Field("script.py", description="Filename with appropriate extension.")
    language: str = Field("python", description="Programming language.")


class SaveJsonInput(BaseModel):
    data:     str = Field(..., description="JSON string to save.")
    filename: str = Field("data.json", description="Output filename.")


# =============================================================================
#  Tools
# =============================================================================

class SaveTextFileTool(BaseTool):
    """
    Save text, Markdown, or code content to a file.

    Use when the user asks to:
      - "Save this as a file"
      - "Write a report and save it"
      - "Create a script and save it"
      - "Export this to Markdown"

    Returns the file path so the user can find it.
    """
    name: str        = "save_text_file"
    description: str = (
        "Save text, Markdown, or prose content to a file. "
        "Use when the user wants to keep or download the output. "
        "Inputs: content (str), filename (str, e.g. 'report.md'), "
        "format (txt|md|py|js|ts|json|csv|html)."
    )
    args_schema: type[BaseModel] = SaveTextInput

    def _run(
        self,
        content:  str,
        filename: str = "output.txt",
        format:   str = "txt",
        **kwargs,
    ) -> str:
        ext_map = {
            "md": ".md", "markdown": ".md",
            "py": ".py", "python": ".py",
            "js": ".js", "javascript": ".js",
            "ts": ".ts", "typescript": ".ts",
            "json": ".json",
            "csv": ".csv",
            "html": ".html",
            "txt": ".txt",
        }
        ext  = ext_map.get(format.lower(), ".txt")
        name = _safe_filename(filename, ext)
        path = _OUTPUT_DIR / name

        # Avoid silent overwrite — add counter if file exists
        counter = 1
        stem    = path.stem
        while path.exists():
            path = _OUTPUT_DIR / f"{stem}_{counter}{ext}"
            counter += 1

        path.write_text(content, encoding="utf-8")
        logger.info("Saved file: %s (%d bytes)", path, len(content.encode("utf-8")))
        return f"File saved: {path}\n({len(content)} characters, {path.stat().st_size} bytes)"


class SaveCodeFileTool(BaseTool):
    """
    Save generated code to a file with the correct extension.

    Use when the CodeAgent produces a script the user wants to keep.
    Automatically detects the extension from the language.
    """
    name: str        = "save_code_file"
    description: str = (
        "Save generated code to a file. Automatically uses the correct "
        "file extension for the language. "
        "Inputs: code (str), filename (str), language (str, default python)."
    )
    args_schema: type[BaseModel] = SaveCodeInput

    _LANG_EXT: dict[str, str] = {
        "python":     ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java":       ".java",
        "c++":        ".cpp",
        "c":          ".c",
        "rust":       ".rs",
        "go":         ".go",
        "sql":        ".sql",
        "bash":       ".sh",
        "shell":      ".sh",
        "html":       ".html",
        "css":        ".css",
    }

    def _run(
        self,
        code:     str,
        filename: str  = "script.py",
        language: str  = "python",
        **kwargs,
    ) -> str:
        ext  = self._LANG_EXT.get(language.lower(), ".py")
        name = _safe_filename(filename, ext)
        path = _OUTPUT_DIR / name

        counter = 1
        stem    = path.stem
        while path.exists():
            path = _OUTPUT_DIR / f"{stem}_{counter}{ext}"
            counter += 1

        path.write_text(code, encoding="utf-8")
        lines = code.count("\n") + 1
        logger.info("Saved code: %s (%d lines)", path, lines)
        return f"Code saved: {path}\n({lines} lines, {language})"


class SaveJsonFileTool(BaseTool):
    """
    Save structured JSON data to a file with pretty-printing.
    Use when the user wants to export data, API responses, or structured results.
    """
    name: str        = "save_json_file"
    description: str = (
        "Save structured JSON data to a .json file with pretty-printing. "
        "Inputs: data (JSON string), filename (str)."
    )
    args_schema: type[BaseModel] = SaveJsonInput

    def _run(self, data: str, filename: str = "data.json", **kwargs) -> str:
        name = _safe_filename(filename, ".json")
        path = _OUTPUT_DIR / name

        # Validate + pretty-print
        try:
            obj        = json.loads(data)
            pretty     = json.dumps(obj, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # Save as-is if not valid JSON
            pretty = data

        counter = 1
        stem    = path.stem
        while path.exists():
            path = _OUTPUT_DIR / f"{stem}_{counter}.json"
            counter += 1

        path.write_text(pretty, encoding="utf-8")
        logger.info("Saved JSON: %s", path)
        return f"JSON saved: {path}\n({len(pretty)} bytes)"


class ListSavedFilesTool(BaseTool):
    """List recently generated files the agent has saved."""
    name: str        = "list_saved_files"
    description: str = (
        "List files previously saved by the agent in this session. "
        "No inputs required."
    )

    def _run(self, **kwargs) -> str:
        files = sorted(_OUTPUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return "No files have been saved yet."
        lines = [f"Saved files ({len(files)} total):"]
        for f in files[:20]:
            size = f.stat().st_size
            lines.append(f"  {f.name}  ({size:,} bytes)  →  {f}")
        return "\n".join(lines)


def get_file_output_tools() -> list[BaseTool]:
    """Return all file-output tools."""
    return [
        SaveTextFileTool(),
        SaveCodeFileTool(),
        SaveJsonFileTool(),
        ListSavedFilesTool(),
    ]
