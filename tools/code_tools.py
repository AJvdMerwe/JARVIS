"""
tools/code_tools.py
────────────────────
LangChain tools for the Code Agent:
  • CodeWriterTool    – generates code from a description.
  • CodeReviewTool    – reviews / explains existing code.
  • CodeExecutorTool  – safely executes Python in a subprocess sandbox.
  • ShellCommandTool  – runs shell commands (opt-in, disabled by default).
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class CodeInput(BaseModel):
    description: str = Field(..., description="What the code should do")
    language: str = Field(default="python", description="Target programming language")
    context: Optional[str] = Field(None, description="Additional context or requirements")


class ReviewInput(BaseModel):
    code: str = Field(..., description="The code to review")
    focus: Optional[str] = Field(None, description="What aspect to focus on (e.g. security, performance)")


class ExecuteInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(default=30, ge=1, le=120, description="Execution timeout in seconds")


# ─────────────────────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────────────────────

class CodeWriterTool(BaseTool):
    """Generate code from a plain-English description."""

    name: str = "code_writer"
    description: str = (
        "Generate code from a plain-English description. "
        "Provide the description of what you want the code to do and the target language."
    )
    args_schema: Type[BaseModel] = CodeInput

    def _run(self, description: str, language: str = "python", context: Optional[str] = None) -> str:
        from core.llm_manager import get_llm

        llm = get_llm()
        prompt = (
            f"Write {language} code that does the following:\n\n"
            f"{description}\n\n"
        )
        if context:
            prompt += f"Additional context:\n{context}\n\n"
        prompt += (
            "Requirements:\n"
            "- Follow best practices and SOLID principles\n"
            "- Add docstrings and comments\n"
            "- Handle edge cases and errors\n"
            "- Return ONLY the code block, no explanation\n"
        )
        response = llm.invoke(prompt)
        return str(response.content)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("Async not implemented")


class CodeReviewTool(BaseTool):
    """Review, explain, or improve existing code."""

    name: str = "code_reviewer"
    description: str = (
        "Review, explain, debug, or improve existing code. "
        "Provide the code and optionally the focus area."
    )
    args_schema: Type[BaseModel] = ReviewInput

    def _run(self, code: str, focus: Optional[str] = None) -> str:
        from core.llm_manager import get_llm

        llm = get_llm()
        focus_str = f" focusing on: {focus}" if focus else ""
        prompt = (
            f"Review the following code{focus_str}.\n\n"
            f"```\n{code}\n```\n\n"
            "Provide:\n"
            "1. A brief explanation of what the code does\n"
            "2. Potential issues or improvements\n"
            "3. A corrected/improved version if needed\n"
        )
        response = llm.invoke(prompt)
        return str(response.content)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class CodeExecutorTool(BaseTool):
    """
    Execute Python code in an isolated subprocess.
    Captures stdout, stderr, and return code.
    """

    name: str = "python_executor"
    description: str = (
        "Execute Python code and return the output. "
        "Use this to test code, compute results, or verify logic."
    )
    args_schema: Type[BaseModel] = ExecuteInput

    # Blocked keywords for basic safety
    _BLOCKED = frozenset([
        "os.remove", "shutil.rmtree", "subprocess.call",
        "__import__('os').system", "open('/etc", "open('/root",
    ])

    def _safety_check(self, code: str) -> Optional[str]:
        for blocked in self._BLOCKED:
            if blocked in code:
                return f"Execution blocked: code contains forbidden pattern '{blocked}'."
        return None

    def _run(self, code: str, timeout: int = 30) -> str:
        violation = self._safety_check(code)
        if violation:
            return violation

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(textwrap.dedent(code))
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            output_parts.append(f"Return code: {result.returncode}")
            return "\n".join(output_parts)
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout} seconds."
        except Exception as exc:
            return f"Execution failed: {exc}"
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


def get_code_tools() -> list[BaseTool]:
    """Return all code-related tools."""
    return [
        CodeWriterTool(),
        CodeReviewTool(),
        CodeExecutorTool(),
    ]
