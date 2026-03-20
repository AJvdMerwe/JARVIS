"""
agents/code_agent.py
─────────────────────
Specialised agent for all code-related tasks:
  • Write code from natural language descriptions.
  • Review, explain, or debug existing code.
  • Execute Python snippets and return output.
  • Suggest architecture and design patterns.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base_agent import AgentResponse, BaseAgent
from core.memory import AssistantMemory
from tools.code_tools import get_code_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert software engineer and code assistant.
You follow SOLID principles, write clean and well-documented code, and always
consider edge cases, error handling, and security implications.

When writing code:
- Choose the most appropriate language unless specified.
- Add docstrings and inline comments.
- Handle errors gracefully.
- Suggest tests where applicable.

When reviewing code:
- Identify bugs, security issues, and performance problems.
- Suggest concrete improvements with reasoning.

Always use the available tools rather than generating code from memory alone.
"""


class CodeAgent(BaseAgent):
    """Agent that writes, reviews, and executes code."""

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
        return "code_agent"

    @property
    def description(self) -> str:
        return (
            "Write, review, debug, and execute code. "
            "Use for programming tasks, code generation, code explanation, "
            "algorithm design, and running Python snippets."
        )

    def get_tools(self) -> list[BaseTool]:
        return get_code_tools()

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        """
        Handle a code-related request by routing directly to the appropriate
        tool(s) based on detected task type, rather than delegating all
        decisions to the ReAct executor.

        Routing strategy
        ----------------
        1. **Review / debug / explain** — query contains an existing code block
           (fenced ``` or heavily indented text) or uses review-intent keywords
           (debug, fix, explain, what does, review, improve).
           → ``code_reviewer`` with an inferred focus area.

        2. **Execute** — query asks to run, test, or evaluate code that is
           already present in the query, or asks to verify output.
           → ``python_executor``, then formats the output with the LLM.

        3. **Write + verify** — a "write" task targeting Python where the
           request includes "test it", "verify", "run it", or "does it work".
           → ``code_writer`` to generate, then ``python_executor`` to run,
           then synthesise both into the final response.

        4. **Write** — all other "write / create / implement / build" requests.
           → ``code_writer`` with the detected language and a cleaned
           description extracted from the query.

        5. **ReAct fallback** — when task type is ambiguous or any direct
           tool call produces empty / error output, the full ReAct executor
           is invoked so the LLM can iterate freely with all tools.

        Args:
            query:    Natural-language task description or code block.
            language: Optional language hint, e.g. ``"typescript"``.

        Returns:
            AgentResponse with generated / reviewed code, execution output,
            and the full tool call log.
        """
        language_hint = kwargs.get("language", "") or self._detect_language(query)
        self._logger.info("CodeAgent handling: %s", query[:80])

        tools        = {t.name: t for t in self.get_tools()}
        writer       = tools.get("code_writer")
        reviewer     = tools.get("code_reviewer")
        executor     = tools.get("python_executor")

        tool_calls: list[tuple[str, str, str]] = []

        try:
            task = self._classify_task(query)
            self._logger.debug("Classified task: %s", task)

            # ─────────────────────────────────────────────────────────────────
            # Route 1 — Review / debug / explain existing code
            # ─────────────────────────────────────────────────────────────────
            if task in ("review", "debug", "explain") and reviewer:
                code_block = self._extract_code_block(query)
                focus      = self._infer_review_focus(task, query)

                review_input = code_block if code_block else query
                review_out   = reviewer._run(code=review_input, focus=focus)
                tool_calls.append(("code_reviewer", review_input[:200], review_out))

                if self._is_sufficient(review_out):
                    return AgentResponse(
                        output=review_out,
                        agent_name=self.name,
                        tool_calls=tool_calls,
                    )

            # ─────────────────────────────────────────────────────────────────
            # Route 2 — Execute code that's already in the query
            # ─────────────────────────────────────────────────────────────────
            elif task == "execute" and executor:
                code_block = self._extract_code_block(query)
                if not code_block:
                    # Nothing to execute — fall through to write + run
                    task = "write_and_verify"
                else:
                    exec_out = executor._run(code=code_block, timeout=30)
                    tool_calls.append(("python_executor", code_block[:200], exec_out))

                    answer = self._format_execution_result(query, code_block, exec_out)
                    return AgentResponse(
                        output=answer,
                        agent_name=self.name,
                        tool_calls=tool_calls,
                    )

            # ─────────────────────────────────────────────────────────────────
            # Route 3 — Write code and verify by running it (Python only)
            # ─────────────────────────────────────────────────────────────────
            if task == "write_and_verify" and writer and executor:
                description = self._clean_description(query)
                lang        = language_hint or "python"

                # Generate the code.
                generated = writer._run(
                    description=description,
                    language=lang,
                    context=self._build_context(query, language_hint),
                )
                tool_calls.append(("code_writer", description[:200], generated))

                # Execute if Python (other languages can't run in the sandbox).
                code_to_run = self._extract_code_block(generated) or generated
                exec_out    = executor._run(code=code_to_run, timeout=30)
                tool_calls.append(("python_executor", code_to_run[:200], exec_out))

                # If execution raised an error, ask the reviewer to fix it.
                if self._execution_failed(exec_out) and reviewer:
                    fix_prompt = (
                        f"This code produced an error when executed.\n\n"
                        f"Code:\n{code_to_run}\n\n"
                        f"Error:\n{exec_out}\n\n"
                        f"Fix the code so it runs correctly."
                    )
                    fixed = reviewer._run(code=fix_prompt, focus="bug fix")
                    tool_calls.append(("code_reviewer", "fix after error", fixed))

                    # Try to run the fixed version.
                    fixed_code = self._extract_code_block(fixed) or fixed
                    if fixed_code and fixed_code != code_to_run:
                        fixed_exec = executor._run(code=fixed_code, timeout=30)
                        tool_calls.append((
                            "python_executor", fixed_code[:200], fixed_exec
                        ))
                        output = self._synthesise_write_verify(
                            query, fixed_code, fixed_exec, was_fixed=True
                        )
                    else:
                        output = self._synthesise_write_verify(
                            query, code_to_run, exec_out, was_fixed=False
                        )
                else:
                    output = self._synthesise_write_verify(
                        query, code_to_run, exec_out, was_fixed=False
                    )

                return AgentResponse(
                    output=output,
                    agent_name=self.name,
                    tool_calls=tool_calls,
                )

            # ─────────────────────────────────────────────────────────────────
            # Route 4 — Write code (no execution required)
            # ─────────────────────────────────────────────────────────────────
            elif task == "write" and writer:
                description = self._clean_description(query)
                lang        = language_hint or self._detect_language(query) or "python"

                generated = writer._run(
                    description=description,
                    language=lang,
                    context=self._build_context(query, language_hint),
                )
                tool_calls.append(("code_writer", description[:200], generated))

                if self._is_sufficient(generated):
                    return AgentResponse(
                        output=generated,
                        agent_name=self.name,
                        tool_calls=tool_calls,
                    )

            # ─────────────────────────────────────────────────────────────────
            # Route 5 — ReAct fallback
            # ─────────────────────────────────────────────────────────────────
            self._logger.info(
                "Falling back to ReAct executor "
                "(task=%s, tool_calls so far=%d).",
                task, len(tool_calls),
            )
            return self._run_react(query, language_hint)

        except Exception as exc:
            self._logger.error("CodeAgent error: %s", exc, exc_info=True)
            try:
                self._logger.info("Attempting ReAct fallback after error.")
                return self._run_react(query, language_hint)
            except Exception:
                return AgentResponse(
                    output=(
                        f"I encountered an error while processing your "
                        f"code request: {exc}"
                    ),
                    agent_name=self.name,
                    error=str(exc),
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_react(self, query: str, language_hint: str = "") -> AgentResponse:
        """Full ReAct executor fallback — LLM decides tool usage iteratively."""
        full_query = query
        if language_hint:
            full_query = f"[Language: {language_hint}] {query}"
        result     = self._executor.invoke({"input": full_query})
        output     = result.get("output", "No output generated.")
        tool_calls = self._extract_tool_calls(result.get("intermediate_steps", []))
        return AgentResponse(
            output=output,
            agent_name=self.name,
            tool_calls=tool_calls,
        )

    # ── Task classification ───────────────────────────────────────────────────

    _WRITE_PATTERNS = re.compile(
        r"\b(write|create|generate|implement|build|make|code|develop|"
        r"design|scaffold|produce|add|define|construct)\b",
        re.IGNORECASE,
    )
    _REVIEW_PATTERNS = re.compile(
        r"\b(review|explain|what does|what is this|describe|"
        r"improve|optimise|optimize|refactor|clean up|simplify|"
        r"is this correct|check (this|the|my)|analyse|analyze|"
        r"look at|read (this|the|my))\b",
        re.IGNORECASE,
    )
    _DEBUG_PATTERNS = re.compile(
        r"\b(debug|fix|find (the )?bug|why (is|does|isn.t|doesn.t)|"
        r"not working|broken|fails?|error|exception|traceback|"
        r"solve (this|the)|wrong output|unexpected)\b",
        re.IGNORECASE,
    )
    _EXPLAIN_PATTERNS = re.compile(
        r"\b(explain|how does|how do|walk me through|step by step|"
        r"what does this do|tell me about (this|the)|"
        r"understand|help me understand|comment|annotate)\b",
        re.IGNORECASE,
    )
    _EXECUTE_PATTERNS = re.compile(
        r"\b(run|execute|eval|test|try|output|what (does it|will it) (print|return|output)|"
        r"compute|calculate (using|with) (this|the) (code|script)|"
        r"show (me )?(the )?result)\b",
        re.IGNORECASE,
    )
    _VERIFY_PATTERNS = re.compile(
        r"\b(and (run|test|verify|execute|check)|"
        r"then (run|test|verify|execute)|"
        r"does it work|make sure it works|"
        r"verify (it|the (output|result))|"
        r"show (the )?output|run it)\b",
        re.IGNORECASE,
    )

    def _classify_task(self, query: str) -> str:
        """
        Classify the query into one of:
        ``write`` · ``write_and_verify`` · ``review`` · ``debug``
        ``explain`` · ``execute`` · ``unknown``
        """
        has_code  = bool(self._extract_code_block(query))
        # Default to Python when no other language is explicitly mentioned
        _OTHER_LANGS = {
            "javascript", "typescript", "java", "rust", "c++", "golang",
            "go ", "ruby", "swift", "kotlin", "php", "scala", "haskell",
        }
        is_python = (
                "python" in query.lower()
                or not any(kw in query.lower() for kw in _OTHER_LANGS)
        )

        if has_code:
            if self._EXECUTE_PATTERNS.search(query):
                return "execute"
            if self._DEBUG_PATTERNS.search(query):
                return "debug"
            if self._EXPLAIN_PATTERNS.search(query):
                return "explain"
            if self._REVIEW_PATTERNS.search(query):
                return "review"
            # Code present but no clear intent → review it
            return "review"

        if self._DEBUG_PATTERNS.search(query):
            return "debug"
        if self._EXPLAIN_PATTERNS.search(query) and not self._WRITE_PATTERNS.search(query):
            return "explain"
        if self._REVIEW_PATTERNS.search(query) and not self._WRITE_PATTERNS.search(query):
            return "review"

        # Write task — check for verify/run intent first (no existing code block)
        if self._WRITE_PATTERNS.search(query):
            wants_run = (
                    self._VERIFY_PATTERNS.search(query)
                    or self._EXECUTE_PATTERNS.search(query)
            )
            if is_python and wants_run:
                return "write_and_verify"
            return "write"

        if self._EXECUTE_PATTERNS.search(query):
            return "execute"

        return "unknown"

    # ── Code extraction ───────────────────────────────────────────────────────

    _FENCED_BLOCK = re.compile(
        r"```(?:\w+)?\n?(.*?)```",
        re.DOTALL,
    )
    _INDENTED_BLOCK = re.compile(
        r"(?:^|\n)((?:[ \t]{4,}[^\n]+\n?){3,})",
    )

    @classmethod
    def _extract_code_block(cls, text: str) -> str:
        """
        Return the first fenced code block from the text, or the first
        block of 3+ lines indented by ≥4 spaces. Returns empty string
        if no code block is found.
        """
        fenced = cls._FENCED_BLOCK.search(text)
        if fenced:
            return fenced.group(1).strip()
        indented = cls._INDENTED_BLOCK.search(text)
        if indented:
            return indented.group(1).strip()
        return ""

    # ── Language detection ────────────────────────────────────────────────────

    _LANGUAGE_KEYWORDS: dict[str, list[str]] = {
        "python":     ["python", ".py", "pandas", "numpy", "django", "flask", "pydantic"],
        "javascript": ["javascript", "js", "node", "nodejs", "react", "vue", "express"],
        "typescript": ["typescript", "ts", ".tsx", ".ts"],
        "java":       ["java", "spring", "maven", "gradle", "jvm"],
        "rust":       ["rust", "cargo", ".rs"],
        "go":         ["golang", " go ", "goroutine"],
        "c++":        ["c++", "cpp", "#include", "std::"],
        "sql":        ["sql", "select", "insert into", "create table"],
        "bash":       ["bash", "shell", "sh ", "#!/"],
        "ruby":       ["ruby", "rails", ".rb"],
        "swift":      ["swift", "xcode", ".swift"],
    }

    @classmethod
    def _detect_language(cls, query: str) -> str:
        """
        Return the first language keyword matched in the query, or empty
        string when none is detected.
        """
        q = query.lower()
        for lang, keywords in cls._LANGUAGE_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return lang
        return ""

    # ── Review focus inference ────────────────────────────────────────────────

    @staticmethod
    def _infer_review_focus(task: str, query: str) -> str:
        """Return a short focus string for CodeReviewTool based on task type."""
        if task == "debug":
            return "bug identification and fix"
        if task == "explain":
            return "explanation and documentation"
        q = query.lower()
        if any(kw in q for kw in ["security", "vulnerability", "injection"]):
            return "security"
        if any(kw in q for kw in ["performance", "speed", "fast", "slow", "optimis"]):
            return "performance"
        if any(kw in q for kw in ["clean", "refactor", "simplif", "readab"]):
            return "code quality and readability"
        if any(kw in q for kw in ["test", "coverage", "unit test"]):
            return "testability"
        return "general review"

    # ── Description cleaning ──────────────────────────────────────────────────

    _WRITE_FILLER = re.compile(
        r"(?i)^(please\s+)?(can you\s+)?(write|create|generate|implement|build|"
        r"make|code|develop|design|produce|add|define|construct)\s+(me\s+)?(a\s+|an\s+)?",
    )

    @classmethod
    def _clean_description(cls, query: str) -> str:
        """
        Strip imperative filler words from a write-task query so the
        CodeWriterTool receives a clean description rather than a full
        natural-language sentence.
        """
        cleaned = cls._WRITE_FILLER.sub("", query).strip().rstrip(".")
        return cleaned or query

    @staticmethod
    def _build_context(query: str, language_hint: str) -> Optional[str]:
        """Build an optional context string for CodeWriterTool."""
        parts: list[str] = []
        if language_hint:
            parts.append(f"Preferred language: {language_hint}")
        # Surface any explicit constraints mentioned in the query
        for phrase in ("must use", "should use", "requires", "without", "no external"):
            idx = query.lower().find(phrase)
            if idx != -1:
                parts.append(query[idx:idx + 100].split("\n")[0].strip())
        return "\n".join(parts) if parts else None

    # ── Execution helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _execution_failed(exec_output: str) -> bool:
        """
        Return True if the executor output indicates a runtime failure.
        Matches Python tracebacks, non-zero return codes, and timeout messages.
        """
        failure_signals = [
            "Traceback (most recent call last)",
            "Error:",
            "Exception:",
            "Return code: 1",
            "Return code: 2",
            "timed out",
            "Execution blocked",
            "Execution failed",
            "SyntaxError",
            "NameError",
            "TypeError",
            "AttributeError",
            "ImportError",
            "ModuleNotFoundError",
        ]
        return any(sig in exec_output for sig in failure_signals)

    def _format_execution_result(
            self,
            query:     str,
            code:      str,
            exec_out:  str,
    ) -> str:
        """Ask the LLM to produce a readable summary of the execution result."""
        prompt = (
            f"The user asked: {query}\n\n"
            f"The following Python code was executed:\n"
            f"```python\n{code}\n```\n\n"
            f"Execution output:\n{exec_out}\n\n"
            f"Summarise what happened in plain language. "
            f"If there was an error, explain it clearly. "
            f"Show the code and its output in your response."
        )
        response = self._llm.invoke(prompt)
        return str(response.content).strip()

    def _synthesise_write_verify(
            self,
            query:     str,
            code:      str,
            exec_out:  str,
            was_fixed: bool,
    ) -> str:
        """
        Combine generated code and its execution output into a single
        coherent response.
        """
        fix_note = " (fixed after an initial error)" if was_fixed else ""
        prompt = (
            f"The user asked: {query}\n\n"
            f"Here is the Python code{fix_note}:\n"
            f"```python\n{code}\n```\n\n"
            f"Execution result:\n{exec_out}\n\n"
            f"Present the code and its output in a clear, well-formatted response. "
            f"Explain what the code does and confirm it works correctly. "
            f"If the output shows an error, acknowledge it and suggest next steps."
        )
        response = self._llm.invoke(prompt)
        return str(response.content).strip()

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_sufficient(text: str, min_chars: int = 50) -> bool:
        """Return True when the text has enough content to be a real answer."""
        return len(text.strip()) >= min_chars