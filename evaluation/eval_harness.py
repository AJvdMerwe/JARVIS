"""
evaluation/eval_harness.py
───────────────────────────
Offline evaluation harness for testing agent quality without a live LLM.

Concepts:
  • EvalCase      – a single test: input query → expected output criteria.
  • EvalCriteria  – flexible pass/fail rules (contains, regex, LLM judge, etc.).
  • EvalResult    – outcome of running one EvalCase.
  • EvalSuite     – a named collection of EvalCases.
  • Evaluator     – runs a suite against a real agent and reports results.

Usage::

    suite = EvalSuite("code_agent_smoke")
    suite.add(EvalCase(
        query       = "Write a Python function that returns the factorial of n",
        criteria    = [Contains("def factorial"), Contains("return"), NoError()],
        agent_hint  = "code",
    ))

    evaluator = Evaluator()
    report = evaluator.run(suite)
    report.print_summary()

The LLMJudge criterion asks the configured LLM to rate whether the actual
response satisfies a natural-language specification.  All other criteria
are deterministic and run fully offline.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents import AgentResponse, Orchestrator


# ─────────────────────────────────────────────────────────────────────────────
#  Criteria
# ─────────────────────────────────────────────────────────────────────────────

class EvalCriteria(ABC):
    """Abstract base for a single pass/fail assertion on an AgentResponse."""

    @abstractmethod
    def check(self, response: AgentResponse) -> tuple[bool, str]:
        """
        Returns (passed: bool, reason: str).
        ``reason`` is shown in the report regardless of pass/fail.
        """


class Contains(EvalCriteria):
    """Assert that the output contains a given substring (case-insensitive by default)."""

    def __init__(self, substring: str, case_sensitive: bool = False) -> None:
        self.substring = substring
        self.case_sensitive = case_sensitive

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        text = response.output if self.case_sensitive else response.output.lower()
        needle = self.substring if self.case_sensitive else self.substring.lower()
        passed = needle in text
        return passed, f"Contains({self.substring!r}): {'✓' if passed else '✗ not found'}"


class NotContains(EvalCriteria):
    """Assert that the output does NOT contain a given substring."""

    def __init__(self, substring: str, case_sensitive: bool = False) -> None:
        self.substring = substring
        self.case_sensitive = case_sensitive

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        text = response.output if self.case_sensitive else response.output.lower()
        needle = self.substring if self.case_sensitive else self.substring.lower()
        passed = needle not in text
        return passed, f"NotContains({self.substring!r}): {'✓' if passed else '✗ found'}"


class MatchesRegex(EvalCriteria):
    """Assert that the output matches a regex pattern."""

    def __init__(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        self.pattern = pattern
        self.flags = flags
        self._re = re.compile(pattern, flags)

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        passed = bool(self._re.search(response.output))
        return passed, f"Regex({self.pattern!r}): {'✓' if passed else '✗ no match'}"


class NoError(EvalCriteria):
    """Assert that the response has no error field."""

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        passed = response.error is None
        reason = "NoError: ✓" if passed else f"NoError: ✗ error='{response.error}'"
        return passed, reason


class UsedTools(EvalCriteria):
    """Assert that the agent called at least one tool (or a specific tool)."""

    def __init__(self, tool_name: Optional[str] = None) -> None:
        self.tool_name = tool_name

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        if self.tool_name:
            passed = any(t[0] == self.tool_name for t in response.tool_calls)
            return passed, f"UsedTool({self.tool_name!r}): {'✓' if passed else '✗'}"
        passed = len(response.tool_calls) > 0
        return passed, f"UsedAnyTool: {'✓' if passed else '✗ no tools called'}"


class HasReferences(EvalCriteria):
    """Assert that the response includes at least N source references."""

    def __init__(self, min_refs: int = 1) -> None:
        self.min_refs = min_refs

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        n = len(response.references)
        passed = n >= self.min_refs
        return passed, f"HasReferences(min={self.min_refs}): {n} found {'✓' if passed else '✗'}"


class MinLength(EvalCriteria):
    """Assert that the output is at least N characters long."""

    def __init__(self, min_chars: int) -> None:
        self.min_chars = min_chars

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        n = len(response.output)
        passed = n >= self.min_chars
        return passed, f"MinLength({self.min_chars}): {n} chars {'✓' if passed else '✗'}"


class AgentIs(EvalCriteria):
    """Assert that the response came from the expected agent."""

    def __init__(self, expected_agent: str) -> None:
        self.expected = expected_agent

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        passed = response.agent_name == self.expected
        return passed, f"AgentIs({self.expected!r}): got '{response.agent_name}' {'✓' if passed else '✗'}"


class LLMJudge(EvalCriteria):
    """
    Ask the LLM to judge whether the response satisfies a specification.
    This is the only criterion that requires a live LLM.
    """

    def __init__(self, specification: str) -> None:
        self.specification = specification

    def check(self, response: AgentResponse) -> tuple[bool, str]:
        from core.llm_manager import get_llm

        llm = get_llm()
        prompt = (
            f"You are a strict evaluator. Respond ONLY with 'PASS' or 'FAIL'.\n\n"
            f"Specification: {self.specification}\n\n"
            f"Response to evaluate:\n{response.output[:1500]}\n\n"
            "Does the response satisfy the specification? Answer PASS or FAIL."
        )
        try:
            result = llm.invoke(prompt)
            verdict = str(result.content).strip().upper()
            passed = verdict.startswith("PASS")
            return passed, f"LLMJudge: {verdict}"
        except Exception as exc:
            return False, f"LLMJudge: ERROR ({exc})"


# ─────────────────────────────────────────────────────────────────────────────
#  EvalCase + EvalResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """
    A single evaluation test case.

    Args:
        query:      The input query to send to the agent.
        criteria:   List of pass/fail criteria to check the response against.
        agent_hint: Optional forced intent (code/news/search/document).
        description: Human-readable description shown in the report.
        tags:       Optional tags for filtering (e.g. ["smoke", "regression"]).
    """
    query:       str
    criteria:    list[EvalCriteria]
    agent_hint:  Optional[str]     = None
    description: str               = ""
    tags:        list[str]         = field(default_factory=list)


@dataclass
class EvalResult:
    """Outcome of running a single EvalCase."""
    case:        EvalCase
    response:    Optional[AgentResponse]
    latency_ms:  float
    passed:      bool
    checks:      list[tuple[bool, str]]   # (passed, reason) per criterion
    error:       Optional[str] = None

    def summary_line(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        desc = self.case.description or self.case.query[:60]
        return f"  {status}  [{self.latency_ms:6.0f}ms]  {desc}"


# ─────────────────────────────────────────────────────────────────────────────
#  EvalSuite
# ─────────────────────────────────────────────────────────────────────────────

class EvalSuite:
    """A named, ordered collection of EvalCases."""

    def __init__(self, name: str) -> None:
        self.name  = name
        self.cases: list[EvalCase] = []

    def add(self, case: EvalCase) -> "EvalSuite":
        self.cases.append(case)
        return self

    def filter_tags(self, *tags: str) -> "EvalSuite":
        """Return a new EvalSuite containing only cases matching any of the given tags."""
        filtered = EvalSuite(self.name + "_filtered")
        for c in self.cases:
            if any(t in c.tags for t in tags):
                filtered.cases.append(c)
        return filtered

    def __len__(self) -> int:
        return len(self.cases)


# ─────────────────────────────────────────────────────────────────────────────
#  EvalReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    suite_name: str
    results:    list[EvalResult]
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total(self)    -> int: return len(self.results)
    @property
    def passed(self)   -> int: return sum(1 for r in self.results if r.passed)
    @property
    def failed(self)   -> int: return self.total - self.passed
    @property
    def pass_rate(self)-> float: return self.passed / self.total if self.total else 0.0
    @property
    def avg_latency(self) -> float:
        return sum(r.latency_ms for r in self.results) / self.total if self.total else 0.0

    def print_summary(self) -> None:
        bar = "═" * 60
        print(f"\n{bar}")
        print(f"  Eval Suite: {self.suite_name}")
        print(f"  Ran at:     {self.started_at}")
        print(bar)
        for r in self.results:
            print(r.summary_line())
            for passed, reason in r.checks:
                mark = "  ✓" if passed else "  ✗"
                print(f"      {mark} {reason}")
        print(bar)
        print(f"  Result: {self.passed}/{self.total} passed  ({self.pass_rate:.0%})  "
              f"avg latency {self.avg_latency:.0f}ms")
        print(bar)

    def to_json(self) -> dict:
        return {
            "suite":      self.suite_name,
            "started_at": self.started_at,
            "total":      self.total,
            "passed":     self.passed,
            "failed":     self.failed,
            "pass_rate":  self.pass_rate,
            "avg_latency_ms": self.avg_latency,
            "results": [
                {
                    "query":   r.case.query,
                    "passed":  r.passed,
                    "latency": r.latency_ms,
                    "checks":  [{"passed": p, "reason": reason} for p, reason in r.checks],
                    "error":   r.error,
                }
                for r in self.results
            ],
        }

    def save(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        print(f"  Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs an EvalSuite against the real Orchestrator and produces an EvalReport.

    Args:
        session_id: Orchestrator session to use (isolated from user sessions).
        stop_on_fail: Abort remaining tests after the first failure.
    """

    def __init__(
        self,
        session_id: str = "eval",
        stop_on_fail: bool = False,
    ) -> None:
        self._orchestrator = Orchestrator(session_id=session_id)
        self._stop_on_fail = stop_on_fail

    def run(self, suite: EvalSuite) -> EvalReport:
        """Execute every EvalCase in the suite and return the report."""
        print(f"\nRunning eval suite '{suite.name}' ({len(suite)} cases)…")
        results: list[EvalResult] = []

        for i, case in enumerate(suite.cases, 1):
            desc = case.description or case.query[:50]
            print(f"  [{i}/{len(suite)}] {desc}")

            t0 = time.monotonic()
            response: Optional[AgentResponse] = None
            error: Optional[str] = None

            try:
                kwargs: dict[str, Any] = {}
                if case.agent_hint:
                    kwargs["intent"] = case.agent_hint
                response = self._orchestrator.run(case.query, **kwargs)
            except Exception as exc:
                error = str(exc)
                response = AgentResponse(
                    output="", agent_name="error", error=error
                )

            latency = (time.monotonic() - t0) * 1000

            # Evaluate criteria
            checks: list[tuple[bool, str]] = []
            for criterion in case.criteria:
                try:
                    passed, reason = criterion.check(response)
                except Exception as exc:
                    passed, reason = False, f"Criterion error: {exc}"
                checks.append((passed, reason))

            overall_passed = all(p for p, _ in checks) and error is None

            result = EvalResult(
                case=case,
                response=response,
                latency_ms=latency,
                passed=overall_passed,
                checks=checks,
                error=error,
            )
            results.append(result)

            # Clear memory between cases to avoid cross-contamination
            self._orchestrator.clear_memory()

            if self._stop_on_fail and not overall_passed:
                print(f"  Stopping on first failure (stop_on_fail=True).")
                break

        return EvalReport(suite_name=suite.name, results=results)
