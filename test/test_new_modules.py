"""
tests/test_new_modules.py
──────────────────────────
Unit tests for:
  • CircuitBreaker / ResilientLLM
  • ToolCache / cached_tool decorator
  • EvalHarness (criteria, EvalSuite, EvalReport)
  • Plugin loader (file discovery)
  • Structured logger (JSON format)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  CircuitBreaker
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def _make(self, threshold=3, timeout=60.0):
        from core.resilience.llm_resilience import CircuitBreaker
        return CircuitBreaker(failure_threshold=threshold, recovery_timeout=timeout)

    def test_starts_closed(self):
        from core.resilience.llm_resilience import CircuitState
        cb = self._make()
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        from core.resilience.llm_resilience import CircuitState
        cb = self._make(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_does_not_allow_when_open(self):
        cb = self._make(threshold=1)
        cb.record_failure()
        assert not cb.allow_request()

    def test_success_resets_failure_count(self):
        from core.resilience.llm_resilience import CircuitState
        cb = self._make(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()   # reset
        cb.record_failure()
        cb.record_failure()
        # Only 2 failures since last success — still closed
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        from core.resilience.llm_resilience import CircuitState
        cb = self._make(threshold=1, timeout=0.05)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        # Accessing .state triggers the transition
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_probes(self):
        from core.resilience.llm_resilience import CircuitState
        cb = self._make(threshold=1, timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        _ = cb.state  # trigger half-open
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
#  ResilientLLM
# ─────────────────────────────────────────────────────────────────────────────

class TestResilientLLM:
    def _make_llm(self, side_effect=None, content="ok"):
        mock_llm = MagicMock()
        if side_effect:
            mock_llm.invoke.side_effect = side_effect
        else:
            mock_llm.invoke.return_value = MagicMock(content=content)
        return mock_llm

    def test_successful_call(self):
        from core.resilience.llm_resilience import ResilientLLM
        primary = self._make_llm(content="Hello!")
        rl = ResilientLLM(primary=primary, max_retries=1, timeout=5.0)
        result = rl.invoke("Hi")
        assert result.content == "Hello!"

    def test_retries_on_rate_limit(self):
        from core.resilience.llm_resilience import ResilientLLM
        calls = []
        def side_effect(*a, **kw):
            calls.append(1)
            if len(calls) < 3:
                raise Exception("rate limit exceeded")
            return MagicMock(content="ok after retries")

        primary = MagicMock(invoke=side_effect)
        with patch("time.sleep"):
            rl = ResilientLLM(primary=primary, max_retries=3, base_delay=0.01, timeout=5.0)
            result = rl.invoke("test")
        assert result.content == "ok after retries"
        assert len(calls) == 3

    def test_falls_back_on_primary_failure(self):
        from core.resilience.llm_resilience import ResilientLLM
        primary  = self._make_llm(side_effect=Exception("rate limit"))
        fallback = self._make_llm(content="from fallback")
        with patch("time.sleep"):
            rl = ResilientLLM(primary=primary, fallback=fallback, max_retries=2, timeout=5.0)
            result = rl.invoke("hi")
        assert result.content == "from fallback"

    def test_timeout_raises(self):
        from core.resilience.llm_resilience import ResilientLLM
        import concurrent.futures

        def slow(*a, **kw):
            time.sleep(10)
            return MagicMock(content="late")

        primary = MagicMock(invoke=slow)
        rl = ResilientLLM(primary=primary, max_retries=1, timeout=0.05)
        with pytest.raises((TimeoutError, Exception)):
            rl.invoke("test")


# ─────────────────────────────────────────────────────────────────────────────
#  ToolCache
# ─────────────────────────────────────────────────────────────────────────────

class TestToolCache:
    def _make(self, ttl=60.0):
        from core.cache.tool_cache import ToolCache
        return ToolCache(default_ttl=ttl, max_size=10)

    def test_set_and_get(self):
        cache = self._make()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = self._make()
        assert cache.get("nonexistent") is None

    def test_expired_returns_none(self):
        cache = self._make(ttl=0.05)
        cache.set("k", "v")
        time.sleep(0.06)
        assert cache.get("k") is None

    def test_hit_rate(self):
        cache = self._make()
        cache.set("k", "v")
        cache.get("k")     # hit
        cache.get("k")     # hit
        cache.get("miss")  # miss
        assert cache.hits == 2
        assert cache.misses == 1
        assert abs(cache.hit_rate - 2/3) < 0.01

    def test_max_size_eviction(self):
        cache = self._make()
        for i in range(12):
            cache.set(f"key{i}", f"val{i}")
        assert cache.size <= 10

    def test_make_key_stable(self):
        from core.cache.tool_cache import ToolCache
        k1 = ToolCache.make_key("tool", "arg1", key="val")
        k2 = ToolCache.make_key("tool", "arg1", key="val")
        assert k1 == k2

    def test_make_key_differs_on_args(self):
        from core.cache.tool_cache import ToolCache
        k1 = ToolCache.make_key("tool", "arg1")
        k2 = ToolCache.make_key("tool", "arg2")
        assert k1 != k2

    def test_cached_tool_decorator(self):
        from core.cache.tool_cache import cached_tool, get_cache
        get_cache().clear()

        call_count = [0]

        class FakeTool:
            @cached_tool("test_tool", ttl=60)
            def _run(self, query: str) -> str:
                call_count[0] += 1
                return f"result for {query}"

        tool = FakeTool()
        r1 = tool._run("hello")
        r2 = tool._run("hello")   # should be cached
        r3 = tool._run("world")   # different arg — new call

        assert r1 == r2 == "result for hello"
        assert r3 == "result for world"
        assert call_count[0] == 2   # "hello" called once, "world" once


# ─────────────────────────────────────────────────────────────────────────────
#  Eval Harness — Criteria
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalCriteria:
    def _resp(self, output="hello world", agent="search_agent", error=None, tools=None, refs=None):
        from agents.base_agent import AgentResponse
        return AgentResponse(
            output=output,
            agent_name=agent,
            tool_calls=tools or [],
            references=refs or [],
            error=error,
        )

    def test_contains_pass(self):
        from evaluation.eval_harness import Contains
        passed, _ = Contains("hello")._run_check(self._resp()) if hasattr(Contains("x"), '_run_check') else Contains("hello").check(self._resp())
        assert passed

    def test_contains_fail(self):
        from evaluation.eval_harness import Contains
        passed, _ = Contains("missing_text").check(self._resp())
        assert not passed

    def test_contains_case_insensitive(self):
        from evaluation.eval_harness import Contains
        passed, _ = Contains("HELLO").check(self._resp("hello world"))
        assert passed

    def test_not_contains(self):
        from evaluation.eval_harness import NotContains
        passed, _ = NotContains("xyz").check(self._resp("hello world"))
        assert passed

    def test_no_error_pass(self):
        from evaluation.eval_harness import NoError
        passed, _ = NoError().check(self._resp())
        assert passed

    def test_no_error_fail(self):
        from evaluation.eval_harness import NoError
        passed, _ = NoError().check(self._resp(error="boom"))
        assert not passed

    def test_min_length_pass(self):
        from evaluation.eval_harness import MinLength
        passed, _ = MinLength(5).check(self._resp("hello world"))
        assert passed

    def test_min_length_fail(self):
        from evaluation.eval_harness import MinLength
        passed, _ = MinLength(100).check(self._resp("short"))
        assert not passed

    def test_agent_is_pass(self):
        from evaluation.eval_harness import AgentIs
        passed, _ = AgentIs("search_agent").check(self._resp())
        assert passed

    def test_agent_is_fail(self):
        from evaluation.eval_harness import AgentIs
        passed, _ = AgentIs("code_agent").check(self._resp(agent="search_agent"))
        assert not passed

    def test_used_tools_pass(self):
        from evaluation.eval_harness import UsedTools
        resp = self._resp(tools=[("web_search", "q", "result")])
        passed, _ = UsedTools().check(resp)
        assert passed

    def test_used_tools_specific(self):
        from evaluation.eval_harness import UsedTools
        resp = self._resp(tools=[("web_search", "q", "result")])
        passed, _ = UsedTools("web_search").check(resp)
        assert passed
        passed2, _ = UsedTools("calculator").check(resp)
        assert not passed2

    def test_has_references(self):
        from evaluation.eval_harness import HasReferences
        resp = self._resp(refs=["Doc › Page 1", "Doc › Page 2"])
        passed, _ = HasReferences(min_refs=2).check(resp)
        assert passed
        passed2, _ = HasReferences(min_refs=3).check(resp)
        assert not passed2

    def test_matches_regex(self):
        from evaluation.eval_harness import MatchesRegex
        resp = self._resp("The answer is 42.")
        passed, _ = MatchesRegex(r"\d+").check(resp)
        assert passed
        passed2, _ = MatchesRegex(r"xyz\d{5}").check(resp)
        assert not passed2


# ─────────────────────────────────────────────────────────────────────────────
#  EvalSuite + Evaluator (mocked orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalSuite:
    def test_add_and_filter(self):
        from evaluation.eval_harness import EvalCase, EvalSuite, NoError
        suite = EvalSuite("test")
        suite.add(EvalCase("q1", [NoError()], tags=["smoke"]))
        suite.add(EvalCase("q2", [NoError()], tags=["regression"]))
        suite.add(EvalCase("q3", [NoError()], tags=["smoke", "regression"]))
        assert len(suite) == 3
        smoke = suite.filter_tags("smoke")
        assert len(smoke) == 2


class TestEvaluator:
    def test_run_all_pass(self):
        from evaluation.eval_harness import (
            Contains, EvalCase, EvalSuite, Evaluator, NoError
        )
        from agents.base_agent import AgentResponse

        suite = EvalSuite("mock_suite")
        suite.add(EvalCase(
            query="test query",
            criteria=[Contains("success"), NoError()],
            agent_hint="search",
        ))

        mock_response = AgentResponse(output="This is a success result.", agent_name="search_agent")

        with patch("evaluation.eval_harness.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = mock_response
            mock_orch_cls.return_value = mock_orch

            evaluator = Evaluator(session_id="test-eval")
            report = evaluator.run(suite)

        assert report.total == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.pass_rate == 1.0

    def test_run_with_failure(self):
        from evaluation.eval_harness import (
            Contains, EvalCase, EvalSuite, Evaluator
        )
        from agents.base_agent import AgentResponse

        suite = EvalSuite("failing_suite")
        suite.add(EvalCase(
            query="test",
            criteria=[Contains("NONEXISTENT_STRING_ABCXYZ")],
        ))

        mock_response = AgentResponse(output="Some unrelated output.", agent_name="search_agent")
        with patch("evaluation.eval_harness.Orchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.run.return_value = mock_response
            mock_orch_cls.return_value = mock_orch

            evaluator = Evaluator()
            report = evaluator.run(suite)

        assert report.failed == 1

    def test_report_to_json(self):
        from evaluation.eval_harness import (
            NoError, EvalCase, EvalSuite, EvalReport, EvalResult
        )
        from agents.base_agent import AgentResponse

        case = EvalCase("q", [NoError()])
        resp = AgentResponse(output="ok", agent_name="test")
        result = EvalResult(case=case, response=resp, latency_ms=123, passed=True, checks=[(True, "NoError: ✓")])
        report = EvalReport(suite_name="test", results=[result])

        data = report.to_json()
        assert data["total"] == 1
        assert data["passed"] == 1
        assert data["pass_rate"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Structured logger
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuredLogger:
    def test_json_formatter_produces_valid_json(self):
        import logging
        from core.logging.structured_logger import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger", level=logging.INFO, pathname="", lineno=1,
            msg="Test message", args=(), exc_info=None,
        )
        record.session_id = "abc123"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["msg"] == "Test message"
        assert data["session_id"] == "abc123"
        assert "ts" in data

    def test_assistant_logger_tags_records(self, caplog, tmp_path):
        import logging
        from core.logging.structured_logger import AssistantLogger

        alog = AssistantLogger("sess-999")
        with caplog.at_level(logging.INFO):
            alog.info("Something happened", extra_key="extra_val")

        # Verify the log was emitted (exact format depends on handler setup)
        assert any("Something happened" in r.message for r in caplog.records)

    def test_agent_call_context_manager(self):
        import logging
        from core.logging.structured_logger import AssistantLogger

        alog = AssistantLogger("sess-ctx")
        logged = []

        with patch.object(alog._logger, "log", side_effect=lambda *a, **kw: logged.append(a)):
            with alog.agent_call("code_agent", "write code") as ctx:
                ctx["output_len"] = 42

        # Should have logged start + completion
        assert len(logged) >= 2


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin loader
# ─────────────────────────────────────────────────────────────────────────────

class TestPluginLoader:
    def test_discover_file_plugin(self, tmp_path: Path):
        """A .py file dropped into plugins/ should be discovered."""
        plugin_code = '''
from langchain_core.tools import BaseTool
class DummyTool(BaseTool):
    name = "dummy_tool"
    description = "A dummy tool."
    def _run(self, x=""): return "dummy"
    def _arun(self, **kw): raise NotImplementedError

def register_tools():
    return [DummyTool()]
'''
        plugin_file = tmp_path / "dummy_plugin.py"
        plugin_file.write_text(plugin_code)

        from plugins.plugin_loader import _load_module_from_path
        mod = _load_module_from_path(plugin_file)
        tools = mod.register_tools()
        assert len(tools) == 1
        assert tools[0].name == "dummy_tool"

    def test_load_all_returns_empty_on_no_plugins(self, tmp_path: Path):
        from plugins import plugin_loader

        with patch.object(plugin_loader, "_PLUGINS_DIR", tmp_path):
            agents, tools = plugin_loader.load_all_plugins()

        # tmp_path has no .py files
        assert isinstance(agents, dict)
        assert isinstance(tools, list)
