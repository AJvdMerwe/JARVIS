"""
tests/test_final_modules.py
────────────────────────────
Tests for:
  • AsyncAgentRunner (single, fanout, timeout)
  • UserPreferences (load, save, reset, coercion, style_prompt)
  • TaskScheduler (registration, run_now, enable/disable, stats)
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import AgentResponse


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_agent(name: str, output: str = "ok", delay: float = 0.0, error: str = "") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent._llm = MagicMock()

    def _run(query, **kwargs):
        if delay:
            time.sleep(delay)
        return AgentResponse(
            output=output,
            agent_name=name,
            error=error or None,
        )

    agent.run.side_effect = _run
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  AsyncAgentRunner
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncAgentRunner:
    @pytest.mark.asyncio
    async def test_run_async_returns_response(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()
        agent  = _mock_agent("search_agent", output="Python is a language.")

        resp = await runner.run_async(agent, "What is Python?")
        assert resp.output == "Python is a language."
        assert resp.agent_name == "search_agent"
        assert resp.error is None

    @pytest.mark.asyncio
    async def test_run_async_timeout_returns_error_response(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner(default_timeout=0.1)
        slow_agent = _mock_agent("slow_agent", delay=2.0)

        resp = await runner.run_async(slow_agent, "slow query", timeout=0.05)
        assert resp.error == "timeout"
        assert "timed out" in resp.output.lower()

    @pytest.mark.asyncio
    async def test_run_async_exception_handled(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()
        broken = MagicMock()
        broken.name = "broken_agent"
        broken.run.side_effect = RuntimeError("unexpected crash")

        resp = await runner.run_async(broken, "query")
        assert resp.error == "unexpected crash"

    @pytest.mark.asyncio
    async def test_fanout_runs_agents_in_parallel(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()

        agents = [
            _mock_agent("code_agent",   output="Here is code.",   delay=0.05),
            _mock_agent("search_agent", output="Here is info.",   delay=0.05),
            _mock_agent("news_agent",   output="Here is news.",   delay=0.05),
        ]

        t0 = time.monotonic()
        result = await runner.fanout("Multi-agent query", agents)
        elapsed = time.monotonic() - t0

        # Sequential would be ≥ 0.15s; parallel should finish in < 0.12s
        assert elapsed < 0.12, f"Agents ran sequentially (took {elapsed:.2f}s)"
        assert len(result.responses) == 3
        assert len(result.succeeded) == 3

    @pytest.mark.asyncio
    async def test_fanout_empty_agents(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()
        result = await runner.fanout("query", agents=[])
        assert result.responses == []
        assert result.latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_fanout_partial_failure(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()
        agents = [
            _mock_agent("good_agent", output="works"),
            _mock_agent("bad_agent",  error="oops"),
        ]
        result = await runner.fanout("query", agents)
        assert len(result.succeeded) == 1
        assert len(result.failed) == 1

    @pytest.mark.asyncio
    async def test_fanout_merge_output(self):
        from core.async_runner import AsyncAgentRunner
        runner = AsyncAgentRunner()
        agents = [
            _mock_agent("code_agent",   output="```python\nprint('hi')\n```"),
            _mock_agent("search_agent", output="Python info here."),
        ]
        result = await runner.fanout("Python", agents)
        merged = result.merge_output()
        assert "code_agent" in merged
        assert "search_agent" in merged

    def test_fanout_result_all_references(self):
        from core.async_runner import FanOutResult
        r1 = AgentResponse(output="a", agent_name="a1", references=["Doc A", "Doc B"])
        r2 = AgentResponse(output="b", agent_name="a2", references=["Doc B", "Doc C"])
        result = FanOutResult(query="q", responses=[r1, r2], latency_ms=10)
        refs = result.all_references()
        assert len(refs) == 3        # deduplicated
        assert "Doc B" in refs


# ─────────────────────────────────────────────────────────────────────────────
#  UserPreferences
# ─────────────────────────────────────────────────────────────────────────────

class TestUserPreferences:
    def test_defaults(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="test")
        assert prefs.response_style == "concise"
        assert prefs.preferred_agent == "auto"
        assert prefs.language == "en"
        assert prefs.voice_enabled is False
        assert prefs.max_results == 5

    def test_save_and_load(self, tmp_path):
        from core.user_prefs.preferences import UserPreferences, _PREFS_DIR
        with patch("core.user_prefs.preferences._PREFS_DIR", tmp_path):
            prefs = UserPreferences(user_id="alice")
            prefs.response_style = "technical"
            prefs.news_topics = ["AI", "science"]
            prefs.max_results = 10
            prefs.save()

            loaded = UserPreferences.load("alice")
            assert loaded.response_style == "technical"
            assert "AI" in loaded.news_topics
            assert loaded.max_results == 10

    def test_reset_removes_file(self, tmp_path):
        from core.user_prefs.preferences import UserPreferences
        with patch("core.user_prefs.preferences._PREFS_DIR", tmp_path):
            prefs = UserPreferences(user_id="bob")
            prefs.response_style = "detailed"
            prefs.save()
            assert (tmp_path / "bob.json").exists()

            prefs.reset()
            assert not (tmp_path / "bob.json").exists()
            assert prefs.response_style == "concise"   # back to default

    def test_news_topics_validator_comma_string(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="x", news_topics="AI, climate, finance")
        assert prefs.news_topics == ["AI", "climate", "finance"]

    def test_language_validator_lowercases(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="x", language="EN")
        assert prefs.language == "en"

    def test_agent_style_prompt_concise(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="x", response_style="concise")
        prompt = prefs.agent_style_prompt()
        assert "brief" in prompt.lower() or "concise" in prompt.lower()

    def test_agent_style_prompt_with_language(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="x", response_style="friendly", language="fr")
        prompt = prefs.agent_style_prompt()
        assert "fr" in prompt

    def test_agent_style_prompt_with_custom_instructions(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="x", custom_instructions="Always cite sources.")
        prompt = prefs.agent_style_prompt()
        assert "Always cite sources." in prompt

    def test_agent_style_prompt_with_topics(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="x", news_topics=["machine learning", "robotics"])
        prompt = prefs.agent_style_prompt()
        assert "machine learning" in prompt

    def test_get_preferences_caches(self, tmp_path):
        from core.user_prefs.preferences import _prefs_cache, get_preferences, invalidate_cache
        with patch("core.user_prefs.preferences._PREFS_DIR", tmp_path):
            invalidate_cache("cache_test")
            p1 = get_preferences("cache_test")
            p2 = get_preferences("cache_test")
            assert p1 is p2    # same object from cache

    def test_safe_user_id_characters(self, tmp_path):
        """User IDs with special chars should be sanitised for filenames."""
        from core.user_prefs.preferences import UserPreferences
        path = UserPreferences._path("user@example.com/dangerous")
        assert "/" not in path.name
        assert "@" not in path.name

    def test_summary_string(self):
        from core.user_prefs import UserPreferences
        prefs = UserPreferences(user_id="alice", response_style="technical")
        s = prefs.summary()
        assert "alice" in s
        assert "technical" in s


# ─────────────────────────────────────────────────────────────────────────────
#  TaskScheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskScheduler:
    def test_register_and_run_now(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler(tick_interval=60)
        calls = []
        sched.register("test_task", lambda: calls.append(1), every_minutes=1)
        sched.run_now("test_task")
        assert calls == [1]

    def test_task_decorator(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler()
        calls = []

        @sched.task(every_minutes=5)
        def my_task():
            calls.append("run")

        sched.run_now("my_task")
        assert calls == ["run"]

    def test_enable_disable(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler()
        calls = []
        sched.register("togglable", lambda: calls.append(1), every_minutes=0.001)
        sched.disable("togglable")
        # Task is disabled — should not be due even if interval passed
        task = sched._tasks["togglable"]
        assert not task.is_due(time.monotonic() + 1000)

        sched.enable("togglable")
        assert task.is_due(time.monotonic() + 1000)

    def test_run_now_unknown_task_raises(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler()
        with pytest.raises(KeyError):
            sched.run_now("nonexistent")

    def test_stats_returned(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler()
        sched.register("task_a", lambda: None, every_minutes=30)
        sched.register("task_b", lambda: None, every_minutes=60)
        stats = sched.stats()
        assert len(stats) == 2
        names = [s["name"] for s in stats]
        assert "task_a" in names
        assert "task_b" in names

    def test_error_count_increments(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler()
        sched.register("failing", lambda: (_ for _ in ()).throw(RuntimeError("boom")), every_minutes=1)
        sched.run_now("failing")
        assert sched._tasks["failing"].error_count == 1
        assert sched._tasks["failing"].last_error == "boom"

    def test_run_count_increments(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler()
        sched.register("counter", lambda: None, every_minutes=1)
        for _ in range(3):
            sched.run_now("counter")
        assert sched._tasks["counter"].run_count == 3

    def test_is_due_after_interval(self):
        from core.scheduler.task_scheduler import ScheduledTask
        task = ScheduledTask(
            name="t", func=lambda: None,
            interval_s=10.0,
            last_run=time.monotonic() - 11.0,  # 11s ago
        )
        assert task.is_due(time.monotonic())

    def test_is_not_due_before_interval(self):
        from core.scheduler.task_scheduler import ScheduledTask
        task = ScheduledTask(
            name="t", func=lambda: None,
            interval_s=60.0,
            last_run=time.monotonic() - 5.0,   # only 5s ago
        )
        assert not task.is_due(time.monotonic())

    def test_start_and_stop(self):
        from core.scheduler import TaskScheduler
        sched = TaskScheduler(tick_interval=0.05)
        sched.register("noop", lambda: None, every_minutes=999)
        sched.start()
        assert sched.is_running
        sched.stop(timeout=1.0)
        assert not sched.is_running

    def test_get_scheduler_singleton(self):
        from core.scheduler import get_scheduler
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_builtin_tasks_registered(self):
        from core.scheduler import get_scheduler
        s = get_scheduler()
        names = {t["name"] for t in s.stats()}
        assert "cache_purge" in names
        assert "trace_rotation" in names
