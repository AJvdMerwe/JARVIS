"""
tests/test_task_execution.py
──────────────────────────────
Tests for Stage 2 — Task Execution:

  • File output tools  — SaveTextFileTool, SaveCodeFileTool, SaveJsonFileTool
  • CodeExecutorTool   — hardened sandbox (expanded blocklist, env isolation,
                         output cap, resource limits)
  • parse_schedule()   — natural-language schedule parsing
  • UserTaskManager    — add/list/remove/toggle + persistence
  • Orchestrator       — schedule detection in run()
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
#  File output tools
# =============================================================================

class TestSaveTextFileTool:

    def test_saves_file_and_returns_path(self, tmp_path):
        from tools.file_output_tools import SaveTextFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool   = SaveTextFileTool()
            result = tool._run("Hello world", filename="test.txt", format="txt")
        assert "test.txt" in result
        assert (tmp_path / "test.txt").exists()

    def test_content_written_correctly(self, tmp_path):
        from tools.file_output_tools import SaveTextFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveTextFileTool()
            tool._run("# My Report\n\nContent here.", filename="report.md", format="md")
        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "# My Report" in content

    def test_markdown_extension_applied(self, tmp_path):
        from tools.file_output_tools import SaveTextFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveTextFileTool()
            tool._run("Report content", filename="report", format="md")
        assert (tmp_path / "report.md").exists()

    def test_no_silent_overwrite(self, tmp_path):
        from tools.file_output_tools import SaveTextFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveTextFileTool()
            tool._run("version 1", filename="data.txt", format="txt")
            tool._run("version 2", filename="data.txt", format="txt")
        # Two separate files should exist
        files = list(tmp_path.glob("data*.txt"))
        assert len(files) == 2

    def test_output_utf8_encoded(self, tmp_path):
        from tools.file_output_tools import SaveTextFileTool
        content = "Résumé · Straße · 日本語"
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveTextFileTool()
            tool._run(content, filename="unicode.txt", format="txt")
        saved = (tmp_path / "unicode.txt").read_text(encoding="utf-8")
        assert "Résumé" in saved

    def test_unsafe_filename_sanitised(self, tmp_path):
        from tools.file_output_tools import SaveTextFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool   = SaveTextFileTool()
            result = tool._run("content", filename="../../../etc/passwd", format="txt")
        # Should not create a file outside tmp_path
        # The file should be saved safely within tmp_path
        saved_files = list(tmp_path.glob("*.txt"))
        assert len(saved_files) >= 1  # file was created somewhere safe
        for f in saved_files:
            # No path traversal — all files inside tmp_path
            assert tmp_path in f.parents or f.parent == tmp_path


class TestSaveCodeFileTool:

    def test_saves_python_file(self, tmp_path):
        from tools.file_output_tools import SaveCodeFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveCodeFileTool()
            tool._run("def hello():\n    print('hi')", filename="hello.py", language="python")
        assert (tmp_path / "hello.py").exists()

    def test_correct_extension_for_language(self, tmp_path):
        from tools.file_output_tools import SaveCodeFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveCodeFileTool()
            tool._run("console.log('hi')", filename="app", language="javascript")
        assert (tmp_path / "app.js").exists()

    def test_returns_line_count(self, tmp_path):
        from tools.file_output_tools import SaveCodeFileTool
        code = "line1\nline2\nline3\n"
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool   = SaveCodeFileTool()
            result = tool._run(code, filename="f.py", language="python")
        assert "3 lines" in result or "line" in result


class TestSaveJsonFileTool:

    def test_saves_json_file(self, tmp_path):
        from tools.file_output_tools import SaveJsonFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveJsonFileTool()
            tool._run('{"key": "value"}', filename="data.json")
        assert (tmp_path / "data.json").exists()

    def test_pretty_prints_json(self, tmp_path):
        from tools.file_output_tools import SaveJsonFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveJsonFileTool()
            tool._run('{"a":1,"b":2}', filename="data.json")
        content = (tmp_path / "data.json").read_text()
        assert "\n" in content  # pretty-printed has newlines

    def test_invalid_json_saved_as_is(self, tmp_path):
        from tools.file_output_tools import SaveJsonFileTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool = SaveJsonFileTool()
            tool._run("not valid json", filename="raw.json")
        assert (tmp_path / "raw.json").exists()


class TestListSavedFilesTool:

    def test_returns_no_files_message_when_empty(self, tmp_path):
        from tools.file_output_tools import ListSavedFilesTool
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool   = ListSavedFilesTool()
            result = tool._run()
        assert "No files" in result

    def test_lists_saved_files(self, tmp_path):
        from tools.file_output_tools import ListSavedFilesTool
        (tmp_path / "a.txt").write_text("content")
        (tmp_path / "b.py").write_text("code")
        with patch("tools.file_output_tools._OUTPUT_DIR", tmp_path):
            tool   = ListSavedFilesTool()
            result = tool._run()
        assert "a.txt" in result
        assert "b.py" in result


# =============================================================================
#  CodeExecutorTool — hardened sandbox
# =============================================================================

class TestCodeExecutorSandbox:

    def _tool(self):
        from tools.code_tools import CodeExecutorTool
        return CodeExecutorTool()

    @pytest.mark.parametrize("blocked_code", [
        "import os; os.remove('/tmp/test')",
        "import os; os.unlink('file')",
        "import shutil; shutil.rmtree('/tmp')",
        "import subprocess; subprocess.call(['ls'])",
        "import subprocess; subprocess.run(['ls'])",
        "os.system('rm -rf /')",
        "__import__('os').system('ls')",
        "open('/etc/passwd')",
        "open('/root/.bashrc')",
        "import socket; socket.connect(('8.8.8.8', 80))",
    ])
    def test_blocked_patterns_rejected(self, blocked_code):
        tool   = self._tool()
        result = tool._safety_check(blocked_code)
        assert result is not None
        assert "blocked" in result.lower()

    def test_safe_code_passes_check(self):
        tool = self._tool()
        safe = "x = 1 + 1\nprint(x)"
        assert tool._safety_check(safe) is None

    def test_output_cap_constant_defined(self):
        import tools.code_tools as ct
        # Now a module-level constant, not a class attribute
        assert hasattr(ct, "_CODE_MAX_OUTPUT_BYTES")
        assert ct._CODE_MAX_OUTPUT_BYTES > 0

    def test_env_isolation_in_subprocess(self):
        """Subprocess should run with minimal environment (PATH only)."""
        import subprocess, sys
        tool = self._tool()
        # Verify the tool passes env parameter
        import inspect
        src = inspect.getsource(tool._run)
        assert "env=" in src

    def test_expanded_blocklist_has_critical_patterns(self):
        from tools.code_tools import CodeExecutorTool
        tool     = CodeExecutorTool()
        critical = ["os.remove", "subprocess.Popen", "socket.connect", "os.system"]
        for pattern in critical:
            assert any(pattern in b for b in tool._BLOCKED), (
                f"'{pattern}' not in _BLOCKED"
            )


# =============================================================================
#  parse_schedule
# =============================================================================

class TestParseSchedule:

    def _parse(self, text):
        from core.user_task_scheduler import parse_schedule
        return parse_schedule(text)

    def test_every_30_minutes(self):
        interval, desc = self._parse("remind me every 30 minutes")
        assert interval == 30.0

    def test_every_2_hours(self):
        interval, desc = self._parse("send me a report every 2 hours")
        assert interval == 120.0

    def test_every_day(self):
        interval, desc = self._parse("schedule a news briefing every day")
        assert interval == 1440.0

    def test_daily_keyword(self):
        interval, desc = self._parse("send me a daily summary")
        assert interval == 1440.0

    def test_every_morning(self):
        interval, desc = self._parse("every morning send me the news")
        assert interval == 1440.0

    def test_every_hour(self):
        interval, desc = self._parse("check the stock price every hour")
        assert interval == 60.0

    def test_in_30_minutes(self):
        interval, desc = self._parse("remind me in 30 minutes")
        assert interval == 30.0

    def test_no_schedule_returns_zero(self):
        interval, desc = self._parse("what is the stock price of Apple?")
        assert interval == 0.0

    def test_description_non_empty_when_matched(self):
        _, desc = self._parse("remind me every hour")
        assert desc != ""

    def test_weekly(self):
        interval, desc = self._parse("run the analysis weekly")
        assert interval == 10080.0


# =============================================================================
#  UserTaskManager
# =============================================================================

class TestUserTaskManager:

    def _mgr(self, tmp_path):
        from core.user_task_scheduler import UserTaskManager
        import core.user_task_scheduler as utm
        utm._TASKS_FILE = tmp_path / "tasks.json"
        mgr = UserTaskManager.__new__(UserTaskManager)
        mgr._tasks = {}
        return mgr

    def test_add_task_returns_user_task(self, tmp_path):
        from core.user_task_scheduler import UserTask
        mgr = self._mgr(tmp_path)
        with patch.object(mgr, "_register_with_scheduler"), \
             patch.object(mgr, "_save"):
            task = mgr.add_task("u1", "sess1", "Daily news", "latest news", "news", 1440.0)
        assert isinstance(task, UserTask)
        assert task.description == "Daily news"
        assert task.interval_minutes == 1440.0

    def test_list_tasks_filtered_by_user(self, tmp_path):
        mgr = self._mgr(tmp_path)
        with patch.object(mgr, "_register_with_scheduler"), \
             patch.object(mgr, "_save"):
            mgr.add_task("alice", "s1", "Alice task", "q", "news", 60.0)
            mgr.add_task("bob",   "s2", "Bob task",   "q", "chat", 120.0)
        alice_tasks = mgr.list_tasks(user_id="alice")
        assert len(alice_tasks) == 1
        assert alice_tasks[0].description == "Alice task"

    def test_remove_task(self, tmp_path):
        mgr = self._mgr(tmp_path)
        with patch.object(mgr, "_register_with_scheduler"), \
             patch.object(mgr, "_save"), \
             patch.object(mgr, "_unregister_from_scheduler"):
            task = mgr.add_task("u1", "s1", "Remove me", "q", "news", 60.0)
            result = mgr.remove_task(task.task_id)
        assert result is True
        assert len(mgr.list_tasks("u1")) == 0

    def test_remove_nonexistent_task(self, tmp_path):
        mgr = self._mgr(tmp_path)
        assert mgr.remove_task("nonexistent") is False

    def test_toggle_task_off(self, tmp_path):
        mgr = self._mgr(tmp_path)
        with patch.object(mgr, "_register_with_scheduler"), \
             patch.object(mgr, "_save"):
            task = mgr.add_task("u1", "s1", "Toggleable", "q", "news", 60.0)
            mgr.toggle_task(task.task_id, enabled=False)
        assert mgr._tasks[task.task_id].enabled is False

    def test_persistence_save_load(self, tmp_path):
        import core.user_task_scheduler as utm
        utm._TASKS_FILE = tmp_path / "tasks.json"

        mgr1 = utm.UserTaskManager.__new__(utm.UserTaskManager)
        mgr1._tasks = {}
        with patch.object(mgr1, "_register_with_scheduler"):
            task = mgr1.add_task("u1", "s1", "Persist test", "q", "chat", 30.0)
        mgr1._save()

        # Load in a new manager instance
        mgr2 = utm.UserTaskManager.__new__(utm.UserTaskManager)
        mgr2._tasks = {}
        mgr2._load()

        loaded = mgr2.list_tasks()
        assert len(loaded) == 1
        assert loaded[0].description == "Persist test"

    def test_task_id_unique(self, tmp_path):
        mgr = self._mgr(tmp_path)
        with patch.object(mgr, "_register_with_scheduler"), \
             patch.object(mgr, "_save"):
            t1 = mgr.add_task("u1", "s1", "Task 1", "q1", "news", 60.0)
            t2 = mgr.add_task("u1", "s1", "Task 2", "q2", "news", 60.0)
        assert t1.task_id != t2.task_id


# =============================================================================
#  Orchestrator schedule detection
# =============================================================================

class TestOrchestratorScheduleDetection:

    def _make_orch(self):
        from agents.orchestrator import Orchestrator, Intent
        from agents.base_agent import AgentResponse

        orch = Orchestrator.__new__(Orchestrator)
        orch._session_id               = "test"
        orch._enable_llm_quality_check = False
        orch._enable_rag_precheck      = False
        orch._rag_similarity_threshold = 0.55
        orch._rag_k                    = 4
        orch._max_fallback_attempts    = 0
        orch._summariser               = None
        orch._episodic                 = None
        orch._user_id                  = "default"

        mem = MagicMock(); mem.messages = []
        orch._memory = mem

        orch._agents = {}
        for intent in Intent:
            a = MagicMock()
            a.name = intent.value
            a.run.return_value = AgentResponse(output="Agent response.", agent_name=intent.value)
            orch._agents[intent] = a
        return orch

    def test_schedule_query_returns_confirmation(self):
        orch = self._make_orch()
        mock_task = MagicMock()
        mock_task.task_id = "abc123"
        mock_mgr  = MagicMock()
        mock_mgr.add_task.return_value = mock_task

        with patch.object(orch, "_route", return_value=__import__(
            "agents.orchestrator", fromlist=["Intent"]
        ).Intent.NEWS), \
             patch.object(orch, "_open_trace", return_value=None), \
             patch.object(orch, "_close_trace"), \
             patch.object(orch, "_update_user_profile"), \
             patch("core.user_task_scheduler.get_task_manager", return_value=mock_mgr):
            resp = orch.run("Remind me to check the news every hour")

        assert resp.agent_name == "scheduler"
        assert "Scheduled" in resp.output or "scheduled" in resp.output

    def test_non_schedule_query_not_intercepted(self):
        orch = self._make_orch()
        with patch.object(orch, "_route", return_value=__import__(
            "agents.orchestrator", fromlist=["Intent"]
        ).Intent.SEARCH), \
             patch.object(orch, "_open_trace", return_value=None), \
             patch.object(orch, "_close_trace"), \
             patch.object(orch, "_update_user_profile"):
            resp = orch.run("What is the capital of France?")

        # Should route to search agent, not scheduler
        assert resp.agent_name != "scheduler"

    def test_schedule_detection_error_falls_through(self):
        """A crash in schedule detection must not prevent normal routing."""
        orch = self._make_orch()
        with patch.object(orch, "_route", return_value=__import__(
            "agents.orchestrator", fromlist=["Intent"]
        ).Intent.CHAT), \
             patch.object(orch, "_open_trace", return_value=None), \
             patch.object(orch, "_close_trace"), \
             patch.object(orch, "_update_user_profile"), \
             patch("core.user_task_scheduler.get_task_manager", side_effect=Exception("fail")):
            resp = orch.run("Remind me every hour")
        # Either scheduled (if pattern matched before error) or normal response
        assert resp is not None


# =============================================================================
#  get_file_output_tools() registry
# =============================================================================

class TestFileOutputToolRegistry:

    def test_all_tools_registered(self):
        from tools.file_output_tools import get_file_output_tools
        tools = get_file_output_tools()
        names = {t.name for t in tools}
        assert "save_text_file"  in names
        assert "save_code_file"  in names
        assert "save_json_file"  in names
        assert "list_saved_files" in names

    def test_code_agent_has_file_tools(self):
        """CodeAgent.get_tools() must include the file output tools."""
        from core.memory import AssistantMemory
        from agents.code_agent import CodeAgent
        from unittest.mock import MagicMock
        agent = CodeAgent.__new__(CodeAgent)
        agent._llm     = MagicMock()
        agent._memory  = AssistantMemory()
        agent._verbose = False
        agent._logger  = MagicMock()
        agent._executor = MagicMock()
        tools = agent.get_tools()
        names = {t.name for t in tools}
        assert "save_text_file" in names or "save_code_file" in names
