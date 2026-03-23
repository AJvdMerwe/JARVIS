"""
core/user_task_scheduler.py
─────────────────────────────
Per-user task scheduling: lets users set reminders and recurring tasks
via natural language in the REPL or API.

Examples the orchestrator can trigger:
  "Remind me to check the earnings report in 30 minutes"
  "Send me a news briefing every morning"
  "Run the portfolio analysis every Friday at 9am"

User tasks are persisted to JSON and reloaded on startup.
They run on the shared TaskScheduler daemon thread.

Architecture
────────────
  UserTask         — a single user-defined scheduled item (data model)
  UserTaskManager  — CRUD for user tasks + registration with the system scheduler
  parse_schedule() — natural-language → (interval_minutes, description)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)

_TASKS_FILE = settings.log_path.parent / "user_tasks.json"


# =============================================================================
#  Data model
# =============================================================================

@dataclass
class UserTask:
    """A single user-defined scheduled task."""
    task_id:          str
    user_id:          str
    session_id:       str
    description:      str          # human-readable description
    query:            str          # query to run when the task fires
    intent:           str          # agent intent to use (e.g. "news", "finance")
    interval_minutes: float        # how often to run (in minutes)
    enabled:          bool   = True
    created_at:       float  = field(default_factory=time.time)
    last_run:         float  = 0.0
    run_count:        int    = 0

    @property
    def next_run_in_minutes(self) -> float:
        elapsed = (time.time() - self.last_run) / 60
        return max(0.0, self.interval_minutes - elapsed)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UserTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
#  Natural-language schedule parser
# =============================================================================

# Patterns: "every 30 minutes", "every hour", "every day", "every morning"
_INTERVAL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"every\s+(\d+)\s+minute", re.I),       1.0),      # multiplier
    (re.compile(r"every\s+(\d+)\s+hour", re.I),         60.0),
    (re.compile(r"in\s+(\d+)\s+minute", re.I),          1.0),
    (re.compile(r"in\s+(\d+)\s+hour", re.I),            60.0),
    (re.compile(r"every\s+half[\s-]hour", re.I),        30.0),     # fixed
    (re.compile(r"every\s+hour\b", re.I),               60.0),
    (re.compile(r"every\s+day\b|daily", re.I),          1440.0),
    (re.compile(r"every\s+morning\b", re.I),            1440.0),
    (re.compile(r"every\s+evening\b", re.I),            1440.0),
    (re.compile(r"every\s+week\b|weekly", re.I),        10080.0),
    (re.compile(r"every\s+(?:mon|tue|wed|thu|fri|sat|sun)", re.I), 10080.0),
]


def parse_schedule(text: str) -> tuple[float, str]:
    """
    Parse a natural-language schedule description into an interval in minutes.

    Returns
    -------
    tuple[float, str]
        (interval_minutes, canonical_description)
        interval_minutes is 0.0 if no schedule was detected.
    """
    for pattern, multiplier in _INTERVAL_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                n = float(m.group(1))
                interval = n * multiplier
            except (IndexError, ValueError):
                interval = multiplier   # fixed-value patterns
            desc = _describe_interval(interval)
            return interval, desc

    return 0.0, ""


def _describe_interval(minutes: float) -> str:
    if minutes < 60:
        return f"every {int(minutes)} minute{'s' if minutes != 1 else ''}"
    if minutes < 1440:
        hours = minutes / 60
        return f"every {int(hours)} hour{'s' if hours != 1 else ''}"
    if minutes < 10080:
        days = minutes / 1440
        return f"every {int(days)} day{'s' if days != 1 else ''}"
    weeks = minutes / 10080
    return f"every {int(weeks)} week{'s' if weeks != 1 else ''}"


# =============================================================================
#  UserTaskManager
# =============================================================================

class UserTaskManager:
    """
    CRUD interface for user-defined scheduled tasks.

    Tasks are persisted to ``_TASKS_FILE`` so they survive restarts.
    They are registered with the system ``TaskScheduler`` so they actually run.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, UserTask] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not _TASKS_FILE.exists():
            return
        try:
            raw = json.loads(_TASKS_FILE.read_text(encoding="utf-8"))
            for item in raw:
                t = UserTask.from_dict(item)
                self._tasks[t.task_id] = t
            logger.info("Loaded %d user task(s) from %s.", len(self._tasks), _TASKS_FILE)
        except Exception as exc:
            logger.warning("Failed to load user tasks: %s", exc)

    def _save(self) -> None:
        try:
            _TASKS_FILE.write_text(
                json.dumps([t.to_dict() for t in self._tasks.values()], indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save user tasks: %s", exc)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def add_task(
        self,
        user_id:          str,
        session_id:       str,
        description:      str,
        query:            str,
        intent:           str   = "chat",
        interval_minutes: float = 60.0,
    ) -> UserTask:
        """
        Create and register a new user task.

        Parameters
        ----------
        user_id          : str
        session_id       : str
        description      : str   Human-readable label (e.g. "Daily news briefing")
        query            : str   Query to run when the task fires
        intent           : str   Agent intent: news | finance | search | chat
        interval_minutes : float How often to run

        Returns
        -------
        UserTask
        """
        import hashlib
        task_id = hashlib.sha256(
            f"{user_id}:{description}:{time.time()}".encode()
        ).hexdigest()[:12]

        task = UserTask(
            task_id=task_id,
            user_id=user_id,
            session_id=session_id,
            description=description,
            query=query,
            intent=intent,
            interval_minutes=interval_minutes,
        )
        self._tasks[task_id] = task
        self._save()
        self._register_with_scheduler(task)
        logger.info(
            "Added user task '%s' for user '%s' (%s).",
            description, user_id, _describe_interval(interval_minutes),
        )
        return task

    def list_tasks(self, user_id: Optional[str] = None) -> list[UserTask]:
        """Return all tasks, optionally filtered by user."""
        tasks = list(self._tasks.values())
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        return sorted(tasks, key=lambda t: t.created_at)

    def remove_task(self, task_id: str) -> bool:
        """Remove a task by ID. Returns True if found and removed."""
        if task_id not in self._tasks:
            return False
        self._unregister_from_scheduler(task_id)
        del self._tasks[task_id]
        self._save()
        return True

    def toggle_task(self, task_id: str, enabled: bool) -> bool:
        """Enable or disable a task without removing it."""
        if task_id not in self._tasks:
            return False
        self._tasks[task_id].enabled = enabled
        self._save()
        return True

    # ── Scheduler integration ─────────────────────────────────────────────────

    def _register_with_scheduler(self, task: UserTask) -> None:
        """Register the task with the global TaskScheduler."""
        try:
            from core.scheduler import get_scheduler
            sched = get_scheduler()

            def _runner(t=task) -> None:
                if not t.enabled:
                    return
                try:
                    from agents.orchestrator import Orchestrator
                    orch = Orchestrator(session_id=t.session_id)
                    resp = orch.run(t.query, intent=t.intent if t.intent != "auto" else None)
                    t.last_run  = time.time()
                    t.run_count += 1
                    self._save()
                    logger.info(
                        "User task '%s' ran (count=%d): %s",
                        t.description, t.run_count, resp.output[:100],
                    )
                except Exception as exc:
                    logger.error("User task '%s' failed: %s", t.description, exc)

            sched.register(
                name=f"user_task_{task.task_id}",
                func=_runner,
                every_minutes=task.interval_minutes,
                enabled=task.enabled,
            )
        except Exception as exc:
            logger.warning("Could not register task with scheduler: %s", exc)

    def _unregister_from_scheduler(self, task_id: str) -> None:
        try:
            from core.scheduler import get_scheduler
            sched = get_scheduler()
            name  = f"user_task_{task_id}"
            with sched._lock:
                sched._tasks.pop(name, None)
        except Exception:
            pass

    def reload_all(self) -> int:
        """Re-register all loaded tasks with the scheduler (called on startup)."""
        count = 0
        for task in self._tasks.values():
            self._register_with_scheduler(task)
            count += 1
        return count


# =============================================================================
#  Singleton
# =============================================================================

_manager: Optional[UserTaskManager] = None


def get_task_manager() -> UserTaskManager:
    """Return the singleton UserTaskManager."""
    global _manager
    if _manager is None:
        _manager = UserTaskManager()
    return _manager
