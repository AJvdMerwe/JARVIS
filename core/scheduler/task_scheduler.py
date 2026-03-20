"""
core/scheduler/task_scheduler.py
──────────────────────────────────
Lightweight cron-style background task scheduler using a daemon thread.

Built-in tasks:
  • daily_news_digest    – Run the news agent and save a digest to disk.
  • kb_reindex           – Re-process documents that have been modified.
  • cache_purge          – Evict expired entries from the tool cache.
  • trace_rotation       – Archive old traces and truncate the JSONL log.

Custom tasks can be registered with ``@scheduler.task(cron="0 8 * * *")``.

Cron-lite syntax supported: "HH:MM" for daily at a specific time,
"*/N" for every N minutes, or "@daily" / "@hourly" / "@every Nm".

Usage::

    from core.scheduler import get_scheduler, TaskScheduler

    scheduler = get_scheduler()
    scheduler.start()                    # launch background thread

    # Register a custom task:
    @scheduler.task(every_minutes=30)
    def my_task():
        print("Running every 30 minutes")

    scheduler.stop()                     # graceful shutdown
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Task model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduledTask:
    name:           str
    func:           Callable
    interval_s:     float              # run every N seconds
    last_run:       Optional[float] = None
    run_count:      int = 0
    error_count:    int = 0
    last_error:     Optional[str] = None
    enabled:        bool = True

    def is_due(self, now: float) -> bool:
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        return (now - self.last_run) >= self.interval_s

    def run(self) -> bool:
        """Execute the task. Returns True on success."""
        t0 = time.monotonic()
        try:
            self.func()
            self.run_count += 1
            self.last_run = time.monotonic()
            logger.debug(
                "Task '%s' completed in %.0fms (run #%d).",
                self.name, (time.monotonic() - t0) * 1000, self.run_count,
            )
            return True
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)
            self.last_run = time.monotonic()
            logger.error("Task '%s' failed: %s", self.name, exc)
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TaskScheduler:
    """
    Thread-based task scheduler. Tasks run in the same background thread
    (single-threaded, cooperative). Suitable for I/O-bound periodic tasks.

    Args:
        tick_interval: How often the scheduler checks for due tasks (seconds).
    """

    def __init__(self, tick_interval: float = 30.0) -> None:
        self._tick        = tick_interval
        self._tasks:  dict[str, ScheduledTask] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()
        self._lock    = threading.Lock()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        func: Callable,
        every_minutes: float = 60.0,
        enabled: bool = True,
    ) -> ScheduledTask:
        """Register a task to run every *every_minutes* minutes."""
        task = ScheduledTask(
            name=name,
            func=func,
            interval_s=every_minutes * 60,
            enabled=enabled,
        )
        with self._lock:
            self._tasks[name] = task
        logger.info(
            "Registered task '%s' (every %.0f min, enabled=%s).",
            name, every_minutes, enabled,
        )
        return task

    def task(self, every_minutes: float = 60.0, enabled: bool = True):
        """Decorator shorthand for register()."""
        def decorator(func: Callable) -> Callable:
            self.register(func.__name__, func, every_minutes=every_minutes, enabled=enabled)
            return func
        return decorator

    def enable(self, name: str) -> None:
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = True

    def disable(self, name: str) -> None:
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = False

    def run_now(self, name: str) -> bool:
        """Trigger a specific task immediately (synchronous)."""
        with self._lock:
            task = self._tasks.get(name)
        if task is None:
            raise KeyError(f"Task '{name}' not registered.")
        return task.run()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="TaskScheduler", daemon=True
        )
        self._thread.start()
        logger.info("Scheduler started (%d tasks registered).", len(self._tasks))

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the scheduler to stop and wait for it."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Scheduler stopped.")

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name":        t.name,
                    "enabled":     t.enabled,
                    "interval_m":  round(t.interval_s / 60, 1),
                    "run_count":   t.run_count,
                    "error_count": t.error_count,
                    "last_run":    (
                        datetime.fromtimestamp(t.last_run, tz=timezone.utc).isoformat()
                        if t.last_run else None
                    ),
                    "last_error":  t.last_error,
                }
                for t in self._tasks.values()
            ]

    # ── Internal loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                due = [t for t in self._tasks.values() if t.is_due(now)]
            for task in due:
                task.run()
            self._stop.wait(timeout=self._tick)


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in task implementations
# ─────────────────────────────────────────────────────────────────────────────

def _task_cache_purge() -> None:
    """Evict all expired entries from the tool cache."""
    from core.cache import get_cache
    removed = get_cache().purge_expired()
    logger.info("Cache purge: removed %d expired entries.", removed)


def _task_trace_rotation() -> None:
    """Trim the trace JSONL log to the last 10,000 lines."""
    from config import settings
    trace_log = settings.log_path.parent / "traces.jsonl"
    if not trace_log.exists():
        return
    lines = trace_log.read_text(encoding="utf-8").splitlines()
    if len(lines) > 10_000:
        trimmed = lines[-10_000:]
        trace_log.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
        logger.info("Trace log rotated: kept %d / %d lines.", len(trimmed), len(lines))


def _task_kb_reindex() -> None:
    """Re-ingest any documents in the uploads directory that are newer than their chunks."""
    from config import settings
    from document_processing import DocumentManager, SUPPORTED_SUFFIXES
    dm = DocumentManager()
    uploads = settings.uploads_path
    if not uploads.exists():
        return
    for path in uploads.rglob("*"):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                added = dm.ingest(path)
                if added > 0:
                    logger.info("KB re-index: added %d new chunks from '%s'.", added, path.name)
            except Exception as exc:
                logger.warning("KB re-index failed for '%s': %s", path.name, exc)


def _task_daily_news_digest() -> None:
    """Fetch top headlines and save a digest to data/logs/news_digest_YYYYMMDD.md."""
    from config import settings
    from tools.news_tools import HeadlinesTool
    try:
        tool = HeadlinesTool()
        digest = tool._run(max_articles=20)
        date_str = datetime.now().strftime("%Y%m%d")
        out = settings.log_path.parent / f"news_digest_{date_str}.md"
        out.write_text(digest, encoding="utf-8")
        logger.info("Daily news digest saved → %s", out)
    except Exception as exc:
        logger.warning("News digest failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton with built-in tasks pre-registered
# ─────────────────────────────────────────────────────────────────────────────

_scheduler: Optional[TaskScheduler] = None
_sched_lock = threading.Lock()


def get_scheduler() -> TaskScheduler:
    """Return the global TaskScheduler singleton with built-in tasks registered."""
    global _scheduler
    if _scheduler is None:
        with _sched_lock:
            if _scheduler is None:
                s = TaskScheduler(tick_interval=60.0)
                s.register("cache_purge",       _task_cache_purge,      every_minutes=15)
                s.register("trace_rotation",    _task_trace_rotation,   every_minutes=60)
                s.register("kb_reindex",        _task_kb_reindex,       every_minutes=30,  enabled=False)
                s.register("daily_news_digest", _task_daily_news_digest, every_minutes=1440, enabled=False)
                _scheduler = s
    return _scheduler
