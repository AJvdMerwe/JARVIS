"""
core/logging/structured_logger.py
───────────────────────────────────
Structured JSON logging for production observability.

Emits every log record as a JSON object with consistent fields:
  timestamp, level, logger, message, [extra fields...]

This makes logs easy to ingest into Elasticsearch, Loki, Datadog, etc.

Also provides:
  • ``AssistantLogger``  – per-session logger that auto-tags every record
    with session_id, agent_name, and latency_ms.
  • ``log_agent_call``   – context manager that times and logs a full
    agent invocation (in + out + tool_calls + latency).
  • ``setup_logging``    – call once at startup to configure handlers.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from config import settings


# ─────────────────────────────────────────────────────────────────────────────
#  JSON formatter
# ─────────────────────────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """
    Formats a LogRecord as a single-line JSON string.
    Extra keyword arguments passed to logger.info() etc. are included.
    """

    RESERVED = frozenset(logging.LogRecord.__dict__.keys()) | {
        "message", "asctime", "exc_info", "exc_text", "stack_info",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "line":    record.lineno,
        }

        # Include any extra fields set via logger.info("…", extra={…})
        for key, val in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = val

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ─────────────────────────────────────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = settings.log_level,
    log_path: Optional[Path] = None,
    json_console: bool = False,
) -> None:
    """
    Configure root logger with:
      • A human-readable console handler (INFO+).
      • A JSON file handler (DEBUG+) for structured log ingestion.

    Args:
        level:        Minimum log level string ("DEBUG", "INFO", etc.).
        log_path:     Where to write JSON logs. Defaults to settings.log_path.
        json_console: If True, console output is also JSON (useful in Docker).
    """
    log_path = log_path or settings.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    # Remove any existing handlers (avoid duplicates on re-import)
    root.handlers.clear()

    # ── Console handler ──────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    if json_console:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)-30s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(console_handler)

    # ── JSON file handler ────────────────────────────────────────────────────
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-session assistant logger
# ─────────────────────────────────────────────────────────────────────────────

class AssistantLogger:
    """
    Logger bound to a session. All records emitted through this logger
    automatically include ``session_id`` and an optional ``agent_name``.

    Usage::

        alog = AssistantLogger("session-abc123")
        alog.info("User message received", user_input="Hello")
        with alog.agent_call("code_agent", "write a sort") as ctx:
            response = agent.run("write a sort")
            ctx["tool_calls"] = len(response.tool_calls)
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._logger = logging.getLogger(f"assistant.session.{session_id}")

    def _log(self, level: int, msg: str, **extra: Any) -> None:
        self._logger.log(
            level, msg,
            extra={"session_id": self._session_id, **extra},
        )

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, **extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, **extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, **extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, **extra)

    @contextmanager
    def agent_call(
        self, agent_name: str, query: str
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager that logs the start, end, latency, and outcome
        of a single agent invocation.

        Usage::

            with alog.agent_call("code_agent", "sort a list") as ctx:
                response = agent.run("sort a list")
                ctx["output_len"] = len(response.output)
                ctx["tool_calls"] = len(response.tool_calls)
        """
        ctx: dict[str, Any] = {}
        t0 = time.monotonic()
        self.info(
            "Agent call started",
            agent=agent_name,
            query_preview=query[:120],
        )
        try:
            yield ctx
            latency = (time.monotonic() - t0) * 1000
            self.info(
                "Agent call completed",
                agent=agent_name,
                latency_ms=round(latency, 1),
                **ctx,
            )
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            self.error(
                "Agent call failed",
                agent=agent_name,
                latency_ms=round(latency, 1),
                error=str(exc),
                **ctx,
            )
            raise
