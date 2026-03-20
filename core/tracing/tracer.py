"""
core/tracing/tracer.py
───────────────────────
Lightweight request tracer that captures structured spans for each
agent invocation without requiring a full APM stack.

Each ``Trace`` contains:
  • Request metadata (session, query, timestamp).
  • A list of ``Span`` objects — one per logical step (routing, LLM call,
    tool call, vector search).
  • Total latency and final outcome.

Traces are written as newline-delimited JSON to a dedicated log file
(``data/logs/traces.jsonl``) and are also queryable in-process via the
``TraceStore``.

Usage::

    tracer = get_tracer()

    with tracer.trace(session_id="abc", query="Write a sort") as t:
        with t.span("routing"):
            intent = orchestrator.route_only(query)
        with t.span("agent_run", agent="code_agent"):
            response = agent.run(query)
        t.set_outcome(response.output[:100], error=response.error)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    name:       str
    started_at: float = field(default_factory=time.monotonic)
    ended_at:   Optional[float] = None
    metadata:   dict[str, Any]  = field(default_factory=dict)
    error:      Optional[str]   = None

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return (time.monotonic() - self.started_at) * 1000
        return (self.ended_at - self.started_at) * 1000

    def end(self, error: Optional[str] = None) -> None:
        self.ended_at = time.monotonic()
        if error:
            self.error = error

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "duration_ms": round(self.duration_ms, 2),
            "metadata":    self.metadata,
            "error":       self.error,
        }


@dataclass
class Trace:
    trace_id:   str
    session_id: str
    query:      str
    started_at: float = field(default_factory=time.monotonic)
    wall_time:  str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spans:      list[Span] = field(default_factory=list)
    outcome:    Optional[str] = None
    error:      Optional[str] = None
    agent_name: Optional[str] = None

    @property
    def total_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def set_outcome(self, output_preview: str, error: Optional[str] = None,
                    agent_name: Optional[str] = None) -> None:
        self.outcome    = output_preview[:200] if output_preview else None
        self.error      = error
        self.agent_name = agent_name

    def to_dict(self) -> dict:
        return {
            "trace_id":   self.trace_id,
            "session_id": self.session_id,
            "query":      self.query[:200],
            "wall_time":  self.wall_time,
            "total_ms":   round(self.total_ms, 2),
            "agent_name": self.agent_name,
            "outcome":    self.outcome,
            "error":      self.error,
            "spans":      [s.to_dict() for s in self.spans],
        }

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Generator[Span, None, None]:
        """Context manager that records a child span."""
        s = Span(name=name, metadata=metadata)
        self.spans.append(s)
        try:
            yield s
            s.end()
        except Exception as exc:
            s.end(error=str(exc))
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  TraceStore
# ─────────────────────────────────────────────────────────────────────────────

class TraceStore:
    """
    In-process circular buffer of recent traces + JSONL file sink.

    Args:
        max_in_memory: Max traces to keep in RAM (oldest evicted).
        trace_log:     Path to the JSONL file sink.
    """

    def __init__(
        self,
        max_in_memory: int = 500,
        trace_log: Optional[Path] = None,
    ) -> None:
        self._max     = max_in_memory
        self._traces: list[Trace] = []
        self._lock    = threading.Lock()
        self._log_path = trace_log or (settings.log_path.parent / "traces.jsonl")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: Trace) -> None:
        """Persist a completed trace."""
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > self._max:
                self._traces.pop(0)

        # Append to JSONL
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")
        except Exception as exc:
            logger.debug("Trace write failed: %s", exc)

    def recent(self, n: int = 20) -> list[Trace]:
        with self._lock:
            return list(self._traces[-n:])

    def for_session(self, session_id: str) -> list[Trace]:
        with self._lock:
            return [t for t in self._traces if t.session_id == session_id]

    def stats(self) -> dict:
        with self._lock:
            if not self._traces:
                return {"total": 0, "avg_ms": 0, "error_rate": 0.0}
            errors = sum(1 for t in self._traces if t.error)
            avg_ms = sum(t.total_ms for t in self._traces) / len(self._traces)
            return {
                "total":      len(self._traces),
                "avg_ms":     round(avg_ms, 1),
                "error_rate": round(errors / len(self._traces), 3),
                "agents":     self._agent_breakdown(),
            }

    def _agent_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self._traces:
            if t.agent_name:
                counts[t.agent_name] = counts.get(t.agent_name, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
#  Tracer — public API
# ─────────────────────────────────────────────────────────────────────────────

class Tracer:
    """
    Entry point for creating and recording traces.

    Usage::

        tracer = get_tracer()

        with tracer.trace("sess-1", "sort a list") as t:
            with t.span("routing"):
                ...
            with t.span("llm_call", model="llama3"):
                ...
            t.set_outcome("Here is sorted list...", agent_name="code_agent")
    """

    def __init__(self, store: Optional[TraceStore] = None) -> None:
        self._store   = store or TraceStore()
        self._counter = 0
        self._lock    = threading.Lock()

    def _next_id(self) -> str:
        with self._lock:
            self._counter += 1
            ts = int(time.time() * 1000)
            return f"tr-{ts}-{self._counter:04d}"

    @contextmanager
    def trace(self, session_id: str, query: str) -> Generator[Trace, None, None]:
        """
        Context manager for a full request trace.

        Automatically records the trace on exit, even on exception.
        """
        t = Trace(trace_id=self._next_id(), session_id=session_id, query=query)
        try:
            yield t
        except Exception as exc:
            t.error = str(exc)
            raise
        finally:
            self._store.record(t)
            logger.debug(
                "Trace %s | session=%s | %.0fms | agent=%s | error=%s",
                t.trace_id, session_id, t.total_ms, t.agent_name, t.error,
            )

    @property
    def store(self) -> TraceStore:
        return self._store

    def stats(self) -> dict:
        return self._store.stats()


# ── Singleton ─────────────────────────────────────────────────────────────────

_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = Tracer()
    return _tracer
