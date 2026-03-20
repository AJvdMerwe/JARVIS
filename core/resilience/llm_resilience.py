"""
core/resilience/llm_resilience.py
───────────────────────────────────
Production-grade resilience wrapper around any LangChain ChatModel.

Features:
  • Exponential back-off with jitter on rate-limit / server errors.
  • Hard timeout per invocation (avoids hanging forever on a stalled model).
  • Automatic backend failover: if the primary LLM fails N times, switch to
    the secondary backend transparently and alert via logging.
  • Circuit-breaker: after K consecutive failures the circuit opens and
    requests fail-fast with a clear error rather than hammering a dead server.
  • Structured per-call metrics (latency, retries, which backend was used).
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit breaker states
# ─────────────────────────────────────────────────────────────────────────────

class CircuitState(str, Enum):
    CLOSED   = "closed"    # normal — requests flow through
    OPEN     = "open"      # tripped — requests fail immediately
    HALF_OPEN = "half_open" # probe — one request let through to test recovery


@dataclass
class CallMetrics:
    backend: str
    latency_ms: float
    retries: int
    success: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit breaker
# ─────────────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Thread-safe circuit breaker.

    Args:
        failure_threshold:  Consecutive failures before opening.
        recovery_timeout:   Seconds before moving OPEN → HALF_OPEN.
        half_open_probes:   Successful probes required to close again.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_probes: int = 2,
    ) -> None:
        self._threshold = failure_threshold
        self._timeout = recovery_timeout
        self._probes_needed = half_open_probes

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._probes = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> CircuitState:
        """Evaluate state transitions based on elapsed time (must hold lock)."""
        if self._state == CircuitState.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self._timeout:
                self._state = CircuitState.HALF_OPEN
                self._probes = 0
                logger.info("Circuit → HALF_OPEN (probing recovery).")
        return self._state

    def allow_request(self) -> bool:
        with self._lock:
            state = self._current_state()
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                return True  # let one probe through
            return False  # OPEN — fail fast

    def record_success(self) -> None:
        with self._lock:
            state = self._current_state()
            self._failures = 0
            if state == CircuitState.HALF_OPEN:
                self._probes += 1
                if self._probes >= self._probes_needed:
                    self._state = CircuitState.CLOSED
                    logger.info("Circuit → CLOSED (recovery confirmed).")

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # Single probe failure → back to OPEN
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning("Circuit → OPEN (probe failed, backing off).")
            elif self._failures >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit → OPEN (%d consecutive failures).", self._failures
                )


# ─────────────────────────────────────────────────────────────────────────────
#  Retryable errors (retry on these; give up on others)
# ─────────────────────────────────────────────────────────────────────────────

_RETRYABLE_SUBSTRINGS = (
    "rate limit",
    "too many requests",
    "503",
    "502",
    "504",
    "timeout",
    "connection error",
    "server error",
    "overloaded",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


# ─────────────────────────────────────────────────────────────────────────────
#  Resilient LLM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ResilientLLM:
    """
    Wraps a primary (and optional fallback) LangChain ChatModel with:
      - Retry with exponential back-off + jitter
      - Per-call timeout
      - Circuit breaker
      - Automatic failover to a secondary backend

    Usage::

        from core.resilience import ResilientLLM
        llm = ResilientLLM()
        response = llm.invoke("Hello!")
        print(llm.last_metrics)
    """

    def __init__(
        self,
        primary: Optional[BaseChatModel] = None,
        fallback: Optional[BaseChatModel] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 120.0,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        from core.llm_manager import get_llm
        self._primary  = primary  or get_llm()
        self._fallback = fallback
        self._max_retries  = max_retries
        self._base_delay   = base_delay
        self._max_delay    = max_delay
        self._timeout      = timeout
        self._circuit      = CircuitBreaker(failure_threshold, recovery_timeout)
        self.last_metrics: Optional[CallMetrics] = None

    # ── Public API (mirrors BaseChatModel) ──────────────────────────────────

    def invoke(
        self,
        input: Any,   # str | list[BaseMessage] | PromptValue
        **kwargs: Any,
    ) -> BaseMessage:
        """Invoke with retry + circuit breaker + failover."""
        return self._call_with_resilience(self._primary, "primary", input, **kwargs)

    def stream(self, input: Any, **kwargs: Any):
        """Stream tokens (no retry — streaming is not idempotent)."""
        yield from self._primary.stream(input, **kwargs)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _call_with_resilience(
        self,
        llm: BaseChatModel,
        backend_label: str,
        input: Any,
        **kwargs: Any,
    ) -> BaseMessage:
        if not self._circuit.allow_request():
            # Try fallback before giving up
            if self._fallback and backend_label == "primary":
                logger.warning("Circuit open on primary — using fallback directly.")
                return self._invoke_once(self._fallback, "fallback", input, **kwargs)
            raise RuntimeError(
                f"LLM circuit breaker is OPEN ({self._circuit.state}). "
                "The model server appears to be unavailable. Try again later."
            )

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._invoke_once(llm, backend_label, input, **kwargs)
                self._circuit.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                self._circuit.record_failure()
                if not _is_retryable(exc) or attempt == self._max_retries:
                    break
                delay = min(
                    self._base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1),
                    self._max_delay,
                )
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs…",
                    attempt, self._max_retries, exc, delay,
                )
                time.sleep(delay)

        # All retries on primary exhausted — try fallback
        if self._fallback and backend_label == "primary":
            logger.warning("Primary LLM exhausted retries, switching to fallback.")
            try:
                return self._invoke_once(self._fallback, "fallback", input, **kwargs)
            except Exception as fallback_exc:
                logger.error("Fallback LLM also failed: %s", fallback_exc)
                raise fallback_exc from last_exc

        raise last_exc

    def _invoke_once(
        self,
        llm: BaseChatModel,
        label: str,
        input: Any,
        **kwargs: Any,
    ) -> BaseMessage:
        """Single timed invocation, records metrics."""
        import concurrent.futures

        t0 = time.monotonic()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(llm.invoke, input, **kwargs)
                try:
                    result = future.result(timeout=self._timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(
                        f"LLM ({label}) did not respond within {self._timeout}s."
                    )
            latency = (time.monotonic() - t0) * 1000
            self.last_metrics = CallMetrics(
                backend=label, latency_ms=latency, retries=0, success=True
            )
            logger.debug("LLM %s responded in %.0fms.", label, latency)
            return result
        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            self.last_metrics = CallMetrics(
                backend=label, latency_ms=latency, retries=0,
                success=False, error=str(exc),
            )
            raise
