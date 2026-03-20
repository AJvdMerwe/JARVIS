"""
api/rate_limiter.py
────────────────────
Simple in-memory sliding-window rate limiter for the FastAPI server.

Per-client limits keyed by IP address (or X-Forwarded-For in proxied
deployments). Configurable via environment variables:

  RATE_LIMIT_REQUESTS=60   # requests per window
  RATE_LIMIT_WINDOW=60     # window size in seconds

Usage in server.py:
    from api.rate_limiter import RateLimitMiddleware
    app.add_middleware(RateLimitMiddleware)
"""
from __future__ import annotations

import collections
import logging
import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket-style sliding window rate limiter.

    Thread-safe via collections.deque's atomic append/popleft.
    Each client gets its own deque of timestamps.

    Args:
        max_requests: Maximum requests allowed per window.
        window_seconds: Rolling window length in seconds.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests    = max_requests
        self.window_seconds  = window_seconds
        # client_id → deque of timestamps (oldest first)
        self._windows: dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque()
        )

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check whether a request from client_id is within limits.

        Returns:
            (allowed: bool, remaining: int)  — requests remaining in window.
        """
        now  = time.monotonic()
        cutoff = now - self.window_seconds
        window = self._windows[client_id]

        # Evict timestamps outside the current window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            return False, 0

        window.append(now)
        remaining = self.max_requests - len(window)
        return True, remaining

    def reset(self, client_id: str) -> None:
        """Clear the rate-limit window for a specific client (useful for tests)."""
        self._windows.pop(client_id, None)

    def stats(self) -> dict:
        return {
            "tracked_clients": len(self._windows),
            "max_requests":    self.max_requests,
            "window_seconds":  self.window_seconds,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        import os
        max_req = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
        window  = float(os.environ.get("RATE_LIMIT_WINDOW",   "60"))
        _limiter = RateLimiter(max_requests=max_req, window_seconds=window)
    return _limiter


# ── Middleware ────────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces per-IP rate limits.

    Adds headers to every response:
      X-RateLimit-Limit     — max requests per window
      X-RateLimit-Remaining — remaining requests
      Retry-After           — seconds until window resets (on 429 only)
    """

    EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        limiter = get_rate_limiter()
        client_ip = self._get_client_ip(request)
        allowed, remaining = limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning("Rate limit exceeded for %s on %s", client_ip, request.url.path)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after": int(limiter.window_seconds),
                },
                headers={
                    "X-RateLimit-Limit":     str(limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After":           str(int(limiter.window_seconds)),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
