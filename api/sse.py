"""
api/sse.py
───────────
Server-Sent Events (SSE) streaming endpoint.

Provides an alternative to the WebSocket endpoint for clients that
prefer a simpler HTTP/1.1 streaming interface (curl, EventSource in
browsers, etc.).

Endpoint:
  GET /stream?query=...&session_id=...&intent=...

Client receives a stream of SSE events:
  data: {"type": "start",     "agent": "code_agent"}
  data: {"type": "token",     "content": "Here "}
  data: {"type": "reference", "ref": "Doc › Page 3"}
  data: {"type": "done",      "latency_ms": 1234}
  data: {"type": "error",     "message": "..."}

Example (curl):
  curl -N "http://localhost:8080/stream?query=What+is+Python&session_id=my-session"

Example (JavaScript):
  const es = new EventSource('/stream?query=hello&session_id=s1');
  es.onmessage = e => console.log(JSON.parse(e.data));
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Streaming"])


# ─────────────────────────────────────────────────────────────────────────────
#  SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _sse_heartbeat() -> str:
    """Keep-alive comment (SSE comment syntax)."""
    return ": heartbeat\n\n"


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming generator
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_response(
    query: str,
    session_id: str,
    intent: Optional[str],
) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the agent in a thread and streams
    the response word-by-word as SSE events.
    """
    from api.server import _get_or_create_orchestrator

    orch = _get_or_create_orchestrator(session_id)

    # Detect intent (fast, no LLM)
    detected = orch.route_only(query)
    yield _sse_event({"type": "start", "agent": detected.value})

    t0 = time.monotonic()
    kwargs = {}
    if intent:
        kwargs["intent"] = intent

    # Run agent in thread pool (agents are synchronous)
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: orch.run(query, **kwargs)
        )
    except Exception as exc:
        yield _sse_event({"type": "error", "message": str(exc)})
        return

    latency_ms = (time.monotonic() - t0) * 1000

    # Stream tokens word-by-word with natural pacing
    words = response.output.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield _sse_event({"type": "token", "content": chunk})
        await asyncio.sleep(0.004)   # ~250 words/sec pacing

    # Emit references
    for ref in response.references:
        yield _sse_event({"type": "reference", "ref": ref})

    # Done
    yield _sse_event({
        "type":            "done",
        "agent":           response.agent_name,
        "latency_ms":      round(latency_ms, 1),
        "tool_call_count": len(response.tool_calls),
        "has_error":       response.error is not None,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Route
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/stream",
    summary="Stream a chat response via Server-Sent Events",
    description=(
        "Send a query and receive a token-by-token streamed response "
        "as Server-Sent Events. Suitable for browser EventSource or curl -N."
    ),
)
async def stream_chat(
    query:      str            = Query(..., min_length=1, max_length=4096, description="Your question"),
    session_id: str            = Query(default="sse-default", description="Session ID for memory"),
    intent:     Optional[str]  = Query(default=None, description="Force agent: code|news|search|document"),
) -> StreamingResponse:
    """SSE streaming chat endpoint."""

    async def event_stream() -> AsyncGenerator[str, None]:
        # Initial heartbeat so the client knows the connection is live
        yield _sse_heartbeat()
        async for event in _stream_response(query, session_id, intent):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            "Connection":       "keep-alive",
        },
    )
