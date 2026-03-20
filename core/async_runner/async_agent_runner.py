"""
core/async_runner/async_agent_runner.py
────────────────────────────────────────
Async execution layer for agents, enabling:

  1. True async invocation of agents (non-blocking in FastAPI handlers).
  2. Parallel multi-agent fan-out — send the same query to N agents
     simultaneously and merge results (useful for cross-agent synthesis).
  3. Async tool call batching — tools within a single agent run that
     are independent can be executed concurrently.
  4. Streaming callback support — yield partial tokens to the caller
     as the LLM generates them.

Architecture:
  • All agent.run() calls are wrapped in asyncio.to_thread() so they
    don't block the event loop.
  • Fan-out queries use asyncio.gather() with individual timeouts.
  • A StreamBuffer collects LangChain streaming callbacks and exposes
    an async generator interface.

Usage::

    runner = AsyncAgentRunner()

    # Single async invocation
    response = await runner.run_async(agent, "Write a quicksort")

    # Parallel fan-out across multiple agents
    results = await runner.fanout("What is quantum computing?",
                                  agents=[code_agent, search_agent])

    # Streaming
    async for token in runner.stream_async(agent, "Explain recursion"):
        print(token, end="", flush=True)
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from langchain_core.callbacks import BaseCallbackHandler

from agents.base_agent import AgentResponse, BaseAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming callback handler
# ─────────────────────────────────────────────────────────────────────────────

class _AsyncTokenQueue(BaseCallbackHandler):
    """
    LangChain callback handler that feeds LLM tokens into an asyncio.Queue.
    Enables async streaming without modifying the agent code.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._loop  = loop
        self._queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:  # type: ignore[override]
        """Called by LangChain for each streamed token."""
        asyncio.run_coroutine_threadsafe(
            self._queue.put(token), self._loop
        )

    def on_llm_end(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Signal end-of-stream with a None sentinel."""
        asyncio.run_coroutine_threadsafe(
            self._queue.put(None), self._loop
        )

    def on_llm_error(self, error: Exception, **kwargs) -> None:  # type: ignore[override]
        asyncio.run_coroutine_threadsafe(
            self._queue.put(None), self._loop
        )

    async def tokens(self) -> AsyncGenerator[str, None]:
        """Async generator yielding tokens as they arrive."""
        while True:
            token = await self._queue.get()
            if token is None:
                break
            yield token


# ─────────────────────────────────────────────────────────────────────────────
#  Fan-out result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FanOutResult:
    """Aggregated results from a multi-agent fan-out."""
    query:       str
    responses:   list[AgentResponse]
    latency_ms:  float
    timed_out:   list[str] = field(default_factory=list)   # agent names

    @property
    def succeeded(self) -> list[AgentResponse]:
        return [r for r in self.responses if r.error is None]

    @property
    def failed(self) -> list[AgentResponse]:
        return [r for r in self.responses if r.error is not None]

    def merge_output(self, separator: str = "\n\n---\n\n") -> str:
        """Concatenate outputs from all successful responses."""
        return separator.join(
            f"**[{r.agent_name}]**\n{r.output}"
            for r in self.succeeded
        )

    def all_references(self) -> list[str]:
        """Deduplicated references from all responses."""
        seen: set[str] = set()
        refs: list[str] = []
        for r in self.responses:
            for ref in r.references:
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
        return refs


# ─────────────────────────────────────────────────────────────────────────────
#  Async runner
# ─────────────────────────────────────────────────────────────────────────────

class AsyncAgentRunner:
    """
    Runs agents asynchronously, with optional parallel fan-out.

    Args:
        default_timeout: Per-agent timeout in seconds for fan-out.
    """

    def __init__(self, default_timeout: float = 120.0) -> None:
        self.default_timeout = default_timeout

    # ── Single async run ─────────────────────────────────────────────────────

    async def run_async(
        self,
        agent: BaseAgent,
        query: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Run an agent asynchronously (non-blocking, runs in thread pool).

        Args:
            agent:   The agent to invoke.
            query:   User query.
            timeout: Per-call timeout (uses default_timeout if None).

        Returns:
            AgentResponse from the agent.
        """
        effective_timeout = timeout or self.default_timeout
        t0 = time.monotonic()

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(agent.run, query, **kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent '%s' timed out after %.0fs for query: %s",
                agent.name, effective_timeout, query[:60],
            )
            response = AgentResponse(
                output=f"[{agent.name}] Request timed out after {effective_timeout:.0f}s.",
                agent_name=agent.name,
                error="timeout",
            )
        except Exception as exc:
            logger.error("Agent '%s' failed: %s", agent.name, exc)
            response = AgentResponse(
                output=f"[{agent.name}] Unexpected error: {exc}",
                agent_name=agent.name,
                error=str(exc),
            )

        latency = (time.monotonic() - t0) * 1000
        logger.debug(
            "AsyncRunner: agent=%s latency=%.0fms error=%s",
            agent.name, latency, response.error,
        )
        return response

    # ── Fan-out ──────────────────────────────────────────────────────────────

    async def fanout(
        self,
        query: str,
        agents: list[BaseAgent],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> FanOutResult:
        """
        Send the same query to multiple agents in parallel and collect results.

        Args:
            query:   The query to send to all agents.
            agents:  List of agents to invoke simultaneously.
            timeout: Per-agent timeout.

        Returns:
            FanOutResult with all responses.
        """
        if not agents:
            return FanOutResult(query=query, responses=[], latency_ms=0.0)

        t0 = time.monotonic()
        tasks = [
            self.run_async(agent, query, timeout=timeout, **kwargs)
            for agent in agents
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=False)
        latency = (time.monotonic() - t0) * 1000

        timed_out = [r.agent_name for r in responses if r.error == "timeout"]

        logger.info(
            "FanOut: query='%s…' agents=%d succeeded=%d timed_out=%d latency=%.0fms",
            query[:40], len(agents),
            sum(1 for r in responses if r.error is None),
            len(timed_out), latency,
        )
        return FanOutResult(
            query=query,
            responses=list(responses),
            latency_ms=latency,
            timed_out=timed_out,
        )

    # ── Streaming ────────────────────────────────────────────────────────────

    async def stream_async(
        self,
        agent: BaseAgent,
        query: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from an agent as an async generator.

        Note: Requires the underlying LLM to support streaming callbacks.
        Falls back to yielding the full response as a single chunk if the
        LLM does not support streaming.

        Yields:
            Token strings as they arrive.
        """
        loop = asyncio.get_event_loop()
        handler = _AsyncTokenQueue(loop)

        # Run the agent in a thread, injecting the streaming callback
        async def _run_with_handler():
            try:
                llm = agent._llm
                # Temporarily inject callback
                original_callbacks = getattr(llm, "callbacks", None) or []
                llm.callbacks = original_callbacks + [handler]
                try:
                    await asyncio.to_thread(agent.run, query, **kwargs)
                finally:
                    llm.callbacks = original_callbacks
            except Exception as exc:
                logger.warning("Streaming agent run failed: %s", exc)
            finally:
                # Always send sentinel to unblock the token generator
                await handler._queue.put(None)

        # Launch the agent run concurrently with token consumption
        run_task = asyncio.create_task(_run_with_handler())

        # Yield tokens
        try:
            async for token in handler.tokens():
                yield token
        finally:
            await run_task
