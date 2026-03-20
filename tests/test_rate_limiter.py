"""
tests/test_rate_limiter.py
───────────────────────────
Unit and integration tests for the API rate limiter and the
resilience+cache pipeline working in combination.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  RateLimiter unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiter:
    def _make(self, max_requests=5, window=60.0):
        from api.rate_limiter import RateLimiter
        return RateLimiter(max_requests=max_requests, window_seconds=window)

    def test_allows_under_limit(self):
        rl = self._make(max_requests=5)
        for _ in range(5):
            allowed, remaining = rl.is_allowed("client1")
            assert allowed

    def test_blocks_over_limit(self):
        rl = self._make(max_requests=3)
        for _ in range(3):
            rl.is_allowed("client1")
        allowed, remaining = rl.is_allowed("client1")
        assert not allowed
        assert remaining == 0

    def test_remaining_decrements(self):
        rl = self._make(max_requests=5)
        _, r1 = rl.is_allowed("c1")
        _, r2 = rl.is_allowed("c1")
        assert r1 == 4
        assert r2 == 3

    def test_different_clients_isolated(self):
        rl = self._make(max_requests=2)
        for _ in range(2):
            rl.is_allowed("alice")
        # alice is at limit, bob is not
        blocked, _ = rl.is_allowed("alice")
        allowed, _ = rl.is_allowed("bob")
        assert not blocked
        assert allowed

    def test_window_expiry_allows_again(self):
        rl = self._make(max_requests=2, window=0.1)
        rl.is_allowed("c1")
        rl.is_allowed("c1")
        # at limit
        assert not rl.is_allowed("c1")[0]
        # wait for window to expire
        time.sleep(0.15)
        assert rl.is_allowed("c1")[0]

    def test_reset_clears_client(self):
        rl = self._make(max_requests=2)
        rl.is_allowed("c1")
        rl.is_allowed("c1")
        assert not rl.is_allowed("c1")[0]
        rl.reset("c1")
        assert rl.is_allowed("c1")[0]

    def test_stats(self):
        rl = self._make(max_requests=10, window=30)
        rl.is_allowed("c1")
        stats = rl.stats()
        assert stats["tracked_clients"] == 1
        assert stats["max_requests"] == 10
        assert stats["window_seconds"] == 30


# ─────────────────────────────────────────────────────────────────────────────
#  Rate limiter middleware via HTTP
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimitMiddleware:
    @pytest.fixture()
    def test_client(self):
        from unittest.mock import MagicMock, patch
        from agents.base_agent import AgentResponse

        mock_orch = MagicMock()
        mock_orch.run.return_value = AgentResponse(
            output="ok", agent_name="search_agent"
        )
        mock_orch.route_only.return_value = MagicMock(value="search")

        with patch("api.server._get_or_create_orchestrator", return_value=mock_orch):
            # Use tight limits for testing
            with patch("api.rate_limiter.get_rate_limiter") as mock_get_rl:
                from api.rate_limiter import RateLimiter
                tight_rl = RateLimiter(max_requests=3, window_seconds=60)
                mock_get_rl.return_value = tight_rl

                from fastapi.testclient import TestClient
                from api.server import app
                with TestClient(app) as client:
                    yield client, tight_rl

    def test_rate_limit_headers_present(self, test_client):
        client, rl = test_client
        resp = client.post("/chat", json={"query": "hello", "session_id": "rl-test"})
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_enforced(self, test_client):
        client, rl = test_client
        # Exhaust the 3-request limit
        for _ in range(3):
            client.post("/chat", json={"query": "hello", "session_id": "rl2"})
        # 4th should be blocked
        resp = client.post("/chat", json={"query": "hello", "session_id": "rl2"})
        assert resp.status_code == 429
        assert "retry_after" in resp.json()

    def test_health_exempt_from_rate_limit(self, test_client):
        client, rl = test_client
        # Exhaust limit for this IP
        rl._windows.clear()
        for _ in range(3):
            rl.is_allowed("testclient")
        # Health check must still pass regardless of rate limit state
        with patch("document_processing.DocumentManager") as mock_dm:
            mock_dm.return_value.total_chunks = 0
            resp = client.get("/health")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
#  Resilience + Cache integration (combined pipeline test)
# ─────────────────────────────────────────────────────────────────────────────

class TestResilienceCachePipeline:
    """
    Verify that the circuit breaker and tool cache work together correctly:
    - Tool results are cached after first call.
    - When primary LLM fails, fallback kicks in.
    - Cached results are served even when LLM is down.
    """

    def test_cache_prevents_redundant_calls(self):
        """Same tool call twice → only one real invocation."""
        from core.cache.tool_cache import ToolCache, cached_tool

        cache = ToolCache(default_ttl=60.0)
        call_count = [0]

        class SearchTool:
            @cached_tool("web_search", ttl=60)
            def _run(self, query: str) -> str:
                call_count[0] += 1
                return f"Results for {query}"

        tool = SearchTool()

        with patch("core.cache.tool_cache.get_cache", return_value=cache):
            r1 = tool._run("quantum computing")
            r2 = tool._run("quantum computing")   # should hit cache
            r3 = tool._run("machine learning")    # different query → new call

        assert r1 == r2
        assert call_count[0] == 2   # "quantum computing" × 1 + "machine learning" × 1
        assert cache.hits == 1
        assert cache.misses == 2

    def test_circuit_opens_on_repeated_failures(self):
        """After threshold failures, circuit opens and subsequent calls fail fast."""
        from core.resilience.llm_resilience import CircuitBreaker, CircuitState, ResilientLLM

        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = Exception("connection refused")

        rl = ResilientLLM(
            primary=failing_llm,
            max_retries=1,
            base_delay=0.0,
            failure_threshold=2,
            timeout=1.0,
        )
        # Two calls to trip the breaker (each uses 1 retry = 2 total failures)
        for _ in range(2):
            try:
                rl.invoke("test")
            except Exception:
                pass

        assert rl._circuit.state == CircuitState.OPEN

        # Next call should fail fast without hitting the LLM
        call_count_before = failing_llm.invoke.call_count
        try:
            rl.invoke("another call")
        except RuntimeError as e:
            assert "circuit breaker" in str(e).lower() or "OPEN" in str(e)
        assert failing_llm.invoke.call_count == call_count_before  # no new calls

    def test_fallback_serves_response_when_primary_down(self):
        """ResilientLLM falls back to secondary when primary is exhausted."""
        from core.resilience.llm_resilience import ResilientLLM

        primary  = MagicMock()
        primary.invoke.side_effect = Exception("rate limit exceeded")

        fallback = MagicMock()
        fallback.invoke.return_value = MagicMock(content="from fallback LLM")

        with patch("time.sleep"):
            rl = ResilientLLM(
                primary=primary,
                fallback=fallback,
                max_retries=2,
                base_delay=0.0,
                timeout=5.0,
            )
            result = rl.invoke("query")

        assert result.content == "from fallback LLM"
        assert rl.last_metrics.backend == "fallback"

    def test_metrics_captured_on_success(self):
        """CallMetrics is populated correctly after a successful invocation."""
        from core.resilience.llm_resilience import ResilientLLM

        fast_llm = MagicMock()
        fast_llm.invoke.return_value = MagicMock(content="fast response")

        rl = ResilientLLM(primary=fast_llm, max_retries=1, timeout=5.0)
        rl.invoke("hello")

        assert rl.last_metrics is not None
        assert rl.last_metrics.success
        assert rl.last_metrics.latency_ms >= 0
        assert rl.last_metrics.backend == "primary"

    def test_cache_stats_accumulate_correctly(self):
        """Hit/miss counters and hit_rate property are correct."""
        from core.cache.tool_cache import ToolCache

        cache = ToolCache(default_ttl=60.0)
        cache.set("k1", "v1")
        cache.set("k2", "v2")

        cache.get("k1")   # hit
        cache.get("k1")   # hit
        cache.get("k3")   # miss
        cache.get("k4")   # miss

        assert cache.hits == 2
        assert cache.misses == 2
        assert cache.hit_rate == pytest.approx(0.5)
        assert cache.size == 2

    def test_circuit_half_open_recovers(self):
        """Circuit transitions OPEN→HALF_OPEN→CLOSED after recovery."""
        from core.resilience.llm_resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            half_open_probes=1,
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_cache_persists_and_reloads(self, tmp_path):
        """Disk-persisted cache survives reload."""
        from core.cache.tool_cache import ToolCache

        persist_file = tmp_path / "cache.json"
        c1 = ToolCache(default_ttl=300.0, persist_path=persist_file)
        c1.set("key_abc", "value_xyz")
        c1.save_to_disk()

        c2 = ToolCache(default_ttl=300.0, persist_path=persist_file)
        result = c2.get("key_abc")
        assert result == "value_xyz"
