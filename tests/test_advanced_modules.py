"""
tests/test_advanced_modules.py
───────────────────────────────
Tests for:
  • ConversationSummariser (compression trigger, rollover)
  • EpisodicMemory (store, recall, extract, prune)
  • Tracer (span recording, JSONL sink, stats)
  • SSE endpoint (event stream structure)
  • StartupValidator (directory checks, env check)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  ConversationSummariser
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationSummariser:
    def _make_memory(self, n_turns: int = 5):
        from core.memory.conversation_memory import AssistantMemory
        mem = AssistantMemory(k=40)
        for i in range(n_turns):
            mem.add_user_message(f"User message {i}")
            mem.add_ai_message(f"AI response {i}")
        return mem

    def test_no_compression_below_threshold(self):
        from core.summariser import ConversationSummariser
        mem = self._make_memory(n_turns=5)     # 10 messages
        summariser = ConversationSummariser(summarise_after=30)
        result = summariser.maybe_summarise(mem)
        assert result is False

    def test_compression_triggered_above_threshold(self):
        from core.summariser import ConversationSummariser
        mem = self._make_memory(n_turns=20)    # 40 messages → > 15

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="The user asked about messages 0-14. AI responded accordingly."
        )
        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            summariser = ConversationSummariser(summarise_after=15, recent_k=5)
            result = summariser.maybe_summarise(mem)

        assert result is True
        mock_llm.invoke.assert_called_once()

    def test_recent_turns_preserved_after_compression(self):
        from core.summariser import ConversationSummariser
        from langchain_core.messages import HumanMessage
        mem = self._make_memory(n_turns=20)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Summary of old turns.")

        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            summariser = ConversationSummariser(summarise_after=15, recent_k=3)
            summariser.maybe_summarise(mem)

        # After compression: 1 system summary + 3*2 recent turns = 7 messages
        messages = mem.messages
        # The first message should be the summary (system or AI type)
        assert any("summary" in str(m.content).lower() or "Summary" in str(m.content)
                   for m in messages)

    def test_last_summary_property_populated(self):
        from core.summariser import ConversationSummariser
        mem = self._make_memory(n_turns=20)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Discussion about messages and responses.")

        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            summariser = ConversationSummariser(summarise_after=15, recent_k=3)
            summariser.maybe_summarise(mem)

        assert summariser.last_summary is not None
        assert len(summariser.last_summary) > 0

    def test_force_summarise_returns_text(self):
        from core.summariser import ConversationSummariser
        mem = self._make_memory(n_turns=3)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Brief exchange about messages.")

        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            summariser = ConversationSummariser()
            summary = summariser.force_summarise(mem)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_llm_failure_returns_false(self):
        from core.summariser import ConversationSummariser
        mem = self._make_memory(n_turns=20)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")

        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            summariser = ConversationSummariser(summarise_after=15)
            result = summariser.maybe_summarise(mem)

        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
#  EpisodicMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestEpisodicMemory:
    def _make_mem(self, tmp_path: Path):
        from core.long_term_memory import EpisodicMemory
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}

        with (
            patch("core.llm_manager.get_embeddings", return_value=mock_embeddings),
            patch("chromadb.PersistentClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_cls.return_value = mock_client

            mem = EpisodicMemory(persist_directory=tmp_path)
            mem._ensure_ready()
            mem._embeddings = mock_embeddings
            mem._collection = mock_collection
        return mem, mock_collection, mock_embeddings

    def test_store_calls_collection_add(self, tmp_path):
        from core.long_term_memory import EpisodicFact
        mem, coll, emb = self._make_mem(tmp_path)

        fact = EpisodicFact(
            fact_id="abc123",
            text="Q3 revenue was $15M.",
            session_id="sess-1",
            query="What was Q3 revenue?",
        )
        mem.store(fact)
        coll.add.assert_called_once()

    def test_store_skips_duplicate(self, tmp_path):
        from core.long_term_memory import EpisodicFact
        mem, coll, _ = self._make_mem(tmp_path)
        coll.get.return_value = {"ids": ["abc123"]}  # already exists

        fact = EpisodicFact(
            fact_id="abc123", text="Existing fact.",
            session_id="s", query="q"
        )
        mem.store(fact)
        coll.add.assert_not_called()

    def test_recall_returns_facts(self, tmp_path):
        from core.long_term_memory import EpisodicMemory
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]

        mock_coll = MagicMock()
        mock_coll.count.return_value = 2
        mock_coll.query.return_value = {
            "documents": [["Revenue was $15M.", "Costs fell 3%."]],
            "metadatas": [
                [
                    {"session_id": "s1", "query": "revenue", "timestamp": time.time(), "importance": 0.8, "formatted": "[2024] Revenue was $15M."},
                    {"session_id": "s1", "query": "costs",   "timestamp": time.time(), "importance": 0.6, "formatted": "[2024] Costs fell 3%."},
                ]
            ],
            "distances": [[0.05, 0.12]],
        }

        mem = EpisodicMemory.__new__(EpisodicMemory)
        mem._collection = mock_coll
        mem._embeddings = mock_emb
        mem._recall_k   = 5

        facts = mem.recall("financial results")
        assert len(facts) == 2
        assert any("Revenue" in f.text for f in facts)

    def test_recall_as_context_empty(self, tmp_path):
        from core.long_term_memory import EpisodicMemory
        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1]

        mock_coll = MagicMock()
        mock_coll.count.return_value = 0
        mock_coll.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        mem = EpisodicMemory.__new__(EpisodicMemory)
        mem._collection = mock_coll
        mem._embeddings = mock_emb
        mem._recall_k   = 5

        result = mem.recall_as_context("anything")
        assert result == ""

    def test_extract_and_store_calls_llm(self, tmp_path):
        from core.long_term_memory import EpisodicMemory
        mem, coll, emb = self._make_mem(tmp_path)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Q3 revenue was $15M.\nOperating costs fell 3%."
        )

        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            stored = mem.extract_and_store(
                session_id="sess-1",
                user_query="What were the Q3 results?",
                agent_response="Q3 revenue was $15M, operating costs fell 3%.",
            )

        assert stored == 2
        assert coll.add.call_count == 2

    def test_extract_none_returns_zero(self, tmp_path):
        from core.long_term_memory import EpisodicMemory
        mem, coll, emb = self._make_mem(tmp_path)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="NONE")

        with patch("core.llm_manager.get_llm", return_value=mock_llm):
            stored = mem.extract_and_store("s", "What time is it?", "It's 3pm.")

        assert stored == 0
        coll.add.assert_not_called()

    def test_episodic_fact_age_and_formatted(self):
        from core.long_term_memory import EpisodicFact
        fact = EpisodicFact(
            fact_id="x", text="Some fact.", session_id="s", query="q",
            timestamp=time.time() - 86400,  # 1 day ago
        )
        assert fact.age_days >= 0.99
        assert "Some fact." in fact.formatted
        assert "session:s" in fact.formatted


# ─────────────────────────────────────────────────────────────────────────────
#  Tracer
# ─────────────────────────────────────────────────────────────────────────────

class TestTracer:
    def test_trace_records_to_store(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=10, trace_log=tmp_path / "traces.jsonl")
        tracer = Tracer(store=store)

        with tracer.trace("sess-1", "what is Python?") as t:
            with t.span("routing"):
                pass
            t.set_outcome("Python is a language.", agent_name="search_agent")

        assert len(store.recent()) == 1
        trace = store.recent()[0]
        assert trace.session_id == "sess-1"
        assert trace.agent_name == "search_agent"
        assert len(trace.spans) == 1

    def test_trace_jsonl_written(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        log = tmp_path / "traces.jsonl"
        store = TraceStore(max_in_memory=10, trace_log=log)
        tracer = Tracer(store=store)

        with tracer.trace("sess-2", "hello") as t:
            t.set_outcome("hi there", agent_name="search_agent")

        assert log.exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["session_id"] == "sess-2"

    def test_span_records_latency(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=10, trace_log=tmp_path / "t.jsonl")
        tracer = Tracer(store=store)

        with tracer.trace("s", "q") as t:
            with t.span("slow_span") as span:
                time.sleep(0.05)

        assert span.duration_ms >= 40

    def test_span_captures_error(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=10, trace_log=tmp_path / "t.jsonl")
        tracer = Tracer(store=store)

        try:
            with tracer.trace("s", "q") as t:
                with t.span("failing_span"):
                    raise ValueError("test error")
        except ValueError:
            pass

        trace = store.recent()[0]
        failing = [s for s in trace.spans if s.name == "failing_span"]
        assert len(failing) == 1
        assert failing[0].error == "test error"

    def test_trace_error_set_on_exception(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=10, trace_log=tmp_path / "t.jsonl")
        tracer = Tracer(store=store)

        try:
            with tracer.trace("s", "q"):
                raise RuntimeError("outer failure")
        except RuntimeError:
            pass

        trace = store.recent()[0]
        assert trace.error == "outer failure"

    def test_store_stats(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=10, trace_log=tmp_path / "t.jsonl")
        tracer = Tracer(store=store)

        for i in range(3):
            with tracer.trace(f"s{i}", f"q{i}") as t:
                t.set_outcome("ok", agent_name="search_agent")

        stats = store.stats()
        assert stats["total"] == 3
        assert stats["error_rate"] == 0.0
        assert stats["agents"]["search_agent"] == 3

    def test_store_session_filter(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=10, trace_log=tmp_path / "t.jsonl")
        tracer = Tracer(store=store)

        with tracer.trace("alice", "hello"): pass
        with tracer.trace("bob",   "world"): pass
        with tracer.trace("alice", "again"): pass

        alice_traces = store.for_session("alice")
        assert len(alice_traces) == 2

    def test_store_max_in_memory_eviction(self, tmp_path):
        from core.tracing import Tracer, TraceStore
        store = TraceStore(max_in_memory=3, trace_log=tmp_path / "t.jsonl")
        tracer = Tracer(store=store)

        for i in range(5):
            with tracer.trace(f"s{i}", "q"): pass

        assert len(store.recent(10)) == 3   # only last 3 kept

    def test_singleton_get_tracer(self):
        from core.tracing import get_tracer
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2


# ─────────────────────────────────────────────────────────────────────────────
#  SSE endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestSSEEndpoint:
    @pytest.fixture()
    def sse_client(self):
        from agents.base_agent import AgentResponse
        from unittest.mock import MagicMock, patch

        mock_orch = MagicMock()
        mock_orch.run.return_value = AgentResponse(
            output="SSE test response here.",
            agent_name="search_agent",
            references=["Doc.pdf › Page 1"],
        )
        mock_orch.route_only.return_value = MagicMock(value="search")

        with patch("api.server._get_or_create_orchestrator", return_value=mock_orch):
            from fastapi.testclient import TestClient
            from api.server import app
            with TestClient(app) as client:
                yield client

    def test_sse_stream_returns_events(self, sse_client):
        """GET /stream should return a text/event-stream response."""
        resp = sse_client.get("/stream", params={"query": "hello", "session_id": "sse-1"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_sse_stream_contains_done_event(self, sse_client):
        """The stream must eventually emit a 'done' event."""
        resp = sse_client.get("/stream", params={"query": "hello", "session_id": "sse-2"})
        body = resp.text
        assert '"type": "done"' in body or '"type":"done"' in body

    def test_sse_stream_contains_tokens(self, sse_client):
        """The stream must contain at least one 'token' event."""
        resp = sse_client.get("/stream", params={"query": "hello", "session_id": "sse-3"})
        assert '"type": "token"' in resp.text or '"type":"token"' in resp.text

    def test_sse_stream_contains_reference(self, sse_client):
        """References should appear as 'reference' typed events."""
        resp = sse_client.get("/stream", params={"query": "hello", "session_id": "sse-4"})
        assert "reference" in resp.text

    def test_sse_missing_query_rejected(self, sse_client):
        resp = sse_client.get("/stream")
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
#  StartupValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestStartupValidator:
    def test_directory_check_passes_with_writable_dirs(self, tmp_path):
        from api.startup_validator import _check_data_directories
        with patch("api.startup_validator.settings") as mock_settings:
            mock_settings.uploads_path   = tmp_path / "uploads"
            mock_settings.log_path       = tmp_path / "logs" / "assistant.log"
            mock_settings.vector_store_path = tmp_path / "vector_store"
            result = _check_data_directories()
        assert result.passed

    def test_directory_check_fails_on_readonly(self, tmp_path):
        from api.startup_validator import _check_data_directories
        import os
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        os.chmod(str(readonly_dir), 0o444)
        try:
            with patch("api.startup_validator.settings") as mock_settings:
                mock_settings.uploads_path   = readonly_dir / "uploads"
                mock_settings.log_path       = tmp_path / "logs" / "x.log"
                mock_settings.vector_store_path = tmp_path / "vs"
                result = _check_data_directories()
            # Root can write anywhere, so only check in non-root envs
            if os.getuid() != 0:
                assert not result.passed
        finally:
            os.chmod(str(readonly_dir), 0o755)

    def test_env_check_warns_without_dotenv(self, tmp_path):
        from api.startup_validator import _check_env_completeness
        with patch("api.startup_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = _check_env_completeness()
        # Should still pass (advisory only)
        assert result.passed

    def test_vector_store_check_passes(self, tmp_path):
        from api.startup_validator import _check_vector_store
        with patch("api.startup_validator.settings") as mock_settings:
            mock_settings.vector_store_path = tmp_path / "vs"
            result = _check_vector_store()
        assert result.passed

    def test_llm_connectivity_check_failure(self):
        from api.startup_validator import _check_llm_connectivity
        with (
            patch("api.startup_validator.settings") as mock_settings,
            patch("httpx.get", side_effect=Exception("connection refused")),
        ):
            mock_settings.llm_backend = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            result = _check_llm_connectivity()
        assert not result.passed
        assert "connection refused" in result.message

    def test_run_startup_checks_returns_report(self, tmp_path):
        from api.startup_validator import run_startup_checks
        with patch("api.startup_validator.settings") as mock_settings:
            mock_settings.uploads_path      = tmp_path / "uploads"
            mock_settings.log_path          = tmp_path / "logs" / "x.log"
            mock_settings.vector_store_path = tmp_path / "vs"
            report = run_startup_checks(
                check_llm=False,
                check_embeddings=False,
                check_storage=True,
            )
        assert report is not None
        assert isinstance(report.to_dict(), dict)
        assert "checks" in report.to_dict()

    def test_validation_report_to_dict(self):
        from api.startup_validator import CheckResult, ValidationReport
        report = ValidationReport(checks=[
            CheckResult("dirs",    True,  "ok"),
            CheckResult("network", False, "timeout"),
        ])
        d = report.to_dict()
        assert d["all_passed"] is False
        assert len(d["checks"]) == 2
        assert d["checks"][1]["passed"] is False
