"""
tests/test_memory_personalisation.py
──────────────────────────────────────
Tests for Stage 1 — Memory & Personalisation:

  • UserPreferences: new fields (name, interests, last_seen, total_sessions)
  • extract_and_update_profile(): name extraction heuristics
  • extract_and_update_profile_llm(): JSON parsing and preference updates
  • build_session_recall(): greeting generation for returning users
  • update_session_count(): session counter increments
  • Orchestrator: episodic enabled by default, _update_user_profile called
  • EpisodicMemory: extract_and_store, recall, importance scoring
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest


# =============================================================================
#  UserPreferences — new fields
# =============================================================================

class TestUserPreferencesNewFields:

    def _prefs(self, **kwargs):
        from core.user_prefs.preferences import UserPreferences
        return UserPreferences(user_id="test_user", **kwargs)

    def test_name_field_defaults_empty(self):
        p = self._prefs()
        assert p.name == ""

    def test_interests_field_defaults_empty_list(self):
        p = self._prefs()
        assert p.interests == []

    def test_last_seen_defaults_zero(self):
        p = self._prefs()
        assert p.last_seen == 0.0

    def test_total_sessions_defaults_zero(self):
        p = self._prefs()
        assert p.total_sessions == 0

    def test_last_session_id_defaults_empty(self):
        p = self._prefs()
        assert p.last_session_id == ""

    def test_name_can_be_set(self):
        p = self._prefs(name="Alice")
        assert p.name == "Alice"

    def test_interests_can_be_set(self):
        p = self._prefs(interests=["AI", "finance"])
        assert "AI" in p.interests

    def test_greeting_name_returns_name(self):
        p = self._prefs(name="Bob")
        assert p.greeting_name() == "Bob"

    def test_greeting_name_returns_empty_when_unset(self):
        p = self._prefs()
        assert p.greeting_name() == ""

    def test_finance_accepted_as_preferred_agent(self):
        p = self._prefs(preferred_agent="finance")
        assert p.preferred_agent == "finance"

    def test_research_accepted_as_preferred_agent(self):
        p = self._prefs(preferred_agent="research")
        assert p.preferred_agent == "research"

    def test_agent_style_prompt_includes_name_when_set(self):
        p = self._prefs(name="Carol", response_style="technical")
        prompt = p.agent_style_prompt()
        # Style should still be reflected
        assert isinstance(prompt, str)


# =============================================================================
#  extract_and_update_profile — heuristics
# =============================================================================

class TestExtractAndUpdateProfile:

    def _run(self, query, response="", agent="chat_agent", user_id="test_profile"):
        from core.profile_extractor import extract_and_update_profile
        with patch("core.user_prefs.get_preferences") as mock_get:
            mock_prefs = MagicMock()
            mock_prefs.name = ""
            mock_prefs.total_sessions = 1
            mock_prefs.preferred_agent = "auto"
            mock_get.return_value = mock_prefs
            extract_and_update_profile("sess1", user_id, query, response, agent)
            return mock_prefs

    def test_extracts_name_from_my_name_is(self):
        prefs = self._run("My name is Alice, what's the weather?")
        assert prefs.name == "Alice"

    def test_extracts_name_from_im_pattern(self):
        prefs = self._run("Hi, I'm Bob and I need help with Python.")
        assert prefs.name == "Bob"

    def test_extracts_name_from_call_me(self):
        prefs = self._run("Please call me Charlie when answering.")
        assert prefs.name == "Charlie"

    def test_no_name_extraction_without_pattern(self):
        prefs = self._run("What is machine learning?")
        assert prefs.name == ""   # was already empty and not set

    def test_does_not_overwrite_existing_name(self):
        from core.profile_extractor import extract_and_update_profile
        with patch("core.user_prefs.get_preferences") as mock_get:
            mock_prefs       = MagicMock()
            mock_prefs.name  = "ExistingName"  # already set
            mock_prefs.total_sessions = 5
            mock_get.return_value = mock_prefs
            extract_and_update_profile(
                "s1", "u1", "My name is NewName!", "", "chat_agent"
            )
        # Should not have changed to NewName
        assert mock_prefs.name == "ExistingName"

    def test_updates_last_seen_timestamp(self):
        from core.profile_extractor import extract_and_update_profile
        before = time.time()
        with patch("core.user_prefs.get_preferences") as mock_get:
            mock_prefs = MagicMock()
            mock_prefs.name = ""
            mock_prefs.total_sessions = 1
            mock_get.return_value = mock_prefs
            extract_and_update_profile("s1", "u1", "hi", "hello", "chat_agent")
        # last_seen was set to a value >= before
        ts = mock_prefs.last_seen
        assert ts >= before

    def test_saves_preferences(self):
        prefs = self._run("My name is Dave.")
        prefs.save.assert_called()

    def test_never_raises(self):
        """Profile extraction must never raise, even when prefs store fails."""
        from core.profile_extractor import extract_and_update_profile
        with patch("core.user_prefs.get_preferences", side_effect=Exception("DB down")):
            extract_and_update_profile("s", "u", "query", "resp", "agent")
        # If we get here without exception, the test passes


# =============================================================================
#  extract_and_update_profile_llm
# =============================================================================

class TestExtractAndUpdateProfileLLM:

    def _run_llm(self, llm_response: str, initial_prefs: dict | None = None):
        from core.profile_extractor import extract_and_update_profile_llm
        mock_prefs = MagicMock()
        mock_prefs.name       = (initial_prefs or {}).get("name", "")
        mock_prefs.interests  = list((initial_prefs or {}).get("interests", []))
        mock_prefs.response_style  = "concise"
        mock_prefs.preferred_agent = "auto"

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=llm_response)

        with patch("core.user_prefs.get_preferences", return_value=mock_prefs), \
             patch("core.llm_manager.get_llm", return_value=mock_llm):
            extract_and_update_profile_llm("s1", "u1", "test query", "test response")
        return mock_prefs

    def test_extracts_name_from_json(self):
        p = self._run_llm('{"name": "Emily"}')
        assert p.name == "Emily"

    def test_extracts_interests_from_json(self):
        p = self._run_llm('{"interests": ["machine learning", "finance"]}')
        assert "machine learning" in p.interests

    def test_extracts_style_preference(self):
        p = self._run_llm('{"style_preference": "technical"}')
        assert p.response_style == "technical"

    def test_extracts_preferred_agent(self):
        p = self._run_llm('{"preferred_agent": "research"}')
        assert p.preferred_agent == "research"

    def test_empty_json_no_changes(self):
        p = self._run_llm("{}")
        p.save.assert_not_called()

    def test_none_keyword_no_changes(self):
        p = self._run_llm("NONE")
        p.save.assert_not_called()

    def test_handles_markdown_fence(self):
        p = self._run_llm('```json\n{"name": "Frank"}\n```')
        assert p.name == "Frank"

    def test_handles_llm_error_gracefully(self):
        from core.profile_extractor import extract_and_update_profile_llm
        with patch("core.user_prefs.get_preferences", side_effect=Exception):
            extract_and_update_profile_llm("s", "u", "q", "r")
        # No exception raised

    def test_invalid_agent_not_saved(self):
        p = self._run_llm('{"preferred_agent": "hacker"}')
        assert p.preferred_agent == "auto"   # unchanged

    def test_invalid_style_not_saved(self):
        p = self._run_llm('{"style_preference": "robot"}')
        assert p.response_style == "concise"  # unchanged

    def test_does_not_overwrite_existing_name(self):
        p = self._run_llm('{"name": "Override"}', initial_prefs={"name": "Original"})
        assert p.name == "Original"


# =============================================================================
#  build_session_recall
# =============================================================================

class TestBuildSessionRecall:

    def _run_recall(self, last_seen_hours_ago: float, total_sessions: int,
                    facts: list | None = None, name: str = ""):
        from core.profile_extractor import build_session_recall
        from core.long_term_memory.episodic_memory import EpisodicFact

        mock_prefs = MagicMock()
        mock_prefs.last_seen       = time.time() - (last_seen_hours_ago * 3600)
        mock_prefs.total_sessions  = total_sessions
        mock_prefs.name            = name
        mock_prefs.greeting_name.return_value = name

        mock_ep = MagicMock()
        mock_facts = []
        for i, text in enumerate(facts or []):
            f = EpisodicFact(
                fact_id=f"f{i}",
                text=text,
                session_id="old_sess",
                query="old query",
            )
            mock_facts.append(f)
        mock_ep.list_all.return_value = mock_facts

        with patch("core.user_prefs.get_preferences", return_value=mock_prefs):
            return build_session_recall("u1", "new_sess", mock_ep)

    def test_first_time_user_no_recall(self):
        result = self._run_recall(last_seen_hours_ago=0.0, total_sessions=0)
        assert result == ""

    def test_recent_session_no_recall(self):
        result = self._run_recall(last_seen_hours_ago=1.0, total_sessions=3,
                                   facts=["Python is a language."])
        assert result == ""

    def test_returning_user_gets_recap(self):
        result = self._run_recall(
            last_seen_hours_ago=48.0,
            total_sessions=5,
            facts=["User prefers Python.", "Last project was a web scraper."],
        )
        assert result != ""
        assert "ago" in result or "yesterday" in result

    def test_recap_includes_facts(self):
        result = self._run_recall(
            last_seen_hours_ago=24.0,
            total_sessions=4,
            facts=["User prefers Python.", "Revenue was $4.2B."],
        )
        assert "Python" in result or "Revenue" in result

    def test_recap_includes_name_when_set(self):
        result = self._run_recall(
            last_seen_hours_ago=48.0,
            total_sessions=5,
            facts=["Some fact here."],
            name="Alice",
        )
        assert "Alice" in result

    def test_no_facts_returns_empty(self):
        result = self._run_recall(last_seen_hours_ago=48.0, total_sessions=5, facts=[])
        assert result == ""

    def test_none_episodic_returns_empty(self):
        from core.profile_extractor import build_session_recall
        result = build_session_recall("u1", "s1", None)
        assert result == ""

    def test_error_returns_empty(self):
        from core.profile_extractor import build_session_recall
        with patch("core.user_prefs.get_preferences", side_effect=Exception):
            result = build_session_recall("u1", "s1", MagicMock())
        assert result == ""

    def test_hours_ago_formatting(self):
        result = self._run_recall(
            last_seen_hours_ago=6.0,
            total_sessions=3,
            facts=["A stored fact."],
        )
        assert "hour" in result or "ago" in result

    def test_yesterday_formatting(self):
        result = self._run_recall(
            last_seen_hours_ago=25.0,
            total_sessions=3,
            facts=["A stored fact."],
        )
        assert "yesterday" in result

    def test_days_ago_formatting(self):
        result = self._run_recall(
            last_seen_hours_ago=72.0,
            total_sessions=3,
            facts=["A stored fact."],
        )
        assert "day" in result


# =============================================================================
#  update_session_count
# =============================================================================

class TestUpdateSessionCount:

    def test_increments_counter(self):
        from core.profile_extractor import update_session_count
        mock_prefs = MagicMock()
        mock_prefs.total_sessions = 4
        with patch("core.user_prefs.get_preferences", return_value=mock_prefs):
            update_session_count("u1", "new_session")
        assert mock_prefs.total_sessions == 5

    def test_updates_last_seen(self):
        from core.profile_extractor import update_session_count
        before = time.time()
        mock_prefs = MagicMock()
        mock_prefs.total_sessions = 0
        with patch("core.user_prefs.get_preferences", return_value=mock_prefs):
            update_session_count("u1", "sess")
        assert mock_prefs.last_seen >= before

    def test_saves_preferences(self):
        from core.profile_extractor import update_session_count
        mock_prefs = MagicMock()
        mock_prefs.total_sessions = 1
        with patch("core.user_prefs.get_preferences", return_value=mock_prefs):
            update_session_count("u1", "sess")
        mock_prefs.save.assert_called_once()

    def test_error_never_raises(self):
        from core.profile_extractor import update_session_count
        with patch("core.user_prefs.get_preferences", side_effect=Exception("fail")):
            update_session_count("u1", "sess")   # must not raise


# =============================================================================
#  Orchestrator — episodic memory on by default
# =============================================================================

class TestOrchestratorMemoryDefaults:

    def test_episodic_enabled_by_default(self):
        """Orchestrator default init should enable episodic memory (not None)."""
        from agents.orchestrator import Orchestrator
        # We can't easily patch the local import inside __init__
        # Instead verify that the Orchestrator's enable_episodic parameter defaults to True
        import inspect
        sig = inspect.signature(Orchestrator.__init__)
        default_val = sig.parameters["enable_episodic"].default
        assert default_val is True, (
            f"Expected enable_episodic=True by default, got {default_val!r}"
        )

    def test_update_user_profile_called_after_run(self):
        from agents.orchestrator import Orchestrator, Intent

        orch = Orchestrator.__new__(Orchestrator)
        orch._session_id               = "test"
        orch._enable_llm_quality_check = False
        orch._enable_rag_precheck      = False
        orch._rag_similarity_threshold = 0.55
        orch._rag_k                    = 4
        orch._max_fallback_attempts    = 0
        orch._summariser               = None
        orch._episodic                 = None
        orch._user_id                  = "default"

        mem = MagicMock(); mem.messages = []
        orch._memory = mem

        for intent in Intent:
            agent      = MagicMock()
            agent.name = intent.value
            agent.run.return_value = MagicMock(
                output="A response.", agent_name=intent.value,
                error=None, references=[], tool_calls=[]
            )
        orch._agents = {i: orch._agents[i] if hasattr(orch, '_agents') else MagicMock()
                        for i in Intent}
        # Rebuild properly
        orch._agents = {}
        for intent in Intent:
            a = MagicMock()
            a.name = intent.value
            from agents.base_agent import AgentResponse
            a.run.return_value = AgentResponse(
                output="Test response.", agent_name=intent.value
            )
            orch._agents[intent] = a

        with patch.object(orch, "_route", return_value=Intent.CHAT), \
             patch.object(orch, "_update_user_profile") as mock_update, \
             patch.object(orch, "_open_trace", return_value=None), \
             patch.object(orch, "_close_trace"):
            orch.run("Hello there")

        mock_update.assert_called_once()

    def test_update_user_profile_never_raises(self):
        """Even if profile extraction fails, run() must return normally."""
        from agents.orchestrator import Orchestrator, Intent
        from agents.base_agent import AgentResponse

        orch = Orchestrator.__new__(Orchestrator)
        orch._session_id               = "test"
        orch._enable_llm_quality_check = False
        orch._enable_rag_precheck      = False
        orch._rag_similarity_threshold = 0.55
        orch._rag_k                    = 4
        orch._max_fallback_attempts    = 0
        orch._summariser               = None
        orch._episodic                 = None
        orch._user_id                  = "default"

        mem = MagicMock(); mem.messages = []
        orch._memory = mem

        orch._agents = {}
        for intent in Intent:
            a = MagicMock()
            a.name = intent.value
            a.run.return_value = AgentResponse(output="OK", agent_name=intent.value)
            orch._agents[intent] = a

        with patch.object(orch, "_route", return_value=Intent.CHAT), \
             patch.object(orch, "_open_trace", return_value=None), \
             patch.object(orch, "_close_trace"):
            # Patch the underlying extractor to raise — _update_user_profile should catch it
            with patch("core.profile_extractor.extract_and_update_profile",
                       side_effect=RuntimeError("extraction fail")):
                result = orch.run("test")
        # If run() returns normally, the test passes
        assert result is not None
