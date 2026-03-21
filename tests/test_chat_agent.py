"""
tests/test_chat_agent.py
─────────────────────────
Tests for:
  • ChatAgent  — direct LLM chat, history injection, streaming, user prefs
  • Orchestrator routing — chat intent detection and dispatch
  • Integration — multi-turn conversation through the orchestrator
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from agents.base_agent import AgentResponse
from agents.orchestrator import Intent, _keyword_route, _llm_route


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_chat_agent(response_text: str = "Hello! I'm doing well, thanks."):
    """Build a ChatAgent with a mock LLM."""
    from agents.chat_agent import ChatAgent
    from core.memory import AssistantMemory

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=response_text)
    mock_llm.stream.return_value = iter([
        MagicMock(content=tok) for tok in response_text.split()
    ])

    agent = ChatAgent(llm=mock_llm, memory=AssistantMemory())
    return agent, mock_llm


# ─────────────────────────────────────────────────────────────────────────────
#  ChatAgent unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChatAgent:

    def test_name_and_description(self):
        from agents.chat_agent import ChatAgent
        agent = ChatAgent.__new__(ChatAgent)
        assert agent.name == "chat_agent"
        assert "conversational" in agent.description.lower() or "chat" in agent.description.lower()

    def test_get_tools_empty(self):
        from agents.chat_agent import ChatAgent
        agent = ChatAgent.__new__(ChatAgent)
        assert agent.get_tools() == []

    def test_run_returns_agent_response(self):
        agent, mock_llm = _make_chat_agent("The sky is blue because of Rayleigh scattering.")
        resp = agent.run("Why is the sky blue?")

        assert isinstance(resp, AgentResponse)
        assert resp.agent_name == "chat_agent"
        assert "Rayleigh" in resp.output
        assert resp.error is None

    def test_run_invokes_llm_with_messages(self):
        agent, mock_llm = _make_chat_agent("Sure, here's a haiku.")
        agent.run("Write a haiku about rain")

        # LLM should have been called with a list of messages
        mock_llm.invoke.assert_called_once()
        messages = mock_llm.invoke.call_args[0][0]
        assert isinstance(messages, list)
        assert len(messages) >= 2  # system + human

    def test_run_includes_system_message(self):
        from langchain_core.messages import SystemMessage
        agent, mock_llm = _make_chat_agent("Response")
        agent.run("Hello")

        messages = mock_llm.invoke.call_args[0][0]
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert len(system_msgs[0].content) > 20

    def test_run_includes_user_message(self):
        from langchain_core.messages import HumanMessage
        agent, mock_llm = _make_chat_agent("Response")
        agent.run("Tell me about recursion")

        messages = mock_llm.invoke.call_args[0][0]
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert any("recursion" in m.content for m in human_msgs)

    def test_run_injects_conversation_history(self):
        """Previous turns should appear between system and new human message."""
        from langchain_core.messages import HumanMessage, AIMessage
        agent, mock_llm = _make_chat_agent("Follow-up answer.")

        # Prime the memory with a past exchange
        agent._memory.add_user_message("What is Python?")
        agent._memory.add_ai_message("Python is a high-level programming language.")

        agent.run("What about its typing system?")

        messages = mock_llm.invoke.call_args[0][0]
        content_str = " ".join(str(m.content) for m in messages)
        assert "Python" in content_str
        assert "programming language" in content_str

    def test_run_handles_llm_exception(self):
        from agents.chat_agent import ChatAgent
        from core.memory import AssistantMemory

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM server down")
        agent = ChatAgent(llm=mock_llm, memory=AssistantMemory())

        resp = agent.run("Hello")
        assert resp.error == "LLM server down"
        assert "error" in resp.output.lower()

    def test_stream_yields_tokens(self):
        agent, mock_llm = _make_chat_agent("Hello from the stream.")
        tokens = list(agent.stream("Say hello"))

        assert len(tokens) > 0
        assert "".join(tokens).strip() != ""

    def test_stream_calls_llm_stream(self):
        agent, mock_llm = _make_chat_agent("Streamed answer.")
        list(agent.stream("question"))  # consume generator
        mock_llm.stream.assert_called_once()

    def test_custom_system_prompt(self):
        from agents.chat_agent import ChatAgent
        from langchain_core.messages import SystemMessage
        from core.memory import AssistantMemory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Ahoy!")
        agent = ChatAgent(
            llm=mock_llm,
            memory=AssistantMemory(),
            system_prompt="You are a pirate. Always speak like a pirate.",
        )
        agent.run("Hello")

        messages = mock_llm.invoke.call_args[0][0]
        system_content = next(m.content for m in messages if isinstance(m, SystemMessage))
        assert "pirate" in system_content.lower()

    def test_system_override_per_call(self):
        from langchain_core.messages import SystemMessage
        agent, mock_llm = _make_chat_agent("Override response.")
        agent.run("Hello", system_override="You are a legal advisor. Be precise.")

        messages = mock_llm.invoke.call_args[0][0]
        system_content = next(m.content for m in messages if isinstance(m, SystemMessage))
        assert "legal" in system_content.lower()

    def test_user_prefs_injected_into_system(self, tmp_path):
        """Style preferences from UserPreferences should appear in the system prompt."""
        from langchain_core.messages import SystemMessage
        from agents.chat_agent import ChatAgent
        from core.memory import AssistantMemory
        from core.user_prefs import UserPreferences

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Technical answer.")

        prefs = UserPreferences(user_id="techuser", response_style="technical")

        with patch("core.user_prefs.preferences._PREFS_DIR", tmp_path):
            prefs.save()
            with patch("core.user_prefs.preferences.get_preferences", return_value=prefs):
                agent = ChatAgent(llm=mock_llm, memory=AssistantMemory())
                agent.run("Explain TCP/IP", user_id="techuser")

        messages = mock_llm.invoke.call_args[0][0]
        system_content = " ".join(m.content for m in messages if isinstance(m, SystemMessage))
        assert "technical" in system_content.lower() or "precise" in system_content.lower()

    def test_build_system_without_prefs(self):
        from agents.chat_agent import ChatAgent, _DEFAULT_SYSTEM
        agent = ChatAgent.__new__(ChatAgent)
        agent._system_prompt = _DEFAULT_SYSTEM
        result = agent._build_system()
        assert result == _DEFAULT_SYSTEM

    def test_run_no_tool_calls_in_response(self):
        agent, _ = _make_chat_agent("Just a direct answer.")
        resp = agent.run("What is 2+2?")
        # ChatAgent never uses tools
        assert resp.tool_calls == []

    def test_run_no_references_in_response(self):
        agent, _ = _make_chat_agent("Pure LLM answer.")
        resp = agent.run("Tell me a story")
        assert resp.references == []


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator routing with chat
# ─────────────────────────────────────────────────────────────────────────────

class TestChatRouting:

    @pytest.mark.parametrize("query", [
        "hello there, how are you?",
        "can you help me with something?",
        "tell me a story about a dragon",
        "write a poem about autumn",
        "explain quantum mechanics simply",
        "what do you think about coffee?",
        "brainstorm startup ideas for me",
        "describe what you can do",
        "who are you?",
        "let's talk about philosophy",
    ])
    def test_conversational_queries_route_to_chat(self, query: str):
        result = _keyword_route(query)
        assert result == Intent.CHAT, (
            f"Expected CHAT for conversational query {query!r}, got {result}"
        )

    @pytest.mark.parametrize("query", [
        "write a python function to reverse a list",
        "debug this error: TypeError on line 42",
        "implement a binary search tree",
    ])
    def test_code_queries_not_routed_to_chat(self, query: str):
        result = _keyword_route(query)
        assert result == Intent.CODE

    @pytest.mark.parametrize("query", [
        "what are today's top headlines?",
        "latest breaking news",
        "give me a news briefing",
    ])
    def test_news_queries_not_routed_to_chat(self, query: str):
        result = _keyword_route(query)
        assert result == Intent.NEWS

    def test_llm_router_returns_chat_for_ambiguous(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="chat")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_route("let's have a philosophical discussion")
        assert result == Intent.CHAT

    def test_llm_router_fallback_is_chat(self):
        """Unknown response from LLM should fall back to CHAT."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="UNKNOWN_CATEGORY_XYZ")
        with patch("agents.orchestrator.get_llm", return_value=mock_llm):
            result = _llm_route("something completely ambiguous")
        assert result == Intent.CHAT


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator dispatches to ChatAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorChatDispatch:

    def _make_orchestrator(self):
        from agents.orchestrator import Orchestrator
        mock_response = AgentResponse(output="I'm doing great, thanks!", agent_name="chat_agent")
        mock_chat_agent = MagicMock()
        mock_chat_agent.run.return_value = mock_response

        with patch("agents.orchestrator.PersistentMemory"):
            orch = Orchestrator.__new__(Orchestrator)
            orch._session_id = "test-chat"
            orch._memory = MagicMock()
            orch._agents = {intent: MagicMock() for intent in Intent}
            orch._agents[Intent.CHAT] = mock_chat_agent
        return orch, mock_chat_agent

    def test_orchestrator_routes_greeting_to_chat(self):
        orch, mock_agent = self._make_orchestrator()
        mock_agent.run.return_value = AgentResponse(
            output="Hi! I'm doing well.", agent_name="chat_agent"
        )
        with patch.object(orch, "_route", return_value=Intent.CHAT):
            resp = orch.run("Hello! How are you?")

        mock_agent.run.assert_called_once_with("Hello! How are you?")
        assert resp.agent_name == "chat_agent"

    def test_chat_agent_registered_in_orchestrator(self):
        from agents.orchestrator import Orchestrator
        with patch("agents.orchestrator.PersistentMemory"), \
                patch("agents.orchestrator.ChatAgent") as mock_chat_cls, \
                patch("agents.orchestrator.CodeAgent"), \
                patch("agents.orchestrator.NewsAgent"), \
                patch("agents.orchestrator.SearchAgent"), \
                patch("agents.orchestrator.DocumentAgent"), \
                patch("agents.orchestrator.FinancialAgent"):
            orch = Orchestrator(session_id="test")
        mock_chat_cls.assert_called_once()
        assert Intent.CHAT in orch._agents

    def test_chat_intent_in_intent_enum(self):
        assert Intent.CHAT in Intent
        assert Intent.CHAT.value == "chat"

    def test_unknown_routes_to_chat_not_search(self):
        """UNKNOWN intent should now dispatch to chat, not search."""
        orch, mock_chat = self._make_orchestrator()
        mock_chat.run.return_value = AgentResponse(output="ok", agent_name="chat_agent")
        with patch.object(orch, "_route", return_value=Intent.CHAT):
            orch.run("some truly ambiguous input xyz123")
        mock_chat.run.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-turn conversation integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurnConversation:

    def test_history_grows_across_turns(self):
        from agents.chat_agent import ChatAgent
        from core.memory import AssistantMemory

        memory = AssistantMemory(k=20)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Turn response.")
        agent = ChatAgent(llm=mock_llm, memory=memory)

        # Simulate three conversation turns
        agent.run("Turn one question")
        memory.save_context("Turn one question", "Turn response.")
        agent.run("Turn two question")
        memory.save_context("Turn two question", "Turn response.")
        agent.run("Turn three question")

        messages = memory.messages
        assert len(messages) >= 4  # at least 2 saved turns = 4 messages

    def test_second_turn_includes_first_turn_context(self):
        """The second LLM call should include the first exchange in its messages."""
        from langchain_core.messages import HumanMessage, AIMessage
        from agents.chat_agent import ChatAgent
        from core.memory import AssistantMemory

        memory = AssistantMemory(k=20)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Second answer.")
        agent = ChatAgent(llm=mock_llm, memory=memory)

        # First turn — manually save to memory
        agent.run("What is Python?")
        memory.save_context("What is Python?", "Python is a programming language.")

        # Second turn — history should include first exchange
        agent.run("What about its type system?")

        second_call_messages = mock_llm.invoke.call_args[0][0]
        all_content = " ".join(str(m.content) for m in second_call_messages)
        assert "Python" in all_content

    def test_chat_agent_memory_cleared_on_clear(self):
        from agents.chat_agent import ChatAgent
        from core.memory import AssistantMemory

        memory = AssistantMemory(k=10)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="OK")
        agent = ChatAgent(llm=mock_llm, memory=memory)

        memory.save_context("q1", "a1")
        memory.save_context("q2", "a2")
        assert len(memory.messages) > 0

        agent.clear_memory()
        assert len(memory.messages) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation suite integration
# ─────────────────────────────────────────────────────────────────────────────

class TestChatEvalSuite:

    def test_chat_suite_exists_in_registry(self):
        from evaluation.builtin_suites import SUITES
        assert "chat" in SUITES

    def test_chat_suite_has_cases(self):
        from evaluation.builtin_suites import chat_suite
        suite = chat_suite()
        assert len(suite) > 0

    def test_smoke_suite_includes_chat_cases(self):
        from evaluation.builtin_suites import smoke_suite
        suite = smoke_suite()
        chat_cases = [c for c in suite.cases if c.agent_hint == "chat"]
        assert len(chat_cases) > 0

    def test_chat_suite_cases_have_correct_agent_hint(self):
        from evaluation.builtin_suites import chat_suite
        suite = chat_suite()
        for case in suite.cases:
            assert case.agent_hint == "chat"

    def test_all_suites_includes_chat(self):
        from evaluation.builtin_suites import get_all_suites
        suite = get_all_suites()
        names = {c.agent_hint for c in suite.cases}
        assert "chat" in names