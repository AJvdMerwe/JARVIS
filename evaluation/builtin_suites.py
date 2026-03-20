"""
evaluation/builtin_suites.py
─────────────────────────────
Pre-built EvalSuites covering all four agents.

Run with:
    python evaluation/run_evals.py --suite smoke
    python evaluation/run_evals.py --suite all
"""
from __future__ import annotations

from .eval_harness import (
    AgentIs, Contains, EvalCase, EvalSuite,
    HasReferences, LLMJudge, MinLength, MatchesRegex,
    NoError, NotContains, UsedTools,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Code Agent
# ─────────────────────────────────────────────────────────────────────────────

def code_suite() -> EvalSuite:
    suite = EvalSuite("code_agent")

    suite.add(EvalCase(
        query="Write a Python function that returns the nth Fibonacci number using memoisation.",
        description="Python fibonacci with memoisation",
        agent_hint="code",
        criteria=[
            AgentIs("code_agent"),
            Contains("def "),
            Contains("fibonacci"),
            Contains("memo") or Contains("cache") or Contains("lru"),
            NoError(),
            MinLength(100),
        ],
        tags=["smoke", "python"],
    ))

    suite.add(EvalCase(
        query="Write a TypeScript interface for a User object with id, name, email, and createdAt fields.",
        description="TypeScript interface generation",
        agent_hint="code",
        criteria=[
            AgentIs("code_agent"),
            Contains("interface"),
            Contains("id"),
            Contains("email"),
            NoError(),
        ],
        tags=["typescript"],
    ))

    suite.add(EvalCase(
        query="Review this code and find the bug: def divide(a, b): return a / b",
        description="Code review - division by zero bug",
        agent_hint="code",
        criteria=[
            AgentIs("code_agent"),
            Contains("zero") or Contains("ZeroDivision"),
            NoError(),
            MinLength(50),
        ],
        tags=["review"],
    ))

    suite.add(EvalCase(
        query="Write a Python class implementing a min-heap with push and pop operations.",
        description="Min-heap implementation",
        agent_hint="code",
        criteria=[
            AgentIs("code_agent"),
            Contains("class"),
            Contains("push") or Contains("heappush"),
            Contains("pop") or Contains("heappop"),
            NoError(),
            MinLength(200),
        ],
        tags=["data-structure"],
    ))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  News Agent
# ─────────────────────────────────────────────────────────────────────────────

def news_suite() -> EvalSuite:
    suite = EvalSuite("news_agent")

    suite.add(EvalCase(
        query="Give me a brief summary of today's top headlines.",
        description="Top headlines retrieval",
        agent_hint="news",
        criteria=[
            AgentIs("news_agent"),
            UsedTools(),
            NoError(),
            MinLength(100),
        ],
        tags=["smoke", "headlines"],
    ))

    suite.add(EvalCase(
        query="What's the latest news about artificial intelligence?",
        description="Topic news - AI",
        agent_hint="news",
        criteria=[
            AgentIs("news_agent"),
            UsedTools(),
            Contains("AI") or Contains("artificial intelligence"),
            NoError(),
        ],
        tags=["topic"],
    ))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  Search Agent
# ─────────────────────────────────────────────────────────────────────────────

def search_suite() -> EvalSuite:
    suite = EvalSuite("search_agent")

    suite.add(EvalCase(
        query="What is the boiling point of water in Celsius and Fahrenheit?",
        description="Factual question - boiling point",
        agent_hint="search",
        criteria=[
            AgentIs("search_agent"),
            Contains("100"),
            Contains("212") or Contains("fahrenheit"),
            NoError(),
        ],
        tags=["smoke", "factual"],
    ))

    suite.add(EvalCase(
        query="Calculate: sqrt(144) + 2 ** 8",
        description="Calculator - sqrt and power",
        agent_hint="search",
        criteria=[
            AgentIs("search_agent"),
            UsedTools("calculator"),
            Contains("268"),   # sqrt(144)=12, 2**8=256, total=268
            NoError(),
        ],
        tags=["calculator"],
    ))

    suite.add(EvalCase(
        query="Who wrote the novel '1984'?",
        description="Wikipedia lookup - author",
        agent_hint="search",
        criteria=[
            AgentIs("search_agent"),
            Contains("Orwell") or Contains("George"),
            NoError(),
        ],
        tags=["factual", "wikipedia"],
    ))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  Document Agent (works without pre-ingested docs — tests routing + tools)
# ─────────────────────────────────────────────────────────────────────────────

def document_suite() -> EvalSuite:
    suite = EvalSuite("document_agent")

    suite.add(EvalCase(
        query="List all documents in the knowledge base.",
        description="List documents tool invocation",
        agent_hint="document",
        criteria=[
            AgentIs("document_agent"),
            UsedTools("list_documents"),
            NoError(),
        ],
        tags=["smoke"],
    ))

    suite.add(EvalCase(
        query="Search the knowledge base for information about revenue.",
        description="Document search - revenue",
        agent_hint="document",
        criteria=[
            AgentIs("document_agent"),
            UsedTools("search_documents"),
            NoError(),
        ],
        tags=["search"],
    ))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  Routing suite — tests the orchestrator's auto-routing
# ─────────────────────────────────────────────────────────────────────────────

def routing_suite() -> EvalSuite:
    suite = EvalSuite("orchestrator_routing")

    routing_cases = [
        ("Write a bubble sort in Python",               "code_agent"),
        ("What are the latest AI headlines today?",     "news_agent"),
        ("What is quantum entanglement?",               "search_agent"),
        ("What does the uploaded PDF say about costs?", "document_agent"),
    ]

    for query, expected_agent in routing_cases:
        suite.add(EvalCase(
            query=query,
            description=f"Routes to {expected_agent}",
            criteria=[AgentIs(expected_agent), NoError()],
            tags=["routing", "smoke"],
        ))

    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke suite — one case per agent, fast to run
# ─────────────────────────────────────────────────────────────────────────────

def smoke_suite() -> EvalSuite:
    suite = EvalSuite("smoke")
    for child_suite in [code_suite(), news_suite(), search_suite(), document_suite(), routing_suite()]:
        smoke_cases = [c for c in child_suite.cases if "smoke" in c.tags]
        suite.cases.extend(smoke_cases)
    return suite


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────

SUITES: dict[str, callable] = {
    "smoke":    smoke_suite,
    "code":     code_suite,
    "news":     news_suite,
    "search":   search_suite,
    "document": document_suite,
    "routing":  routing_suite,
}


def get_all_suites() -> EvalSuite:
    combined = EvalSuite("all")
    for fn in SUITES.values():
        combined.cases.extend(fn().cases)
    return combined
