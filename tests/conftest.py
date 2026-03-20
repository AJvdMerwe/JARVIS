"""
tests/conftest.py
──────────────────
Shared pytest fixtures and configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path so imports work from the tests directory
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Silence noisy loggers during tests ────────────────────────────────────

import logging
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# ─── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """
    Prevent any test from accidentally calling a real LLM.
    Tests that need LLM behaviour should mock get_llm() themselves.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Mocked LLM response.")
    monkeypatch.setattr("core.llm_manager.get_llm", lambda: mock_llm)


@pytest.fixture()
def mock_embeddings():
    """Return a mock embeddings model that produces deterministic vectors."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [[0.1, 0.2, 0.3]] * len(texts)
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock
