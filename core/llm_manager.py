"""
core/llm_manager.py
────────────────────
Factory that returns a LangChain ChatModel backed by either:
  • Ollama  – easy local inference, no extra server needed.
  • vLLM   – OpenAI-compatible server, higher throughput for multi-user use.

Both are exposed via the same LangChain interface so agents never need to
know which backend is active.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Backend factories
# ─────────────────────────────────────────────────────────────────────────────

def _build_ollama_llm() -> BaseChatModel:
    """Instantiate the Ollama-backed chat model."""
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    logger.info(
        "LLM backend → Ollama | model=%s | url=%s",
        settings.ollama_model,
        settings.ollama_base_url,
    )
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.1,
        num_predict=2048,
    )


def _build_vllm_llm() -> BaseChatModel:
    """
    Instantiate the vLLM-backed chat model.
    vLLM exposes an OpenAI-compatible REST API; we use langchain-openai with a
    custom base_url pointing at the vLLM server.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    logger.info(
        "LLM backend → vLLM  | model=%s | url=%s",
        settings.vllm_model,
        settings.vllm_base_url,
    )
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=0.1,
        max_tokens=2048,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a cached LangChain chat model for the configured backend.
    The first call initialises the model; subsequent calls return the
    already-created instance.
    """
    backend = settings.llm_backend
    if backend == "vllm":
        return _build_vllm_llm()
    return _build_ollama_llm()


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached embeddings model.
    Uses sentence-transformers locally (no external API calls).
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import]

    logger.info("Embeddings model → %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
