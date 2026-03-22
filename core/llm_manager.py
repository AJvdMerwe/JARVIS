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
    Return a cached LangChain embeddings model for the configured backend.

    Backend selection is driven by ``settings.embedding_backend``:

    ``huggingface`` (default)
        Loads a sentence-transformers model in-process via
        ``langchain_community.embeddings.HuggingFaceEmbeddings``.
        Runs on the device specified by ``settings.embedding_device``
        (defaults to ``cpu`` to avoid competing with Docling / the LLM for
        VRAM on low-memory GPUs).

        Environment variables:
            EMBEDDING_BACKEND=huggingface
            EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
            EMBEDDING_DEVICE=cpu          # cpu | cuda | mps | auto

    ``ollama``
        Sends embedding requests to the running Ollama server.
        Zero application VRAM — the model runs inside the Ollama process.
        Ideal when the GPU is already under pressure from Docling or the
        LLM backend.

        Environment variables:
            EMBEDDING_BACKEND=ollama
            OLLAMA_EMBEDDING_MODEL=nomic-embed-text   # or nomic-embed-text-v2-moe
            OLLAMA_BASE_URL=http://localhost:11434

    Returns
    -------
    LangChain Embeddings
        An object implementing ``embed_documents(texts)`` and
        ``embed_query(text)``.
    """
    backend = settings.embedding_backend.lower()

    if backend == "ollama":
        return _build_ollama_embeddings()
    return _build_huggingface_embeddings()


def _resolve_hf_device(device_setting: str) -> str:
    """
    Resolve ``"auto"`` to the best available device, or return the
    explicit setting unchanged.

    Priority order for ``auto``: MPS (Apple) → CUDA → CPU.
    """
    if device_setting != "auto":
        return device_setting
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_huggingface_embeddings():
    """
    Build an in-process HuggingFace sentence-transformers embeddings model.

    The device is resolved from ``settings.embedding_device`` so it can be
    set to ``cpu`` (safe, default) or ``cuda`` / ``mps`` when you have the
    headroom.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "langchain-community is not installed. "
            "Run: pip install langchain-community sentence-transformers"
        ) from exc

    device = _resolve_hf_device(settings.embedding_device)
    logger.info(
        "Embeddings backend → HuggingFace | model=%s | device=%s",
        settings.embedding_model, device,
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_ollama_embeddings():
    """
    Build an Ollama-backed embeddings model.

    Requests are forwarded to the Ollama HTTP server, keeping the embedding
    model completely out of the application process.  This means zero
    application-side VRAM, making it the right choice when the GPU is already
    saturated by Docling layout models or the LLM.

    Supported Ollama embedding models (pull with ``ollama pull <model>``):
        nomic-embed-text          — 137 M params, 768-dim, fast
        nomic-embed-text-v2-moe   — MoE architecture, better quality
        mxbai-embed-large         — 335 M params, 1024-dim, highest quality
        all-minilm                — 23 M params, tiny and very fast
    """
    # Prefer langchain_ollama (newer, dedicated package) over community shim
    try:
        from langchain_ollama import OllamaEmbeddings  # type: ignore[import]
        logger.info(
            "Embeddings backend → Ollama (langchain_ollama) | model=%s | url=%s",
            settings.ollama_embedding_model,
            settings.ollama_base_url,
        )
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
    except ImportError:
        pass

    # Fallback to langchain_community shim
    try:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore[import]
        logger.info(
            "Embeddings backend → Ollama (langchain_community) | model=%s | url=%s",
            settings.ollama_embedding_model,
            settings.ollama_base_url,
        )
        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )
    except ImportError as exc:
        raise ImportError(
            "Neither langchain-ollama nor langchain-community is installed. "
            "Run: pip install langchain-ollama"
        ) from exc


def clear_embeddings_cache() -> None:
    """
    Evict the cached embeddings instance so the next call to
    ``get_embeddings()`` re-creates it from the current settings.

    Useful when switching backends at runtime (e.g. in tests or an admin
    endpoint that reloads configuration).
    """
    get_embeddings.cache_clear()
    logger.info("Embeddings cache cleared — next call will reinitialise.")
