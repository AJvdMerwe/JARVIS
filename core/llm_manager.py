"""
core/llm_manager.py
────────────────────
Factory that returns a LangChain ChatModel backed by:
  * Ollama  - easy local inference, no extra server needed.
  * vLLM    - OpenAI-compatible server, higher GPU throughput.
  * SGLang  - high-performance inference using the Ollama-compatible API
              (RadixAttention, chunked prefill, speculative decoding).

All three are exposed via the same LangChain BaseChatModel interface so
agents never need to know which backend is active.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
#  Backend connectivity probe
# =============================================================================

def probe_backend(backend: str, url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """
    Check that the inference server at ``url`` is reachable and responding.

    Returns (ok: bool, message: str).
    Used at startup to fail fast with a clear error rather than silently
    falling back to a different backend.

    Ollama / SGLang probe : GET <url>/api/tags
    vLLM probe            : GET <url>/models
    """
    import urllib.request
    import urllib.error

    if backend in ("ollama", "sglang"):
        probe_url = url.rstrip("/") + "/api/tags"
    else:
        probe_url = url.rstrip("/") + "/models"

    try:
        req = urllib.request.Request(probe_url, headers={"User-Agent": "jarvis-probe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, f"{backend} server reachable at {url}"
            return False, f"{backend} server returned HTTP {resp.status}"
    except OSError as exc:
        return False, f"Cannot reach {backend} server at {url} — {exc}"


def require_backend(raise_on_fail: bool = True) -> bool:
    """
    Probe the configured LLM backend and raise ``RuntimeError`` if it is
    unreachable (when ``raise_on_fail=True``).

    This is called at application startup so misconfiguration is caught
    immediately instead of silently falling back to a different backend.

    Set ``LLM_BACKEND_SKIP_PROBE=true`` in the environment to disable
    (useful in unit tests that mock the LLM entirely).
    """
    import os
    if os.getenv("LLM_BACKEND_SKIP_PROBE", "").lower() in ("1", "true", "yes"):
        return True

    backend = settings.llm_backend
    if backend == "ollama":
        url = settings.ollama_base_url
    elif backend == "vllm":
        url = settings.vllm_base_url
    elif backend == "sglang":
        url = settings.sglang_base_url
    else:
        return True  # unknown backend — let it fail at call time

    ok, msg = probe_backend(backend, url)
    if ok:
        logger.info("Backend probe OK: %s", msg)
        return True

    error_msg = (
        f"\n\n{'='*60}\n"
        f"  LLM BACKEND NOT REACHABLE\n"
        f"  Backend  : {backend}\n"
        f"  URL      : {url}\n"
        f"  Error    : {msg}\n"
        f"{'='*60}\n"
    )

    if backend == "sglang":
        error_msg += (
            f"  Start SGLang with:\n"
            f"    bash start.sh sglang   (managed mode)\n"
            f"  or manually:\n"
            f"    python -m sglang.launch_server \\\n"
            f"        --model-path {settings.sglang_model} \\\n"
            f"        --port {settings.sglang_base_url.split(':')[-1].split('/')[0]} \\\n"
            f"        --host 0.0.0.0 --trust-remote-code\n"
        )
    elif backend == "ollama":
        error_msg += "  Start Ollama with:  ollama serve\n"
    elif backend == "vllm":
        error_msg += "  Start vLLM with:  bash scripts/start_vllm.sh\n"
    error_msg += f"{'='*60}\n"

    if raise_on_fail:
        raise RuntimeError(error_msg)
    logger.error(error_msg)
    return False


# =============================================================================
#  Backend factories
# =============================================================================

def _build_ollama_llm() -> BaseChatModel:
    """Instantiate the Ollama-backed chat model."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError("langchain-ollama is not installed. Run: pip install langchain-ollama") from exc
    logger.info("LLM backend -> Ollama | model=%s | url=%s", settings.ollama_model, settings.ollama_base_url)
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
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai") from exc
    logger.info("LLM backend -> vLLM | model=%s | url=%s", settings.vllm_model, settings.vllm_base_url)
    return ChatOpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        model=settings.vllm_model,
        temperature=0.1,
        max_tokens=2048,
    )


def _build_sglang_llm() -> BaseChatModel:
    """
    Instantiate the SGLang-backed chat model via the Ollama-compatible API.

    SGLang exposes the same wire format as Ollama at /api/chat, /api/generate
    and /api/embeddings.  We use ChatOllama (langchain-ollama) pointed at the
    SGLang server - no OpenAI key required, no extra dependencies.

    Configuration (.env):
        LLM_BACKEND=sglang
        SGLANG_BASE_URL=http://localhost:11435
        SGLANG_MODEL=meta-llama/Llama-3.1-8B-Instruct
        SGLANG_NUM_PREDICT=2048

    Start the server:
        pip install sglang[all]
        python -m sglang.launch_server \\
            --model-path meta-llama/Llama-3.1-8B-Instruct \\
            --port 11435 --host 0.0.0.0 --trust-remote-code --dtype float16
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError("langchain-ollama is not installed. Run: pip install langchain-ollama") from exc
    logger.info(
        "LLM backend -> SGLang (Ollama-compat API) | model=%s | url=%s | speculative=%s",
        settings.sglang_model, settings.sglang_base_url, settings.sglang_enable_speculative,
    )
    return ChatOllama(
        base_url=settings.sglang_base_url,
        model=settings.sglang_model,
        temperature=0.1,
        num_predict=settings.sglang_num_predict,
    )


# =============================================================================
#  Public API
# =============================================================================

@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a cached LangChain chat model for the configured backend.
    Backend is LLM_BACKEND: ollama | vllm | sglang.

    Probes the backend server on first call and raises RuntimeError if it
    is unreachable — no silent fallback to a different backend ever occurs.
    """
    require_backend(raise_on_fail=True)
    backend = settings.llm_backend
    if backend == "vllm":
        return _build_vllm_llm()
    if backend == "sglang":
        return _build_sglang_llm()
    return _build_ollama_llm()


@lru_cache(maxsize=1)
def get_reasoning_llm() -> BaseChatModel:
    """
    Return a cached Ollama-backed LLM configured for the reasoning/thinking model.

    Always uses Ollama regardless of LLM_BACKEND - the reasoning model is
    typically a specialist local model (deepseek-r1, qwen3) not available on
    vLLM or SGLang servers.

    Configured via:
        OLLAMA_REASONING_MODEL=deepseek-r1:7b
        OLLAMA_BASE_URL=http://localhost:11434
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError("langchain-ollama is not installed. Run: pip install langchain-ollama") from exc
    logger.info(
        "Reasoning LLM -> Ollama | model=%s | url=%s",
        settings.ollama_reasoning_model, settings.ollama_base_url,
    )
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_reasoning_model,
        temperature=0.1,
        num_predict=4096,
        num_ctx=8192,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Return a cached LangChain embeddings model.
    Backend: EMBEDDING_BACKEND = huggingface | ollama.
    """
    backend = settings.embedding_backend.lower()
    if backend == "ollama":
        return _build_ollama_embeddings()
    return _build_huggingface_embeddings()


def get_backend_info() -> dict:
    """Return a summary of the active LLM backend configuration."""
    backend = settings.llm_backend
    info: dict = {"backend": backend}
    if backend == "ollama":
        info.update({"url": settings.ollama_base_url, "model": settings.ollama_model, "api": "ollama"})
    elif backend == "vllm":
        info.update({"url": settings.vllm_base_url, "model": settings.vllm_model, "api": "openai-compatible"})
    elif backend == "sglang":
        info.update({
            "url":         settings.sglang_base_url,
            "model":       settings.sglang_model,
            "speculative": settings.sglang_enable_speculative,
            "api":         "ollama-compatible",
        })
    return info


def clear_llm_cache() -> None:
    """Evict all cached LLM instances."""
    for fn in (get_llm, get_reasoning_llm):
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    logger.info("LLM cache cleared.")


def clear_embeddings_cache() -> None:
    """Evict the cached embeddings instance."""
    get_embeddings.cache_clear()
    logger.info("Embeddings cache cleared.")


# =============================================================================
#  Embedding helpers
# =============================================================================

def _resolve_hf_device(device_setting: str) -> str:
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
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-community is not installed. Run: pip install langchain-community sentence-transformers"
        ) from exc
    device = _resolve_hf_device(settings.embedding_device)
    logger.info("Embeddings backend -> HuggingFace | model=%s | device=%s", settings.embedding_model, device)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_ollama_embeddings():
    """Ollama-backed embeddings - zero application VRAM."""
    try:
        from langchain_ollama import OllamaEmbeddings
        logger.info(
            "Embeddings backend -> Ollama (langchain_ollama) | model=%s | url=%s",
            settings.ollama_embedding_model, settings.ollama_base_url,
        )
        return OllamaEmbeddings(base_url=settings.ollama_base_url, model=settings.ollama_embedding_model)
    except ImportError:
        pass
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        logger.info(
            "Embeddings backend -> Ollama (langchain_community) | model=%s | url=%s",
            settings.ollama_embedding_model, settings.ollama_base_url,
        )
        return OllamaEmbeddings(base_url=settings.ollama_base_url, model=settings.ollama_embedding_model)
    except ImportError as exc:
        raise ImportError(
            "Neither langchain-ollama nor langchain-community is installed. Run: pip install langchain-ollama"
        ) from exc
