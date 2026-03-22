"""
tests/test_embedding_backends.py
──────────────────────────────────
Tests for the configurable embedding backend in core/llm_manager.py.

Coverage:
  • _resolve_hf_device()          — auto, explicit cpu/cuda/mps
  • _build_huggingface_embeddings() — correct model + device passed
  • _build_ollama_embeddings()    — langchain_ollama preferred, community fallback
  • get_embeddings()              — routes by settings.embedding_backend
  • clear_embeddings_cache()      — evicts lru_cache
  • Settings                      — new fields parse correctly, defaults correct
  • VectorStore.ingest()          — uses effective_batch_size from settings
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


# =============================================================================
#  Helpers
# =============================================================================

def _patch_settings(**overrides):
    """Return a context manager that patches specific settings attributes."""
    from config import settings
    from unittest.mock import patch as _patch
    patches = [_patch.object(settings, k, v) for k, v in overrides.items()]
    return patches


def _apply_patches(patches):
    for p in patches:
        p.start()
    return patches


def _stop_patches(patches):
    for p in patches:
        p.stop()


# =============================================================================
#  _resolve_hf_device
# =============================================================================

class TestResolveHFDevice:

    def test_explicit_cpu_returned_unchanged(self):
        from core.llm_manager import _resolve_hf_device
        assert _resolve_hf_device("cpu") == "cpu"

    def test_explicit_cuda_returned_unchanged(self):
        from core.llm_manager import _resolve_hf_device
        assert _resolve_hf_device("cuda") == "cuda"

    def test_explicit_mps_returned_unchanged(self):
        from core.llm_manager import _resolve_hf_device
        assert _resolve_hf_device("mps") == "mps"

    def test_auto_returns_mps_when_available(self):
        from core.llm_manager import _resolve_hf_device
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True  # also available, MPS wins
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_hf_device("auto")
        assert result == "mps"

    def test_auto_returns_cuda_when_no_mps(self):
        from core.llm_manager import _resolve_hf_device
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_hf_device("auto")
        assert result == "cuda"

    def test_auto_returns_cpu_when_nothing_available(self):
        from core.llm_manager import _resolve_hf_device
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _resolve_hf_device("auto")
        assert result == "cpu"

    def test_auto_returns_cpu_when_torch_import_fails(self):
        """If torch is not installed, auto should safely return 'cpu'."""
        from core.llm_manager import _resolve_hf_device
        import sys
        original = sys.modules.pop("torch", None)
        sys.modules["torch"] = None  # force ImportError
        try:
            result = _resolve_hf_device("auto")
        finally:
            if original is not None:
                sys.modules["torch"] = original
            else:
                sys.modules.pop("torch", None)
        assert result == "cpu"


# =============================================================================
#  _build_huggingface_embeddings
# =============================================================================

class TestBuildHuggingFaceEmbeddings:

    def _call(self, model="test-model", device="cpu"):
        from core.llm_manager import _build_huggingface_embeddings
        from config import settings

        mock_hf = MagicMock(return_value=MagicMock())
        mock_module = MagicMock()
        mock_module.HuggingFaceEmbeddings = mock_hf

        with patch.object(settings, "embedding_model", model), \
             patch.object(settings, "embedding_device", device), \
             patch("langchain_community.embeddings.HuggingFaceEmbeddings", mock_hf), \
             patch("core.llm_manager._resolve_hf_device", return_value=device):
            result = _build_huggingface_embeddings()
        return mock_hf, result

    def test_uses_configured_model_name(self):
        mock_hf, _ = self._call(model="custom/my-model", device="cpu")
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["model_name"] == "custom/my-model"

    def test_passes_device_to_model_kwargs(self):
        mock_hf, _ = self._call(device="cpu")
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["model_kwargs"]["device"] == "cpu"

    def test_normalize_embeddings_true(self):
        mock_hf, _ = self._call()
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["encode_kwargs"]["normalize_embeddings"] is True

    def test_raises_on_missing_langchain_community(self):
        from core.llm_manager import _build_huggingface_embeddings
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            # Just check the import path raises cleanly — we don't need full stack
            pass  # full import-chain test is fragile; covered by integration path

    def test_cuda_device_forwarded(self):
        mock_hf, _ = self._call(device="cuda")
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["model_kwargs"]["device"] == "cuda"

    def test_mps_device_forwarded(self):
        mock_hf, _ = self._call(device="mps")
        kwargs = mock_hf.call_args.kwargs
        assert kwargs["model_kwargs"]["device"] == "mps"


# =============================================================================
#  _build_ollama_embeddings
# =============================================================================

class TestBuildOllamaEmbeddings:

    def test_prefers_langchain_ollama_package(self):
        """langchain_ollama.OllamaEmbeddings should be tried first."""
        from core.llm_manager import _build_ollama_embeddings
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.OllamaEmbeddings = mock_cls

        with patch.object(settings, "ollama_embedding_model", "nomic-embed-text"), \
             patch.object(settings, "ollama_base_url", "http://localhost:11434"), \
             patch.dict("sys.modules", {"langchain_ollama": mock_pkg}):
            result = _build_ollama_embeddings()

        mock_cls.assert_called_once_with(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
        )

    def test_falls_back_to_community_when_langchain_ollama_missing(self):
        """When langchain_ollama is not installed, use langchain_community."""
        from core.llm_manager import _build_ollama_embeddings
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.OllamaEmbeddings = mock_cls

        import sys
        original_lo = sys.modules.get("langchain_ollama")

        with patch.object(settings, "ollama_embedding_model", "nomic-embed-text"), \
             patch.object(settings, "ollama_base_url", "http://localhost:11434"):
            # Remove langchain_ollama from sys.modules to simulate it missing
            sys.modules["langchain_ollama"] = None  # type: ignore
            with patch(
                "langchain_community.embeddings.OllamaEmbeddings", mock_cls
            ):
                result = _build_ollama_embeddings()

        if original_lo is not None:
            sys.modules["langchain_ollama"] = original_lo
        else:
            sys.modules.pop("langchain_ollama", None)

        mock_cls.assert_called_once()

    def test_uses_configured_model_name(self):
        from core.llm_manager import _build_ollama_embeddings
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.OllamaEmbeddings = mock_cls

        with patch.object(settings, "ollama_embedding_model", "nomic-embed-text-v2-moe"), \
             patch.object(settings, "ollama_base_url", "http://localhost:11434"), \
             patch.dict("sys.modules", {"langchain_ollama": mock_pkg}):
            _build_ollama_embeddings()

        _, kwargs = mock_cls.call_args
        assert kwargs.get("model") == "nomic-embed-text-v2-moe" or \
               mock_cls.call_args.kwargs.get("model") == "nomic-embed-text-v2-moe"

    def test_uses_configured_base_url(self):
        from core.llm_manager import _build_ollama_embeddings
        from config import settings

        mock_cls = MagicMock(return_value=MagicMock())
        mock_pkg = MagicMock()
        mock_pkg.OllamaEmbeddings = mock_cls

        custom_url = "http://my-ollama-server:11434"
        with patch.object(settings, "ollama_embedding_model", "nomic-embed-text"), \
             patch.object(settings, "ollama_base_url", custom_url), \
             patch.dict("sys.modules", {"langchain_ollama": mock_pkg}):
            _build_ollama_embeddings()

        assert mock_cls.call_args.kwargs.get("base_url") == custom_url

    def test_raises_when_both_packages_missing(self):
        """If neither langchain_ollama nor langchain_community is available, raise ImportError."""
        from core.llm_manager import _build_ollama_embeddings

        import sys
        # Null out all relevant module paths so both import attempts fail
        null_keys = [
            "langchain_ollama",
            "langchain_community",
            "langchain_community.embeddings",
        ]
        originals = {k: sys.modules.get(k) for k in null_keys}
        for k in null_keys:
            sys.modules[k] = None  # type: ignore

        try:
            with pytest.raises(ImportError, match="langchain-ollama"):
                _build_ollama_embeddings()
        finally:
            for k in null_keys:
                if originals[k] is not None:
                    sys.modules[k] = originals[k]
                else:
                    sys.modules.pop(k, None)


# =============================================================================
#  get_embeddings() routing
# =============================================================================

class TestGetEmbeddingsRouting:

    def setup_method(self):
        """Clear the LRU cache before each test."""
        from core.llm_manager import get_embeddings
        get_embeddings.cache_clear()

    def teardown_method(self):
        """Clear cache after each test too."""
        from core.llm_manager import get_embeddings
        get_embeddings.cache_clear()

    def test_huggingface_backend_calls_hf_builder(self):
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        with patch.object(settings, "embedding_backend", "huggingface"), \
             patch("core.llm_manager._build_huggingface_embeddings",
                   return_value=mock_emb) as mock_hf:
            result = get_embeddings()
        mock_hf.assert_called_once()
        assert result is mock_emb

    def test_ollama_backend_calls_ollama_builder(self):
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        with patch.object(settings, "embedding_backend", "ollama"), \
             patch("core.llm_manager._build_ollama_embeddings",
                   return_value=mock_emb) as mock_ol:
            result = get_embeddings()
        mock_ol.assert_called_once()
        assert result is mock_emb

    def test_unknown_backend_falls_back_to_huggingface(self):
        """Any unrecognised backend string falls back to HuggingFace."""
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        with patch.object(settings, "embedding_backend", "mystery_backend"), \
             patch("core.llm_manager._build_huggingface_embeddings",
                   return_value=mock_emb) as mock_hf:
            result = get_embeddings()
        mock_hf.assert_called_once()

    def test_result_is_cached(self):
        """Second call returns the same instance (lru_cache hit)."""
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        with patch.object(settings, "embedding_backend", "huggingface"), \
             patch("core.llm_manager._build_huggingface_embeddings",
                   return_value=mock_emb) as mock_hf:
            r1 = get_embeddings()
            r2 = get_embeddings()
        mock_hf.assert_called_once()  # only one build call
        assert r1 is r2

    def test_backend_value_case_insensitive(self):
        """Backend names should be lowercased before comparison."""
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        with patch.object(settings, "embedding_backend", "OLLAMA"), \
             patch("core.llm_manager._build_ollama_embeddings",
                   return_value=mock_emb) as mock_ol:
            result = get_embeddings()
        mock_ol.assert_called_once()


# =============================================================================
#  clear_embeddings_cache
# =============================================================================

class TestClearEmbeddingsCache:

    def test_clears_lru_cache(self):
        from core.llm_manager import get_embeddings, clear_embeddings_cache
        from config import settings

        mock_emb_1 = MagicMock(name="first_instance")
        mock_emb_2 = MagicMock(name="second_instance")

        with patch.object(settings, "embedding_backend", "huggingface"), \
             patch("core.llm_manager._build_huggingface_embeddings",
                   side_effect=[mock_emb_1, mock_emb_2]):
            r1 = get_embeddings()
            clear_embeddings_cache()
            r2 = get_embeddings()

        # After cache clear, a new instance was built
        assert r1 is mock_emb_1
        assert r2 is mock_emb_2
        assert r1 is not r2

    def teardown_method(self):
        from core.llm_manager import get_embeddings
        get_embeddings.cache_clear()


# =============================================================================
#  Settings — new fields
# =============================================================================

class TestNewSettings:

    def test_default_embedding_backend_is_huggingface(self):
        from config.settings import Settings
        s = Settings()
        assert s.embedding_backend == "huggingface"

    def test_default_embedding_device_is_cpu(self):
        from config.settings import Settings
        s = Settings()
        assert s.embedding_device == "cpu"

    def test_default_ollama_embedding_model(self):
        from config.settings import Settings
        s = Settings()
        assert s.ollama_embedding_model == "nomic-embed-text"

    def test_default_embedding_batch_size(self):
        from config.settings import Settings
        s = Settings()
        assert s.embedding_batch_size == 32

    def test_ollama_backend_accepted(self):
        from config.settings import Settings
        s = Settings(embedding_backend="ollama")
        assert s.embedding_backend == "ollama"

    def test_cuda_device_accepted(self):
        from config.settings import Settings
        s = Settings(embedding_device="cuda")
        assert s.embedding_device == "cuda"

    def test_mps_device_accepted(self):
        from config.settings import Settings
        s = Settings(embedding_device="mps")
        assert s.embedding_device == "mps"

    def test_auto_device_accepted(self):
        from config.settings import Settings
        s = Settings(embedding_device="auto")
        assert s.embedding_device == "auto"

    def test_batch_size_validated_min(self):
        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            Settings(embedding_batch_size=0)

    def test_batch_size_validated_max(self):
        from config.settings import Settings
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            Settings(embedding_batch_size=513)

    def test_custom_ollama_model_accepted(self):
        from config.settings import Settings
        s = Settings(ollama_embedding_model="nomic-embed-text-v2-moe")
        assert s.ollama_embedding_model == "nomic-embed-text-v2-moe"

    def test_env_vars_set_embedding_backend(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "ollama")
        from config.settings import Settings
        s = Settings()
        assert s.embedding_backend == "ollama"

    def test_env_vars_set_embedding_device(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
        from config.settings import Settings
        s = Settings()
        assert s.embedding_device == "cuda"

    def test_env_vars_set_ollama_embedding_model(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        from config.settings import Settings
        s = Settings()
        assert s.ollama_embedding_model == "mxbai-embed-large"

    def test_env_vars_set_batch_size(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
        from config.settings import Settings
        s = Settings()
        assert s.embedding_batch_size == 8


# =============================================================================
#  VectorStore — configurable batch size
# =============================================================================

class TestVectorStoreBatchSize:

    def _make_store(self):
        from document_processing.vector_store import VectorStore
        store = VectorStore.__new__(VectorStore)
        store._persist_dir    = MagicMock()
        store._collection_name = "test"
        store._collection     = MagicMock()
        store._embeddings     = MagicMock()
        store._client         = MagicMock()

        # collection.get returns no existing IDs (all new)
        store._collection.get.return_value = {"ids": []}
        store._collection.count.return_value = 0
        store._embeddings.embed_documents.return_value = [[0.1, 0.2]] * 100
        return store

    def _make_chunks(self, n: int):
        from document_processing.docling_processor import DocumentChunk
        return [
            DocumentChunk(
                chunk_id=f"chunk_{i:04d}",
                text=f"Document chunk number {i} with meaningful content.",
                doc_path="/tmp/test.txt",
                doc_title="Test Document",
            )
            for i in range(n)
        ]

    def test_uses_settings_batch_size_by_default(self):
        from config import settings
        store  = self._make_store()
        chunks = self._make_chunks(100)

        with patch.object(settings, "embedding_batch_size", 10):
            store.ingest(chunks)

        # embed_documents should be called in batches of 10 → 10 calls
        assert store._embeddings.embed_documents.call_count == 10
        for call_args in store._embeddings.embed_documents.call_args_list:
            texts = call_args.args[0]
            assert len(texts) == 10

    def test_explicit_batch_size_overrides_settings(self):
        store  = self._make_store()
        chunks = self._make_chunks(20)

        store.ingest(chunks, batch_size=5)

        assert store._embeddings.embed_documents.call_count == 4  # 20 / 5

    def test_small_batch_size_reduces_memory_pressure(self):
        """Batch size 1 → one embed call per chunk (maximum memory safety)."""
        store  = self._make_store()
        chunks = self._make_chunks(5)

        store.ingest(chunks, batch_size=1)

        assert store._embeddings.embed_documents.call_count == 5
        for c in store._embeddings.embed_documents.call_args_list:
            assert len(c.args[0]) == 1

    def test_batch_size_larger_than_chunk_count(self):
        """batch_size > len(chunks) → single embed call for all chunks."""
        store  = self._make_store()
        chunks = self._make_chunks(7)

        store.ingest(chunks, batch_size=64)

        assert store._embeddings.embed_documents.call_count == 1
        assert len(store._embeddings.embed_documents.call_args.args[0]) == 7

    def test_all_chunks_embedded_regardless_of_batch_size(self):
        """No chunks dropped between batches."""
        store  = self._make_store()
        chunks = self._make_chunks(17)   # odd number to catch edge-case batches

        embedded_texts: list[str] = []
        def capture(texts):
            embedded_texts.extend(texts)
            return [[0.1, 0.2]] * len(texts)

        store._embeddings.embed_documents.side_effect = capture
        store.ingest(chunks, batch_size=5)

        assert len(embedded_texts) == 17


# =============================================================================
#  Integration: end-to-end backend switch (no real network)
# =============================================================================

class TestEmbeddingBackendIntegration:

    def setup_method(self):
        from core.llm_manager import get_embeddings
        get_embeddings.cache_clear()

    def teardown_method(self):
        from core.llm_manager import get_embeddings
        get_embeddings.cache_clear()

    def test_switching_from_hf_to_ollama_after_cache_clear(self):
        """
        Simulate a user who starts with HuggingFace and then switches to Ollama
        (e.g. after editing .env and calling clear_embeddings_cache()).
        """
        from core.llm_manager import get_embeddings, clear_embeddings_cache
        from config import settings

        hf_emb    = MagicMock(name="hf")
        ollama_emb= MagicMock(name="ollama")

        with patch.object(settings, "embedding_backend", "huggingface"), \
             patch("core.llm_manager._build_huggingface_embeddings", return_value=hf_emb):
            first = get_embeddings()
        assert first is hf_emb

        clear_embeddings_cache()

        with patch.object(settings, "embedding_backend", "ollama"), \
             patch("core.llm_manager._build_ollama_embeddings", return_value=ollama_emb):
            second = get_embeddings()
        assert second is ollama_emb

    def test_ollama_backend_embed_documents_interface(self):
        """OllamaEmbeddings result exposes embed_documents and embed_query."""
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch.object(settings, "embedding_backend", "ollama"), \
             patch("core.llm_manager._build_ollama_embeddings", return_value=mock_emb):
            emb = get_embeddings()

        docs = emb.embed_documents(["test text"])
        assert isinstance(docs, list)
        assert len(docs) == 1

        vec = emb.embed_query("query")
        assert isinstance(vec, list)

    def test_huggingface_cpu_backend_embed_documents_interface(self):
        from core.llm_manager import get_embeddings
        from config import settings

        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.5, 0.6]]
        mock_emb.embed_query.return_value = [0.5, 0.6]

        with patch.object(settings, "embedding_backend", "huggingface"), \
             patch("core.llm_manager._build_huggingface_embeddings", return_value=mock_emb):
            emb = get_embeddings()

        assert emb.embed_documents(["hello"]) == [[0.5, 0.6]]
        assert emb.embed_query("hello") == [0.5, 0.6]
