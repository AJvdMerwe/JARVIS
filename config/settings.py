"""
config/settings.py
──────────────────
Centralised, validated settings loaded from the environment / .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # ── vLLM ────────────────────────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    vllm_api_key: str = "EMPTY"

    # ── Backend selector ────────────────────────────────────────────────────
    llm_backend: Literal["ollama", "vllm"] = "ollama"

    # ── Whisper ─────────────────────────────────────────────────────────────
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = "base"
    voice_enabled: bool = True
    voice_language: str = "en"

    # ── Vector store ────────────────────────────────────────────────────────
    vector_store_path: Path = Path("./data/vector_store")
    vector_store_collection: str = "assistant_docs"

    # Embedding backend: "huggingface" (in-process, may use GPU)
    #                 or "ollama"      (out-of-process, zero app VRAM)
    embedding_backend: Literal["huggingface", "ollama"] = "huggingface"

    # HuggingFace model path / repo name (only used when backend=huggingface)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Device for HuggingFace embeddings: "cpu" | "cuda" | "mps" | "auto"
    # Default "cpu" — safe for low-VRAM setups.
    # Set "cuda" explicitly only when you have headroom after Docling + LLM.
    embedding_device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"

    # Ollama embedding model (only used when backend=ollama)
    # Good low-VRAM choices: nomic-embed-text, nomic-embed-text-v2-moe, mxbai-embed-large
    ollama_embedding_model: str = "nomic-embed-text"

    # Number of texts embedded per call to the embedding model.
    # Smaller batches reduce peak memory usage at the cost of throughput.
    embedding_batch_size: int = Field(32, ge=1, le=512)

    chunk_size: int = Field(512, ge=64, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=512)

    # ── Document ingestion ──────────────────────────────────────────────────
    uploads_path: Path = Path("./data/uploads")

    # ── News ────────────────────────────────────────────────────────────────
    news_max_articles: int = Field(10, ge=1, le=50)
    news_rss_feeds: str = (
        "https://feeds.bbci.co.uk/news/rss.xml,"
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml,"
        "https://feeds.a.dj.com/rss/RSSWorldNews.xml,"
        "https://techcrunch.com/feed/,"
        "https://feeds.feedburner.com/TechCrunch/"
    )

    # ── Agent ────────────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(15, ge=1, le=50)
    agent_verbose: bool = False
    memory_max_tokens: int = Field(4096, ge=512)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_path: Path = Path("./data/logs/assistant.log")

    # ── Derived helpers ──────────────────────────────────────────────────────
    @field_validator("vector_store_path", "uploads_path", "log_path", mode="before")
    @classmethod
    def _ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def rss_feed_list(self) -> list[str]:
        return [f.strip() for f in self.news_rss_feeds.split(",") if f.strip()]


# Singleton – import this everywhere
settings = Settings()

# Ensure required directories exist at import time
for _dir in (
    settings.vector_store_path,
    settings.uploads_path,
    settings.log_path.parent,
):
    _dir.mkdir(parents=True, exist_ok=True)
