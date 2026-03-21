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
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"

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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
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
