"""
core/cache/tool_cache.py
─────────────────────────
TTL-based in-memory cache for expensive tool calls (web searches,
RSS fetches, Wikipedia lookups). Avoids hitting the same endpoint
multiple times within a session.

Features:
  • Thread-safe LRU-ish cache (dict with timestamp).
  • Per-key TTL; global max size with LRU eviction.
  • `@cached_tool` decorator for wrapping BaseTool._run() methods.
  • Cache hit/miss counters for observability.
  • Optional serialisation to disk for cross-session persistence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Cache entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    value: str
    created_at: float = field(default_factory=time.monotonic)
    hits: int = 0

    def is_expired(self, ttl: float) -> bool:
        return (time.monotonic() - self.created_at) > ttl


# ─────────────────────────────────────────────────────────────────────────────
#  ToolCache
# ─────────────────────────────────────────────────────────────────────────────

class ToolCache:
    """
    Thread-safe TTL cache for tool call results.

    Args:
        default_ttl:  Default time-to-live in seconds.
        max_size:     Maximum number of entries (oldest evicted first).
        persist_path: Optional path to persist cache across restarts.
    """

    def __init__(
        self,
        default_ttl: float = 300.0,     # 5 minutes
        max_size: int = 512,
        persist_path: Optional[Path] = None,
    ) -> None:
        self._default_ttl = default_ttl
        self._max_size    = max_size
        self._persist_path = persist_path
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.hits   = 0
        self.misses = 0

        if persist_path:
            self._load_from_disk()

    # ── Key construction ─────────────────────────────────────────────────────

    @staticmethod
    def make_key(tool_name: str, *args: Any, **kwargs: Any) -> str:
        """Stable SHA-256 key from tool name + arguments."""
        payload = json.dumps(
            {"tool": tool_name, "args": args, "kwargs": kwargs},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    # ── Core operations ──────────────────────────────────────────────────────

    def get(self, key: str, ttl: Optional[float] = None) -> Optional[str]:
        """Return cached value or None if missing/expired."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.is_expired(effective_ttl):
                del self._store[key]
                self.misses += 1
                return None
            entry.hits += 1
            self.hits += 1
            logger.debug("Cache HIT  key=%s (hits=%d)", key[:8], entry.hits)
            return entry.value

    def set(self, key: str, value: str) -> None:
        """Store a value. Evicts oldest entry if over max_size."""
        with self._lock:
            if len(self._store) >= self._max_size:
                oldest = min(self._store, key=lambda k: self._store[k].created_at)
                del self._store[oldest]
                logger.debug("Cache evicted key=%s (max_size reached).", oldest[:8])
            self._store[key] = CacheEntry(value=value)
            logger.debug("Cache SET   key=%s", key[:8])

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> int:
        """Clear all entries. Returns count removed."""
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    def purge_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with self._lock:
            expired = [
                k for k, v in self._store.items()
                if v.is_expired(self._default_ttl)
            ]
            for k in expired:
                del self._store[k]
            return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "max_size": self._max_size,
            "default_ttl": self._default_ttl,
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            now = time.monotonic()
            loaded = 0
            for key, entry_dict in data.items():
                age = time.time() - entry_dict.get("wall_time", 0)
                if age < self._default_ttl:
                    entry = CacheEntry(
                        value=entry_dict["value"],
                        created_at=now - age,
                        hits=entry_dict.get("hits", 0),
                    )
                    self._store[key] = entry
                    loaded += 1
            logger.info("Loaded %d cache entries from disk.", loaded)
        except Exception as exc:
            logger.warning("Could not load cache from disk: %s", exc)

    def save_to_disk(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        now_wall = time.time()
        now_mono = time.monotonic()
        data = {
            k: {
                "value": v.value,
                "wall_time": now_wall - (now_mono - v.created_at),
                "hits": v.hits,
            }
            for k, v in self._store.items()
        }
        self._persist_path.write_text(json.dumps(data, indent=2))
        logger.debug("Saved %d cache entries to disk.", len(data))


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton + per-tool TTLs
# ─────────────────────────────────────────────────────────────────────────────

# Default TTLs per tool (seconds)
_TOOL_TTLS: dict[str, float] = {
    "get_headlines":    180.0,   # 3 min — news changes fast
    "search_topic_news":180.0,
    "web_search":       300.0,   # 5 min
    "wikipedia_lookup": 3600.0,  # 1 hour — Wikipedia rarely changes
    "fetch_webpage":    600.0,   # 10 min
    "search_documents": 60.0,    # 1 min — KB may be updated
}

_cache: Optional[ToolCache] = None
_cache_lock = threading.Lock()


def get_cache() -> ToolCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                from config import settings
                persist = settings.log_path.parent / "tool_cache.json"
                _cache = ToolCache(
                    default_ttl=300.0,
                    max_size=512,
                    persist_path=persist,
                )
    return _cache


# ─────────────────────────────────────────────────────────────────────────────
#  @cached_tool decorator
# ─────────────────────────────────────────────────────────────────────────────

def cached_tool(tool_name: str, ttl: Optional[float] = None):
    """
    Decorator for BaseTool._run() methods.

    Usage::

        class MyTool(BaseTool):
            @cached_tool("my_tool", ttl=120)
            def _run(self, query: str) -> str:
                return expensive_api_call(query)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> str:
            cache = get_cache()
            effective_ttl = ttl or _TOOL_TTLS.get(tool_name, cache._default_ttl)
            # args[0] is self; skip it for the cache key
            key = cache.make_key(tool_name, *args[1:], **kwargs)
            cached = cache.get(key, ttl=effective_ttl)
            if cached is not None:
                return cached
            result = fn(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator
