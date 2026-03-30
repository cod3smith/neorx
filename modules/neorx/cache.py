"""
NeoRx Caching Layer
============================

Transparent caching for API responses and computed graphs.

Backends
--------
- **file** (default): JSON files in ``~/.neorx/cache/``
- **redis**: When ``REDIS_URL`` is set (Docker deployment)

Cache TTLs
----------
- API responses: 24 hours (biomedical data changes infrequently)
- Computed graphs: 7 days
- Invalidate with ``cache.clear()`` or ``--no-cache`` CLI flag
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# TTLs in seconds
API_TTL = 86_400       # 24 hours
GRAPH_TTL = 604_800    # 7 days

CACHE_DIR = Path.home() / ".neorx" / "cache"


def _cache_key(source: str, **params: Any) -> str:
    """Derive a deterministic cache key from source + parameters."""
    raw = json.dumps({"source": source, **params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── File Backend ────────────────────────────────────────────────────

class FileCache:
    """Simple JSON file-based cache."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if missing / expired."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("expires_at", 0) < time.time():
                path.unlink(missing_ok=True)
                return None
            logger.debug("Cache HIT (file): %s", key)
            return data["value"]
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl: int = API_TTL) -> None:
        """Store a value with a TTL in seconds."""
        path = self.cache_dir / f"{key}.json"
        data = {"value": value, "expires_at": time.time() + ttl, "created": time.time()}
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
        logger.debug("Cache SET (file): %s (TTL=%ds)", key, ttl)

    def clear(self) -> int:
        """Remove all cached entries.  Returns count removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info("File cache cleared: %d entries removed.", count)
        return count


# ── Redis Backend ───────────────────────────────────────────────────

class RedisCache:
    """Redis-backed cache for Docker deployments."""

    def __init__(self, url: str | None = None) -> None:
        import redis  # type: ignore[import-untyped]

        self.url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client: Any = redis.from_url(self.url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        raw = self._client.get(f"ct:{key}")
        if raw is None:
            return None
        logger.debug("Cache HIT (redis): %s", key)
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: int = API_TTL) -> None:
        self._client.setex(f"ct:{key}", ttl, json.dumps(value, default=str))
        logger.debug("Cache SET (redis): %s (TTL=%ds)", key, ttl)

    def clear(self) -> int:
        keys = self._client.keys("ct:*")
        if keys:
            self._client.delete(*keys)
        count = len(keys) if keys else 0
        logger.info("Redis cache cleared: %d entries.", count)
        return count


# ── Factory ─────────────────────────────────────────────────────────

_instance: FileCache | RedisCache | None = None


def get_cache() -> FileCache | RedisCache:
    """Get the configured cache backend (singleton).

    Checks ``NEORX_CACHE_BACKEND`` env var:
    - ``"redis"`` → attempt Redis, fall back to file
    - ``"file"`` (default) → file-based cache
    """
    global _instance
    if _instance is not None:
        return _instance

    backend = os.environ.get("NEORX_CACHE_BACKEND", "file")

    if backend == "redis":
        try:
            inst = RedisCache()
            inst._client.ping()
            _instance = inst
            logger.info("Using Redis cache backend.")
            return _instance
        except Exception:
            logger.warning("Redis unavailable — falling back to file cache.")

    _instance = FileCache()
    logger.info("Using file cache backend (%s).", _instance.cache_dir)
    return _instance


def reset_cache_instance() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None


def cached_api_call(source: str, ttl: int = API_TTL, **params: Any) -> Any | None:
    """Check the cache for a previous API response.

    Returns the cached value or ``None`` on miss.
    """
    key = _cache_key(source, **params)
    return get_cache().get(key)


def store_api_response(
    source: str,
    value: Any,
    ttl: int = API_TTL,
    **params: Any,
) -> None:
    """Store an API response in the cache."""
    key = _cache_key(source, **params)
    get_cache().set(key, value, ttl)
