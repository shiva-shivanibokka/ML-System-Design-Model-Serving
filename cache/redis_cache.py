"""
Redis prediction cache.

Cache strategy:
  - Key: sha256(text)[:16] — short hash, avoids storing raw text as key
  - Value: JSON-serialised PredictionResult fields
  - TTL: configurable (default 300s)
  - Hit: return cached result in <5ms, skip model inference entirely
  - Miss: run inference, store result, return

Why cache?
  Cache hits eliminate model inference (~30-80ms) entirely.
  For repeated queries (monitoring dashboards, health checks, A/B test scripts
  that replay the same texts) the cache hit rate is often 30-50%.

Cache key design:
  We hash the text so we never store raw PII in Redis keys.
  We include the model_version in the key so v1 and v2 never share cache entries
  — this is important during canary when both versions are live.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

import structlog

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from configs.settings import settings

log = structlog.get_logger(__name__)


class PredictionCache:
    """
    Redis-backed prediction cache with graceful degradation.

    If Redis is unavailable (connection refused, timeout), the cache
    silently falls through to model inference. The API never fails
    because of a cache outage.
    """

    def __init__(self) -> None:
        self._client: Optional[object] = None
        self._available: bool = False
        self._hits: int = 0
        self._misses: int = 0
        self._errors: int = 0

    def connect(self) -> bool:
        """
        Attempt to connect to Redis. Returns True if successful.
        Called once during FastAPI lifespan startup.
        """
        if not REDIS_AVAILABLE:
            log.warning("redis_package_not_installed", fallback="cache_disabled")
            return False

        try:
            self._client = redis.Redis(
                host=settings.cache.host,
                port=settings.cache.port,
                db=settings.cache.db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=1,
            )
            self._client.ping()
            self._available = True
            log.info(
                "cache_connected",
                host=settings.cache.host,
                port=settings.cache.port,
            )
            return True
        except Exception as e:
            log.warning("cache_unavailable", error=str(e), fallback="inference_only")
            self._available = False
            return False

    def _make_key(self, text: str, model_version: str) -> str:
        """
        Build a cache key: pred:{version}:{sha256_prefix}
        Version is part of the key so v1 and v2 never share cached entries.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{settings.cache.key_prefix}{model_version}:{text_hash}"

    def get(self, text: str, model_version: str) -> Optional[dict]:
        """
        Look up a cached prediction. Returns dict or None.
        Never raises — cache miss and cache error both return None.
        """
        if not self._available:
            return None

        # Don't cache very long texts — the key/value overhead isn't worth it
        if len(text) > settings.cache.max_text_length_for_cache:
            return None

        key = self._make_key(text, model_version)
        try:
            raw = self._client.get(key)
            if raw is not None:
                self._hits += 1
                return json.loads(raw)
            self._misses += 1
            return None
        except Exception as e:
            self._errors += 1
            log.debug("cache_get_error", error=str(e))
            return None

    def set(self, text: str, model_version: str, result_dict: dict) -> None:
        """
        Store a prediction result. TTL from config. Never raises.
        """
        if not self._available:
            return

        if len(text) > settings.cache.max_text_length_for_cache:
            return

        key = self._make_key(text, model_version)
        try:
            self._client.setex(
                name=key,
                time=settings.cache.ttl_seconds,
                value=json.dumps(result_dict),
            )
        except Exception as e:
            self._errors += 1
            log.debug("cache_set_error", error=str(e))

    def stats(self) -> dict:
        """Return cache performance stats for Prometheus and the Gradio UI."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_rate": round(hit_rate, 4),
            "available": self._available,
        }

    def flush(self) -> int:
        """
        Flush all prediction cache entries (keys matching our prefix).
        Used after a model version is rolled back to avoid serving stale v2 cache.
        Returns number of keys deleted.
        """
        if not self._available:
            return 0
        try:
            keys = self._client.keys(f"{settings.cache.key_prefix}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            log.warning("cache_flush_error", error=str(e))
            return 0

    @property
    def is_available(self) -> bool:
        return self._available
