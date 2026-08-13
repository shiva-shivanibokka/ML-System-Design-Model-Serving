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

import structlog

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from configs.settings import settings

log = structlog.get_logger(__name__)


class _InProcessBackend:
    """
    Dict-backed stand-in that speaks the four Redis calls this cache uses.

    The managed deployment runs a single instance with no Redis reachable, and
    a cache that is switched off is a feature the demo can no longer show. One
    process means one cache, so a dict is not an approximation of Redis here —
    it is the same thing without the network hop or the hosted dependency.

    ponytail: no size bound; entries expire by TTL and the process is
    short-lived. Add an LRU eviction if it ever holds a long-running instance.
    """

    # Bound here because setex's keyword argument is also called `time`, to
    # match redis-py's signature exactly and keep the call sites unchanged.
    _now = staticmethod(time.time)

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._now() >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def setex(self, name: str, time: int, value: str) -> None:  # noqa: A002
        self._store[name] = (self._now() + time, value)

    def keys(self, pattern: str) -> list[str]:
        prefix = pattern.rstrip("*")
        return [k for k in list(self._store) if k.startswith(prefix)]

    def delete(self, *keys: str) -> int:
        return sum(self._store.pop(k, None) is not None for k in keys)


class PredictionCache:
    """
    Prediction cache with graceful degradation.

    Prefers Redis. If Redis is unreachable it falls back to an in-process
    store rather than switching the cache off, and reports which backend is
    live so the difference is never silent.
    """

    def __init__(self) -> None:
        self._client: object | None = None
        self._available: bool = False
        self._backend: str = "none"
        self._hits: int = 0
        self._misses: int = 0
        self._errors: int = 0

    def connect(self) -> bool:
        """
        Attempt to connect to Redis, falling back to the in-process backend.

        Returns True if Redis specifically is live. The cache is usable either
        way — check `backend` to find out which one answered.
        """
        if not REDIS_AVAILABLE:
            log.warning("redis_package_not_installed", fallback="in_process_cache")
            self._use_in_process()
            return False

        # An empty REDIS_HOST means "there is no Redis here", which is not the
        # same as "Redis is down". Without this the connect attempt spends
        # several seconds resolving a hostname that was never going to exist —
        # paid on every cold start, to learn something already known.
        if not settings.cache.host:
            log.info("redis_not_configured", fallback="in_process_cache")
            self._use_in_process()
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
            self._backend = "redis"
            log.info(
                "cache_connected",
                host=settings.cache.host,
                port=settings.cache.port,
            )
            return True
        except Exception as e:
            log.warning("redis_unavailable", error=str(e), fallback="in_process_cache")
            self._use_in_process()
            return False

    def _use_in_process(self) -> None:
        self._client = _InProcessBackend()
        self._available = True
        self._backend = "in_process"
        log.info("cache_connected", backend="in_process", ttl=settings.cache.ttl_seconds)

    def _make_key(self, text: str, model_version: str) -> str:
        """
        Build a cache key: pred:{version}:{sha256_prefix}
        Version is part of the key so v1 and v2 never share cached entries.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{settings.cache.key_prefix}{model_version}:{text_hash}"

    def get(self, text: str, model_version: str) -> dict | None:
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
            "backend": self._backend,
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

    @property
    def backend(self) -> str:
        """ "redis" | "in_process" | "none" — which store is answering."""
        return self._backend
