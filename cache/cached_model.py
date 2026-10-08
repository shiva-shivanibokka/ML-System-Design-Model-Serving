"""A model wrapper that serves repeats from cache without bypassing the router.

Why this exists
---------------
The cache used to sit in front of everything, in `api.main.predict`: look up the
text, and on a hit return immediately. That one early `return` cancelled the
three mechanisms this project exists to demonstrate.

  * **The canary split never happened.** The cache key was chosen from the
    deployment *state* (`"v2" if state == "full" else "v1"`), so on a hit the
    router — and with it the weighted random draw — never ran. Measured at a 50%
    split, 60 identical requests put **zero** traffic on v2 while `/metrics`
    reported `canary_v2_traffic_fraction 0.5`.
  * **Shadow mode never ran v2.** Five identical requests produced zero
    comparisons, so the disagreement monitor was sampling cache *misses* only.
  * **Drift never saw cached requests**, so the "first 200 requests" reference
    window was built from misses only.

The fix is to move the cache one layer down. The router still decides which
version serves and still dispatches shadow v2 on every request; it is the
individual model call that gets short-circuited. Caching is an inference
optimisation, not a routing decision, and this puts it where it belongs.

A cached result keeps the `model_version` of whatever produced it and is marked
`from_cache`, so latency statistics can exclude it rather than being flattened
by zeros.
"""

from __future__ import annotations

import structlog

from models.base import BaseModel, PredictionResult, WarmupResult

log = structlog.get_logger()


class CachedModel(BaseModel):
    """Wraps a model so repeated inputs skip inference but nothing else.

    Everything the router, circuit breaker and state machine rely on is
    delegated to the wrapped model, so this is transparent to them.
    """

    def __init__(self, model: BaseModel, cache) -> None:
        super().__init__(version=model.version)
        self._model = model
        self._cache = cache

    # ---- delegation -------------------------------------------------------
    def load(self) -> None:
        self._model.load()

    def warmup(self, *args, **kwargs) -> WarmupResult:
        return self._model.warmup(*args, **kwargs)

    def _run_inference(self, text: str) -> tuple[str, float]:
        return self._model._run_inference(text)

    @property
    def is_ready(self) -> bool:
        return self._model.is_ready

    def __getattr__(self, name: str):
        # Anything not defined here belongs to the wrapped model. Guarded
        # against the wrapper's own attributes to avoid infinite recursion
        # during __init__.
        if name in ("_model", "_cache"):
            raise AttributeError(name)
        return getattr(self._model, name)

    # ---- the one behaviour that changes ----------------------------------
    def predict(self, text: str) -> PredictionResult:
        cached = self._cache.get(text, self.version)
        if cached is not None:
            return PredictionResult(
                label=cached["label"],
                score=cached["score"],
                # A cache hit did no inference. Reporting 0.0 as a latency would
                # drag every percentile toward zero, so callers that aggregate
                # latency check `from_cache` and skip these.
                latency_ms=0.0,
                from_cache=True,
                model_version=cached.get("model_version", self.version),
                input_text=text,
                input_length=len(text),
            )

        result = self._model.predict(text)
        self._cache.set(
            text,
            self.version,
            {
                "label": result.label,
                # Store the rounded value the API returns, so a cached response
                # is byte-identical to the fresh one it replaces. Storing full
                # precision made /predict return 0.9994489550590515 on a hit and
                # 0.9994 on a miss for the same input.
                "score": round(result.score, 4),
                "model_version": result.model_version,
                "model_used": result.model_version,
            },
        )
        return result


__all__ = ["CachedModel"]
