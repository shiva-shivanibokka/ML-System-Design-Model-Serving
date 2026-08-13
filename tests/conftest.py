"""
Shared fixtures.

The important constraint here is that CI must never download model weights.
DistilBERT is ~260MB and the two models take about a minute to load and warm,
which would dominate every run and make the suite depend on the HuggingFace
Hub being reachable. Every test that needs a model gets a stub whose behaviour
is deterministic, so assertions can be exact rather than approximate.
"""

from __future__ import annotations

import os

import pytest

# Must be set before configs.settings is imported anywhere: an empty host is
# how this codebase says "there is no Redis here", and tests should exercise
# the in-process path rather than wait on a connection that cannot succeed.
os.environ.setdefault("REDIS_HOST", "")
os.environ.setdefault("EPHEMERAL_STATE", "1")

from models.base import BaseModel  # noqa: E402


class StubModel(BaseModel):
    """
    A model that returns whatever it is told to, instantly.

    `script` maps input text to (label, score). Anything not in the script gets
    `default`. Setting `raises` makes every prediction raise, which is how the
    router's fallback and the circuit breaker get exercised.
    """

    def __init__(
        self,
        version: str = "v1",
        script: dict[str, tuple[str, float]] | None = None,
        default: tuple[str, float] = ("POSITIVE", 0.99),
        raises: bool = False,
        latency_ms: float = 1.0,
    ) -> None:
        super().__init__(version=version)
        self.script = script or {}
        self.default = default
        self.raises = raises
        self.latency_ms = latency_ms
        self.calls: list[str] = []

    def load(self) -> None:
        self._loaded = True
        # BaseModel.predict() gates on _ready, which normally only flips after
        # warmup(). Setting it here makes the stub usable without a warm-up
        # pass while leaving the real readiness logic untouched.
        self._ready = True

    def _run_inference(self, text: str) -> tuple[str, float]:
        self.calls.append(text)
        if self.raises:
            raise RuntimeError(f"{self.version} inference failed")
        return self.script.get(text, self.default)

    # predict() is deliberately NOT overridden. The base class builds the
    # PredictionResult and measures latency, so tests exercise that real code
    # path rather than a parallel one that could drift away from it.


@pytest.fixture
def stub_v1() -> StubModel:
    m = StubModel(version="v1", default=("POSITIVE", 0.99), latency_ms=10.0)
    m.load()  # BaseModel.predict() refuses to run until the model is ready
    return m


@pytest.fixture
def stub_v2() -> StubModel:
    m = StubModel(version="v2", default=("POSITIVE", 0.97), latency_ms=5.0)
    m.load()
    return m


@pytest.fixture
def state_machine():
    """A fresh state machine per test, with persistence and threads disabled."""
    from deployment.state_machine import DeploymentStateMachine

    sm = DeploymentStateMachine()
    yield sm
    sm.stop()


@pytest.fixture
def breaker():
    from deployment.circuit_breaker import CircuitBreaker

    return CircuitBreaker()


@pytest.fixture
def monitor():
    from monitoring.disagreement import DisagreementMonitor

    return DisagreementMonitor()


@pytest.fixture
def cache():
    from cache.redis_cache import PredictionCache

    c = PredictionCache()
    c.connect()
    return c
