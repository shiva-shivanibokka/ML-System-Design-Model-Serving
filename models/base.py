"""
Abstract base class for all model versions.
Every model version must implement this interface so the router,
circuit breaker, and deployment state machine can treat v1 and v2 identically.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class PredictionResult:
    """Unified output schema for any model version."""

    label: str  # "POSITIVE" | "NEGATIVE"
    score: float  # confidence in [0.0, 1.0] for the predicted label
    latency_ms: float  # wall-clock inference time in milliseconds
    model_version: str  # "v1" | "v2"
    input_text: str  # original input (used for drift/disagreement logging)
    input_length: int  # character count of input_text


@dataclass
class WarmupResult:
    """Result of the post-load warm-up pass."""

    version: str
    num_requests: int
    first_latency_ms: float  # cold inference (JIT compilation penalty visible here)
    last_latency_ms: float  # warm inference
    speedup_ratio: float  # first / last — shows JIT warm-up effect
    ready: bool


class BaseModel(ABC):
    """
    Abstract base for model v1 and v2.

    Lifecycle:
        model = ModelVx()
        model.load()          # download weights, build pipeline
        warmup = model.warmup()   # run dummy inferences, flip ready=True
        result = model.predict(text)
    """

    def __init__(self, version: str):
        self.version = version
        self._ready: bool = False
        self._loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Download and initialise the model. Must be called before predict()."""
        ...

    @abstractmethod
    def _run_inference(self, text: str) -> tuple[str, float]:
        """
        Run a single forward pass.
        Returns: (label, confidence_score)
        """
        ...

    def predict(self, text: str) -> PredictionResult:
        """
        Public inference entry point. Measures wall-clock latency.
        Raises RuntimeError if model is not ready.
        """
        if not self._ready:
            raise RuntimeError(
                f"Model {self.version} is not ready. Call load() then warmup() first."
            )

        t0 = time.perf_counter()
        label, score = self._run_inference(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResult(
            label=label,
            score=score,
            latency_ms=latency_ms,
            model_version=self.version,
            input_text=text,
            input_length=len(text),
        )

    def warmup(self, dummy_text: str, num_requests: int = 10) -> WarmupResult:
        """
        Run N dummy inferences after model load.

        Purpose:
        - PyTorch JIT / XLA compiles the computation graph on the first call.
          Without warm-up the first real user request pays this penalty (~200-500ms).
        - The warm-up also surfaces OOM or config errors before traffic arrives.
        - The /ready endpoint only flips to True after this completes.

        Returns a WarmupResult with first vs last latency to document the JIT effect.
        """
        if not self._loaded:
            raise RuntimeError(f"Model {self.version} must be loaded before warm-up.")

        latencies: List[float] = []
        for _ in range(num_requests):
            t0 = time.perf_counter()
            self._run_inference(dummy_text)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        self._ready = True

        first = latencies[0]
        last = latencies[-1]
        speedup = first / last if last > 0 else 1.0

        return WarmupResult(
            version=self.version,
            num_requests=num_requests,
            first_latency_ms=round(first, 2),
            last_latency_ms=round(last, 2),
            speedup_ratio=round(speedup, 2),
            ready=True,
        )

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_loaded(self) -> bool:
        return self._loaded
