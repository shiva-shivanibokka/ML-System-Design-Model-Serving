"""
Circuit Breaker for Model V2.

The circuit breaker pattern prevents cascading failures when v2 starts
malfunctioning (OOM, timeout, corrupt weights, GPU errors).

States:
  CLOSED    → normal operation. Requests flow through to v2.
  OPEN      → circuit is tripped. All requests fall back to v1 immediately.
              No requests sent to v2. Fail-fast with <1ms response time.
  HALF_OPEN → after timeout_seconds, one probe request is allowed through.
              If it succeeds → CLOSED. If it fails → OPEN again.

Why this matters in production:
  Without a circuit breaker:
    - v2 starts timing out (e.g. OOM on GPU)
    - Every request waits call_timeout_seconds before falling back to v1
    - At 100 req/s, 5s timeout = 500 in-flight requests piling up
    - Server threads exhausted → entire API goes down
    - This is how a v2 bug takes down a v1 service

  With a circuit breaker:
    - After failure_threshold failures → circuit opens
    - All subsequent requests return v1 in <1ms (no network call to v2)
    - After timeout_seconds → one probe test
    - If v2 recovered → circuit closes, traffic resumes
    - If v2 still broken → circuit stays open, restart the timer

Reference: Netflix Hystrix, resilience4j, Go's sony/gobreaker.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from enum import Enum
from typing import TypeVar

import structlog

from configs.settings import settings

log = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal — requests flow through
    OPEN = "open"  # Tripped — all requests fail-fast to v1
    HALF_OPEN = "half_open"  # Recovery probe — one request allowed


class CircuitBreakerTimeoutError(Exception):
    """A call exceeded call_timeout_seconds and was abandoned by the caller."""


class CircuitBreakerOpenError(Exception):
    """Raised when a call is blocked by an open circuit."""

    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker for v2 inference calls.

    Usage:
        cb = CircuitBreaker()
        try:
            result = cb.call(model_v2.predict, text)
        except CircuitBreakerOpenError:
            result = model_v1.predict(text)  # immediate fallback
    """

    def __init__(self) -> None:
        cfg = settings.circuit_breaker
        self._failure_threshold: int = cfg.failure_threshold
        self._success_threshold: int = cfg.success_threshold
        self._timeout_seconds: int = cfg.timeout_seconds
        self._call_timeout: float = cfg.call_timeout_seconds

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: float = 0.0
        self._total_blocked: int = 0
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Core call method
    # -----------------------------------------------------------------------

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute fn(*args, **kwargs) through the circuit breaker.

        - If OPEN and timeout not elapsed: raises CircuitBreakerOpenError
        - If OPEN and timeout elapsed: transitions to HALF_OPEN, allows one call
        - If HALF_OPEN and call succeeds: transitions to CLOSED
        - If HALF_OPEN and call fails: transitions back to OPEN
        - If CLOSED and call fails: increments counter, opens if threshold reached
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed < self._timeout_seconds:
                    self._total_blocked += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit OPEN — v2 blocked for {self._timeout_seconds - elapsed:.0f}s more"
                    )
                # Timeout elapsed — allow one probe
                log.info(
                    "circuit_half_open",
                    elapsed_s=round(elapsed, 1),
                    timeout_s=self._timeout_seconds,
                )
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

        # Execute the call outside the lock to avoid blocking other threads
        self._total_calls += 1
        try:
            result = self._call_with_timeout(fn, *args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(str(e))
            raise

    def _call_with_timeout(self, fn, *args, **kwargs):
        """Run fn, failing the call if it exceeds call_timeout_seconds.

        This used to be absent. `self._call_timeout` was loaded from config and
        never read by anything, so `fn()` ran unbounded: a 7-second call under a
        5-second timeout returned normally with failures=0 and the breaker still
        closed. The class docstring builds the entire case for the breaker on
        this timeout ("Every request waits call_timeout_seconds before falling
        back to v1 ... server threads exhausted"), so the one failure mode it
        exists to catch was the one it could not detect. A hung v2 hung the
        request forever; only raised exceptions ever tripped it.

        The worker thread is not killed on timeout -- Python cannot safely do
        that -- so a hung call still occupies its thread. What this changes is
        that the CALLER stops waiting and the breaker counts a failure, which is
        what lets the breaker open and shed load to v1.
        """
        if not self._call_timeout or self._call_timeout <= 0:
            return fn(*args, **kwargs)

        # Deliberately NOT a `with` block: ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True), which blocks until the worker finishes -- so a
        # 7-second call under a 5-second timeout still took 7 seconds to return,
        # and the timeout bought the caller nothing. shutdown(wait=False) lets
        # the caller leave at the deadline and the orphaned thread finish on its
        # own.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=self._call_timeout)
            except FuturesTimeoutError as exc:
                raise CircuitBreakerTimeoutError(f"call exceeded {self._call_timeout}s") from exc
        finally:
            pool.shutdown(wait=False)

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    log.info(
                        "circuit_closed",
                        after_successes=self._success_count,
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # The threshold counts CONSECUTIVE failures, so one success
                # starts the count over. This used to decrement by one, which
                # is a sliding window, not a consecutive count: after 4 failures
                # and 1 success the count stood at 3, so two more failures
                # opened the breaker where a true reset would have needed five.
                # The log line below it has always said "consecutive_failures",
                # and README describes it that way in two places.
                self._failure_count = 0

    def _on_failure(self, error: str) -> None:
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — back to OPEN
                log.warning(
                    "circuit_open_from_half_open",
                    error=error,
                    timeout_s=self._timeout_seconds,
                )
                self._state = CircuitState.OPEN

            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self._failure_threshold
            ):
                log.warning(
                    "circuit_opened",
                    consecutive_failures=self._failure_count,
                    threshold=self._failure_threshold,
                    error=error,
                )
                self._state = CircuitState.OPEN

    # -----------------------------------------------------------------------
    # Manual controls (for Gradio UI and admin API)
    # -----------------------------------------------------------------------

    def force_open(self) -> None:
        """Manually open the circuit (admin operation)."""
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
        log.info("circuit_force_opened")

    def force_close(self) -> None:
        """Manually close the circuit after a fix (admin operation)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
        log.info("circuit_force_closed")

    def reset(self) -> None:
        """Reset all state and counters. Used between deployment stages."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._total_blocked = 0
            self._total_calls = 0
            self._total_failures = 0
        log.info("circuit_reset")

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def get_status(self) -> dict:
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_blocked": self._total_blocked,
                "failure_rate": round(self._total_failures / self._total_calls, 4)
                if self._total_calls > 0
                else 0.0,
                "thresholds": {
                    "failure_threshold": self._failure_threshold,
                    "success_threshold": self._success_threshold,
                    "timeout_seconds": self._timeout_seconds,
                },
            }


# Module-level singleton
circuit_breaker = CircuitBreaker()
