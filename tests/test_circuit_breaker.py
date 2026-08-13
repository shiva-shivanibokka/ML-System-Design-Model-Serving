"""
Circuit breaker.

The breaker is the floor under the rollout: even when the canary thresholds
are not tripped, a model that is failing must stop being called. The states
worth testing are the transitions, not the happy path.
"""

from __future__ import annotations

import pytest

from deployment.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


def _ok() -> str:
    return "fine"


def _boom() -> str:
    raise RuntimeError("v2 exploded")


def test_starts_closed_and_passes_calls_through(breaker: CircuitBreaker):
    assert breaker.state is CircuitState.CLOSED
    assert breaker.call(_ok) == "fine"


def test_opens_after_consecutive_failures(breaker: CircuitBreaker):
    threshold = breaker.get_status()["thresholds"]["failure_threshold"]

    for _ in range(threshold):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)

    assert breaker.state is CircuitState.OPEN
    assert breaker.is_open is True


def test_success_resets_the_failure_run(breaker: CircuitBreaker):
    """
    The threshold counts *consecutive* failures. One success in the middle
    means the model is not reliably broken, and the count starts over.
    """
    threshold = breaker.get_status()["thresholds"]["failure_threshold"]

    for _ in range(threshold - 1):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)
    breaker.call(_ok)
    with pytest.raises(RuntimeError):
        breaker.call(_boom)

    assert breaker.state is CircuitState.CLOSED


def test_open_blocks_calls_without_invoking_them(breaker: CircuitBreaker):
    threshold = breaker.get_status()["thresholds"]["failure_threshold"]
    for _ in range(threshold):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)

    calls = []

    def _tracked():
        calls.append(1)
        return "should not run"

    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(_tracked)

    # The point of an open breaker is that the failing dependency is not
    # touched at all — blocking after calling would save nothing.
    assert calls == []
    assert breaker.get_status()["total_blocked"] >= 1


def test_half_open_after_timeout_then_closes_on_successes(breaker: CircuitBreaker, monkeypatch):
    status = breaker.get_status()["thresholds"]
    for _ in range(status["failure_threshold"]):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)
    assert breaker.state is CircuitState.OPEN

    # Jump past the timeout rather than sleeping through it.
    breaker._last_failure_time -= status["timeout_seconds"] + 1

    for _ in range(status["success_threshold"]):
        assert breaker.call(_ok) == "fine"

    assert breaker.state is CircuitState.CLOSED


def test_half_open_reopens_when_the_probe_fails(breaker: CircuitBreaker):
    status = breaker.get_status()["thresholds"]
    for _ in range(status["failure_threshold"]):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)

    breaker._last_failure_time -= status["timeout_seconds"] + 1

    with pytest.raises(RuntimeError):
        breaker.call(_boom)

    # A failed probe means the dependency is still sick; going straight back to
    # OPEN avoids hammering it once per timeout window.
    assert breaker.state is CircuitState.OPEN


def test_reset_closes_it_manually(breaker: CircuitBreaker):
    threshold = breaker.get_status()["thresholds"]["failure_threshold"]
    for _ in range(threshold):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)
    assert breaker.state is CircuitState.OPEN

    breaker.reset()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.call(_ok) == "fine"


def test_status_payload_shape(breaker: CircuitBreaker):
    status = breaker.get_status()
    for key in (
        "state",
        "failure_count",
        "total_calls",
        "total_failures",
        "total_blocked",
        "failure_rate",
        "thresholds",
    ):
        assert key in status, f"missing {key}"
