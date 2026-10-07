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

    # The run must start over, so (threshold - 1) more failures still leave it
    # closed. Asserting only CLOSED after ONE further failure was true under the
    # old decrement-by-one behaviour too, so the test named for this claim could
    # not detect the difference: after 4 failures and 1 success the counter sat
    # at 3, and two more failures opened the breaker where a real reset needs
    # five.
    for _ in range(threshold - 1):
        with pytest.raises(RuntimeError):
            breaker.call(_boom)
    assert (
        breaker.state is CircuitState.CLOSED
    ), "a success did not reset the consecutive-failure run"

    # ...and the very next failure, the threshold-th of the new run, opens it.
    with pytest.raises(RuntimeError):
        breaker.call(_boom)
    assert breaker.state is CircuitState.OPEN


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


def test_a_hanging_call_times_out_and_counts_as_a_failure(breaker: CircuitBreaker):
    """call_timeout_seconds must actually bound the call.

    It was loaded from config and never read: a 7-second call under a 5-second
    timeout returned normally, failures stayed at 0 and the breaker stayed
    closed. The breaker's own docstring builds its justification on this timeout
    ("Every request waits call_timeout_seconds before falling back to v1"), so
    the single failure mode it exists to catch -- a hung v2 -- was the one it
    could not see.
    """
    import time

    from deployment.circuit_breaker import CircuitBreakerTimeoutError

    budget = breaker._call_timeout
    assert budget and budget > 0, "this test needs a configured timeout"

    t0 = time.perf_counter()
    with pytest.raises(CircuitBreakerTimeoutError):
        breaker.call(lambda: time.sleep(budget * 2) or "never returned")
    elapsed = time.perf_counter() - t0

    # The caller must not wait for the hung call to finish.
    assert elapsed < budget * 1.8, (
        f"caller waited {elapsed:.1f}s for a {budget}s timeout -- "
        "the executor is blocking on shutdown"
    )
    assert (
        breaker.get_status()["failure_count"] >= 1
    ), "a timed-out call must count as a failure, or the breaker can never open on a hang"
