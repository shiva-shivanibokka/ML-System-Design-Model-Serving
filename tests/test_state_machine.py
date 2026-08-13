"""
Deployment state machine.

The rollback paths matter most: a canary that cannot pull itself back is just
a slower way to ship a bad model, and the noise guard is what stops it doing
so on three unlucky requests.
"""

from __future__ import annotations

import pytest

from configs.settings import settings
from deployment.state_machine import (
    PROGRESSION_ORDER,
    DeploymentState,
    DeploymentStateMachine,
)


def test_starts_in_shadow(state_machine: DeploymentStateMachine):
    assert state_machine.state is DeploymentState.SHADOW
    assert state_machine.v2_traffic_fraction == 0.0


def test_promote_walks_the_progression_in_order(state_machine: DeploymentStateMachine):
    seen = [state_machine.state]
    for _ in range(len(PROGRESSION_ORDER) - 1):
        result = state_machine.promote()
        assert result["ok"] is True
        seen.append(state_machine.state)

    assert seen == PROGRESSION_ORDER
    assert state_machine.state is DeploymentState.FULL


def test_cannot_promote_past_full(state_machine: DeploymentStateMachine):
    for _ in range(len(PROGRESSION_ORDER) - 1):
        state_machine.promote()

    result = state_machine.promote()
    assert result["ok"] is False
    assert "full" in result["reason"].lower()
    assert state_machine.state is DeploymentState.FULL


def test_traffic_fraction_rises_monotonically(state_machine: DeploymentStateMachine):
    fractions = [state_machine.v2_traffic_fraction]
    for _ in range(len(PROGRESSION_ORDER) - 1):
        state_machine.promote()
        fractions.append(state_machine.v2_traffic_fraction)

    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_manual_rollback_leaves_the_progression(state_machine: DeploymentStateMachine):
    state_machine.promote()
    state_machine.promote()
    assert state_machine.state is DeploymentState.CANARY_25

    result = state_machine.rollback(trigger="manual")
    assert result["ok"] is True
    assert state_machine.state is DeploymentState.ROLLED_BACK
    # Rolled back means no live traffic on the candidate, whatever it was before.
    assert state_machine.v2_traffic_fraction == 0.0


def test_promote_from_rolled_back_restarts_at_shadow(
    state_machine: DeploymentStateMachine,
):
    state_machine.promote()
    state_machine.rollback()
    assert state_machine.state is DeploymentState.ROLLED_BACK

    state_machine.promote()
    assert state_machine.state is DeploymentState.SHADOW


def test_rollback_is_a_no_op_when_already_out_of_the_progression(
    state_machine: DeploymentStateMachine,
):
    """
    Rolling back from shadow reports failure rather than pretending to act.
    The API route keys its cache flush off this, so a false success here would
    empty the cache for nothing. Cache flushing itself is covered in the API
    tests, because the state machine deliberately does no I/O.
    """
    assert state_machine.state is DeploymentState.SHADOW
    result = state_machine.rollback()
    assert result["ok"] is False
    assert state_machine.state is DeploymentState.SHADOW


# ---------------------------------------------------------------------------
# Automatic rollback
# ---------------------------------------------------------------------------


def _drive_v2(sm: DeploymentStateMachine, n: int, latency_ms: float, errors: int = 0):
    """Send n v2 requests through the machine, the first `errors` of them failing."""
    for i in range(n):
        sm.record_request(model_used="v2", latency_ms=latency_ms, error=i < errors)


def test_auto_rollback_on_error_rate(state_machine: DeploymentStateMachine):
    state_machine.promote()  # canary_5 — auto-rollback only applies on canary+
    cfg = settings.deployment.rollback

    n = cfg.min_requests_before_check + 10
    failing = int(n * (cfg.error_rate_threshold + 0.15)) + 1
    _drive_v2(state_machine, n, latency_ms=5.0, errors=failing)

    assert state_machine.state is DeploymentState.ROLLED_BACK
    triggers = [e["trigger"] for e in state_machine.get_audit_log()]
    assert "auto_rollback_error_rate" in triggers


def test_auto_rollback_on_p99_latency(state_machine: DeploymentStateMachine):
    state_machine.promote()
    cfg = settings.deployment.rollback

    # Establish a v1 baseline, then make v2 comfortably slower than the multiple.
    for _ in range(50):
        state_machine.record_request(model_used="v1", latency_ms=10.0, error=False)
    _drive_v2(
        state_machine,
        cfg.min_requests_before_check + 5,
        latency_ms=10.0 * cfg.latency_p99_multiplier * 3,
    )

    assert state_machine.state is DeploymentState.ROLLED_BACK
    triggers = [e["trigger"] for e in state_machine.get_audit_log()]
    assert "auto_rollback_latency" in triggers


def test_noise_guard_holds_below_min_requests(state_machine: DeploymentStateMachine):
    """
    Every request failing is not enough to roll back if there have not been
    many requests. Two bad calls out of two is noise, not evidence.
    """
    state_machine.promote()
    cfg = settings.deployment.rollback

    _drive_v2(
        state_machine,
        cfg.min_requests_before_check - 1,
        latency_ms=5.0,
        errors=cfg.min_requests_before_check - 1,
    )

    assert state_machine.state is DeploymentState.CANARY_5


def test_shadow_never_auto_rolls_back(state_machine: DeploymentStateMachine):
    """
    In shadow, v2 serves nobody. A 100% failure rate there is information, not
    an incident, and must not move the machine.
    """
    cfg = settings.deployment.rollback
    n = cfg.min_requests_before_check + 20
    _drive_v2(state_machine, n, latency_ms=5.0, errors=n)

    assert state_machine.state is DeploymentState.SHADOW


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def test_audit_records_every_transition_with_metrics(
    state_machine: DeploymentStateMachine,
):
    state_machine.promote()
    state_machine.rollback(trigger="manual_rollback")

    entries = state_machine.get_audit_log()
    assert [e["trigger"] for e in entries][-2:] == ["manual_promote", "manual_rollback"]

    for entry in entries:
        # An audit line that records the transition but not the evidence cannot
        # explain the decision afterwards, which is the only reason it exists.
        assert {"timestamp", "from_state", "to_state", "trigger"} <= set(entry)
        assert "v2_error_rate_at_event" in entry
        assert "v2_p99_latency_ms" in entry
        assert "requests_seen" in entry


def test_status_payload_is_complete_on_a_fresh_machine(
    state_machine: DeploymentStateMachine,
):
    status = state_machine.get_status()
    for key in (
        "state",
        "v2_traffic_fraction",
        "v2_requests",
        "v2_error_rate",
        "v2_p99_latency_ms",
        "v1_p99_latency_ms",
        "total_requests",
        "time_in_state_seconds",
        "state_durability",
        "rollback_thresholds",
    ):
        assert key in status, f"missing {key}"


@pytest.mark.parametrize("state", list(DeploymentState))
def test_every_state_has_a_traffic_fraction(state, state_machine):
    from deployment.state_machine import V2_TRAFFIC_FRACTION

    assert state in V2_TRAFFIC_FRACTION
    assert 0.0 <= V2_TRAFFIC_FRACTION[state] <= 1.0
