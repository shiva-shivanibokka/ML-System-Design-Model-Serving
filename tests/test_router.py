"""
Request router.

This is where a bad v2 becomes a user-visible failure or does not. The
fallbacks are the whole point: whatever goes wrong with the candidate model,
somebody still gets an answer, and it comes from v1.
"""

from __future__ import annotations

import random

import pytest

from deployment.circuit_breaker import CircuitBreakerOpenError
from deployment.router import RequestRouter
from deployment.state_machine import DeploymentState
from tests.conftest import StubModel


@pytest.fixture
def router(stub_v1, stub_v2, state_machine, monkeypatch):
    """A router wired to stubs and to a fresh state machine."""
    import deployment.router as router_mod

    monkeypatch.setattr(router_mod, "state_machine", state_machine)
    r = RequestRouter()
    r.set_models(stub_v1, stub_v2)
    r.state_machine = state_machine
    return r


@pytest.mark.asyncio
async def test_shadow_returns_v1_and_still_runs_v2(router, stub_v1, stub_v2):
    user, shadow, used = await router.route("hello", trace_id="t1")

    assert used == "v1"
    assert user.model_version == "v1"
    # v2 ran, but its answer never reached the caller. That is what shadow is.
    assert stub_v2.calls == ["hello"]
    assert shadow is not None
    assert shadow.model_version == "v2"


@pytest.mark.asyncio
async def test_shadow_survives_a_broken_v2(router, stub_v1, state_machine):
    """A candidate model that throws must not affect the user's answer."""
    router._model_v2 = StubModel(version="v2", raises=True)
    router._model_v2.load()

    user, shadow, used = await router.route("hello", trace_id="t2")

    assert used == "v1"
    assert user.label == "POSITIVE"
    assert shadow is None


@pytest.mark.asyncio
async def test_canary_sends_traffic_to_v2_when_the_draw_selects_it(
    router, state_machine, monkeypatch
):
    state_machine.promote()  # canary_5
    monkeypatch.setattr(random, "random", lambda: 0.0)  # always inside the slice

    _, _, used = await router.route("hello", trace_id="t3")
    assert used == "v2"


@pytest.mark.asyncio
async def test_canary_sends_traffic_to_v1_when_the_draw_does_not(
    router, state_machine, monkeypatch
):
    state_machine.promote()
    monkeypatch.setattr(random, "random", lambda: 0.999)  # outside the slice

    _, _, used = await router.route("hello", trace_id="t4")
    assert used == "v1"


@pytest.mark.asyncio
async def test_canary_falls_back_to_v1_when_v2_raises(router, state_machine, monkeypatch):
    state_machine.promote()
    monkeypatch.setattr(random, "random", lambda: 0.0)
    router._model_v2 = StubModel(version="v2", raises=True)
    router._model_v2.load()

    user, _, used = await router.route("hello", trace_id="t5")

    # The user gets an answer regardless; the fallback is named so the failure
    # is visible in metrics rather than silently absorbed.
    assert user.model_version == "v1"
    assert "fallback" in used


@pytest.mark.asyncio
async def test_canary_falls_back_when_the_breaker_is_open(router, state_machine, monkeypatch):
    state_machine.promote()
    monkeypatch.setattr(random, "random", lambda: 0.0)

    def _blocked(*a, **k):
        raise CircuitBreakerOpenError("circuit open")

    monkeypatch.setattr(router, "_safe_v2_predict_or_raise", _blocked)

    user, _, used = await router.route("hello", trace_id="t6")
    assert user.model_version == "v1"
    assert "fallback" in used


@pytest.mark.asyncio
async def test_full_sends_everything_to_v2(router, state_machine):
    while state_machine.state is not DeploymentState.FULL:
        state_machine.promote()

    user, _, used = await router.route("hello", trace_id="t7")
    assert used == "v2"
    assert user.model_version == "v2"


@pytest.mark.asyncio
async def test_full_still_falls_back_if_v2_dies(router, state_machine):
    while state_machine.state is not DeploymentState.FULL:
        state_machine.promote()
    router._model_v2 = StubModel(version="v2", raises=True)
    router._model_v2.load()

    user, _, used = await router.route("hello", trace_id="t8")
    assert user.model_version == "v1"
    assert "fallback" in used


@pytest.mark.asyncio
async def test_rolled_back_sends_everything_to_v1(router, state_machine, stub_v2):
    state_machine.promote()
    state_machine.rollback()
    assert state_machine.state is DeploymentState.ROLLED_BACK

    _, _, used = await router.route("hello", trace_id="t9")
    assert used == "v1"
    # Rolled back means the candidate is not consulted at all, not even silently.
    assert stub_v2.calls == []


@pytest.mark.asyncio
async def test_canary_split_is_roughly_the_configured_fraction(router, state_machine, monkeypatch):
    """
    Statistical rather than exact: the router draws per request, so the split
    is only right in aggregate. Seeded so the test cannot flake.

    Auto-rollback is disabled for the duration. BaseModel.predict measures real
    wall-clock latency, so under a slow interpreter — coverage instrumentation
    is enough — v2's p99 can drift past 2x v1's purely from scheduling noise,
    the machine rolls back mid-loop, and every remaining request goes to v1.
    That behaviour is correct and has its own tests; here it would just be
    measuring the wrong thing.
    """
    state_machine.promote()
    state_machine.promote()  # canary_25
    fraction = state_machine.v2_traffic_fraction
    monkeypatch.setattr(state_machine, "_check_rollback_thresholds", lambda: None)
    random.seed(1234)

    used = [(await router.route("x", trace_id="s"))[2] for _ in range(400)]
    share = used.count("v2") / len(used)

    assert state_machine.state is DeploymentState.CANARY_25
    assert abs(share - fraction) < 0.08, f"expected ~{fraction}, saw {share}"
