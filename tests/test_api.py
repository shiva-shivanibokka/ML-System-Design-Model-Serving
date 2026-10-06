"""
API integration tests, driven through FastAPI's TestClient.

No weights are downloaded. api.main loads its models on a background thread
during lifespan; here that loader is replaced with stubs before the client
starts, so the app under test is the real one but the models are not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import StubModel


@pytest.fixture
def client(monkeypatch):
    import api.main as main

    v1 = StubModel(version="v1", default=("POSITIVE", 0.99), latency_ms=10.0)
    v2 = StubModel(version="v2", default=("POSITIVE", 0.97), latency_ms=4.0)

    monkeypatch.setattr(main, "model_v1", v1)
    monkeypatch.setattr(main, "model_v2", v2)

    def _instant_load() -> None:
        v1.load()
        v2.load()
        main._load_stage = "ready"
        main.request_router.set_models(v1, v2)

    monkeypatch.setattr(main, "_load_models", _instant_load)

    with TestClient(main.app) as c:
        # Reset the module-level singletons: they persist across tests
        # otherwise, and a state machine left in canary makes later
        # assertions depend on execution order.
        main.state_machine.reset()
        main.disagreement_monitor.reset()
        main.circuit_breaker.reset()
        main.cache.flush()
        c.v1, c.v2 = v1, v2
        yield c


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def test_health_is_200_and_names_its_backends(client):
    body = client.get("/health").json()
    assert body["status"] in ("ok", "degraded")
    assert body["cache_backend"] in ("redis", "in_process", "none")
    # A fallback that inherits the name of the thing it replaced would hide a
    # real Redis outage, so this must be specific.
    assert body["redis_available"] is (body["cache_backend"] == "redis")


def test_ready_reports_the_load_stage(client):
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    assert "stage" in r.json()


def test_every_read_endpoint_answers_on_a_cold_instance(client):
    """
    REGRESSION: /monitoring/disagreement returned 500 before any prediction had
    been made, which on a scale-to-zero service is how every visitor arrives.
    """
    for path in (
        "/health",
        "/deployment/status",
        "/deployment/audit",
        "/monitoring/disagreement",
        "/monitoring/disagreement/recent?n=5",
        "/monitoring/disagreement/comparisons?n=5",
        "/monitoring/drift",
        "/monitoring/drift/history",
        "/monitoring/cache",
        "/circuit-breaker/status",
    ):
        assert client.get(path).status_code == 200, f"{path} did not answer 200"


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def test_predict_returns_the_full_contract(client):
    body = client.post("/predict", json={"text": "a decent enough film"}).json()
    for key in (
        "label",
        "score",
        "model_version",
        "model_used",
        "deployment_state",
        "latency_ms",
        "cache_hit",
        "trace_id",
    ):
        assert key in body, f"missing {key}"
    assert body["label"] == "POSITIVE"
    assert body["cache_hit"] is False


def test_repeat_prediction_is_a_cache_hit_and_skips_inference(client):
    text = "exactly the same sentence"
    client.post("/predict", json={"text": text})
    calls_after_first = len(client.v1.calls)

    body = client.post("/predict", json={"text": text}).json()
    assert body["cache_hit"] is True
    # The point of the cache is that inference does not run again.
    assert len(client.v1.calls) == calls_after_first


def test_shadow_serves_v1_but_still_runs_v2(client):
    assert client.get("/deployment/status").json()["state"] == "shadow"

    body = client.post("/predict", json={"text": "shadow mode check"}).json()
    assert body["model_used"] == "v1"
    # v2 ran silently — that is what makes the comparison possible at all.
    assert client.v2.calls == ["shadow mode check"]

    stats = client.get("/monitoring/disagreement").json()
    assert stats["total_comparisons"] >= 1


def test_empty_text_is_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_trace_id_is_returned_in_the_header(client):
    r = client.post("/predict", json={"text": "trace check"})
    assert r.headers.get("X-Trace-ID")


# ---------------------------------------------------------------------------
# Deployment control
# ---------------------------------------------------------------------------


def test_promote_and_rollback_round_trip(client):
    start = client.get("/deployment/status").json()["state"]
    assert start == "shadow"

    assert client.post("/deployment/promote").json()["ok"] is True
    assert client.get("/deployment/status").json()["state"] == "canary_5"

    body = client.post("/deployment/rollback").json()
    assert body["ok"] is True
    assert client.get("/deployment/status").json()["state"] == "rolled_back"


def test_rollback_from_shadow_does_not_flush_the_cache(client):
    """
    Rolling back when there is nothing to roll back is a no-op, and must not
    throw away every valid v1 answer as a side effect.
    """
    client.post("/predict", json={"text": "cached answer"})
    before = client.get("/monitoring/cache").json()

    body = client.post("/deployment/rollback").json()
    assert body["ok"] is False
    assert body["cache_flushed"] is False

    # The cached entry survives, so a repeat is still a hit.
    assert client.post("/predict", json={"text": "cached answer"}).json()["cache_hit"] is True
    assert client.get("/monitoring/cache").json()["hits"] > before["hits"]


def test_rollback_after_a_promotion_does_flush(client):
    client.post("/predict", json={"text": "will be dropped"})
    client.post("/deployment/promote")

    body = client.post("/deployment/rollback").json()
    assert body["ok"] is True
    assert body["cache_flushed"] is True
    # An answer from the withdrawn model must not outlive the rollback.
    assert client.post("/predict", json={"text": "will be dropped"}).json()["cache_hit"] is False


def test_audit_grows_with_each_transition(client):
    before = client.get("/deployment/audit").json()["total_entries"]
    client.post("/deployment/promote")
    after = client.get("/deployment/audit").json()

    assert after["total_entries"] == before + 1
    assert after["entries"][-1]["trigger"] == "manual_promote"


def test_circuit_breaker_reset_endpoint(client):
    body = client.post("/circuit-breaker/reset").json()
    assert body["ok"] is True
    assert body["state"] == "closed"


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def test_panel_is_served_and_root_redirects_to_it(client):
    assert client.get("/ui/").status_code == 200
    assert client.get("/ui/styles.css").status_code == 200
    assert client.get("/ui/app.js").status_code == 200

    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/ui/"


def _force_state(main, target: str):
    """Walk the state machine to `target` by promoting, without the auto timer."""
    from deployment.state_machine import DeploymentState

    seen = set()
    while main.state_machine.state.value != target:
        before = main.state_machine.state.value
        if before in seen:
            raise AssertionError(f"cannot reach {target}, stuck at {before}")
        seen.add(before)
        main.state_machine.promote(triggered_by="test")
    assert main.state_machine.state is DeploymentState(target)


# ---------------------------------------------------------------------------
# The cache must not cancel the deployment machinery
#
# These three pin the defect that made the whole project misreport itself: the
# cache check in api.main returned before request_router.route() was ever
# called, so on a cache hit the canary draw did not happen, v2 never ran in
# shadow, and neither the disagreement monitor nor the drift detector saw the
# request. Measured before the fix, at a 50% canary split, 40 identical
# requests put ZERO traffic on v2 while /metrics reported a 0.5 fraction.
# ---------------------------------------------------------------------------


def test_cache_hits_still_honour_the_canary_split(client):
    """A repeated input must still be split across versions."""
    import api.main as main

    _force_state(main, "canary_50")
    payload = {"text": "the same sentence every time"}

    versions = set()
    for _ in range(60):
        body = client.post("/predict", json=payload).json()
        versions.add(body["model_used"])

    assert "v2" in versions, (
        f"60 requests at a 50% canary split never reached v2 (saw {versions}). "
        "The cache is returning before the router draws."
    )


def test_shadow_mode_runs_v2_even_when_the_answer_is_cached(client):
    """Shadow mode compares v1 and v2 on every request, cached or not."""
    import api.main as main

    _force_state(main, "shadow")
    payload = {"text": "shadow mode should always compare"}

    client.post("/predict", json=payload)  # populates the cache
    before = main.disagreement_monitor.get_stats()["total_comparisons"]
    for _ in range(5):
        client.post("/predict", json=payload)  # all cache hits
    after = main.disagreement_monitor.get_stats()["total_comparisons"]

    assert after - before == 5, (
        f"5 cached requests in shadow produced {after - before} comparisons, expected 5. "
        "v2 is not running on cache hits, so the shadow comparison is sampling "
        "only cache misses."
    )


def test_drift_detector_sees_cached_requests_too(client):
    """Drift is about the input distribution, which does not care about caching."""
    import api.main as main

    payload = {"text": "drift should count every request"}
    client.post("/predict", json=payload)
    before = main.drift_detector.get_status()["total_records"]
    for _ in range(5):
        client.post("/predict", json=payload)
    after = main.drift_detector.get_status()["total_records"]

    assert after - before == 5, (
        f"5 cached requests recorded {after - before} drift samples, expected 5. "
        "The reference window is being built from cache misses only."
    )
