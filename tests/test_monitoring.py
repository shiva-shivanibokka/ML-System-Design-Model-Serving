"""
Disagreement monitor, drift detector and prediction cache.

The first test here is a regression: the empty-window branch of get_stats used
to return a different set of keys from the populated branch, so the endpoint
answered 500 on any freshly started instance — which, on a service that scales
to zero, is the state every visitor arrives in.
"""

from __future__ import annotations

from api import schemas
from monitoring.drift import DriftDetector

# ---------------------------------------------------------------------------
# Disagreement
# ---------------------------------------------------------------------------


def test_empty_window_validates_against_the_response_model(monitor):
    """REGRESSION: a cold instance must not 500 on /monitoring/disagreement."""
    schemas.DisagreementStatsResponse(**monitor.get_stats())


def test_populated_window_validates_too(monitor):
    monitor.record("POSITIVE", "NEGATIVE", 0.91, 0.72, 40)
    monitor.record("POSITIVE", "POSITIVE", 0.95, 0.93, 30)
    schemas.DisagreementStatsResponse(**monitor.get_stats())


def test_both_branches_return_the_same_keys(monitor):
    """
    The bug was structural, not a typo: two return statements drifted apart.
    Comparing the key sets directly is what stops that happening again.
    """
    empty = set(monitor.get_stats())
    monitor.record("POSITIVE", "POSITIVE", 0.9, 0.9, 10)
    populated = set(monitor.get_stats())
    assert empty == populated


def test_counts_agreements_and_disagreements(monitor):
    monitor.record("POSITIVE", "NEGATIVE", 0.9, 0.6, 20)
    monitor.record("POSITIVE", "POSITIVE", 0.9, 0.88, 20)
    monitor.record("NEGATIVE", "NEGATIVE", 0.8, 0.79, 20)

    stats = monitor.get_stats()
    assert stats["total_comparisons"] == 3
    assert stats["disagreements_in_window"] == 1
    assert stats["agreements_in_window"] == 2
    assert stats["disagreement_rate"] == round(1 / 3, 4)


def test_direction_breakdown_records_which_way_they_split(monitor):
    monitor.record("POSITIVE", "NEGATIVE", 0.9, 0.6, 20)
    monitor.record("POSITIVE", "NEGATIVE", 0.8, 0.5, 20)
    monitor.record("NEGATIVE", "POSITIVE", 0.7, 0.6, 20)

    breakdown = monitor.get_stats()["direction_breakdown"]
    assert breakdown["POSITIVE_to_NEGATIVE"] == 2
    assert breakdown["NEGATIVE_to_POSITIVE"] == 1


def test_recent_disagreements_excludes_agreements(monitor):
    monitor.record("POSITIVE", "POSITIVE", 0.9, 0.9, 20)
    monitor.record("POSITIVE", "NEGATIVE", 0.9, 0.4, 20)

    recent = monitor.get_recent_disagreements(10)
    assert len(recent) == 1
    assert recent[0]["v1_label"] != recent[0]["v2_label"]


def test_recent_comparisons_includes_agreements(monitor):
    """
    v2 is v1 quantized, so the two agree almost always and the
    disagreements-only view stays empty. The comparisons view is what gives the
    panel a signal to draw.
    """
    monitor.record("POSITIVE", "POSITIVE", 0.9, 0.88, 20)
    monitor.record("POSITIVE", "NEGATIVE", 0.9, 0.4, 20)

    comparisons = monitor.get_recent_comparisons(10)
    assert len(comparisons) == 2
    assert [c["agrees"] for c in comparisons] == [True, False]
    # The gap is what gets plotted, so it must survive rounding.
    assert comparisons[0]["confidence_gap"] > 0


def test_recent_comparisons_respects_n(monitor):
    for i in range(10):
        monitor.record("POSITIVE", "POSITIVE", 0.9, 0.9 - i / 100, 20)
    assert len(monitor.get_recent_comparisons(4)) == 4


def test_window_is_bounded(monitor):
    """The rolling window must not grow without limit on a long-lived process."""
    limit = monitor._window_size
    for _ in range(limit + 50):
        monitor.record("POSITIVE", "POSITIVE", 0.9, 0.9, 20)

    stats = monitor.get_stats()
    assert stats["window_size"] == limit
    assert stats["total_comparisons"] == limit + 50


def test_alert_needs_both_a_high_rate_and_enough_samples(monitor):
    """One disagreement out of one is a 100% rate and means nothing."""
    monitor.record("POSITIVE", "NEGATIVE", 0.9, 0.4, 20)
    assert monitor.get_stats()["alert_active"] is False


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_drift_status_validates_before_any_check():
    schemas.DriftStatusResponse(**DriftDetector().get_status())


def test_reference_window_freezes_then_checks_run():
    d = DriftDetector()
    ref_size = d._reference_size
    detection_size = d._detection_size

    for _ in range(ref_size):
        d.record(text_length=50, confidence=0.9)
    assert d.get_status()["reference_window_frozen"] is True

    for _ in range(detection_size):
        d.record(text_length=50, confidence=0.9)

    status = d.get_status()
    assert status["checks_run"] >= 1
    assert status["last_check"] is not None
    schemas.DriftStatusResponse(**status)


def test_identical_distributions_do_not_drift():
    d = DriftDetector()
    for _ in range(d._reference_size + d._detection_size):
        d.record(text_length=50, confidence=0.9)

    last = d.get_status()["last_check"]
    assert last["any_drift"] is False


def test_shifted_distribution_is_detected():
    d = DriftDetector()
    for i in range(d._reference_size):
        d.record(text_length=20 + i % 5, confidence=0.95)
    # A very different input profile should move the score off zero.
    for i in range(d._detection_size):
        d.record(text_length=800 + i % 5, confidence=0.55)

    last = d.get_status()["last_check"]
    assert last["text_length_drift_score"] > 0
    assert last["method"] in ("evidently", "jensen_shannon")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_falls_back_in_process_when_no_redis(cache):
    """
    An empty REDIS_HOST means there is no Redis here, not that Redis is down.
    The feature must keep working, and must say which store answered.
    """
    assert cache.backend == "in_process"
    assert cache.stats()["available"] is True


def test_cache_round_trip_and_stats(cache):
    assert cache.get("hello", "v1") is None
    cache.set("hello", "v1", {"label": "POSITIVE", "score": 0.9})
    assert cache.get("hello", "v1")["label"] == "POSITIVE"

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    schemas.CacheStatsResponse(**stats)


def test_cache_flush_empties_it(cache):
    cache.set("a", "v1", {"label": "POSITIVE", "score": 0.9})
    cache.set("b", "v1", {"label": "NEGATIVE", "score": 0.8})
    assert cache.flush() >= 2
    assert cache.get("a", "v1") is None


def test_cache_never_raises_when_the_backend_is_broken(cache):
    """
    Reads may degrade; they may not take the request down with them. A cache
    that raises turns a slow path into a 500.
    """

    class Exploding:
        def get(self, *a, **k):
            raise RuntimeError("backend gone")

        def setex(self, *a, **k):
            raise RuntimeError("backend gone")

        def keys(self, *a, **k):
            raise RuntimeError("backend gone")

        def delete(self, *a, **k):
            raise RuntimeError("backend gone")

    cache._client = Exploding()
    assert cache.get("anything", "v1") is None
    cache.set("anything", "v1", {"label": "POSITIVE", "score": 0.9})
    assert cache.stats()["errors"] >= 1
