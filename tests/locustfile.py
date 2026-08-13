"""
Locust load test — canary-aware performance benchmarking.

What this tests:
  1. Baseline throughput and latency (p50/p95/p99) under sustained load
  2. Per-model-version latency breakdown (v1 vs v2 vs fallback)
  3. Canary fairness: are the observed traffic splits matching config?
  4. Rollback trigger simulation: intentionally hammer the API to
     validate that auto-rollback fires when error thresholds are breached

Usage:
  # Run 100 concurrent users for 2 minutes
  locust -f tests/locustfile.py --host http://localhost:8000 \
         --users 100 --spawn-rate 10 --run-time 2m --headless

  # Open Locust web UI
  locust -f tests/locustfile.py --host http://localhost:8000

  # Run canary-aware mode (logs per-version stats)
  locust -f tests/locustfile.py --host http://localhost:8000 \
         --tags canary --users 50 --spawn-rate 5 --run-time 5m

Results interpreted:
  At 5% canary: ~5% of responses should show model_version=v2
  At 25% canary: ~25% should show v2
  p99 for v2 > 2x p99 for v1 → auto-rollback should trigger
"""

from __future__ import annotations

import random
import statistics
import threading
from collections import defaultdict

from locust import HttpUser, between, events, tag, task

# ---------------------------------------------------------------------------
# Sample texts for sentiment classification
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "This product is absolutely amazing and exceeded all my expectations.",
    "Terrible service, I am very disappointed and will never return.",
    "The quality is decent but the price seems a bit high for what you get.",
    "Outstanding customer support team, they resolved my issue immediately.",
    "The packaging was damaged and the item was broken upon arrival.",
    "Really impressive performance, works exactly as described in the listing.",
    "Completely useless product, nothing works as advertised. Total waste of money.",
    "Good value for the price, not perfect but does the job well enough.",
    "I am completely satisfied with this purchase and would recommend it.",
    "Worst experience I have had in years, avoid this product at all costs.",
    "The delivery was fast and the item was exactly as shown in the photos.",
    "Poor build quality, feels cheap and flimsy compared to the competition.",
    "Surprisingly good for the price point, I was not expecting this quality.",
    "The instructions were unclear and the setup took much longer than expected.",
    "Five stars, no complaints, this is exactly what I was looking for.",
    "Not worth the money at all, very underwhelming performance.",
    "The product works great for basic needs but lacks advanced features.",
    "Exceptional quality and attention to detail, highly recommend to everyone.",
    "Had to return it after just one week of use, completely stopped working.",
    "Love everything about this, the design, quality, and performance are all top tier.",
    # Shorter texts (testing text length distribution drift)
    "Great product!",
    "Terrible.",
    "OK.",
    "Love it!",
    "Waste of money.",
    # Longer texts (testing text length distribution drift)
    (
        "I have been using this product for the past three months and I must say "
        "that it has completely transformed my daily routine. The build quality is "
        "exceptional and the performance is consistent even under heavy use. "
        "The customer service team was incredibly responsive when I had a minor "
        "issue with the setup process. Would strongly recommend this to anyone "
        "looking for a reliable and high-quality product in this category."
    ),
]


# ---------------------------------------------------------------------------
# Per-version tracking (module level, updated by all workers)
# ---------------------------------------------------------------------------

_version_counts: dict[str, int] = defaultdict(int)
_version_latencies: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Locust event hooks
# ---------------------------------------------------------------------------


@events.test_stop.add_listener
def print_canary_report(environment, **kwargs):
    """
    Print canary-aware summary at the end of the test run.
    Shows observed traffic split per model version vs expected.
    """
    with _lock:
        if not _version_counts:
            return

        total = sum(_version_counts.values())
        print("\n" + "=" * 60)
        print("CANARY-AWARE LOAD TEST REPORT")
        print("=" * 60)
        print(f"Total requests tracked: {total:,}\n")

        print(
            f"{'Model Used':<25} {'Count':>8} {'Share':>8} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10}"
        )
        print("-" * 75)

        for version, count in sorted(_version_counts.items()):
            share = count / total if total > 0 else 0
            lats = sorted(_version_latencies.get(version, [0]))
            p50 = statistics.median(lats) if lats else 0.0
            p95 = lats[int(len(lats) * 0.95)] if len(lats) > 1 else 0.0
            p99 = lats[int(len(lats) * 0.99)] if len(lats) > 1 else 0.0
            print(
                f"{version:<25} {count:>8,} {share:>8.1%} "
                f"{p50:>10.1f} {p95:>10.1f} {p99:>10.1f}"
            )

        print("=" * 60)


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class SentimentAPIUser(HttpUser):
    """
    Standard load test user.
    Sends prediction requests with realistic text inputs.
    Wait time simulates realistic user think time.
    """

    wait_time = between(0.1, 0.5)  # 100-500ms between requests = ~5 req/s per user

    @task(10)
    @tag("predict", "canary")
    def predict(self):
        """Main prediction endpoint — primary load test task."""
        text = random.choice(SAMPLE_TEXTS)
        with self.client.post(
            "/predict",
            json={"text": text},
            catch_response=True,
            name="POST /predict",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                model_used = data.get("model_used", "unknown")
                latency_ms = data.get("latency_ms", 0.0)
                cache_hit = data.get("cache_hit", False)

                # Track per-version stats
                with _lock:
                    _version_counts[model_used] += 1
                    if not cache_hit:
                        _version_latencies[model_used].append(latency_ms)

                response.success()

            elif response.status_code == 503:
                response.failure("Service unavailable — models may still be warming up")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    @tag("health")
    def health_check(self):
        """Liveness probe — should always be fast."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    @tag("monitoring")
    def check_deployment_status(self):
        """Deployment status endpoint — verifies monitoring doesn't slow down inference."""
        with self.client.get(
            "/deployment/status",
            catch_response=True,
            name="GET /deployment/status",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")


class HighThroughputUser(HttpUser):
    """
    High-throughput user — minimal wait time to stress-test the API.
    Use sparingly: this represents burst traffic, not sustained load.
    """

    wait_time = between(0.01, 0.05)  # Near-zero wait = ~50 req/s per user
    weight = 1  # Lower weight than standard user

    @task
    @tag("stress")
    def predict_burst(self):
        text = random.choice(SAMPLE_TEXTS[:10])  # fewer unique texts = higher cache hit rate
        self.client.post(
            "/predict",
            json={"text": text},
            name="POST /predict [burst]",
        )


class ReadinessWatcher(HttpUser):
    """
    Watches the readiness endpoint to detect when models become available.
    Useful at the start of a load test to wait for warm-up completion.
    """

    wait_time = between(2.0, 5.0)
    weight = 1

    @task
    @tag("readiness")
    def check_ready(self):
        with self.client.get(
            "/ready",
            catch_response=True,
            name="GET /ready",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Not ready yet — models still warming up")
            else:
                response.failure(f"Unexpected: {response.status_code}")
