"""
Prometheus metrics definitions.

All metrics are defined at module level (not inside request handlers).
This is the correct Prometheus pattern — metrics must be singletons
registered with the default registry exactly once.

Metric naming follows Prometheus conventions:
  - snake_case
  - Unit suffix (_seconds, _total, _bytes)
  - Namespace prefix: model_serving_

Histogram bucket strategy:
  Buckets are chosen to match the latency SLOs in config.yaml.
  p50=50ms, p95=200ms, p99=500ms.
  Fine-grained buckets below 100ms where most requests should land.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Inference latency (most important metric)
# ---------------------------------------------------------------------------
INFERENCE_LATENCY = Histogram(
    name="model_serving_inference_latency_seconds",
    documentation=(
        "End-to-end inference latency in seconds, labelled by model version "
        "and deployment state. Used to compute p50/p95/p99 per version."
    ),
    labelnames=["model_version", "deployment_state", "cache_hit"],
    buckets=[
        0.005,
        0.010,
        0.025,
        0.050,
        0.075,
        0.100,
        0.150,
        0.200,
        0.300,
        0.500,
        0.750,
        1.0,
        2.0,
        5.0,
    ],
)

# ---------------------------------------------------------------------------
# Request counts
# ---------------------------------------------------------------------------
INFERENCE_REQUESTS = Counter(
    name="model_serving_requests_total",
    documentation=(
        "Total prediction requests, labelled by model_version "
        "(v1 | v2 | v1_cb_fallback | v1_error_fallback) and deployment_state."
    ),
    labelnames=["model_version", "deployment_state"],
)

INFERENCE_ERRORS = Counter(
    name="model_serving_errors_total",
    documentation=(
        "Total inference errors (exceptions, timeouts, circuit breaker blocks). "
        "Labelled by model_version and error_type."
    ),
    labelnames=["model_version", "error_type"],
)

# ---------------------------------------------------------------------------
# Cache metrics
# ---------------------------------------------------------------------------
CACHE_HITS = Counter(
    name="model_serving_cache_hits_total",
    documentation="Redis cache hits — requests served without model inference.",
    labelnames=["model_version"],
)

CACHE_MISSES = Counter(
    name="model_serving_cache_misses_total",
    documentation="Redis cache misses — requests that required model inference.",
    labelnames=["model_version"],
)

# ---------------------------------------------------------------------------
# Deployment state machine
# ---------------------------------------------------------------------------
DEPLOYMENT_TRANSITIONS = Counter(
    name="model_serving_deployment_transitions_total",
    documentation=(
        "Count of deployment state transitions. "
        "Labelled by from_state, to_state, trigger "
        "(manual_promote | auto_promote | auto_rollback_error_rate | "
        "auto_rollback_latency | manual_rollback)."
    ),
    labelnames=["from_state", "to_state", "trigger"],
)

DEPLOYMENT_STATE = Gauge(
    name="model_serving_deployment_state",
    documentation=(
        "Current deployment state encoded as integer: "
        "shadow=0, canary_5=1, canary_25=2, canary_50=3, full=4, rolled_back=5. "
        "Useful for Grafana state timeline panels."
    ),
)

CANARY_V2_TRAFFIC_FRACTION = Gauge(
    name="model_serving_canary_v2_traffic_fraction",
    documentation="Current fraction of traffic routed to v2 (0.0 to 1.0).",
)

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
CIRCUIT_BREAKER_STATE = Gauge(
    name="model_serving_circuit_breaker_state",
    documentation=(
        "Circuit breaker state: closed=0, open=1, half_open=2. "
        "Alert if this stays at 1 for > 60 seconds."
    ),
)

CIRCUIT_BREAKER_BLOCKED = Counter(
    name="model_serving_circuit_breaker_blocked_total",
    documentation="Requests blocked by the open circuit breaker (fail-fast to v1).",
)

# ---------------------------------------------------------------------------
# Disagreement monitoring (shadow mode)
# ---------------------------------------------------------------------------
DISAGREEMENT_RATE = Gauge(
    name="model_serving_v1_v2_disagreement_rate",
    documentation=(
        "Rolling disagreement rate between v1 and v2 predictions in shadow mode. "
        "High values (>30%) indicate v2 is fundamentally different from v1."
    ),
)

DISAGREEMENT_EVENTS = Counter(
    name="model_serving_disagreement_events_total",
    documentation=(
        "Total cases where v1 and v2 predicted different labels in shadow mode. "
        "Labelled by v1_label and v2_label."
    ),
    labelnames=["v1_label", "v2_label"],
)

# ---------------------------------------------------------------------------
# Drift detection (Evidently)
# ---------------------------------------------------------------------------
DRIFT_DETECTED = Gauge(
    name="model_serving_drift_detected",
    documentation=(
        "1 if Evidently drift detection flagged input distribution drift "
        "in the last check window, 0 otherwise. "
        "Labelled by drift_type (text_length | confidence_score)."
    ),
    labelnames=["drift_type"],
)

DRIFT_SCORE = Gauge(
    name="model_serving_drift_score",
    documentation=(
        "Jensen-Shannon divergence score from last Evidently drift check. "
        "Values > threshold indicate drift."
    ),
    labelnames=["drift_type"],
)

# ---------------------------------------------------------------------------
# Model readiness
# ---------------------------------------------------------------------------
MODEL_READY = Gauge(
    name="model_serving_model_ready",
    documentation=(
        "1 if model has completed warm-up and is serving traffic, 0 otherwise. "
        "Labelled by model_version. Used by Kubernetes readiness probe."
    ),
    labelnames=["model_version"],
)

MODEL_WARMUP_LATENCY = Gauge(
    name="model_serving_warmup_latency_seconds",
    documentation=(
        "First inference latency during warm-up (cold JIT). "
        "Shows the penalty paid without warm-up."
    ),
    labelnames=["model_version", "pass_number"],
)

# ---------------------------------------------------------------------------
# State integer mapping (for DEPLOYMENT_STATE gauge)
# ---------------------------------------------------------------------------
DEPLOYMENT_STATE_INT = {
    "shadow": 0,
    "canary_5": 1,
    "canary_25": 2,
    "canary_50": 3,
    "full": 4,
    "rolled_back": 5,
}


def update_deployment_gauges(state_value: str, v2_fraction: float) -> None:
    """
    Update deployment-related gauges after every state transition.
    Called by DeploymentStateMachine._transition().
    """
    DEPLOYMENT_STATE.set(DEPLOYMENT_STATE_INT.get(state_value, -1))
    CANARY_V2_TRAFFIC_FRACTION.set(v2_fraction)


def update_circuit_breaker_gauge(state_value: str) -> None:
    """Update circuit breaker gauge after state change."""
    mapping = {"closed": 0, "open": 1, "half_open": 2}
    CIRCUIT_BREAKER_STATE.set(mapping.get(state_value, -1))
