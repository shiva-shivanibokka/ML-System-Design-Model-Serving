"""
Pydantic v2 request/response schemas for all API endpoints.

Schema design principles:
  - Every field has a description (powers the auto-generated OpenAPI docs)
  - Response schemas include all metadata needed for debugging and monitoring
  - Strict typing — no Optional fields that could silently drop data
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Input schema for POST /predict"""

    text: str = Field(
        ...,
        description="Text to classify. Sentiment analysis: POSITIVE or NEGATIVE.",
        min_length=1,
        max_length=2000,
        examples=["This product is absolutely amazing, I highly recommend it!"],
    )

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        # min_length is checked before this validator runs, so "   " satisfies
        # min_length=1 and then strips to "". Without this the model is asked
        # to classify an empty string and the caller gets a confident answer
        # about nothing.
        stripped = v.strip()
        if not stripped:
            raise ValueError("text must contain at least one non-whitespace character")
        return stripped


class PredictResponse(BaseModel):
    """Output schema for POST /predict"""

    # Core prediction
    label: str = Field(
        ...,
        description="Predicted sentiment label: POSITIVE or NEGATIVE.",
    )
    score: float = Field(
        ...,
        description="Model confidence score in [0.0, 1.0] for the predicted label.",
        ge=0.0,
        le=1.0,
    )

    # Routing metadata
    model_version: str = Field(
        ...,
        description="Which model version served this request: v1 or v2.",
    )
    model_used: str = Field(
        ...,
        description=(
            "Routing decision: v1 | v2 | v1_cb_fallback | v1_error_fallback. "
            "v1_cb_fallback means circuit breaker was open and v1 was used."
        ),
    )
    deployment_state: str = Field(
        ...,
        description="Deployment state at time of request: shadow | canary_5 | etc.",
    )

    # Performance
    latency_ms: float = Field(
        ...,
        description="Model inference latency in milliseconds (excludes cache lookup).",
    )
    cache_hit: bool = Field(
        ...,
        description="True if this response was served from Redis cache.",
    )

    # Tracing
    trace_id: str = Field(
        ...,
        description="Unique request trace ID (X-Trace-ID header). For log correlation.",
    )


# ---------------------------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for GET /health — liveness probe."""

    status: str = Field(
        ...,
        description="ok | degraded | down",
    )
    uptime_seconds: float = Field(
        ...,
        description="Seconds since the API process started.",
    )
    redis_available: bool = Field(
        ...,
        description=(
            "True only if Redis itself is reachable. The in-process fallback "
            "does not set this — see cache_backend for what is actually serving."
        ),
    )
    cache_backend: str = Field(
        default="none",
        description="Which cache answered: redis | in_process | none.",
    )
    models_stage: str = Field(
        default="ready",
        description=(
            "Model loader stage. /health is liveness only and stays 200 while "
            "models load, so this says whether inference is available yet."
        ),
    )
    deployment_state: str = Field(
        ...,
        description="Current deployment state.",
    )


class ReadinessResponse(BaseModel):
    """Response for GET /ready — readiness probe (Kubernetes)."""

    ready: bool = Field(
        ...,
        description=(
            "True only after all models have loaded AND completed warm-up. "
            "The service accepts no traffic until this is True."
        ),
    )
    v1_ready: bool = Field(
        ...,
        description="True if model v1 has completed warm-up.",
    )
    v2_ready: bool = Field(
        ...,
        description="True if model v2 has completed warm-up.",
    )
    warmup_details: dict[str, dict] = Field(
        default_factory=dict,
        description="Per-model warm-up results (first vs last inference latency).",
    )
    stage: str = Field(
        default="ready",
        description=(
            "Where the loader is: not_started | loading_v1 | warming_v1 | "
            "loading_v2 | warming_v2 | ready | failed."
        ),
    )
    load_elapsed_seconds: float = Field(
        default=0.0,
        description="Seconds since the loader started. Lets a caller show progress.",
    )
    load_error: str | None = Field(
        default=None,
        description="Exception from the loader, if stage is 'failed'.",
    )


# ---------------------------------------------------------------------------
# Deployment control
# ---------------------------------------------------------------------------


class DeploymentStatusResponse(BaseModel):
    """Response for GET /deployment/status"""

    state: str
    v2_traffic_fraction: float
    v2_requests: int
    v2_errors: int
    v2_error_rate: float
    v2_p99_latency_ms: float
    v1_p99_latency_ms: float
    total_requests: int
    time_in_state_seconds: float
    auto_progression_enabled: bool
    # "disk" when state survives a restart, "ephemeral" when it lives only for
    # the life of this instance. Stated rather than assumed.
    state_durability: str = "disk"
    rollback_thresholds: dict
    circuit_breaker: dict


class PromoteResponse(BaseModel):
    """Response for POST /deployment/promote"""

    ok: bool
    from_state: str | None = None
    to_state: str | None = None
    reason: str | None = None


class RollbackResponse(BaseModel):
    """Response for POST /deployment/rollback"""

    ok: bool
    from_state: str | None = None
    to_state: str | None = None
    reason: str | None = None
    cache_flushed: bool = False


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------


class DisagreementStatsResponse(BaseModel):
    """Response for GET /monitoring/disagreement"""

    total_comparisons: int
    window_size: int
    disagreements_in_window: int
    agreements_in_window: int
    disagreement_rate: float
    alert_active: bool
    alert_threshold: float
    direction_breakdown: dict[str, int]
    avg_confidence_gap_all: float
    avg_confidence_gap_on_disagreements: float


class DriftStatusResponse(BaseModel):
    """Response for GET /monitoring/drift"""

    reference_window_frozen: bool
    reference_size: int
    total_records: int
    checks_run: int
    last_check: dict | None = None
    thresholds: dict


class AuditLogResponse(BaseModel):
    """Response for GET /deployment/audit"""

    entries: list[dict]
    total_entries: int


class CacheStatsResponse(BaseModel):
    """Response for GET /monitoring/cache"""

    hits: int
    misses: int
    errors: int
    hit_rate: float
    available: bool
    backend: str = "none"  # "redis" | "in_process" | "none"
