"""
FastAPI application — the central serving layer.

Lifecycle (FastAPI lifespan):
  1. Configure structlog (JSON logging)
  2. Start the model loader on a background thread (v1 → warm-up → v2 → warm-up).
     Startup does not block on it, so the port binds in under a second and the
     load is observable through /ready rather than hidden inside a dead socket.
  3. Connect the prediction cache (Redis, else in-process)
  4. Load deployment state (from disk, or in-memory when EPHEMERAL_STATE=1)
  5. Start auto-progression background thread
  6. Inject models into router
  → Server begins accepting traffic; /predict returns 503 until /ready is 200

  On shutdown:
  → Stop auto-progression thread
  → Log shutdown event

Endpoints:
  POST /predict                   — main inference endpoint
  GET  /health                    — liveness probe (always returns 200 if process alive)
  GET  /ready                     — readiness probe (200 only after warm-up complete)
  GET  /deployment/status         — deployment state + metrics
  POST /deployment/promote        — manually advance to next stage
  POST /deployment/rollback       — manually roll back to shadow
  GET  /deployment/audit          — full audit log of state transitions
  GET  /monitoring/disagreement   — shadow mode v1/v2 disagreement stats
  GET  /monitoring/drift          — Evidently drift detection status
  GET  /monitoring/cache          — Redis cache hit/miss stats
  GET  /circuit-breaker/status    — circuit breaker state
  POST /circuit-breaker/reset     — manually reset circuit breaker
  GET  /metrics                   — Prometheus metrics (scraped by prometheus container)
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from api.middleware import TraceIDMiddleware, configure_structlog
from api.schemas import (
    AuditLogResponse,
    CacheStatsResponse,
    DeploymentStatusResponse,
    DisagreementStatsResponse,
    DriftStatusResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PromoteResponse,
    ReadinessResponse,
    RollbackResponse,
)
from cache.redis_cache import PredictionCache
from configs.settings import settings
from deployment.circuit_breaker import circuit_breaker
from deployment.router import router as request_router
from deployment.state_machine import state_machine
from models.model_v1 import ModelV1
from models.model_v2 import ModelV2
from monitoring.disagreement import disagreement_monitor
from monitoring.drift import drift_detector
from monitoring.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    INFERENCE_ERRORS,
    INFERENCE_LATENCY,
    INFERENCE_REQUESTS,
    MODEL_READY,
    MODEL_WARMUP_LATENCY,
    update_circuit_breaker_gauge,
    update_deployment_gauges,
)

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (shared across all requests)
# ---------------------------------------------------------------------------
model_v1 = ModelV1()
model_v2 = ModelV2()
cache = PredictionCache()

# Warm-up results stored for /ready endpoint
_warmup_results: dict = {}
_startup_time: float = 0.0

# Loading two transformer models and warming them takes tens of seconds. Doing
# that inside lifespan means the socket is not bound until it finishes, so on a
# scale-to-zero runtime the first visitor after an idle period stares at a blank
# tab for the whole load with no way to tell a slow start from a dead service.
#
# Loading on a background thread binds the port immediately and turns that wait
# into something observable: /health answers at once, /ready keeps returning 503
# with the current stage until warm-up completes. That is what a readiness probe
# is for — this just lets the probe do its job.
_load_stage: str = "not_started"
_load_error: str | None = None
_load_started_at: float = 0.0
LOAD_STAGES = (
    "not_started",
    "loading_v1",
    "warming_v1",
    "loading_v2",
    "warming_v2",
    "ready",
    "failed",
)


def _load_models() -> None:
    """Load and warm both models. Runs on a background thread."""
    global _load_stage, _load_error

    try:
        # ── Model v1 ────────────────────────────────────────────────────────
        _load_stage = "loading_v1"
        log.info("loading_v1")
        model_v1.load()

        _load_stage = "warming_v1"
        warmup_v1 = model_v1.warmup(
            dummy_text=settings.models.warmup.dummy_text,
            num_requests=settings.models.warmup.num_requests,
        )
        _warmup_results["v1"] = {
            "first_latency_ms": warmup_v1.first_latency_ms,
            "last_latency_ms": warmup_v1.last_latency_ms,
            "speedup_ratio": warmup_v1.speedup_ratio,
        }
        MODEL_READY.labels(model_version="v1").set(1)
        MODEL_WARMUP_LATENCY.labels(model_version="v1", pass_number="first").set(
            warmup_v1.first_latency_ms / 1000.0
        )
        MODEL_WARMUP_LATENCY.labels(model_version="v1", pass_number="last").set(
            warmup_v1.last_latency_ms / 1000.0
        )
        log.info(
            "v1_ready",
            first_latency_ms=warmup_v1.first_latency_ms,
            last_latency_ms=warmup_v1.last_latency_ms,
            speedup_ratio=warmup_v1.speedup_ratio,
        )

        # ── Model v2 ────────────────────────────────────────────────────────
        _load_stage = "loading_v2"
        log.info("loading_v2")
        model_v2.load()

        _load_stage = "warming_v2"
        warmup_v2 = model_v2.warmup(
            dummy_text=settings.models.warmup.dummy_text,
            num_requests=settings.models.warmup.num_requests,
        )
        _warmup_results["v2"] = {
            "first_latency_ms": warmup_v2.first_latency_ms,
            "last_latency_ms": warmup_v2.last_latency_ms,
            "speedup_ratio": warmup_v2.speedup_ratio,
        }
        MODEL_READY.labels(model_version="v2").set(1)
        MODEL_WARMUP_LATENCY.labels(model_version="v2", pass_number="first").set(
            warmup_v2.first_latency_ms / 1000.0
        )
        MODEL_WARMUP_LATENCY.labels(model_version="v2", pass_number="last").set(
            warmup_v2.last_latency_ms / 1000.0
        )
        log.info(
            "v2_ready",
            first_latency_ms=warmup_v2.first_latency_ms,
            last_latency_ms=warmup_v2.last_latency_ms,
            speedup_ratio=warmup_v2.speedup_ratio,
            quantized=True,
        )

        _load_stage = "ready"
        log.info(
            "models_ready",
            elapsed_seconds=round(time.monotonic() - _load_started_at, 1),
        )
    except Exception as e:
        # A failed load must be loud. /ready reports the exception instead of
        # holding at 503 forever, which would read identically to a slow start.
        _load_stage = "failed"
        _load_error = f"{type(e).__name__}: {e}"
        log.error("model_load_failed", error=_load_error, exc_info=True)


# ---------------------------------------------------------------------------
# Lifespan (replaces @app.on_event deprecated pattern)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before yield = startup. Everything after = shutdown.
    """
    global _startup_time, _load_started_at

    configure_structlog()
    _startup_time = time.monotonic()
    _load_started_at = _startup_time
    log.info("startup_begin", service="ml-model-serving")

    # ── Models (background) ─────────────────────────────────────────────────
    # Kicked off here, awaited by nobody. /ready is the gate on model traffic.
    threading.Thread(target=_load_models, daemon=True, name="model-loader").start()

    # ── Redis cache ──────────────────────────────────────────────────────────
    cache.connect()

    # Postgres was connected here and then never written to — the audit trail
    # has always lived in memory plus a JSONL file. Keeping an engine open for
    # a database nothing writes to bought a dependency, a cold-start connect,
    # and a README claim that was not true. Removed rather than implemented:
    # the deployed service runs with no attached services at all, so a
    # Postgres audit sink could never be exercised in the demo anyway.

    # ── Deployment state ────────────────────────────────────────────────────
    state_machine.load_persisted_state()
    state_machine.start_auto_progression()
    update_deployment_gauges(
        state_machine.state.value,
        state_machine.v2_traffic_fraction,
    )
    update_circuit_breaker_gauge(circuit_breaker.state.value)

    # ── Inject models into router ────────────────────────────────────────────
    request_router.set_models(model_v1, model_v2)

    log.info(
        "startup_complete",
        deployment_state=state_machine.state.value,
        cache_backend=cache.backend,
        models=_load_stage,  # still loading — /ready is the gate
    )

    yield  # ── Server is live ──────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("shutdown_begin")
    state_machine.stop()
    log.info("shutdown_complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML System Design: Model Serving",
    description=(
        "Production-grade model serving with shadow mode, canary deployment, "
        "circuit breaker, Evidently drift detection, and disagreement monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add trace ID middleware
app.add_middleware(TraceIDMiddleware)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ---------------------------------------------------------------------------
# POST /predict — main inference endpoint
# ---------------------------------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify sentiment of input text",
    tags=["Inference"],
)
async def predict(request_body: PredictRequest, request: Request) -> PredictResponse:
    """
    Classify the sentiment of input text.

    Routing is determined by the current deployment state:
    - **shadow**: v1 serves user, v2 runs silently for comparison
    - **canary**: weighted random routing (5/25/50% to v2)
    - **full**: all traffic to v2 (v1 fallback if circuit opens)
    - **rolled_back**: all traffic to v1

    Cache: identical inputs return cached results in <5ms.
    """
    if not model_v1.is_ready or not model_v2.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Models not ready (stage: {_load_stage}"
                + (f", error: {_load_error}" if _load_error else "")
                + "). Poll /ready."
            ),
        )

    trace_id = getattr(request.state, "trace_id", "unknown")
    text = request_body.text
    deployment_state = state_machine.state.value

    # ── Cache check ───────────────────────────────────────────────────────
    # Determine which model version will likely serve (for cache key)
    # In shadow/rolled_back/canary: primary is v1; in full: primary is v2
    primary_version = "v2" if deployment_state == "full" else "v1"
    cached = cache.get(text, primary_version)
    if cached:
        CACHE_HITS.labels(model_version=primary_version).inc()
        INFERENCE_REQUESTS.labels(
            model_version=f"{primary_version}_cached",
            deployment_state=deployment_state,
        ).inc()
        return PredictResponse(
            label=cached["label"],
            score=cached["score"],
            model_version=cached["model_version"],
            model_used=cached.get("model_used", primary_version),
            deployment_state=deployment_state,
            latency_ms=0.0,
            cache_hit=True,
            trace_id=trace_id,
        )
    CACHE_MISSES.labels(model_version=primary_version).inc()

    # ── Route request ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        user_result, shadow_v2_result, model_used = await request_router.route(
            text=text,
            trace_id=trace_id,
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(
            model_version="router",
            error_type=type(e).__name__,
        ).inc()
        log.error("predict_router_error", error=str(e), trace_id=trace_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}",
        ) from e
    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    # ── Prometheus metrics ────────────────────────────────────────────────
    INFERENCE_LATENCY.labels(
        model_version=user_result.model_version,
        deployment_state=deployment_state,
        cache_hit="false",
    ).observe(user_result.latency_ms / 1000.0)

    INFERENCE_REQUESTS.labels(
        model_version=model_used,
        deployment_state=deployment_state,
    ).inc()

    # ── Disagreement monitoring (shadow mode) ─────────────────────────────
    if shadow_v2_result is not None:
        disagreement_monitor.record(
            v1_label=user_result.label,
            v2_label=shadow_v2_result.label,
            v1_score=user_result.score,
            v2_score=shadow_v2_result.score,
            input_length=user_result.input_length,
        )

    # ── Drift detection (track input distribution) ─────────────────────────
    drift_detector.record(
        text_length=user_result.input_length,
        confidence=user_result.score,
    )

    # ── Cache write ───────────────────────────────────────────────────────
    cache.set(
        text,
        user_result.model_version,
        {
            "label": user_result.label,
            "score": user_result.score,
            "model_version": user_result.model_version,
            "model_used": model_used,
        },
    )

    # ── Update circuit breaker gauge ──────────────────────────────────────
    update_circuit_breaker_gauge(circuit_breaker.state.value)
    update_deployment_gauges(
        state_machine.state.value,
        state_machine.v2_traffic_fraction,
    )

    log.info(
        "predict_complete",
        label=user_result.label,
        score=round(user_result.score, 4),
        model_used=model_used,
        deployment_state=deployment_state,
        latency_ms=round(user_result.latency_ms, 1),
        total_latency_ms=round(total_latency_ms, 1),
        trace_id=trace_id,
    )

    return PredictResponse(
        label=user_result.label,
        score=round(user_result.score, 4),
        model_version=user_result.model_version,
        model_used=model_used,
        deployment_state=deployment_state,
        latency_ms=round(user_result.latency_ms, 1),
        cache_hit=False,
        trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# GET /health — liveness probe
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """
    Liveness probe. Returns 200 if the process is alive.
    Kubernetes kills and restarts the pod if this returns non-200.
    Does NOT check model readiness — use /ready for that.
    """
    return HealthResponse(
        status="ok" if _load_stage == "ready" else "degraded",
        uptime_seconds=round(time.monotonic() - _startup_time, 1),
        # Reports Redis specifically, not "some cache is up" — the in-process
        # fallback keeps the feature alive but is not Redis, and conflating the
        # two would hide a real outage behind a working demo.
        redis_available=cache.backend == "redis",
        cache_backend=cache.backend,
        models_stage=_load_stage,
        deployment_state=state_machine.state.value,
    )


# ---------------------------------------------------------------------------
# GET /ready — readiness probe
# ---------------------------------------------------------------------------


@app.get(
    "/ready",
    summary="Readiness probe",
    tags=["Health"],
)
async def ready() -> JSONResponse:
    """
    Readiness probe. Returns 200 only after both models have completed warm-up.

    Kubernetes will NOT send traffic to this pod until this returns 200.
    This prevents the cold JIT-compilation penalty from hitting real users.

    Warm-up effect: first inference = 200-500ms (JIT compile + cache cold)
                    after warm-up  = 30-80ms   (hot path)

    While models are still loading this returns 503 with the current stage and
    elapsed seconds, so a caller can show progress instead of guessing whether
    the service is slow or broken. A load that raises returns stage "failed"
    with the exception rather than holding at 503 indefinitely.
    """
    v1_ready = model_v1.is_ready
    v2_ready = model_v2.is_ready
    both_ready = v1_ready and v2_ready

    body = ReadinessResponse(
        ready=both_ready,
        v1_ready=v1_ready,
        v2_ready=v2_ready,
        warmup_details=_warmup_results,
        stage=_load_stage,
        load_elapsed_seconds=(
            round(time.monotonic() - _load_started_at, 1) if _load_started_at else 0.0
        ),
        load_error=_load_error,
    )

    return JSONResponse(
        content=body.model_dump(),
        status_code=200 if both_ready else 503,
    )


# ---------------------------------------------------------------------------
# Deployment control endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/deployment/status",
    response_model=DeploymentStatusResponse,
    summary="Current deployment state and per-version metrics",
    tags=["Deployment"],
)
async def deployment_status() -> DeploymentStatusResponse:
    """
    Returns the full deployment state snapshot including:
    - Current state (shadow / canary_5 / etc.)
    - v2 traffic fraction
    - v2 error rate and p99 latency (used for rollback decisions)
    - Circuit breaker state
    """
    sm_status = state_machine.get_status()
    cb_status = circuit_breaker.get_status()
    return DeploymentStatusResponse(
        **sm_status,
        circuit_breaker=cb_status,
    )


@app.post(
    "/deployment/promote",
    response_model=PromoteResponse,
    summary="Manually promote to next deployment stage",
    tags=["Deployment"],
)
async def promote() -> PromoteResponse:
    """
    Manually advance deployment state: shadow → canary_5 → canary_25 → canary_50 → full.
    Auto-progression also does this automatically on a timer (configurable).
    """
    result = state_machine.promote(triggered_by="manual")
    update_deployment_gauges(
        state_machine.state.value,
        state_machine.v2_traffic_fraction,
    )
    return PromoteResponse(**result)


@app.post(
    "/deployment/rollback",
    response_model=RollbackResponse,
    summary="Manually roll back to shadow mode",
    tags=["Deployment"],
)
async def rollback() -> RollbackResponse:
    """
    Immediately roll back v2 to shadow mode.
    All traffic returns to v1. Redis cache is flushed to remove any v2 predictions.
    """
    result = state_machine.rollback(trigger="manual_rollback", note="Triggered via API")
    # Only flush when a rollback actually happened. Rolling back while already
    # in shadow is a no-op on the state machine, and emptying the cache anyway
    # would throw away every valid v1 answer to no purpose.
    flushed = cache.flush() if result.get("ok") else 0
    update_deployment_gauges(
        state_machine.state.value,
        state_machine.v2_traffic_fraction,
    )
    return RollbackResponse(**result, cache_flushed=flushed > 0)


@app.get(
    "/deployment/audit",
    response_model=AuditLogResponse,
    summary="Deployment state transition audit log",
    tags=["Deployment"],
)
async def audit_log() -> AuditLogResponse:
    """
    Returns all deployment state transitions with timestamps, triggers,
    and per-version metrics at the time of each event.

    Every auto-rollback, manual promotion, and auto-progression is logged here.
    """
    entries = state_machine.get_audit_log()
    return AuditLogResponse(entries=entries, total_entries=len(entries))


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/monitoring/disagreement",
    response_model=DisagreementStatsResponse,
    summary="Shadow mode v1 vs v2 disagreement rate",
    tags=["Monitoring"],
)
async def disagreement_stats() -> DisagreementStatsResponse:
    """
    In shadow mode, v1 and v2 both run on every request.
    This endpoint shows how often they disagree on the predicted label.

    High disagreement (>30%) before promoting to canary is a red flag.
    """
    stats = disagreement_monitor.get_stats()
    return DisagreementStatsResponse(**stats)


@app.get(
    "/monitoring/drift",
    response_model=DriftStatusResponse,
    summary="Evidently input distribution drift detection status",
    tags=["Monitoring"],
)
async def drift_status() -> DriftStatusResponse:
    """
    Evidently AI drift detection status.
    Compares the current input distribution (text lengths, confidence scores)
    against the reference window (first N requests).

    Drift signals: text_length drift (different types of inputs) and
    confidence_score drift (model calibration change).
    """
    status_data = drift_detector.get_status()
    return DriftStatusResponse(**status_data)


@app.get(
    "/monitoring/cache",
    response_model=CacheStatsResponse,
    summary="Redis cache performance stats",
    tags=["Monitoring"],
)
async def cache_stats() -> CacheStatsResponse:
    """Cache hit rate and error counts since server start."""
    stats = cache.stats()
    return CacheStatsResponse(**stats)


@app.get(
    "/monitoring/disagreement/recent",
    summary="Recent v1 vs v2 disagreement cases",
    tags=["Monitoring"],
)
async def recent_disagreements(n: int = 20) -> dict:
    """
    Returns the N most recent cases where v1 and v2 predicted different labels.
    Useful for manual inspection: are the disagreements on borderline cases
    or are they systematic errors?
    """
    return {
        "recent_disagreements": disagreement_monitor.get_recent_disagreements(n),
        "n_requested": n,
    }


@app.get(
    "/monitoring/disagreement/comparisons",
    summary="Recent v1 vs v2 comparisons, agreements included",
    tags=["Monitoring"],
)
async def recent_comparisons(n: int = 40) -> dict:
    """
    The N most recent shadow comparisons whether or not the labels matched.

    v2 is v1 quantized to int8, so the two agree on essentially all ordinary
    input and the disagreements-only list stays empty. The confidence gap is
    non-zero on every comparison, so this is the series that actually shows
    what quantization does to the model's answers.
    """
    return {
        "comparisons": disagreement_monitor.get_recent_comparisons(n),
        "n_requested": n,
    }


@app.get(
    "/monitoring/drift/history",
    summary="Full drift detection history",
    tags=["Monitoring"],
)
async def drift_history() -> dict:
    """All historical drift check results since server start."""
    return {"history": drift_detector.get_history()}


# ---------------------------------------------------------------------------
# Circuit breaker endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/circuit-breaker/status",
    summary="Circuit breaker state and counters",
    tags=["Circuit Breaker"],
)
async def cb_status() -> dict:
    """
    Returns circuit breaker state: closed | open | half_open.
    If open, v2 is blocked and all traffic falls back to v1.
    """
    return circuit_breaker.get_status()


@app.post(
    "/circuit-breaker/reset",
    summary="Manually reset circuit breaker to CLOSED",
    tags=["Circuit Breaker"],
)
async def cb_reset() -> dict:
    """
    Manually close the circuit breaker after a v2 fix.
    Only use this after confirming the underlying v2 issue is resolved.
    """
    circuit_breaker.reset()
    update_circuit_breaker_gauge(circuit_breaker.state.value)
    return {"ok": True, "state": circuit_breaker.state.value}


# ---------------------------------------------------------------------------
# Control panel (same process, same port)
# ---------------------------------------------------------------------------
# The panel is a static page that talks to this API over HTTP, mounted into
# this app so the whole demo is one container behind one URL: no second
# service to deploy, no CORS, no cross-origin URL to keep in sync, and nothing
# extra that can expire.
#
# It replaced a Gradio app, which is why MOUNT_UI still exists — compose runs
# the API without a panel, and API-only remains a valid way to run this.
#
# StaticFiles(html=True) serves index.html for "/ui/" itself. Assets are
# referenced relatively from that page, so nothing is emitted root-absolute
# and there is no redirect layer to maintain.
if os.getenv("MOUNT_UI", "1") == "1":
    try:
        from fastapi.responses import RedirectResponse
        from fastapi.staticfiles import StaticFiles

        _web_dir = Path(__file__).resolve().parent.parent / "web"
        if not (_web_dir / "index.html").exists():
            raise FileNotFoundError(f"no index.html under {_web_dir}")

        app.mount("/ui", StaticFiles(directory=str(_web_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def _root() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

        log.info("ui_mounted", path="/ui", directory=str(_web_dir))
    except Exception as e:
        # API-only is a valid way to run this; a missing panel must not take
        # the service down with it.
        log.warning("ui_mount_failed", error=str(e), fallback="api_only")
