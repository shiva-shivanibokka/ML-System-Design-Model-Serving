"""
Request middleware: OpenTelemetry-style trace IDs + per-request structured logging.

Trace ID pattern:
  Every inbound request gets a unique X-Trace-ID header.
  If the client provides one (e.g. an upstream service), we honour it.
  Otherwise we generate one (UUID4).

  The trace_id flows through:
    middleware → router → model inference → cache → response body → logs

  This means you can take any response body's trace_id, grep the logs,
  and see the full lifecycle of that request: which model ran, latency,
  deployment state, cache hit/miss, disagreement result.

  This is the foundation of distributed tracing — in production you'd
  ship these trace IDs to Jaeger or Zipkin. Here they flow through
  structlog JSON logs, which is the same principle.

Why structlog:
  structlog produces JSON logs where every field is a key-value pair.
  Unlike Python's logging module which produces unstructured strings,
  JSON logs can be:
    - Searched by field in CloudWatch, Datadog, Grafana Loki
    - Aggregated (count requests per deployment_state per minute)
    - Correlated (find all logs for trace_id=abc123)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

log = structlog.get_logger(__name__)

TRACE_ID_HEADER = "X-Trace-ID"


class TraceIDMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that:
    1. Reads or generates a trace ID for every request
    2. Binds the trace ID to the structlog context (flows into all log lines)
    3. Measures total request wall-clock time
    4. Adds X-Trace-ID and X-Request-Duration-Ms to response headers
    5. Logs a single request summary line at the end of every request
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or inherit trace ID
        trace_id = request.headers.get(TRACE_ID_HEADER) or str(uuid.uuid4())[:16]

        # Bind to structlog context — all log calls within this request
        # will automatically include trace_id=... without passing it around
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        # Store trace_id in request state so endpoint handlers can access it
        request.state.trace_id = trace_id

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            # Add trace ID and timing to response headers
            response.headers[TRACE_ID_HEADER] = trace_id
            response.headers["X-Request-Duration-Ms"] = f"{duration_ms:.1f}"

            log.info(
                "request_complete",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 1),
            )
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            log.error(
                "request_failed",
                error=str(exc),
                duration_ms=round(duration_ms, 1),
                exc_info=True,
            )
            raise
        finally:
            structlog.contextvars.clear_contextvars()


def configure_structlog() -> None:
    """
    Configure structlog for JSON output.
    Called once at application startup (in FastAPI lifespan).

    Output format:
        {"event": "request_complete", "trace_id": "a1b2c3d4",
         "method": "POST", "path": "/predict", "status_code": 200,
         "duration_ms": 45.2, "timestamp": "2025-01-15T10:23:45Z",
         "level": "info"}
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
