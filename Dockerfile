# =============================================================================
# ML System Design: Model Serving — Dockerfile
# =============================================================================
# Multi-stage build is overkill for this service (no compiled assets).
# Single stage: Python 3.11 slim base, no root, non-blocking cache.

FROM python:3.11-slim

# Metadata
LABEL maintainer="ml-system-design"
LABEL description="Production model serving with canary deployment and circuit breaker"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=app:app . .

# Create data directories with correct ownership
RUN mkdir -p data/audit_log && chown -R app:app data/

USER app

# HuggingFace cache directory (models downloaded to this path)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

# Default environment variables (overridden by docker-compose)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 8000 7860

# Health check (calls our /health liveness endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
