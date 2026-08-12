# =============================================================================
# ML System Design: Model Serving — Dockerfile
# =============================================================================
# Single stage: Python 3.11 slim base, non-root, CPU-only torch, model weights
# baked in at build time.

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

# CPU-only torch, installed first and from PyTorch's own index.
# `pip install torch` on Linux resolves to the CUDA build and drags in ~2.5GB of
# nvidia-* wheels — all of it dead weight for a service that runs device=-1, and
# all of it image layers to pull on every cold start.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.0

# Remaining dependencies. torch==2.3.0 is already satisfied by 2.3.0+cpu above,
# so nothing here re-fetches it.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# HuggingFace cache directory (weights are baked into this path below)
ENV HF_HOME=/app/.cache/huggingface

# Download the model weights at build time, not at first request. v1 and v2 are
# the same checkpoint (v2 quantizes it after load), so this is one ~268MB
# download shared by both, and the container never needs the network to serve.
ARG MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
name='${MODEL_NAME}'; \
AutoTokenizer.from_pretrained(name); \
AutoModelForSequenceClassification.from_pretrained(name); \
print('weights cached')"

# Copy application source
COPY --chown=app:app . .

# Create data directories with correct ownership
RUN mkdir -p data/audit_log && chown -R app:app data/ && \
    chown -R app:app /app/.cache

USER app

# Default environment variables (overridden by docker-compose / Cloud Run)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
# No network calls at runtime — the weights are already in HF_HOME. This makes
# a missing layer fail loudly at start rather than silently re-downloading.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# Liveness only — /health answers as soon as the port binds, while models load
# in the background. /ready is the gate on inference.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Managed runtimes inject the port; default to 8000 for compose and local runs.
# One worker: the deployment state machine, circuit breaker, disagreement
# window and cache are all in-process, and a second worker would give each its
# own copy — two disagreeing control planes behind one URL.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
