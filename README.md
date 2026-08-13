# ML System Design: Production Model Serving

A production-grade model deployment system built around the question every ML engineering interview asks:

> **"You have a new model. How do you deploy it without breaking production?"**

This project implements the full answer: shadow mode validation, canary progressive delivery, circuit breaker failover, Evidently AI drift detection, disagreement rate monitoring, and a complete audit trail — all wired into a real FastAPI serving layer with Redis, Prometheus, Grafana, and Docker.

**Live demo: [model-serving on Cloud Run](https://model-serving-548930096299.us-central1.run.app)** — control panel at `/ui`, API docs at `/docs`.

---

## Running it hosted

The full stack is six containers. The hosted demo is one, and the difference is
deliberate — see [Hosted vs. local](#hosted-vs-local) for what changes and why.

Two things about the hosted instance are worth knowing before you click:

**It scales to zero, so the first request is a cold start.** Both models load
and warm up on a background thread while the API is already answering, so the
page and every monitoring tab come up immediately and a banner reports which
stage the loader is on. The first prediction waits for warm-up (roughly 20-40s
from cold) and then answers; everything after that is single-digit-millisecond
cached or ~10-30ms warm.

**Deployment state is in memory and resets when the instance is recycled.**
`GET /deployment/status` says so directly — `"state_durability": "ephemeral"`.
If you promote to canary_25 and come back an hour later, it will read `shadow`
again. This is a demo control plane; writing state to a container filesystem
that gets wiped would not have made it durable, only quietly inconsistent.

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
                         │         REQUEST PIPELINE                  │
                         │                                           │
  POST /predict          │  1. Redis cache check  (<5ms on hit)      │
       │                 │  2. Deployment state read                  │
       ▼                 │  3. Traffic routing decision               │
  TraceID Middleware      │  4. Model inference (v1 or v2)            │
  (X-Trace-ID)           │  5. Shadow logging (if SHADOW state)      │
       │                 │  6. Drift + disagreement monitoring        │
       ▼                 │  7. Cache write                            │
  Request Router ──────► │  8. Prometheus metrics update             │
       │                 └──────────────────────────────────────────┘
       ▼

 DEPLOYMENT STATE MACHINE
 ┌──────────┐   promote   ┌──────────┐   promote   ┌───────────┐
 │  SHADOW  │────────────►│ CANARY_5 │────────────►│ CANARY_25 │
 │ (0% v2)  │             │ (5% v2)  │             │ (25% v2)  │
 └──────────┘             └──────────┘             └───────────┘
       ▲                        │ auto-rollback           │
       │    rolled_back         │ (error_rate > 5%        │ promote
       │◄───────────────────────┤  OR p99 > 2x v1)        ▼
       │                        │                   ┌───────────┐
       │                        │     promote       │ CANARY_50 │
       │                        │                   │ (50% v2)  │
       │                        │                   └───────────┘
       │                        │                         │ promote
       │                        │                         ▼
       │                        │                    ┌──────────┐
       └────────────────────────┴───────────────────►│   FULL   │
                   rollback (any state)              │ (100% v2)│
                                                     └──────────┘

 CIRCUIT BREAKER (wraps all v2 calls)
 CLOSED → [N consecutive failures] → OPEN → [timeout_s] → HALF_OPEN → probe → CLOSED
                                      │                        │
                                   fail-fast               probe fails → OPEN
                                  return v1 (<1ms)
```

---

## What Makes This Different From a Standard FastAPI + Redis Project

| Production Problem | Solution Implemented |
|---|---|
| "How do you test v2 before users see it?" | Shadow mode: v2 runs silently on every request, zero user impact |
| "How do you roll out v2 safely?" | Canary state machine: 5% → 25% → 50% → 100%, auto-progression on timer |
| "What if v2 starts failing?" | Auto-rollback: triggers if v2 error_rate > 5% OR p99 > 2× v1 p99 |
| "What if v2 crashes entirely?" | Circuit breaker: OPEN state returns v1 in <1ms (fail-fast) |
| "Are v1 and v2 actually the same model?" | Disagreement rate: tracks % of shadow requests where labels differ |
| "Are canary inputs different from training?" | Evidently AI drift: Jensen-Shannon divergence on text length + confidence |
| "Why did this request take 800ms?" | Trace IDs: X-Trace-ID flows through every component and log line |
| "Why is v2 slow on the first request?" | Warm-up: 10 dummy inferences post-load, /ready only flips after warm-up |
| "What happened when v2 was rolled back?" | Audit log: every transition recorded with the metrics as they stood, in memory and appended to JSONL |

---

## Models

| Version | Architecture | Precision | Expected CPU Latency |
|---|---|---|---|
| v1 | DistilBERT SST-2 fine-tuned | FP32 (full precision) | 30-80ms |
| v2 | Same weights, INT8 dynamic quantization | INT8 (quantized) | 20-55ms (~30% faster) |

**Why INT8 quantization as v2?**
It's the most common real-world "v2" scenario: same model architecture, different serving optimization. INT8 produces slightly different confidence scores even when the label agrees — this makes shadow disagreement monitoring meaningful and demonstrates a real divergence between versions.

---

## Stack

| Component | Technology |
|---|---|
| Inference API | FastAPI (async) + Uvicorn |
| Models | HuggingFace Transformers (DistilBERT) + PyTorch INT8 quantization |
| Deployment state | In-memory state machine + Redis persistence |
| Traffic routing | Weighted random routing (deployment/router.py) |
| Circuit breaker | Custom implementation (deployment/circuit_breaker.py) |
| Drift detection | Evidently AI DataDriftPreset + scipy Jensen-Shannon fallback |
| Disagreement monitoring | Rolling window tracker (monitoring/disagreement.py) |
| Caching | Redis (hash-keyed, per-version, TTL=300s) |
| Metrics | Prometheus (prometheus_client) + Grafana dashboards |
| Structured logging | structlog (JSON, trace_id-bound) |
| Audit trail | In-memory ring buffer + JSONL append (no database) |
| Load testing | Locust (canary-aware per-version latency breakdown) |
| Containerization | Docker + docker-compose (5 services) |
| Control panel | Static HTML/CSS/JS, no build step, served by FastAPI at `/ui` |

---

## Quickstart

### 1. Clone and configure

```bash
git clone <repo-url>
cd ML-System-Design-Model-Serving
cp .env.example .env   # no secrets needed — models are public HuggingFace weights
```

### 2. Start all services

```bash
docker-compose up --build
```

**First start takes ~3-5 minutes**: DistilBERT (~268MB) downloads from HuggingFace Hub, then both v1 and v2 run warm-up passes before `/ready` flips to true.

### 3. Access services

| Service | URL |
|---|---|
| FastAPI docs (OpenAPI) | http://localhost:8000/docs |
| Control panel | http://localhost:8000/ui/ |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

### 4. Test the API

```bash
# Wait for readiness
curl http://localhost:8000/ready

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product exceeded all my expectations!"}'

# Check deployment state
curl http://localhost:8000/deployment/status

# Promote to canary (5%)
curl -X POST http://localhost:8000/deployment/promote

# Check shadow disagreement stats
curl http://localhost:8000/monitoring/disagreement

# Check drift detection
curl http://localhost:8000/monitoring/drift
```

### 5. Run load test

```bash
# Open Locust web UI with canary-aware tracking
docker-compose --profile load-test up locust

# Headless: 100 users, 2 minutes
docker run --rm --network host \
  -v $(pwd)/tests:/mnt/locust \
  locustio/locust:2.27.0 \
  -f /mnt/locust/locustfile.py \
  --host http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 2m --headless
```

---

## Deployment Walkthrough

### Step 1: Start in Shadow Mode (default)

The system starts in `shadow` state. v2 runs on every request but results are never returned to users. Watch the disagreement rate build up on the Disagreement tab of the control panel, or via:

```bash
curl http://localhost:8000/monitoring/disagreement
```

A disagreement rate < 5% means v2 is behaviorally similar to v1. Safe to proceed.

### Step 2: Promote to Canary 5%

```bash
curl -X POST http://localhost:8000/deployment/promote
```

5% of traffic now goes to v2. The state machine monitors error rate and p99 latency continuously. If either threshold is breached, auto-rollback fires automatically.

### Step 3: Auto-progression

With `auto_progression.enabled: true` (default), the system automatically advances from 5% → 25% → 50% → 100% after each stage runs cleanly for the configured duration.

### Step 4: Rollback (manual or automatic)

```bash
# Manual rollback
curl -X POST http://localhost:8000/deployment/rollback

# Auto-rollback fires when:
#   v2 error rate > 5%   OR
#   v2 p99 > 2x v1 p99
# Check what triggered it:
curl http://localhost:8000/deployment/audit
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Classify text sentiment |
| GET | `/health` | Liveness probe (always 200 if process alive) |
| GET | `/ready` | Readiness probe (200 only after warm-up) |
| GET | `/deployment/status` | State + per-version metrics |
| POST | `/deployment/promote` | Advance to next stage |
| POST | `/deployment/rollback` | Roll back to shadow |
| GET | `/deployment/audit` | State transition history |
| GET | `/monitoring/disagreement` | Shadow mode v1/v2 disagreement stats |
| GET | `/monitoring/drift` | Evidently drift detection status |
| GET | `/monitoring/cache` | Redis cache hit/miss stats |
| GET | `/circuit-breaker/status` | Circuit breaker state |
| POST | `/circuit-breaker/reset` | Manually close circuit |
| GET | `/metrics` | Prometheus metrics |

---

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `model_serving_inference_latency_seconds` | Histogram | p50/p95/p99 per version and deployment state |
| `model_serving_requests_total` | Counter | Request count by model_version and deployment_state |
| `model_serving_errors_total` | Counter | Errors by model_version and error_type |
| `model_serving_deployment_state` | Gauge | Current state as integer (0=shadow, 4=full) |
| `model_serving_canary_v2_traffic_fraction` | Gauge | Current v2 traffic fraction (0.0 to 1.0) |
| `model_serving_deployment_transitions_total` | Counter | State transitions by trigger |
| `model_serving_circuit_breaker_state` | Gauge | 0=closed, 1=open, 2=half_open |
| `model_serving_v1_v2_disagreement_rate` | Gauge | Rolling disagreement rate in shadow mode |
| `model_serving_drift_detected` | Gauge | 1 if drift detected, per drift_type |
| `model_serving_model_ready` | Gauge | 1 after warm-up complete, per model_version |
| `model_serving_warmup_latency_seconds` | Gauge | First vs last warm-up inference latency |

---

## Interview Talking Points

**"How do you deploy a new model without risk?"**
Shadow mode. v2 runs on every request but results are discarded. Zero user impact. You accumulate behavioral data (disagreement rate, confidence distributions) before any user ever sees v2.

**"How do you know when to promote from shadow to canary?"**
Disagreement rate < 5% is the behavioral signal. Evidently drift score < threshold confirms the canary inputs match the shadow inputs. Both signals together mean v2 is ready.

**"What triggers automatic rollback?"**
Two independent signals: error rate > 5% OR p99 latency > 2× v1 p99. The threshold is configurable. The minimum request count (20) prevents rollback from noise on the first few requests.

**"What happens if v2 crashes entirely?"**
The circuit breaker opens after N consecutive failures. All subsequent requests get v1 in < 1ms (fail-fast — no waiting for v2 timeout). After 30 seconds, one probe request tests if v2 recovered.

**"How do you debug a slow request?"**
Every response includes a `trace_id`. Search the structlog JSON logs for that trace_id — every log line in the request lifecycle is bound to it. This is the same principle as Jaeger/Zipkin distributed tracing.

**"What's the difference between this and what RecSys does with Thompson Sampling?"**
Thompson Sampling in RecSys optimizes which model to use based on click-through rate — it's online learning. Shadow/canary is about safely deploying a new version — deployment risk management. Completely different concern.

---

## Hosted vs. local

`docker-compose up` runs the architecture as designed: five containers, real
Redis, Prometheus scraping, Grafana dashboards. The hosted demo runs one
container with no attached services, which changes three things. All three are
reported by the API rather than assumed, so you can tell from outside which
mode you are looking at.

| | docker-compose | Cloud Run |
|---|---|---|
| **Cache** | Redis | In-process dict with the same TTL. `GET /health` reports `cache_backend`, and `redis_available` stays `false` — the fallback keeps the feature working but is never described as Redis. |
| **Deployment state** | JSON + JSONL on a volume | In memory. `state_durability: "ephemeral"` on `/deployment/status`. The in-memory audit log at `/deployment/audit` is unaffected. |
| **Drift detection** | Evidently | scipy Jensen-Shannon divergence. `/monitoring/drift` reports `method`, so which path ran is visible. Evidently pulls ~400MB of transitive dependencies for a number `monitoring/drift.py` already computes without it. |

Two smaller deployment details are load-bearing:

**One instance, one worker.** The state machine, circuit breaker, disagreement
window and cache all live in process. A second worker or a second instance
would give each its own copy — two control planes behind one URL, disagreeing
with each other and with nothing reporting the split. `--max-instances=1` and
`--workers 1` are what make the single-container story honest rather than
merely convenient.

**Empty means absent, not broken.** `REDIS_HOST=""` tells the service there is
no Redis here, so it skips the connect instead of spending cold-start seconds
resolving a hostname that was never going to answer. Leaving the compose
default in place cost seconds on every single cold start — a large share of the
startup budget, spent learning something already known.

### Deploying it

```bash
gcloud run deploy model-serving \
  --source . --region us-central1 --allow-unauthenticated \
  --port 8000 --memory 4Gi --cpu 2 \
  --min-instances 0 --max-instances 1 --timeout 300 --cpu-boost \
  --set-env-vars "^@^EPHEMERAL_STATE=1@MOUNT_UI=1@REDIS_HOST="
```

The image installs torch from PyTorch's CPU index and bakes the DistilBERT
weights in at build time. Both matter: `pip install torch` on Linux resolves to
the CUDA build and drags in ~2.5GB of `nvidia-*` wheels for a service that runs
`device=-1`, and downloading weights at first request would move ~268MB onto
the cold-start path. `HF_HUB_OFFLINE=1` then makes a missing layer fail loudly
at startup rather than silently re-downloading in production.

---

## Key Design References

- **Canary deployment**: Martin Fowler, "Canary Release" (2010) — the definitive write-up
- **Circuit breaker**: Michael Nygard, "Release It!" (2018) — Netflix Hystrix implements this
- **Shadow mode**: Google SRE Book — "testing in production" chapter
- **Evidently AI**: evidently.ai — open-source ML observability, used at Booking.com
- **INT8 quantization**: Bhandare et al., "Efficient 8-Bit Quantization of Transformer NLP Models" (2019)
- **Readiness vs liveness probes**: Kubernetes docs — the distinction matters at container startup
