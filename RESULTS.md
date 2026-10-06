# Independent review and fixes (branch `sop-eval`)

An adversarial review of this repository found that **the three mechanisms the
project exists to demonstrate were each broken**, and that one tracked file was
publishing the author's Windows username. Every defect below was reproduced
before being fixed. Measurements are from this machine: Windows 11, Python 3.12,
CPU-only `torch==2.3.0`.

---

## 1. One early `return` cancelled the canary, shadow mode and drift

`api/main.py` looked up the cache **before** routing, and the key came from the
deployment *state* rather than from the routing decision:

```python
primary_version = "v2" if deployment_state == "full" else "v1"
cached = cache.get(text, primary_version)
if cached:
    return PredictResponse(...)        # router never runs
```

Because the router never ran on a hit, three things silently stopped happening.

**The canary split did not happen.** At a 50% split, 60 identical requests:

```
AssertionError: 60 requests at a 50% canary split never reached v2 (saw {'v1'}).
```

Zero traffic reached v2 while `/metrics` reported
`model_serving_canary_v2_traffic_fraction 0.5`. The README's *"5% of traffic now
goes to v2"* was untrue whenever the cache was warm — and the cache is on by
default.

**Shadow mode did not run v2.** Five cached requests produced **0** comparisons,
so the disagreement monitor was sampling cache *misses* only.

**Drift never saw cached requests**, so the "first 200 requests" reference window
was built from misses only.

**Fix.** Caching moved one layer down, into `cache/cached_model.py`. The router
still makes the routing decision and still dispatches shadow v2 on every request;
it is the individual model call that is short-circuited. Caching is an inference
optimisation, not a routing decision.

Three regression tests now pin each mechanism, and each one fails against the old
code with the message quoted above.

A knock-on correctness fix: a cache hit used to return the stored score at full
precision while a fresh response rounded to 4 dp, so the same input returned
`0.9994489550590515` or `0.9994` depending on cache state. The cached value is
now stored rounded.

---

## 2. Auto-promotion had no health gate at all

The progression loop checked elapsed time and nothing else — not error rate, not
latency, not disagreement, not drift — and then wrote `"Auto-promoted after Ns
clean run"` into the audit log. Reproduced: a canary at 100% v2 failure rate
promoted itself, with the audit entry asserting it had run cleanly.

**Fix.** `_auto_progression_blockers()` applies the same thresholds the rollback
check already uses, so a stage cannot promote through a condition that would have
rolled it back. The minimum-request gate matters most: without a sample, a stage
carrying no v2 traffic at all is indistinguishable from a healthy one — which is
exactly the situation defect 1 created. The audit note now records the actual
request count and error rate instead of asserting cleanliness.

Three tests, two of them negative cases, because a gate with only passing tests
is not a gate.

Also removed: a dead `if current == DeploymentState.ROLLED_BACK: continue` guard.
`ROLLED_BACK` is never a key of `AUTO_PROGRESSION_DURATIONS`, so it was
unreachable.

---

## 3. The drift score meant the opposite of itself depending on what was installed

With `evidently` importable, `monitoring/drift.py` read Evidently's `drift_score`
— a K-S **p-value** — and stored it in the same field as this module's
Jensen-Shannon **divergence**:

```
IDENTICAL distributions -> drift_score 1.0, any_drift False
SHIFTED   distributions -> drift_score 0.0, any_drift True
```

Exactly backwards from the scipy path, where 0 means identical. `web/app.js` drew
the bar as `score/limit` with a tooltip saying *"0 means identical"*, so the
dashboard pinned both drift bars at 100% when there was **no** drift, and any
Prometheus alert on `model_serving_drift_score > 0.1` fired permanently.

**Fix: the Evidently branch was removed, not repaired.** It had 0% test coverage,
and `requirements-serve.txt` — the file the Docker image and Cloud Run actually
installed — deliberately excludes Evidently. The path that shipped was never the
path the code preferred. One metric defined one way is worth more here than two
that disagree about their own sign. The README's *"Evidently AI drift detection"*
claims were corrected rather than the behaviour dressed up.

---

## 4. `call_timeout_seconds` was loaded and never read

`deployment/circuit_breaker.py` stored `self._call_timeout` and nothing ever used
it. Reproduced: a 7-second call under a 5-second timeout returned normally, with
`failures=0` and the breaker closed.

The class docstring builds the entire case for the breaker on this timeout
(*"Every request waits call_timeout_seconds before falling back to v1 … server
threads exhausted"*), so **the one failure mode it exists to catch was the one it
could not detect**. A hung v2 hung the request forever; only raised exceptions
ever tripped it.

**Fix.** Calls now run under a timeout. One subtlety worth recording: the obvious
implementation, `with ThreadPoolExecutor(...) as pool`, still took 7 seconds —
`__exit__` calls `shutdown(wait=True)` and blocks on the hung worker, so the
timeout bought the caller nothing. `shutdown(wait=False)` lets the caller leave
at the deadline.

```
before: no timeout, 7.0s, failures=0
after:  caller returned after 5.0s (timeout 5.0s), failure counted
```

---

## 5. The breaker did not count consecutive failures, and its test could not fail

`_on_success` decremented the failure count rather than resetting it — a sliding
window, not a consecutive count. After 4 failures and 1 success the counter stood
at 3, so two more failures opened the breaker where a true reset needs five.
Meanwhile the log line said `consecutive_failures` and the README says
"consecutive" twice.

`test_success_resets_the_failure_run` is docstringed *"the count starts over"* and
then asserted only `state is CLOSED` after a single further failure — true under
**both** semantics, so the test named for the claim could not detect the
difference. It now asserts that `threshold - 1` further failures leave it closed
and the next one opens it.

```
after 4 failures, count = 4
after 1 success,  count = 0   (was 3)
```

---

## 6. A tracked file was publishing the author's username

`.coverage` was tracked — a 53 KB SQLite file whose `file` table held 19
**absolute** paths:

```
C:\Users\<account>\OneDrive\Desktop\GITHUB REPOS\ML-System-Design-Model-Serving\api\main.py
```

The account name is redacted here on purpose: writing the real string into this
file would republish the exact thing the fix removes.

Untracked and added to `.gitignore`, along with `.coverage.*`, `coverage.xml` and
`htmlcov/`. It also meant any local `pytest --cov` run dirtied a tracked file.

**Still outstanding:** the file remains in the repository's *history*. Removing it
there requires a force-push, which has not been done.

---

## 7. Claims corrected

| Claim | Measured |
|---|---|
| `▶ Open the live demo` + three `run.app` links | the service returns **503** |
| trial "ends around 19 September 2026" (future tense) | that date has passed |
| "tests-71" badge | **78** |
| "v1 30-80ms, v2 20-55ms (~30% faster)" | v1 p50 **32.6ms** / p99 64.2; v2 p50 **25.3ms** / p99 28.5 — **~20% faster**, measured over 50 warm requests each |
| "The full stack is six containers" / "five containers" / "(5 services)" | **four** (redis, gateway, prometheus, grafana), plus locust behind a profile |
| "Prometheus + **Grafana dashboards**" | there are no dashboard JSONs in the repo; `GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH` pointed at a file that does not exist |
| docker-compose "uses Evidently" | the image installs `requirements-serve.txt`, which excludes it |
| shadow mode "the user never waits for v2" | v2 is awaited sequentially; a shadow request costs both inferences |
| `state_machine.py` "PostgreSQL audit table" | Postgres was removed from the project |
| drift "reference window = training distribution baseline" | it is the first 200 requests *this process served*, and it re-baselines on restart |
| drift "is v2 systematically less confident than v1" | only the user-facing score is recorded, so v2's confidence never reaches the detector |
| quickstart step 1: `cp .env.example .env` | the file did not exist; `.gitignore` even had a `!.env.example` rule for it |

---

## 8. Other changes

- **`.env.example` created** so the quickstart's first command runs.
- **Dead dependencies removed**: `gradio`, `plotly`, `sqlalchemy`, `psycopg2-binary`.
  Nothing imports any of them; `gradio 4.36.1` is a year-old pin with known
  advisories that the documented local install pulled in.
- **A v2 error no longer deflates v2's p99.** The error path pushed `0.0` into the
  v2 latency window, so errors dragged the percentile *down* — making the
  latency-based rollback least likely to fire exactly when v2 was failing most.
  `record_request` now accepts `None` for "no timing".
- **Cache hits are excluded from latency aggregation** rather than recorded as
  0.0 ms, and `cache_hit` on the response is now real instead of hardcoded `False`.
- **Redis is no longer published to the host** in `docker-compose.yml`. Binding
  6379 made `docker compose up` fail whenever any other Redis was running, after
  which the gateway came up "healthy" on its in-process cache — quietly making
  the README's "real Redis" row false with no error anywhere but `/health`.
- Obsolete `version: "3.9"` removed from `docker-compose.yml`; a stray internal
  marker word removed from `cache/redis_cache.py`.

## Known limitations, stated rather than fixed

- **A timed-out call still occupies its worker thread.** Python cannot safely kill
  a running thread. What the timeout changes is that the *caller* stops waiting
  and the breaker counts a failure, which is what lets it open and shed load.
- **HALF_OPEN is not a single-probe gate.** With `success_threshold: 2`, at least
  two calls reach v2 before it closes, and under concurrency arbitrarily many do.
  The README describes "one probe request".
- **`configs/config.yaml` is not the single source of truth it claims.** Several
  keys are loaded and never read: `deployment.canary_traffic_splits` (the real
  splits are hardcoded in `state_machine.py`), `monitoring.latency_slo` (buckets
  are hardcoded in `metrics.py`), and the whole `api.*` block.
- **`models/model_v1.py` and `model_v2.py` have no test coverage of `load()` or
  `_run_inference()`** — the deliberate consequence of CI never downloading
  weights.
- **No LICENSE file.**

## Verification

```
78 passed                (pytest tests/)
ruff check  — All checks passed!
ruff format — 23 files already formatted
coverage    — 85%
canary      — v2 reached at a 50% split with a warm cache
shadow      — 5 cached requests produce 5 comparisons
drift       — 5 cached requests produce 5 records
breaker     — 7s call under a 5s timeout: caller returns at 5.0s, failure counted
breaker     — 4 failures then 1 success leaves the consecutive count at 0
```
