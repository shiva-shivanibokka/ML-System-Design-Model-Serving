"""
Deployment State Machine.

States:
  SHADOW     → v2 runs silently on every request. User always gets v1.
                Used to validate v2 behavior at zero risk.
  CANARY_5   → 5% of requests go to v2. Auto-rollback if thresholds breached.
  CANARY_25  → 25% of requests go to v2.
  CANARY_50  → 50% of requests go to v2.
  FULL       → 100% of requests go to v2. v1 kept warm for instant rollback.
  ROLLED_BACK → v2 demoted back to shadow after a triggered rollback.

Transitions:
  Manual promotion:   SHADOW → CANARY_5 → CANARY_25 → CANARY_50 → FULL
  Auto-progression:   Automatic after configured clean-run duration per stage.
  Manual rollback:    Any canary/full state → ROLLED_BACK (demotes to shadow)
  Auto-rollback:      Triggered when error_rate > threshold OR p99 > 2× v1 p99

Every transition is logged to:
  1. In-memory audit log (last N entries)
  2. JSONL file on disk (audit_log/transitions.jsonl)
  3. PostgreSQL audit table (async, non-blocking)
  4. Prometheus counter (deployment_transitions_total)

State persistence:
  Current state is persisted to JSON on every transition so it survives
  container restarts. On startup, state is loaded from this file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional

import structlog

from configs.settings import settings

log = structlog.get_logger(__name__)

# On a scale-to-zero managed runtime the container filesystem is scratch space:
# it is wiped when the instance is recycled, and a second instance would get its
# own copy. Writing state there does not make it durable, it makes it
# *inconsistent* — the file would claim canary_50 on one instance and shadow on
# the next, with nothing reporting the discrepancy.
#
# So the hosted deployment declares the truth instead of faking it: state lives
# in memory for the life of the instance, and /deployment/status says so. Local
# and docker-compose runs leave this unset and keep the JSON + JSONL files.
EPHEMERAL_STATE = os.getenv("EPHEMERAL_STATE", "0") == "1"


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class DeploymentState(str, Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5"
    CANARY_25 = "canary_25"
    CANARY_50 = "canary_50"
    FULL = "full"
    ROLLED_BACK = "rolled_back"


# Ordered progression for auto-advance logic
PROGRESSION_ORDER: List[DeploymentState] = [
    DeploymentState.SHADOW,
    DeploymentState.CANARY_5,
    DeploymentState.CANARY_25,
    DeploymentState.CANARY_50,
    DeploymentState.FULL,
]

# Traffic fraction v2 receives in each state
V2_TRAFFIC_FRACTION: Dict[DeploymentState, float] = {
    DeploymentState.SHADOW: 0.0,  # shadow = no live traffic to v2
    DeploymentState.CANARY_5: 0.05,
    DeploymentState.CANARY_25: 0.25,
    DeploymentState.CANARY_50: 0.50,
    DeploymentState.FULL: 1.00,
    DeploymentState.ROLLED_BACK: 0.0,
}

# How long each canary stage must stay clean before auto-progression (seconds)
AUTO_PROGRESSION_DURATIONS: Dict[DeploymentState, int] = {
    DeploymentState.CANARY_5: settings.deployment.auto_progression.canary_5_duration_minutes
    * 60,
    DeploymentState.CANARY_25: settings.deployment.auto_progression.canary_25_duration_minutes
    * 60,
    DeploymentState.CANARY_50: settings.deployment.auto_progression.canary_50_duration_minutes
    * 60,
}


# ---------------------------------------------------------------------------
# Audit entry
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    timestamp: str
    from_state: str
    to_state: str
    trigger: str  # "manual_promote" | "manual_rollback" | "auto_promote" | "auto_rollback_error_rate" | "auto_rollback_latency" | "startup"
    v2_error_rate_at_event: float
    v2_p99_latency_ms: float
    v1_p99_latency_ms: float
    requests_seen: int
    note: str = ""


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class DeploymentStateMachine:
    """
    Thread-safe deployment state machine.

    Designed for a single-process FastAPI server (uvicorn workers=1).
    Uses a threading.Lock for state mutations since FastAPI runs async
    coroutines on a thread pool.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: DeploymentState = DeploymentState(
            settings.deployment.initial_state
        )
        self._state_entered_at: float = time.monotonic()
        self._audit_log: Deque[AuditEntry] = deque(
            maxlen=settings.monitoring.audit.max_entries_in_memory
        )
        self._total_requests: int = 0

        # Counters used for rollback decision — updated by the router
        self._v2_requests: int = 0
        self._v2_errors: int = 0
        self._v2_latencies: Deque[float] = deque(maxlen=200)
        self._v1_latencies: Deque[float] = deque(maxlen=200)

        # Auto-progression background thread
        self._auto_thread: Optional[threading.Thread] = None
        self._stop_auto: threading.Event = threading.Event()

        # Ensure persistence dirs exist
        if not EPHEMERAL_STATE:
            Path(settings.deployment.state_persistence_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            Path(settings.deployment.audit_log_path).parent.mkdir(
                parents=True, exist_ok=True
            )

    # -----------------------------------------------------------------------
    # Startup / shutdown
    # -----------------------------------------------------------------------

    def load_persisted_state(self) -> None:
        """Load state from disk on startup (survives container restart)."""
        path = Path(settings.deployment.state_persistence_path)
        if EPHEMERAL_STATE:
            log.info(
                "ephemeral_state_mode",
                using=self._state.value,
                note="state resets when this instance is recycled",
            )
        elif path.exists():
            try:
                data = json.loads(path.read_text())
                loaded_state = DeploymentState(
                    data.get("state", settings.deployment.initial_state)
                )
                with self._lock:
                    self._state = loaded_state
                log.info("deployment_state_loaded", state=loaded_state.value)
            except Exception as e:
                log.warning(
                    "state_load_failed", error=str(e), fallback=self._state.value
                )
        else:
            log.info("no_persisted_state", using=self._state.value)

        self._record_audit(
            from_state=self._state,
            to_state=self._state,
            trigger="startup",
            note="State loaded on container start",
        )

    def start_auto_progression(self) -> None:
        """Start background thread that advances canary stages on timer."""
        if not settings.deployment.auto_progression.enabled:
            return
        self._stop_auto.clear()
        self._auto_thread = threading.Thread(
            target=self._auto_progression_loop,
            daemon=True,
            name="deployment-auto-progression",
        )
        self._auto_thread.start()
        log.info("auto_progression_started")

    def stop(self) -> None:
        """Called during FastAPI shutdown."""
        self._stop_auto.set()

    # -----------------------------------------------------------------------
    # State reads (no lock needed — reads are atomic on CPython)
    # -----------------------------------------------------------------------

    @property
    def state(self) -> DeploymentState:
        return self._state

    @property
    def v2_traffic_fraction(self) -> float:
        return V2_TRAFFIC_FRACTION[self._state]

    def is_shadow_active(self) -> bool:
        return self._state == DeploymentState.SHADOW

    def is_canary_active(self) -> bool:
        return self._state in (
            DeploymentState.CANARY_5,
            DeploymentState.CANARY_25,
            DeploymentState.CANARY_50,
        )

    def get_status(self) -> dict:
        """Full status snapshot for the API /deployment/status endpoint."""
        with self._lock:
            v2_error_rate = (
                self._v2_errors / self._v2_requests if self._v2_requests > 0 else 0.0
            )
            v2_p99 = self._percentile(self._v2_latencies, 99)
            v1_p99 = self._percentile(self._v1_latencies, 99)
            time_in_state = time.monotonic() - self._state_entered_at

            return {
                "state": self._state.value,
                "v2_traffic_fraction": V2_TRAFFIC_FRACTION[self._state],
                "v2_requests": self._v2_requests,
                "v2_errors": self._v2_errors,
                "v2_error_rate": round(v2_error_rate, 4),
                "v2_p99_latency_ms": round(v2_p99, 1),
                "v1_p99_latency_ms": round(v1_p99, 1),
                "total_requests": self._total_requests,
                "time_in_state_seconds": round(time_in_state, 1),
                "auto_progression_enabled": settings.deployment.auto_progression.enabled,
                "state_durability": "ephemeral" if EPHEMERAL_STATE else "disk",
                "rollback_thresholds": {
                    "error_rate": settings.deployment.rollback.error_rate_threshold,
                    "latency_multiplier": settings.deployment.rollback.latency_p99_multiplier,
                },
            }

    def get_audit_log(self) -> List[dict]:
        """Return recent audit entries as list of dicts."""
        with self._lock:
            return [asdict(e) for e in self._audit_log]

    # -----------------------------------------------------------------------
    # State mutations (require lock)
    # -----------------------------------------------------------------------

    def promote(self, triggered_by: str = "manual") -> dict:
        """
        Advance to the next stage in the progression.
        Returns new state info.
        """
        with self._lock:
            current = self._state
            if current == DeploymentState.FULL:
                return {"ok": False, "reason": "Already at full deployment"}
            if current == DeploymentState.ROLLED_BACK:
                next_state = DeploymentState.SHADOW
            else:
                try:
                    idx = PROGRESSION_ORDER.index(current)
                    next_state = PROGRESSION_ORDER[idx + 1]
                except (ValueError, IndexError):
                    return {
                        "ok": False,
                        "reason": f"Cannot promote from {current.value}",
                    }

            self._transition(
                to_state=next_state,
                trigger=f"{triggered_by}_promote",
                note=f"Promoted by {triggered_by}",
            )
            return {
                "ok": True,
                "from_state": current.value,
                "to_state": next_state.value,
            }

    def rollback(self, trigger: str = "manual", note: str = "") -> dict:
        """
        Roll back to shadow mode. Flushes cache to avoid stale v2 predictions.
        """
        with self._lock:
            current = self._state
            if current in (DeploymentState.SHADOW, DeploymentState.ROLLED_BACK):
                return {"ok": False, "reason": "Already in shadow/rolled-back state"}

            self._transition(
                to_state=DeploymentState.ROLLED_BACK,
                trigger=trigger,
                note=note or f"Rollback triggered: {trigger}",
            )
            # Reset v2 counters so next canary starts clean
            self._v2_requests = 0
            self._v2_errors = 0
            self._v2_latencies.clear()
            return {"ok": True, "from_state": current.value, "to_state": "rolled_back"}

    # -----------------------------------------------------------------------
    # Metrics update (called by router on every request)
    # -----------------------------------------------------------------------

    def record_request(
        self,
        model_used: str,
        latency_ms: float,
        error: bool,
    ) -> None:
        """
        Update per-version request counters and latency windows.
        Also checks rollback thresholds after each v2 request.
        """
        with self._lock:
            self._total_requests += 1
            if model_used == "v1":
                self._v1_latencies.append(latency_ms)
            elif model_used == "v2":
                self._v2_requests += 1
                self._v2_latencies.append(latency_ms)
                if error:
                    self._v2_errors += 1
                # Check rollback thresholds after recording
                self._check_rollback_thresholds()

    def _check_rollback_thresholds(self) -> None:
        """
        Called under lock after every v2 request.
        Triggers automatic rollback if thresholds are breached.
        Must be called while holding self._lock.
        """
        cfg = settings.deployment.rollback
        if self._v2_requests < cfg.min_requests_before_check:
            return
        if self._state not in (
            DeploymentState.CANARY_5,
            DeploymentState.CANARY_25,
            DeploymentState.CANARY_50,
            DeploymentState.FULL,
        ):
            return

        error_rate = self._v2_errors / self._v2_requests
        v2_p99 = self._percentile(self._v2_latencies, 99)
        v1_p99 = self._percentile(self._v1_latencies, 99)

        if error_rate > cfg.error_rate_threshold:
            log.warning(
                "auto_rollback_triggered",
                reason="error_rate",
                v2_error_rate=round(error_rate, 4),
                threshold=cfg.error_rate_threshold,
            )
            self._transition(
                to_state=DeploymentState.ROLLED_BACK,
                trigger="auto_rollback_error_rate",
                note=f"v2 error rate {error_rate:.2%} exceeded threshold {cfg.error_rate_threshold:.2%}",
            )
            return

        if v1_p99 > 0 and v2_p99 > v1_p99 * cfg.latency_p99_multiplier:
            log.warning(
                "auto_rollback_triggered",
                reason="latency_p99",
                v2_p99_ms=round(v2_p99, 1),
                v1_p99_ms=round(v1_p99, 1),
                multiplier=cfg.latency_p99_multiplier,
            )
            self._transition(
                to_state=DeploymentState.ROLLED_BACK,
                trigger="auto_rollback_latency",
                note=(
                    f"v2 p99 {v2_p99:.1f}ms > {cfg.latency_p99_multiplier}x "
                    f"v1 p99 {v1_p99:.1f}ms"
                ),
            )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _record_audit(
        self,
        from_state: DeploymentState,
        to_state: DeploymentState,
        trigger: str,
        note: str = "",
    ) -> None:
        """
        Append an audit entry without changing state.

        Used for markers that belong in the trail but are not transitions — the
        "startup" entry that records which state the process booted into. It is
        deliberately not _transition(): that one moves the machine, resets the
        stage timer and increments the Prometheus transition counter, none of
        which is true of simply starting up.

        Takes the lock itself; callers must not hold it.
        """
        with self._lock:
            entry = AuditEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                from_state=from_state.value,
                to_state=to_state.value,
                trigger=trigger,
                v2_error_rate_at_event=(
                    round(self._v2_errors / self._v2_requests, 4)
                    if self._v2_requests > 0
                    else 0.0
                ),
                v2_p99_latency_ms=round(self._percentile(self._v2_latencies, 99), 1),
                v1_p99_latency_ms=round(self._percentile(self._v1_latencies, 99), 1),
                requests_seen=self._total_requests,
                note=note,
            )
            self._audit_log.append(entry)
            self._append_audit_file(entry)

        log.info(
            "deployment_audit",
            from_state=from_state.value,
            to_state=to_state.value,
            trigger=trigger,
            note=note,
        )

    def _transition(
        self,
        to_state: DeploymentState,
        trigger: str,
        note: str = "",
    ) -> None:
        """
        Execute a state transition. Caller must hold self._lock.
        Writes to in-memory log, disk JSONL, and Prometheus counter.
        """
        from_state = self._state
        v2_error_rate = (
            self._v2_errors / self._v2_requests if self._v2_requests > 0 else 0.0
        )
        v2_p99 = self._percentile(self._v2_latencies, 99)
        v1_p99 = self._percentile(self._v1_latencies, 99)

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            from_state=from_state.value,
            to_state=to_state.value,
            trigger=trigger,
            v2_error_rate_at_event=round(v2_error_rate, 4),
            v2_p99_latency_ms=round(v2_p99, 1),
            v1_p99_latency_ms=round(v1_p99, 1),
            requests_seen=self._total_requests,
            note=note,
        )

        self._state = to_state
        self._state_entered_at = time.monotonic()
        self._audit_log.append(entry)

        # Persist state to disk
        self._persist_state()

        # Append to JSONL audit file
        self._append_audit_file(entry)

        # Prometheus counter (import here to avoid circular at module load)
        try:
            from monitoring.metrics import DEPLOYMENT_TRANSITIONS

            DEPLOYMENT_TRANSITIONS.labels(
                from_state=from_state.value,
                to_state=to_state.value,
                trigger=trigger,
            ).inc()
        except Exception:
            pass

        log.info(
            "deployment_transition",
            from_state=from_state.value,
            to_state=to_state.value,
            trigger=trigger,
            note=note,
        )

    def _persist_state(self) -> None:
        """Write current state to JSON file. Called under lock."""
        if EPHEMERAL_STATE:
            return
        try:
            path = Path(settings.deployment.state_persistence_path)
            path.write_text(
                json.dumps(
                    {
                        "state": self._state.value,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
        except Exception as e:
            log.warning("state_persist_failed", error=str(e))

    def _append_audit_file(self, entry: AuditEntry) -> None:
        """Append audit entry to JSONL file. Called under lock."""
        # The in-memory audit log (GET /deployment/audit) is unaffected — only
        # the on-disk copy is skipped.
        if EPHEMERAL_STATE:
            return
        try:
            path = Path(settings.deployment.audit_log_path)
            with open(path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            log.warning("audit_file_write_failed", error=str(e))

    def _auto_progression_loop(self) -> None:
        """
        Background thread: checks if the current canary stage has been
        running cleanly long enough to auto-promote to the next stage.
        Polls every 15 seconds.
        """
        while not self._stop_auto.wait(15):
            with self._lock:
                current = self._state
                if current not in AUTO_PROGRESSION_DURATIONS:
                    continue

                # Has there been a rollback? Skip.
                if current == DeploymentState.ROLLED_BACK:
                    continue

                duration_required = AUTO_PROGRESSION_DURATIONS[current]
                time_in_state = time.monotonic() - self._state_entered_at

                if time_in_state >= duration_required:
                    try:
                        idx = PROGRESSION_ORDER.index(current)
                        next_state = PROGRESSION_ORDER[idx + 1]
                    except (ValueError, IndexError):
                        continue

                    log.info(
                        "auto_progression_eligible",
                        current=current.value,
                        next=next_state.value,
                        time_in_state_s=round(time_in_state, 1),
                    )
                    self._transition(
                        to_state=next_state,
                        trigger="auto_promote",
                        note=(
                            f"Auto-promoted after {time_in_state:.0f}s clean run "
                            f"(required {duration_required}s)"
                        ),
                    )

    @staticmethod
    def _percentile(data: Deque[float], p: int) -> float:
        """Compute p-th percentile from a deque. Returns 0 if empty."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return sorted_data[idx]


# Module-level singleton — shared across the FastAPI process
state_machine = DeploymentStateMachine()
