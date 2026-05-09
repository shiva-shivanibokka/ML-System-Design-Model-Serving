"""
Disagreement Monitor — v1 vs v2 shadow mode comparison.

In shadow mode, v1 and v2 both run on every request.
The disagreement monitor tracks how often they predict different labels.

Why this matters:
  A disagreement rate of 2% on a sentiment model means 1 in 50 users
  would get a different answer if v2 were live. That might be acceptable
  or alarming depending on context:

  - Low disagreement (<5%): v2 is behaviorally similar to v1. Safe to promote.
  - Medium disagreement (5-30%): v2 is meaningfully different. Inspect cases.
  - High disagreement (>30%): v2 is fundamentally different. Deep investigation
    required before any canary traffic.

  The disagreement monitor also records WHICH direction disagreements go:
  - POSITIVE → NEGATIVE: v2 is more pessimistic than v1
  - NEGATIVE → POSITIVE: v2 is more optimistic than v1
  These directional patterns reveal systematic model bias changes.

  Confidence gap analysis:
  Even when v1 and v2 agree on the label, they may have very different
  confidence scores. A large confidence gap (v1=0.99, v2=0.62) means
  v2 is less certain — which affects downstream thresholds.

Metric logged to Prometheus: model_serving_v1_v2_disagreement_rate (Gauge)
Alert: if rate > 30% after 50+ samples (configurable in config.yaml)
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import structlog

from configs.settings import settings

log = structlog.get_logger(__name__)


@dataclass
class DisagreementRecord:
    """Single shadow comparison record."""

    v1_label: str
    v2_label: str
    v1_score: float
    v2_score: float
    agrees: bool
    confidence_gap: float  # abs(v1_score - v2_score)
    input_length: int


class DisagreementMonitor:
    """
    Rolling window disagreement tracker for shadow mode.

    Thread-safe. Updated on every shadow mode request.
    """

    def __init__(self) -> None:
        cfg = settings.monitoring.disagreement
        self._window_size: int = cfg.window_size
        self._alert_threshold: float = cfg.alert_threshold
        self._min_samples: int = cfg.min_samples_before_alert

        self._records: Deque[DisagreementRecord] = deque(maxlen=self._window_size)
        self._lock = threading.Lock()

        # Lifetime counters (not windowed)
        self._total_comparisons: int = 0
        self._total_disagreements: int = 0

    def record(
        self,
        v1_label: str,
        v2_label: str,
        v1_score: float,
        v2_score: float,
        input_length: int,
    ) -> bool:
        """
        Record a shadow comparison and return True if disagreement alert triggered.

        Called by the router after every shadow mode request where v2 ran successfully.
        """
        agrees = v1_label == v2_label
        confidence_gap = abs(v1_score - v2_score)

        record = DisagreementRecord(
            v1_label=v1_label,
            v2_label=v2_label,
            v1_score=v1_score,
            v2_score=v2_score,
            agrees=agrees,
            confidence_gap=confidence_gap,
            input_length=input_length,
        )

        alert_triggered = False
        with self._lock:
            self._records.append(record)
            self._total_comparisons += 1
            if not agrees:
                self._total_disagreements += 1

            # Update Prometheus gauge
            current_rate = self._compute_disagreement_rate()
            self._update_prometheus(current_rate, v1_label, v2_label, agrees)

            # Log disagreements at info level (they're notable)
            if not agrees:
                log.info(
                    "shadow_disagreement",
                    v1_label=v1_label,
                    v2_label=v2_label,
                    v1_score=round(v1_score, 4),
                    v2_score=round(v2_score, 4),
                    confidence_gap=round(confidence_gap, 4),
                    rolling_rate=round(current_rate, 4),
                )

            # Alert threshold check
            if (
                self._total_comparisons >= self._min_samples
                and current_rate > self._alert_threshold
            ):
                log.warning(
                    "disagreement_rate_alert",
                    rate=round(current_rate, 4),
                    threshold=self._alert_threshold,
                    window_size=len(self._records),
                )
                alert_triggered = True

        return alert_triggered

    def _compute_disagreement_rate(self) -> float:
        """Compute disagreement rate over the current window. Must hold lock."""
        if not self._records:
            return 0.0
        disagreements = sum(1 for r in self._records if not r.agrees)
        return disagreements / len(self._records)

    def _update_prometheus(
        self,
        rate: float,
        v1_label: str,
        v2_label: str,
        agrees: bool,
    ) -> None:
        """Update Prometheus metrics. Must hold lock."""
        try:
            from monitoring.metrics import DISAGREEMENT_RATE, DISAGREEMENT_EVENTS

            DISAGREEMENT_RATE.set(rate)
            if not agrees:
                DISAGREEMENT_EVENTS.labels(
                    v1_label=v1_label,
                    v2_label=v2_label,
                ).inc()
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Return full disagreement statistics for the API and Gradio UI."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_comparisons": self._total_comparisons,
                "window_size": 0,
                "disagreement_rate": 0.0,
                "alert_active": False,
                "direction_breakdown": {},
                "avg_confidence_gap": 0.0,
                "avg_confidence_gap_on_disagreements": 0.0,
            }

        disagreements = [r for r in records if not r.agrees]
        agreements = [r for r in records if r.agrees]
        rate = len(disagreements) / len(records)

        # Direction breakdown: which label transitions are happening
        direction_breakdown: Dict[str, int] = {}
        for r in disagreements:
            key = f"{r.v1_label}_to_{r.v2_label}"
            direction_breakdown[key] = direction_breakdown.get(key, 0) + 1

        avg_gap_all = sum(r.confidence_gap for r in records) / len(records)
        avg_gap_disagree = (
            sum(r.confidence_gap for r in disagreements) / len(disagreements)
            if disagreements
            else 0.0
        )

        return {
            "total_comparisons": self._total_comparisons,
            "window_size": len(records),
            "disagreements_in_window": len(disagreements),
            "agreements_in_window": len(agreements),
            "disagreement_rate": round(rate, 4),
            "alert_active": (
                self._total_comparisons >= self._min_samples
                and rate > self._alert_threshold
            ),
            "alert_threshold": self._alert_threshold,
            "direction_breakdown": direction_breakdown,
            "avg_confidence_gap_all": round(avg_gap_all, 4),
            "avg_confidence_gap_on_disagreements": round(avg_gap_disagree, 4),
        }

    def get_recent_disagreements(self, n: int = 20) -> List[dict]:
        """Return the N most recent disagreement records for display."""
        with self._lock:
            records = list(self._records)

        disagreements = [r for r in records if not r.agrees]
        recent = disagreements[-n:]
        return [
            {
                "v1_label": r.v1_label,
                "v2_label": r.v2_label,
                "v1_score": round(r.v1_score, 4),
                "v2_score": round(r.v2_score, 4),
                "confidence_gap": round(r.confidence_gap, 4),
                "input_length": r.input_length,
            }
            for r in recent
        ]

    def reset(self) -> None:
        """Reset for a new deployment round."""
        with self._lock:
            self._records.clear()
            self._total_comparisons = 0
            self._total_disagreements = 0


# Module-level singleton
disagreement_monitor = DisagreementMonitor()
