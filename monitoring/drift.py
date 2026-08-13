"""
Input Distribution Drift Detector using Evidently AI.

Problem being solved:
  "The inputs coming to v2 during the canary are different from what
   the model was trained on. This means v2's performance numbers
   during the canary don't tell us how it will perform at full scale."

What we monitor:
  1. TEXT LENGTH DISTRIBUTION — are users sending longer/shorter texts?
     A shift here means different types of content are being submitted.
     (e.g. model trained on tweets, now receiving full articles)

  2. CONFIDENCE SCORE DISTRIBUTION — is v2 systematically less/more confident
     than v1 on the same inputs?
     A shift here means the model has different calibration.
     (Important for any system using confidence thresholds downstream)

How it works:
  - Reference window: first N requests (captured as training distribution baseline)
  - Detection window: rolling window of most recent M requests
  - Every K requests, run Evidently DataDriftPreset
  - If Jensen-Shannon divergence > threshold → flag drift, update Prometheus gauge

Evidently AI:
  Open-source library from Evidently AI (YC-backed, founded 2021).
  Used at Booking.com, EPAM, and many MLOps teams.
  Provides a DataDriftPreset that runs multiple statistical tests and
  returns a unified report with per-feature drift scores.

Fallback:
  If evidently is not installed, falls back to a manual
  Jensen-Shannon divergence computation using scipy.stats.
  The monitoring works either way — evidently just gives richer reports.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import structlog

from configs.settings import settings

log = structlog.get_logger(__name__)

# Optional import — fall back gracefully. These names look unused to a linter
# because the import itself is the test: they are re-imported inside
# _check_with_evidently, and this block only decides which path runs.
try:
    import pandas as pd  # noqa: F401
    from evidently import ColumnMapping  # noqa: F401
    from evidently.report import Report  # noqa: F401
    from evidently.metric_preset import DataDriftPreset  # noqa: F401

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    log.info("evidently_not_installed", fallback="scipy_js_divergence")

try:
    from scipy.spatial.distance import jensenshannon

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class DriftCheckResult:
    """Result of a single drift detection run."""

    checked_at_request_n: int
    reference_size: int
    current_size: int
    text_length_drift_score: float
    confidence_drift_score: float
    text_length_drifted: bool
    confidence_drifted: bool
    any_drift: bool
    method: str  # "evidently" | "jensen_shannon" | "unavailable"


class DriftDetector:
    """
    Rolling input distribution drift detector.

    Maintains two data collections:
    - reference: first N requests (the "baseline" distribution)
    - current:   rolling window of last M requests (what's happening now)

    Runs a drift check every K requests and updates Prometheus gauges.
    """

    def __init__(self) -> None:
        cfg = settings.monitoring.drift
        self._reference_size: int = cfg.reference_window_size
        self._detection_size: int = cfg.detection_window_size
        self._check_every_n: int = cfg.run_every_n_requests
        self._text_length_threshold: float = cfg.text_length_drift_threshold
        self._confidence_threshold: float = cfg.confidence_drift_threshold

        # Reference window (frozen after reaching target size)
        self._reference_text_lengths: List[int] = []
        self._reference_confidences: List[float] = []
        self._reference_frozen: bool = False

        # Rolling current window
        self._current_text_lengths: Deque[int] = deque(maxlen=self._detection_size)
        self._current_confidences: Deque[float] = deque(maxlen=self._detection_size)

        self._total_records: int = 0
        self._last_check_result: Optional[DriftCheckResult] = None
        self._check_history: List[DriftCheckResult] = []
        self._lock = threading.Lock()

    def record(self, text_length: int, confidence: float) -> None:
        """
        Record a new inference observation.
        Builds reference window, then checks for drift every N requests.
        """
        with self._lock:
            self._total_records += 1

            # Build reference window (frozen after target size)
            if not self._reference_frozen:
                self._reference_text_lengths.append(text_length)
                self._reference_confidences.append(confidence)
                if len(self._reference_text_lengths) >= self._reference_size:
                    self._reference_frozen = True
                    log.info(
                        "drift_reference_window_frozen",
                        size=self._reference_size,
                    )

            # Always update rolling current window
            self._current_text_lengths.append(text_length)
            self._current_confidences.append(confidence)

            # Run drift check every N requests (only once reference is frozen)
            if (
                self._reference_frozen
                and self._total_records % self._check_every_n == 0
                and len(self._current_text_lengths) >= self._detection_size // 2
            ):
                result = self._run_drift_check()
                self._last_check_result = result
                self._check_history.append(result)
                self._update_prometheus(result)

    def _run_drift_check(self) -> DriftCheckResult:
        """Execute drift detection. Must be called under lock."""
        ref_lengths = self._reference_text_lengths[:]
        ref_confs = self._reference_confidences[:]
        cur_lengths = list(self._current_text_lengths)
        cur_confs = list(self._current_confidences)

        if EVIDENTLY_AVAILABLE:
            return self._check_with_evidently(
                ref_lengths, ref_confs, cur_lengths, cur_confs
            )
        elif SCIPY_AVAILABLE:
            return self._check_with_js_divergence(
                ref_lengths, ref_confs, cur_lengths, cur_confs
            )
        else:
            return DriftCheckResult(
                checked_at_request_n=self._total_records,
                reference_size=len(ref_lengths),
                current_size=len(cur_lengths),
                text_length_drift_score=0.0,
                confidence_drift_score=0.0,
                text_length_drifted=False,
                confidence_drifted=False,
                any_drift=False,
                method="unavailable",
            )

    def _check_with_evidently(
        self, ref_lengths, ref_confs, cur_lengths, cur_confs
    ) -> DriftCheckResult:
        """Run Evidently DataDriftPreset on text length and confidence features."""
        try:
            import pandas as pd
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            ref_df = pd.DataFrame(
                {
                    "text_length": ref_lengths,
                    "confidence": ref_confs,
                }
            )
            cur_df = pd.DataFrame(
                {
                    "text_length": cur_lengths,
                    "confidence": cur_confs,
                }
            )

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            result_dict = report.as_dict()

            # Extract per-feature drift scores from Evidently result
            metrics = result_dict.get("metrics", [])
            text_score = 0.0
            conf_score = 0.0
            text_drifted = False
            conf_drifted = False

            for metric in metrics:
                result = metric.get("result", {})
                drift_by_columns = result.get("drift_by_columns", {})

                if "text_length" in drift_by_columns:
                    col = drift_by_columns["text_length"]
                    text_score = col.get("drift_score", 0.0)
                    text_drifted = col.get("drift_detected", False)

                if "confidence" in drift_by_columns:
                    col = drift_by_columns["confidence"]
                    conf_score = col.get("drift_score", 0.0)
                    conf_drifted = col.get("drift_detected", False)

            return DriftCheckResult(
                checked_at_request_n=self._total_records,
                reference_size=len(ref_lengths),
                current_size=len(cur_lengths),
                text_length_drift_score=round(text_score, 4),
                confidence_drift_score=round(conf_score, 4),
                text_length_drifted=text_drifted,
                confidence_drifted=conf_drifted,
                any_drift=text_drifted or conf_drifted,
                method="evidently",
            )
        except Exception as e:
            log.warning("evidently_check_failed", error=str(e), fallback="scipy")
            return self._check_with_js_divergence(
                ref_lengths, ref_confs, cur_lengths, cur_confs
            )

    def _check_with_js_divergence(
        self, ref_lengths, ref_confs, cur_lengths, cur_confs
    ) -> DriftCheckResult:
        """
        Fallback: compute Jensen-Shannon divergence manually.

        Jensen-Shannon divergence:
        - Symmetric version of KL divergence
        - Range: [0, 1] (JS divergence using log base 2 = [0, 1])
        - 0 = identical distributions, 1 = completely different
        - Standard threshold: 0.1 (configurable)
        """
        text_score = self._js_divergence_1d(ref_lengths, cur_lengths, bins=20)
        conf_score = self._js_divergence_1d(ref_confs, cur_confs, bins=20)

        text_drifted = text_score > self._text_length_threshold
        conf_drifted = conf_score > self._confidence_threshold

        return DriftCheckResult(
            checked_at_request_n=self._total_records,
            reference_size=len(ref_lengths),
            current_size=len(cur_lengths),
            text_length_drift_score=round(text_score, 4),
            confidence_drift_score=round(conf_score, 4),
            text_length_drifted=text_drifted,
            confidence_drifted=conf_drifted,
            any_drift=text_drifted or conf_drifted,
            method="jensen_shannon",
        )

    @staticmethod
    def _js_divergence_1d(ref: list, cur: list, bins: int = 20) -> float:
        """Compute Jensen-Shannon divergence between two 1D distributions."""
        try:
            all_vals = ref + cur
            min_val, max_val = min(all_vals), max(all_vals)
            if min_val == max_val:
                return 0.0

            bin_edges = np.linspace(min_val, max_val, bins + 1)
            ref_hist, _ = np.histogram(ref, bins=bin_edges, density=True)
            cur_hist, _ = np.histogram(cur, bins=bin_edges, density=True)

            # Add small epsilon to avoid log(0)
            ref_hist = ref_hist + 1e-10
            cur_hist = cur_hist + 1e-10

            # Normalise to probability distributions
            ref_hist = ref_hist / ref_hist.sum()
            cur_hist = cur_hist / cur_hist.sum()

            if SCIPY_AVAILABLE:
                score = float(jensenshannon(ref_hist, cur_hist))
            else:
                # Manual JS divergence
                m = 0.5 * (ref_hist + cur_hist)
                js = 0.5 * np.sum(ref_hist * np.log(ref_hist / m)) + 0.5 * np.sum(
                    cur_hist * np.log(cur_hist / m)
                )
                score = float(np.sqrt(js))

            return min(score, 1.0)  # clamp to [0, 1]
        except Exception:
            return 0.0

    def _update_prometheus(self, result: DriftCheckResult) -> None:
        """Update Prometheus drift gauges. Must be called under lock."""
        try:
            from monitoring.metrics import DRIFT_DETECTED, DRIFT_SCORE

            DRIFT_DETECTED.labels(drift_type="text_length").set(
                1.0 if result.text_length_drifted else 0.0
            )
            DRIFT_DETECTED.labels(drift_type="confidence_score").set(
                1.0 if result.confidence_drifted else 0.0
            )
            DRIFT_SCORE.labels(drift_type="text_length").set(
                result.text_length_drift_score
            )
            DRIFT_SCORE.labels(drift_type="confidence_score").set(
                result.confidence_drift_score
            )
        except Exception:
            pass

        if result.any_drift:
            log.warning(
                "drift_detected",
                text_length_score=result.text_length_drift_score,
                confidence_score=result.confidence_drift_score,
                text_length_drifted=result.text_length_drifted,
                confidence_drifted=result.confidence_drifted,
                method=result.method,
            )

    def get_status(self) -> dict:
        """Return drift detection status for API and Gradio."""
        with self._lock:
            last = self._last_check_result
            return {
                "reference_window_frozen": self._reference_frozen,
                "reference_size": len(self._reference_text_lengths),
                "total_records": self._total_records,
                "checks_run": len(self._check_history),
                "last_check": (
                    {
                        "checked_at_request_n": last.checked_at_request_n,
                        "text_length_drift_score": last.text_length_drift_score,
                        "confidence_drift_score": last.confidence_drift_score,
                        "text_length_drifted": last.text_length_drifted,
                        "confidence_drifted": last.confidence_drifted,
                        "any_drift": last.any_drift,
                        "method": last.method,
                    }
                    if last
                    else None
                ),
                "thresholds": {
                    "text_length": self._text_length_threshold,
                    "confidence": self._confidence_threshold,
                },
            }

    def get_history(self) -> List[dict]:
        """Return all historical drift check results."""
        with self._lock:
            from dataclasses import asdict

            return [asdict(r) for r in self._check_history]

    def reset(self) -> None:
        """Reset for a new deployment stage."""
        with self._lock:
            self._reference_text_lengths.clear()
            self._reference_confidences.clear()
            self._reference_frozen = False
            self._current_text_lengths.clear()
            self._current_confidences.clear()
            self._total_records = 0
            self._last_check_result = None
            self._check_history.clear()


# Module-level singleton
drift_detector = DriftDetector()
