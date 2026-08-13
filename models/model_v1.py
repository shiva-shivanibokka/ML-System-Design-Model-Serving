"""
Model V1 — DistilBERT SST-2 (full precision FP32).

This is the stable production model. It runs at full float32 precision.
All traffic goes through v1 until the deployment state machine promotes v2.
"""

from __future__ import annotations

import structlog
from transformers import pipeline

from configs.settings import settings
from models.base import BaseModel

log = structlog.get_logger(__name__)


class ModelV1(BaseModel):
    """
    DistilBERT fine-tuned on SST-2, full FP32 precision.

    HuggingFace model: distilbert-base-uncased-finetuned-sst-2-english
    Size: ~268MB
    Expected inference latency (CPU): 30-80ms per request
    """

    def __init__(self) -> None:
        super().__init__(version="v1")
        self._pipeline = None

    def load(self) -> None:
        """
        Download model from HuggingFace Hub (cached after first download)
        and build the transformers inference pipeline.
        """
        log.info("loading_model", version=self.version, model=settings.models.v1.name)
        self._pipeline = pipeline(
            task="text-classification",
            model=settings.models.v1.name,
            device=-1,  # -1 = CPU; set to 0 for first GPU
            truncation=True,
            max_length=settings.models.v1.max_length,
            # No return_all_scores/top_k: the default already returns the single
            # top label as [{label, score}], which is what _run_inference
            # unpacks. top_k=1 is NOT equivalent — it nests one level deeper
            # ([[{...}]]) and would break the caller.
        )
        self._loaded = True
        log.info("model_loaded", version=self.version)

    def _run_inference(self, text: str) -> tuple[str, float]:
        """Single forward pass. Returns (label, confidence_score)."""
        result = self._pipeline(text)[0]
        label: str = result["label"]  # "POSITIVE" | "NEGATIVE"
        score: float = float(result["score"])
        return label, score
