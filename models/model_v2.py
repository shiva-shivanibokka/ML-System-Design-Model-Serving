"""
Model V2 — DistilBERT SST-2 (INT8 dynamic quantization).

This is the candidate model being evaluated via shadow mode and canary rollout.
INT8 dynamic quantization is applied post-load using torch.quantization.
This simulates a realistic v2 scenario: same architecture, different precision/optimization.

Why INT8 quantization as v2?
  - It's a genuinely different model that produces slightly different confidence scores
    even when the label agrees — this makes shadow disagreement monitoring meaningful.
  - It's ~30% faster and ~4x smaller in memory — a real production tradeoff.
  - It represents the most common type of "v2" in production: same weights, different
    serving optimization (quantization, pruning, ONNX export).

Production equivalent: INT8 quantization is used at Google, Meta, and Hugging Face
to reduce inference cost for BERT-family models without retraining.
"""

from __future__ import annotations

import structlog
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from configs.settings import settings
from models.base import BaseModel

log = structlog.get_logger(__name__)


class ModelV2(BaseModel):
    """
    DistilBERT fine-tuned on SST-2, INT8 dynamic quantization.

    Quantization applied via torch.quantization.quantize_dynamic — targets
    all nn.Linear layers (the dominant cost in transformer inference).

    Expected latency improvement over v1 (CPU): ~20-40% faster
    Memory reduction: ~4x (FP32 → INT8 for weight storage)
    """

    def __init__(self) -> None:
        super().__init__(version="v2")
        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """
        1. Download same weights as v1 from HuggingFace Hub.
        2. Apply INT8 dynamic quantization to all Linear layers.
        3. Build inference pipeline around the quantized model.

        Dynamic quantization:
        - Weights are quantized to INT8 at load time (static).
        - Activations are quantized dynamically per-batch at runtime.
        - No calibration dataset needed — works out of the box.
        - Linear layers are targeted because attention + FFN are ~95% of compute.
        """
        model_name = settings.models.v2.name
        log.info(
            "loading_model",
            version=self.version,
            model=model_name,
            quantization="int8_dynamic",
        )

        # Load tokenizer and model separately so we can quantize the model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Apply INT8 dynamic quantization
        # torch.quantization.quantize_dynamic replaces all nn.Linear layers
        # with dynamically quantized counterparts (IntX8Linear)
        self._model = torch.quantization.quantize_dynamic(
            self._model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        self._model.eval()

        log.info(
            "quantization_applied",
            version=self.version,
            method="dynamic_int8",
            target_layers="nn.Linear",
        )

        # Build a HuggingFace pipeline using the quantized model + tokenizer
        self._pipeline = pipeline(
            task="text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=-1,  # CPU — quantized INT8 is CPU-optimized
            truncation=True,
            max_length=settings.models.v2.max_length,
            # No return_all_scores/top_k: the default already returns the single
            # top label as [{label, score}], which is what _run_inference
            # unpacks. top_k=1 is NOT equivalent — it nests one level deeper
            # ([[{...}]]) and would break the caller.
        )

        self._loaded = True
        log.info("model_loaded", version=self.version, quantized=True)

    def _run_inference(self, text: str) -> tuple[str, float]:
        """Single forward pass through the quantized model."""
        result = self._pipeline(text)[0]
        label: str = result["label"]
        score: float = float(result["score"])
        return label, score
