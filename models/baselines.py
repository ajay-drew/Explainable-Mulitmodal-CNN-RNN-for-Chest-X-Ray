"""
Baseline models for comparison against the zero-shot XRV pipeline.

All baselines share the same interface:
    predict(image_tensor) → dict with 'logits', 'probs', 'features'

Classes:
    ZeroShotBaseline    — XRV built-in pathology head, no modification
    CNNOnlyBaseline     — Same as ZeroShotBaseline (identical in zero-shot mode)
    TextOnlyBaseline    — RadBERT CLS → linear probe (for ablation)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot baseline (primary)
# ─────────────────────────────────────────────────────────────────────────────

class ZeroShotBaseline(nn.Module):
    """
    TorchXRayVision DenseNet121 used exactly as released — no modification.

    This is the strongest baseline: pretrained on MIMIC + CheXpert + NIH + RSNA.
    The model's own 18-label head is used; outputs are mapped to our 15 labels.
    Zero fine-tuning, zero gradient updates.

    Args:
        weights:       XRV weights string.
        target_labels: Our 15 label names.
    """

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        target_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # Defer import to avoid top-level dependency issues in environments
        # without torchxrayvision installed
        from models.image_encoder import XRVImageEncoder
        self.encoder = XRVImageEncoder(weights=weights, target_labels=target_labels)
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, 224, 224) float32 tensor, range [-1024, 1024].

        Returns:
            dict with 'logits' (B, 15), 'probs' (B, 15), 'features' (B, 1024).
        """
        out = self.encoder(x)
        probs = torch.sigmoid(out["logits"])
        return {
            "logits": out["logits"],
            "probs": probs,
            "features": out["features"],
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for forward with no_grad context."""
        self.eval()
        return self.forward(x)


# ─────────────────────────────────────────────────────────────────────────────
# CNN-only baseline (same as zero-shot in this architecture)
# ─────────────────────────────────────────────────────────────────────────────

class CNNOnlyBaseline(ZeroShotBaseline):
    """
    CNN-only baseline: frozen XRV DenseNet121, no text branch.
    Identical to ZeroShotBaseline in the zero-training setting.
    Provided as a named alternative for evaluation tables.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Text-only baseline (RadBERT CLS → linear probe)
# ─────────────────────────────────────────────────────────────────────────────

class TextOnlyBaseline(nn.Module):
    """
    Text-only baseline: RadBERT frozen CLS → untrained linear layer → 15 classes.

    Because the projection is NOT fine-tuned (zero training), this is essentially
    a random linear projection from RadBERT CLS features. Its AUROC will be near
    0.5 for most pathologies, demonstrating the value of the image branch.

    Args:
        model_name:    RadBERT HuggingFace identifier.
        num_classes:   Number of output classes.
    """

    def __init__(
        self,
        model_name: str = "StanfordAIMI/RadBERT",
        num_classes: int = 15,
    ) -> None:
        super().__init__()
        from models.text_encoder import RadBERTEncoder
        self.encoder = RadBERTEncoder(model_name=model_name)
        # Linear probe — intentionally NOT trained in zero-shot mode
        self.head = nn.Linear(512, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (B, seq_len) LongTensor.
            attention_mask: (B, seq_len) LongTensor.

        Returns:
            dict with 'logits' (B, num_classes), 'probs' (B, num_classes).
        """
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(enc_out["projected"])
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probs": probs}

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward with no_grad."""
        self.eval()
        return self.forward(input_ids, attention_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_inference_time(
    model: nn.Module,
    dummy_input: torch.Tensor,
    n_runs: int = 10,
) -> float:
    """
    Measure average inference time in milliseconds.

    Args:
        model:       Model to benchmark.
        dummy_input: Batch tensor appropriate for the model.
        n_runs:      Number of forward passes to average over.

    Returns:
        Average inference time in milliseconds.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return sum(times) / len(times)
