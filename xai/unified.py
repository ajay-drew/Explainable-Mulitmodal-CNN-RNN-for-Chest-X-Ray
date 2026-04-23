"""
UnifiedExplainer: orchestrates Grad-CAM, SHAP, Integrated Gradients,
attention weights, faithfulness scoring, and NL summaries into a single
ExplanationResult for any chest X-ray + optional radiology report input.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from xai.nlp_summary import NLExplainer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    """
    Container for all XAI outputs from a single inference.

    Fields:
        disease_name:          Predicted pathology label.
        confidence:            Prediction confidence in [0, 1].
        heatmap:               (H, W) float32 Grad-CAM heatmap, range [0, 1].
        heatmap_overlay:       PIL Image with heatmap overlaid on X-ray.
        token_shap:            SHAP token attributions [(token, score), ...].
        token_ig:              IG token attributions [(token, score), ...].
        attention_weights:     Dict token → average attention weight.
        faithfulness_score:    Float in [0, 1].
        nl_summary:            Natural language explanation string.
        modality_contributions: Dict with 'image_pct' and 'text_pct'.
        all_predictions:       Dict label → confidence for all 15 classes.
    """

    disease_name: str = ""
    confidence: float = 0.0
    heatmap: Optional[np.ndarray] = None
    heatmap_overlay: Optional[Image.Image] = None
    token_shap: List[Tuple[str, float]] = field(default_factory=list)
    token_ig: List[Tuple[str, float]] = field(default_factory=list)
    attention_weights: Dict[str, float] = field(default_factory=dict)
    faithfulness_score: float = 0.0
    nl_summary: str = ""
    modality_contributions: Dict[str, float] = field(
        default_factory=lambda: {"image_pct": 100.0, "text_pct": 0.0}
    )
    all_predictions: Dict[str, float] = field(default_factory=dict)

    def as_json(self) -> str:
        """Serialize to JSON string (PIL Image and numpy arrays excluded)."""
        safe = {
            "disease_name": self.disease_name,
            "confidence": self.confidence,
            "faithfulness_score": self.faithfulness_score,
            "nl_summary": self.nl_summary,
            "modality_contributions": self.modality_contributions,
            "all_predictions": self.all_predictions,
            "token_shap": self.token_shap,
            "token_ig": self.token_ig,
            "attention_weights": self.attention_weights,
        }
        return json.dumps(safe, indent=2)

    def as_html(self) -> str:
        """Render a styled HTML report string summarising the explanation."""
        pred_rows = "".join(
            f"<tr><td>{lbl}</td><td>{conf*100:.1f}%</td></tr>"
            for lbl, conf in sorted(
                self.all_predictions.items(), key=lambda x: x[1], reverse=True
            )
            if conf > 0.01
        )
        shap_rows = "".join(
            f"<tr><td>{tok}</td><td style='color:{'red' if s>0 else 'blue'}'>"
            f"{s:+.4f}</td></tr>"
            for tok, s in self.token_shap[:10]
        )
        html = f"""
        <html><body style="font-family:Arial,sans-serif; max-width:800px; margin:auto">
        <h2>XAI Explanation Report</h2>
        <h3>Prediction: {self.disease_name} ({self.confidence*100:.1f}%)</h3>
        <p><b>NL Summary:</b> {self.nl_summary}</p>
        <p><b>Faithfulness Score:</b> {self.faithfulness_score:.3f}</p>
        <h4>All Class Probabilities</h4>
        <table border="1" cellpadding="4"><tr><th>Label</th><th>Confidence</th></tr>
        {pred_rows}</table>
        <h4>Top SHAP Token Attributions</h4>
        <table border="1" cellpadding="4"><tr><th>Token</th><th>Score</th></tr>
        {shap_rows}</table>
        <h4>Modality Contributions</h4>
        <p>Image: {self.modality_contributions.get('image_pct',100):.1f}% &nbsp;|&nbsp;
           Text: {self.modality_contributions.get('text_pct',0):.1f}%</p>
        </body></html>
        """
        return html


# ─────────────────────────────────────────────────────────────────────────────
# UnifiedExplainer
# ─────────────────────────────────────────────────────────────────────────────

class UnifiedExplainer:
    """
    Orchestrates Grad-CAM, SHAP, IG, attention extraction, faithfulness,
    and NL summary generation for a chest X-ray + optional radiology report.

    Args:
        image_encoder:    XRVImageEncoder instance.
        text_encoder:     RadBERTEncoder instance (or None if no text branch).
        tokenizer:        HuggingFace tokenizer for RadBERT.
        device:           Torch device.
        gradcam_method:   One of "gradcam", "gradcam++", "eigencam".
        n_ig_steps:       IG integration steps.
        n_shap_samples:   Unused here (SHAP uses full masking, not sampling).
        labels:           Ordered list of 15 pathology label names.
    """

    def __init__(
        self,
        image_encoder,
        text_encoder=None,
        tokenizer=None,
        device: Optional[torch.device] = None,
        gradcam_method: str = "gradcam",
        n_ig_steps: int = 50,
        n_shap_samples: int = 100,
        labels: Optional[List[str]] = None,
    ) -> None:
        from xai.gradcam import XRVGradCAM, compute_faithfulness
        from xai.text_attribution import (
            IntegratedGradientsAttributor,
            SHAPAttributor,
            extract_attention_weights,
        )

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.labels = labels or [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration",
            "Mass", "No Finding", "Nodule", "Pleural Thickening",
            "Pneumonia", "Pneumothorax",
        ]

        self.gradcam = XRVGradCAM(image_encoder, method=gradcam_method)
        self._compute_faithfulness = compute_faithfulness
        self.nl_explainer = NLExplainer()

        if text_encoder is not None and tokenizer is not None:
            self.shap = SHAPAttributor(
                text_encoder, tokenizer, device=self.device
            )
            self.ig = IntegratedGradientsAttributor(
                text_encoder, tokenizer, device=self.device, n_steps=n_ig_steps
            )
        else:
            self.shap = None
            self.ig = None

        self._extract_attention = extract_attention_weights

    def explain(
        self,
        image_tensor: torch.Tensor,
        original_pil: Image.Image,
        report_text: Optional[str] = None,
        threshold: float = 0.5,
    ) -> ExplanationResult:
        """
        Run the full XAI pipeline for one image + optional radiology report.

        Args:
            image_tensor:  (1, 1, 224, 224) float32 tensor, range [-1024, 1024].
            original_pil:  Original chest X-ray PIL Image for overlay.
            report_text:   Optional radiology report string.
            threshold:     Confidence threshold for positive predictions.

        Returns:
            ExplanationResult with all XAI outputs populated.
        """
        result = ExplanationResult()
        image_tensor = image_tensor.to(self.device)

        # ── 1. Image inference ──────────────────────────────────────────────
        self.image_encoder.eval()
        with torch.no_grad():
            enc_out = self.image_encoder(image_tensor)

        logits = enc_out["logits"][0]      # (15,)
        probs = torch.sigmoid(logits)      # (15,)
        image_features = enc_out["features"][0]  # (1024,)

        all_preds = {
            self.labels[i]: float(probs[i]) for i in range(len(self.labels))
        }
        result.all_predictions = all_preds

        # Top predicted class
        top_idx = int(probs.argmax())
        result.disease_name = self.labels[top_idx]
        result.confidence = float(probs[top_idx])

        # ── 2. Grad-CAM ─────────────────────────────────────────────────────
        try:
            heatmap = self.gradcam.generate_heatmap(image_tensor, top_idx)
            overlay = self.gradcam.overlay_heatmap(original_pil, heatmap)
            result.heatmap = heatmap
            result.heatmap_overlay = overlay
        except Exception as exc:
            logger.warning(f"Grad-CAM failed: {exc}")
            result.heatmap = np.zeros((224, 224), dtype=np.float32)

        # ── 3. Faithfulness ─────────────────────────────────────────────────
        if result.heatmap is not None and result.heatmap.any():
            try:
                result.faithfulness_score = self._compute_faithfulness(
                    self.image_encoder, image_tensor, result.heatmap, top_idx
                )
            except Exception as exc:
                logger.warning(f"Faithfulness computation failed: {exc}")

        # ── 4. Text attributions (if report provided) ────────────────────────
        text_features = None
        if report_text and self.text_encoder and self.tokenizer:
            try:
                from xai.text_attribution import _tokenize
                enc = _tokenize(report_text, self.tokenizer)
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                # SHAP
                result.token_shap = self.shap.explain(report_text, target_class=0)

                # Integrated Gradients
                result.token_ig = self.ig.explain(report_text, target_class=0)

                # Attention weights
                result.attention_weights = self._extract_attention(
                    self.text_encoder, input_ids, attention_mask,
                    self.device, self.tokenizer
                )

                # Text features for modality contribution
                with torch.no_grad():
                    text_out = self.text_encoder(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                text_features = text_out["projected"][0]  # (512,)

            except Exception as exc:
                logger.warning(f"Text attribution failed: {exc}")

        # ── 5. Modality contributions ────────────────────────────────────────
        if text_features is not None:
            # Cosine similarity proxy: compare norm of each feature vector
            img_norm = float(image_features.norm())
            txt_norm = float(text_features.norm())
            total = img_norm + txt_norm + 1e-9
            result.modality_contributions = {
                "image_pct": round(img_norm / total * 100, 1),
                "text_pct": round(txt_norm / total * 100, 1),
            }
        else:
            result.modality_contributions = {"image_pct": 100.0, "text_pct": 0.0}

        # ── 6. NL Summary ────────────────────────────────────────────────────
        if result.heatmap is not None:
            result.nl_summary = self.nl_explainer.explain_image_prediction(
                disease_name=result.disease_name,
                confidence=result.confidence,
                heatmap=result.heatmap,
            )
        if text_features is not None and result.token_shap:
            mod_line = self.nl_explainer.explain_modality_contribution(
                result.modality_contributions["image_pct"],
                result.modality_contributions["text_pct"],
            )
            result.nl_summary += " " + mod_line

        return result
