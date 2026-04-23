"""
Rule-based natural language explanation summaries.
No LLM calls — pure template filling based on heatmap geometry and token scores.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# Anatomical region names for each heatmap quadrant
_QUADRANT_LABELS = {
    "upper_left":  "the left upper lobe",
    "upper_right": "the right upper lobe",
    "lower_left":  "the left lower lobe",
    "lower_right": "the right lower lobe",
}

# Ordinal suffix for secondary region
_ORDINAL = {1: "second", 2: "third", 3: "fourth"}

# Disease-specific radiological context
_DISEASE_CONTEXT = {
    "Atelectasis":       "partial or complete lung collapse with linear opacities",
    "Cardiomegaly":      "an enlarged cardiac silhouette relative to the thoracic diameter",
    "Consolidation":     "homogeneous opacity replacing normal aeration",
    "Edema":             "bilateral perihilar opacities and Kerley B lines",
    "Effusion":          "blunting of the costophrenic angle and fluid layering",
    "Emphysema":         "hyperinflation with flattened diaphragms and increased lucency",
    "Fibrosis":          "reticular interstitial opacities and honeycombing",
    "Hernia":            "herniated abdominal contents visible above the diaphragm",
    "Infiltration":      "patchy or diffuse areas of increased opacity",
    "Mass":              "a focal dense opacity with well- or ill-defined margins",
    "No Finding":        "a normal chest radiograph with no focal abnormality",
    "Nodule":            "a rounded opacity smaller than 3 cm",
    "Pleural Thickening": "thickening of the pleural lining along the chest wall",
    "Pneumonia":         "lobar or segmental consolidation with air bronchograms",
    "Pneumothorax":      "absence of lung markings with a visible pleural line",
}


class NLExplainer:
    """
    Generates natural language explanations for model predictions.
    All logic is rule-based template filling — no external model calls.
    """

    def explain_image_prediction(
        self,
        disease_name: str,
        confidence: float,
        heatmap: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Generate a natural language explanation for a chest X-ray prediction.

        Args:
            disease_name:  Predicted pathology label string.
            confidence:    Prediction confidence in [0, 1].
            heatmap:       2D float32 numpy array (H, W), range [0, 1].
            image_shape:   (H, W) of the original image (ignored; heatmap dims used).

        Returns:
            Explanation string describing the prediction and anatomical focus.
        """
        # Divide heatmap into 4 quadrants and compute mean activation
        h, w = heatmap.shape
        mid_h, mid_w = h // 2, w // 2

        quadrant_means = {
            "upper_left":  float(heatmap[:mid_h, :mid_w].mean()),
            "upper_right": float(heatmap[:mid_h, mid_w:].mean()),
            "lower_left":  float(heatmap[mid_h:, :mid_w].mean()),
            "lower_right": float(heatmap[mid_h:, mid_w:].mean()),
        }

        # Sort quadrants by mean activation (descending)
        sorted_quadrants = sorted(
            quadrant_means.items(), key=lambda x: x[1], reverse=True
        )
        primary_region = _QUADRANT_LABELS[sorted_quadrants[0][0]]
        secondary_region = _QUADRANT_LABELS[sorted_quadrants[1][0]]

        context = _DISEASE_CONTEXT.get(
            disease_name,
            "radiological findings consistent with this pathology",
        )
        conf_pct = confidence * 100 if confidence <= 1.0 else confidence

        summary = (
            f"The model predicted {disease_name} with {conf_pct:.0f}% confidence. "
            f"Analysis focused primarily on {primary_region}, with secondary attention "
            f"to {secondary_region}. "
            f"This pattern is consistent with typical radiological findings for "
            f"{disease_name}: {context}."
        )
        return summary

    def explain_text_prediction(
        self,
        label: str,
        confidence: float,
        top_tokens: List[Tuple[str, float]],
    ) -> str:
        """
        Generate a natural language explanation for a text classification prediction.

        Args:
            label:       Predicted label string (e.g. "positive", "negative").
            confidence:  Prediction confidence in [0, 1].
            top_tokens:  List of (token, score) pairs sorted by |score| descending.

        Returns:
            Explanation string describing the classification signal.
        """
        conf_pct = confidence * 100 if confidence <= 1.0 else confidence

        # Top-3 tokens for the summary
        top_3 = [tok for tok, _ in top_tokens[:3]]
        if top_3:
            token_list = ", ".join(f'"{t}"' for t in top_3)
        else:
            token_list = "(no significant tokens identified)"

        # Determine dominant language pattern
        pos_score = sum(s for _, s in top_tokens if s > 0)
        neg_score = sum(abs(s) for _, s in top_tokens if s < 0)
        dominant = "Positive" if pos_score >= neg_score else "Negative"

        summary = (
            f"The model classified this text as {label.upper()} with "
            f"{conf_pct:.0f}% confidence. "
            f"The most influential terms were: {token_list}. "
            f"{dominant} language patterns dominated the classification signal."
        )
        return summary

    def explain_modality_contribution(
        self,
        image_similarity: float,
        text_similarity: float,
    ) -> str:
        """
        Generate a sentence describing relative modality contributions.

        Args:
            image_similarity: Image feature cosine similarity score (proxy).
            text_similarity:  Text feature cosine similarity score (proxy).

        Returns:
            One-sentence contribution summary.
        """
        total = image_similarity + text_similarity + 1e-9
        img_pct = (image_similarity / total) * 100
        txt_pct = (text_similarity / total) * 100

        if img_pct > txt_pct + 10:
            dominant = "imaging features"
        elif txt_pct > img_pct + 10:
            dominant = "report text features"
        else:
            dominant = "both imaging and text features equally"

        return (
            f"The prediction was driven primarily by {dominant} "
            f"(image: {img_pct:.0f}%, text: {txt_pct:.0f}%)."
        )
