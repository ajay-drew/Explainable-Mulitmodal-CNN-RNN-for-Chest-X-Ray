"""Explainability (XAI) module."""

from .gradcam import GradCAM, generate_gradcam_heatmap
from .text_attribution import (
    TextAttributor,
    IntegratedGradientsAttributor,
    SHAPAttributor,
)
from .unified import UnifiedExplainer

__all__ = [
    "GradCAM",
    "generate_gradcam_heatmap",
    "TextAttributor",
    "IntegratedGradientsAttributor",
    "SHAPAttributor",
    "UnifiedExplainer",
]
