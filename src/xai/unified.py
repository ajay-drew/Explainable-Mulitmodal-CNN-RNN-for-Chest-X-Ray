"""
Unified Multimodal Explainability.

Per PROJECT_PLAN §3.5 and §5.6:
- Combine image Grad-CAM with text token attributions
- Novel ensemble approach (weighted attribution scores)
- Faithfulness evaluation (>92% target)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .gradcam import GradCAM, overlay_heatmap
from .text_attribution import TextAttributor, SHAPAttributor


@dataclass
class MultimodalExplanation:
    """Container for unified multimodal explanation."""
    
    # Image explanation
    image_heatmap: np.ndarray  # [H, W]
    image_overlay: Optional[np.ndarray] = None  # [H, W, 3]
    
    # Text explanation
    token_attributions: List[Tuple[str, float]] = None  # [(token, score), ...]
    highlighted_text: Optional[str] = None  # HTML
    
    # Unified scores
    image_contribution: float = 0.0  # How much image contributed
    text_contribution: float = 0.0   # How much text contributed
    
    # Prediction info
    predicted_class: str = ""
    prediction_probability: float = 0.0
    
    # Faithfulness score
    faithfulness_score: Optional[float] = None


class UnifiedExplainer:
    """
    Unified explainer combining image and text explanations.
    
    Per PROJECT_PLAN §5.6:
    - Image: Grad-CAM heatmaps
    - Text: SHAP/IG token attributions
    - Unified: Weighted combination with faithfulness evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        text_attributor: Optional[TextAttributor] = None,
        tokenizer = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize unified explainer.
        
        Args:
            model: Multimodal classifier
            target_layer: CNN layer for Grad-CAM
            text_attributor: Text attribution method (default: SHAP)
            tokenizer: Tokenizer for decoding tokens
            class_names: List of class names
        """
        self.model = model
        self.gradcam = GradCAM(model, target_layer)
        self.text_attributor = text_attributor or SHAPAttributor()
        self.tokenizer = tokenizer
        self.class_names = class_names or []
    
    def explain(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None,
        original_image: Optional[np.ndarray] = None,
        original_text: Optional[str] = None,
        top_k_tokens: int = 10,
        compute_faithfulness: bool = True,
    ) -> MultimodalExplanation:
        """
        Generate unified multimodal explanation.
        
        Args:
            image: Input image [1, C, H, W]
            input_ids: Tokenized report [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            target_class: Class to explain (None = predicted)
            original_image: Original image for overlay
            original_text: Original text for display
            top_k_tokens: Number of top tokens to return
            compute_faithfulness: Whether to compute faithfulness score
            
        Returns:
            MultimodalExplanation with all components
        """
        self.model.eval()
        device = image.device
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(image, input_ids, attention_mask)
            probs = torch.sigmoid(logits)
        
        if target_class is None:
            target_class = probs[0].argmax().item()
        
        pred_prob = probs[0, target_class].item()
        class_name = self.class_names[target_class] if target_class < len(self.class_names) else str(target_class)
        
        # === Image Explanation (Grad-CAM) ===
        heatmap = self.gradcam.generate(
            image, input_ids, attention_mask, target_class
        ).cpu().numpy()
        
        # Create overlay if original image provided
        image_overlay = None
        if original_image is not None:
            image_overlay = overlay_heatmap(original_image, heatmap)
        
        # === Text Explanation ===
        text_attributions = self.text_attributor.attribute(
            self.model, input_ids, attention_mask, image, target_class
        )
        
        top_tokens = []
        if self.tokenizer is not None:
            top_tokens = self.text_attributor.get_top_tokens(
                text_attributions, input_ids, self.tokenizer, top_k_tokens
            )
        
        # === Compute Modality Contributions ===
        image_contrib, text_contrib = self._compute_contributions(
            image, input_ids, attention_mask, target_class
        )
        
        # === Faithfulness Evaluation ===
        faithfulness = None
        if compute_faithfulness:
            faithfulness = self._evaluate_faithfulness(
                image, input_ids, attention_mask,
                heatmap, text_attributions, target_class
            )
        
        return MultimodalExplanation(
            image_heatmap=heatmap,
            image_overlay=image_overlay,
            token_attributions=top_tokens,
            highlighted_text=original_text,
            image_contribution=image_contrib,
            text_contribution=text_contrib,
            predicted_class=class_name,
            prediction_probability=pred_prob,
            faithfulness_score=faithfulness,
        )
    
    def _compute_contributions(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
    ) -> Tuple[float, float]:
        """
        Compute relative contribution of each modality.
        
        Uses ablation: measure prediction drop when removing each modality.
        """
        with torch.no_grad():
            # Full prediction
            full_logits = self.model(image, input_ids, attention_mask)
            full_prob = torch.sigmoid(full_logits[0, target_class]).item()
            
            # Zero out image features
            zero_image = torch.zeros_like(image)
            image_ablated_logits = self.model(zero_image, input_ids, attention_mask)
            image_ablated_prob = torch.sigmoid(image_ablated_logits[0, target_class]).item()
            
            # Zero out text (use all padding)
            zero_text = torch.zeros_like(input_ids)
            zero_mask = torch.zeros_like(attention_mask)
            text_ablated_logits = self.model(image, zero_text, zero_mask)
            text_ablated_prob = torch.sigmoid(text_ablated_logits[0, target_class]).item()
        
        # Contribution = drop in probability when ablated
        image_contrib = max(0, full_prob - image_ablated_prob)
        text_contrib = max(0, full_prob - text_ablated_prob)
        
        # Normalize
        total = image_contrib + text_contrib + 1e-8
        return image_contrib / total, text_contrib / total
    
    def _evaluate_faithfulness(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heatmap: np.ndarray,
        text_attributions: torch.Tensor,
        target_class: int,
        top_percent: float = 0.2,
    ) -> float:
        """
        Evaluate faithfulness of explanations.
        
        Per PROJECT_PLAN: >92% faithfulness target.
        
        Method: Remove top attributed features and measure prediction drop.
        Higher drop = more faithful explanation.
        """
        with torch.no_grad():
            # Baseline prediction
            base_logits = self.model(image, input_ids, attention_mask)
            base_prob = torch.sigmoid(base_logits[0, target_class]).item()
        
        # === Image faithfulness ===
        # Mask top regions of heatmap
        threshold = np.percentile(heatmap, 100 * (1 - top_percent))
        mask = (heatmap >= threshold).astype(np.float32)
        
        # Create masked image
        mask_tensor = torch.from_numpy(mask).to(image.device).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=image.shape[-2:], mode='bilinear'
        )
        masked_image = image * (1 - mask_tensor)
        
        with torch.no_grad():
            image_masked_logits = self.model(masked_image, input_ids, attention_mask)
            image_masked_prob = torch.sigmoid(image_masked_logits[0, target_class]).item()
        
        image_faithfulness = (base_prob - image_masked_prob) / (base_prob + 1e-8)
        
        # === Text faithfulness ===
        # Mask top tokens
        n_tokens = int(top_percent * attention_mask.sum().item())
        top_token_indices = text_attributions.abs().topk(max(1, n_tokens)).indices
        
        masked_ids = input_ids.clone()
        for idx in top_token_indices:
            masked_ids[0, idx] = 0  # Mask with PAD
        
        with torch.no_grad():
            text_masked_logits = self.model(image, masked_ids, attention_mask)
            text_masked_prob = torch.sigmoid(text_masked_logits[0, target_class]).item()
        
        text_faithfulness = (base_prob - text_masked_prob) / (base_prob + 1e-8)
        
        # Combined faithfulness (average)
        faithfulness = (image_faithfulness + text_faithfulness) / 2
        
        return max(0, min(1, faithfulness))


def create_explanation_report(
    explanation: MultimodalExplanation,
    save_path: Optional[str] = None,
) -> str:
    """
    Create a human-readable explanation report.
    
    Args:
        explanation: MultimodalExplanation object
        save_path: Optional path to save HTML report
        
    Returns:
        HTML report string
    """
    html = f"""
    <html>
    <head>
        <title>Multimodal Explanation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .positive {{ background-color: rgba(0, 255, 0, 0.3); }}
            .negative {{ background-color: rgba(255, 0, 0, 0.3); }}
        </style>
    </head>
    <body>
        <h1>Chest X-Ray Diagnosis Explanation</h1>
        
        <div class="section">
            <h2>Prediction</h2>
            <p><strong>Class:</strong> {explanation.predicted_class}</p>
            <p><strong>Probability:</strong> {explanation.prediction_probability:.2%}</p>
        </div>
        
        <div class="section">
            <h2>Modality Contributions</h2>
            <p><strong>Image:</strong> {explanation.image_contribution:.1%}</p>
            <p><strong>Text:</strong> {explanation.text_contribution:.1%}</p>
        </div>
        
        <div class="section">
            <h2>Top Contributing Tokens</h2>
            <ul>
    """
    
    for token, score in (explanation.token_attributions or []):
        css_class = "positive" if score > 0 else "negative"
        html += f'<li class="{css_class}">"{token}": {score:.4f}</li>'
    
    html += f"""
            </ul>
        </div>
        
        <div class="section">
            <h2>Faithfulness Score</h2>
            <p>{explanation.faithfulness_score:.2%} (target: >92%)</p>
        </div>
    </body>
    </html>
    """
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(html)
    
    return html
