"""
Text Attribution for explainability.

Per PROJECT_PLAN §10:
- SHAP for token-level importance
- Integrated Gradients for attribution
- Works with BERT encoder (RadBERT / Bio_ClinicalBERT)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class TextAttributor(ABC):
    """Base class for text attribution methods."""
    
    @abstractmethod
    def attribute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        Compute attribution scores for tokens.
        
        Returns:
            Attribution scores [seq_len]
        """
        pass
    
    @abstractmethod
    def get_top_tokens(
        self,
        attributions: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top attributed tokens."""
        pass


class IntegratedGradientsAttributor(TextAttributor):
    """
    Integrated Gradients for text attribution.
    
    Per PROJECT_PLAN §10:
    - Gradient of output w.r.t. input embeddings
    - Sum along path from baseline to input
    """
    
    def __init__(self, n_steps: int = 50):
        """
        Args:
            n_steps: Number of interpolation steps
        """
        self.n_steps = n_steps
    
    def attribute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            model: Multimodal classifier
            input_ids: Token IDs [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            image: Input image [1, C, H, W]
            target_class: Class to explain
            
        Returns:
            Attribution scores [seq_len]
        """
        model.eval()
        
        # Get input embeddings
        text_encoder = model.text_encoder
        embeddings = text_encoder.get_input_embeddings(input_ids)  # [1, seq_len, embed_dim]
        
        # Baseline: zero embeddings
        baseline = torch.zeros_like(embeddings)
        
        # Interpolate between baseline and input
        scaled_inputs = []
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            scaled = baseline + alpha * (embeddings - baseline)
            scaled_inputs.append(scaled)
        
        # Compute gradients at each step
        gradients = []
        for scaled_embedding in scaled_inputs:
            scaled_embedding.requires_grad_(True)
            
            # Forward pass with embeddings (need to modify text encoder to accept embeddings)
            # This is a simplified version; full implementation needs model modification
            
            # For now, compute gradient of logits w.r.t. embeddings
            outputs = self._forward_with_embeddings(
                model, scaled_embedding, attention_mask, image
            )
            
            # Backward for target class
            model.zero_grad()
            outputs[0, target_class].backward(retain_graph=True)
            
            gradients.append(scaled_embedding.grad.detach())
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients = (input - baseline) * avg_gradients
        integrated_grads = (embeddings - baseline) * avg_gradients
        
        # Sum over embedding dimension to get per-token attribution
        attributions = integrated_grads.sum(dim=-1).squeeze()  # [seq_len]
        
        return attributions
    
    def _forward_with_embeddings(
        self,
        model: nn.Module,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using embeddings instead of input_ids.
        
        Note: This requires model modification to accept embeddings.
        For now, this is a placeholder.
        """
        # TODO: Modify TextEncoder to accept embeddings directly
        # This is needed for proper IG computation
        raise NotImplementedError(
            "Full IG implementation requires model modification to accept embeddings. "
            "Use captum library for production implementation."
        )
    
    def get_top_tokens(
        self,
        attributions: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top attributed tokens."""
        # Get absolute attributions
        abs_attr = attributions.abs()
        
        # Get top k indices
        top_indices = abs_attr.topk(min(top_k, len(abs_attr))).indices
        
        # Decode tokens
        results = []
        for idx in top_indices:
            token_id = input_ids[0, idx].item()
            token = tokenizer.decode([token_id])
            score = attributions[idx].item()
            results.append((token, score))
        
        return results


class SHAPAttributor(TextAttributor):
    """
    SHAP-based text attribution.
    
    Per PROJECT_PLAN §10:
    - Token-level Shapley values
    - Uses masking/perturbation
    """
    
    def __init__(self, n_samples: int = 100, mask_token_id: int = 0):
        """
        Args:
            n_samples: Number of perturbation samples
            mask_token_id: Token ID to use for masking (typically PAD or MASK)
        """
        self.n_samples = n_samples
        self.mask_token_id = mask_token_id
    
    def attribute(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        Compute SHAP attribution via perturbation.
        
        Simplified implementation: measures effect of masking each token.
        For full SHAP, use the shap library.
        """
        model.eval()
        device = input_ids.device
        seq_len = input_ids.size(1)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_logits = model(image, input_ids, attention_mask)
            baseline_prob = torch.sigmoid(baseline_logits[0, target_class])
        
        # Compute importance of each token by masking
        attributions = torch.zeros(seq_len, device=device)
        
        for i in range(seq_len):
            # Skip padding tokens
            if attention_mask[0, i] == 0:
                continue
            
            # Mask token i
            masked_ids = input_ids.clone()
            masked_ids[0, i] = self.mask_token_id
            
            # Get prediction with masked token
            with torch.no_grad():
                masked_logits = model(image, masked_ids, attention_mask)
                masked_prob = torch.sigmoid(masked_logits[0, target_class])
            
            # Attribution = difference (how much prediction drops)
            attributions[i] = baseline_prob - masked_prob
        
        return attributions
    
    def get_top_tokens(
        self,
        attributions: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get top attributed tokens."""
        # Get absolute attributions
        abs_attr = attributions.abs()
        
        # Get top k indices
        top_indices = abs_attr.topk(min(top_k, len(abs_attr))).indices
        
        # Decode tokens
        results = []
        for idx in top_indices:
            token_id = input_ids[0, idx].item()
            token = tokenizer.decode([token_id])
            score = attributions[idx].item()
            results.append((token, score))
        
        return results


def highlight_text(
    text: str,
    token_scores: List[Tuple[str, float]],
    positive_color: str = "green",
    negative_color: str = "red",
) -> str:
    """
    Generate HTML with highlighted tokens based on attribution.
    
    Args:
        text: Original text
        token_scores: List of (token, score) tuples
        positive_color: Color for positive attributions
        negative_color: Color for negative attributions
        
    Returns:
        HTML string with highlighted tokens
    """
    # Simple HTML generation
    html_parts = ["<p>"]
    
    for token, score in token_scores:
        if score > 0:
            color = positive_color
            opacity = min(abs(score), 1.0)
        else:
            color = negative_color
            opacity = min(abs(score), 1.0)
        
        style = f"background-color: rgba({color}, {opacity:.2f})"
        html_parts.append(f'<span style="{style}">{token}</span> ')
    
    html_parts.append("</p>")
    return "".join(html_parts)
