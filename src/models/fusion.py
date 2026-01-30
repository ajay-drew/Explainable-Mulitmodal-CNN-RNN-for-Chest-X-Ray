"""
Multimodal Fusion Module.

Per PROJECT_PLAN §3.3 and §5.4:
- Fuse CNN (image) and BERT (text) features
- Attention-based fusion (self-attention or cross-modal attention)
- Output: Unified embedding for classification
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """Base class for fusion modules."""
    
    def __init__(self, image_dim: int, text_dim: int, output_dim: int):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class ConcatFusion(FusionModule):
    """
    Simple concatenation fusion.
    
    Concatenates image and text features, then projects to output dim.
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        output_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__(image_dim, text_dim, output_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(image_dim + text_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse image and text features via concatenation.
        
        Args:
            image_features: [batch, image_dim]
            text_features: [batch, text_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        # Concatenate
        fused = torch.cat([image_features, text_features], dim=-1)
        
        # Project
        return self.projection(fused)


class AttentionFusion(FusionModule):
    """
    Attention-based fusion (per PROJECT_PLAN).
    
    Uses self-attention over image and text features to learn
    weighted combination.
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__(image_dim, text_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        
        # Project both modalities to same dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse image and text features via attention.
        
        Args:
            image_features: [batch, image_dim]
            text_features: [batch, text_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        batch_size = image_features.size(0)
        
        # Project to common dimension
        image_proj = self.image_proj(image_features)  # [B, hidden]
        text_proj = self.text_proj(text_features)      # [B, hidden]
        
        # Stack as sequence for attention [B, 2, hidden]
        combined = torch.stack([image_proj, text_proj], dim=1)
        
        # Self-attention over modalities
        attended, attn_weights = self.attention(
            combined, combined, combined,
            need_weights=True,
        )
        
        # Flatten attended features [B, 2*hidden]
        attended = attended.view(batch_size, -1)
        
        # Project to output
        return self.output_proj(attended)


class CrossModalAttention(FusionModule):
    """
    Cross-modal attention fusion (inspired by MCX-Net).
    
    Text features guide attention over image features (or vice versa).
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__(image_dim, text_dim, output_dim)
        
        # Project both modalities
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention: text attends to image
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Combine attended image with original text
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-modal fusion: text guides image attention.
        
        Args:
            image_features: [batch, image_dim]
            text_features: [batch, text_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        # Project
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, hidden]
        text_proj = self.text_proj(text_features).unsqueeze(1)      # [B, 1, hidden]
        
        # Cross-attention: query=text, key/value=image
        attended, _ = self.cross_attention(
            query=text_proj,
            key=image_proj,
            value=image_proj,
        )
        attended = attended.squeeze(1)  # [B, hidden]
        text_proj = text_proj.squeeze(1)
        
        # Combine
        combined = torch.cat([attended, text_proj], dim=-1)
        return self.combine(combined)


def get_fusion_module(
    fusion_type: str,
    image_dim: int,
    text_dim: int,
    output_dim: int,
    **kwargs,
) -> FusionModule:
    """
    Factory function for fusion modules.
    
    Args:
        fusion_type: One of "concat", "attention", "cross_attention"
        image_dim: Image feature dimension
        text_dim: Text feature dimension
        output_dim: Output dimension
        **kwargs: Additional arguments for the fusion module
        
    Returns:
        FusionModule instance
    """
    if fusion_type == "concat":
        return ConcatFusion(image_dim, text_dim, output_dim, **kwargs)
    elif fusion_type == "attention":
        return AttentionFusion(image_dim, text_dim, output_dim, **kwargs)
    elif fusion_type == "cross_attention":
        return CrossModalAttention(image_dim, text_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
