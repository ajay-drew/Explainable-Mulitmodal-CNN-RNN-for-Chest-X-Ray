"""
Text Encoder (BERT) for radiology report feature extraction.

Per PROJECT_PLAN §9:
- Uses RadBERT-RoBERTa-4m (UCSD-VA-health/RadBERT-RoBERTa-4m)
- Alternative: Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
- Output: Pooled features (768-dim) for fusion
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextEncoder(nn.Module):
    """
    BERT-based encoder for radiology reports.
    
    Uses RadBERT or Bio_ClinicalBERT from Hugging Face Hub
    for text feature extraction.
    """
    
    def __init__(
        self,
        model_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
        pooling: str = "cls",  # "cls", "mean", "max"
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: HuggingFace model name
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze encoder weights
            output_dim: Optional projection dimension
            pooling: Pooling strategy ("cls", "mean", "max")
        """
        super().__init__()
        
        self.model_name = model_name
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone
        
        # Load BERT/RoBERTa model
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)
        
        # Get hidden size from config
        self.feature_dim = self.backbone.config.hidden_size  # 768 for BERT/RoBERTa-base
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Optional projection layer
        self.projection = None
        if output_dim is not None and output_dim != self.feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.output_dim = output_dim
        else:
            self.output_dim = self.feature_dim
    
    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings to get sentence representation.
        
        Args:
            last_hidden_state: Token embeddings [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Pooled representation [batch, hidden]
        """
        if self.pooling == "cls":
            # Use [CLS] token (first token)
            return last_hidden_state[:, 0, :]
        
        elif self.pooling == "mean":
            # Mean pooling over non-padded tokens
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling == "max":
            # Max pooling over non-padded tokens
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state[mask == 0] = -1e9
            return torch.max(last_hidden_state, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from tokenized reports.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Feature tensor [batch, output_dim]
        """
        # Get BERT outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Pool to get fixed-size representation
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        
        # Optional projection
        if self.projection is not None:
            pooled = self.projection(pooled)
        
        return pooled
    
    def get_token_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get token-level embeddings for XAI.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Tuple of (token_embeddings [batch, seq_len, hidden], attention_mask)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state, attention_mask
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get input embeddings for Integrated Gradients.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            
        Returns:
            Input embeddings [batch, seq_len, embedding_dim]
        """
        return self.backbone.embeddings.word_embeddings(input_ids)
