"""
Multimodal Classifier for chest X-ray diagnosis.

Per PROJECT_PLAN §5.5:
- Multi-label classification (13 diseases + No Findings)
- BCE loss with sigmoid activation
- Dropout for regularization
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion import get_fusion_module, FusionModule


class MultimodalClassifier(nn.Module):
    """
    Full multimodal CNN-RNN classifier for chest X-ray diagnosis.
    
    Architecture:
    1. Image Encoder (TorchXRayVision DenseNet121)
    2. Text Encoder (RadBERT/Bio_ClinicalBERT)
    3. Multimodal Fusion (attention-based)
    4. Classification Head (MLP with sigmoid for multi-label)
    """
    
    def __init__(
        self,
        # Image encoder config
        image_model_name: str = "densenet121-res224-mimic_nb",
        image_pretrained: bool = True,
        freeze_image_encoder: bool = False,
        
        # Text encoder config
        text_model_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m",
        text_pretrained: bool = True,
        freeze_text_encoder: bool = False,
        text_pooling: str = "cls",
        
        # Fusion config
        fusion_type: str = "attention",
        fusion_hidden_dim: int = 512,
        fusion_dropout: float = 0.5,
        
        # Classifier config
        num_classes: int = 14,
        classifier_hidden_dims: Optional[List[int]] = None,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # === Image Encoder ===
        self.image_encoder = ImageEncoder(
            model_name=image_model_name,
            pretrained=image_pretrained,
            freeze_backbone=freeze_image_encoder,
        )
        
        # === Text Encoder ===
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            pretrained=text_pretrained,
            freeze_backbone=freeze_text_encoder,
            pooling=text_pooling,
        )
        
        # === Fusion Module ===
        self.fusion = get_fusion_module(
            fusion_type=fusion_type,
            image_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            output_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )
        
        # === Classification Head ===
        classifier_hidden_dims = classifier_hidden_dims or [256]
        classifier_layers = []
        
        in_dim = fusion_hidden_dim
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
            ])
            in_dim = hidden_dim
        
        # Final classification layer (no activation - use BCEWithLogitsLoss)
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the multimodal classifier.
        
        Args:
            image: Input images [batch, channels, height, width]
            input_ids: Tokenized report IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Logits [batch, num_classes] (apply sigmoid for probabilities)
        """
        # Extract image features
        image_features = self.image_encoder(image)
        
        # Extract text features
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Fuse modalities
        fused_features = self.fusion(image_features, text_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_features(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get intermediate features for analysis/XAI.
        
        Returns:
            Dictionary with image_features, text_features, fused_features
        """
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        fused_features = self.fusion(image_features, text_features)
        
        return {
            "image_features": image_features,
            "text_features": text_features,
            "fused_features": fused_features,
        }
    
    def predict(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            image: Input images
            input_ids: Tokenized reports
            attention_mask: Attention mask
            threshold: Classification threshold (per PROJECT_PLAN: 0.5)
            
        Returns:
            Tuple of (probabilities, binary_predictions)
        """
        logits = self.forward(image, input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()
        
        return probabilities, predictions


def create_model(config) -> MultimodalClassifier:
    """
    Create multimodal classifier from config.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        MultimodalClassifier instance
    """
    return MultimodalClassifier(
        image_model_name=config.model.image_encoder_name,
        image_pretrained=config.model.image_encoder_pretrained,
        freeze_image_encoder=config.model.freeze_image_encoder,
        text_model_name=config.model.text_encoder_name,
        text_pretrained=True,
        freeze_text_encoder=config.model.freeze_text_encoder,
        fusion_type=config.model.fusion_type,
        fusion_hidden_dim=config.model.fusion_hidden_dim,
        fusion_dropout=config.model.fusion_dropout,
        num_classes=config.model.num_classes,
        classifier_hidden_dims=config.model.classifier_hidden_dims,
        classifier_dropout=config.model.classifier_dropout,
    )
