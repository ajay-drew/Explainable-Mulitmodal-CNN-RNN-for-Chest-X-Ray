"""
Image Encoder (CNN) for chest X-ray feature extraction.

Per PROJECT_PLAN §8:
- Uses TorchXRayVision DenseNet121 with MIMIC-CXR weights
- Model: densenet121-res224-mimic_nb or densenet121-res224-mimic_ch
- Output: Feature vector for fusion
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import torchxrayvision as xrv
    HAS_XRV = True
except ImportError:
    HAS_XRV = False


class ImageEncoder(nn.Module):
    """
    CNN encoder for chest X-ray images.
    
    Uses TorchXRayVision DenseNet121 pretrained on MIMIC-CXR
    for feature extraction.
    """
    
    def __init__(
        self,
        model_name: str = "densenet121-res224-mimic_nb",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize image encoder.
        
        Args:
            model_name: TorchXRayVision model weights name
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze encoder weights
            output_dim: Optional projection dimension (if None, use raw features)
        """
        super().__init__()
        
        if not HAS_XRV:
            raise ImportError(
                "torchxrayvision is required. Install with: pip install torchxrayvision"
            )
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        # Load TorchXRayVision DenseNet
        if pretrained:
            self.backbone = xrv.models.DenseNet(weights=model_name)
        else:
            self.backbone = xrv.models.DenseNet(weights=None)
        
        # Get feature dimension from backbone
        # DenseNet121 outputs 1024-dim features before classifier
        self.feature_dim = 1024
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from chest X-ray images.
        
        Args:
            x: Input images [batch, channels, height, width]
               Expected: [B, 1, 224, 224] grayscale
               
        Returns:
            Feature tensor [batch, output_dim]
        """
        # TorchXRayVision expects specific input format
        # Normalize to [-1024, 1024] range if not already
        
        # Extract features using backbone's features method
        features = self.backbone.features(x)
        
        # Global average pooling
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        
        # Optional projection
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate feature maps for Grad-CAM.
        
        Args:
            x: Input images [batch, channels, height, width]
            
        Returns:
            Feature maps [batch, channels, h, w] before pooling
        """
        return self.backbone.features(x)


class ImageEncoderTorchvision(nn.Module):
    """
    Alternative image encoder using torchvision ResNet.
    
    Fallback if TorchXRayVision is not available or for comparison.
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        
        import torchvision.models as models
        
        # Load ResNet
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
        elif model_name == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet152(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove classifier
        self.backbone.fc = nn.Identity()
        
        # Freeze if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Adapt input for grayscale (1 channel -> 3 channels)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)
        
        # Optional projection
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        # Adapt grayscale to RGB
        x = self.input_adapter(x)
        
        # Extract features
        features = self.backbone(x)
        
        # Project if needed
        if self.projection is not None:
            features = self.projection(features)
        
        return features
