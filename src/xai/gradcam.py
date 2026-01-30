"""
Grad-CAM for image explainability.

Per PROJECT_PLAN §3.5 and §5.6:
- Generate visual saliency heatmaps
- Highlight disease-relevant regions in chest X-rays
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates heatmaps showing which image regions influenced
    the model's prediction for each class.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The full model (for forward pass)
            target_layer: The convolutional layer to extract activations from
                         (typically the last conv layer before pooling)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for a single image.
        
        Args:
            image: Input image [1, C, H, W]
            input_ids: Tokenized report [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            target_class: Class index to explain (None = use predicted class)
            
        Returns:
            Heatmap tensor [H, W] with values in [0, 1]
        """
        self.model.eval()
        
        # Enable gradients for this input
        image.requires_grad_(True)
        
        # Forward pass
        logits = self.model(image, input_ids, attention_mask)
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_for_all_classes(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        class_names: List[str],
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate Grad-CAM heatmaps for all predicted classes.
        
        Args:
            image: Input image [1, C, H, W]
            input_ids: Tokenized report [1, seq_len]
            attention_mask: Attention mask [1, seq_len]
            class_names: List of class names
            threshold: Prediction threshold
            
        Returns:
            Dictionary mapping class names to heatmaps
        """
        # Get predictions
        with torch.no_grad():
            logits = self.model(image, input_ids, attention_mask)
            probs = torch.sigmoid(logits)
        
        # Find predicted classes
        predicted_indices = (probs[0] >= threshold).nonzero(as_tuple=True)[0]
        
        heatmaps = {}
        for idx in predicted_indices:
            class_name = class_names[idx.item()]
            heatmap = self.generate(image, input_ids, attention_mask, target_class=idx.item())
            heatmaps[class_name] = heatmap
        
        return heatmaps


def generate_gradcam_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_name: str = "image_encoder.backbone.features",
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function to generate Grad-CAM heatmap.
    
    Args:
        model: Multimodal classifier
        image: Input image [1, C, H, W]
        input_ids: Tokenized report [1, seq_len]
        attention_mask: Attention mask [1, seq_len]
        target_layer_name: Dot-separated path to target layer
        target_class: Class to explain
        
    Returns:
        Heatmap as numpy array [H, W]
    """
    # Get target layer
    target_layer = model
    for name in target_layer_name.split("."):
        target_layer = getattr(target_layer, name)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = gradcam.generate(image, input_ids, attention_mask, target_class)
    
    return heatmap.cpu().numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image [H, W] or [H, W, C]
        heatmap: Heatmap [H, W] with values in [0, 1]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
        
    Returns:
        Overlaid image [H, W, 3]
    """
    import matplotlib.pyplot as plt
    
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR,
        )
    ) / 255.0
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Drop alpha
    
    # Normalize image to [0, 1]
    if image.max() > 1:
        image = image / 255.0
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Overlay
    overlaid = (1 - alpha) * image + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    
    return (overlaid * 255).astype(np.uint8)
