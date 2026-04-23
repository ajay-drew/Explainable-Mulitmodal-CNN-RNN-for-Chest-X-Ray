"""
Grad-CAM heatmap generation for TorchXRayVision DenseNet121.

Uses the pytorch-grad-cam library for robust, multi-method support.
Input images must be single-channel grayscale; the library's 3-channel
requirement is handled internally.

Supported methods:
    "gradcam"   → GradCAM
    "gradcam++" → GradCAMPlusPlus
    "eigencam"  → EigenCAM
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


def _import_cam_class(method: str):
    """Import the appropriate CAM class by method name."""
    method = method.lower().replace("-", "").replace("+", "p")
    if method in ("gradcam", "grad_cam"):
        from pytorch_grad_cam import GradCAM
        return GradCAM
    elif method in ("gradcampp", "gradcam++"):
        from pytorch_grad_cam import GradCAMPlusPlus
        return GradCAMPlusPlus
    elif method == "eigencam":
        from pytorch_grad_cam import EigenCAM
        return EigenCAM
    else:
        raise ValueError(
            f"Unknown CAM method: '{method}'. "
            "Choose from 'gradcam', 'gradcam++', 'eigencam'."
        )


class _XRVWrapper(nn.Module):
    """
    Wraps XRVImageEncoder so that pytorch-grad-cam gets standard (B, C, H, W)
    3-channel input and returns class logits directly.

    pytorch-grad-cam expects:
      - forward(x) → (B, num_classes) logits
      - 3-channel RGB input

    TorchXRayVision DenseNet expects:
      - 1-channel input, range [-1024, 1024]

    This wrapper accepts 3-channel input (duplicating channels internally),
    strips back to single channel, and routes through the XRV model.
    """

    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) from pytorch-grad-cam
        # Convert back to single-channel by averaging
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        out = self.encoder(x)
        return out["logits"]  # (B, 15)


class XRVGradCAM:
    """
    Grad-CAM generator for TorchXRayVision DenseNet121.

    Args:
        encoder:  XRVImageEncoder instance (frozen backbone).
        method:   CAM method string: "gradcam", "gradcam++", or "eigencam".
    """

    def __init__(self, encoder, method: str = "gradcam") -> None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        self.encoder = encoder
        self.method = method
        self._ClassifierOutputTarget = ClassifierOutputTarget

        # Build wrapper and target layer
        self.wrapped = _XRVWrapper(encoder)
        self.target_layer = encoder.get_gradcam_target_layer()

        CAMClass = _import_cam_class(method)
        self.cam = CAMClass(
            model=self.wrapped,
            target_layers=[self.target_layer],
        )
        logger.info(f"XRVGradCAM initialised with method='{method}'.")

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Args:
            image_tensor: (1, 1, 224, 224) float32, range [-1024, 1024].
            target_class: Index into our 15-label list.

        Returns:
            Grayscale heatmap as float32 numpy array (224, 224), range [0, 1].
        """
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # pytorch-grad-cam expects 3-channel input
        # Normalise [-1024, 1024] → [0, 1] for the wrapper's RGB conversion
        img_norm = (image_tensor + 1024.0) / 2048.0  # (1, 1, 224, 224), [0, 1]
        img_3ch = img_norm.repeat(1, 3, 1, 1)         # (1, 3, 224, 224)

        targets = [self._ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=img_3ch, targets=targets)
        return grayscale_cam[0]  # (224, 224), float32 [0, 1]

    def overlay_heatmap(
        self,
        original_pil: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = None,
    ) -> Image.Image:
        """
        Overlay a Grad-CAM heatmap on the original X-ray image.

        Args:
            original_pil: Original chest X-ray PIL Image (any mode).
            heatmap:      Float32 numpy array (H, W), range [0, 1].
            alpha:        Overlay transparency (default 0.4).
            colormap:     OpenCV colormap constant. Defaults to COLORMAP_JET.

        Returns:
            PIL Image (RGB) with heatmap overlay.
        """
        import cv2
        from pytorch_grad_cam.utils.image import show_cam_on_image

        if colormap is None:
            colormap = cv2.COLORMAP_JET

        # Prepare base image as float32 RGB [0, 1]
        img_rgb = original_pil.convert("RGB").resize((224, 224), Image.LANCZOS)
        img_float = np.array(img_rgb, dtype=np.float32) / 255.0

        # show_cam_on_image returns uint8 RGB numpy array
        overlay = show_cam_on_image(img_float, heatmap, use_rgb=True, colormap=colormap)
        return Image.fromarray(overlay)

    def generate_and_overlay(
        self,
        image_tensor: torch.Tensor,
        original_pil: Image.Image,
        target_class: int,
        alpha: float = 0.4,
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Convenience method: generate heatmap and overlay in one call.

        Returns:
            (heatmap_array, overlay_pil)
        """
        heatmap = self.generate_heatmap(image_tensor, target_class)
        overlay = self.overlay_heatmap(original_pil, heatmap, alpha=alpha)
        return heatmap, overlay


def compute_faithfulness(
    encoder,
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    top_k: int = 5,
    patch_size: int = 16,
) -> float:
    """
    Faithfulness metric: perturb top-k heatmap patches to zero and measure
    the drop in prediction confidence.

    Score = (original_conf - perturbed_conf) / original_conf
    Clamped to [0, 1]. Higher = more faithful explanation.

    Args:
        encoder:      XRVImageEncoder.
        image_tensor: (1, 1, 224, 224) input tensor.
        heatmap:      (224, 224) float32 heatmap from Grad-CAM.
        target_class: Target class index.
        top_k:        Number of patches to occlude.
        patch_size:   Size of each occlusion patch in pixels.

    Returns:
        Faithfulness score in [0, 1].
    """
    encoder.eval()
    with torch.no_grad():
        out = encoder(image_tensor)
        probs = torch.sigmoid(out["logits"])
        original_conf = probs[0, target_class].item()

    # Find top-k patches by average heatmap activation
    H, W = heatmap.shape
    patches = []
    step = patch_size
    for r in range(0, H - step + 1, step):
        for c in range(0, W - step + 1, step):
            score = heatmap[r:r + step, c:c + step].mean()
            patches.append((score, r, c))
    patches.sort(key=lambda x: x[0], reverse=True)

    # Occlude top-k patches
    perturbed = image_tensor.clone()
    for _, r, c in patches[:top_k]:
        perturbed[0, :, r:r + step, c:c + step] = -1024.0  # XRV "black"

    with torch.no_grad():
        out_p = encoder(perturbed)
        probs_p = torch.sigmoid(out_p["logits"])
        perturbed_conf = probs_p[0, target_class].item()

    if original_conf < 1e-6:
        return 0.0
    score = (original_conf - perturbed_conf) / original_conf
    return float(max(0.0, min(1.0, score)))
