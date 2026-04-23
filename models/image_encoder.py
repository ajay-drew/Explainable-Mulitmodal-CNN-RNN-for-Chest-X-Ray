"""
Image encoder wrapping TorchXRayVision DenseNet121.

The ENTIRE backbone is frozen immediately after loading.
Predictions come from the model's own pretrained 18-class pathology head.
The 18 XRV outputs are mapped to our 15 target labels.

Input:  (B, 1, 224, 224) float32 tensor, range [-1024, 1024]
Output: dict with keys
          'logits'   — (B, 15) mapped to our label set
          'features' — (B, 1024) global average pooled features
          'xrv_logits' — (B, 18) raw XRV outputs (for zero-shot baseline)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchxrayvision as xrv

logger = logging.getLogger(__name__)

# Our 15 target labels
TARGET_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# Mapping: our label → XRV pathology string used in xrv model.pathologies
# XRV uses "Pleural_Thickening", we call it "Pleural Thickening"
TARGET_TO_XRV: Dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Hernia": "Hernia",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "No Finding": "No Finding",
    "Nodule": "Nodule",
    "Pleural Thickening": "Pleural_Thickening",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


class XRVImageEncoder(nn.Module):
    """
    Frozen TorchXRayVision DenseNet121 image encoder.

    The backbone and its built-in pathology head are both frozen.
    A forward hook extracts 1024-dim global average pooled features.
    The built-in 18-class logits are remapped to 15 target labels.

    Args:
        weights:        XRV weight string (default "densenet121-res224-all").
        target_labels:  Ordered list of 15 output label names.
    """

    def __init__(
        self,
        weights: str = "densenet121-res224-all",
        target_labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.target_labels = target_labels or TARGET_LABELS
        self.num_classes = len(self.target_labels)

        logger.info(f"Loading TorchXRayVision DenseNet121 ({weights}) …")
        self.model = xrv.models.DenseNet(weights=weights)
        self.model.eval()

        # Freeze ALL parameters — backbone + built-in head
        self.freeze_backbone()

        # Build mapping: our label index → XRV output index
        xrv_pathologies = [p for p in self.model.pathologies]
        self._target_indices: List[Optional[int]] = []
        for label in self.target_labels:
            xrv_name = TARGET_TO_XRV.get(label, label)
            try:
                idx = xrv_pathologies.index(xrv_name)
                self._target_indices.append(idx)
            except ValueError:
                logger.warning(
                    f"Label '{label}' (XRV: '{xrv_name}') not found in XRV pathologies. "
                    "Output for this class will be zeros."
                )
                self._target_indices.append(None)

        # Feature extraction hook — captures global average pooled features
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._register_feature_hook()

        logger.info(
            f"XRVImageEncoder ready. {self.count_trainable_parameters()} "
            "trainable parameters (should be 0)."
        )

    def _register_feature_hook(self) -> None:
        """Register a forward hook on the adaptive average pooling layer."""
        # TorchXRayVision DenseNet: self.model.features → denseblocks → norm → relu
        # The global avg pool is self.model.features followed by adaptive_avg_pool2d in forward
        # We hook into the batch norm after the final dense block.
        def _hook(module, input, output):
            # output shape: (B, 1024, H, W)
            # Global average pool
            pooled = output.mean(dim=[2, 3])  # (B, 1024)
            self._feature_cache["features"] = pooled

        # Register on the last norm layer inside features
        try:
            self.model.features.norm5.register_forward_hook(_hook)
        except AttributeError:
            # Fallback: hook the entire features block
            self.model.features.register_forward_hook(
                lambda m, inp, out: self._feature_cache.__setitem__(
                    "features", out.mean(dim=[2, 3]) if out.ndim == 4 else out
                )
            )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (including XRV's own head)."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.debug("XRVImageEncoder: backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze backbone parameters.
        NOTE: This method is provided for completeness but should NEVER be
        called in the default zero-shot pipeline.
        """
        for param in self.model.parameters():
            param.requires_grad = True
        logger.warning(
            "XRVImageEncoder: backbone UNFROZEN. This deviates from the "
            "zero-training philosophy."
        )

    def count_trainable_parameters(self) -> int:
        """Return the number of parameters with requires_grad=True."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Tensor (B, 1, 224, 224), float32, range [-1024, 1024].

        Returns:
            dict:
              'logits'      — (B, 15) raw scores for our target labels
              'features'    — (B, 1024) global average pooled features
              'xrv_logits'  — (B, n_xrv) full XRV output
        """
        # XRV DenseNet expects (B, 1, 224, 224)
        xrv_out = self.model(x)  # (B, n_xrv)

        # Extract features via hook
        features = self._feature_cache.get("features")
        if features is None:
            # Fallback: run features block manually
            feat_map = self.model.features(x)
            import torch.nn.functional as F
            features = F.relu(feat_map, inplace=False).mean(dim=[2, 3])

        # Map XRV outputs → our 15 labels
        batch_size = xrv_out.shape[0]
        logits = torch.zeros(
            batch_size, self.num_classes, device=xrv_out.device, dtype=xrv_out.dtype
        )
        for our_idx, xrv_idx in enumerate(self._target_indices):
            if xrv_idx is not None and xrv_idx < xrv_out.shape[1]:
                logits[:, our_idx] = xrv_out[:, xrv_idx]

        return {
            "logits": logits,
            "features": features,
            "xrv_logits": xrv_out,
        }

    def get_gradcam_target_layer(self):
        """Return the target layer for Grad-CAM (final DenseBlock)."""
        return self.model.features.denseblock4


def load_image_encoder(
    weights: str = "densenet121-res224-all",
    device: Optional[torch.device] = None,
) -> XRVImageEncoder:
    """
    Convenience loader. Returns encoder in eval mode on the specified device.

    Args:
        weights: XRV weights string.
        device:  Target device. Defaults to CPU.

    Returns:
        XRVImageEncoder in eval mode.
    """
    dev = device or torch.device("cpu")
    encoder = XRVImageEncoder(weights=weights)
    encoder.to(dev)
    encoder.eval()
    return encoder
