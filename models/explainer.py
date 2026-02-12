import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class Explainer:
    def __init__(self, model):
        self.model = model
        # Use the last encoder layer's layernorm (before attention) for better spatial features
        self.target_layers = [model.vit.encoder.layer[-1].layernorm_before]
            
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x).logits

        def reshape_transform(tensor, height=14, width=14):
            # Drop class token and reshape
            result = tensor[:, 1:, :].reshape(tensor.size(0),
                                              height, width, tensor.size(2))
            
            # Bring channels to first dim: (B, H, W, C) -> (B, C, H, W)
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        self.cam = GradCAM(model=ModelWrapper(self.model), target_layers=self.target_layers, reshape_transform=reshape_transform)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int, original_image: Image.Image) -> np.ndarray:
        """
        Generates Grad-CAM heatmap and returns the overlay image as a numpy array (RGB).
        """
        try:
            # Generate grayscale CAM
            # Note: pytorch-grad-cam expects inputs with shape (batch, channels, height, width)
            # targets expects a list of ClassifierOutputTarget, but for simple classification 
            # passing the class index as an integer often works or we wrap it.
            # Library update: targets=[ClassifierOutputTarget(target_class)] is preferred.
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            
            targets = [ClassifierOutputTarget(target_class)]
            
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            
            # In this version of the library, grayscale_cam is (batch, height, width)
            grayscale_cam = grayscale_cam[0, :]
            
            # Resize original image to match tensor input size usually 224x224
            # We should use the normalized float image required by show_cam_on_image
            # but show_cam_on_image expects the image to be float32 in [0, 1]
            
            # Prepare original image for overlay
            img_resized = original_image.resize((224, 224))
            img_cw = np.array(img_resized)
            img_float = np.float32(img_cw) / 255.0
            
            visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            return visualization
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            raise e
