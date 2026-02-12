import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import logging
from config import MODEL_NAME, CLASS_NAMES
from models.explainer import Explainer

logger = logging.getLogger(__name__)

class ModelDetector:
    _instance = None
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on device: {self.device}")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize Explainer
            # Note: Explainer needs the model instance
            self.explainer = Explainer(self.model)
            
            logger.info("Model and Explainer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelDetector()
        return cls._instance

    def predict(self, image: Image.Image):
        """
        Runs inference on the image.
        Returns:
            - prediction (str): Predicted class name
            - confidence (float): Confidence score (0-100)
            - probabilities (dict): Map of all class probabilities
            - heatmap_overlay (np.ndarray): Generated Grad-CAM heatmap
        """
        try:
            # Preprocess
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            # We need gradients for Grad-CAM? 
            # Usual inference is with torch.no_grad(), but Grad-CAM requires gradients.
            # However, for the FORWARD pass to get predictions, we can use no_grad if we re-run for CAM.
            # Or we can just run forward with grad enabled? 
            # Efficiency: Run once. But Grad-CAM usually hooks into the model. 
            # Standard flow: 
            # 1. Forward pass (can be no_grad if we don't need backprop YET, but GradCAM needs it).
            # ACTUALLY: pytorch-grad-cam handles the forward pass internally when we call generate().
            # So we should run a lightweight forward pass for prediction first.
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
            # Get prediction
            score, predicted_class_idx_tensor = torch.max(probs, 1)
            predicted_class_idx = predicted_class_idx_tensor.item()
            confidence = score.item() * 100
            
            probabilities = {
                class_name: probs[0][i].item() * 100 
                for i, class_name in enumerate(CLASS_NAMES)
            }
            
            predicted_label = CLASS_NAMES[predicted_class_idx]
            
            # Generate Heatmap
            # Note: input_tensor for GradCAM should be the same as fed to model.
            # inputs['pixel_values'] is the tensor.
            heatmap = self.explainer.generate_heatmap(
                inputs['pixel_values'], 
                predicted_class_idx,
                image
            )
            
            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities,
                "heatmap": heatmap
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise e

# Global loader function
def load_model():
    return ModelDetector.get_instance()
