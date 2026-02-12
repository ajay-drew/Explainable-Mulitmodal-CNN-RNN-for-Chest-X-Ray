from fastapi import APIRouter, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import os
import time
import logging
from PIL import Image
import numpy as np
import cv2

from config import MAX_FILE_SIZE_BYTES, HEATMAP_DIR, HEATMAP_EXPIRY_SECONDS
from utils.preprocessing import validate_image
from utils.file_handler import generate_heatmap_id, cleanup_old_heatmaps
from utils.metrics import Metrics
from models.detector import ModelDetector
from api.schemas import PredictionResponse, HealthResponse, ErrorResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    start_time = time.time()
    try:
        # Validate Image
        image = await validate_image(file, MAX_FILE_SIZE_BYTES)
        
        # Run Inference
        detector = ModelDetector.get_instance()
        result = detector.predict(image)
        
        # Save Heatmap
        heatmap_id = generate_heatmap_id()
        heatmap_filename = f"{heatmap_id}.png"
        heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
        
        # Overlay heatmap is a floating point [0,1] or uint8 depending on return
        # show_cam_on_image returns float32 if image_weight is involved or sometimes uint8 if converted manually
        # pytorch-grad-cam show_cam_on_image returns [H, W, 3] usually float or uint8.
        # Check type
        heatmap_overlay = result['heatmap']
        
        # Convert to 0-255 uint8 if float
        if heatmap_overlay.max() <= 1.0:
            heatmap_overlay = (heatmap_overlay * 255).astype(np.uint8)
            
        # Save using PIL or cv2. PIL expects RGB. cv2 expects BGR.
        # We'll use PIL for consistency.
        heatmap_pil = Image.fromarray(heatmap_overlay)
        heatmap_pil.save(heatmap_path)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update metrics
        Metrics.update(
            predicted_class=result['prediction'],
            confidence=result['confidence'],
            inference_time=inference_time
        )
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_old_heatmaps, HEATMAP_DIR, HEATMAP_EXPIRY_SECONDS)
        
        logger.info(f"[RESPONSE] 200 - heatmap_id: {heatmap_id}")
        
        return {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "heatmap_id": heatmap_id
        }
        
    except HTTPException as he:
        # Pass through HTTP exceptions
        Metrics.update("", 0.0, 0.0, error=True)
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        Metrics.update("", 0.0, 0.0, error=True)
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")

@router.get("/explain/{heatmap_id}")
async def get_heatmap(heatmap_id: str):
    filename = f"{heatmap_id}.png"
    filepath = os.path.join(HEATMAP_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Heatmap not found. May have expired.")
        
    return FileResponse(filepath, media_type="image/png", filename=filename)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    detector = ModelDetector.get_instance()
    metrics = Metrics.get_metrics()
    
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "device": str(detector.device),
        "metrics": metrics
    }
