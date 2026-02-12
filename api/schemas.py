from pydantic import BaseModel
from typing import Dict, Union

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    heatmap_id: str

class MetricsData(BaseModel):
    total_predictions: int
    predictions_by_class: Dict[str, int]
    average_confidence: float
    average_inference_time_ms: float
    error_rate_percent: float
    uptime_seconds: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    metrics: MetricsData

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
