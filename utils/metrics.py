from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
import time

@dataclass
class MetricsParams:
    total_predictions: int = 0
    total_errors: int = 0
    predictions_by_class: Dict[str, int] = field(default_factory=lambda: {"NORMAL": 0, "PNEUMONIA": 0})
    total_inference_time: float = 0.0
    total_confidence_sum: float = 0.0
    uptime_start: datetime = field(default_factory=datetime.now)

class Metrics:
    _instance = None
    _params = None

    @classmethod
    def initialize(cls):
        if cls._instance is None:
            cls._instance = Metrics()
            cls._params = MetricsParams()

    @classmethod
    def get_metrics(cls):
        if cls._params is None:
            cls.initialize()
            
        avg_confidence = 0.0
        if cls._params.total_predictions > 0:
            avg_confidence = (cls._params.total_confidence_sum / cls._params.total_predictions)

        avg_inference_time = 0.0
        if cls._params.total_predictions > 0:
            avg_inference_time = (cls._params.total_inference_time / cls._params.total_predictions) * 1000 # to ms

        error_rate = 0.0
        total_requests = cls._params.total_predictions + cls._params.total_errors
        if total_requests > 0:
            error_rate = (cls._params.total_errors / total_requests) * 100
            
        uptime_seconds = (datetime.now() - cls._params.uptime_start).total_seconds()

        return {
            "total_predictions": cls._params.total_predictions,
            "predictions_by_class": cls._params.predictions_by_class,
            "average_confidence": round(avg_confidence, 2),
            "average_inference_time_ms": round(avg_inference_time, 2),
            "error_rate_percent": round(error_rate, 2),
            "uptime_seconds": int(uptime_seconds)
        }

    @classmethod
    def update(cls, predicted_class: str, confidence: float, inference_time: float, error: bool = False):
        if cls._params is None:
            cls.initialize()

        if error:
            cls._params.total_errors += 1
        else:
            cls._params.total_predictions += 1
            if predicted_class in cls._params.predictions_by_class:
                cls._params.predictions_by_class[predicted_class] += 1
            cls._params.total_confidence_sum += confidence
            cls._params.total_inference_time += inference_time
