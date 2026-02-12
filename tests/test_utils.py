import pytest
from utils.metrics import Metrics
from utils.file_handler import generate_heatmap_id
import time

def test_metrics_singleton():
    Metrics._instance = None # Reset
    Metrics.initialize()
    metrics1 = Metrics.get_metrics()
    Metrics.update("NORMAL", 90.0, 0.1)
    metrics2 = Metrics.get_metrics()
    
    assert metrics2['total_predictions'] == 1
    assert metrics2['predictions_by_class']['NORMAL'] == 1

def test_metrics_averages():
    Metrics._instance = None
    Metrics.initialize()
    
    Metrics.update("NORMAL", 100.0, 0.1)
    Metrics.update("PNEUMONIA", 50.0, 0.3)
    
    data = Metrics.get_metrics()
    assert data['total_predictions'] == 2
    assert data['average_confidence'] == 75.0
    assert data['average_inference_time_ms'] == 200.0 # (0.1+0.3)/2 * 1000

def test_generate_heatmap_id():
    hid = generate_heatmap_id()
    assert isinstance(hid, str)
    assert "heatmap_" in hid
    assert len(hid.split("_")) >= 3
