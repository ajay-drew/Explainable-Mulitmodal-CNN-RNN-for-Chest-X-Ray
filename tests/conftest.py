import pytest
import pytest_asyncio
import sys
import os
from unittest.mock import MagicMock, patch
from httpx import AsyncClient
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from models.detector import ModelDetector
from utils.metrics import Metrics

@pytest.fixture
def mock_model_detector():
    """
    Mocks the ModelDetector to avoid loading the heavy ViT model.
    Returns a mock that provides a valid prediction response.
    """
    with patch('models.detector.ModelDetector.get_instance') as mock_get_instance:
        mock_instance = MagicMock()
        
        # Mock predict method return value
        dummy_heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
        
        mock_instance.predict.return_value = {
            "prediction": "PNEUMONIA",
            "confidence": 95.5,
            "probabilities": {"NORMAL": 4.5, "PNEUMONIA": 95.5},
            "heatmap": dummy_heatmap
        }
        
        mock_instance.model = MagicMock() # To pass health check
        mock_instance.device = "cpu"
        
        mock_get_instance.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def test_app(mock_model_detector):
    """
    Returns the FastAPI app with dependencies mocked.
    """
    Metrics.initialize()
    return app

@pytest_asyncio.fixture
async def client(test_app):
    """
    Async client for valid testing of FastAPI routes.
    """
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        yield ac

@pytest.fixture
def valid_image_bytes():
    """
    Creates a small valid in-memory image for upload testing.
    """
    img = Image.new('RGB', (100, 100), color = 'red')
    import io
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()
