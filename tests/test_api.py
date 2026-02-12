import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "metrics" in data

@pytest.mark.asyncio
async def test_predict_success(client: AsyncClient, valid_image_bytes):
    response = await client.post(
        "/predict", 
        files={"file": ("test.png", valid_image_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "heatmap_id" in data
    assert data["prediction"] == "PNEUMONIA" # From mock

@pytest.mark.asyncio
async def test_predict_invalid_file(client: AsyncClient):
    # Sending text file instead of image
    response = await client.post(
        "/predict", 
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    # PIL validation should fail -> 400
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_heatmap_download_404(client: AsyncClient):
    response = await client.get("/explain/heatmap_non_existent")
    assert response.status_code == 404
