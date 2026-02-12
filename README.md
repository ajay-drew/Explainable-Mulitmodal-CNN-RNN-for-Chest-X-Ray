# Pneumonia Detection API with Explainable AI

This project is a **FastAPI-based backend** for detecting pneumonia from chest X-ray images. It uses a fine-tuned Vision Transformer (ViT) model and generates Grad-CAM heatmaps to explain its predictions.

## Features

- **Pneumonia Detection**: Classifies Chest X-Rays as `NORMAL` or `PNEUMONIA`.
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps highlighting affected areas.
- **FastAPI Backend**: High-performance, asynchronous API.
- **Metrics Tracking**: In-memory tracking of accuracy, latency, and error rates.
- **Automatic Cleanup**: Temporary files are automatically deleted after 1 hour.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (Recommended) or CPU (Slower)
- loose 6GB+ VRAM if using GPU.

## Installation

1.  Clone the repository or navigate to the project directory:
    ```bash
    cd pneumonia-detection-api
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Starting the Application

1. **Start the Backend Server**:
   ```bash
   python main.py
   ```
   The backend will start at `http://localhost:8000`.

2. **Start the Streamlit Frontend** (in a new terminal):
   ```bash
   streamlit run frontend/app.py
   ```
   The frontend will open in your browser at `http://localhost:8501`.

### Running Tests

Execute the test suite to verify API endpoints and utilities:

```bash
pytest
```

### API Documentation

Interactive Swagger UI is available at: `http://localhost:7779/docs`

### API Endpoints

#### 1. Predict (`POST /predict`)
Uploads an image and returns the prediction along with a heatmap ID.

**Request:**
- `file`: Image file (JPG, PNG) - Max 10MB

**Example:**
```bash
curl -X POST "http://localhost:7779/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/chest_xray.jpg"
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 98.45,
  "probabilities": {
    "NORMAL": 1.55,
    "PNEUMONIA": 98.45
  },
  "heatmap_id": "heatmap_20240212_103000_a1b2c3d4"
}
```

#### 2. Get Explanation (`GET /explain/{heatmap_id}`)
Downloads the generated Grad-CAM heatmap.

**Example:**
```bash
curl "http://localhost:7779/explain/heatmap_20240212_103000_a1b2c3d4" --output heatmap.png
```

#### 3. Health Check (`GET /health`)
Returns system status and metrics.

**Example:**
```bash
curl "http://localhost:7779/health"
```

## Directory Structure

```
pneumonia-detection-api/
├── main.py                 # App entry point
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── models/
│   ├── detector.py         # Model inference
│   └── explainer.py        # Grad-CAM logic
├── api/
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Pydantic models
├── utils/
│   ├── preprocessing.py    # Image validation/transform
│   ├── file_handler.py     # File I/O
│   └── metrics.py          # Metrics tracking
└── tmp/
    ├── uploads/            # Temp uploads
    └── heatmaps/           # Generated heatmaps
```

## Troubleshooting

- **CUDA Out of Memory**: If running on a GPU with limited memory, try closing other GPU-intensive applications.
- **Model Load Fail**: Ensure you have internet access for the initial model download.
- **Import Errors**: Ensure all dependencies in `requirements.txt` are installed.
