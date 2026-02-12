from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from contextlib import asynccontextmanager

from config import LOG_DIR
from api.routes import router
from models.detector import load_model, ModelDetector
from utils.metrics import Metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI application started")
    try:
        load_model()  # Initializes global model instance
        Metrics.initialize()
        logger.info("Ready to accept requests")
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("FastAPI application shutting down")

app = FastAPI(
    title="Pneumonia Detection API with XAI",
    description="Hybrid CNN-RNN Framework - Medical Imaging Module",
    version="0.4.0",
    lifespan=lifespan
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    # In development, auto-reload can be useful. 
    # But for heavy model loading, it might be annoying.
    uvicorn.run("main:app", host="0.0.0.0", port=7779, reload=False)
