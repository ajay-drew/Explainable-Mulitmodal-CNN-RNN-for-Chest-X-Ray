import os

# Model Settings
MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# File Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp", "uploads")
HEATMAP_DIR = os.path.join(BASE_DIR, "tmp", "heatmaps")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Constraints
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
HEATMAP_EXPIRY_SECONDS = 3600  # 1 hour

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
