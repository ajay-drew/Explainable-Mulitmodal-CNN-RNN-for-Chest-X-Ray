import os
import aiofiles
import uuid
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def generate_heatmap_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"heatmap_{timestamp}_{unique_id}"

async def save_upload_to_tmp(file_content: bytes, filename: str, upload_dir: str) -> str:
    """Saves raw bytes to a temporary file."""
    filepath = os.path.join(upload_dir, filename)
    async with aiofiles.open(filepath, 'wb') as out_file:
        await out_file.write(file_content)
    return filepath

def delete_file(filepath: str):
    """Safely deletes a file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Deleted file: {filepath}")
    except Exception as e:
        logger.error(f"Error deleting file {filepath}: {e}")

async def cleanup_old_heatmaps(heatmap_dir: str, expiry_seconds: int):
    """Background task to remove old heatmaps."""
    try:
        now = datetime.now()
        threshold = now - timedelta(seconds=expiry_seconds)
        
        if not os.path.exists(heatmap_dir):
            return

        for filename in os.listdir(heatmap_dir):
            filepath = os.path.join(heatmap_dir, filename)
            # Check creation time
            if os.path.isfile(filepath):
                timestamp = os.path.getctime(filepath)
                file_time = datetime.fromtimestamp(timestamp)
                
                if file_time < threshold:
                    delete_file(filepath)
                    logger.info(f"Cleaned up expired heatmap: {filename}")
                    
    except Exception as e:
        logger.error(f"Error during heatmap cleanup: {e}")
