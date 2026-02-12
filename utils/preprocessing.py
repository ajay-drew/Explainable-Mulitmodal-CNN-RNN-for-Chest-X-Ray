from PIL import Image
import io
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)

async def validate_image(file: UploadFile, max_size_bytes: int) -> Image.Image:
    """
    Validates file size, format, and integrity.
    Returns a PIL Image object converted to RGB.
    """
    # Check file size (Note: Request body size checking happens before this usually, 
    # but specific file size can be checked if we read it. 
    # FastAPI UploadFile is a SpooledTemporaryFile.)
    
    # Read file content
    content = await file.read()
    
    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=400, 
            detail=f"File size exceeds limit of {max_size_bytes / (1024*1024)}MB"
        )
        
    try:
        image = Image.open(io.BytesIO(content))
        image.verify() # Verify it's an image
        image = Image.open(io.BytesIO(content)) # Re-open after verify
        
        # Convert to RGB (standardize channels)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file or corrupted data.")
    finally:
        await file.seek(0) # Reset file pointer if needed elsewhere, though we consumed it here.
