from schemas.dfv_schemas import PredictionResponse
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from services.dfv_service import ModelService
import os
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dfv", tags=["deepfakevoice"])

# Global instance of ModelService (initialized once)
model_service = ModelService()

def get_dfv_service() -> ModelService:
    """
    Dependency to get the shared ModelService instance.
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DeepFake Voice service not ready (model not initialized)"
        )
    return model_service

@router.get("/health")
def dfv_health_check(
    model_service: ModelService = Depends(get_dfv_service)
):
    return model_service.health_check()

@router.get("/labels")
def get_labels(
    model_service: ModelService = Depends(get_dfv_service)
) -> JSONResponse:
    """
    Get available labels for DeepFake Voice classification.
    """
    try:
        return model_service.get_available_labels()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_labels: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve labels: {str(e)}"
        )


@router.post("/detect")
async def detect_audio(
    file: UploadFile = File(..., 
                            media_type="audio/*", 
                            description="Audio file (WAV, MP3, etc.)"),
    model_service: ModelService = Depends(get_dfv_service)
) -> JSONResponse:
    """
    Detect if an audio file contains deepfake voice.
    Supports common audio formats (WAV, MP3, etc.) via librosa.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file provided"
        )
    
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing uploaded audio file: {file.filename}, size: {file.size} bytes")
        
        # Detect
        result = model_service.detect_audio_file(temp_file_path)
        
        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()
        
        return PredictionResponse(**result)
    
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error processing audio file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio detection failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_e:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {str(cleanup_e)}")
