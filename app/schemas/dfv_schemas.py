from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="REAL or FAKE")
    confidence: float = Field(..., description="Confidence score (0-1)")
    num_segments: int = Field(..., description="Number of audio segments splitted")
    segments_processed: int = Field(..., description="Number of audio segments analyzed")
    confidence: float = Field(..., description="Confidence score (0-1)")
    audio_duration: float = Field(..., description="Audio duration in seconds")
    timestamp: str = Field(..., description="Prediction timestamp")
