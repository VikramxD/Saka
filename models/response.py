from pydantic import BaseModel, Field, HttpUrl
from .metrics import ProcessingMetrics

class VideoResponse(BaseModel):
    """Response model for enhanced video."""
    output_url: HttpUrl = Field(..., description="S3 URL of the enhanced video")
    metrics: ProcessingMetrics = Field(..., description="Processing metrics")
