from pydantic import BaseModel, Field

class VideoRequest(BaseModel):
    """Input request model for video enhancement."""
    video: bytes = Field(..., description="Input video file bytes")
    calculate_ssim: bool = Field(
        default=False, 
        description="Whether to calculate SSIM metric"
    )
