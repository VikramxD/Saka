from typing import Optional
from pydantic import BaseModel, Field

class ProcessingMetrics(BaseModel):
    """Processing metrics for enhanced video."""
    ram_usage_mb: float = Field(..., description="RAM usage in megabytes")
    processing_time_sec: float = Field(..., description="Processing time in seconds")
    ssim_score: Optional[float] = Field(None, description="SSIM score if calculated")

    def dict(self, *args, **kwargs):
        """Override dict to exclude ssim_score if None."""
        data = super().dict(*args, **kwargs)
        if self.ssim_score is None:
            data.pop('ssim_score')
        return data
