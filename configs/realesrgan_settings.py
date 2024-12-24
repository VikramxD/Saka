from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class UpscalerSettings(BaseSettings):
    """Configuration settings for Real-ESRGAN video upscaling."""

    output_dir: Path = Field(default=Path("../outputs"), description="Base directory for processed videos")
    model_name: str = Field(default="realesr-animevideov3", description="Real-ESRGAN model name")
    scale_factor: int = Field(default=2, description="Video upscaling factor", ge=1, le=4)
    tile_size: int = Field( default=0, description="Tile size for processing (0 for auto)")
    face_enhance: bool = Field(default=False, description="Enable face enhancement")
    use_half_precision: bool = Field(default=True, description="Use FP16 (half) precision for faster processing")
    gpu_device: int = Field(default=0, description="GPU device ID to use")

    class Config:
        env_prefix = "UPSCALER_"
