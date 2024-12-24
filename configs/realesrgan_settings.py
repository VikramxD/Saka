from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class UpscalerSettings(BaseSettings):
    """Configuration settings for batch video upscaling."""

    input_dir: Path = Field(Path('/root/pixelupbench/data/anime'),description="Directory containing input videos")
    output_dir: Path = Field(default=Path("../test_results"), description="Base directory for outputs")
    model_name: str = Field(default="realesr-animevideov3", description="Model name")
    scale_factor: int = Field(default=2)
    tile_size: int = Field(default=0)
    face_enhance: bool = Field(default=False)
    use_half_precision: bool = Field(default=True)
    gpu_device: int = Field(default=0)

    class Config:
        env_prefix = "UPSCALER_"