"""
Configuration settings for video enhancement using Spandrel.

This module defines the configuration settings for the video enhancement system
using Pydantic for validation and environment variable support.

Features:
    - Type validation and coercion
    - Environment variable support
    - Automatic directory creation
    - Path validation
    - Default values for all settings
    - Comprehensive validation rules

Environment Variables:
    - MODEL_NAME: Model name to use (default: realesr-animevideov3)
    - SCALE_FACTOR: Upscaling factor (1-4, default: 4)
    - MODEL_PATH: Path to the model file
    - OUTPUT_DIR: Output directory path
    - LOG_DIR: Log directory path
    - INPUT_PATH: Input video path (optional)
    - S3_*: S3 storage configuration variables
    - API_*: API server configuration variables
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class S3Settings(BaseSettings):
    """S3 storage configuration settings."""
    access_key: str = Field('', description="AWS access key or compatible S3 provider key")
    secret_key: str = Field('', description="AWS secret key or compatible S3 provider secret")
    endpoint_url: Optional[str] = Field(None, description="Optional: Custom S3 endpoint URL")
    region: str = Field("ap-south-1", description="S3 region")
    bucket_name: str = Field("sakaresults", description="S3 bucket name")
    storage_path: str = Field(".", description="Base path in bucket for storing videos")
    url_expiration: int = Field(3600, description="Presigned URL expiration time in seconds")
    upload_chunk_size: int = Field(8388608, description="Upload chunk size in bytes (8MB)")
    max_concurrent_uploads: int = Field(4, description="Maximum concurrent upload threads")
    max_retries: int = Field(3, description="Maximum retry attempts for failed operations")
    retry_delay: int = Field(1, description="Delay between retries in seconds")

    class Config:
        env_prefix = "S3_"

class APISettings(BaseSettings):
    """API server configuration settings."""
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    workers: int = Field(4, description="Number of workers")
    timeout: int = Field(300, description="Request timeout in seconds")
    max_video_size: int = Field(104857600, description="Maximum video size in bytes (100MB)")
    allowed_extensions: List[str] = Field(
        default=[".mp4", ".avi", ".mkv", ".mov"],
        description="Allowed video file extensions"
    )

    class Config:
        env_prefix = "API_"

class UpscalerSettings(BaseSettings):
    """
    Configuration settings for video enhancement using Spandrel.
    
    This class uses Pydantic's BaseSettings to provide automatic environment variable
    loading, type validation, and directory management for the video enhancement system.
    
    Attributes:
        model_name (str): Name of the model to use
        scale_factor (int): Upscaling factor (1-4)
        model_path (Path): Path to the model file
        calculate_ssim (bool): Whether to calculate SSIM metrics
        output_dir (Path): Directory for output files
        log_dir (Path): Directory for log files
        input_path (Optional[Path]): Input video path
        s3 (S3Settings): S3 storage configuration
        api (APISettings): API server configuration
    """
    
    # Model settings
    model_name: str = Field(
        default="4x UltraSharp",
        description="Name of the model to use",
        env="MODEL_NAME"
    )
    scale_factor: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Upscaling factor",
        env="SCALE_FACTOR"
    )
    model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "4x_UltraSharp.pth",
        description="Path to the model file",
        env="MODEL_PATH"
    )
    
    # Processing settings
    calculate_ssim: bool = Field(
        default=False,
        description="Whether to calculate SSIM metrics",
        env="CALCULATE_SSIM"
    )
    
    # Directory settings - relative to project root
    output_dir: Path = Field(
        default=PROJECT_ROOT / "outputs",
        description="Directory for output files",
        env="OUTPUT_DIR"
    )
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Directory for log files",
        env="LOG_DIR"
    )
    
    # Input settings
    input_path: Optional[Path] = Field(
        default=None,
        description="Input video path",
        env="INPUT_PATH"
    )

    # S3 settings
    s3: S3Settings = S3Settings()

    # API settings
    api: APISettings = APISettings()

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        case_sensitive=False
    )

    @field_validator("output_dir", "log_dir")
    @classmethod
    def create_directories(cls, v: Path | str) -> Path:
        """Create directories if they don't exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("input_path")
    @classmethod
    def validate_input_path(cls, v: Optional[Path | str]) -> Optional[Path]:
        """Validate input path if provided."""
        if v is not None:
            if isinstance(v, str):
                v = Path(v)
            if not v.exists():
                raise ValueError(f"Input video path does not exist: {v}")
        return v

    def __str__(self) -> str:
        """Return a human-readable string representation of settings."""
        return (
            f"UpscalerSettings(\n"
            f"  Model: {self.model_name}\n"
            f"  Scale: {self.scale_factor}x\n"
            f"  Model Path: {self.model_path}\n"
            f"  Calculate SSIM: {self.calculate_ssim}\n"
            f"  Output Dir: {self.output_dir}\n"
            f"  Log Dir: {self.log_dir}\n"
            f"  Input Path: {self.input_path or 'Not set'}\n"
            f"  S3 Bucket: {self.s3.bucket_name}\n"
            f"  API Port: {self.api.port}\n"
            f")"
        )
