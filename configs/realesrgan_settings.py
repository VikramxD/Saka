"""
Configuration settings for Real-ESRGAN video upscaling.

This module defines the configuration settings for the Real-ESRGAN video upscaling system
using Pydantic for validation and environment variable support.

Features:
    - Type validation and coercion
    - Environment variable support
    - Automatic directory creation
    - Path validation
    - Default values for all settings
    - Comprehensive validation rules

Environment Variables:
    - REALESRGAN_MODEL_NAME: Model name to use (default: realesr-animevideov3)
    - REALESRGAN_SCALE_FACTOR: Upscaling factor (1-4, default: 4)
    - REALESRGAN_TILE_SIZE: Tile size for processing (default: 0)
    - REALESRGAN_FACE_ENHANCE: Enable face enhancement (default: false)
    - REALESRGAN_USE_HALF_PRECISION: Use FP16 for inference (default: true)
    - REALESRGAN_OUTPUT_DIR: Output directory path
    - REALESRGAN_MODEL_DIR: Real-ESRGAN model directory
    - REALESRGAN_LOG_DIR: Log directory path
    - REALESRGAN_INPUT_PATH: Input video path (optional)

Example:
    >>> settings = UpscalerSettings()  # Load from environment
    >>> settings = UpscalerSettings(model_name="realesr-animevideov3")  # Direct init
    >>> print(settings.model_name)
    'realesr-animevideov3'
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent

class UpscalerSettings(BaseSettings):
    """
    Settings for Real-ESRGAN video upscaling with environment variable support.
    
    This class uses Pydantic's BaseSettings to provide automatic environment variable
    loading, type validation, and directory management for the video upscaling system.
    
    Attributes:
        model_name (str): Name of the Real-ESRGAN model to use
        scale_factor (int): Upscaling factor (1-4)
        tile_size (int): Tile size for processing (0 means no tiling)
        face_enhance (bool): Whether to enhance faces in the video
        use_half_precision (bool): Whether to use FP16 for inference
        calculate_ssim (bool): Whether to calculate SSIM metrics
        output_dir (Path): Directory for output files
        realesrgan_dir (Path): Directory containing Real-ESRGAN installation
        log_dir (Path): Directory for log files
        input_path (Optional[Path]): Input video path
    
    Example:
        >>> settings = UpscalerSettings(
        ...     model_name="realesr-animevideov3",
        ...     scale_factor=4,
        ...     face_enhance=True
        ... )
        >>> print(settings.output_dir)
        PosixPath('/path/to/outputs')
    """
    
    # Model settings
    model_name: str = Field(
        default="realesr-animevideov3",
        description="Name of the Real-ESRGAN model to use",
        env="REALESRGAN_MODEL_NAME"
    )
    scale_factor: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Upscaling factor",
        env="REALESRGAN_SCALE_FACTOR"
    )
    tile_size: int = Field(
        default=0,
        ge=0,
        description="Tile size for processing. 0 means no tiling",
        env="REALESRGAN_TILE_SIZE"
    )
    face_enhance: bool = Field(
        default=False,
        description="Whether to enhance faces in the video",
        env="REALESRGAN_FACE_ENHANCE"
    )
    use_half_precision: bool = Field(
        default=True,
        description="Whether to use half precision (FP16) for inference",
        env="REALESRGAN_USE_HALF_PRECISION"
    )
    
    # Processing settings
    calculate_ssim: bool = Field(
        default=False,
        description="Whether to calculate SSIM metrics",
        env="REALESRGAN_CALCULATE_SSIM"
    )
    
    # Directory settings - relative to project root
    output_dir: Path = Field(
        default=PROJECT_ROOT / "outputs",
        description="Directory for output files",
        env="REALESRGAN_OUTPUT_DIR"
    )
    realesrgan_dir: Path = Field(
        default=PROJECT_ROOT / "Real-ESRGAN",
        description="Directory containing Real-ESRGAN installation",
        env="REALESRGAN_MODEL_DIR"
    )
    log_dir: Path = Field(
        default=PROJECT_ROOT / "logs",
        description="Directory for log files",
        env="REALESRGAN_LOG_DIR"
    )
    
    # Input settings
    input_path: Optional[Path] = Field(
        default=None,
        description="Input video path",
        env="REALESRGAN_INPUT_PATH"
    )

    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,
        case_sensitive=False
    )

    @validator("output_dir", "realesrgan_dir", "log_dir")
    def create_directories(cls, v) -> Path:
        """
        Create directories if they don't exist.
        
        Args:
            v: Directory path to create
            
        Returns:
            Path object of the created directory
            
        Raises:
            OSError: If directory creation fails
        """
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("input_path")
    def validate_input_path(cls, v) -> Optional[Path]:
        """
        Validate input path if provided.
        
        Args:
            v: Input path to validate
            
        Returns:
            Path object if valid, None if not provided
            
        Raises:
            ValueError: If path doesn't exist
        """
        if v is not None:
            if isinstance(v, str):
                v = Path(v)
            if not v.exists():
                raise ValueError(f"Input video path does not exist: {v}")
        return v

    def get_model_settings(self) -> dict:
        """
        Return a dictionary of model-specific configuration.
        
        Returns:
            Dictionary containing model settings:
                - model_name: Name of the model
                - scale: Upscaling factor
                - tile: Tile size
                - face_enhance: Face enhancement flag
                - fp32: Full precision flag (inverse of half precision)
        """
        return {
            "model_name": self.model_name,
            "scale": self.scale_factor,
            "tile": self.tile_size,
            "face_enhance": self.face_enhance,
            "fp32": not self.use_half_precision
        }
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of settings.
        
        Returns:
            Formatted string containing all settings values
        """
        return (
            f"UpscalerSettings(\n"
            f"  Model: {self.model_name}\n"
            f"  Scale: {self.scale_factor}x\n"
            f"  Tile Size: {self.tile_size}\n"
            f"  Face Enhance: {self.face_enhance}\n"
            f"  Half Precision: {self.use_half_precision}\n"
            f"  Calculate SSIM: {self.calculate_ssim}\n"
            f"  Output Dir: {self.output_dir}\n"
            f"  Real-ESRGAN Dir: {self.realesrgan_dir}\n"
            f"  Log Dir: {self.log_dir}\n"
            f"  Input Path: {self.input_path or 'Not set'}\n"
            f")"
        )
