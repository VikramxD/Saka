"""
Configuration settings for Real-ESRGAN video upscaling.

Features:
    - Batch video processing with progress tracking
    - Detailed performance metrics collection
    - Automated environment setup
    - Error handling and recovery
    - JSON-based metrics export

Dependencies:
    - torch with CUDA support
    - Real-ESRGAN
    - OpenCV
    - tqdm for progress tracking

Typical usage:
    settings = UpscalerSettings(model_name="realesr-animevideov3")
    upscaler = VideoUpscaler(settings)
    metrics = upscaler.process_video(video_path)
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Get the parent directory of the project
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class UpscalerSettings:
    """Settings for video upscaling."""
    # Model settings
    model_name: str = "realesr-animevideov3"
    scale_factor: int = 4
    tile_size: int = 0
    face_enhance: bool = False
    use_half_precision: bool = True
    
    # Processing settings
    calculate_ssim: bool = False  # Whether to calculate SSIM metrics
    
    # Directory settings - relative to project root
    output_dir: Path = PROJECT_ROOT / "outputs"
    realesrgan_dir: Path = PROJECT_ROOT / "Real-ESRGAN"
    log_dir: Path = PROJECT_ROOT / "logs"
    
    def __post_init__(self):
        """Convert string paths to Path objects and create directories."""
        # Convert string paths to Path objects if they're strings
        if isinstance(self.output_dir, str):
            self.output_dir = PROJECT_ROOT / self.output_dir
        if isinstance(self.realesrgan_dir, str):
            self.realesrgan_dir = PROJECT_ROOT / self.realesrgan_dir
        if isinstance(self.log_dir, str):
            self.log_dir = PROJECT_ROOT / self.log_dir
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def __str__(self):
        """Pretty string representation of settings."""
        return (
            f"UpscalerSettings(\n"
            f"  Model: {self.model_name}\n"
            f"  Scale: {self.scale_factor}x\n"
            f"  Face enhance: {self.face_enhance}\n"
            f"  Half precision: {self.use_half_precision}\n"
            f"  Calculate SSIM: {self.calculate_ssim}\n"
            f"  Output dir: {self.output_dir}\n"
            f"  Log dir: {self.log_dir}\n"
            f")"
        )
