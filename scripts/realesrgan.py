from pathlib import Path
from typing import Dict, Any
import cv2
import torch
import subprocess
from loguru import logger
import git
from configs.realesrgan_settings import UpscalerSettings



"""
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
    - loguru for logging
    - tqdm for progress tracking

Typical usage:
    settings = UpscalerSettings(input_dir=Path("videos"), model_name="RealESRGAN_x4plus")
    upscaler = VideoUpscaler(settings)
    metrics = upscaler.process_batch()
"""


class VideoUpscaler:
    """
    Video upscaling system using Real-ESRGAN.

    This class implements video upscaling using Real-ESRGAN,
    focusing solely on the upscaling process.

    Attributes:
        settings (UpscalerSettings): Configuration settings for the upscaler
        realesrgan_path (Path): Path to Real-ESRGAN installation
    """

    REALESRGAN_REPO: str = "https://github.com/xinntao/Real-ESRGAN.git"

    def __init__(self, settings: UpscalerSettings) -> None:
        """
        Initialize the upscaler with provided settings.

        Args:
            settings: Configuration settings for video processing

        Raises:
            RuntimeError: If CUDA GPU is not available
        """
        self.settings = settings
        self.realesrgan_path = self._setup_environment()
        logger.info(f"Using model: {settings.model_name}")

    def _setup_environment(self) -> Path:
        """Set up the processing environment and dependencies."""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not detected. CUDA-capable GPU is required.")

        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        realesrgan_path = Path("../Real-ESRGAN")

        if not realesrgan_path.exists():
            logger.info("Setting up Real-ESRGAN...")
            git.Repo.clone_from(self.REALESRGAN_REPO, realesrgan_path)
            subprocess.run(["pip", "install", "-r", str(realesrgan_path / "requirements.txt")], check=True)

        return realesrgan_path

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Process a single video through the upscaling pipeline.

        Args:
            video_path: Path to the input video file

        Returns:
            Dictionary containing:
                - input_resolution: {width, height}
                - output_resolution: {width, height}
                - output_path: Path to processed video
        """
        # Get input video information
        cap = cv2.VideoCapture(str(video_path))
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate output resolution
        output_width = input_width * self.settings.scale_factor
        output_height = input_height * self.settings.scale_factor

        # Create output directory structure
        output_dir = self.settings.output_dir / self.settings.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_output.mp4"

        # Process video
        cmd = [
            "python",
            str(self.realesrgan_path / "inference_realesrgan_video.py"),
            "-i", str(video_path),
            "-o", str(output_path),
            "-n", self.settings.model_name,
            "-s", str(self.settings.scale_factor),
            "-t", str(self.settings.tile_size),
        ]

        if not self.settings.use_half_precision:
            cmd.append("--fp32")
        if self.settings.face_enhance:
            cmd.append("--face_enhance")

        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"Processing failed: {process.stderr}")

        return {
            "input_resolution": {"width": input_width, "height": input_height},
            "output_resolution": {"width": output_width, "height": output_height},
            "output_path": str(output_path)
        }

