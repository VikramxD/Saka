"""
Real-ESRGAN Video Upscaling System.

This module provides functionality for upscaling videos using the Real-ESRGAN model.
It handles the complete pipeline from model setup to video processing and quality assessment.

Key Features:
    - Automated Real-ESRGAN setup and dependency management
    - Video upscaling with configurable parameters
    - Progress tracking with rich
    - Quality assessment using SSIM metrics
    - Comprehensive logging
    - Error handling and recovery

Dependencies:
    - Real-ESRGAN
    - OpenCV for video processing
    - torch for model inference
    - rich for progress tracking
    - loguru for logging
    - psutil for system monitoring

Example:
    >>> from configs.realesrgan_settings import UpscalerSettings
    >>> settings = UpscalerSettings()
    >>> upscaler = VideoUpscaler(settings)
    >>> result = upscaler.process_video(Path("input.mp4"))
"""

from pathlib import Path
from typing import Dict, Any
import cv2
import torch
import subprocess
import time
import psutil
import json
from loguru import logger
from skimage.metrics import structural_similarity as ssim
from configs.realesrgan_settings import UpscalerSettings
import numpy as np
import os
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn,
)
from rich.console import Console


def setup_logger(log_dir: Path) -> None:
    """
    Configure loguru logger with file and console outputs.
    
    Sets up two logging handlers:
    1. File handler: Logs all messages to a rotating log file
    2. Console handler: Displays colored output in the terminal
    
    Args:
        log_dir: Directory where log files will be stored
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video_enhancer.log"
    logger.remove()
    logger.add(
        log_file,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        rotation="1 week"
    )
    logger.add(
        lambda msg: print(msg),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )


def setup_realesrgan(realesrgan_dir: Path) -> Path:
    """
    Set up Real-ESRGAN repository and install dependencies.

    This is a one-time setup that should be run before using the VideoUpscaler.
    It will:
    1. Clone the Real-ESRGAN repository if not present
    2. Install dependencies using uv package manager
    3. Set up the package in editable mode

    Args:
        realesrgan_dir: Directory where Real-ESRGAN should be installed

    Returns:
        Path to the Real-ESRGAN installation directory

    Raises:
        subprocess.CalledProcessError: If any installation step fails
    """
    if not realesrgan_dir.exists():
        logger.info("Cloning Real-ESRGAN repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", str(realesrgan_dir)],
            check=True
        )
        
        # Change to Real-ESRGAN directory
        original_dir = os.getcwd()
        os.chdir(str(realesrgan_dir))
        
        try:
            # Install dependencies using uv
            logger.info("Installing dependencies with uv...")
            result = subprocess.run(
                ["uv", "pip", "install", "-r", "requirements.txt"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.debug(f"Dependencies installation output: {result.stdout}")
            
            # Install package in editable mode
            logger.info("Installing Real-ESRGAN in editable mode...")
            result = subprocess.run(
                ["uv", "pip", "install", "-e", ".", "--no-build-isolation"],
                check=True,
                capture_output=True,
                text=True
            )
            logger.debug(f"Package installation output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during setup: {e.stdout}\n{e.stderr}")
            raise
        finally:
            # Always change back to original directory
            os.chdir(original_dir)
            
        logger.success("Real-ESRGAN setup completed successfully")
    else:
        logger.info("Real-ESRGAN repository already exists")
    
    return realesrgan_dir


class VideoUpscaler:
    """
    Video upscaling system using Real-ESRGAN.

    This class implements a complete video upscaling pipeline using the Real-ESRGAN model.
    It handles model setup, video processing, and quality assessment.

    Attributes:
        settings (UpscalerSettings): Configuration settings for the upscaler

    Example:
        >>> settings = UpscalerSettings(model_name="realesr-animevideov3")
        >>> upscaler = VideoUpscaler(settings)
        >>> result = upscaler.process_video(Path("input.mp4"))
    """

    def __init__(self, settings: UpscalerSettings):
        """
        Initialize the video upscaler.

        Args:
            settings: Configuration settings for the upscaler, including model parameters
                     and directory paths
        """
        setup_logger(settings.log_dir)
        self.settings = settings
        logger.info(f"Initializing VideoUpscaler with settings: {settings}")

    def calculate_spatial_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate spatial SSIM between original and upscaled frames.

        The upscaled frame (frame2) will be downscaled to match frame1's dimensions
        for accurate comparison.

        Args:
            frame1: Original frame
            frame2: Upscaled frame to compare against

        Returns:
            Spatial SSIM score between 0 and 1
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Downscale the upscaled frame to match original dimensions
        if gray2.shape != gray1.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]), interpolation=cv2.INTER_AREA)
            
        return ssim(gray1, gray2, data_range=255)

    def calculate_temporal_ssim(self, frames: np.ndarray) -> float:
        """
        Calculate temporal SSIM across consecutive frames.

        Measures the temporal consistency between consecutive frames in a video.

        Args:
            frames: Array of video frames to analyze

        Returns:
            Temporal SSIM score between 0 and 1
        """
        num_frames = len(frames)
        temporal_ssim_scores = []

        for i in range(1, num_frames):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            temporal_ssim_scores.append(ssim(prev_frame, curr_frame, data_range=255))

        return np.mean(temporal_ssim_scores) if temporal_ssim_scores else 0.0

    def calculate_st_ssim(self, original_video: Path, enhanced_video: Path) -> float:
        """
        Calculate Spatio-Temporal SSIM between original and enhanced videos.

        Combines both spatial and temporal SSIM metrics to provide a comprehensive
        quality assessment of the upscaled video.

        Args:
            original_video: Path to original input video
            enhanced_video: Path to upscaled output video

        Returns:
            ST-SSIM score between 0 and 1

        Raises:
            ValueError: If videos cannot be opened or have no frames
        """
        # Wait longer for the enhanced video to be fully written
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            if not enhanced_video.exists():
                logger.warning(f"Enhanced video not found, attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                continue
                
            if enhanced_video.stat().st_size == 0:
                logger.warning(f"Enhanced video file is empty, attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                continue

            try:
                # Try opening both videos
                orig_cap = cv2.VideoCapture(str(original_video))
                enh_cap = cv2.VideoCapture(str(enhanced_video))

                if not orig_cap.isOpened():
                    logger.warning(f"Could not open original video: {original_video}")
                    orig_cap.release()
                    if enh_cap.isOpened():
                        enh_cap.release()
                    time.sleep(retry_delay)
                    continue

                if not enh_cap.isOpened():
                    logger.warning(f"Could not open enhanced video: {enhanced_video}")
                    enh_cap.release()
                    if orig_cap.isOpened():
                        orig_cap.release()
                    time.sleep(retry_delay)
                    continue

                # Both videos opened successfully
                break
            except Exception as e:
                logger.warning(f"Error opening videos (attempt {attempt + 1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                continue
        else:
            logger.error("Failed to open videos after all retries")
            return 0.0

        try:
            # Get video properties
            total_frames = min(
                int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(enh_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )

            if total_frames == 0:
                logger.warning("No frames found in videos")
                return 0.0

            # Initialize frame storage
            orig_frames = []
            enh_frames = []
            spatial_scores = []

            # Process frames with progress bar
            with Progress() as progress:
                task_id = progress.add_task("Calculating ST-SSIM", total=total_frames)
                frame_count = 0
                while frame_count < total_frames:
                    orig_ret, orig_frame = orig_cap.read()
                    enh_ret, enh_frame = enh_cap.read()

                    if not (orig_ret and enh_ret):
                        break

                    try:
                        # Ensure frames are valid
                        if orig_frame is None or enh_frame is None:
                            continue

                        # Calculate spatial SSIM (will handle downscaling internally)
                        spatial_score = self.calculate_spatial_ssim(orig_frame, enh_frame)
                        if not np.isnan(spatial_score):
                            spatial_scores.append(spatial_score)
                        
                        # Store grayscale frames for temporal SSIM
                        # For temporal SSIM, we keep original resolution for each video
                        orig_frames.append(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY))
                        enh_frames.append(cv2.cvtColor(enh_frame, cv2.COLOR_BGR2GRAY))
                        
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                        continue
                    finally:
                        frame_count += 1
                        progress.update(task_id, advance=1)

        finally:
            # Ensure videos are properly closed
            orig_cap.release()
            enh_cap.release()

        if not spatial_scores or not orig_frames:
            logger.warning("No valid frames processed")
            return 0.0

        # Calculate temporal SSIM for each video separately at their native resolutions
        temporal_score_orig = self.calculate_temporal_ssim(np.array(orig_frames))
        temporal_score_enh = self.calculate_temporal_ssim(np.array(enh_frames))
        
        # Average the temporal scores
        temporal_score = (temporal_score_orig + temporal_score_enh) / 2

        # Calculate final ST-SSIM score (50% spatial, 50% temporal)
        spatial_mean = np.mean(spatial_scores)
        st_ssim = spatial_mean * 0.5 + temporal_score * 0.5

        return float(st_ssim) if not np.isnan(st_ssim) else 0.0

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a single video through the upscaling pipeline.

        This method:
        1. Validates input video
        2. Runs Real-ESRGAN upscaling
        3. Calculates quality metrics if enabled
        4. Monitors system resources
        5. Tracks progress

        Args:
            video_path: Path to the input video file

        Returns:
            Dictionary containing:
                - Processing time
                - Output path
                - SSIM score (if enabled)
                - System resource usage
                - Model parameters used

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If upscaling process fails
        """
        start_time = time.time()
        
        # Get input video information
        cap = cv2.VideoCapture(video_path)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate output resolution
        output_width = input_width * self.settings.scale_factor
        output_height = input_height * self.settings.scale_factor

        # Create outputs directory if it doesn't exist
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        # Set output path - Real-ESRGAN will create subfolder and add _out.mp4
        input_file = Path(video_path)
        output_path = self.settings.output_dir / input_file.stem
        final_output = output_path / f"{input_file.stem}_out.mp4"

        # Process video
        cmd = [
            "python",
            str(self.settings.realesrgan_dir / "inference_realesrgan_video.py"),
            "-i", video_path,
            "-o", str(output_path),
            "-n", self.settings.model_name,
            "-s", str(self.settings.scale_factor),
            "-t", str(self.settings.tile_size),
        ]

        # Add optional arguments based on settings
        model_settings = self.settings.get_model_settings()
        if model_settings["fp32"]:
            cmd.append("--fp32")
        if model_settings["face_enhance"]:
            cmd.append("--face_enhance")

        # Create rich progress display
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn(),
            console=console,
            expand=True
        )
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        with progress:
            # Create the main task
            task_id = progress.add_task(
                f"[cyan]Enhancing[/cyan] {Path(video_path).name}",
                total=100,
                start=True
            )

            def update_progress(line: str) -> None:
                if not line:
                    return
                
                line = line.strip().lower()
                
                try:
                    # Try frame-based pattern first (e.g., "frame: 22/180")
                    if 'frame' in line and '/' in line:
                        current = int(''.join(filter(str.isdigit, line.split('/')[0])))
                        total = int(''.join(filter(str.isdigit, line.split('/')[1])))
                        if total > 0:
                            percent = (current / total) * 100
                            progress.update(task_id, completed=percent)
                    
                    # Try percentage pattern (e.g., "50%")
                    elif '%' in line:
                        percent_str = ''.join(filter(lambda x: x.isdigit() or x == '.', line.split('%')[0]))
                        if percent_str:
                            try:
                                percent = float(percent_str)
                                progress.update(task_id, completed=percent)
                            except ValueError:
                                pass
                except Exception as e:
                    pass

            # Process output in real-time
            while True:
                # Read stdout
                output = process.stdout.readline()
                if output:
                    update_progress(output)
                
                # Read stderr
                error = process.stderr.readline()
                if error and "error" in error.lower():
                    logger.error(f"Real-ESRGAN error: {error.strip()}")
                
                # Check if process is still running
                if process.poll() is not None:
                    break
            
            # Process any remaining output
            for line in process.stdout:
                update_progress(line)
            
            for line in process.stderr:
                if "error" in line.lower():
                    logger.error(f"Real-ESRGAN error: {line.strip()}")

            # Ensure progress bar shows completion
            progress.update(task_id, completed=100)
        
        if process.returncode != 0:
            raise RuntimeError(f"Video processing failed with return code {process.returncode}")
            
        # Wait for the output file to be fully written
        max_retries = 10
        retry_delay = 1
        for attempt in range(max_retries):
            if final_output.exists() and final_output.stat().st_size > 0:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        else:
            raise RuntimeError("Output file was not created or is empty")

        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time
        ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)

        # Prepare result dictionary
        result = {
            "video_url": str(final_output),
            "input_resolution": {
                "width": input_width,
                "height": input_height
            },
            "output_resolution": {
                "width": output_width,
                "height": output_height
            },
            "processing_time": round(processing_time, 2),
            "ram_usage_mb": round(ram_usage, 2),
            "model_settings": model_settings
        }

        # Calculate SSIM if enabled
        if self.settings.calculate_ssim:
            ssim_score = self.calculate_st_ssim(Path(video_path), final_output)
            result["ssim_score"] = round(ssim_score, 3)

        return result


if __name__ == "__main__":
    try:
        # Set up input path and settings
        input_path = '/home/jovyan/video-enhancer/scripts/hxh.mp4'
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        # Load settings with custom configuration
        settings = UpscalerSettings(
            model_name="realesr-animevideov3",  # Change model if needed
            scale_factor=4,                      # Change scale factor if needed
            face_enhance=False,                  # Set to True to enhance faces
            use_half_precision=True,             # Set to False for FP32 precision
            input_path=input_path                # Set the input path
        )
        
        # Perform one-time setup of Real-ESRGAN
        setup_realesrgan(settings.realesrgan_dir)
        
        # Create upscaler instance
        upscaler = VideoUpscaler(settings)
        
        # Process video
        result = upscaler.process_video(input_path)
        logger.info(f"Video processing completed: {json.dumps(result, indent=2)}")
            
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        raise
