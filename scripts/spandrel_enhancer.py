"""
Video Enhancement with Spandrel

This module provides functionality for video enhancement using Spandrel model loading.
It handles the complete pipeline from model loading to video processing and quality assessment.

Key Features:
    - Efficient model loading with Spandrel
    - Video upscaling with configurable parameters
    - Progress tracking with rich
    - Quality assessment using SSIM metrics
    - Comprehensive logging and monitoring
    - Error handling and recovery
    - Integration with Prometheus metrics

Dependencies:
    - spandrel for model loading
    - torch for model inference
    - opencv-python for video processing
    - rich for progress tracking
    - prometheus_client for metrics
    - loguru for logging
    - psutil for system monitoring

Example:
    >>> from configs.settings import get_settings
    >>> settings = get_settings()
    >>> upscaler = VideoUpscaler(settings)
    >>> result = upscaler.process_video("input.mp4")
"""

from pathlib import Path
from typing import Dict, Any, List
import cv2
import torch
import time
import psutil
import json
from loguru import logger
from skimage.metrics import structural_similarity as ssim
from spandrel import ModelLoader
from configs.spandrel_settings import UpscalerSettings
import numpy as np
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


class VideoUpscaler:
    """
    Video upscaling system using Spandrel-loaded models.

    This class implements a complete video upscaling pipeline using models loaded through Spandrel.
    It handles model loading, video processing, and quality assessment.

    Attributes:
        settings (UpscalerSettings): Configuration settings for the upscaler
        model: The loaded Spandrel model

    Example:
        >>> settings = UpscalerSettings(model_name="realesr-animevideov3")
        >>> upscaler = VideoUpscaler(settings)
        >>> result = upscaler.process_video("input.mp4")
    """

    def __init__(self, settings: UpscalerSettings):
        """
        Initialize the video upscaler.

        Args:
            settings: Configuration settings for the upscaler
        """
        setup_logger(settings.log_dir)
        self.settings = settings
        self.model = None
        self._load_model()
        
        # Create output directory
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load model using Spandrel."""
        logger.info(f"Loading model: {self.settings.model_name}")
        
        try:
            model_loader = ModelLoader()
            self.model = model_loader.load_from_file(str(self.settings.model_path))
            self.model.cuda().eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

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
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
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

    def detect_content_type(self, frame: np.ndarray) -> str:
        """Determine content type using edge density analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size
        content_type = "anime" if edge_density > 0.1 else "realistic"
        logger.debug(f"Detected content type: {content_type} (edge density: {edge_density:.3f})")
        return content_type

    def process_frame(self, frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """Process a single frame through the model."""
        try:
            # Convert to RGB and normalize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float().div(255.0)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
            
            # Process with model
            with torch.no_grad():
                output_tensor = self.model(frame_tensor)
            
            # Convert back to numpy
            output_frame = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_frame = (output_frame * 255.0).clip(0, 255).astype(np.uint8)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if scale_factor != 4.0:
                h, w = frame.shape[:2]
                target_h = int(h * scale_factor)
                target_w = int(w * scale_factor)
                logger.debug(f"Downsampling from 4x to {scale_factor}x ({target_w}x{target_h})")
                output_frame = cv2.resize(output_frame, (target_w, target_h), 
                                        interpolation=cv2.INTER_LANCZOS4)
            
            return output_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            raise

    def process_chunk(self, frames: List[np.ndarray], scale_factor: float) -> List[np.ndarray]:
        """Process a chunk of frames in parallel."""
        try:
            processed_frames = []
            for frame in frames:
                processed_frame = self.process_frame(frame, scale_factor)
                processed_frames.append(processed_frame)
            return processed_frames
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            raise

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a single video through the upscaling pipeline.

        This method:
        1. Validates input video
        2. Runs model inference
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
            RuntimeError: If processing fails
        """
        start_time = time.time()
        
        # Get input video information
        cap = cv2.VideoCapture(video_path)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate output resolution
        output_width = input_width * self.settings.scale_factor
        output_height = input_height * self.settings.scale_factor
        
        # Set up output path
        input_file = Path(video_path)
        output_path = self.settings.output_dir / f"{input_file.stem}_enhanced.mp4"
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps,
            (output_width, output_height)
        )
        
        try:
            # Process frames with progress tracking
            with Progress() as progress:
                task_id = progress.add_task(
                    f"[cyan]Enhancing[/cyan] {input_file.name}",
                    total=total_frames
                )
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Convert frame to tensor and process
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                        input_tensor = input_tensor.unsqueeze(0)
                        
                        # Run inference
                        output_tensor = self.model(input_tensor)
                        
                        # Convert back to frame
                        output_frame = (
                            output_tensor.squeeze(0)
                            .permute(1, 2, 0)
                            .mul(255)
                            .byte()
                            .cpu()
                            .numpy()
                        )
                        
                        # Write frame
                        out.write(output_frame)
                        
                    progress.update(task_id, advance=1)
                    
        finally:
            cap.release()
            out.release()
            
        # Calculate metrics
        end_time = time.time()
        processing_time = end_time - start_time
        ram_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Prepare result
        result = {
            "video_url": str(output_path),
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
            "model_settings": {
                "model_name": self.settings.model_name,
                "scale": self.settings.scale_factor
            }
        }
        
        # Calculate SSIM if enabled
        if self.settings.calculate_ssim:
            ssim_score = self.calculate_st_ssim(Path(video_path), output_path)
            result["ssim_score"] = round(ssim_score, 3)
            
        return result


if __name__ == "__main__":
    try:
        # Set up input path and settings
        input_path = 'input.mp4'
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        # Load settings
        settings = UpscalerSettings(
            model_name="realesr-animevideov3",
            scale_factor=4,
            input_path=input_path
        )
        
        # Process video
        upscaler = VideoUpscaler(settings)
        result = upscaler.process_video(input_path)
        logger.info(f"Video processing completed: {json.dumps(result, indent=2)}")
            
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        raise
