from pathlib import Path
from typing import Dict, Any
import cv2
import torch
import subprocess
import time
import psutil
import json
from tqdm import tqdm
from loguru import logger
from skimage.metrics import structural_similarity as ssim
from configs.realesrgan_settings import UpscalerSettings
import numpy as np
import os

def setup_logger(log_dir: Path):
    """Set up loguru logger configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "video_enhancer.log"
    
    # Remove default handler and add our custom handlers
    logger.remove()
    
    # Add handlers for both file and console
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
    Video upscaling system using Real-ESRGAN.

    This class implements video upscaling using Real-ESRGAN,
    focusing solely on the upscaling process.

    Attributes:
        settings (UpscalerSettings): Configuration settings for the upscaler
        realesrgan_path (Path): Path to Real-ESRGAN installation
    """

    REALESRGAN_REPO: str = "https://github.com/xinntao/Real-ESRGAN.git"

    def __init__(self, settings: UpscalerSettings):
        """
        Initialize the video upscaler.

        Args:
            settings: Configuration settings for the upscaler
        """
        # Set up logging first
        setup_logger(settings.log_dir)
        
        self.settings = settings
        logger.info(f"Initializing VideoUpscaler with settings: {settings}")
        
        # Set up Real-ESRGAN
        self.realesrgan_path = self._setup_realesrgan()

    def _setup_realesrgan(self) -> Path:
        """
        Set up Real-ESRGAN repository and install dependencies.
        Uses uv for package management.
        """
        realesrgan_path = self.settings.realesrgan_dir
        
        if not realesrgan_path.exists():
            logger.info("Cloning Real-ESRGAN repository...")
            subprocess.run(
                ["git", "clone", self.REALESRGAN_REPO, str(realesrgan_path)],
                check=True
            )
            
            # Change to Real-ESRGAN directory
            original_dir = os.getcwd()
            os.chdir(str(realesrgan_path))
            
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
        
        return realesrgan_path

    def calculate_spatial_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate spatial SSIM between original and upscaled frames.
        The upscaled frame (frame2) will be downscaled to match frame1's dimensions.
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Downscale the upscaled frame to match original dimensions
        if gray2.shape != gray1.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]), interpolation=cv2.INTER_AREA)
            
        return ssim(gray1, gray2, data_range=255)

    def calculate_temporal_ssim(self, frames: np.ndarray) -> float:
        """Calculate temporal SSIM across consecutive frames."""
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
        Handles different resolutions by downscaling the enhanced video for comparison.
        
        Args:
            original_video: Path to original input video
            enhanced_video: Path to upscaled output video
            
        Returns:
            float: ST-SSIM score between 0 and 1
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
            with tqdm(total=total_frames, desc="Calculating ST-SSIM", unit="frame") as pbar:
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
                        pbar.update(1)

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

    def process_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Process a single video through the upscaling pipeline.

        Args:
            video_path: Path to the input video file

        Returns:
            Dictionary containing processing details and results
        """
        start_time = time.time()
        
        # Get input video information
        cap = cv2.VideoCapture(str(video_path))
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate output resolution
        output_width = input_width * self.settings.scale_factor
        output_height = input_height * self.settings.scale_factor

        # Create outputs directory if it doesn't exist
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        # Set output path - just the folder name
        output_path = self.settings.output_dir / video_path.stem

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

        # Create progress bar for video processing
        pbar = tqdm(total=100, desc="Processing video", unit="%")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        def update_progress(line):
            if not line:
                return
                
            logger.debug(f"Progress line: {line.strip()}")
            
            try:
                if 'frame' in line.lower() and '/' in line:
                    # Try to find pattern like "frame: 22/180"
                    parts = line.split('/')
                    if len(parts) >= 2:
                        current = int(''.join(filter(str.isdigit, parts[0])))
                        total = int(''.join(filter(str.isdigit, parts[1])))
                        if total > 0:
                            progress = (current / total) * 100
                            pbar.n = min(round(progress), 100)
                            pbar.refresh()
                elif '%' in line:
                    # Try to find percentage pattern
                    percent_str = ''.join(filter(lambda x: x.isdigit() or x == '.', line.split('%')[0]))
                    if percent_str:
                        progress = float(percent_str)
                        pbar.n = min(round(progress), 100)
                        pbar.refresh()
            except Exception as e:
                logger.debug(f"Error parsing progress: {str(e)} from line: {line.strip()}")

        # Monitor both stdout and stderr for progress updates
        while True:
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if not stdout_line and not stderr_line and process.poll() is not None:
                break
                
            if stdout_line:
                update_progress(stdout_line)
            if stderr_line:
                update_progress(stderr_line)

        pbar.close()
        return_code = process.wait()

        if return_code != 0:
            error_output = process.stderr.read()
            raise RuntimeError(f"Processing failed: {error_output}")

        # The actual output path after Real-ESRGAN processing
        final_output = output_path / f"{video_path.stem}_out.mp4"

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
            "ram_usage_mb": round(ram_usage, 1),
            "processing_time_sec": round(processing_time, 1),
            "message": "Video processed successfully."
        }

        # Calculate SSIM if enabled
        if self.settings.calculate_ssim:
            logger.info("Calculating SSIM score...")
            ssim_score = self.calculate_st_ssim(video_path, final_output)
            result["ssim_score"] = round(ssim_score, 3)
            logger.info(f"SSIM Score: {result['ssim_score']}")

        logger.info(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    input_path = "hxh.mp4"
    model_name = "realesr-animevideov3"
    scale_factor = 4
    face_enhance = False
    log_dir = Path("logs")

    try:
        settings = UpscalerSettings(
            model_name=model_name,
            scale_factor=scale_factor,
            face_enhance=face_enhance,
            log_dir=log_dir
        )

        upscaler = VideoUpscaler(settings)
        upscaler.process_video(Path(input_path))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "message": "Processing failed"
        }, indent=2))
