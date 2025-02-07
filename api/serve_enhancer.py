import os
import uuid
import time
import tempfile
import base64
from pathlib import Path
import shutil
from typing import Dict, Any, Optional, List, Tuple
import json
import psutil
import litserve as ls
import torch
from fastapi import UploadFile, FastAPI
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from scripts.spandrel_enhancer import VideoUpscaler
from api.storage import S3Handler
from configs.spandrel_settings import UpscalerSettings
from loguru import logger
import requests
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, multiprocess
from prometheus_client import make_asgi_app
import cv2
import numpy as np

# Set up multiprocess mode for Prometheus
METRICS_DIR = "/tmp/prometheus_multiproc_dir"

# Clean up old metrics files
if os.path.exists(METRICS_DIR):
    shutil.rmtree(METRICS_DIR)
os.makedirs(METRICS_DIR, mode=0o777, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = METRICS_DIR

# Create multiprocess registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class VideoEnhancerMetrics(ls.Logger):
    """Prometheus metrics logger for video enhancement service."""
    
    def __init__(self):
        super().__init__()
        
        # Processing time metrics (split by operation)
        self.processing_duration = Histogram(
            "video_processing_seconds",
            "Time spent processing video",
            ["operation"],  # inference/upload
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=registry
        )
        
        # Video size metrics
        self.video_size = Histogram(
            "video_size_bytes",
            "Video file size in bytes",
            ["type"],  # input/output
            buckets=[1e6, 5e6, 10e6, 50e6, 100e6, 500e6, 1e9],  # 1MB to 1GB
            registry=registry
        )
        
        # Request counters
        self.requests_total = Counter(
            "video_requests_total",
            "Total number of video enhancement requests",
            ["status"],  # success/failure
            registry=registry
        )
        
        # SSIM metrics
        self.ssim_score = Gauge(  # Changed to Gauge for current average
            "video_ssim_score",
            "SSIM quality score for enhanced videos",
            registry=registry
        )
        
        # Resource usage metrics
        self.gpu_memory_usage = Gauge(
            "gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            registry=registry
        )
        
        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            registry=registry
        )
        
        self.ram_usage = Gauge(
            "ram_usage_bytes",
            "RAM usage in bytes",
            registry=registry
        )
        
        # Resolution metrics
        self.resolution_scale = Histogram(
            "resolution_scale_factor",
            "Scale factor between input and output resolution",
            buckets=[1, 2, 4, 8, 16],
            registry=registry
        )

    def process(self, key: str, value: Any):
        """Process metrics based on key-value pairs logged during processing."""
        
        if key == "processing_time":
            self.processing_duration.labels(operation="inference").observe(value)
            
        elif key == "upload_time":
            self.processing_duration.labels(operation="upload").observe(value)
            
        elif key == "video_size":
            if isinstance(value, dict):
                if "input" in value:
                    self.video_size.labels(type="input").observe(value["input"])
                if "output" in value:
                    self.video_size.labels(type="output").observe(value["output"])
            
        elif key == "request_status":
            self.requests_total.labels(status=value).inc()
            
        elif key == "ssim_score" and value is not None:
            self.ssim_score.set(value)  # Using set instead of observe for current value
            
        elif key == "resolution_scale" and value is not None:
            self.resolution_scale.observe(value)
            
        # Update resource usage metrics
        self._update_resource_metrics()

    def _update_resource_metrics(self):
        """Update system resource usage metrics."""
        try:
            # CPU usage
            self.cpu_usage.set(psutil.cpu_percent())
            
            # RAM usage
            ram = psutil.virtual_memory()
            self.ram_usage.set(ram.used)
            
            # GPU memory if available
            try:
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated()
                    self.gpu_memory_usage.set(gpu_mem)
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")

class VideoMetrics(BaseModel):
    """Metrics for video processing."""
    processing_time: float = Field(..., description="Time taken for model inference in seconds")
    ram_usage_mb: float = Field(..., description="RAM usage in megabytes")
    input_resolution: Dict[str, int] = Field(..., description="Input video resolution")
    output_resolution: Dict[str, int] = Field(..., description="Output video resolution")
    ssim_score: Optional[float] = Field(None, description="SSIM quality score if calculated")

class VideoRequest(BaseModel):
    """Request model for video enhancement."""
    video_data: str = Field(..., description="Base64 encoded video data")
    calculate_ssim: bool = Field(default=False, description="Whether to calculate SSIM score (computationally expensive)")
    
    @field_validator('video_data')
    @classmethod
    def validate_video_data(cls, v: str) -> str:
        try:
            data = base64.b64decode(v)
            if len(data) == 0:
                raise ValueError("Empty video data")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 video data: {str(e)}")

class VideoResponse(BaseModel):
    """Response model for video enhancement request."""
    status: str = Field(..., description="Status of the request (success/error)")
    output_url: Optional[str] = Field(None, description="S3 URL of the enhanced video")
    metrics: Optional[VideoMetrics] = Field(None, description="Processing metrics")
    model_settings: Dict[str, Any] = Field(..., description="Model settings used for enhancement")
    error: Optional[str] = Field(None, description="Error message if status is error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")

class VideoEnhancerAPI(ls.LitAPI):
    """
    Video enhancement API using Spandrel model.

    This class provides an API for processing video enhancement requests using Spandrel model loading.
    It handles video upload, enhancement, and delivery of results through S3 storage.
    Features:
        - Parallel chunk processing for faster video enhancement
        - Dynamic content type detection (anime vs realistic)
        - Automatic video reassembly
        - S3 storage integration
        - Comprehensive metrics tracking
    """

    def setup(self, device: str):
        """Initialize model and metrics."""
        self.settings = UpscalerSettings()
        self.device = device
        
        # Initialize model with Spandrel
        logger.info("Loading model with Spandrel...")
        self.upscaler = VideoUpscaler(self.settings)
        self.s3 = S3Handler(self.settings.s3)
        
        # Set model name for metrics
        self.metrics_logger = VideoEnhancerMetrics()
        self.metrics_logger.model_name = self.settings.model_name

    def decode_request(self, request: Dict[str, Any]) -> VideoRequest:
        """Decode and validate the incoming request."""
        try:
            return VideoRequest.parse_obj(request)
        except Exception as e:
            logger.error(f"Error decoding request: {e}")
            raise ValueError(f"Failed to decode request: {e}")

    def split_video_by_fps(self, video_path: Path) -> List[Tuple[int, List[np.ndarray]]]:
        """Split video into temporal chunks for parallel processing."""
        logger.info(f"Starting video split for: {video_path}")
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties: FPS={fps}, Frames={total_frames}")
        
        duration = total_frames / fps
        logger.debug(f"Video stats - FPS: {fps}, Frames: {total_frames}, Duration: {duration}s")
        
        try:
            chunks = []
            for chunk_idx in range(int(duration)):
                frames = []
                start_frame = chunk_idx * fps
                end_frame = min((chunk_idx + 1) * fps, total_frames)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(int(start_frame), int(end_frame)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                if frames:
                    chunks.append((chunk_idx, frames))
                    logger.debug(f"Added chunk {chunk_idx} with {len(frames)} frames")
            
            if not chunks:
                raise ValueError("No valid frames could be read from the video")
            
            return chunks, fps
            
        finally:
            cap.release()

    def combine_chunks(self, chunks: List[Tuple[int, List[np.ndarray]]], output_path: Path, fps: float):
        """Combine processed chunks into final video."""
        logger.info(f"Combining {len(chunks)} chunks into final video")
        
        chunks.sort(key=lambda x: x[0])  # Sort by chunk index
        h, w = chunks[0][1][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        try:
            total_frames = sum(len(frames) for _, frames in chunks)
            processed_frames = 0
            
            for chunk_idx, frames in chunks:
                for frame in frames:
                    out.write(frame)
                    processed_frames += 1
                logger.debug(f"Written chunk {chunk_idx} ({processed_frames}/{total_frames} frames)")
        finally:
            out.release()

    def predict(self, request: VideoRequest | List[VideoRequest]) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Process video enhancement request(s) with parallel chunk processing."""
        if isinstance(request, list):
            return [self._process_single_request(req) for req in request]
        return self._process_single_request(request)
    
    def _process_single_request(self, request: VideoRequest) -> Dict[str, Any]:
        """Process a single video enhancement request using parallel chunk processing."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                request_id = str(uuid.uuid4())
                
                # Save input video
                input_path = temp_dir / f"input_{request_id}.mp4"
                video_data = base64.b64decode(request.video_data)
                with open(input_path, "wb") as f:
                    f.write(video_data)

                # Log input video size
                self.metrics_logger.video_size.labels(type="input").observe(len(video_data))

                # Split video into chunks
                start_time = time.time()
                chunks, fps = self.split_video_by_fps(input_path)
                
                # Process chunks in parallel
                processed_chunks = []
                for chunk_idx, frames in chunks:
                    result = self.upscaler.process_chunk(frames, self.settings.scale_factor)
                    processed_chunks.append((chunk_idx, result))
                
                # Combine chunks
                output_path = temp_dir / f"output_{request_id}.mp4"
                self.combine_chunks(processed_chunks, output_path, fps)
                
                inference_time = time.time() - start_time
                self.metrics_logger.processing_duration.labels(operation="inference").observe(inference_time)

                # Upload to S3
                if output_path.exists():
                    output_size = output_path.stat().st_size
                    self.metrics_logger.video_size.labels(type="output").observe(output_size)
                    
                    upload_start = time.time()
                    s3_path = f"videos/enhanced_{request_id}.mp4"
                    video_url = self.s3.upload_video(output_path, s3_path)
                    upload_time = time.time() - upload_start
                    self.metrics_logger.processing_duration.labels(operation="upload").observe(upload_time)

                    # Calculate metrics
                    input_resolution = self._get_video_resolution(input_path)
                    output_resolution = self._get_video_resolution(output_path)
                    if input_resolution and output_resolution:
                        scale = output_resolution["width"] / input_resolution["width"]

                    # Calculate SSIM if requested
                    ssim_score = None
                    if request.calculate_ssim:
                        ssim_score = self._calculate_ssim(input_path, output_path)
                        if ssim_score is not None:
                            self.metrics_logger.ssim_score.set(ssim_score)

                    # Log request status
                    self.metrics_logger.requests_total.labels(status="success").inc()

                    # Log GPU memory usage
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(self.device)
                        self.metrics_logger.gpu_memory_usage.set(gpu_memory)

                    # Prepare response
                    return {
                        "status": "success",
                        "output_url": video_url,
                        "metrics": VideoMetrics(
                            processing_time=inference_time,
                            ram_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                            input_resolution=input_resolution,
                            output_resolution=output_resolution,
                            ssim_score=ssim_score
                        ),
                        "model_settings": {
                            "model_name": self.settings.model_name,
                            "scale_factor": self.settings.scale_factor,
                            "calculate_ssim": request.calculate_ssim
                        },
                        "timestamp": datetime.utcnow()
                    }

                else:
                    raise RuntimeError("Failed to generate output video")

        except Exception as e:
            logger.exception("Error processing video request")
            self.metrics_logger.requests_total.labels(status="failure").inc()
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }

    def _get_video_resolution(self, video_path: Path) -> Optional[Dict[str, int]]:
        """Get video resolution."""
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return {"width": width, "height": height}
        return None

    def _calculate_ssim(self, original_path: Path, enhanced_path: Path) -> Optional[float]:
        """Calculate SSIM between original and enhanced videos."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Read first frames for comparison
            cap_orig = cv2.VideoCapture(str(original_path))
            cap_enh = cv2.VideoCapture(str(enhanced_path))
            
            if not (cap_orig.isOpened() and cap_enh.isOpened()):
                return None
            
            ret1, frame1 = cap_orig.read()
            ret2, frame2 = cap_enh.read()
            
            if not (ret1 and ret2):
                return None
                
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Resize enhanced frame to match original
            if gray1.shape != gray2.shape:
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            score = ssim(gray1, gray2, data_range=255)
            return float(score)
            
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return None
        finally:
            if 'cap_orig' in locals():
                cap_orig.release()
            if 'cap_enh' in locals():
                cap_enh.release()

def main():
    """Main entry point for the Video Enhancer API."""
    settings = UpscalerSettings()
    api = VideoEnhancerAPI()
    
    # Initialize Prometheus metrics
    prometheus_logger = VideoEnhancerMetrics()
    prometheus_logger.mount(path="/metrics", app=make_asgi_app(registry=registry))
    
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices='auto',
        workers_per_device=settings.api.workers,
        track_requests=True,
        fast_queue=True,
        loggers=[prometheus_logger],
        max_batch_size=16,
        batch_timeout=0.05
    )
    
    logger.info("Starting API server with Prometheus metrics...")
    try:
        server.run(
            host=settings.api.host,
            port=settings.api.port,
            workers=settings.api.workers
        )
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        logger.info("Server shutdown complete")

if __name__ == "__main__":
    main()