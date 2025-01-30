import os
import uuid
import time
import tempfile
import base64
from pathlib import Path
import shutil
from typing import Dict, Any, Optional, List
import json
import psutil
import litserve as ls
from fastapi import UploadFile, FastAPI
from pydantic import BaseModel, Field, validator
from datetime import datetime
from scripts.realesrgan import VideoUpscaler
from api.storage import S3Handler
from configs.settings import get_settings
from loguru import logger
import requests
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, multiprocess
from prometheus_client import make_asgi_app

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
                import torch
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
    
    @validator('video_data')
    def validate_video_data(cls, v):
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
    Video enhancement API using Real-ESRGAN model.

    This class provides an API for processing video enhancement requests using the Real-ESRGAN model.
    It handles video upload, enhancement, and delivery of results through S3 storage.
    """

    def setup(self, device: str):
        """Set up API dependencies."""
        self.settings = get_settings()
        self.upscaler = VideoUpscaler(self.settings.realesrgan)
        self.s3 = S3Handler(self.settings.s3)
        self.device = device
        logger.info(f"Initialized API with device: {device}")

    def decode_request(self, request: Dict[str, Any]) -> VideoRequest:
        """Decode and validate the incoming request."""
        try:
            return VideoRequest.parse_obj(request)
        except Exception as e:
            logger.error(f"Error decoding request: {e}")
            raise ValueError(f"Failed to decode request: {e}")

    def predict(self, request: VideoRequest | List[VideoRequest]) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Process video enhancement request(s).
        
        Args:
            request: Single VideoRequest or list of VideoRequest objects
            
        Returns:
            Single result dictionary or list of result dictionaries
        """
        # Handle batched requests
        if isinstance(request, list):
            return [self._process_single_request(req) for req in request]
        
        # Handle single request
        return self._process_single_request(request)
    
    def _process_single_request(self, request: VideoRequest) -> Dict[str, Any]:
        """Process a single video enhancement request."""
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
                self.log("video_size", {"input": len(video_data)})

                # Process video
                start_time = time.time()
                self.settings.realesrgan.calculate_ssim = request.calculate_ssim
                result = self.upscaler.process_video(str(input_path))
                inference_time = time.time() - start_time
                self.log("processing_time", inference_time)

                # Upload to S3 if successful
                if "video_url" in result:
                    output_path = Path(result["video_url"])
                    if output_path.exists():
                        # Log output video size
                        self.log("video_size", {"output": output_path.stat().st_size})
                        
                        # Upload to S3
                        upload_start = time.time()
                        s3_path = f"videos/enhanced_{request_id}.mp4"
                        result["video_url"] = self.s3.upload_video(output_path, s3_path)
                        upload_time = time.time() - upload_start
                        self.log("upload_time", upload_time)

                        # Log resolution scale
                        if "input_resolution" in result and "output_resolution" in result:
                            input_res = result["input_resolution"]
                            output_res = result["output_resolution"]
                            scale = output_res["width"] / input_res["width"]
                            self.log("resolution_scale", scale)

                # Log SSIM score if available
                if "ssim_score" in result:
                    self.log("ssim_score", result["ssim_score"])

                # Log request status
                self.log("request_status", "success")
                return result

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            self.log("request_status", "failure")
            return {"error": str(e)}

    def encode_response(self, result: Dict[str, Any]) -> VideoResponse:
        """Encode the processing result into a standardized response.
        
        Args:
            result: Raw processing result dictionary or error string
            
        Returns:
            VideoResponse object
        """
        # Handle string error results
        if isinstance(result, str):
            return VideoResponse(
                status="error",
                error=result,
                model_settings={},
                timestamp=datetime.utcnow()
            )
            
        # Handle dictionary results
        try:
            if "error" in result:
                return VideoResponse(
                    status="error",
                    error=result["error"],
                    model_settings={},
                    timestamp=datetime.utcnow()
                )

            # Create metrics object
            metrics = VideoMetrics(
                processing_time=result.get("processing_time", 0),
                ram_usage_mb=result.get("ram_usage_mb", 0),
                input_resolution=result.get("input_resolution", {}),
                output_resolution=result.get("output_resolution", {}),
                ssim_score=result.get("ssim_score")
            )

            # Create success response
            return VideoResponse(
                status="success",
                output_url=result["video_url"],
                metrics=metrics,
                model_settings=result.get("model_settings", {}),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error encoding response: {str(e)}")
            return VideoResponse(
                status="error",
                error=str(e),
                model_settings={},
                timestamp=datetime.utcnow()
            )

def main():
    """Main entry point for the Video Enhancer API."""
    settings = get_settings()
    api = VideoEnhancerAPI()
    api.setup("cpu")
    
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
        batch_timeout=0.05,
       
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