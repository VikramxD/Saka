import os
import uuid
import time
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import psutil
import litserve as ls
from fastapi import UploadFile, FastAPI
from pydantic import BaseModel, Field, validator
from datetime import datetime
from models import VideoResponse, ProcessingMetrics, VideoRequest
from scripts.realesrgan import VideoUpscaler
from api.storage import S3Handler
from configs.settings import get_settings
from loguru import logger
import requests
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, multiprocess
from prometheus_client import make_asgi_app

# Set up multiprocess mode for Prometheus
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

# Create multiprocess registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class VideoMetrics(BaseModel):
    """Metrics for video processing."""
    inference_time: float = Field(..., description="Time taken for model inference in seconds")
    upload_time: float = Field(..., description="Time taken for S3 upload in seconds")
    input_size: int = Field(..., description="Size of input video in bytes")
    output_size: int = Field(..., description="Size of enhanced video in bytes")
    ssim_score: Optional[float] = Field(None, description="SSIM quality score if calculated")

class VideoRequest(BaseModel):
    """Request model for video enhancement."""
    video_data: str = Field(..., description="Base64 encoded video data")
    calculate_ssim: bool = Field(False, description="Whether to calculate SSIM score")
    
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
    """Response model for individual video enhancement request."""
    status: str = Field(..., description="Status of the request (success/error)")
    request_id: str = Field(..., description="Unique identifier for the request")
    output_url: Optional[str] = Field(None, description="S3 URL of the enhanced video")
    metrics: Optional[VideoMetrics] = Field(None, description="Processing metrics")
    error: Optional[str] = Field(None, description="Error message if status is error")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")

class BatchResponse(BaseModel):
    """Response model for batch processing."""
    batch_size: int = Field(..., description="Number of requests in the batch")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the batch response")
    responses: List[VideoResponse] = Field(..., description="List of individual responses")
    metrics: Dict[str, int] = Field(..., description="Batch-level metrics")

class VideoEnhancerMetrics(ls.Logger):
    """Prometheus metrics logger for video enhancement service."""
    
    def __init__(self):
        super().__init__()
        
        # Processing time metrics
        self.processing_duration = Histogram(
            "video_processing_seconds",
            "Time spent processing video",
            ["operation"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=registry
        )
        
        # File size metrics
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
        self.ssim_score = Histogram(
            "video_ssim_score",
            "SSIM quality score for enhanced videos",
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0],
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
        
        # Batch processing metrics
        self.batch_size = Histogram(
            "batch_size",
            "Number of videos processed in each batch",
            buckets=[1, 2, 4, 8, 16, 32],
            registry=registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            "request_queue_size",
            "Number of requests waiting in queue",
            registry=registry
        )

    def process(self, key: str, value: Any):
        """Process metrics based on key-value pairs logged during processing."""
        
        if key == "inference_time":
            self.processing_duration.labels(operation="inference").observe(value)
            
        elif key == "upload_time":
            self.processing_duration.labels(operation="upload").observe(value)
            
        elif key == "input_size":
            self.video_size.labels(type="input").observe(value)
            
        elif key == "output_size":
            self.video_size.labels(type="output").observe(value)
            
        elif key == "request_status":
            self.requests_total.labels(status=value).inc()
            
        elif key == "ssim_score" and value is not None:
            self.ssim_score.observe(value)
            
        elif key == "batch_size":
            self.batch_size.observe(value)
            
        elif key == "queue_size":
            self.queue_size.set(value)
            
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

class VideoEnhancerAPI(ls.LitAPI):
    """
    Video enhancement API with batching support

    This class provides an API for processing video enhancement requests using the Real-ESRGAN model.
    It includes methods for setting up dependencies, batched processing, and proper request/response handling.
    The API supports uploading enhanced videos to S3 and calculating SSIM metrics.
    """

    def __init__(self):
        super().__init__()
        self.batch_size = 4
        self.max_batch_wait_time = 0.5

    def setup(self, device: str):
        """Set up API dependencies."""
        self.settings = get_settings()
        self.upscaler = VideoUpscaler(self.settings.realesrgan)
        self.s3 = S3Handler(self.settings.s3)
        self.device = device
        logger.info(f"Initialized API with device: {device}")

    def decode_request(self, request: VideoRequest) -> VideoRequest:
        """Decode and validate the incoming request using the VideoRequest model."""
        try:
            if isinstance(request, dict):
                request = VideoRequest.parse_obj(request)
            return request
        except Exception as e:
            logger.error(f"Error decoding request: {e}")
            self.log("request_status", "decode_failed")
            raise ValueError(f"Failed to decode request: {e}")

    def predict(self, batch: List[VideoRequest]) -> List[VideoResponse]:
        """Process a batch of video enhancement requests."""
        responses = []
        self.log("batch_size", len(batch))
        
        for request in batch:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)
                    request_id = str(uuid.uuid4())
                    
                    # Save input video
                    input_path = temp_dir / f"input_{request_id}.mp4"
                    video_data = base64.b64decode(request.video_data)
                    with open(input_path, "wb") as f:
                        f.write(video_data)
                    
                    self.log("input_size", len(video_data))

                    # Process video
                    inference_start = time.time()
                    result = self.upscaler.process_video(str(input_path))
                    inference_time = time.time() - inference_start
                    self.log("inference_time", inference_time)

                    # Handle output
                    output_path = result["video_url"] if isinstance(result, dict) else result
                    if not output_path:
                        raise ValueError("No output path in result")

                    # Upload to S3
                    upload_start = time.time()
                    s3_path = f"videos/enhanced_{request_id}.mp4"
                    output_url = self.s3.upload_video(Path(output_path), s3_path)
                    upload_time = time.time() - upload_start
                    self.log("upload_time", upload_time)

                    # Calculate metrics
                    video_size = Path(output_path).stat().st_size
                    self.log("output_size", video_size)

                    # Create metrics model
                    metrics = VideoMetrics(
                        inference_time=inference_time,
                        upload_time=upload_time,
                        input_size=len(video_data),
                        output_size=video_size,
                        ssim_score=result.get("ssim_score") if isinstance(result, dict) else None
                    )

                    # Create response model
                    response = VideoResponse(
                        status="success",
                        request_id=request_id,
                        output_url=output_url,
                        metrics=metrics
                    )
                    self.log("request_status", "success")
                
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                self.log("request_status", "failed")
                response = VideoResponse(
                    status="error",
                    request_id=request_id,
                    error=str(e)
                )
            
            responses.append(response)
        
        return responses

    def encode_response(self, responses) -> BatchResponse:
        """Encode the response(s) using the BatchResponse model.
        
        Args:
            responses: Either a single VideoResponse or a list of VideoResponse objects
        """
        # Convert single response to list if needed
        if isinstance(responses, VideoResponse):
            responses = [responses]
        elif not isinstance(responses, list):
            raise ValueError(f"Unexpected response type: {type(responses)}")
            
        return BatchResponse(
            batch_size=len(responses),
            responses=responses,
            metrics={
                "success_count": sum(1 for r in responses if r.status == "success"),
                "error_count": sum(1 for r in responses if r.status == "error")
            }
        )

def main():
    """Main entry point for the Video Enhancer API."""
    import argparse
    import threading
    
    settings = get_settings()
    api = VideoEnhancerAPI()
    api.setup("cpu")
    prometheus_logger = VideoEnhancerMetrics()
    prometheus_logger.mount(path="/metrics", app=make_asgi_app(registry=registry))
    
    # Create LitServer instance with Prometheus logger
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices='auto',
        workers_per_device=settings.api.workers,
        max_batch_size=16,
        track_requests=True,
        fast_queue=True,
        loggers=[prometheus_logger]
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