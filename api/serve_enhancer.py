import os
import uuid
import time
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, List
import json
import psutil
import litserve as ls
from fastapi import UploadFile, FastAPI
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
    Video enhancement API 

    This class provides an API for processing video enhancement requests using the Real-ESRGAN model.
    It includes methods for setting up dependencies, decoding requests, and processing video enhancement tasks.
    The API also supports uploading enhanced videos to S3 and calculating SSIM metrics.
    """

    def __init__(self):
        super().__init__()

    def setup(self, device: str):
        """Set up API dependencies."""
        self.settings = get_settings()
        self.upscaler = VideoUpscaler(self.settings.realesrgan)
        self.s3 = S3Handler(self.settings.s3)
        logger.info(f"Initialized API with device: {device}")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the incoming request."""
        try:
            return request
        except Exception as e:
            logger.error(f"Error decoding request: {e}")
            self.log("request_status", "decode_failed")
            raise ValueError(f"Failed to decode request: {e}")

    def predict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple video enhancement requests in a batch."""
        responses = []
        
        # Log batch size
        self.log("batch_size", len(data))
        
        for request_data in data:
            try:
                video_b64 = request_data.get("video_data", "")
                calculate_ssim = request_data.get("calculate_ssim", False)

                if not video_b64:
                    self.log("request_status", "invalid_input")
                    responses.append({
                        "status": "error",
                        "message": "No video data provided"
                    })
                    continue

                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)

                    # Decode and save input video
                    input_path = temp_dir / f"input_{uuid.uuid4()}.mp4"
                    video_data = base64.b64decode(video_b64)
                    with open(input_path, "wb") as f:
                        f.write(video_data)
                    
                    # Log input video size
                    input_size = len(video_data)
                    self.log("input_size", input_size)

                    # Process video
                    inference_start = time.time()
                    result = self.upscaler.process_video(str(input_path))
                    inference_time = time.time() - inference_start
                    self.log("inference_time", inference_time)

                    # Get output path from result
                    if isinstance(result, dict):
                        output_path = result.get("video_url")
                        if not output_path:
                            self.log("request_status", "no_output")
                            raise ValueError("No output path in result")
                    else:
                        output_path = result

                    upload_start = time.time()
                    s3_path = f"videos/enhanced_{uuid.uuid4()}.mp4"
                    output_url = self.s3.upload_video(Path(output_path), s3_path)
                    self.log("upload_time", time.time() - upload_start)

                    # Calculate metrics
                    processing_time = inference_time
                    video_size = Path(output_path).stat().st_size
                    self.log("output_size", video_size)

                    # Prepare response
                    response = {
                        "status": "complete",
                        "message": "Video enhancement complete",
                        "output_url": output_url,
                        "inference_time": processing_time,
                        "video_size": video_size
                    }

                    # Add SSIM score if calculated
                    if calculate_ssim and isinstance(result, dict) and "ssim_score" in result:
                        ssim = result["ssim_score"]
                        response["ssim_score"] = ssim
                        self.log("ssim_score", ssim)

                    self.log("request_status", "success")
                    responses.append(response)

            except Exception as e:
                logger.error(f"Error processing video: {e}")
                self.log("request_status", "error")
                responses.append({
                    "status": "error",
                    "message": f"Video processing failed: {str(e)}"
                })

        return responses

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
        loggers=prometheus_logger

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