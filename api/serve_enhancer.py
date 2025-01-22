import os
import time
import uuid
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from loguru import logger
import litserve as ls
from fastapi import UploadFile, File, Form, Header, HTTPException
from prometheus_client import (
    CollectorRegistry,
    Histogram,
    Gauge,
    multiprocess,
    make_asgi_app
)

from models import VideoResponse, ProcessingMetrics
from scripts.realesrgan import VideoUpscaler
from api.storage import S3Handler
from configs.settings import get_settings, PrometheusSettings

PROMETHEUS_DIR = "/tmp/prometheus_multiproc_dir"
os.environ["PROMETHEUS_MULTIPROC_DIR"] = PROMETHEUS_DIR
Path(PROMETHEUS_DIR).mkdir(parents=True, exist_ok=True)
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

class VideoEnhancerMetrics(ls.Logger):
    """Prometheus metrics logger for video enhancement."""
    
    def __init__(self, config: PrometheusSettings = None):
        super().__init__()
        if config is None:
            config = get_settings().prometheus
        self.config = config
        prefix = config.prefix
        
        # Processing time metrics
        self.processing_time = Histogram(
            f"{prefix}_processing_seconds",
            "Time spent processing video",
            ["operation"],
            registry=registry
        )
        
        # Video metrics
        self.video_size = Histogram(
            f"{prefix}_video_size_bytes",
            "Size of processed videos",
            buckets=[1e6, 1e7, 1e8, 1e9],
            registry=registry
        )
        
        # Resource metrics
        self.ram_usage = Gauge(
            f"{prefix}_ram_usage_bytes",
            "Current RAM usage",
            registry=registry,
            multiprocess_mode='livesum'
        )
        
        self.gpu_memory = Gauge(
            f"{prefix}_gpu_memory_bytes",
            "Current GPU memory usage",
            registry=registry,
            multiprocess_mode='livesum'
        )
        
        # Mount metrics endpoint
        self.mount(path=config.path, app=make_asgi_app(registry=registry))

    def process(self, key: str, value: float):
        """Process metrics from API."""
        logger.debug("Processing metric: {} = {}", key, value)
        
        if key == "inference_time":
            self.processing_time.labels(operation="inference").observe(value)
        elif key == "video_size":
            self.video_size.observe(value)
        elif key == "ram_usage":
            self.ram_usage.set(value)
        elif key == "gpu_memory":
            self.gpu_memory.set(value)
        elif key.startswith("error_"):
            logger.error("Error occurred: {}", key)

class VideoEnhancerAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize components during startup."""
        try:
            self.settings = get_settings()
            self.upscaler = VideoUpscaler(self.settings.realesrgan)
            self.storage = S3Handler(self.settings.s3)
            logger.info("Video Enhancer API initialized on device: {}", device)
            
        except Exception as e:
            logger.exception("Failed to initialize API")
            raise

    def predict(self, request: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the video and return VideoResponse.
        
        Args:
            request: List of request dictionaries, each containing:
                - video_base64: Base64 encoded video bytes
                - calculate_ssim: Whether to calculate SSIM
            
        Returns:
            List of dictionaries with output_url and metrics
            
        Raises:
            ValueError: If video is not provided
        """
        results = []
        for single_request in request:
            video_base64 = single_request.get("video_base64")
            if not video_base64:
                raise ValueError("No video provided in request")

            calculate_ssim = single_request.get("calculate_ssim", False)
            
            # Create temp file for video
            temp_files = []
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                temp_files.append(Path(temp_video.name))
                video_bytes = base64.b64decode(video_base64)
                temp_video.write(video_bytes)
                input_path = Path(temp_video.name)

            try:
                start_time = time.perf_counter()
                self.log("video_size", len(video_bytes))

                self.upscaler.settings.calculate_ssim = calculate_ssim
                result = self.upscaler.process_video(input_path)
                self.log("inference_time", time.perf_counter() - start_time)
                self.log("ram_usage", result["ram_usage_mb"] * 1e6)  # Convert to bytes

                output_path = f"videos/{uuid.uuid4()}/enhanced.mp4"
                output_url = self.storage.upload_video(
                    Path(result["video_url"]),
                    output_path
                )

                results.append({
                    "output_url": output_url,
                    "metrics": {
                        "ram_usage_mb": result["ram_usage_mb"],
                        "processing_time_sec": time.perf_counter() - start_time,
                        "ssim_score": result.get("ssim_score") if calculate_ssim else None
                    }
                })

            except Exception as e:
                error_type = type(e).__name__.lower()
                self.log(f"error_{error_type}", 1)
                logger.exception("Processing failed")
                raise

            finally:
                # Cleanup temporary files
                for temp_file in temp_files:
                    temp_file.unlink(missing_ok=True)
                    
        return results

if __name__ == "__main__":
    settings = get_settings()
    metrics_logger = VideoEnhancerMetrics()
    api = VideoEnhancerAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices = 'auto',
        workers_per_device = settings.api.workers,
        max_batch_size=16,
        loggers=metrics_logger,
        track_requests = True,

    )
    
    server.run(
        host=settings.api.host,
        port=settings.api.port
    )