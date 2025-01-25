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

class VideoEnhancerAPI(ls.LitAPI):
    """Video enhancement API with RabbitMQ task queue integration."""
    
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
            raise ValueError(f"Failed to decode request: {e}")

    def predict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the video enhancement request."""
        try:
            request_data = data[0]
            video_b64 = request_data.get("video_data", "")
            calculate_ssim = request_data.get("calculate_ssim", False)
            
            if not video_b64:
                raise ValueError("No video data provided")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Decode and save input video
                input_path = temp_dir / "input.mp4"
                with open(input_path, "wb") as f:
                    f.write(base64.b64decode(video_b64))
                
                # Process video
                start_time = time.time()
                result = self.upscaler.process_video(str(input_path))
                end_time = time.time()
                
                # Get output path from result
                if isinstance(result, dict):
                    output_path = result.get("video_url")
                    if not output_path:
                        raise ValueError("No output path in result")
                else:
                    output_path = result
                
                # Upload to S3
                s3_path = "videos/enhanced.mp4"
                output_url = self.s3.upload_video(Path(output_path), s3_path)
                
                # Calculate metrics
                processing_time = end_time - start_time
                video_size = Path(output_path).stat().st_size
                
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
                    response["ssim_score"] = result["ssim_score"]
                
                return [response]
                
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return [{
                "status": "error",
                "message": f"Video processing failed: {str(e)}"
            }]

def main():
    """Main entry point for the Video Enhancer API."""
    import argparse
    import threading
    
    settings = get_settings()
    api = VideoEnhancerAPI()
    api.setup("cpu")  # Initialize main API instance
    
    # Create LitServer instance
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices='auto',
        workers_per_device=settings.api.workers,
        max_batch_size=16,
        track_requests=True,
        fast_queue=True
    )
    
    # Start server
    logger.info("Starting API server...")
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