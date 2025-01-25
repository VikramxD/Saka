import os
import uuid
import time
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, List
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app
import json
import psutil
import litserve as ls
from fastapi import UploadFile
from models import VideoResponse, ProcessingMetrics, VideoRequest
from scripts.realesrgan import VideoUpscaler
from api.storage import S3Handler
from api.rabbitmq_handler import RabbitMQHandler
from configs.settings import get_settings
from loguru import logger
import requests

# Set up multiprocess metrics directory
PROMETHEUS_DIR = "/tmp/prometheus_multiproc_dir"
os.environ["PROMETHEUS_MULTIPROC_DIR"] = PROMETHEUS_DIR
Path(PROMETHEUS_DIR).mkdir(parents=True, exist_ok=True)

# Clear existing metrics files
for f in Path(PROMETHEUS_DIR).glob("*.db"):
    f.unlink()

# Create registry for multiprocess mode
registry = CollectorRegistry()

class VideoEnhancerMetrics(ls.Logger):
    """Prometheus metrics for video enhancement API."""
    
    def __init__(self):
        """Initialize metrics."""
        super().__init__()
        prefix = "video_enhancer"
        
        self.processing_time = Histogram(
            f"{prefix}_processing_seconds",
            "Time spent processing video",
            ["operation"],
            registry=registry
        )
        
        self.video_size = Histogram(
            f"{prefix}_video_size_bytes",
            "Size of video in bytes",
            ["operation"],
            registry=registry
        )
        
        self.ssim_score = Histogram(
            f"{prefix}_ssim_score",
            "SSIM quality score",
            ["operation"],
            registry=registry
        )
    
    def process(self, key: str, value: float):
        """Process metrics from API."""
        logger.debug("Processing metric: {} = {}", key, value)
        
        if key == "inference_time":
            self.processing_time.labels(operation="enhance").observe(value)
        elif key == "video_size":
            self.video_size.labels(operation="enhance").observe(value)
        elif key == "ssim_score":
            self.ssim_score.labels(operation="enhance").observe(value)

class VideoEnhancerAPI(ls.LitAPI):
    """Video enhancement API with RabbitMQ task queue integration."""
    
    def __init__(self):
        super().__init__()
        
    def setup(self, device: str):
        """Set up API dependencies."""
        self.settings = get_settings()
        self.upscaler = VideoUpscaler(self.settings.realesrgan)
        self.s3 = S3Handler(self.settings.s3)
        self.rabbitmq = RabbitMQHandler(self.settings.rabbitmq)
        logger.info(f"Initialized API with device: {device}")
        
    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the incoming request."""
        try:
            # Handle raw request directly
            return request
        except Exception as e:
            logger.error(f"Error decoding request: {e}")
            raise ValueError(f"Failed to decode request: {e}")

    def predict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the video enhancement request or check task status."""
        try:
            # Get request data
            request_data = data[0]
            
            # Check if this is a task status request
            if 'task_id' in request_data and 'video' not in request_data:
                task_id = request_data['task_id']
                return [self.get_task_status(task_id)]
            
            # Otherwise, handle video enhancement request
            video_b64 = request_data.get("video", "")
            calculate_ssim = request_data.get("calculate_ssim", False)
            webhook_url = request_data.get("webhook_url")
            
            if not video_b64:
                raise ValueError("No video data provided")
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Create task data
            task_data = {
                "task_id": task_id,
                "video_data": video_b64,
                "calculate_ssim": calculate_ssim,
                "webhook_url": webhook_url
            }
            
            # Submit task to RabbitMQ
            self.rabbitmq.publish_task(task_data)
            
            # Return task ID for status checking
            return [{
                "status": "pending",
                "task_id": task_id,
                "message": "Task submitted successfully"
            }]
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            return [{
                "status": "error",
                "message": str(e)
            }]

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of task from RabbitMQ."""
        try:
            result = self.rabbitmq.get_result(task_id)
            if result:
                return result
            else:
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "Task is queued for processing"
                }
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {
                "task_id": task_id,
                "status": "error",
                "message": str(e)
            }

    def task_status(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle task status request."""
        try:
            request_data = data[0]
            task_id = request_data.get("task_id")
            if not task_id:
                raise ValueError("No task ID provided")
                
            status = self.get_task_status(task_id)
            return [status]
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return [{
                "status": "error",
                "message": str(e)
            }]

    def process_video(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a video enhancement task."""
        try:
            # Extract task info
            task_id = task_data['task_id']
            video_b64 = task_data['video_data']
            calculate_ssim = task_data.get('calculate_ssim', False)
            
            # Decode base64 video
            video_data = base64.b64decode(video_b64)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_in:
                temp_in.write(video_data)
                input_path = temp_in.name
            
            # Create temp output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_out:
                output_path = temp_out.name
            
            # Process video
            start_time = time.time()
            output_path = self.upscaler.process_video(input_path)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            # Calculate SSIM if requested
            ssim_score = None
            if calculate_ssim:
                ssim_score = self.upscaler.calculate_st_ssim(Path(input_path), Path(output_path))
            
            # Upload to S3
            s3_key = f"enhanced/{task_id}.mp4"
            self.s3.upload_file(output_path, s3_key)
            output_url = self.s3.get_url(s3_key)
            
            # Cleanup temp files
            os.unlink(input_path)
            os.unlink(output_path)
            
            # Prepare metrics
            metrics = {
                "input_size_mb": round(input_size, 2),
                "output_size_mb": round(output_size, 2),
                "processing_time_sec": round(processing_time, 2)
            }
            if ssim_score is not None:
                metrics["ssim_score"] = round(ssim_score, 4)
            
            return {
                "status": "success",
                "task_id": task_id,
                "output_url": output_url,
                "metrics": metrics,
                "message": "Video enhancement completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                "status": "error",
                "task_id": task_data.get('task_id'),
                "message": str(e)
            }

    def run_consumer(self):
        """Run the consumer to process tasks from RabbitMQ."""
        logger.info("Starting consumer...")
        
        def callback(ch, method, properties, body):
            try:
                # Parse task data
                task_data = json.loads(body)
                logger.info(f"Processing task {task_data.get('task_id')}")
                
                # Process video
                result = self.process_video(task_data)
                
                # Store result
                self.rabbitmq.update_result(result['task_id'], result)
                
                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except Exception as e:
                logger.error(f"Error in consumer callback: {e}")
                # Acknowledge message even on error to avoid infinite retries
                ch.basic_ack(delivery_tag=method.delivery_tag)
        
        # Start consuming
        try:
            self.rabbitmq.channel.basic_qos(prefetch_count=1)
            self.rabbitmq.channel.basic_consume(
                queue=self.rabbitmq.task_queue,
                on_message_callback=callback
            )
            logger.info("Started consuming from queue")
            self.rabbitmq.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.rabbitmq.channel.stop_consuming()
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise

    def submit_task(self, task_id: str, video_file: UploadFile):
        """Submit a video enhancement task to the RabbitMQ queue."""
        try:
            task_data = {
                "task_id": task_id,
                "video_base64": base64.b64encode(video_file.file.read()).decode('utf-8')
            }
            self.rabbitmq.publish_task(task_data)
            
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            raise Exception(
                status_code=500,
                detail=f"Failed to submit task: {str(e)}"
            )
            
class TaskStatusCallback(ls.Callback):
    """Callback to notify webhook endpoints about task status changes."""
    
    def __init__(self):
        super().__init__()
        self.tasks = {}  # Store task info
    
    def _notify_webhook(self, task_id: str, status: Dict[str, Any]):
        """Send notification to webhook if configured."""
        task_info = self.tasks.get(task_id, {})
        webhook_url = task_info.get('webhook_url')
        
        if webhook_url:
            try:
                requests.post(webhook_url, json=status)
                logger.info(f"Notified webhook for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to notify webhook for task {task_id}: {e}")
    
    def on_after_decode_request(self, lit_api, *args, **kwargs) -> None:
        """Store task info and notify webhook about task received."""
        if not kwargs.get('request'):
            return
            
        request = kwargs['request']
        if not isinstance(request, list):
            return
            
        request_data = request[0]
        task_id = request_data.get('task_id')
        webhook_url = request_data.get('webhook_url')
        
        if task_id and webhook_url and 'video' in request_data:  # Only track video enhancement tasks
            self.tasks[task_id] = {
                'webhook_url': webhook_url,
                'start_time': time.time()
            }
            
            # Notify webhook about task received
            status = {
                'task_id': task_id,
                'status': 'received',
                'timestamp': time.time(),
                'message': 'Task received and queued for processing'
            }
            self._notify_webhook(task_id, status)
    
    def on_after_predict(self, lit_api, *args, **kwargs) -> None:
        """Notify webhook about task processing started."""
        if not kwargs.get('result'):
            return
            
        result = kwargs['result']
        if not isinstance(result, list):
            return
            
        result_data = result[0]
        task_id = result_data.get('task_id')
        if task_id in self.tasks:
            status = {
                'task_id': task_id,
                'status': 'processing',
                'timestamp': time.time(),
                'message': 'Task processing started'
            }
            self._notify_webhook(task_id, status)
    
    def on_after_encode_response(self, lit_api, *args, **kwargs) -> None:
        """Notify webhook about task completion."""
        if not kwargs.get('response'):
            return
            
        response = kwargs['response']
        if not isinstance(response, list):
            return
            
        response_data = response[0]
        task_id = response_data.get('task_id')
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            processing_time = time.time() - task_info['start_time']
            
            status = {
                'task_id': task_id,
                'status': response_data.get('status', 'unknown'),
                'timestamp': time.time(),
                'processing_time': processing_time,
                'message': response_data.get('message', 'Task completed'),
                'result': response_data
            }
            self._notify_webhook(task_id, status)
            
            # Cleanup task info
            del self.tasks[task_id]

def main():
    """Main entry point for the Video Enhancer API."""
    import argparse
    import threading
    
    settings = get_settings()
    
    # Initialize API
    api = VideoEnhancerAPI()
    
    # Initialize metrics logger and callbacks
    metrics_logger = VideoEnhancerMetrics()
    metrics_logger.mount(path="/metrics", app=make_asgi_app(registry=registry))
    
    task_status_callback = TaskStatusCallback()
    
    # Start consumer thread
    def run_consumer():
        consumer_api = VideoEnhancerAPI()
        consumer_api.setup("cpu")
        try:
            consumer_api.run_consumer()
        except KeyboardInterrupt:
            logger.info("Shutting down consumer thread...")
        except Exception as e:
            logger.error(f"Consumer thread error: {e}")
    
    consumer_thread = threading.Thread(target=run_consumer, daemon=True)
    consumer_thread.start()
    logger.info("Started consumer thread")
    
    # Create LitServer instance with metrics logger and callbacks
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices='auto',
        workers_per_device=settings.api.workers,
        max_batch_size=16,
        track_requests=True,
        loggers=[metrics_logger],
        callbacks=[task_status_callback]
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
        if hasattr(api, 'rabbitmq'):
            api.rabbitmq.close()

if __name__ == "__main__":
    main()