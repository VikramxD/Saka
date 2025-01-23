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
import json

from models import VideoResponse, ProcessingMetrics
from scripts.realesrgan import VideoUpscaler
from api.storage import S3Handler
from api.rabbitmq_handler import RabbitMQHandler
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
    """Video enhancement API with RabbitMQ task queue integration.
    
    This API provides endpoints for video enhancement using Real-ESRGAN,
    with asynchronous processing via RabbitMQ task queue. It supports:
    - Asynchronous video processing
    - Task status tracking
    - Metrics collection
    - S3 storage integration
    
    The API can run in two modes:
    1. API mode: Handles HTTP requests and submits tasks to RabbitMQ
    2. Worker mode: Processes tasks from RabbitMQ queue
    """
    
    def setup(self, device: str):
        """Initialize API components during startup.
        
        Args:
            device (str): Device to use for processing (e.g., 'cuda:0')
            
        Initializes:
            - Video upscaler model
            - S3 storage handler
            - RabbitMQ connection
            - Metrics collection
            
        Raises:
            Exception: If initialization of any component fails
        """
        try:
            settings = get_settings()
            
            # Initialize RabbitMQ
            self.rabbitmq = RabbitMQHandler(settings.rabbitmq)
            logger.info("Initialized RabbitMQ connection")
            
            # Initialize upscaler if needed
            if device:
                self.upscaler = VideoUpscaler(settings.realesrgan)
                logger.info(f"Initialized video upscaler on {device}")
            else:
                self.upscaler = None
                
            # Initialize S3 storage
            self.s3 = S3Handler(settings.s3)
            logger.info("Initialized S3 storage")
            
        except Exception as e:
            logger.error(f"Failed to initialize API: {e}")
            raise
            
    def preprocess_request(self, request_data):
        """Preprocess request before predict.
        
        Args:
            request_data: Raw request data
            
        Returns:
            Preprocessed request data
        """
        logger.debug(f"Preprocessing request type: {type(request_data)}")
        
        # Extract request from LitServer format
        if isinstance(request_data, dict):
            if "inputs" in request_data:
                # LitServer format with inputs array
                if not request_data["inputs"]:
                    return {
                        "status": "ERROR",
                        "error": "Empty inputs array"
                    }
                request = request_data["inputs"][0]
            elif "body" in request_data:
                # LitServer format with body
                request = request_data["body"]
                if isinstance(request, str):
                    try:
                        request = json.loads(request)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON body: {e}")
                        return {
                            "status": "ERROR",
                            "error": "Invalid JSON in request body"
                        }
            else:
                request = request_data
        else:
            request = request_data
            
        logger.debug(f"Preprocessed request type: {type(request)}")
        logger.debug(f"Preprocessed request keys: {request.keys() if isinstance(request, dict) else 'not a dict'}")
        return request
            
    def predict(self, request_data):
        """Process video enhancement request.
        
        Args:
            request_data: Request data from LitServer
            
        Returns:
            List containing a single response dictionary
        """
        logger.debug(f"Raw predict request type: {type(request_data)}")
        logger.debug(f"Raw predict request keys: {request_data.keys() if isinstance(request_data, dict) else 'not a dict'}")
        
        try:
            # Format request for LitServer
            if isinstance(request_data, dict):
                if "inputs" not in request_data and "body" not in request_data:
                    request_data = {"inputs": [request_data]}
            else:
                request_data = {"inputs": [request_data]}
            
            # Preprocess request
            request = self.preprocess_request(request_data)
            if isinstance(request, dict) and "status" in request and request["status"] == "ERROR":
                return [request]
                
            # Validate request is a dict
            if not isinstance(request, dict):
                return [{
                    "status": "ERROR",
                    "error": f"Request must be a dictionary, got {type(request)}"
                }]
            
            # Handle both video upload and status check
            if "task_id" in request:
                # Get task status
                task_id = request["task_id"]
                if not task_id:
                    return [{
                        "status": "ERROR",
                        "error": "task_id cannot be empty"
                    }]
                    
                # Get task result
                result = self.rabbitmq.get_result(task_id)
                if result is None:
                    return [{
                        "task_id": task_id,
                        "status": "PENDING"
                    }]
                return [result]
                
            # Process new video request
            if "video_base64" not in request:
                logger.error(f"Missing video_base64 in request keys: {request.keys()}")
                return [{
                    "status": "ERROR",
                    "error": "video_base64 is required for video upload"
                }]
                
            # Create task and submit to RabbitMQ
            task_id = str(uuid.uuid4())
            task_data = {
                "task_id": task_id,
                **request
            }
            logger.debug(f"Publishing task: {task_data.keys()}")
            self.rabbitmq.publish_task(task_data)
            
            response = [{
                "task_id": task_id,
                "status": "PENDING"
            }]
            logger.debug(f"Predict response: {response}")
            return response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing request: {error_msg}")
            logger.exception(e)  # Log full traceback
            response = [{
                "status": "ERROR",
                "error": error_msg
            }]
            logger.debug(f"Error response: {response}")
            return response

    def encode_response(self, output):
        """Encode the response.
        
        Args:
            output: List containing a single response dictionary
            
        Returns:
            List containing a single encoded response dictionary
        """
        return output

    def get_task_status(self, request):
        """Get the status of a video enhancement task.
        
        Args:
            request: Request data from LitServer
            
        Returns:
            List containing a single task status response
        """
        return self.predict(request)

    def process_video_task(self, task_data: Dict[str, Any]):
        """Process a video task from the RabbitMQ queue.
        
        Args:
            task_data: Dictionary containing task_id and video_base64
        """
        temp_file = None
        try:
            task_id = task_data["task_id"]
            video_base64 = task_data["video_base64"]
            calculate_ssim = task_data.get("calculate_ssim", False)
            
            logger.info(f"Processing task {task_id}")
            
            # Decode base64 and save to temp file
            video_bytes = base64.b64decode(video_base64)
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.write(video_bytes)
            temp_file.close()
            
            logger.info(f"Saved video to temporary file: {temp_file.name}")
            
            # Process video
            try:
                # TODO: Add actual video processing here
                time.sleep(2)  # Simulate processing
                
                # Update task result
                result = {
                    "task_id": task_id,
                    "status": "COMPLETED",
                    "output_url": "https://example.com/output.mp4"
                }
                if calculate_ssim:
                    result["metrics"] = {"ssim": 0.95}
                    
                self.rabbitmq.update_result(task_id, result)
                logger.info(f"Task {task_id} completed successfully")
                
            except Exception as e:
                error_msg = f"Failed to process video: {str(e)}"
                logger.error(error_msg)
                self.rabbitmq.update_result(task_id, {
                    "task_id": task_id,
                    "status": "ERROR",
                    "error": error_msg
                })
                
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            if task_id:
                self.rabbitmq.update_result(task_id, {
                    "task_id": task_id,
                    "status": "ERROR",
                    "error": f"Task processing failed: {str(e)}"
                })
        finally:
            # Cleanup temp file
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                    logger.info(f"Cleaned up temporary file: {temp_file.name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup temporary file: {str(e)}")

    def run_consumer(self):
        """Run in consumer mode to process tasks from RabbitMQ queue.
        
        This method:
        1. Sets up a connection to RabbitMQ
        2. Starts consuming tasks from the queue
        3. Processes each task using process_video_task
        4. Handles cleanup on shutdown
        """
        try:
            logger.info("Starting Video Enhancer in consumer mode...")
            self.rabbitmq.consume_tasks(self.process_video_task)
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
            self.rabbitmq.close()
        except Exception as e:
            logger.error(f"Consumer failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    
    settings = get_settings()
    metrics_logger = VideoEnhancerMetrics(settings.prometheus)
    
    def run_api_server():
        """Run the API server for handling HTTP requests."""
        api = VideoEnhancerAPI()
        server = ls.LitServer(
            api,
            accelerator="auto",
            devices='auto',
            workers_per_device=settings.api.workers,
            max_batch_size=16,
            loggers=[metrics_logger],  # Pass logger as a list
            track_requests=True
        )
        server.run(
            host=settings.api.host,
            port=settings.api.port
        )
    
    def run_consumer():
        """Run the consumer for processing tasks."""
        api = VideoEnhancerAPI()
        api.setup("cuda:0")
        api.run_consumer()
    
    # Start processes based on mode
    mode = os.environ.get("MODE", "all").lower()
    processes = []
    
    try:
        if mode in ["all", "api"]:
            api_process = multiprocessing.Process(target=run_api_server)
            api_process.start()
            processes.append(api_process)
            logger.info("Started API server process")
            
        if mode in ["all", "consumer"]:
            consumer_process = multiprocessing.Process(target=run_consumer)
            consumer_process.start()
            processes.append(consumer_process)
            logger.info("Started consumer process")
            
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        for process in processes:
            process.terminate()
            process.join()