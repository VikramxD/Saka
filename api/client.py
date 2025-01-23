import os
import base64
from pathlib import Path
from typing import Dict, Any
import httpx
import time
from loguru import logger

class VideoEnhancerClient:
    """Client for interacting with the Video Enhancer API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 3600,
        poll_interval: float = 1.0
    ):
        """Initialize the client."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client = httpx.Client(timeout=timeout)

    def enhance_video(
        self,
        video_path: str,
        calculate_ssim: bool = False,
        wait_for_result: bool = True
    ) -> Dict[str, Any]:
        """Enhance a video using the server.
        
        Args:
            video_path: Path to video file
            calculate_ssim: Whether to calculate SSIM
            wait_for_result: Whether to wait for processing to complete
            
        Returns:
            Task info with status and result
        """
        # Read and encode video
        try:
            logger.debug(f"Reading video file: {video_path}")
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            logger.debug(f"Successfully encoded video ({len(video_bytes)} bytes)")
        except Exception as e:
            raise RuntimeError(f"Failed to read video file: {e}")

        # Make request with JSON body
        request = {
            "video_base64": video_base64,
            "calculate_ssim": calculate_ssim
        }
        logger.debug(f"Sending request: {request.keys()}")
        
        # Submit task
        try:
            url = f"{self.base_url}/predict"
            logger.debug(f"POST {url}")
            
            # Send request directly without LitServer format
            response = self.client.post(
                url,
                json=request
            )
            response.raise_for_status()
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response body: {response.text}")
            
            task_info = response.json()
            logger.debug(f"Predict response: {task_info}")
            
            # Check for error response
            if isinstance(task_info, list) and len(task_info) > 0:
                task_info = task_info[0]
                logger.debug(f"Extracted task info: {task_info}")
                
            if task_info.get("status") == "ERROR":
                raise RuntimeError(task_info.get("error", "Unknown error"))
                
            if not wait_for_result:
                return task_info
                
            # Poll for task completion
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                # Get task status
                status_request = {
                    "task_id": task_info["task_id"]
                }
                logger.debug(f"Checking status: {status_request}")
                
                response = self.client.post(
                    url,
                    json=status_request
                )
                response.raise_for_status()
                
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                    
                logger.debug(f"Status response: {result}")
                
                if result.get("status") == "ERROR":
                    raise RuntimeError(result.get("error", "Unknown error"))
                    
                if result.get("status") != "PENDING":
                    return result
                    
                time.sleep(self.poll_interval)
                
            raise TimeoutError("Task timed out")
            
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP request failed: {e}")

def main():
    """Example usage of the client."""
    import argparse
    parser = argparse.ArgumentParser(description='Enhance a video using the Video Enhancer API')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--calculate-ssim', action='store_true', help='Calculate SSIM metric')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    args = parser.parse_args()

    # Create client and enhance video
    client = VideoEnhancerClient(base_url=args.api_url)
    try:
        print(f"Enhancing video: {args.video_path}")
        response = client.enhance_video(args.video_path, args.calculate_ssim)
        print("\nAPI Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main()
