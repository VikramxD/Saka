import os
import base64
from pathlib import Path
from typing import Dict, Any
import httpx
from loguru import logger

class VideoEnhancerClient:
    """Client for interacting with the Video Enhancer API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 3600
    ):
        """Initialize the client."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def enhance_video(
        self,
        video_path: str,
        calculate_ssim: bool = False
    ) -> Dict[str, Any]:
        """Enhance a video using the API.
        
        Args:
            video_path: Path to the video file
            calculate_ssim: Whether to calculate SSIM metric
            
        Returns:
            JSON response with output_url and metrics
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            httpx.HTTPError: If API request fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Read video file and encode as base64
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        # Make request with JSON body
        json_data = {
            "video_base64": video_base64,
            "calculate_ssim": calculate_ssim
        }

        response = self.client.post(
            f"{self.base_url}/predict",
            json=json_data
        )
        response.raise_for_status()
        return response.json()

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
