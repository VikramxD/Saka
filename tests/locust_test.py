import os
import base64
import time
import random
from locust import HttpUser, task, between, events, constant_throughput
from pathlib import Path

class VideoEnhancerUser(HttpUser):
    """Simulated user for load testing the video enhancement API."""
    
    # Use constant throughput to ensure consistent load
    wait_time = constant_throughput(1)  # 1 request per second per user
    
    def on_start(self):
        """Load test videos on startup."""
        self.videos = {}
        video_dir = Path(__file__).parent.parent / "videos"
        
        # Load all videos from the videos directory
        for video_file in ["one_piece.mp4", "hxh.mp4", "dbz.mp4"]:
            video_path = video_dir / video_file
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found at {video_path}")
            
            with open(video_path, "rb") as f:
                self.videos[video_file] = base64.b64encode(f.read()).decode()
        
        if not self.videos:
            raise RuntimeError("No videos found for testing")
    
    @task(3)
    def enhance_video_concurrent(self):
        """Task to test video enhancement endpoint with concurrent requests."""
        # Randomly select multiple videos for concurrent processing
        selected_videos = random.sample(list(self.videos.keys()), 
                                     k=min(2, len(self.videos)))  # Process 2 videos concurrently
        
        for video_name in selected_videos:
            video_data = self.videos[video_name]
            payload = {
                "video_data": video_data,
                "calculate_ssim": False
            }
            
            start_time = time.time()
            with self.client.post(
                "/predict", 
                json=payload, 
                catch_response=True, 
                name=f"enhance_video_{video_name}"
            ) as response:
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    if result["status"] == "success":
                        response.success()
                        events.request.fire(
                            request_type="Video Enhancement",
                            name=f"Processing Time ({video_name})",
                            response_time=duration * 1000,
                            response_length=len(video_data),
                            context=result.get("metrics", {})
                        )
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        response.failure(f"API returned error for {video_name}: {error_msg}")
                else:
                    response.failure(f"Request failed for {video_name} with status code: {response.status_code}")
    
    @task(5)
    def check_metrics(self):
        """Task to monitor Prometheus metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics endpoint failed with status code: {response.status_code}")

def on_test_start(environment, **kwargs):
    """Called when the test is starting."""
    print("Starting video enhancement load test...")
    print("Available endpoints:")
    print("  - POST /predict : Video enhancement endpoint (concurrent)")
    print("  - GET /metrics : Prometheus metrics endpoint")
    print("\nConcurrency settings:")
    print("  - Each user processes 2 videos concurrently")
    print("  - Constant throughput of 1 request per second per user")
    print("  - Recommended to start with 5-10 users for high concurrency")

def on_test_stop(environment, **kwargs):
    """Called when the test is ending."""
    print("\nTest completed!")
    print("Check Grafana dashboard for detailed metrics visualization")

# Register test lifecycle hooks
events.test_start.add_listener(on_test_start)
events.test_stop.add_listener(on_test_stop)

if __name__ == "__main__":
    """
    Run with: 
    locust -f tests/locust_test.py --host http://localhost:8000 --users 10 --spawn-rate 2
    
    This will:
    - Start 10 concurrent users
    - Spawn 2 new users per second
    - Each user will process 2 videos concurrently
    - Total concurrent requests = up to 20 (10 users × 2 videos)
    """
    pass 