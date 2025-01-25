import os
import base64
import requests
import time
from pathlib import Path
import concurrent.futures
from typing import List

def encode_video(video_path: Path) -> str:
    """Encode video file to base64."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def enhance_video(video_path: Path, api_url: str = "http://localhost:8000/predict") -> dict:
    """Enhance a single video using the API."""
    try:
        # Encode video
        video_data = encode_video(video_path)
        
        # Prepare request
        payload = {
            "video_data": video_data,
            "calculate_ssim": True
        }
        
        # Send request
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        return {
            "video": video_path.name,
            "status": "success",
            "result": response.json()
        }
    except Exception as e:
        return {
            "video": video_path.name,
            "status": "error",
            "error": str(e)
        }

def batch_enhance(video_dir: Path, max_workers: int = 3) -> List[dict]:
    """Enhance multiple videos in parallel."""
    # Get all video files
    video_files = [
        f for f in video_dir.glob("*.mp4")
        if f.is_file()
    ]
    
    if not video_files:
        raise ValueError(f"No video files found in {video_dir}")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(enhance_video, video_path): video_path
            for video_path in video_files
        }
        
        # Get results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            result = future.result()
            results.append(result)
            
            # Print progress
            status = result["status"]
            video_name = result["video"]
            if status == "success":
                print(f" Enhanced {video_name}")
            else:
                print(f" Failed to enhance {video_name}: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    # Directory containing videos to enhance
    video_dir = Path("../scripts")  # Update this path to where your videos are
    
    print(f"Starting batch enhancement of videos in {video_dir}")
    start_time = time.time()
    
    try:
        results = batch_enhance(video_dir)
        
        # Print summary
        success_count = sum(1 for r in results if r["status"] == "success")
        total_count = len(results)
        
        print("\nBatch Enhancement Summary:")
        print(f"Total videos processed: {total_count}")
        print(f"Successfully enhanced: {success_count}")
        print(f"Failed: {total_count - success_count}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during batch enhancement: {e}")
