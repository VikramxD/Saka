import asyncio
import aiohttp
import base64
import time
from pathlib import Path
import json
from typing import List, Dict
import argparse
from datetime import datetime

async def encode_video(video_path: str) -> str:
    """Encode video file to base64."""
    with open(video_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

async def send_request(session: aiohttp.ClientSession, video_data: str, api_url: str, calculate_ssim: bool = False) -> dict:
    """Send a single request to the video enhancement API."""
    payload = {
        "video_data": video_data,
        "calculate_ssim": calculate_ssim
    }
    
    start_time = time.time()
    async with session.post(api_url, json=payload) as response:
        result = await response.json()
        duration = time.time() - start_time
        return {
            "status": response.status,
            "duration": duration,
            "response": result
        }

async def process_video(session: aiohttp.ClientSession, video_path: str, api_url: str, calculate_ssim: bool = False) -> dict:
    """Process a single video and return results."""
    print(f"Processing video: {Path(video_path).name}")
    video_data = await encode_video(video_path)
    return await send_request(session, video_data, api_url, calculate_ssim)

async def process_videos(video_paths: List[str], api_url: str, calculate_ssim: bool = False) -> List[dict]:
    """Process multiple videos concurrently."""
    print(f"Processing {len(video_paths)} videos...")
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_video(session, video_path, api_url, calculate_ssim)
            for video_path in video_paths
        ]
        return await asyncio.gather(*tasks)

def analyze_results(results: List[dict], video_paths: List[str]):
    """Analyze and print test results."""
    total_time = sum(result["duration"] for result in results)
    successful = sum(1 for result in results if result["status"] == 200)
    failed = len(results) - successful

    print("\nProcessing Results:")
    print(f"Total Processing Time: {total_time:.2f}s")
    print(f"Total Videos: {len(video_paths)}")
    print(f"Successfully Processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Average Time per Video: {total_time/len(video_paths):.2f}s")
    
    # Print individual video results
    print("\nIndividual Video Results:")
    for video_path, result in zip(video_paths, results):
        print(f"\n=== {Path(video_path).name} ===")
        if result["status"] == 200:
            response = result["response"]
            if response["status"] == "success":
                print(f"Status: Success")
                print(f"Output URL: {response['output_url']}")
                print(f"Metrics:")
                metrics = response["metrics"]
                print(f"  Processing Time: {metrics['processing_time']:.2f}s")
                print(f"  RAM Usage: {metrics['ram_usage_mb']:.2f} MB")
                print(f"  Input Resolution: {metrics['input_resolution']}")
                print(f"  Output Resolution: {metrics['output_resolution']}")
                if metrics.get("ssim_score") is not None:
                    print(f"  SSIM Score: {metrics['ssim_score']:.3f}")
                print(f"Model Settings: {response['model_settings']}")
            else:
                print(f"Status: Failed")
                print(f"Error: {response['error']}")
        else:
            print(f"Status: HTTP Error {result['status']}")
            print(f"Response: {result['response']}")

    # Print and save raw JSON response
    print("\nRaw JSON Response:")
    formatted_json = json.dumps(results, indent=2)
    print(formatted_json)
    
    # Save response to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        f.write(formatted_json)
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Test video enhancement API')
    parser.add_argument('--videos_dir', type=str, required=True, help='Directory containing test video files')
    parser.add_argument('--url', type=str, default='http://localhost:8000/predict', help='API endpoint URL')
    parser.add_argument('--concurrent', type=int, default=4, help='Number of concurrent requests')
    parser.add_argument('--calculate_ssim', action='store_true', help='Enable SSIM calculation (computationally expensive)')
    
    args = parser.parse_args()
    
    videos_dir = Path(args.videos_dir)
    if not videos_dir.exists() or not videos_dir.is_dir():
        print(f"Error: Videos directory not found: {args.videos_dir}")
        return
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(str(p) for p in videos_dir.glob(f'*{ext}'))
    
    if not video_paths:
        print(f"Error: No video files found in directory: {args.videos_dir}")
        return
    
    print(f"Starting video processing...")
    print(f"API URL: {args.url}")
    print(f"Found {len(video_paths)} videos in: {args.videos_dir}")
    print(f"SSIM calculation: {'enabled' if args.calculate_ssim else 'disabled'}")
    for video in video_paths:
        print(f"  - {Path(video).name}")
    
    # Process videos with concurrency limit
    semaphore = asyncio.Semaphore(args.concurrent)
    
    async def bounded_process():
        async with semaphore:
            return await process_videos(video_paths, args.url, args.calculate_ssim)
    
    results = asyncio.run(bounded_process())
    analyze_results(results, video_paths)

if __name__ == "__main__":
    main() 