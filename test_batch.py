import asyncio
import aiohttp
import base64
import time
from pathlib import Path
import json
from typing import List
import argparse

async def encode_video(video_path: str) -> str:
    """Encode video file to base64."""
    with open(video_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

async def send_request(session: aiohttp.ClientSession, video_data: str, api_url: str) -> dict:
    """Send a single request to the video enhancement API."""
    payload = {
        "video_data": video_data,
        "calculate_ssim": True
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

async def batch_test(video_path: str, num_requests: int, api_url: str) -> List[dict]:
    """Send multiple requests concurrently to test batching."""
    # Encode video once and reuse
    video_data = await encode_video(video_path)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            task = send_request(session, video_data, api_url)
            tasks.append(task)
        
        print(f"Sending {num_requests} concurrent requests...")
        results = await asyncio.gather(*tasks)
        return results

def analyze_results(results: List[dict]):
    """Analyze and print test results."""
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["status"] == 200)
    failed_requests = total_requests - successful_requests
    
    durations = [r["duration"] for r in results]
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    print("\nTest Results:")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_requests}")
    print(f"\nTiming Statistics:")
    print(f"Average Duration: {avg_duration:.2f}s")
    print(f"Min Duration: {min_duration:.2f}s")
    print(f"Max Duration: {max_duration:.2f}s")
    
    # Analyze batch processing
    batch_sizes = set()
    for r in results:
        if r["status"] == 200:
            batch_sizes.add(r["response"].get("batch_size", 1))
    
    print(f"\nBatch Processing:")
    print(f"Observed Batch Sizes: {sorted(batch_sizes)}")
    
    # Sample response structure
    if results and results[0]["status"] == 200:
        print("\nSample Response Structure:")
        print(json.dumps(results[0]["response"], indent=2))

def main():
    parser = argparse.ArgumentParser(description='Test video enhancement API batching')
    parser.add_argument('--video', type=str, required=True, help='Path to test video file')
    parser.add_argument('--requests', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('--url', type=str, default='http://localhost:8000/predict', help='API endpoint URL')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    print(f"Starting batch test with {args.requests} concurrent requests...")
    print(f"API URL: {args.url}")
    print(f"Test Video: {args.video}")
    
    results = asyncio.run(batch_test(args.video, args.requests, args.url))
    analyze_results(results)

if __name__ == "__main__":
    main() 