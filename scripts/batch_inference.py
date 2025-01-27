#!/usr/bin/env python3
import os
import time
import base64
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger

class BatchInferenceTester:
    """Test harness for video enhancement batch inference."""
    
    def __init__(self, server_url: str, concurrent_requests: int = 4):
        """Initialize the batch inference tester.
        
        Args:
            server_url: URL of the video enhancement server
            concurrent_requests: Number of concurrent requests to make
        """
        self.server_url = server_url.rstrip('/')
        self.concurrent_requests = concurrent_requests
        self.results = []
        
    async def _send_single_request(self, video_path: Path, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Send a single video enhancement request."""
        try:
            # Read and encode video
            with open(video_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "video_data": video_data,
                "calculate_ssim": True
            }
            
            # Send request and measure time
            start_time = time.time()
            async with session.post(f"{self.server_url}/predict", json=payload) as response:
                response_data = await response.json()
                end_time = time.time()
                
            # Add timing information
            result = {
                "video_path": str(video_path),
                "video_size": video_path.stat().st_size,
                "status": response_data.get("status", "unknown"),
                "inference_time": response_data.get("inference_time", 0),
                "total_time": end_time - start_time,
                "ssim_score": response_data.get("ssim_score", None),
                "output_size": response_data.get("video_size", 0),
                "output_url": response_data.get("output_url", None)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            return {
                "video_path": str(video_path),
                "status": "error",
                "error": str(e)
            }

    async def process_videos(self, video_dir: Path) -> List[Dict[str, Any]]:
        """Process all videos in the directory with batching."""
        video_files = list(video_dir.glob("*.mp4"))
        if not video_files:
            raise ValueError(f"No .mp4 files found in {video_dir}")
            
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Create client session for connection pooling
        async with aiohttp.ClientSession() as session:
            # Create tasks for all videos
            tasks = [
                self._send_single_request(video_path, session)
                for video_path in video_files
            ]
            
            # Process in batches using semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            async def bounded_request(task):
                async with semaphore:
                    return await task
                    
            # Run all tasks with progress bar
            results = []
            with tqdm(total=len(tasks), desc="Processing videos") as pbar:
                for task in asyncio.as_completed([bounded_request(task) for task in tasks]):
                    result = await task
                    results.append(result)
                    pbar.update(1)
                    
        return results

    def analyze_results(self, save_dir: Path):
        """Analyze and visualize the batch processing results."""
        if not self.results:
            logger.warning("No results to analyze")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Create output directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        df.to_csv(save_dir / "batch_results.csv", index=False)
        
        # Basic statistics
        stats = {
            "total_videos": len(df),
            "successful": len(df[df.status == "complete"]),
            "failed": len(df[df.status != "complete"]),
            "avg_inference_time": df.inference_time.mean(),
            "avg_total_time": df.total_time.mean(),
            "avg_ssim": df.ssim_score.mean()
        }
        
        with open(save_dir / "statistics.txt", "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        # Visualizations
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Processing times
        plt.subplot(2, 2, 1)
        df.boxplot(column=['inference_time', 'total_time'])
        plt.title('Processing Times Distribution')
        plt.ylabel('Seconds')
        
        # Plot 2: SSIM scores
        plt.subplot(2, 2, 2)
        df.ssim_score.hist()
        plt.title('SSIM Score Distribution')
        
        # Plot 3: Video sizes
        plt.subplot(2, 2, 3)
        plt.scatter(df.video_size, df.output_size)
        plt.xlabel('Input Size (bytes)')
        plt.ylabel('Output Size (bytes)')
        plt.title('Input vs Output Size')
        
        # Save plots
        plt.tight_layout()
        plt.savefig(save_dir / "batch_analysis.png")
        
        logger.info(f"Analysis results saved to {save_dir}")

async def main():
    parser = argparse.ArgumentParser(description="Batch inference tester for video enhancement")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing input videos")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="Video enhancement server URL")
    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("--output-dir", type=Path, default=Path("batch_results"), help="Directory to save results")
    
    args = parser.parse_args()
    tester = BatchInferenceTester(args.server_url, args.concurrent)
    
    try:
        # Run batch processing
        logger.info(f"Starting batch inference with {args.concurrent} concurrent requests")
        start_time = time.time()
        
        results = await tester.process_videos(args.video_dir)
        tester.results = results
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed in {total_time:.2f} seconds")
        
        # Analyze results
        tester.analyze_results(args.output_dir)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
