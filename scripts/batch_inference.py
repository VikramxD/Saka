#!/usr/bin/env python3
"""
Batch Video Enhancement Testing Script

This script provides functionality for testing the video enhancement API with batch processing.
It supports:
- Parallel request processing with multiple workers
- Detailed progress tracking for each video
- Performance metrics collection
- Result analysis and visualization
- Automatic retries
"""

import os
import time
import base64
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn
)
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import numpy as np
import json

async def process_single_video(video_path: Path, session: aiohttp.ClientSession, server_url: str, max_retries: int, progress) -> Dict[str, Any]:
    """Process a single video with retries and progress tracking."""
    video_name = video_path.name
    task_id = progress.add_task(
        f"[cyan]Processing[/cyan] {video_name}",
        total=100,
        start=False
    )
    
    for attempt in range(max_retries):
        try:
            # Update progress - Starting
            progress.update(task_id, completed=5, description=f"[cyan]Reading[/cyan] {video_name}")
            
            # Read and encode video
            with open(video_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Update progress - Video loaded
            progress.update(task_id, completed=10, description=f"[cyan]Uploading[/cyan] {video_name}")
            
            # Prepare request payload
            payload = {
                "video_data": video_data,
                "calculate_ssim": True
            }
            
            # Send request and measure time
            start_time = time.time()
            async with session.post(f"{server_url}/predict", json=payload) as response:
                # Update progress - Processing started
                progress.update(task_id, completed=30, description=f"[cyan]Server processing[/cyan] {video_name}")
                
                response_data = await response.json()
                end_time = time.time()
                
                # Process response
                if response_data.get("status") == "success":
                    # Update progress - Success
                    progress.update(
                        task_id,
                        completed=100,
                        description=f"[green]Completed[/green] {video_name}"
                    )
                    
                    metrics = response_data.get("metrics", {})
                    return {
                        "video_path": str(video_path),
                        "video_size": {
                            "bytes": video_path.stat().st_size,
                            "megabytes": round(video_path.stat().st_size / 1024 / 1024, 2)
                        },
                        "status": "success",
                        "timing": {
                            "processing_time_seconds": round(metrics.get("processing_time", 0), 2),
                            "total_time_seconds": round(end_time - start_time, 2)
                        },
                        "quality_metrics": {
                            "ssim_score": round(metrics.get("ssim_score", 0), 3),
                            "resolution": {
                                "input": metrics.get("input_resolution", {}),
                                "output": metrics.get("output_resolution", {}),
                                "scale_factor": metrics.get("model_settings", {}).get("scale_factor", 4)
                            }
                        },
                        "system_metrics": {
                            "ram_usage_mb": round(metrics.get("ram_usage_mb", 0), 2),
                            "gpu_memory_mb": round(metrics.get("gpu_memory_mb", 0), 2)
                        },
                        "model_info": metrics.get("model_settings", {}),
                        "output": {
                            "url": response_data.get("output_url"),
                            "filename": Path(response_data.get("output_url", "")).name
                        },
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    # Update progress - Retry
                    progress.update(
                        task_id,
                        description=f"[yellow]Retrying[/yellow] {video_name} ({attempt + 1}/{max_retries})"
                    )
                    if attempt == max_retries - 1:
                        progress.update(
                            task_id,
                            description=f"[red]Failed[/red] {video_name}",
                            completed=100
                        )
                        return {
                            "video_path": str(video_path),
                            "status": "error",
                            "error": response_data.get("error", "Unknown error")
                        }
                    await asyncio.sleep(2 ** attempt)
                
        except Exception as e:
            # Update progress - Error
            progress.update(
                task_id,
                description=f"[red]Error[/red] {video_name} ({attempt + 1}/{max_retries})"
            )
            logger.error(f"Error processing {video_path} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                progress.update(
                    task_id,
                    description=f"[red]Failed[/red] {video_name}",
                    completed=100
                )
                return {
                    "video_path": str(video_path),
                    "status": "error",
                    "error": str(e)
                }
            await asyncio.sleep(2 ** attempt)

class BatchInferenceTester:
    """Test harness for video enhancement batch inference."""
    
    def __init__(self, server_url: str, num_workers: int = None, max_retries: int = 3):
        """Initialize the batch inference tester.
        
        Args:
            server_url: URL of the video enhancement server
            num_workers: Number of worker processes (defaults to CPU count)
            max_retries: Maximum retry attempts for failed requests
        """
        self.server_url = server_url.rstrip('/')
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.max_retries = max_retries
        self.results = []
        
    async def process_videos(self, video_dir: Path) -> List[Dict[str, Any]]:
        """Process all videos in parallel using multiple workers."""
        # Collect video files
        video_files = []
        for ext in [".mp4", ".avi", ".mkv", ".mov"]:
            video_files.extend(video_dir.glob(f"*{ext}"))
            
        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")
            
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Process videos with detailed progress tracking
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as progress:
            # Create a semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.num_workers)
            
            # Create client session for connection pooling
            async with aiohttp.ClientSession() as session:
                # Create tasks for all videos
                async def process_with_semaphore(video_path):
                    async with semaphore:
                        return await process_single_video(
                            video_path,
                            session,
                            self.server_url,
                            self.max_retries,
                            progress
                        )
                
                # Process all videos
                tasks = [
                    process_with_semaphore(video_path)
                    for video_path in video_files
                ]
                
                # Wait for all tasks to complete
                results = await asyncio.gather(*tasks)
                
        return results

    def analyze_results(self, save_dir: Path):
        """Analyze and visualize the batch processing results."""
        if not self.results:
            logger.warning("No results to analyze")
            return
            
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Create output directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate statistics
        successful_df = df[df.status == "success"]
        stats = {
            "batch_summary": {
                "total_videos": len(df),
                "successful": len(successful_df),
                "failed": len(df[df.status != "success"]),
                "success_rate_percent": round(len(successful_df) / len(df) * 100, 1)
            },
            "timing_metrics": {
                "average_processing_time": round(successful_df.timing.apply(lambda x: x["processing_time_seconds"]).mean(), 2),
                "average_total_time": round(successful_df.timing.apply(lambda x: x["total_time_seconds"]).mean(), 2),
                "min_processing_time": round(successful_df.timing.apply(lambda x: x["processing_time_seconds"]).min(), 2),
                "max_processing_time": round(successful_df.timing.apply(lambda x: x["processing_time_seconds"]).max(), 2)
            },
            "quality_metrics": {
                "average_ssim": round(successful_df.quality_metrics.apply(lambda x: x["ssim_score"]).mean(), 3),
                "min_ssim": round(successful_df.quality_metrics.apply(lambda x: x["ssim_score"]).min(), 3),
                "max_ssim": round(successful_df.quality_metrics.apply(lambda x: x["ssim_score"]).max(), 3)
            },
            "system_metrics": {
                "average_ram_usage_mb": round(successful_df.system_metrics.apply(lambda x: x["ram_usage_mb"]).mean(), 2),
                "peak_ram_usage_mb": round(successful_df.system_metrics.apply(lambda x: x["ram_usage_mb"]).max(), 2),
                "average_gpu_memory_mb": round(successful_df.system_metrics.apply(lambda x: x.get("gpu_memory_mb", 0)).mean(), 2)
            },
            "batch_info": {
                "start_time": min(df.timestamp),
                "end_time": max(df.timestamp),
                "total_duration_seconds": round(time.time() - time.mktime(time.strptime(min(df.timestamp), "%Y-%m-%d %H:%M:%S")), 2)
            }
        }
        
        # Save detailed results as JSON
        output = {
            "batch_statistics": stats,
            "processed_videos": self.results
        }
        
        with open(save_dir / "batch_results.json", "w") as f:
            json.dump(output, f, indent=2)
            
        # Save summary statistics
        with open(save_dir / "statistics.txt", "w") as f:
            f.write("Batch Processing Summary\n")
            f.write("======================\n\n")
            
            for section, metrics in stats.items():
                f.write(f"{section.replace('_', ' ').title()}\n")
                f.write("-" * len(section) + "\n")
                for key, value in metrics.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
        
        # Create visualizations
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 10))
        
        # Set common style parameters
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12
        })
        
        # Plot 1: Processing times
        ax1 = fig.add_subplot(221)
        successful_df = df[df.status == "success"]
        successful_df[["processing_time", "total_time"]].boxplot(ax=ax1)
        ax1.set_title("Processing Times Distribution")
        ax1.set_ylabel("Seconds")
        
        # Plot 2: SSIM scores
        ax2 = fig.add_subplot(222)
        successful_df.ssim_score.hist(ax=ax2, bins=20)
        ax2.set_title("SSIM Score Distribution")
        ax2.set_xlabel("SSIM Score")
        
        # Plot 3: Video sizes vs Processing time
        ax3 = fig.add_subplot(223)
        ax3.scatter(successful_df.video_size / 1e6, successful_df.processing_time)
        ax3.set_xlabel("Video Size (MB)")
        ax3.set_ylabel("Processing Time (s)")
        ax3.set_title("Video Size vs Processing Time")
        
        # Plot 4: RAM Usage
        ax4 = fig.add_subplot(224)
        successful_df.ram_usage_mb.hist(ax=ax4, bins=20)
        ax4.set_title("RAM Usage Distribution")
        ax4.set_xlabel("RAM Usage (MB)")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_dir / "batch_analysis.png", dpi=300, bbox_inches='tight')
        
        logger.info(f"Analysis results saved to {save_dir}")
        logger.info(f"Success rate: {stats['successful']}/{stats['total_videos']} "
                   f"({stats['successful']/stats['total_videos']*100:.1f}%)")

async def main():
    """Main entry point for batch testing."""
    parser = argparse.ArgumentParser(description="Batch inference tester for video enhancement")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing input videos")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="Video enhancement server URL")
    parser.add_argument("--workers", type=int, default=None, help="Number of concurrent requests (defaults to CPU count)")
    parser.add_argument("--output-dir", type=Path, default=Path("batch_results"), help="Directory to save results")
    parser.add_argument("--retries", type=int, default=3, help="Maximum retry attempts for failed requests")
    
    args = parser.parse_args()
    
    # Configure logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        args.output_dir / "batch_test.log",
        rotation="100 MB",
        level="INFO"
    )
    logger.add(
        lambda msg: print(msg),
        level="INFO",
        colorize=True
    )
    
    try:
        # Initialize tester
        tester = BatchInferenceTester(
            args.server_url, 
            args.workers,
            args.retries
        )
        
        # Run batch processing
        logger.info(f"Starting batch inference with {tester.num_workers} concurrent requests")
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
