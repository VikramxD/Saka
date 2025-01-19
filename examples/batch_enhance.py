import glob
from pathlib import Path
from api.client import VideoEnhancerClient

def enhance_videos(input_dir: str):
    """Enhance all videos in a directory."""
    
    # Initialize client
    client = VideoEnhancerClient(base_url="http://localhost:8000")
    
    # Get all video files
    video_files = glob.glob(f"{input_dir}/*.mp4")
    
    # Process each video
    for video_path in video_files:
        print(f"\nProcessing: {video_path}")
        try:
            # Enhance video
            response = client.enhance_video(video_path)
            
            # Print results
            print("Success!")
            print(f"Enhanced URL: {response['output_url']}")
            print("Metrics:")
            for key, value in response["metrics"].items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhance multiple videos")
    parser.add_argument("input_dir", help="Directory containing input videos")
    args = parser.parse_args()
    
    enhance_videos(args.input_dir)
