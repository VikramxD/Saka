import requests
import base64
import json
from pathlib import Path

def test_video_enhancement():
    """Test the video enhancement API."""
    # Load video file
    video_path = Path("../scripts/hxh.mp4")
    if not video_path.exists():
        print(f"Error: Test video not found at {video_path}")
        return
        
    # Read and encode as base64
    with open(video_path, "rb") as f:
        video_data = f.read()
        video_b64 = base64.b64encode(video_data).decode()
    
    print(f"Sending video of size {len(video_data) / 1024 / 1024:.1f}MB")
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "video_data": video_b64,
            "calculate_ssim": False
        }
    )
    
    # Print raw response
    if response.ok:
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_video_enhancement()
