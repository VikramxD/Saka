import requests
import base64
import time
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

class WebhookHandler(BaseHTTPRequestHandler):
    """Simple webhook handler to print task status updates."""
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        status = json.loads(post_data.decode('utf-8'))
        
        print("\nWebhook received:")
        print(f"Task ID: {status['task_id']}")
        print(f"Status: {status['status']}")
        print(f"Message: {status['message']}")
        if 'processing_time' in status:
            print(f"Processing Time: {status['processing_time']:.2f}s")
        print("----------------------------------------")
        
        self.send_response(200)
        self.end_headers()

def start_webhook_server(port=8001):
    """Start a simple webhook server."""
    server = HTTPServer(('localhost', port), WebhookHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    return server

def check_task_status(task_id: str) -> dict:
    """Check the status of a task."""
    response = requests.post(
        "http://localhost:8000/task_status",
        json={"task_id": task_id}
    )
    
    if not response.ok:
        print(f"Error checking status: {response.text}")
        return None
        
    result = response.json()
    # API returns a list with single item
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return result

def test_video_enhancement():
    # Start webhook server
    webhook_port = 8001
    webhook_server = start_webhook_server(webhook_port)
    print(f"Started webhook server on port {webhook_port}")
    
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
    
    # Submit task
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "video": video_b64,
            "calculate_ssim": False,
            "webhook_url": f"http://localhost:{webhook_port}"
        }
    )
    
    # Check initial response
    if not response.ok:
        print(f"Error submitting task: {response.text}")
        return
        
    result = response.json()
    task_id = result.get("task_id")
    if not task_id:
        print("Error: No task ID received")
        return
        
    print(f"Task submitted with ID: {task_id}")
    print("Waiting for webhook notifications...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping webhook server...")
        webhook_server.shutdown()
        webhook_server.server_close()

if __name__ == "__main__":
    # Example: Check status of a specific task
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        print(f"Checking status of task {task_id}")
        status = check_task_status(task_id)
        if status:
            print(f"Status: {status}")
    else:
        # Run full test
        test_video_enhancement()
