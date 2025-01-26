import base64
import requests

# Read a small portion of video for testing
with open('/home/jovyan/video-enhancer/scripts/dbz.mp4', 'rb') as f:
    # Read first 1MB only for testing
    video_data = f.read(1024*1024)
    video_base64 = base64.b64encode(video_data).decode('utf-8')

# Create request payload as a list with one item
payload = [{
    "video_base64": video_base64
}]

print(f"Sending {len(video_base64)} bytes of base64 data")

# Send request
response = requests.post(
    'http://localhost:8000/predict',
    json=payload
)

# Print response
print(f"\nStatus: {response.status_code}")
if response.ok:
    results = response.json()
    print("Result:", results[0])
else:
    print("Error:", response.text)
