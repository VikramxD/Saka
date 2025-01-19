from api.client import VideoEnhancerClient

client = VideoEnhancerClient()
response = client.enhance_video(
    "/home/jovyan/video-enhancer/scripts/hxh.mp4",
    calculate_ssim=True
)
print(response)