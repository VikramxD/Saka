from api.client import VideoEnhancerClient

client = VideoEnhancerClient()
result = client.enhance_video(
    "/home/jovyan/video-enhancer/scripts/hxh.mp4",
    wait_for_result=True
)
print(result)