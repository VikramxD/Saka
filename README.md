
<div align='center'>

# Video Super Resolution Enhancer API  
<strong>AI-Powered Video Upscaling with Real-ESRGAN and LitServe</strong>    
</div>

---

## 🚀 Key Features

| **AI Capabilities**          | **Performance**              | **Infrastructure**        |
|-------------------------------|-------------------------------|----------------------------|
| 🔥 2x/4x Video Upscaling      | ⚡ GPU Auto-Scaling           | 📊 Real-time Monitoring    |
| 🎨 Anime-Optimized Models     | 🚀 Dynamic Batching           | 🔐 S3-Compatible Storage   |
| ✨ Artifact Removal           | 🔄 Async Processing           | 📈 Quality Metrics (SSIM)  |
| 🤖 Face Enhancement           | 📦 Batch Processing           | 🐳 Docker/K8s Ready       |

---

## 🏗 Architecture Overview

```
flowchart TD
    A[Client Request] --> B[LitServe API]
    B --> C[Model Registry]
    C --> D[GPU Cluster]
    D --> E[S3 Storage]
    D --> F[Prometheus]
    F --> G[Grafana Dashboard]
    B --> H[Result Notification]
```

---

## 🛠 Core Components

### Model Serving (`api/serve_enhancer.py`)
```
from litserve import LitServer, LitAPI

class VideoEnhancerAPI(LitAPI):
    def setup(self, device):
        self.upscaler = VideoUpscaler(settings.realesrgan)
        self.s3 = S3Handler(settings.s3)

server = LitServer(
    VideoEnhancerAPI(),
    accelerator="auto",
    devices='auto',
    workers_per_device=4,
    max_batch_size=16
)
```
- **LitServe Integration**: Native GPU autoscaling and dynamic batching
- **Real-ESRGAN Pipeline**: Direct model execution without queues
- **Metrics Integration**: Built-in Prometheus monitoring

### Key Configuration (`configs/settings.py`)
```
class Settings(BaseSettings):
    realesrgan: UpscalerSettings = UpscalerSettings()
    api: APISettings = APISettings()
    s3: S3Settings = S3Settings()
```

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 11.8+
- Docker 20.10+
- Python 3.10+

### Installation
```
git clone --recursive https://github.com/vikramxD/video-enhancer.git
cd video-enhancer

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### Docker Deployment
```
# docker-compose.yml
services:
  video-enhancer:
    build: .
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8000:8000"
```

---

## 📡 API Endpoints

| Endpoint          | Method | Description                     |
|--------------------|--------|---------------------------------|
| `/predict`        | POST   | Submit video for enhancement    |
| `/metrics`        | GET    | Prometheus metrics endpoint     |

**Example Request:**
```
from api.client import VideoEnhancerClient

client = VideoEnhancerClient()
result = client.enhance_video("input.mp4", calculate_ssim=True)
print(f"Enhanced URL: {result['output_url']}")
```

---

## 📊 Monitoring Stack

![Grafana Dashboard](https://placehold.co/800x400/EEE/31343C?text=Video+Enhancer+Metrics)

Tracked Metrics:
- GPU Utilization
- Processing Latency
- Memory Usage
- Batch Efficiency
- SSIM Quality Scores

Start monitoring:
```
./setup_monitoring.sh
# Access at http://localhost:3000 (admin/admin)
```

---

## 🧩 Key Components

### Model Processing (`scripts/realesrgan.py`)
```
class VideoUpscaler:
    def process_video(self, video_path):
        cmd = [
            "python", "inference_realesrgan_video.py",
            "-n", self.settings.model_name,
            "-s", str(self.settings.scale_factor)
        ]
```

### Performance Features
- Automatic mixed precision (FP16/FP32)
- Tile-based processing for large videos
- Face enhancement options
- Spatial-temporal quality metrics


## 📚 Documentation Structure

```
├── api/
│   ├── client.py          # Client library
│   ├── serve_enhancer.py  # LitServe endpoint
│   └── storage.py         # S3 integration
├── configs/
│   ├── settings.py        # Main configuration
│   └── realesrgan.py       # Model settings
└── scripts/
    └── realesrgan.py      # Processing pipeline
```

---

<div align="center">
  <br>
  📧 **Contact**:VikramxD | 📄 **License**: Apache 2.0
