<div align='center'>

# Video Super Resolution Enhancer API 


&nbsp;

<strong>Transform your videos with state-of-the-art AI upscaling.</strong>    
</div>

----

**Video Super Resolution Enhancer API** is a cutting-edge solution to upscale, restore, and enhance videos using AI models.
<div align='center'>
  
<pre>
✅ 2x and 4x Upscaling        ✅ Real-time Monitoring      
✅ Noise Reduction            ✅ Model Flexibility
✅ GPU Auto-scaling           ✅ Sharpening & Artifact Removal
</pre>

</div>

----

## Architecture for the AI Upscaling Pipeline

```mermaid
 flowchart TD
    subgraph "Request Intake"
        A["Request Queue\n(e.g., RabbitMQ)"]
        B["Job Scheduler\n- Generate Unique Job ID\n- Set Initial task_status: PENDING"]
    end

    subgraph "Model Preparation"
        C["Model Registry\n- Select Appropriate Model"]
        D["Model Optimization\n- Convert to ONNX\n- Convert to TensorRT\n- Apply Quantization"]
        E["Model Caching\n- Store Optimized Models\n- Version Control"]
        F["LitServe Deployment\n- Auto-scale across GPUs\n- Dynamic Resource Allocation"]
    end

    subgraph "Inference Pipeline"
        G["Inference Request\n- Load Cached Optimized Model\n- Perform Inference"]
        H["Result Processing\n- Generate Output\n- Update task_status\n- Track Inference Metrics"]
    end

    subgraph "Storage & Logging"
        I["Object Storage\n(S3/MinIO)\n- Store Inference Results\n- Versioned Storage"]
        J["Prometheus Metrics Collection\n- Memory Usage\n- Model FPS\n- Latency\n- GPU Utilization"]
        K["Grafana Visualization\n- Real-time Dashboards\n- Alerts\n- Performance Tracking"]
        L["Logging System\n- Centralized Logging\n- Stack Trace Capture\n- Error Tracking"]
    end

    subgraph "Task Management"
        M["Task Status Tracking\n- PENDING\n- PROCESSING\n- COMPLETED\n- FAILED"]
        N["Notification System\n- Email/Slack Alerts\n- Status Updates"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    B --> G
    G --> H
    H --> I
    H --> J
    H --> M
    G --> L
    J --> K
    M --> N
    I --> N
    L --> K
```
----

### Stack

- **RabbitMQ**: Robust message queue for handling requests.
- **Prometheus + Grafana**: Self-hosted monitoring and visualization for metrics such as GPU utilization and latency.
- **Litserve**: Model inference serving with GPU autoscaling, batching, and streaming.
- **GitHub Actions**: CI/CD pipelines for seamless deployment and testing.
- **Docker**: Containerization for scalability and ease of deployment.
- **Loguru**: Centralized logging with stack trace capture and error tracking.
- **Model Package**: Repository containing inference code for supported models.
- **AWS S3 or Serverless Providers**: For storing inference results with version control.

----

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- FFmpeg installed on your system

### Installation

1. Clone the repository with submodules:
```bash
# Clone with submodules
git clone --recursive https://github.com/suducrates/video-enhancer.git

# Or if you've already cloned the repository:
git submodule init
git submodule update
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install FFmpeg if not already installed:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install ffmpeg


```

## 💡 Usage

### API Endpoints

- `POST /enhance`: Submit a video for enhancement
- `GET /status/{job_id}`: Check job status
- `GET /result/{job_id}`: Download enhanced video

### Example Usage

```python
import requests

# Submit a video for enhancement
with open('input.mp4', 'rb') as f:
    response = requests.post('http://localhost:8000/predict',
                           files={'video': f},
                           data={'scale': 2,
                                 'model': 'real-esrgan'})
job_id = response.json()['job_id']

# Check status
status = requests.get(f'http://localhost:8000/status/{job_id}').json()
print(f"Job Status: {status['status']}")

# Download result when complete
if status['status'] == 'COMPLETED':
    result = requests.get(f'http://localhost:8000/result/{job_id}')
    with open('enhanced.mp4', 'wb') as f:
        f.write(result.content)
```

## 🛠️ Development

### Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
pytest tests/
```

### Environment Variables

Create a `.env` file with the following variables:

```env
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
S3_BUCKET=video-enhancer-storage
PROMETHEUS_PORT=9090
GPU_MEMORY_FRACTION=0.8
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Out of Memory**
   - Reduce batch size in config
   - Lower video resolution
   - Use CPU fallback mode

2. **Slow Processing**
   - Check GPU utilization
   - Verify CUDA installation
   - Monitor system resources

3. **Failed Jobs**
   - Check logs in Grafana
   - Verify input video format
   - Ensure sufficient storage

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Metrics & Monitoring

Access monitoring dashboards:
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

Key metrics tracked:
- GPU Utilization
- Processing Time
- Queue Length
- Error Rates
- Model Performance

## API Information

### API Endpoints

- `POST /predict`: Enhance a video
- `GET /status/{job_id}`: Check job status
- `GET /result/{job_id}`: Download enhanced video

### Request Format

```json
{
    "video_base64": "<base64 encoded video data>",
    "calculate_ssim": true
}
```

### Response Format

```json
{
    "output_url": "https://storage.example.com/videos/uuid/enhanced.mp4",
    "metrics": {
        "ram_usage_mb": 1024.5,
        "processing_time_sec": 15.3,
        "ssim_score": 0.95
    }
}
```

### Configuration

Key settings in `configs/settings.py`:

```python
class S3Settings:
    bucket_name: str  # S3 bucket for storing enhanced videos
    access_key: str   # S3 access key
    secret_key: str   # S3 secret key
    endpoint_url: str # Optional custom S3 endpoint
    region: str       # S3 region

class RealESRGANSettings:
    model_path: str   # Path to model weights
    device: str       # cuda or cpu
    tile_size: int    # Tile size for processing
    batch_size: int   # Batch size for inference
```

### Error Handling

Common error responses:
- `400 Bad Request`: Invalid input (e.g., no video file, invalid base64)
- `413 Payload Too Large`: Video file too large
- `500 Internal Server Error`: Processing failed
