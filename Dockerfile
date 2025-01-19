# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/outputs \
    ${PROMETHEUS_MULTIPROC_DIR}

# Set permissions
RUN chmod -R 777 \
    /app/logs \
    /app/outputs \
    ${PROMETHEUS_MULTIPROC_DIR}

# Clone and setup RealESRGAN
RUN git clone https://github.com/xinntao/Real-ESRGAN.git \
    && cd Real-ESRGAN \
    && pip install basicsr facexlib gfpgan \
    && pip install -r requirements.txt \
    && python setup.py develop \
    && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth \
        -P experiments/pretrained_models

# Expose ports for API and Prometheus
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["python3", "-m", "api.serve_enhancer"]
