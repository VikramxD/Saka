from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from configs.realesrgan_settings import UpscalerSettings


class PrometheusSettings(BaseSettings):
    prefix: str = Field("video_enhancer", description="Metric name prefix")
    path: str = Field("/metrics", description="Metrics endpoint path")
    enabled: bool = Field(True, description="Enable Prometheus metrics")

    class Config:
        env_prefix = "PROMETHEUS_"


class APISettings(BaseSettings):
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    workers: int = Field(4, description="Number of workers")
    timeout: int = Field(300, description="Request timeout in seconds")
    max_video_size: int = Field(104857600, description="Maximum video size in bytes (100MB)")
    allowed_extensions: List[str] = Field(
        default=[".mp4", ".avi", ".mkv", ".mov"],
        description="Allowed video file extensions"
    )

    class Config:
        env_prefix = "API_"


class TempSettings(BaseSettings):
    directory: str = Field("/tmp/video-enhancer", description="Temporary directory path")
    cleanup_interval: int = Field(3600, description="Cleanup interval in seconds")

    class Config:
        env_prefix = "TEMP_"


class S3Settings(BaseSettings):
    access_key: str = Field('', description="AWS access key or compatible S3 provider key")
    secret_key: str = Field('', description="AWS secret key or compatible S3 provider secret")
    endpoint_url: Optional[str] = Field(None, description="Optional: Custom S3 endpoint URL")
    region: str = Field("", description="S3 region")
    bucket_name: str = Field("diffusion-model-bucket", description="S3 bucket name")
    storage_path: str = Field(".", description="Base path in bucket for storing videos")
    url_expiration: int = Field(3600, description="Presigned URL expiration time in seconds")
    upload_chunk_size: int = Field(8388608, description="Upload chunk size in bytes (8MB)")
    max_concurrent_uploads: int = Field(4, description="Maximum concurrent upload threads")
    max_retries: int = Field(3, description="Maximum retry attempts for failed operations")
    retry_delay: int = Field(1, description="Delay between retries in seconds")

    class Config:
        env_prefix = "S3_"


class RabbitMQSettings(BaseSettings):
    host: str = Field("localhost", description="RabbitMQ host")
    port: int = Field(5672, description="RabbitMQ port")
    username: str = Field("guest", description="RabbitMQ username")
    password: str = Field("guest", description="RabbitMQ password")
    vhost: str = Field("/", description="RabbitMQ virtual host")
    queue_name: str = Field("video_tasks", description="Queue name for video processing tasks")
    exchange_name: str = Field("video_exchange", description="Exchange name for video processing")
    routing_key: str = Field("video.process", description="Routing key for video processing")
    result_queue: str = Field("video_results", description="Queue name for processing results")
    prefetch_count: int = Field(1, description="Number of messages to prefetch")
    connection_retry: int = Field(5, description="Number of connection retry attempts")
    heartbeat: int = Field(600, description="Connection heartbeat in seconds")

    class Config:
        env_prefix = "RABBITMQ_"


class Settings(BaseSettings):
    realesrgan: UpscalerSettings = UpscalerSettings()
    prometheus: PrometheusSettings = PrometheusSettings()
    api: APISettings = APISettings()
    temp: TempSettings = TempSettings()
    s3: S3Settings = S3Settings()
    rabbitmq: RabbitMQSettings = RabbitMQSettings()

    class Config:
        case_sensitive = False
        env_file = ".env"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
