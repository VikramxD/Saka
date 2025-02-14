from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

from configs.spandrel_settings import UpscalerSettings

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
    region: str = Field("ap-south-1", description="S3 region")
    bucket_name: str = Field("sakaresults", description="S3 bucket name")
    storage_path: str = Field(".", description="Base path in bucket for storing videos")
    url_expiration: int = Field(3600, description="Presigned URL expiration time in seconds")
    upload_chunk_size: int = Field(8388608, description="Upload chunk size in bytes (8MB)")
    max_concurrent_uploads: int = Field(4, description="Maximum concurrent upload threads")
    max_retries: int = Field(3, description="Maximum retry attempts for failed operations")
    retry_delay: int = Field(1, description="Delay between retries in seconds")

    class Config:
        env_prefix = "S3_"


class Settings(BaseSettings):
    realesrgan: UpscalerSettings = UpscalerSettings()
    prometheus: PrometheusSettings = PrometheusSettings()
    api: APISettings = APISettings()
    temp: TempSettings = TempSettings()
    s3: S3Settings = S3Settings()

    class Config:
        case_sensitive = False
        env_file = ".env"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
