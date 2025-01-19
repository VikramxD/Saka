import boto3
from pathlib import Path
from typing import Optional
from loguru import logger
from botocore.exceptions import ClientError
from configs.settings import S3Settings

class S3Handler:
    """Handler for S3 storage operations."""
    
    def __init__(self, config: S3Settings):
        """Initialize S3 client with configuration."""
        self.config = config
        self.client = boto3.client(
            's3',
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            endpoint_url=config.endpoint_url,
            region_name=config.region
        )
        logger.info("S3 handler initialized for bucket: {}", config.bucket_name)

    def upload_video(self, local_path: Path, s3_path: str) -> str:
        """Upload video to S3 and return the URL."""
        try:
            file_size = local_path.stat().st_size
            logger.info("Uploading video ({} bytes) to {}", file_size, s3_path)

            # Upload file
            with open(local_path, 'rb') as f:
                self.client.upload_fileobj(
                    f,
                    self.config.bucket_name,
                    s3_path
                )

            # Generate HTTPS URL
            url = f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{s3_path}"
            logger.info("Upload complete: {}", url)
            return url

        except ClientError as e:
            logger.exception("S3 upload failed")
            raise
