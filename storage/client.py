"""
PropelAI Storage Client
S3-compatible storage with local filesystem fallback for development
"""

import os
import uuid
import mimetypes
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, BinaryIO, List, Union
import hashlib

from storage.models import StoredFile, UploadResult


class BaseStorageClient(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def upload_file(
        self,
        file: BinaryIO,
        filename: str,
        tenant_id: str,
        folder: str = "rfps",
        metadata: Optional[dict] = None,
    ) -> UploadResult:
        """Upload a file to storage."""
        pass

    @abstractmethod
    async def download_file(self, key: str) -> Optional[bytes]:
        """Download a file from storage."""
        pass

    @abstractmethod
    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned URL for file access."""
        pass

    @abstractmethod
    async def delete_file(self, key: str) -> bool:
        """Delete a file from storage."""
        pass

    @abstractmethod
    async def list_files(self, tenant_id: str, folder: str = "rfps") -> List[StoredFile]:
        """List files for a tenant."""
        pass

    @abstractmethod
    async def file_exists(self, key: str) -> bool:
        """Check if a file exists."""
        pass


class LocalStorageClient(BaseStorageClient):
    """
    Local filesystem storage for development.
    Mimics S3 structure with tenant/folder organization.
    """

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.getenv("STORAGE_PATH", "./data/storage"))
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, key: str) -> Path:
        """Get full filesystem path for a key."""
        return self.base_path / key

    def _generate_key(self, tenant_id: str, folder: str, filename: str) -> str:
        """Generate a unique storage key."""
        # Add unique ID to prevent collisions
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ".-_")
        return f"{tenant_id}/{folder}/{unique_id}_{safe_filename}"

    async def upload_file(
        self,
        file: BinaryIO,
        filename: str,
        tenant_id: str,
        folder: str = "rfps",
        metadata: Optional[dict] = None,
    ) -> UploadResult:
        """Upload a file to local storage."""
        try:
            key = self._generate_key(tenant_id, folder, filename)
            full_path = self._get_full_path(key)

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Read file content
            content = file.read()
            size = len(content)

            # Write to filesystem
            with open(full_path, "wb") as f:
                f.write(content)

            # Store metadata if provided
            if metadata:
                meta_path = full_path.with_suffix(full_path.suffix + ".meta")
                import json
                with open(meta_path, "w") as f:
                    json.dump(metadata, f)

            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            return UploadResult.success_result(
                key=key,
                filename=filename,
                size=size,
                content_type=content_type,
            )

        except Exception as e:
            return UploadResult.error_result(filename=filename, error=str(e))

    async def download_file(self, key: str) -> Optional[bytes]:
        """Download a file from local storage."""
        try:
            full_path = self._get_full_path(key)
            if not full_path.exists():
                return None
            with open(full_path, "rb") as f:
                return f.read()
        except Exception:
            return None

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """
        For local storage, return a file:// URL.
        In production, this would generate an S3 presigned URL.
        """
        full_path = self._get_full_path(key)
        if full_path.exists():
            return f"file://{full_path.absolute()}"
        return None

    async def delete_file(self, key: str) -> bool:
        """Delete a file from local storage."""
        try:
            full_path = self._get_full_path(key)
            if full_path.exists():
                full_path.unlink()
                # Also delete metadata file if exists
                meta_path = full_path.with_suffix(full_path.suffix + ".meta")
                if meta_path.exists():
                    meta_path.unlink()
                return True
            return False
        except Exception:
            return False

    async def list_files(self, tenant_id: str, folder: str = "rfps") -> List[StoredFile]:
        """List files for a tenant."""
        files = []
        tenant_folder = self.base_path / tenant_id / folder

        if not tenant_folder.exists():
            return files

        for file_path in tenant_folder.iterdir():
            if file_path.is_file() and not file_path.suffix == ".meta":
                key = str(file_path.relative_to(self.base_path))
                stat = file_path.stat()
                content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

                files.append(StoredFile(
                    key=key,
                    filename=file_path.name,
                    content_type=content_type,
                    size=stat.st_size,
                    tenant_id=tenant_id,
                    created_at=datetime.fromtimestamp(stat.st_ctime),
                ))

        return files

    async def file_exists(self, key: str) -> bool:
        """Check if a file exists."""
        return self._get_full_path(key).exists()


class S3StorageClient(BaseStorageClient):
    """
    AWS S3 / S3-compatible storage client.
    Supports AWS S3, MinIO, Google Cloud Storage (S3-compatible), etc.
    """

    def __init__(
        self,
        bucket_name: str = None,
        region: str = None,
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
    ):
        import boto3
        from botocore.config import Config

        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "propelai-storage")
        self.region = region or os.getenv("S3_REGION", "us-east-1")

        # S3 client configuration
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        )

        client_kwargs = {
            "service_name": "s3",
            "region_name": self.region,
            "config": config,
        }

        # Custom endpoint for S3-compatible services (MinIO, etc.)
        endpoint = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint

        # Credentials (use environment or IAM role if not provided)
        ak = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        sk = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        if ak and sk:
            client_kwargs["aws_access_key_id"] = ak
            client_kwargs["aws_secret_access_key"] = sk

        self.client = boto3.client(**client_kwargs)

    def _generate_key(self, tenant_id: str, folder: str, filename: str) -> str:
        """Generate a unique S3 key with tenant isolation."""
        unique_id = uuid.uuid4().hex[:8]
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ".-_")
        return f"{tenant_id}/{folder}/{unique_id}_{safe_filename}"

    async def upload_file(
        self,
        file: BinaryIO,
        filename: str,
        tenant_id: str,
        folder: str = "rfps",
        metadata: Optional[dict] = None,
    ) -> UploadResult:
        """Upload a file to S3."""
        try:
            key = self._generate_key(tenant_id, folder, filename)
            content = file.read()
            size = len(content)
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

            # Calculate MD5 for integrity check
            content_md5 = hashlib.md5(content).hexdigest()

            extra_args = {
                "ContentType": content_type,
                "Metadata": {
                    "original-filename": filename,
                    "tenant-id": tenant_id,
                    "content-md5": content_md5,
                    **(metadata or {}),
                },
            }

            # Upload to S3
            file.seek(0)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=content,
                **extra_args,
            )

            return UploadResult.success_result(
                key=key,
                filename=filename,
                size=size,
                content_type=content_type,
            )

        except Exception as e:
            return UploadResult.error_result(filename=filename, error=str(e))

    async def download_file(self, key: str) -> Optional[bytes]:
        """Download a file from S3."""
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()
        except Exception:
            return None

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned URL for S3 file access."""
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception:
            return None

    async def delete_file(self, key: str) -> bool:
        """Delete a file from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception:
            return False

    async def list_files(self, tenant_id: str, folder: str = "rfps") -> List[StoredFile]:
        """List files for a tenant in S3."""
        files = []
        prefix = f"{tenant_id}/{folder}/"

        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # Extract filename from key
                    filename = obj["Key"].split("/")[-1]
                    # Remove unique prefix if present
                    if "_" in filename:
                        filename = filename.split("_", 1)[1]

                    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

                    files.append(StoredFile(
                        key=obj["Key"],
                        filename=filename,
                        content_type=content_type,
                        size=obj["Size"],
                        tenant_id=tenant_id,
                        created_at=obj["LastModified"],
                        etag=obj.get("ETag", "").strip('"'),
                    ))

        except Exception:
            pass

        return files

    async def file_exists(self, key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception:
            return False


class StorageClient:
    """
    Unified storage client that automatically selects the appropriate backend.
    Uses S3 in production, local filesystem in development.
    """

    _instance: Optional[BaseStorageClient] = None

    @classmethod
    def get_client(cls) -> BaseStorageClient:
        """Get or create the storage client singleton."""
        if cls._instance is None:
            # Check environment to determine backend
            storage_backend = os.getenv("STORAGE_BACKEND", "local")

            if storage_backend == "s3":
                cls._instance = S3StorageClient()
            else:
                cls._instance = LocalStorageClient()

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the client instance (for testing)."""
        cls._instance = None


def get_storage_client() -> BaseStorageClient:
    """FastAPI dependency for storage client."""
    return StorageClient.get_client()
