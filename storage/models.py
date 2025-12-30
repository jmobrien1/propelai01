"""
PropelAI Storage Models
Data models for file storage operations
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class StoredFile:
    """Represents a file stored in the storage backend."""
    key: str  # S3 key / path
    filename: str
    content_type: str
    size: int
    tenant_id: str
    created_at: datetime
    etag: Optional[str] = None
    metadata: Optional[dict] = None

    @property
    def extension(self) -> str:
        """Get file extension."""
        if "." in self.filename:
            return self.filename.rsplit(".", 1)[1].lower()
        return ""

    @property
    def is_pdf(self) -> bool:
        return self.extension == "pdf"

    @property
    def is_docx(self) -> bool:
        return self.extension == "docx"

    @property
    def is_xlsx(self) -> bool:
        return self.extension == "xlsx"


@dataclass
class UploadResult:
    """Result of a file upload operation."""
    success: bool
    key: str
    filename: str
    size: int
    content_type: str
    error: Optional[str] = None

    @classmethod
    def success_result(cls, key: str, filename: str, size: int, content_type: str) -> "UploadResult":
        return cls(
            success=True,
            key=key,
            filename=filename,
            size=size,
            content_type=content_type,
        )

    @classmethod
    def error_result(cls, filename: str, error: str) -> "UploadResult":
        return cls(
            success=False,
            key="",
            filename=filename,
            size=0,
            content_type="",
            error=error,
        )
