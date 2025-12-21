"""
PropelAI Configuration

Centralized configuration management using environment variables.
All configuration values are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache


class Settings:
    """
    Application settings loaded from environment variables.

    Usage:
        from api.config import get_settings
        settings = get_settings()
        print(settings.jwt_secret)
    """

    def __init__(self):
        # Environment
        self.environment = os.environ.get("PROPELAI_ENV", "development")
        self.debug = self.environment != "production"

        # API Settings
        self.api_version = "4.1.0"
        self.api_version_major = 4
        self.api_version_minor = 1
        self.api_version_patch = 0

        # Server
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8000"))

        # Database
        self.database_url = self._get_database_url()

        # Redis (optional)
        self.redis_url = os.environ.get("REDIS_URL")

        # JWT Authentication
        self._jwt_secret = os.environ.get("JWT_SECRET")
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24 * 7  # 7 days

        # Validate JWT secret in production
        if self.environment == "production" and not self._jwt_secret:
            raise RuntimeError(
                "CRITICAL: JWT_SECRET environment variable must be set in production. "
                "Generate a secure secret with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

        # CORS
        cors_origins = os.environ.get("CORS_ORIGINS", "*")
        if cors_origins == "*":
            self.cors_origins: List[str] = ["*"]
            self.cors_allow_credentials = False
        else:
            self.cors_origins = [origin.strip() for origin in cors_origins.split(",")]
            self.cors_allow_credentials = True

        # Security Headers
        self.enable_hsts = os.environ.get("ENABLE_HSTS", "").lower() == "true"
        self.enable_csp = os.environ.get("ENABLE_CSP", "").lower() == "true"

        # File Upload
        self.max_file_size_mb = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
        self.max_file_size = self.max_file_size_mb * 1024 * 1024
        self.allowed_extensions = [".pdf", ".docx", ".doc", ".xlsx", ".xls"]

        # Storage Paths
        self.persistent_data_dir = Path(os.environ.get("DATA_DIR", "/data"))
        self.upload_dir = self.persistent_data_dir / "uploads"
        self.output_dir = self.persistent_data_dir / "outputs"

        # Rate Limiting
        self.rate_limit_login = {"max_requests": 5, "window_seconds": 60}
        self.rate_limit_register = {"max_requests": 3, "window_seconds": 60}
        self.rate_limit_forgot_password = {"max_requests": 3, "window_seconds": 300}
        self.rate_limit_api_general = {"max_requests": 100, "window_seconds": 60}

        # Account Security
        self.account_lockout_threshold = 5  # Lock after N failed attempts
        self.account_lockout_duration_minutes = 15

        # Email Verification
        self.require_email_verification = (
            os.environ.get("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"
        )
        self.email_verification_expiry_hours = 24

        # Password Reset
        self.password_reset_expiry_hours = 1

        # Data Retention (Soft Delete)
        self.default_retention_days = 30

        # Email Service
        self.email_provider = os.environ.get("EMAIL_PROVIDER", "console")
        self.email_from = os.environ.get("EMAIL_FROM", "noreply@propelai.com")
        self.email_from_name = os.environ.get("EMAIL_FROM_NAME", "PropelAI")
        self.app_base_url = os.environ.get("APP_BASE_URL", "http://localhost:8000")

        # SMTP Settings
        self.smtp_host = os.environ.get("SMTP_HOST", "")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER", "")
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"

        # SendGrid
        self.sendgrid_api_key = os.environ.get("SENDGRID_API_KEY", "")

        # LLM API Keys
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", "")

    @property
    def jwt_secret(self) -> str:
        """Get JWT secret, with fallback for development."""
        return self._jwt_secret or "propelai-dev-secret-change-in-production"

    def _get_database_url(self) -> str:
        """Get and normalize database URL."""
        url = os.environ.get("DATABASE_URL", "")

        # Handle Render's postgres:// URL (SQLAlchemy requires postgresql://)
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)

        return url

    @property
    def async_database_url(self) -> str:
        """Get async database URL for asyncpg."""
        if not self.database_url:
            return ""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
