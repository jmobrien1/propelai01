"""
PropelAI v4.1 Integration Tests

Tests for authentication, teams, 2FA, rate limiting, email service, and session management.

Run with: pytest tests/test_auth_v41.py -v
"""

import pytest
import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== Email Service Tests ==============

class TestEmailService:
    """Tests for the email service abstraction"""

    def test_email_config_from_env(self):
        """Test that email config loads from environment"""
        from api.email_service import EmailConfig, EmailProvider

        # Default config (no env vars)
        config = EmailConfig.from_env()
        assert config.provider == EmailProvider.CONSOLE
        assert config.from_email == "noreply@propelai.com"
        assert config.from_name == "PropelAI"

    def test_email_config_smtp(self):
        """Test SMTP configuration"""
        from api.email_service import EmailConfig, EmailProvider

        with patch.dict(os.environ, {
            "EMAIL_PROVIDER": "smtp",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "465",
            "SMTP_USER": "user@example.com",
            "SMTP_PASSWORD": "secret",
        }):
            config = EmailConfig.from_env()
            assert config.provider == EmailProvider.SMTP
            assert config.smtp_host == "smtp.example.com"
            assert config.smtp_port == 465

    def test_email_config_sendgrid(self):
        """Test SendGrid configuration"""
        from api.email_service import EmailConfig, EmailProvider

        with patch.dict(os.environ, {
            "EMAIL_PROVIDER": "sendgrid",
            "SENDGRID_API_KEY": "SG.test-key",
        }):
            config = EmailConfig.from_env()
            assert config.provider == EmailProvider.SENDGRID
            assert config.sendgrid_api_key == "SG.test-key"

    def test_password_reset_template(self):
        """Test password reset email template generation"""
        from api.email_service import EmailTemplate

        subject, html, text = EmailTemplate.password_reset(
            user_name="John Doe",
            reset_url="https://example.com/reset?token=abc123"
        )

        assert "Reset Your PropelAI Password" in subject
        assert "John Doe" in html
        assert "John Doe" in text
        assert "https://example.com/reset?token=abc123" in html
        assert "https://example.com/reset?token=abc123" in text

    def test_team_invitation_template(self):
        """Test team invitation email template generation"""
        from api.email_service import EmailTemplate

        subject, html, text = EmailTemplate.team_invitation(
            invitee_email="new@example.com",
            team_name="Acme Corp",
            inviter_name="Jane Smith",
            invitation_url="https://example.com/invite?token=xyz",
            role="contributor"
        )

        assert "Acme Corp" in subject
        assert "Jane Smith" in html
        assert "Acme Corp" in html
        assert "Contributor" in html  # Role is title-cased
        assert "Can add and edit library content" in html  # Role description

    def test_welcome_template(self):
        """Test welcome email template generation"""
        from api.email_service import EmailTemplate

        subject, html, text = EmailTemplate.welcome(
            user_name="New User",
            login_url="https://example.com/"
        )

        assert "Welcome" in subject
        assert "New User" in html
        assert "https://example.com/" in html

    @pytest.mark.asyncio
    async def test_console_provider_sends(self):
        """Test that console provider 'sends' emails (prints to console)"""
        from api.email_service import ConsoleEmailProvider

        provider = ConsoleEmailProvider()
        result = await provider.send(
            to_email="test@example.com",
            subject="Test Subject",
            html_body="<p>HTML</p>",
            text_body="Text",
            from_email="noreply@example.com",
            from_name="Test"
        )

        assert result is True


# ============== Rate Limiter Tests ==============

class TestRateLimiter:
    """Tests for the rate limiting functionality"""

    @pytest.mark.asyncio
    async def test_memory_rate_limiter_allows_under_limit(self):
        """Test that requests under the limit are allowed"""
        from api.main import RateLimiter

        limiter = RateLimiter()

        # First few requests should be allowed
        for i in range(3):
            is_limited, retry_after = await limiter.is_rate_limited(
                key="test_key",
                max_requests=5,
                window_seconds=60
            )
            assert is_limited is False
            assert retry_after == 0

    @pytest.mark.asyncio
    async def test_memory_rate_limiter_blocks_over_limit(self):
        """Test that requests over the limit are blocked"""
        from api.main import RateLimiter

        limiter = RateLimiter()

        # Use up all allowed requests
        for i in range(5):
            await limiter.is_rate_limited(
                key="block_test",
                max_requests=5,
                window_seconds=60
            )

        # Next request should be blocked
        is_limited, retry_after = await limiter.is_rate_limited(
            key="block_test",
            max_requests=5,
            window_seconds=60
        )

        assert is_limited is True
        assert retry_after > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_different_keys(self):
        """Test that different keys have separate limits"""
        from api.main import RateLimiter

        limiter = RateLimiter()

        # Max out key1
        for i in range(3):
            await limiter.is_rate_limited("key1", 3, 60)

        # key1 should be limited
        is_limited1, _ = await limiter.is_rate_limited("key1", 3, 60)
        assert is_limited1 is True

        # key2 should still be allowed
        is_limited2, _ = await limiter.is_rate_limited("key2", 3, 60)
        assert is_limited2 is False


# ============== Authentication Tests ==============

class TestAuthentication:
    """Tests for authentication functionality"""

    def test_password_hashing(self):
        """Test password hashing and verification"""
        from api.main import hash_password, verify_password

        password = "secure_password_123"
        hashed = hash_password(password)

        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_jwt_token_creation(self):
        """Test JWT token creation"""
        from api.main import create_jwt_token, verify_jwt_token

        token = create_jwt_token(
            user_id="user123",
            email="test@example.com",
            name="Test User"
        )

        assert token is not None
        assert len(token) > 0

    def test_jwt_token_verification(self):
        """Test JWT token verification"""
        from api.main import create_jwt_token, verify_jwt_token

        token = create_jwt_token(
            user_id="user123",
            email="test@example.com",
            name="Test User"
        )

        payload = verify_jwt_token(token)

        assert payload["sub"] == "user123"
        assert payload["email"] == "test@example.com"
        assert payload["name"] == "Test User"

    def test_jwt_token_invalid(self):
        """Test that invalid JWT tokens are rejected"""
        from api.main import verify_jwt_token

        with pytest.raises(ValueError):
            verify_jwt_token("invalid.token.here")

    def test_generate_id(self):
        """Test unique ID generation"""
        from api.main import generate_id

        id1 = generate_id()
        id2 = generate_id()

        assert id1 != id2
        assert len(id1) == 8
        assert id1.isupper()


# ============== 2FA Tests ==============

class TestTwoFactorAuth:
    """Tests for two-factor authentication"""

    def test_backup_code_generation(self):
        """Test backup code generation"""
        from api.main import generate_backup_codes

        codes = generate_backup_codes()

        assert len(codes) == 10
        for code in codes:
            assert len(code) == 8
            assert code.isalnum()
            assert code.isupper()

    def test_backup_codes_unique(self):
        """Test that backup codes are unique"""
        from api.main import generate_backup_codes

        codes = generate_backup_codes()
        unique_codes = set(codes)

        assert len(unique_codes) == len(codes)


# ============== Session Management Tests ==============

class TestSessionManagement:
    """Tests for session management functionality"""

    def test_get_device_info_chrome(self):
        """Test device info extraction for Chrome"""
        from api.main import get_device_info

        request = MagicMock()
        request.headers.get.return_value = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        device = get_device_info(request)
        assert device == "Chrome Browser"

    def test_get_device_info_safari(self):
        """Test device info extraction for Safari"""
        from api.main import get_device_info

        request = MagicMock()
        request.headers.get.return_value = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

        device = get_device_info(request)
        assert device == "Safari Browser"

    def test_get_device_info_iphone(self):
        """Test device info extraction for iPhone"""
        from api.main import get_device_info

        request = MagicMock()
        request.headers.get.return_value = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Mobile/15E148"

        device = get_device_info(request)
        assert device == "iPhone"

    def test_get_device_info_android(self):
        """Test device info extraction for Android"""
        from api.main import get_device_info

        request = MagicMock()
        request.headers.get.return_value = "Mozilla/5.0 (Linux; Android 13; Pixel 7) Mobile"

        device = get_device_info(request)
        assert device == "Android"

    def test_get_client_ip_direct(self):
        """Test client IP extraction (direct connection)"""
        from api.main import get_client_ip

        request = MagicMock()
        request.headers.get.return_value = None
        request.client.host = "192.168.1.100"

        ip = get_client_ip(request)
        assert ip == "192.168.1.100"

    def test_get_client_ip_forwarded(self):
        """Test client IP extraction (behind proxy)"""
        from api.main import get_client_ip

        request = MagicMock()
        request.headers.get.side_effect = lambda h: "203.0.113.1, 10.0.0.1" if h == "x-forwarded-for" else None

        ip = get_client_ip(request)
        assert ip == "203.0.113.1"

    def test_get_client_ip_real_ip(self):
        """Test client IP extraction (X-Real-IP header)"""
        from api.main import get_client_ip

        def header_getter(h):
            if h == "x-forwarded-for":
                return None
            if h == "x-real-ip":
                return "198.51.100.1"
            return None

        request = MagicMock()
        request.headers.get.side_effect = header_getter

        ip = get_client_ip(request)
        assert ip == "198.51.100.1"


# ============== Database Model Tests ==============

class TestDatabaseModels:
    """Tests for database models"""

    def test_user_model_to_dict(self):
        """Test UserModel.to_dict() method"""
        from api.database import UserModel

        user = UserModel(
            id="USR001",
            email="test@example.com",
            name="Test User",
            totp_enabled=False,
        )
        user.created_at = datetime.utcnow()

        data = user.to_dict()

        assert data["id"] == "USR001"
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"
        assert data["totp_enabled"] is False
        assert "password_hash" not in data  # Should not expose password

    def test_team_model_to_dict(self):
        """Test TeamModel.to_dict() method"""
        from api.database import TeamModel

        team = TeamModel(
            id="TEAM001",
            name="Test Team",
            slug="test-team",
            description="A test team",
        )
        team.created_at = datetime.utcnow()
        team.memberships = []

        data = team.to_dict()

        assert data["id"] == "TEAM001"
        assert data["name"] == "Test Team"
        assert data["slug"] == "test-team"
        assert data["member_count"] == 0

    def test_session_model_to_dict(self):
        """Test UserSessionModel.to_dict() method"""
        from api.database import UserSessionModel

        session = UserSessionModel(
            id="SESS001",
            user_id="USR001",
            token_hash="abc123",
            device_info="Chrome Browser",
            ip_address="192.168.1.1",
            is_current=True,
            expires_at=datetime.utcnow() + timedelta(hours=24),
        )
        session.created_at = datetime.utcnow()
        session.last_active = datetime.utcnow()

        data = session.to_dict()

        assert data["id"] == "SESS001"
        assert data["device_info"] == "Chrome Browser"
        assert data["is_active"] is True
        assert "token_hash" not in data  # Should not expose token hash

    def test_invitation_model_to_dict(self):
        """Test TeamInvitationModel.to_dict() method"""
        from api.database import TeamInvitationModel

        invitation = TeamInvitationModel(
            id="INV001",
            team_id="TEAM001",
            email="new@example.com",
            role="contributor",
            token="secure_token",
            status="pending",
            expires_at=datetime.utcnow() + timedelta(days=7),
        )
        invitation.created_at = datetime.utcnow()

        data = invitation.to_dict()

        assert data["id"] == "INV001"
        assert data["email"] == "new@example.com"
        assert data["role"] == "contributor"
        assert data["status"] == "pending"
        assert data["is_expired"] is False
        assert "token" not in data  # Should not expose token


# ============== Integration Tests ==============

class TestIntegration:
    """Integration tests for API endpoints"""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app"""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_check(self, test_client):
        """Test that the API is accessible"""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_rate_limit_config_exists(self):
        """Test that rate limit configurations are defined"""
        from api.main import RATE_LIMITS

        assert "login" in RATE_LIMITS
        assert "register" in RATE_LIMITS
        assert "forgot_password" in RATE_LIMITS
        assert "api_general" in RATE_LIMITS

    def test_jwt_config_exists(self):
        """Test that JWT configuration is defined"""
        from api.main import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS

        assert JWT_SECRET is not None
        assert JWT_ALGORITHM == "HS256"
        assert JWT_EXPIRY_HOURS > 0


# ============== Password Validation Tests ==============

class TestPasswordValidation:
    """Tests for password strength validation"""

    def test_password_too_short(self):
        """Test that short passwords are rejected"""
        from api.main import validate_password_strength

        is_valid, error = validate_password_strength("Short1")
        assert is_valid is False
        assert "8 characters" in error

    def test_password_no_uppercase(self):
        """Test that passwords without uppercase are rejected"""
        from api.main import validate_password_strength

        is_valid, error = validate_password_strength("lowercase123")
        assert is_valid is False
        assert "uppercase" in error.lower()

    def test_password_no_lowercase(self):
        """Test that passwords without lowercase are rejected"""
        from api.main import validate_password_strength

        is_valid, error = validate_password_strength("UPPERCASE123")
        assert is_valid is False
        assert "lowercase" in error.lower()

    def test_password_no_digit(self):
        """Test that passwords without digits are rejected"""
        from api.main import validate_password_strength

        is_valid, error = validate_password_strength("NoDigitsHere")
        assert is_valid is False
        assert "number" in error.lower()

    def test_password_valid(self):
        """Test that valid passwords pass validation"""
        from api.main import validate_password_strength

        is_valid, error = validate_password_strength("ValidPass123")
        assert is_valid is True
        assert error == ""


# ============== Input Sanitization Tests ==============

class TestInputSanitization:
    """Tests for input sanitization functions"""

    def test_sanitize_html_removes_script_tags(self):
        """Test that script tags are removed"""
        from api.main import sanitize_html

        dirty = '<script>alert("XSS")</script>Hello'
        clean = sanitize_html(dirty)
        assert "<script" not in clean.lower()
        assert "Hello" in clean

    def test_sanitize_html_removes_event_handlers(self):
        """Test that event handlers are removed"""
        from api.main import sanitize_html

        dirty = '<img src="x" onerror="alert(1)">'
        clean = sanitize_html(dirty)
        assert "onerror" not in clean.lower()

    def test_sanitize_html_removes_javascript_urls(self):
        """Test that javascript: URLs are removed"""
        from api.main import sanitize_html

        dirty = '<a href="javascript:alert(1)">Click</a>'
        clean = sanitize_html(dirty)
        assert "javascript:" not in clean.lower()

    def test_sanitize_html_removes_style_tags(self):
        """Test that style tags are removed"""
        from api.main import sanitize_html

        dirty = '<style>body{display:none}</style>Content'
        clean = sanitize_html(dirty)
        assert "<style" not in clean.lower()
        assert "Content" in clean

    def test_sanitize_html_removes_iframe(self):
        """Test that iframe tags are removed"""
        from api.main import sanitize_html

        dirty = '<iframe src="http://evil.com"></iframe>Safe'
        clean = sanitize_html(dirty)
        assert "<iframe" not in clean.lower()
        assert "Safe" in clean

    def test_sanitize_html_preserves_safe_content(self):
        """Test that safe content is preserved"""
        from api.main import sanitize_html

        safe = "This is normal text with <b>bold</b> formatting"
        clean = sanitize_html(safe)
        assert "normal text" in clean
        assert "<b>bold</b>" in clean

    def test_sanitize_filename_removes_path(self):
        """Test that path components are removed from filename"""
        from api.main import sanitize_filename

        dirty = "../../../etc/passwd"
        clean = sanitize_filename(dirty)
        assert ".." not in clean
        assert "/" not in clean
        assert "passwd" in clean

    def test_sanitize_filename_removes_null_bytes(self):
        """Test that null bytes are removed from filename"""
        from api.main import sanitize_filename

        dirty = "file\x00.txt"
        clean = sanitize_filename(dirty)
        assert "\x00" not in clean
        assert "file" in clean

    def test_sanitize_filename_replaces_special_chars(self):
        """Test that special characters are replaced"""
        from api.main import sanitize_filename

        dirty = 'file<name>.txt'
        clean = sanitize_filename(dirty)
        assert "<" not in clean
        assert ">" not in clean

    def test_sanitize_filename_limits_length(self):
        """Test that long filenames are truncated"""
        from api.main import sanitize_filename

        dirty = "a" * 300 + ".txt"
        clean = sanitize_filename(dirty)
        assert len(clean) <= 255


# ============== Email Verification Tests ==============

class TestEmailVerification:
    """Tests for email verification functionality"""

    def test_email_verification_template(self):
        """Test email verification template generation"""
        from api.email_service import EmailTemplate

        subject, html, text = EmailTemplate.email_verification(
            user_name="John Doe",
            verification_url="https://example.com/verify?token=abc123"
        )

        assert "Verify" in subject
        assert "John Doe" in html
        assert "John Doe" in text
        assert "https://example.com/verify?token=abc123" in html
        assert "https://example.com/verify?token=abc123" in text
        assert "24 hours" in html  # Expiration warning


# ============== Health Check Tests ==============

class TestHealthChecks:
    """Tests for health check endpoints"""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app"""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test main health check endpoint"""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data

    def test_liveness_probe(self, test_client):
        """Test Kubernetes liveness probe"""
        response = test_client.get("/api/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint"""
        response = test_client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "rfps" in data


# ============== Logging Tests ==============

class TestLogging:
    """Tests for structured logging"""

    def test_json_formatter(self):
        """Test JSON log formatter"""
        from api.main import JSONFormatter
        import logging

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert "timestamp" in data


# ============== Async Test Runner ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
