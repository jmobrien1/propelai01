"""
PropelAI Email Service (v4.1)

Provides email sending abstraction with support for:
- SMTP (default, works with any email provider)
- SendGrid (recommended for production)
- Console output (development/testing)

Usage:
    from api.email_service import email_service, EmailTemplate

    # Send password reset email
    await email_service.send_password_reset(
        to_email="user@example.com",
        reset_token="abc123",
        user_name="John Doe"
    )

    # Send team invitation email
    await email_service.send_team_invitation(
        to_email="invitee@example.com",
        team_name="Acme Corp",
        inviter_name="Jane Smith",
        invitation_token="xyz789",
        role="contributor"
    )

Configuration (environment variables):
    EMAIL_PROVIDER: "smtp" | "sendgrid" | "console" (default: "console")
    EMAIL_FROM: Sender email address
    EMAIL_FROM_NAME: Sender display name (default: "PropelAI")

    For SMTP:
        SMTP_HOST: SMTP server hostname
        SMTP_PORT: SMTP server port (default: 587)
        SMTP_USER: SMTP username
        SMTP_PASSWORD: SMTP password
        SMTP_USE_TLS: "true" | "false" (default: "true")

    For SendGrid:
        SENDGRID_API_KEY: SendGrid API key
"""

import os
import asyncio
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


# ============== Configuration ==============

class EmailProvider(str, Enum):
    CONSOLE = "console"
    SMTP = "smtp"
    SENDGRID = "sendgrid"


@dataclass
class EmailConfig:
    """Email service configuration"""
    provider: EmailProvider
    from_email: str
    from_name: str
    base_url: str  # For building links in emails

    # SMTP settings
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True

    # SendGrid settings
    sendgrid_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "EmailConfig":
        """Load configuration from environment variables"""
        provider_str = os.environ.get("EMAIL_PROVIDER", "console").lower()
        provider = EmailProvider(provider_str) if provider_str in [e.value for e in EmailProvider] else EmailProvider.CONSOLE

        return cls(
            provider=provider,
            from_email=os.environ.get("EMAIL_FROM", "noreply@propelai.com"),
            from_name=os.environ.get("EMAIL_FROM_NAME", "PropelAI"),
            base_url=os.environ.get("APP_BASE_URL", "http://localhost:8000"),
            smtp_host=os.environ.get("SMTP_HOST"),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER"),
            smtp_password=os.environ.get("SMTP_PASSWORD"),
            smtp_use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
            sendgrid_api_key=os.environ.get("SENDGRID_API_KEY"),
        )


# ============== Email Templates ==============

class EmailTemplate:
    """Email template builder"""

    @staticmethod
    def password_reset(user_name: str, reset_url: str) -> tuple[str, str, str]:
        """Returns (subject, html_body, text_body) for password reset email"""
        subject = "Reset Your PropelAI Password"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
        .header h1 {{ color: white; margin: 0; font-size: 24px; }}
        .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
        .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
        .button:hover {{ background: #5a6fd6; }}
        .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Password Reset</h1>
        </div>
        <div class="content">
            <p>Hi {user_name},</p>
            <p>We received a request to reset your PropelAI password. Click the button below to create a new password:</p>
            <p style="text-align: center;">
                <a href="{reset_url}" class="button">Reset Password</a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; color: #667eea;">{reset_url}</p>
            <div class="warning">
                ⚠️ This link will expire in 1 hour. If you didn't request this reset, please ignore this email.
            </div>
        </div>
        <div class="footer">
            <p>PropelAI - Autonomous Proposal Operating System</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""
Hi {user_name},

We received a request to reset your PropelAI password.

Click this link to reset your password:
{reset_url}

This link will expire in 1 hour.

If you didn't request this reset, please ignore this email.

--
PropelAI - Autonomous Proposal Operating System
"""

        return subject, html_body, text_body

    @staticmethod
    def team_invitation(
        invitee_email: str,
        team_name: str,
        inviter_name: str,
        invitation_url: str,
        role: str
    ) -> tuple[str, str, str]:
        """Returns (subject, html_body, text_body) for team invitation email"""
        subject = f"You're invited to join {team_name} on PropelAI"

        role_description = {
            "admin": "Full access to manage the team, members, and all content",
            "contributor": "Can add and edit library content",
            "viewer": "Read-only access to library content"
        }.get(role, "Team member")

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
        .header h1 {{ color: white; margin: 0; font-size: 24px; }}
        .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
        .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
        .button:hover {{ background: #5a6fd6; }}
        .role-badge {{ display: inline-block; background: #e9ecef; padding: 4px 12px; border-radius: 12px; font-size: 14px; }}
        .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
        .info-box {{ background: #e7f3ff; border: 1px solid #b6d4fe; padding: 15px; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤝 Team Invitation</h1>
        </div>
        <div class="content">
            <p>Hi there!</p>
            <p><strong>{inviter_name}</strong> has invited you to join <strong>{team_name}</strong> on PropelAI.</p>

            <div class="info-box">
                <p style="margin: 0;"><strong>Your role:</strong> <span class="role-badge">{role.title()}</span></p>
                <p style="margin: 10px 0 0 0; font-size: 14px; color: #666;">{role_description}</p>
            </div>

            <p style="text-align: center;">
                <a href="{invitation_url}" class="button">Accept Invitation</a>
            </p>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; color: #667eea;">{invitation_url}</p>
            <p style="color: #666; font-size: 14px;">This invitation expires in 7 days.</p>
        </div>
        <div class="footer">
            <p>PropelAI - Autonomous Proposal Operating System</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""
Hi there!

{inviter_name} has invited you to join {team_name} on PropelAI.

Your role: {role.title()}
{role_description}

Click this link to accept the invitation:
{invitation_url}

This invitation expires in 7 days.

--
PropelAI - Autonomous Proposal Operating System
"""

        return subject, html_body, text_body

    @staticmethod
    def welcome(user_name: str, login_url: str) -> tuple[str, str, str]:
        """Returns (subject, html_body, text_body) for welcome email"""
        subject = "Welcome to PropelAI!"

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 8px 8px 0 0; }}
        .header h1 {{ color: white; margin: 0; font-size: 24px; }}
        .content {{ background: #f8f9fa; padding: 30px; border-radius: 0 0 8px 8px; }}
        .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
        .feature {{ padding: 10px 0; border-bottom: 1px solid #dee2e6; }}
        .feature:last-child {{ border-bottom: none; }}
        .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Welcome to PropelAI!</h1>
        </div>
        <div class="content">
            <p>Hi {user_name},</p>
            <p>Welcome to PropelAI - your autonomous proposal operating system. We're excited to have you!</p>

            <h3>What you can do:</h3>
            <div class="feature">📄 <strong>Upload RFPs</strong> - Extract requirements automatically</div>
            <div class="feature">🎯 <strong>Analyze Strategy</strong> - Get win themes and competitive insights</div>
            <div class="feature">📝 <strong>Draft Proposals</strong> - AI-powered writing with citations</div>
            <div class="feature">👥 <strong>Collaborate</strong> - Work with your team in shared workspaces</div>

            <p style="text-align: center;">
                <a href="{login_url}" class="button">Get Started</a>
            </p>
        </div>
        <div class="footer">
            <p>PropelAI - Autonomous Proposal Operating System</p>
        </div>
    </div>
</body>
</html>
"""

        text_body = f"""
Hi {user_name},

Welcome to PropelAI - your autonomous proposal operating system!

What you can do:
- Upload RFPs - Extract requirements automatically
- Analyze Strategy - Get win themes and competitive insights
- Draft Proposals - AI-powered writing with citations
- Collaborate - Work with your team in shared workspaces

Get started: {login_url}

--
PropelAI - Autonomous Proposal Operating System
"""

        return subject, html_body, text_body


# ============== Email Providers ==============

class EmailProviderBase(ABC):
    """Abstract base class for email providers"""

    @abstractmethod
    async def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: str,
        from_name: str
    ) -> bool:
        """Send an email. Returns True if successful."""
        pass


class ConsoleEmailProvider(EmailProviderBase):
    """Console output for development/testing"""

    async def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: str,
        from_name: str
    ) -> bool:
        print("\n" + "=" * 60)
        print("📧 EMAIL (Console Provider - Not Actually Sent)")
        print("=" * 60)
        print(f"From: {from_name} <{from_email}>")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print("-" * 60)
        print(text_body)
        print("=" * 60 + "\n")
        return True


class SMTPEmailProvider(EmailProviderBase):
    """SMTP email provider"""

    def __init__(self, config: EmailConfig):
        self.config = config

    async def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: str,
        from_name: str
    ) -> bool:
        # Run SMTP in thread pool since it's blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_sync,
            to_email, subject, html_body, text_body, from_email, from_name
        )

    def _send_sync(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: str,
        from_name: str
    ) -> bool:
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{from_name} <{from_email}>"
            message["To"] = to_email

            # Attach text and HTML parts
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            message.attach(part1)
            message.attach(part2)

            # Connect and send
            if self.config.smtp_use_tls:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    server.starttls(context=context)
                    if self.config.smtp_user and self.config.smtp_password:
                        server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(from_email, to_email, message.as_string())
            else:
                with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                    if self.config.smtp_user and self.config.smtp_password:
                        server.login(self.config.smtp_user, self.config.smtp_password)
                    server.sendmail(from_email, to_email, message.as_string())

            print(f"[Email] Sent to {to_email} via SMTP")
            return True

        except Exception as e:
            print(f"[Email] SMTP error: {e}")
            return False


class SendGridEmailProvider(EmailProviderBase):
    """SendGrid email provider"""

    def __init__(self, config: EmailConfig):
        self.config = config
        self.api_key = config.sendgrid_api_key

    async def send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
        from_email: str,
        from_name: str
    ) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "personalizations": [{"to": [{"email": to_email}]}],
                        "from": {"email": from_email, "name": from_name},
                        "subject": subject,
                        "content": [
                            {"type": "text/plain", "value": text_body},
                            {"type": "text/html", "value": html_body}
                        ]
                    }
                )

                if response.status_code in (200, 201, 202):
                    print(f"[Email] Sent to {to_email} via SendGrid")
                    return True
                else:
                    print(f"[Email] SendGrid error: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            print(f"[Email] SendGrid error: {e}")
            return False


# ============== Email Service ==============

class EmailService:
    """High-level email service"""

    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or EmailConfig.from_env()
        self.provider = self._create_provider()

    def _create_provider(self) -> EmailProviderBase:
        """Create the appropriate email provider based on configuration"""
        if self.config.provider == EmailProvider.SENDGRID:
            if not self.config.sendgrid_api_key:
                print("[Email] Warning: SendGrid configured but no API key, falling back to console")
                return ConsoleEmailProvider()
            return SendGridEmailProvider(self.config)

        elif self.config.provider == EmailProvider.SMTP:
            if not self.config.smtp_host:
                print("[Email] Warning: SMTP configured but no host, falling back to console")
                return ConsoleEmailProvider()
            return SMTPEmailProvider(self.config)

        else:
            return ConsoleEmailProvider()

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str
    ) -> bool:
        """Send a custom email"""
        return await self.provider.send(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
            from_email=self.config.from_email,
            from_name=self.config.from_name
        )

    async def send_password_reset(
        self,
        to_email: str,
        reset_token: str,
        user_name: str
    ) -> bool:
        """Send password reset email"""
        reset_url = f"{self.config.base_url}/?reset_token={reset_token}"
        subject, html_body, text_body = EmailTemplate.password_reset(user_name, reset_url)
        return await self.send_email(to_email, subject, html_body, text_body)

    async def send_team_invitation(
        self,
        to_email: str,
        team_name: str,
        inviter_name: str,
        invitation_token: str,
        role: str = "viewer"
    ) -> bool:
        """Send team invitation email"""
        invitation_url = f"{self.config.base_url}/?invite_token={invitation_token}"
        subject, html_body, text_body = EmailTemplate.team_invitation(
            to_email, team_name, inviter_name, invitation_url, role
        )
        return await self.send_email(to_email, subject, html_body, text_body)

    async def send_welcome(
        self,
        to_email: str,
        user_name: str
    ) -> bool:
        """Send welcome email to new users"""
        login_url = f"{self.config.base_url}/"
        subject, html_body, text_body = EmailTemplate.welcome(user_name, login_url)
        return await self.send_email(to_email, subject, html_body, text_body)


# Global email service instance
email_service = EmailService()
