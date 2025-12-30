"""
PropelAI JWT Token Management
Handles token creation and verification
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import jwt
from pydantic import BaseModel

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "propelai-dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))


class TokenData(BaseModel):
    """Data stored in JWT token."""
    user_id: str
    tenant_id: str
    email: str
    role: str
    exp: Optional[datetime] = None


def create_access_token(
    user_id: str,
    tenant_id: str,
    email: str,
    role: str = "member",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User's UUID
        tenant_id: Tenant's UUID
        email: User's email
        role: User's role (admin, member, viewer)
        expires_delta: Optional custom expiration time

    Returns:
        JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)

    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenData(
            user_id=payload["user_id"],
            tenant_id=payload["tenant_id"],
            email=payload["email"],
            role=payload["role"],
            exp=datetime.fromtimestamp(payload["exp"]),
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def decode_token_unverified(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode a token without verifying signature.
    Useful for debugging or extracting claims from expired tokens.
    """
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        return None
