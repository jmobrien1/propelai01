"""
PropelAI Authentication Middleware
FastAPI dependencies for authentication and authorization
"""

import os
from typing import Optional, Annotated
from functools import wraps

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from auth.jwt import verify_token, TokenData
from database.connection import get_db, set_tenant_context


# Security scheme
security = HTTPBearer(auto_error=False)

# Development mode flag - allows bypassing auth for local development
DEV_MODE = os.getenv("PROPELAI_ENV", "development") == "development"
DEV_TENANT_ID = os.getenv("DEV_TENANT_ID", "00000000-0000-0000-0000-000000000001")
DEV_USER_ID = os.getenv("DEV_USER_ID", "00000000-0000-0000-0000-000000000001")


class AuthenticatedUser(BaseModel):
    """Authenticated user context."""
    user_id: str
    tenant_id: str
    email: str
    role: str

    def is_admin(self) -> bool:
        return self.role == "admin"

    def is_member(self) -> bool:
        return self.role in ["admin", "member"]

    def is_viewer(self) -> bool:
        return self.role in ["admin", "member", "viewer"]


async def get_current_user(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> AuthenticatedUser:
    """
    FastAPI dependency to get the current authenticated user.

    In development mode, returns a default dev user if no token provided.
    In production, requires valid JWT token.

    Usage:
        @app.get("/protected")
        async def protected_route(user: AuthenticatedUser = Depends(get_current_user)):
            return {"user_id": user.user_id}
    """
    # Development mode bypass
    if DEV_MODE and credentials is None:
        return AuthenticatedUser(
            user_id=DEV_USER_ID,
            tenant_id=DEV_TENANT_ID,
            email="dev@propelai.local",
            role="admin",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(
        user_id=token_data.user_id,
        tenant_id=token_data.tenant_id,
        email=token_data.email,
        role=token_data.role,
    )


async def get_current_tenant(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> str:
    """
    FastAPI dependency to get the current tenant ID.

    Usage:
        @app.get("/items")
        async def get_items(tenant_id: str = Depends(get_current_tenant)):
            return {"tenant_id": tenant_id}
    """
    return user.tenant_id


def require_role(required_roles: list[str]):
    """
    Decorator to require specific roles for an endpoint.

    Usage:
        @app.delete("/users/{user_id}")
        @require_role(["admin"])
        async def delete_user(user_id: str, user: AuthenticatedUser = Depends(get_current_user)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the user in kwargs
            user = kwargs.get("user")
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if user.role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required role: {required_roles}. Your role: {user.role}",
                )

            return await func(*args, **kwargs)
        return wrapper
    return decorator


class AuthenticatedDBSession:
    """
    Database session with tenant context automatically set.

    Usage:
        @app.get("/items")
        async def get_items(
            auth_db: AuthenticatedDBSession = Depends(get_authenticated_db)
        ):
            async with auth_db.session as db:
                # Queries are automatically filtered by tenant
                result = await db.execute(select(Item))
    """
    def __init__(self, session: AsyncSession, user: AuthenticatedUser):
        self.session = session
        self.user = user
        self.tenant_id = user.tenant_id

    async def __aenter__(self) -> AsyncSession:
        await set_tenant_context(self.session, self.tenant_id)
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        else:
            await self.session.commit()


async def get_authenticated_db(
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthenticatedDBSession:
    """
    FastAPI dependency that provides a database session with tenant context.

    The tenant context is automatically set based on the authenticated user,
    enabling Row-Level Security policies.
    """
    return AuthenticatedDBSession(db, user)
