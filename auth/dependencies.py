"""
PropelAI Auth Dependencies
Additional authentication utilities and context management
"""

from dataclasses import dataclass
from typing import Optional
from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession


# Context variable for tenant isolation
_tenant_context: ContextVar[Optional[str]] = ContextVar("tenant_context", default=None)


@dataclass
class TenantContext:
    """
    Tenant context for the current request.
    Provides utilities for tenant-scoped operations.
    """
    tenant_id: str
    user_id: str
    role: str

    def __post_init__(self):
        # Set context variable for access in deeper call stacks
        _tenant_context.set(self.tenant_id)

    @classmethod
    def get_current_tenant_id(cls) -> Optional[str]:
        """Get the current tenant ID from context."""
        return _tenant_context.get()

    def can_access_tenant(self, target_tenant_id: str) -> bool:
        """Check if current user can access the target tenant."""
        # For now, users can only access their own tenant
        # In the future, we might support cross-tenant access for super admins
        return self.tenant_id == target_tenant_id

    def filter_by_tenant(self, query):
        """Add tenant filter to a SQLAlchemy query."""
        from database.models import RFP, Document, Requirement, LibraryDocument
        # This will be enhanced to work with any model that has tenant_id
        return query.filter_by(tenant_id=self.tenant_id)


def get_tenant_id() -> Optional[str]:
    """Get the current tenant ID from context variable."""
    return _tenant_context.get()


def set_tenant_id(tenant_id: str):
    """Set the current tenant ID in context variable."""
    _tenant_context.set(tenant_id)


def clear_tenant_context():
    """Clear the tenant context."""
    _tenant_context.set(None)
