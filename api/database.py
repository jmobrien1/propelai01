"""
PropelAI Database Module (v4.1)

Provides persistent storage using PostgreSQL for RFP data.
File uploads are stored on Render Disk at /data/uploads.

Usage:
    from api.database import get_db, init_db, RFPModel

    # Initialize on startup
    await init_db()

    # Use in endpoints
    async with get_db() as db:
        rfp = await db.get_rfp(rfp_id)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, JSON,
    ForeignKey, create_engine, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import JSONB

# Database URL from environment (Render provides this automatically)
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Convert postgres:// to postgresql:// for SQLAlchemy 2.0
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Async version for asyncpg
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1) if DATABASE_URL else ""

# Base for models
Base = declarative_base()


# ============== Models ==============

class RFPModel(Base):
    """RFP project metadata"""
    __tablename__ = "rfps"

    id = Column(String(50), primary_key=True)
    name = Column(String(500), nullable=False, default="Untitled RFP")
    solicitation_number = Column(String(200), nullable=True)
    agency = Column(String(500), nullable=True)
    due_date = Column(String(100), nullable=True)
    status = Column(String(50), default="created")
    extraction_mode = Column(String(50), nullable=True)

    # File tracking (paths on disk)
    files = Column(JSONB, default=list)  # List of filenames
    file_paths = Column(JSONB, default=list)  # List of full paths

    # Document metadata from guided upload
    document_metadata = Column(JSONB, default=dict)

    # Extracted data (stored as JSON for flexibility)
    stats = Column(JSONB, nullable=True)
    outline = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    requirements = relationship("RequirementModel", back_populates="rfp", cascade="all, delete-orphan")
    amendments = relationship("AmendmentModel", back_populates="rfp", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "name": self.name,
            "solicitation_number": self.solicitation_number,
            "agency": self.agency,
            "due_date": self.due_date,
            "status": self.status,
            "extraction_mode": self.extraction_mode,
            "files": self.files or [],
            "file_paths": self.file_paths or [],
            "document_metadata": self.document_metadata or {},
            "stats": self.stats,
            "outline": self.outline,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "requirements": [r.to_dict() for r in self.requirements] if self.requirements else [],
            "amendments": [a.to_dict() for a in self.amendments] if self.amendments else [],
        }


class RequirementModel(Base):
    """Extracted requirement from RFP"""
    __tablename__ = "requirements"

    id = Column(String(100), primary_key=True)
    rfp_id = Column(String(50), ForeignKey("rfps.id", ondelete="CASCADE"), nullable=False)

    # Requirement content
    text = Column(Text, nullable=False)
    rfp_reference = Column(String(200), nullable=True)  # Original RFP section reference

    # Classification
    category = Column(String(100), nullable=True)  # L_COMPLIANCE, EVALUATION, TECHNICAL
    type = Column(String(100), nullable=True)
    section = Column(String(100), nullable=True)
    subsection = Column(String(200), nullable=True)

    # Priority and binding
    priority = Column(String(20), default="medium")
    binding_level = Column(String(50), nullable=True)  # SHALL, SHOULD, MAY
    binding_keyword = Column(String(50), nullable=True)

    # Confidence
    confidence = Column(Float, default=0.7)
    confidence_level = Column(String(20), nullable=True)
    needs_review = Column(Boolean, default=False)
    review_reasons = Column(JSONB, default=list)

    # Source tracking
    source_page = Column(Integer, nullable=True)
    source_doc = Column(String(500), nullable=True)
    source_content_type = Column(String(50), nullable=True)
    parent_title = Column(String(500), nullable=True)

    # Cross-references
    cross_references = Column(JSONB, default=list)

    # Relationship
    rfp = relationship("RFPModel", back_populates="requirements")

    # Index for faster queries
    __table_args__ = (
        Index('idx_requirements_rfp_id', 'rfp_id'),
        Index('idx_requirements_category', 'category'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rfp_id": self.rfp_id,
            "text": self.text,
            "rfp_reference": self.rfp_reference,
            "category": self.category,
            "type": self.type,
            "section": self.section,
            "subsection": self.subsection,
            "priority": self.priority,
            "binding_level": self.binding_level,
            "binding_keyword": self.binding_keyword,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons or [],
            "source_page": self.source_page,
            "source_doc": self.source_doc,
            "source_content_type": self.source_content_type,
            "parent_title": self.parent_title,
            "cross_references": self.cross_references or [],
        }


class AmendmentModel(Base):
    """RFP amendment tracking"""
    __tablename__ = "amendments"

    id = Column(String(100), primary_key=True)
    rfp_id = Column(String(50), ForeignKey("rfps.id", ondelete="CASCADE"), nullable=False)

    amendment_number = Column(Integer, nullable=False)
    amendment_date = Column(String(100), nullable=True)
    filename = Column(String(500), nullable=True)
    file_path = Column(String(1000), nullable=True)

    # Changes detected
    changes = Column(JSONB, default=list)
    summary = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    rfp = relationship("RFPModel", back_populates="amendments")

    __table_args__ = (
        Index('idx_amendments_rfp_id', 'rfp_id'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rfp_id": self.rfp_id,
            "amendment_number": self.amendment_number,
            "amendment_date": self.amendment_date,
            "filename": self.filename,
            "file_path": self.file_path,
            "changes": self.changes or [],
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ============== Team Workspace Models (v4.1) ==============

from sqlalchemy import Enum as SQLEnum
import enum


class UserRole(str, enum.Enum):
    """Role levels for team members"""
    ADMIN = "admin"
    CONTRIBUTOR = "contributor"
    VIEWER = "viewer"


class UserModel(Base):
    """User accounts"""
    __tablename__ = "users"

    id = Column(String(50), primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)  # NULL if OAuth
    avatar_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Two-Factor Authentication
    totp_secret = Column(String(100), nullable=True)  # TOTP secret key
    totp_enabled = Column(Boolean, default=False)
    totp_backup_codes = Column(JSONB, default=list)  # Backup codes for recovery

    # Account Lockout
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)  # NULL = not locked

    # Email Verification
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(100), nullable=True)
    email_verification_sent_at = Column(DateTime, nullable=True)

    # Relationships
    team_memberships = relationship("TeamMembershipModel", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "is_active": self.is_active,
            "email_verified": self.email_verified or False,
            "totp_enabled": self.totp_enabled or False,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TeamModel(Base):
    """Team workspaces"""
    __tablename__ = "teams"

    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    settings = Column(JSONB, default=dict)
    created_by = Column(String(50), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    memberships = relationship("TeamMembershipModel", back_populates="team", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "settings": self.settings or {},
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "member_count": len(self.memberships) if self.memberships else 0,
        }


class TeamMembershipModel(Base):
    """Team membership with roles"""
    __tablename__ = "team_memberships"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")
    invited_by = Column(String(50), ForeignKey("users.id"), nullable=True)
    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    team = relationship("TeamModel", back_populates="memberships")
    user = relationship("UserModel", back_populates="team_memberships", foreign_keys=[user_id])

    __table_args__ = (
        Index('idx_team_memberships_team_id', 'team_id'),
        Index('idx_team_memberships_user_id', 'user_id'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "role": self.role,
            "invited_by": self.invited_by,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "user": self.user.to_dict() if self.user else None,
        }


class ActivityLogModel(Base):
    """Activity audit trail"""
    __tablename__ = "activity_log"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id", ondelete="CASCADE"), nullable=True)
    user_id = Column(String(50), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(50), nullable=True)
    details = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_activity_log_team_id', 'team_id'),
        Index('idx_activity_log_user_id', 'user_id'),
        Index('idx_activity_log_created_at', 'created_at'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TeamInvitationModel(Base):
    """Team invitations for users who may not have accounts yet"""
    __tablename__ = "team_invitations"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), nullable=False)
    role = Column(String(20), default="viewer")
    token = Column(String(100), nullable=False, unique=True)
    invited_by = Column(String(50), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), default="pending")  # pending, accepted, expired, cancelled
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    accepted_at = Column(DateTime, nullable=True)

    # Relationships
    team = relationship("TeamModel", backref="invitations")
    inviter = relationship("UserModel", foreign_keys=[invited_by])

    __table_args__ = (
        Index('idx_team_invitations_team_id', 'team_id'),
        Index('idx_team_invitations_email', 'email'),
        Index('idx_team_invitations_token', 'token'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "email": self.email,
            "role": self.role,
            "status": self.status,
            "invited_by": self.invited_by,
            "inviter_name": self.inviter.name if self.inviter else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "is_expired": datetime.utcnow() > self.expires_at if self.expires_at else False,
        }


class APIKeyModel(Base):
    """API keys for programmatic access"""
    __tablename__ = "api_keys"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False)  # SHA256 hash of the key
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for identification
    permissions = Column(JSONB, default=list)  # ["read", "write", "admin"]
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_api_keys_team_id', 'team_id'),
        Index('idx_api_keys_user_id', 'user_id'),
        Index('idx_api_keys_key_prefix', 'key_prefix'),
    )

    def to_dict(self, include_prefix: bool = True) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "team_id": self.team_id,
            "user_id": self.user_id,
            "name": self.name,
            "permissions": self.permissions or ["read"],
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_prefix:
            result["key_prefix"] = self.key_prefix
        return result


class UserSessionModel(Base):
    """Active user sessions for session management"""
    __tablename__ = "user_sessions"

    id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), nullable=False)  # Hash of JWT token for revocation
    device_info = Column(String(500), nullable=True)  # User agent / device description
    ip_address = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)  # Approximate location from IP
    is_current = Column(Boolean, default=False)  # Is this the current session?
    last_active = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    # Relationship
    user = relationship("UserModel", backref="sessions")

    __table_args__ = (
        Index('idx_user_sessions_user_id', 'user_id'),
        Index('idx_user_sessions_token_hash', 'token_hash'),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "device_info": self.device_info,
            "ip_address": self.ip_address,
            "location": self.location,
            "is_current": self.is_current,
            "is_active": self.revoked_at is None and datetime.utcnow() < self.expires_at,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


# ============== Database Connection ==============

# Engine and session factory (initialized lazily)
_engine = None
_async_session_factory = None


def _get_engine():
    """Get or create the async engine"""
    global _engine
    if _engine is None and ASYNC_DATABASE_URL:
        _engine = create_async_engine(
            ASYNC_DATABASE_URL,
            echo=False,  # Set to True for SQL debugging
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory():
    """Get or create the session factory"""
    global _async_session_factory
    if _async_session_factory is None:
        engine = _get_engine()
        if engine:
            _async_session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
    return _async_session_factory


async def init_db():
    """Initialize the database (create tables if needed)"""
    engine = _get_engine()
    if engine is None:
        print("[DB] No DATABASE_URL configured, using in-memory store")
        return False

    try:
        async with engine.begin() as conn:
            # Create SQLAlchemy-managed tables
            await conn.run_sync(Base.metadata.create_all)

        # Initialize pgvector extension and Company Library tables
        await init_vector_tables()

        print("[DB] Database initialized successfully")
        return True
    except Exception as e:
        print(f"[DB] Failed to initialize database: {e}")
        return False


async def init_vector_tables():
    """Initialize pgvector extension and Company Library tables"""
    engine = _get_engine()
    if engine is None:
        return False

    # SQL for pgvector and Company Library tables
    init_sql = """
    -- Enable pgvector extension (requires superuser or pre-installed)
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    -- Try to enable vector extension (may already be enabled by hosting provider)
    DO $$
    BEGIN
        CREATE EXTENSION IF NOT EXISTS "vector";
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'pgvector extension not available or already enabled';
    END $$;

    -- Company profile metadata
    CREATE TABLE IF NOT EXISTS company_profiles (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        company_name VARCHAR(500) NOT NULL,
        duns_number VARCHAR(20),
        cage_code VARCHAR(10),
        naics_codes TEXT[],
        set_aside_types TEXT[],
        clearance_level VARCHAR(50),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Capabilities with vector embeddings
    CREATE TABLE IF NOT EXISTS capabilities (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
        name VARCHAR(500) NOT NULL,
        description TEXT NOT NULL,
        category VARCHAR(100),
        keywords TEXT[],
        embedding vector(1536),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Past performance records with vector embeddings
    CREATE TABLE IF NOT EXISTS past_performances (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
        project_name VARCHAR(500) NOT NULL,
        client_name VARCHAR(500),
        client_agency VARCHAR(500),
        contract_number VARCHAR(100),
        contract_value DECIMAL(15,2),
        period_of_performance VARCHAR(100),
        description TEXT NOT NULL,
        relevance_keywords TEXT[],
        metrics JSONB,
        embedding vector(1536),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Key personnel with vector embeddings
    CREATE TABLE IF NOT EXISTS key_personnel (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        role VARCHAR(255),
        years_experience INTEGER,
        clearance_level VARCHAR(50),
        certifications TEXT[],
        bio TEXT NOT NULL,
        expertise_areas TEXT[],
        embedding vector(1536),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Differentiators with vector embeddings
    CREATE TABLE IF NOT EXISTS differentiators (
        id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
        company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
        title VARCHAR(500) NOT NULL,
        description TEXT NOT NULL,
        category VARCHAR(100),
        proof_points TEXT[],
        competitor_comparison TEXT,
        embedding vector(1536),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    try:
        async with engine.begin() as conn:
            from sqlalchemy import text
            # Execute each statement separately
            for statement in init_sql.split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        await conn.execute(text(statement))
                    except Exception as e:
                        # Log but continue - table may already exist
                        if 'already exists' not in str(e).lower():
                            print(f"[DB] Warning: {e}")

        print("[DB] Company Library tables ready")
        return True
    except Exception as e:
        print(f"[DB] Warning: Could not initialize vector tables: {e}")
        return False


@asynccontextmanager
async def get_db_session():
    """Get a database session (async context manager)"""
    factory = _get_session_factory()
    if factory is None:
        yield None
        return

    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def is_db_available() -> bool:
    """Check if database is configured and available"""
    return bool(ASYNC_DATABASE_URL)


# ============== High-Level Database Operations ==============

class DatabaseStore:
    """
    High-level database operations that mirror the in-memory RFPStore interface.
    This allows easy swapping between in-memory and database storage.
    """

    async def create(self, rfp_id: str, data: Dict) -> Dict:
        """Create a new RFP entry"""
        async with get_db_session() as session:
            if session is None:
                raise RuntimeError("Database not available")

            rfp = RFPModel(
                id=rfp_id,
                name=data.get("name", "Untitled RFP"),
                solicitation_number=data.get("solicitation_number"),
                agency=data.get("agency"),
                due_date=data.get("due_date"),
                status="created",
                files=[],
                file_paths=[],
            )
            session.add(rfp)
            await session.flush()
            return rfp.to_dict()

    async def get(self, rfp_id: str) -> Optional[Dict]:
        """Get RFP by ID"""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        async with get_db_session() as session:
            if session is None:
                return None

            result = await session.execute(
                select(RFPModel)
                .options(selectinload(RFPModel.requirements))
                .options(selectinload(RFPModel.amendments))
                .where(RFPModel.id == rfp_id)
            )
            rfp = result.scalar_one_or_none()
            return rfp.to_dict() if rfp else None

    async def update(self, rfp_id: str, updates: Dict) -> Dict:
        """Update RFP fields"""
        from sqlalchemy import select

        async with get_db_session() as session:
            if session is None:
                raise RuntimeError("Database not available")

            result = await session.execute(
                select(RFPModel).where(RFPModel.id == rfp_id)
            )
            rfp = result.scalar_one_or_none()
            if not rfp:
                raise KeyError(f"RFP not found: {rfp_id}")

            # Handle requirements specially - they go to separate table
            if "requirements" in updates:
                # Delete existing requirements
                from sqlalchemy import delete
                await session.execute(
                    delete(RequirementModel).where(RequirementModel.rfp_id == rfp_id)
                )

                # Insert new requirements
                for req_data in updates["requirements"]:
                    req = RequirementModel(
                        id=f"{rfp_id}_{req_data.get('id', '')}",
                        rfp_id=rfp_id,
                        text=req_data.get("text", ""),
                        rfp_reference=req_data.get("rfp_reference"),
                        category=req_data.get("category"),
                        type=req_data.get("type"),
                        section=req_data.get("section"),
                        subsection=req_data.get("subsection"),
                        priority=req_data.get("priority", "medium"),
                        binding_level=req_data.get("binding_level"),
                        binding_keyword=req_data.get("binding_keyword"),
                        confidence=req_data.get("confidence", 0.7),
                        confidence_level=req_data.get("confidence_level"),
                        needs_review=req_data.get("needs_review", False),
                        review_reasons=req_data.get("review_reasons", []),
                        source_page=req_data.get("source_page"),
                        source_doc=req_data.get("source_doc"),
                        source_content_type=req_data.get("source_content_type"),
                        parent_title=req_data.get("parent_title"),
                        cross_references=req_data.get("cross_references", []),
                    )
                    session.add(req)
                del updates["requirements"]

            # Update other fields
            for key, value in updates.items():
                if hasattr(rfp, key):
                    setattr(rfp, key, value)

            rfp.updated_at = datetime.utcnow()
            await session.flush()

            # Reload with relationships
            return await self.get(rfp_id)

    async def delete(self, rfp_id: str) -> bool:
        """Delete RFP"""
        from sqlalchemy import select, delete

        async with get_db_session() as session:
            if session is None:
                return False

            result = await session.execute(
                select(RFPModel).where(RFPModel.id == rfp_id)
            )
            rfp = result.scalar_one_or_none()
            if not rfp:
                return False

            await session.delete(rfp)
            return True

    async def list_all(self) -> List[Dict]:
        """List all RFPs"""
        from sqlalchemy import select

        async with get_db_session() as session:
            if session is None:
                return []

            result = await session.execute(
                select(RFPModel).order_by(RFPModel.created_at.desc())
            )
            rfps = result.scalars().all()
            return [rfp.to_dict() for rfp in rfps]


# Global database store instance
db_store = DatabaseStore()
