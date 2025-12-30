"""
PropelAI Database Models
SQLAlchemy async models with multi-tenancy support
"""

import uuid
from datetime import datetime
from typing import Optional, List, Any

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    Date,
    ForeignKey,
    JSON,
    BigInteger,
    Index,
    text,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ============================================
# Core Tables: Tenants & Users
# ============================================

class Tenant(Base):
    """Multi-tenant organization."""
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(50), default="free")  # free, pro, enterprise
    settings: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    rfps: Mapped[List["RFP"]] = relationship("RFP", back_populates="tenant", cascade="all, delete-orphan")
    library_documents: Mapped[List["LibraryDocument"]] = relationship(
        "LibraryDocument", back_populates="tenant", cascade="all, delete-orphan"
    )


class User(Base):
    """User within a tenant."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), default="member")  # admin, member, viewer
    auth_provider_id: Mapped[Optional[str]] = mapped_column(String(255))  # Auth0/Supabase ID
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="users")

    __table_args__ = (
        Index("idx_users_tenant", "tenant_id"),
        Index("idx_users_email", "email"),
    )


# ============================================
# RFP & Document Tables
# ============================================

class RFP(Base):
    """RFP project."""
    __tablename__ = "rfps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    solicitation_number: Mapped[Optional[str]] = mapped_column(String(100))
    agency: Mapped[Optional[str]] = mapped_column(String(255))
    due_date: Mapped[Optional[datetime]] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(50), default="created")
    processing_mode: Mapped[Optional[str]] = mapped_column(String(50))  # legacy, semantic, best_practices
    processing_progress: Mapped[int] = mapped_column(Integer, default=0)
    processing_message: Mapped[Optional[str]] = mapped_column(Text)
    requirements_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="rfps")
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="rfp", cascade="all, delete-orphan"
    )
    requirements: Mapped[List["Requirement"]] = relationship(
        "Requirement", back_populates="rfp", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_rfps_tenant", "tenant_id"),
        Index("idx_rfps_status", "status"),
    )


class Document(Base):
    """Document within an RFP."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    rfp_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rfps.id", ondelete="CASCADE"),
        nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    s3_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    doc_type: Mapped[Optional[str]] = mapped_column(String(50))  # main_solicitation, sow, amendment
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_status: Mapped[str] = mapped_column(String(50), default="pending")
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    rfp: Mapped["RFP"] = relationship("RFP", back_populates="documents")
    requirements: Mapped[List["Requirement"]] = relationship(
        "Requirement", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_documents_rfp", "rfp_id"),
        Index("idx_documents_tenant", "tenant_id"),
    )


class Requirement(Base):
    """Extracted requirement from an RFP document."""
    __tablename__ = "requirements"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    rfp_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rfps.id", ondelete="CASCADE"),
        nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(100))
    requirement_type: Mapped[Optional[str]] = mapped_column(String(50))  # performance, instruction, evaluation
    priority: Mapped[Optional[str]] = mapped_column(String(20))  # high, medium, low
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    source_page: Mapped[Optional[int]] = mapped_column(Integer)
    bbox_coordinates: Mapped[Optional[dict]] = mapped_column(JSONB)  # {x, y, width, height, page}
    source_snippet: Mapped[Optional[str]] = mapped_column(Text)
    keywords: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    related_requirements: Mapped[Optional[List[uuid.UUID]]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    rfp: Mapped["RFP"] = relationship("RFP", back_populates="requirements")
    document: Mapped["Document"] = relationship("Document", back_populates="requirements")

    __table_args__ = (
        Index("idx_requirements_rfp", "rfp_id"),
        Index("idx_requirements_type", "requirement_type"),
        Index("idx_requirements_tenant", "tenant_id"),
    )


# ============================================
# Company Library Tables
# ============================================

class LibraryDocument(Base):
    """Document in the Company Library."""
    __tablename__ = "library_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    s3_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    doc_type: Mapped[Optional[str]] = mapped_column(String(50))  # resume, past_performance, capability
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    embedding_status: Mapped[str] = mapped_column(String(50), default="pending")
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="library_documents")
    entities: Mapped[List["LibraryEntity"]] = relationship(
        "LibraryEntity", back_populates="document", cascade="all, delete-orphan"
    )
    embeddings: Mapped[List["LibraryEmbedding"]] = relationship(
        "LibraryEmbedding", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_library_documents_tenant", "tenant_id"),
    )


class LibraryEntity(Base):
    """Extracted entity from a library document (resume, past performance, etc.)."""
    __tablename__ = "library_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("library_documents.id", ondelete="CASCADE"),
        nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # resume, past_performance
    extracted_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    document: Mapped["LibraryDocument"] = relationship("LibraryDocument", back_populates="entities")
    embeddings: Mapped[List["LibraryEmbedding"]] = relationship(
        "LibraryEmbedding", back_populates="entity"
    )

    __table_args__ = (
        Index("idx_library_entities_tenant", "tenant_id"),
        Index("idx_library_entities_type", "entity_type"),
    )


class LibraryEmbedding(Base):
    """Vector embedding for library content."""
    __tablename__ = "library_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("library_documents.id", ondelete="CASCADE"),
        nullable=False
    )
    entity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("library_entities.id", ondelete="SET NULL"),
        nullable=True
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[Optional[int]] = mapped_column(Integer)
    embedding = mapped_column(Vector(1536))  # OpenAI ada-002 dimension
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    document: Mapped["LibraryDocument"] = relationship("LibraryDocument", back_populates="embeddings")
    entity: Mapped[Optional["LibraryEntity"]] = relationship("LibraryEntity", back_populates="embeddings")

    __table_args__ = (
        Index("idx_library_embeddings_tenant", "tenant_id"),
        Index(
            "idx_library_embeddings_vector",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


# ============================================
# LangGraph Checkpointing
# ============================================

class Checkpoint(Base):
    """LangGraph checkpoint for persistent workflow state."""
    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    parent_checkpoint_id: Mapped[Optional[str]] = mapped_column(String(255))
    checkpoint_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_checkpoints_thread_created", "thread_id", "created_at"),
    )


# ============================================
# Agent Trace Log (for RLHF/Audit)
# ============================================

class AgentTraceLog(Base):
    """Immutable audit log of agent actions."""
    __tablename__ = "agent_trace_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    proposal_id: Mapped[Optional[str]] = mapped_column(String(50))
    trace_run_id: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    step_type: Mapped[str] = mapped_column(String(20), nullable=False)  # THOUGHT, ACTION, OBSERVATION
    input_state: Mapped[dict] = mapped_column(JSONB, nullable=False)
    output_state: Mapped[dict] = mapped_column(JSONB, nullable=False)
    reasoning_content: Mapped[Optional[str]] = mapped_column(Text)
    tool_calls: Mapped[Optional[dict]] = mapped_column(JSONB)
    tool_outputs: Mapped[Optional[dict]] = mapped_column(JSONB)
    tokens_input: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_output: Mapped[Optional[int]] = mapped_column(Integer)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_trace_tenant", "tenant_id"),
        Index("idx_trace_proposal", "proposal_id"),
        Index("idx_trace_agent", "agent_name"),
        Index("idx_trace_created", "created_at"),
    )


class FeedbackPair(Base):
    """Human feedback pairs for RLHF training."""
    __tablename__ = "feedback_pairs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    trace_log_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_trace_log.id"),
        nullable=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    proposal_id: Mapped[Optional[str]] = mapped_column(String(50))
    section_id: Mapped[Optional[str]] = mapped_column(String(100))
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    original_score: Mapped[Optional[float]] = mapped_column(Float)
    human_edited_text: Mapped[str] = mapped_column(Text, nullable=False)
    edit_type: Mapped[Optional[str]] = mapped_column(String(50))  # tone, fact, strategy, formatting
    prompt_context: Mapped[str] = mapped_column(Text, nullable=False)
    user_role: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_feedback_tenant", "tenant_id"),
        Index("idx_feedback_proposal", "proposal_id"),
    )
