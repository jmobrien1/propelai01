"""
PropelAI Database Repositories
Data access layer for all database operations
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import (
    Tenant,
    User,
    RFP,
    Document,
    Requirement,
    LibraryDocument,
    LibraryEntity,
    LibraryEmbedding,
    Checkpoint,
    AgentTraceLog,
)


# ============================================
# Tenant Repository
# ============================================

class TenantRepository:
    """Repository for tenant operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, plan_tier: str = "free") -> Tenant:
        """Create a new tenant."""
        tenant = Tenant(name=name, plan_tier=plan_tier)
        self.session.add(tenant)
        await self.session.flush()
        return tenant

    async def get_by_id(self, tenant_id: uuid.UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        result = await self.session.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Tenant]:
        """Get tenant by name."""
        result = await self.session.execute(
            select(Tenant).where(Tenant.name == name)
        )
        return result.scalar_one_or_none()

    async def update(self, tenant_id: uuid.UUID, **kwargs) -> Optional[Tenant]:
        """Update tenant fields."""
        await self.session.execute(
            update(Tenant).where(Tenant.id == tenant_id).values(**kwargs)
        )
        return await self.get_by_id(tenant_id)


# ============================================
# User Repository
# ============================================

class UserRepository:
    """Repository for user operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: uuid.UUID,
        email: str,
        name: Optional[str] = None,
        role: str = "member",
        auth_provider_id: Optional[str] = None,
    ) -> User:
        """Create a new user."""
        user = User(
            tenant_id=tenant_id,
            email=email,
            name=name,
            role=role,
            auth_provider_id=auth_provider_id,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID."""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def get_by_auth_provider_id(self, auth_provider_id: str) -> Optional[User]:
        """Get user by auth provider ID (Auth0/Supabase)."""
        result = await self.session.execute(
            select(User).where(User.auth_provider_id == auth_provider_id)
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> List[User]:
        """List all users in a tenant."""
        result = await self.session.execute(
            select(User).where(User.tenant_id == tenant_id)
        )
        return list(result.scalars().all())

    async def update_last_login(self, user_id: uuid.UUID) -> None:
        """Update user's last login timestamp."""
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(last_login=datetime.utcnow())
        )


# ============================================
# RFP Repository
# ============================================

@dataclass
class RFPStats:
    """Statistics for an RFP."""
    total_requirements: int
    by_type: Dict[str, int]
    by_priority: Dict[str, int]
    by_section: Dict[str, int]


class RFPRepository:
    """Repository for RFP operations."""

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID):
        self.session = session
        self.tenant_id = tenant_id

    async def create(
        self,
        name: str,
        solicitation_number: Optional[str] = None,
        agency: Optional[str] = None,
        due_date: Optional[datetime] = None,
    ) -> RFP:
        """Create a new RFP."""
        rfp = RFP(
            tenant_id=self.tenant_id,
            name=name,
            solicitation_number=solicitation_number,
            agency=agency,
            due_date=due_date,
        )
        self.session.add(rfp)
        await self.session.flush()
        return rfp

    async def get_by_id(self, rfp_id: uuid.UUID) -> Optional[RFP]:
        """Get RFP by ID with tenant isolation."""
        result = await self.session.execute(
            select(RFP)
            .options(selectinload(RFP.documents))
            .where(and_(RFP.id == rfp_id, RFP.tenant_id == self.tenant_id))
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
    ) -> List[RFP]:
        """List all RFPs for the tenant."""
        query = select(RFP).where(RFP.tenant_id == self.tenant_id)

        if status:
            query = query.where(RFP.status == status)

        query = query.order_by(RFP.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count(self, status: Optional[str] = None) -> int:
        """Count RFPs for the tenant."""
        query = select(func.count(RFP.id)).where(RFP.tenant_id == self.tenant_id)
        if status:
            query = query.where(RFP.status == status)
        result = await self.session.execute(query)
        return result.scalar() or 0

    async def update(self, rfp_id: uuid.UUID, **kwargs) -> Optional[RFP]:
        """Update RFP fields."""
        kwargs["updated_at"] = datetime.utcnow()
        await self.session.execute(
            update(RFP)
            .where(and_(RFP.id == rfp_id, RFP.tenant_id == self.tenant_id))
            .values(**kwargs)
        )
        return await self.get_by_id(rfp_id)

    async def update_status(
        self,
        rfp_id: uuid.UUID,
        status: str,
        progress: int = 0,
        message: Optional[str] = None,
    ) -> None:
        """Update RFP processing status."""
        await self.session.execute(
            update(RFP)
            .where(and_(RFP.id == rfp_id, RFP.tenant_id == self.tenant_id))
            .values(
                status=status,
                processing_progress=progress,
                processing_message=message,
                updated_at=datetime.utcnow(),
            )
        )

    async def delete(self, rfp_id: uuid.UUID) -> bool:
        """Delete an RFP and all related data."""
        result = await self.session.execute(
            delete(RFP).where(and_(RFP.id == rfp_id, RFP.tenant_id == self.tenant_id))
        )
        return result.rowcount > 0

    async def get_stats(self, rfp_id: uuid.UUID) -> RFPStats:
        """Get requirement statistics for an RFP."""
        # Total count
        total_result = await self.session.execute(
            select(func.count(Requirement.id))
            .where(and_(
                Requirement.rfp_id == rfp_id,
                Requirement.tenant_id == self.tenant_id
            ))
        )
        total = total_result.scalar() or 0

        # By type
        type_result = await self.session.execute(
            select(Requirement.requirement_type, func.count(Requirement.id))
            .where(and_(
                Requirement.rfp_id == rfp_id,
                Requirement.tenant_id == self.tenant_id
            ))
            .group_by(Requirement.requirement_type)
        )
        by_type = {row[0] or "unknown": row[1] for row in type_result}

        # By priority
        priority_result = await self.session.execute(
            select(Requirement.priority, func.count(Requirement.id))
            .where(and_(
                Requirement.rfp_id == rfp_id,
                Requirement.tenant_id == self.tenant_id
            ))
            .group_by(Requirement.priority)
        )
        by_priority = {row[0] or "unknown": row[1] for row in priority_result}

        # By section
        section_result = await self.session.execute(
            select(Requirement.section, func.count(Requirement.id))
            .where(and_(
                Requirement.rfp_id == rfp_id,
                Requirement.tenant_id == self.tenant_id
            ))
            .group_by(Requirement.section)
        )
        by_section = {row[0] or "unknown": row[1] for row in section_result}

        return RFPStats(
            total_requirements=total,
            by_type=by_type,
            by_priority=by_priority,
            by_section=by_section,
        )


# ============================================
# Document Repository
# ============================================

class DocumentRepository:
    """Repository for document operations."""

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID):
        self.session = session
        self.tenant_id = tenant_id

    async def create(
        self,
        rfp_id: uuid.UUID,
        filename: str,
        s3_path: str,
        doc_type: Optional[str] = None,
        file_size: Optional[int] = None,
        mime_type: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> Document:
        """Create a new document."""
        document = Document(
            tenant_id=self.tenant_id,
            rfp_id=rfp_id,
            filename=filename,
            s3_path=s3_path,
            doc_type=doc_type,
            file_size=file_size,
            mime_type=mime_type,
            page_count=page_count,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(self, document_id: uuid.UUID) -> Optional[Document]:
        """Get document by ID."""
        result = await self.session.execute(
            select(Document)
            .where(and_(Document.id == document_id, Document.tenant_id == self.tenant_id))
        )
        return result.scalar_one_or_none()

    async def list_by_rfp(self, rfp_id: uuid.UUID) -> List[Document]:
        """List all documents for an RFP."""
        result = await self.session.execute(
            select(Document)
            .where(and_(Document.rfp_id == rfp_id, Document.tenant_id == self.tenant_id))
            .order_by(Document.created_at)
        )
        return list(result.scalars().all())

    async def update_embedding_status(
        self,
        document_id: uuid.UUID,
        status: str,
    ) -> None:
        """Update document embedding status."""
        await self.session.execute(
            update(Document)
            .where(and_(Document.id == document_id, Document.tenant_id == self.tenant_id))
            .values(embedding_status=status)
        )

    async def delete(self, document_id: uuid.UUID) -> bool:
        """Delete a document."""
        result = await self.session.execute(
            delete(Document)
            .where(and_(Document.id == document_id, Document.tenant_id == self.tenant_id))
        )
        return result.rowcount > 0


# ============================================
# Requirement Repository
# ============================================

class RequirementRepository:
    """Repository for requirement operations."""

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID):
        self.session = session
        self.tenant_id = tenant_id

    async def create(
        self,
        rfp_id: uuid.UUID,
        document_id: uuid.UUID,
        text: str,
        section: Optional[str] = None,
        requirement_type: Optional[str] = None,
        priority: Optional[str] = None,
        confidence: Optional[float] = None,
        source_page: Optional[int] = None,
        bbox_coordinates: Optional[dict] = None,
        keywords: Optional[List[str]] = None,
    ) -> Requirement:
        """Create a new requirement."""
        requirement = Requirement(
            tenant_id=self.tenant_id,
            rfp_id=rfp_id,
            document_id=document_id,
            text=text,
            section=section,
            requirement_type=requirement_type,
            priority=priority,
            confidence=confidence,
            source_page=source_page,
            bbox_coordinates=bbox_coordinates,
            keywords=keywords,
        )
        self.session.add(requirement)
        await self.session.flush()
        return requirement

    async def bulk_create(self, requirements: List[dict]) -> List[Requirement]:
        """Bulk create requirements."""
        created = []
        for req_data in requirements:
            req_data["tenant_id"] = self.tenant_id
            requirement = Requirement(**req_data)
            self.session.add(requirement)
            created.append(requirement)
        await self.session.flush()
        return created

    async def get_by_id(self, requirement_id: uuid.UUID) -> Optional[Requirement]:
        """Get requirement by ID."""
        result = await self.session.execute(
            select(Requirement)
            .where(and_(
                Requirement.id == requirement_id,
                Requirement.tenant_id == self.tenant_id
            ))
        )
        return result.scalar_one_or_none()

    async def list_by_rfp(
        self,
        rfp_id: uuid.UUID,
        requirement_type: Optional[str] = None,
        priority: Optional[str] = None,
        section: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Requirement]:
        """List requirements for an RFP with filters."""
        query = select(Requirement).where(and_(
            Requirement.rfp_id == rfp_id,
            Requirement.tenant_id == self.tenant_id
        ))

        if requirement_type:
            query = query.where(Requirement.requirement_type == requirement_type)
        if priority:
            query = query.where(Requirement.priority == priority)
        if section:
            query = query.where(Requirement.section == section)

        query = query.order_by(Requirement.source_page, Requirement.created_at)
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_by_rfp(
        self,
        rfp_id: uuid.UUID,
        requirement_type: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> int:
        """Count requirements for an RFP."""
        query = select(func.count(Requirement.id)).where(and_(
            Requirement.rfp_id == rfp_id,
            Requirement.tenant_id == self.tenant_id
        ))

        if requirement_type:
            query = query.where(Requirement.requirement_type == requirement_type)
        if priority:
            query = query.where(Requirement.priority == priority)

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def search(
        self,
        rfp_id: uuid.UUID,
        query: str,
        limit: int = 50,
    ) -> List[Requirement]:
        """Search requirements by text."""
        result = await self.session.execute(
            select(Requirement)
            .where(and_(
                Requirement.rfp_id == rfp_id,
                Requirement.tenant_id == self.tenant_id,
                Requirement.text.ilike(f"%{query}%")
            ))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_by_rfp(self, rfp_id: uuid.UUID) -> int:
        """Delete all requirements for an RFP."""
        result = await self.session.execute(
            delete(Requirement)
            .where(and_(
                Requirement.rfp_id == rfp_id,
                Requirement.tenant_id == self.tenant_id
            ))
        )
        return result.rowcount


# ============================================
# Library Repository
# ============================================

class LibraryRepository:
    """Repository for Company Library operations."""

    def __init__(self, session: AsyncSession, tenant_id: uuid.UUID):
        self.session = session
        self.tenant_id = tenant_id

    async def create_document(
        self,
        filename: str,
        s3_path: str,
        doc_type: Optional[str] = None,
        file_size: Optional[int] = None,
    ) -> LibraryDocument:
        """Create a new library document."""
        document = LibraryDocument(
            tenant_id=self.tenant_id,
            filename=filename,
            s3_path=s3_path,
            doc_type=doc_type,
            file_size=file_size,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_document_by_id(self, document_id: uuid.UUID) -> Optional[LibraryDocument]:
        """Get library document by ID."""
        result = await self.session.execute(
            select(LibraryDocument)
            .options(selectinload(LibraryDocument.entities))
            .where(and_(
                LibraryDocument.id == document_id,
                LibraryDocument.tenant_id == self.tenant_id
            ))
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        doc_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LibraryDocument]:
        """List library documents."""
        query = select(LibraryDocument).where(
            LibraryDocument.tenant_id == self.tenant_id
        )

        if doc_type:
            query = query.where(LibraryDocument.doc_type == doc_type)

        query = query.order_by(LibraryDocument.created_at.desc())
        query = query.limit(limit).offset(offset)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create_entity(
        self,
        document_id: uuid.UUID,
        entity_type: str,
        extracted_data: dict,
        confidence: Optional[float] = None,
    ) -> LibraryEntity:
        """Create an extracted entity."""
        entity = LibraryEntity(
            tenant_id=self.tenant_id,
            document_id=document_id,
            entity_type=entity_type,
            extracted_data=extracted_data,
            confidence=confidence,
        )
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def create_embedding(
        self,
        document_id: uuid.UUID,
        chunk_text: str,
        embedding: List[float],
        chunk_index: Optional[int] = None,
        entity_id: Optional[uuid.UUID] = None,
        metadata: Optional[dict] = None,
    ) -> LibraryEmbedding:
        """Create an embedding record."""
        emb = LibraryEmbedding(
            tenant_id=self.tenant_id,
            document_id=document_id,
            entity_id=entity_id,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.session.add(emb)
        await self.session.flush()
        return emb

    async def search_embeddings(
        self,
        query_embedding: List[float],
        limit: int = 10,
        entity_type: Optional[str] = None,
    ) -> List[tuple]:
        """
        Search embeddings by vector similarity.
        Returns list of (LibraryEmbedding, similarity_score) tuples.
        """
        from sqlalchemy import text

        # Build the query with cosine similarity
        query = f"""
            SELECT
                le.id,
                le.chunk_text,
                le.metadata,
                ld.filename,
                ld.doc_type,
                1 - (le.embedding <=> :query_vec) as similarity
            FROM library_embeddings le
            JOIN library_documents ld ON le.document_id = ld.id
            WHERE le.tenant_id = :tenant_id
        """

        if entity_type:
            query += " AND ld.doc_type = :entity_type"

        query += " ORDER BY le.embedding <=> :query_vec LIMIT :limit"

        params = {
            "query_vec": str(query_embedding),
            "tenant_id": str(self.tenant_id),
            "limit": limit,
        }
        if entity_type:
            params["entity_type"] = entity_type

        result = await self.session.execute(text(query), params)
        return list(result.fetchall())

    async def delete_document(self, document_id: uuid.UUID) -> bool:
        """Delete a library document and all related data."""
        result = await self.session.execute(
            delete(LibraryDocument)
            .where(and_(
                LibraryDocument.id == document_id,
                LibraryDocument.tenant_id == self.tenant_id
            ))
        )
        return result.rowcount > 0
