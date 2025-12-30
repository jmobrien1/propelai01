"""
PropelAI Library API Endpoints
RAG-powered Company Library for resumes, past performance, and capabilities
"""

import os
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from auth.dependencies import get_current_user, get_current_tenant
from rag.search import RAGSearch, SearchQuery, SearchMode, SearchResult, get_rag_search
from rag.ingestion import (
    LibraryIngestionPipeline,
    EntityType,
    IngestionResult,
    IngestionStatus,
    get_ingestion_pipeline,
)


router = APIRouter(prefix="/api/library", tags=["library"])


# ============================================================================
# Request/Response Models
# ============================================================================


class SearchRequest(BaseModel):
    """Library search request."""
    query: str = Field(..., description="Search query text")
    mode: str = Field(default="hybrid", description="Search mode: similarity, hybrid, keyword, reranked")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    min_score: float = Field(default=0.5, ge=0, le=1, description="Minimum relevance score")
    entity_types: Optional[List[str]] = Field(default=None, description="Filter by entity types")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter by document IDs")


class SearchResultResponse(BaseModel):
    """Single search result."""
    chunk_id: str
    text: str
    score: float
    document_id: Optional[str]
    document_name: Optional[str]
    page_number: Optional[int]
    section: Optional[str]
    entity_type: Optional[str]
    entity_name: Optional[str]


class SearchResponse(BaseModel):
    """Library search response."""
    query: str
    mode: str
    total_results: int
    results: List[SearchResultResponse]
    search_time_ms: int


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    filename: str
    entity_type: str
    entity_name: str
    chunk_count: int
    status: str
    message: str


class DocumentMetadata(BaseModel):
    """Document metadata."""
    id: str
    filename: str
    entity_type: str
    entity_name: str
    chunk_count: int
    created_at: datetime
    extracted_data: dict


class LibraryStatsResponse(BaseModel):
    """Library statistics."""
    total_documents: int
    total_chunks: int
    documents_by_type: dict
    recent_uploads: List[DocumentMetadata]


class RequirementMatchRequest(BaseModel):
    """Request to find library content for a requirement."""
    requirement_text: str
    entity_types: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class RequirementMatchResponse(BaseModel):
    """Matching library content for a requirement."""
    requirement_text: str
    matches: List[SearchResultResponse]
    suggested_response: Optional[str] = None


# ============================================================================
# Dependencies
# ============================================================================


async def get_search_service() -> RAGSearch:
    """Get RAG search service."""
    # In production, this would use the database pool
    return get_rag_search()


async def get_ingestion_service() -> LibraryIngestionPipeline:
    """Get ingestion pipeline service."""
    return get_ingestion_pipeline()


# ============================================================================
# Search Endpoints
# ============================================================================


@router.post("/search", response_model=SearchResponse)
async def search_library(
    request: SearchRequest,
    tenant_id: str = Depends(get_current_tenant),
    search_service: RAGSearch = Depends(get_search_service),
):
    """
    Search the Company Library using semantic similarity.

    Supports multiple search modes:
    - **similarity**: Pure vector similarity search
    - **hybrid**: Combines vector and keyword search (recommended)
    - **keyword**: Traditional keyword matching
    - **reranked**: Vector search with cross-encoder reranking (most accurate)
    """
    start_time = datetime.utcnow()

    try:
        search_mode = SearchMode(request.mode)
    except ValueError:
        search_mode = SearchMode.HYBRID

    query = SearchQuery(
        query=request.query,
        mode=search_mode,
        top_k=request.top_k,
        min_score=request.min_score,
        document_ids=request.document_ids,
        entity_types=request.entity_types,
    )

    results = await search_service.search(query, tenant_id)

    search_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_results=len(results),
        results=[
            SearchResultResponse(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                document_id=r.document_id,
                document_name=r.document_name,
                page_number=r.page_number,
                section=r.section,
                entity_type=r.entity_type,
                entity_name=r.entity_name,
            )
            for r in results
        ],
        search_time_ms=search_time_ms,
    )


@router.get("/search/quick")
async def quick_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=5, ge=1, le=20),
    tenant_id: str = Depends(get_current_tenant),
    search_service: RAGSearch = Depends(get_search_service),
):
    """Quick search endpoint for autocomplete and suggestions."""
    query = SearchQuery(
        query=q,
        mode=SearchMode.SIMILARITY,
        top_k=limit,
        min_score=0.4,
    )

    results = await search_service.search(query, tenant_id)

    return {
        "query": q,
        "suggestions": [
            {
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "document": r.document_name,
                "entity": r.entity_name,
                "score": round(r.score, 3),
            }
            for r in results
        ],
    }


@router.post("/match-requirement", response_model=RequirementMatchResponse)
async def match_requirement_to_library(
    request: RequirementMatchRequest,
    tenant_id: str = Depends(get_current_tenant),
    search_service: RAGSearch = Depends(get_search_service),
):
    """
    Find library content that matches a proposal requirement.
    Used by the proposal generation engine to pull relevant content.
    """
    results = await search_service.search_for_requirement(
        requirement_text=request.requirement_text,
        tenant_id=tenant_id,
        entity_types=request.entity_types,
    )

    # Limit to top_k
    results = results[:request.top_k]

    return RequirementMatchResponse(
        requirement_text=request.requirement_text,
        matches=[
            SearchResultResponse(
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                document_id=r.document_id,
                document_name=r.document_name,
                page_number=r.page_number,
                section=r.section,
                entity_type=r.entity_type,
                entity_name=r.entity_name,
            )
            for r in results
        ],
    )


# ============================================================================
# Document Management Endpoints
# ============================================================================


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    entity_type: str = Form(..., description="Type: resume, past_performance, capability, etc."),
    entity_name: Optional[str] = Form(default=None, description="Name for the entity"),
    tenant_id: str = Depends(get_current_tenant),
    ingestion_service: LibraryIngestionPipeline = Depends(get_ingestion_service),
):
    """
    Upload a document to the Company Library.

    Supported entity types:
    - **resume**: Personnel resumes
    - **past_performance**: Past performance narratives
    - **capability**: Capability statements
    - **template**: Reusable templates
    - **boilerplate**: Standard boilerplate text
    - **policy**: Company policies
    - **reference**: Reference documents
    """
    # Validate entity type
    try:
        etype = EntityType(entity_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entity type. Must be one of: {[e.value for e in EntityType]}",
        )

    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}",
        )

    # Save file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Ingest document
        result = await ingestion_service.ingest_document(
            file_path=tmp_path,
            tenant_id=tenant_id,
            entity_type=etype,
            entity_name=entity_name,
        )

        if result.status == IngestionStatus.FAILED:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {', '.join(result.errors)}",
            )

        return DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            entity_type=result.entity_type.value,
            entity_name=result.entity_name,
            chunk_count=result.chunk_count,
            status=result.status.value,
            message=f"Successfully ingested {result.chunk_count} chunks",
        )

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents(
    entity_type: Optional[str] = Query(default=None, description="Filter by entity type"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    tenant_id: str = Depends(get_current_tenant),
):
    """List documents in the library."""
    # In production, this would query the database
    # For now, return mock data
    return [
        DocumentMetadata(
            id="doc-1",
            filename="john_smith_resume.pdf",
            entity_type="resume",
            entity_name="John Smith",
            chunk_count=12,
            created_at=datetime.utcnow(),
            extracted_data={
                "skills": ["Python", "AWS", "FastAPI"],
                "years_experience": 10,
            },
        ),
        DocumentMetadata(
            id="doc-2",
            filename="cms_medicare_modernization.pdf",
            entity_type="past_performance",
            entity_name="CMS Medicare Modernization",
            chunk_count=25,
            created_at=datetime.utcnow(),
            extracted_data={
                "agency": "Department of Health and Human Services",
                "contract_value": 15000000,
            },
        ),
    ]


@router.get("/documents/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
):
    """Get document details."""
    # In production, query the database
    return DocumentMetadata(
        id=document_id,
        filename="example.pdf",
        entity_type="resume",
        entity_name="Example Document",
        chunk_count=10,
        created_at=datetime.utcnow(),
        extracted_data={},
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
    ingestion_service: LibraryIngestionPipeline = Depends(get_ingestion_service),
):
    """Delete a document from the library."""
    success = await ingestion_service.delete_document(document_id, tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "deleted", "document_id": document_id}


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    tenant_id: str = Depends(get_current_tenant),
    ingestion_service: LibraryIngestionPipeline = Depends(get_ingestion_service),
):
    """Re-generate embeddings for a document."""
    success = await ingestion_service.update_embeddings(document_id, tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "reindexed", "document_id": document_id}


# ============================================================================
# Library Statistics
# ============================================================================


@router.get("/stats", response_model=LibraryStatsResponse)
async def get_library_stats(
    tenant_id: str = Depends(get_current_tenant),
):
    """Get library statistics for the tenant."""
    # In production, aggregate from database
    return LibraryStatsResponse(
        total_documents=150,
        total_chunks=3500,
        documents_by_type={
            "resume": 75,
            "past_performance": 40,
            "capability": 20,
            "boilerplate": 15,
        },
        recent_uploads=[],
    )


# ============================================================================
# Entity-Specific Endpoints
# ============================================================================


@router.get("/resumes")
async def list_resumes(
    skill: Optional[str] = Query(default=None, description="Filter by skill"),
    certification: Optional[str] = Query(default=None, description="Filter by certification"),
    min_years: Optional[int] = Query(default=None, description="Minimum years experience"),
    limit: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_current_tenant),
):
    """List resumes with optional filters."""
    # In production, query the database with filters
    return {
        "total": 75,
        "resumes": [
            {
                "id": "doc-1",
                "name": "John Smith",
                "skills": ["Python", "AWS", "FastAPI", "PostgreSQL"],
                "certifications": ["AWS Solutions Architect", "PMP"],
                "years_experience": 10,
            },
        ],
    }


@router.get("/past-performance")
async def list_past_performance(
    agency: Optional[str] = Query(default=None, description="Filter by agency"),
    naics: Optional[str] = Query(default=None, description="Filter by NAICS code"),
    min_value: Optional[float] = Query(default=None, description="Minimum contract value"),
    limit: int = Query(default=20, ge=1, le=100),
    tenant_id: str = Depends(get_current_tenant),
):
    """List past performance records with optional filters."""
    return {
        "total": 40,
        "records": [
            {
                "id": "doc-2",
                "name": "CMS Medicare Modernization",
                "agency": "HHS",
                "contract_value": 15000000,
                "naics_codes": ["541511", "541512"],
                "contract_type": "IDIQ",
            },
        ],
    }


@router.get("/similar/{document_id}")
async def find_similar_documents(
    document_id: str,
    limit: int = Query(default=5, ge=1, le=20),
    tenant_id: str = Depends(get_current_tenant),
    search_service: RAGSearch = Depends(get_search_service),
):
    """Find documents similar to the specified document."""
    # First get the document's entity type
    # In production, query the database
    entity_type = "resume"  # Mock

    results = await search_service.find_similar_entities(
        entity_id=document_id,
        entity_type=entity_type,
        tenant_id=tenant_id,
        top_k=limit,
    )

    return {
        "source_document_id": document_id,
        "similar_documents": [
            {
                "document_id": r.document_id,
                "document_name": r.document_name,
                "entity_name": r.entity_name,
                "similarity_score": round(r.score, 3),
            }
            for r in results
        ],
    }
