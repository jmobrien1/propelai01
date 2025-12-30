"""
PropelAI Trust Gate API
Source traceability endpoints for click-to-verify functionality
"""

import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.middleware import get_current_user, AuthenticatedUser
from database.connection import get_db
from database.repositories import RFPRepository, RequirementRepository, DocumentRepository
from storage.client import get_storage_client, BaseStorageClient
from parsing.document_parser import DocumentParser


router = APIRouter(prefix="/api/trust-gate", tags=["Trust Gate"])


# ============================================
# Request/Response Models
# ============================================

class SourceLocation(BaseModel):
    """Source location for a requirement in the PDF."""
    found: bool
    page: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    matched_text: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    confidence: float = 0.0
    document_id: Optional[str] = None
    document_name: Optional[str] = None


class RequirementSourceRequest(BaseModel):
    """Request to find source of a requirement."""
    requirement_text: str = Field(..., min_length=10)


class HighlightRegion(BaseModel):
    """Region to highlight in the PDF viewer."""
    page: int
    x: float
    y: float
    width: float
    height: float
    color: str = "#FFFF00"  # Yellow highlight
    opacity: float = 0.3


class PDFViewerData(BaseModel):
    """Data for the PDF viewer component."""
    document_id: str
    document_name: str
    pdf_url: str
    total_pages: int
    highlights: List[HighlightRegion] = []
    initial_page: int = 1


class VerificationResult(BaseModel):
    """Result of source verification."""
    verified: bool
    requirement_id: str
    source: SourceLocation
    viewer_data: Optional[PDFViewerData] = None


# ============================================
# Endpoints
# ============================================

@router.get("/rfp/{rfp_id}/requirements/{requirement_id}/source")
async def get_requirement_source(
    rfp_id: str,
    requirement_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: BaseStorageClient = Depends(get_storage_client),
) -> SourceLocation:
    """
    Get the source location of a requirement in the original PDF.
    Returns bounding box coordinates for click-to-verify highlighting.
    """
    tenant_id = uuid.UUID(user.tenant_id)

    # Get the requirement
    req_repo = RequirementRepository(db, tenant_id)
    requirement = await req_repo.get_by_id(uuid.UUID(requirement_id))

    if not requirement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requirement not found"
        )

    # Check if we have stored bbox coordinates
    if requirement.bbox_coordinates:
        return SourceLocation(
            found=True,
            page=requirement.bbox_coordinates.get("page", requirement.source_page),
            bbox=requirement.bbox_coordinates,
            matched_text=requirement.source_snippet or requirement.text[:200],
            confidence=requirement.confidence or 1.0,
            document_id=str(requirement.document_id),
        )

    # If no stored bbox, try to find it in the document
    doc_repo = DocumentRepository(db, tenant_id)
    document = await doc_repo.get_by_id(requirement.document_id)

    if not document:
        return SourceLocation(
            found=False,
            page=requirement.source_page,
            confidence=0.0,
        )

    # Download and parse the document to find the source
    try:
        pdf_bytes = await storage.download_file(document.s3_path)
        if not pdf_bytes:
            return SourceLocation(found=False, confidence=0.0)

        parser = DocumentParser()
        parsed = parser.parse_bytes(pdf_bytes, document.filename)

        # Find the requirement text in the document
        source = parsed.find_text_source(requirement.text)

        if source and source.get("found"):
            # Update the requirement with the found coordinates
            # (In production, you'd save this back to the database)
            return SourceLocation(
                found=True,
                page=source.get("page"),
                bbox=source.get("bbox"),
                matched_text=source.get("matched_text"),
                context_before=source.get("context_before"),
                context_after=source.get("context_after"),
                confidence=source.get("confidence", 0.0),
                document_id=str(document.id),
                document_name=document.filename,
            )

    except Exception as e:
        # Log the error but return a graceful response
        pass

    return SourceLocation(
        found=False,
        page=requirement.source_page,
        confidence=0.0,
        document_id=str(document.id) if document else None,
    )


@router.post("/rfp/{rfp_id}/find-source")
async def find_text_source(
    rfp_id: str,
    request: RequirementSourceRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: BaseStorageClient = Depends(get_storage_client),
) -> SourceLocation:
    """
    Find the source location of arbitrary text in the RFP documents.
    Useful for verifying extracted requirements.
    """
    tenant_id = uuid.UUID(user.tenant_id)

    # Get the RFP and its documents
    rfp_repo = RFPRepository(db, tenant_id)
    rfp = await rfp_repo.get_by_id(uuid.UUID(rfp_id))

    if not rfp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="RFP not found"
        )

    doc_repo = DocumentRepository(db, tenant_id)
    documents = await doc_repo.list_by_rfp(uuid.UUID(rfp_id))

    parser = DocumentParser()

    # Search through each document
    for document in documents:
        if not document.filename.lower().endswith('.pdf'):
            continue

        try:
            pdf_bytes = await storage.download_file(document.s3_path)
            if not pdf_bytes:
                continue

            parsed = parser.parse_bytes(pdf_bytes, document.filename)
            source = parsed.find_text_source(request.requirement_text)

            if source and source.get("found") and source.get("confidence", 0) > 0.7:
                return SourceLocation(
                    found=True,
                    page=source.get("page"),
                    bbox=source.get("bbox"),
                    matched_text=source.get("matched_text"),
                    context_before=source.get("context_before"),
                    context_after=source.get("context_after"),
                    confidence=source.get("confidence", 0.0),
                    document_id=str(document.id),
                    document_name=document.filename,
                )

        except Exception:
            continue

    return SourceLocation(found=False, confidence=0.0)


@router.get("/rfp/{rfp_id}/requirements/{requirement_id}/verify")
async def verify_requirement(
    rfp_id: str,
    requirement_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: BaseStorageClient = Depends(get_storage_client),
) -> VerificationResult:
    """
    Get full verification data for a requirement, including PDF viewer setup.
    Returns everything needed to display the click-to-verify UI.
    """
    tenant_id = uuid.UUID(user.tenant_id)

    # Get requirement and its source location
    req_repo = RequirementRepository(db, tenant_id)
    requirement = await req_repo.get_by_id(uuid.UUID(requirement_id))

    if not requirement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requirement not found"
        )

    # Get the source document
    doc_repo = DocumentRepository(db, tenant_id)
    document = await doc_repo.get_by_id(requirement.document_id)

    if not document:
        return VerificationResult(
            verified=False,
            requirement_id=requirement_id,
            source=SourceLocation(found=False, confidence=0.0),
        )

    # Get presigned URL for the PDF
    pdf_url = await storage.get_presigned_url(document.s3_path, expires_in=3600)

    # Get or find the source location
    source = await get_requirement_source(
        rfp_id, requirement_id, user, db, storage
    )

    # Build viewer data
    viewer_data = None
    if source.found and pdf_url:
        highlights = []
        if source.bbox:
            highlights.append(HighlightRegion(
                page=source.page or 1,
                x=source.bbox.get("x", 0),
                y=source.bbox.get("y", 0),
                width=source.bbox.get("width", 100),
                height=source.bbox.get("height", 20),
            ))

        viewer_data = PDFViewerData(
            document_id=str(document.id),
            document_name=document.filename,
            pdf_url=pdf_url,
            total_pages=document.page_count or 1,
            highlights=highlights,
            initial_page=source.page or 1,
        )

    return VerificationResult(
        verified=source.found and source.confidence > 0.8,
        requirement_id=requirement_id,
        source=source,
        viewer_data=viewer_data,
    )


@router.get("/rfp/{rfp_id}/documents/{document_id}/page/{page_number}")
async def get_page_content(
    rfp_id: str,
    document_id: str,
    page_number: int,
    user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: BaseStorageClient = Depends(get_storage_client),
) -> Dict[str, Any]:
    """
    Get the content of a specific page with all text blocks and their locations.
    Used for rendering overlays on the PDF viewer.
    """
    tenant_id = uuid.UUID(user.tenant_id)

    doc_repo = DocumentRepository(db, tenant_id)
    document = await doc_repo.get_by_id(uuid.UUID(document_id))

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        pdf_bytes = await storage.download_file(document.s3_path)
        if not pdf_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document file not found"
            )

        parser = DocumentParser()
        parsed = parser.parse_bytes(pdf_bytes, document.filename)

        if parsed._pdf_document:
            page = parsed._pdf_document.get_page(page_number)
            if page:
                return {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "blocks": [
                        {
                            "text": block.text,
                            "bbox": block.bbox.to_dict(),
                            "type": block.block_type,
                        }
                        for block in page.blocks
                    ],
                    "full_text": page.full_text,
                }

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page_number} not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse document: {str(e)}"
        )


@router.get("/rfp/{rfp_id}/documents/{document_id}/search")
async def search_document(
    rfp_id: str,
    document_id: str,
    q: str = Query(..., min_length=3, description="Search query"),
    user: AuthenticatedUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    storage: BaseStorageClient = Depends(get_storage_client),
) -> List[SourceLocation]:
    """
    Search for text within a specific document.
    Returns all matching locations with bounding boxes.
    """
    tenant_id = uuid.UUID(user.tenant_id)

    doc_repo = DocumentRepository(db, tenant_id)
    document = await doc_repo.get_by_id(uuid.UUID(document_id))

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        pdf_bytes = await storage.download_file(document.s3_path)
        if not pdf_bytes:
            return []

        parser = DocumentParser()
        parsed = parser.parse_bytes(pdf_bytes, document.filename)

        results = []
        for ext in parsed.search(q):
            results.append(SourceLocation(
                found=True,
                page=ext.page,
                bbox=ext.bbox,
                matched_text=ext.text[:500],
                context_before=ext.context_before,
                context_after=ext.context_after,
                confidence=1.0,
                document_id=str(document.id),
                document_name=document.filename,
            ))

        return results[:50]  # Limit to 50 results

    except Exception:
        return []
