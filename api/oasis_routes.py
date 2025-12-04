"""
OASIS+ API Routes
=================

FastAPI router for GSA OASIS+ proposal management endpoints.

Endpoints:
- POST /api/oasis/proposals - Create new OASIS+ proposal
- GET /api/oasis/proposals - List all proposals
- GET /api/oasis/proposals/{id} - Get proposal details
- POST /api/oasis/proposals/{id}/jp1 - Upload J.P-1 Matrix
- POST /api/oasis/proposals/{id}/projects - Add project to library
- POST /api/oasis/proposals/{id}/projects/{pid}/documents - Upload project documents
- POST /api/oasis/proposals/{id}/score - Score projects for domain
- POST /api/oasis/proposals/{id}/optimize - Optimize project selection
- POST /api/oasis/proposals/{id}/artifacts - Generate submission artifacts
- GET /api/oasis/proposals/{id}/scorecard - Get domain scorecard
- GET /api/oasis/proposals/{id}/export - Export Symphony bundle
"""

import os
import sys
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from decimal import Decimal

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import OASIS+ components
try:
    from agents.oasis_plus import (
        OASISOrchestrator,
        OASISProposal,
        Project,
        DomainType,
        BusinessSize,
        OptimizationConstraints,
        ContractType,
    )
    OASIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OASIS+ module not available: {e}")
    OASIS_AVAILABLE = False


# ============== Configuration ==============

OASIS_UPLOAD_DIR = Path(tempfile.gettempdir()) / "propelai_oasis"
OASIS_UPLOAD_DIR.mkdir(exist_ok=True)

OASIS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "propelai_oasis_output"
OASIS_OUTPUT_DIR.mkdir(exist_ok=True)


# ============== Pydantic Models ==============

class ProposalCreate(BaseModel):
    """Request to create a new OASIS+ proposal"""
    contractor_name: str
    contractor_cage: str = ""
    business_size: str = "unrestricted"  # unrestricted, small_business, wosb, etc.
    target_domains: List[str] = []  # Domain type values


class ProposalResponse(BaseModel):
    """Response with proposal details"""
    proposal_id: str
    contractor_name: str
    contractor_cage: str
    business_size: str
    status: str
    domains_loaded: int
    projects_count: int
    created_at: str
    updated_at: str


class ProjectCreate(BaseModel):
    """Request to add a project"""
    title: str
    client_agency: str
    contract_number: str
    task_order_number: Optional[str] = None
    naics_code: str = ""
    psc_code: str = ""
    start_date: str  # YYYY-MM-DD
    end_date: Optional[str] = None
    total_obligated_amount: float = 0.0
    contract_type: str = "ffp"
    is_prime: bool = True
    is_oconus: bool = False
    clearance_level: Optional[str] = None
    scope_description: str = ""


class ProjectResponse(BaseModel):
    """Response with project details"""
    project_id: str
    title: str
    client_agency: str
    contract_number: str
    average_annual_value: float
    documents_count: int
    claims_count: int


class ScoreRequest(BaseModel):
    """Request to score projects"""
    domain: str  # Domain type value
    project_ids: Optional[List[str]] = None  # If None, score all


class OptimizeRequest(BaseModel):
    """Request to optimize project selection"""
    domain: str
    max_projects: int = 5
    target_margin: int = 5
    min_confidence: float = 0.6


class OptimizationResponse(BaseModel):
    """Response with optimization results"""
    domain: str
    total_score: int
    threshold: int
    margin: int
    meets_threshold: bool
    risk_level: str
    selected_projects: List[str]
    risk_factors: List[str]


class ScorecardResponse(BaseModel):
    """Domain scorecard response"""
    domain: str
    business_size: str
    total_score: int
    verified_score: int
    threshold: int
    margin: int
    meets_threshold: bool
    has_safe_margin: bool
    qualifying_projects: List[Dict[str, Any]]
    claims: List[Dict[str, Any]]


# ============== Router Setup ==============

router = APIRouter(prefix="/api/oasis", tags=["OASIS+"])

# Global orchestrator instance
orchestrator: Optional[OASISOrchestrator] = None


def get_orchestrator() -> OASISOrchestrator:
    """Get or create the orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        if not OASIS_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="OASIS+ module not available"
            )
        orchestrator = OASISOrchestrator()
    return orchestrator


# ============== Endpoints ==============

@router.get("/health")
async def oasis_health():
    """Check OASIS+ module availability"""
    return {
        "available": OASIS_AVAILABLE,
        "version": "1.0.0",
        "features": {
            "jp1_parser": True,
            "evidence_hunter": True,
            "optimizer": True,
            "pdf_tagger": True,
            "form_generator": True,
        } if OASIS_AVAILABLE else {}
    }


@router.post("/proposals", response_model=ProposalResponse)
async def create_proposal(request: ProposalCreate):
    """Create a new OASIS+ proposal"""
    orch = get_orchestrator()

    # Parse business size
    try:
        business_size = BusinessSize(request.business_size)
    except ValueError:
        business_size = BusinessSize.UNRESTRICTED

    # Parse target domains
    target_domains = []
    for domain_str in request.target_domains:
        try:
            target_domains.append(DomainType(domain_str))
        except ValueError:
            pass

    proposal = orch.create_proposal(
        contractor_name=request.contractor_name,
        contractor_cage=request.contractor_cage,
        business_size=business_size,
        target_domains=target_domains or None,
    )

    # Create upload directory for this proposal
    proposal_dir = OASIS_UPLOAD_DIR / proposal.proposal_id
    proposal_dir.mkdir(exist_ok=True)

    return ProposalResponse(
        proposal_id=proposal.proposal_id,
        contractor_name=proposal.contractor_name,
        contractor_cage=proposal.contractor_cage,
        business_size=proposal.business_size.value,
        status=proposal.status,
        domains_loaded=len(proposal.domains),
        projects_count=len(proposal.projects),
        created_at=proposal.created_at.isoformat(),
        updated_at=proposal.updated_at.isoformat(),
    )


@router.get("/proposals")
async def list_proposals():
    """List all OASIS+ proposals"""
    orch = get_orchestrator()

    proposals = []
    for proposal in orch.proposals.values():
        proposals.append({
            "proposal_id": proposal.proposal_id,
            "contractor_name": proposal.contractor_name,
            "business_size": proposal.business_size.value,
            "status": proposal.status,
            "domains_loaded": len(proposal.domains),
            "projects_count": len(proposal.projects),
            "created_at": proposal.created_at.isoformat(),
        })

    return {"proposals": proposals, "count": len(proposals)}


@router.get("/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get proposal details and summary"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    return orch.get_summary(proposal_id)


@router.post("/proposals/{proposal_id}/jp1")
async def upload_jp1_matrix(
    proposal_id: str,
    file: UploadFile = File(...),
):
    """Upload and parse J.P-1 Qualifications Matrix"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # Save uploaded file
    proposal_dir = OASIS_UPLOAD_DIR / proposal_id
    proposal_dir.mkdir(exist_ok=True)

    jp1_path = proposal_dir / f"JP1_{file.filename}"
    with open(jp1_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        domains = orch.load_jp1_matrix(proposal_id, str(jp1_path))

        return {
            "success": True,
            "domains_loaded": len(domains),
            "domains": [
                {
                    "type": dt.value,
                    "name": d.name,
                    "criteria_count": len(d.criteria),
                    "threshold_unrestricted": d.unrestricted_threshold,
                    "threshold_small_business": d.small_business_threshold,
                }
                for dt, d in domains.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse J.P-1: {e}")


@router.post("/proposals/{proposal_id}/projects", response_model=ProjectResponse)
async def add_project(proposal_id: str, request: ProjectCreate):
    """Add a project to the proposal library"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date() if request.end_date else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Parse contract type
    try:
        contract_type = ContractType(request.contract_type)
    except ValueError:
        contract_type = ContractType.FIRM_FIXED_PRICE

    project_id = f"PROJ-{uuid.uuid4().hex[:8].upper()}"

    project = Project(
        project_id=project_id,
        title=request.title,
        client_agency=request.client_agency,
        contract_number=request.contract_number,
        task_order_number=request.task_order_number,
        naics_code=request.naics_code,
        psc_code=request.psc_code,
        start_date=start_date,
        end_date=end_date,
        total_obligated_amount=Decimal(str(request.total_obligated_amount)),
        contract_type=contract_type,
        is_prime=request.is_prime,
        is_oconus=request.is_oconus,
        clearance_level=request.clearance_level,
        scope_description=request.scope_description,
    )

    oasis_project = orch.add_project(proposal_id, project)

    # Calculate AAV
    aav = project.calculate_aav()

    return ProjectResponse(
        project_id=project_id,
        title=project.title,
        client_agency=project.client_agency,
        contract_number=project.contract_number,
        average_annual_value=float(aav),
        documents_count=len(oasis_project.documents),
        claims_count=len(oasis_project.claims),
    )


@router.get("/proposals/{proposal_id}/projects")
async def list_projects(proposal_id: str):
    """List all projects in proposal library"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    projects = []
    for oasis_project in proposal.projects.values():
        p = oasis_project.project
        projects.append({
            "project_id": p.project_id,
            "title": p.title,
            "client_agency": p.client_agency,
            "contract_number": p.contract_number,
            "average_annual_value": float(p.calculate_aav()),
            "documents_count": len(oasis_project.documents),
            "claims_count": len(oasis_project.claims),
            "score_by_domain": {
                d.value: s for d, s in oasis_project.score_by_domain.items()
            },
        })

    return {"projects": projects, "count": len(projects)}


@router.post("/proposals/{proposal_id}/projects/{project_id}/documents")
async def upload_project_documents(
    proposal_id: str,
    project_id: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload documents for a project and ingest them"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if project_id not in proposal.projects:
        raise HTTPException(status_code=404, detail="Project not found")

    # Save uploaded files
    project_dir = OASIS_UPLOAD_DIR / proposal_id / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        file_path = project_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_paths.append(str(file_path))

        # Track document in project
        proposal.projects[project_id].documents.append({
            "filename": file.filename,
            "path": str(file_path),
            "uploaded_at": datetime.now().isoformat(),
        })

    # Ingest documents
    try:
        chunks = orch.ingest_project_documents(
            proposal_id, project_id, saved_paths
        )

        return {
            "success": True,
            "files_uploaded": len(saved_paths),
            "chunks_created": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {e}")


@router.post("/proposals/{proposal_id}/score")
async def score_projects(proposal_id: str, request: ScoreRequest):
    """Score projects against domain criteria"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # Parse domain
    try:
        domain = DomainType(request.domain)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {request.domain}")

    if domain not in proposal.domains:
        raise HTTPException(status_code=400, detail="Domain not loaded. Upload J.P-1 first.")

    try:
        if request.project_ids:
            # Score specific projects
            all_claims = {}
            for pid in request.project_ids:
                claims = orch.auto_score_project(proposal_id, pid, domain)
                all_claims[pid] = claims
        else:
            # Score all projects
            all_claims = orch.score_all_projects(proposal_id, domain)

        return {
            "success": True,
            "domain": domain.value,
            "projects_scored": len(all_claims),
            "results": {
                pid: {
                    "claims_count": len(claims),
                    "total_points": sum(c.claimed_points for c in claims),
                }
                for pid, claims in all_claims.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}")


@router.post("/proposals/{proposal_id}/optimize", response_model=OptimizationResponse)
async def optimize_selection(proposal_id: str, request: OptimizeRequest):
    """Optimize project selection for a domain"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # Parse domain
    try:
        domain = DomainType(request.domain)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {request.domain}")

    constraints = OptimizationConstraints(
        max_qualifying_projects=request.max_projects,
        target_margin=request.target_margin,
        min_confidence_score=request.min_confidence,
    )

    try:
        result = orch.optimize_domain(proposal_id, domain, constraints)

        return OptimizationResponse(
            domain=domain.value,
            total_score=result.total_score,
            threshold=result.threshold,
            margin=result.margin,
            meets_threshold=result.meets_threshold,
            risk_level=result.overall_risk,
            selected_projects=[p.project_id for p in result.selected_projects],
            risk_factors=result.risk_factors,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Optimization failed: {e}")


@router.get("/proposals/{proposal_id}/scorecard/{domain}")
async def get_scorecard(proposal_id: str, domain: str):
    """Get the scorecard for a domain"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    try:
        domain_type = DomainType(domain)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")

    scorecard = orch.get_scorecard(proposal_id, domain_type)
    if not scorecard:
        raise HTTPException(status_code=404, detail="Scorecard not available. Run optimization first.")

    return {
        "domain": scorecard.domain.value,
        "business_size": scorecard.business_size.value,
        "total_score": scorecard.total_score,
        "verified_points": scorecard.verified_points,
        "pending_points": scorecard.pending_points,
        "unverified_points": scorecard.unverified_points,
        "threshold": scorecard.threshold,
        "margin": scorecard.margin,
        "meets_threshold": scorecard.meets_threshold,
        "has_safe_margin": scorecard.has_safe_margin,
        "qualifying_projects": [
            {
                "project_id": p.project_id,
                "title": p.title,
                "aav": float(p.calculate_aav()),
            }
            for p in scorecard.qualifying_projects
        ],
        "claims_count": len(scorecard.claims),
        "at_risk_claims": len(scorecard.at_risk_claims),
    }


@router.post("/proposals/{proposal_id}/artifacts")
async def generate_artifacts(
    proposal_id: str,
    domain: str = Form(...),
    generate_pdfs: bool = Form(True),
    generate_jp3: bool = Form(True),
):
    """Generate Symphony submission artifacts"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    try:
        domain_type = DomainType(domain)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")

    # Create output directory
    output_dir = OASIS_OUTPUT_DIR / proposal_id / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": domain,
        "output_dir": str(output_dir),
        "tagged_pdfs": [],
        "jp3_forms": [],
    }

    try:
        if generate_pdfs:
            tagged = orch.generate_tagged_pdfs(proposal_id, domain_type, str(output_dir))
            results["tagged_pdfs"] = [
                {
                    "output_path": t.output_path,
                    "annotations_added": t.annotations_added,
                    "claims_tagged": len(t.claims_tagged),
                }
                for t in tagged
            ]

        if generate_jp3:
            jp3s = orch.generate_jp3_forms(proposal_id, domain_type, str(output_dir))
            results["jp3_forms"] = [
                {
                    "output_path": j.output_path,
                    "project_title": j.project_title,
                    "success": j.success,
                }
                for j in jp3s
            ]

        results["success"] = True
        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Artifact generation failed: {e}")


@router.get("/proposals/{proposal_id}/export")
async def export_symphony_bundle(proposal_id: str, domain: str):
    """Export complete Symphony submission bundle as ZIP"""
    orch = get_orchestrator()

    proposal = orch.get_proposal(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    try:
        domain_type = DomainType(domain)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")

    # Check if artifacts exist
    output_dir = OASIS_OUTPUT_DIR / proposal_id / domain
    if not output_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Artifacts not generated. Call /artifacts first."
        )

    # Create ZIP file
    zip_filename = f"OASIS_{proposal.contractor_name}_{domain}_{datetime.now().strftime('%Y%m%d')}"
    zip_path = OASIS_OUTPUT_DIR / f"{zip_filename}.zip"

    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', output_dir)

    return FileResponse(
        path=str(zip_path),
        filename=f"{zip_filename}.zip",
        media_type="application/zip",
    )


@router.get("/domains")
async def list_domains():
    """List all available OASIS+ domains"""
    if not OASIS_AVAILABLE:
        return {"domains": []}

    return {
        "domains": [
            {
                "value": d.value,
                "name": d.value.replace("_", " ").title(),
            }
            for d in DomainType
        ]
    }


@router.get("/business-sizes")
async def list_business_sizes():
    """List all business size categories"""
    if not OASIS_AVAILABLE:
        return {"sizes": []}

    return {
        "sizes": [
            {
                "value": s.value,
                "name": s.value.replace("_", " ").title(),
                "threshold": 36 if s != BusinessSize.UNRESTRICTED else 42,
            }
            for s in BusinessSize
        ]
    }
