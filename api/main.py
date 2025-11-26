"""
PropelAI API Layer
FastAPI-based REST API for the Autonomous Proposal Operating System

Endpoints:
- POST /proposals - Create new proposal
- GET /proposals/{id} - Get proposal status and state
- POST /proposals/{id}/upload - Upload RFP documents
- POST /proposals/{id}/shred - Run RFP shredding
- POST /proposals/{id}/strategy - Generate win strategy
- POST /proposals/{id}/draft - Generate drafts
- POST /proposals/{id}/redteam - Run red team evaluation
- POST /proposals/{id}/feedback - Submit human feedback (HITL)
- GET /proposals/{id}/export - Export compliance matrix
"""

import os
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import PropelAI components
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import ProposalState, ProposalPhase, create_initial_state
from core.orchestrator import ProposalOrchestrator, create_orchestrator
from agents.compliance_agent import ComplianceAgent, create_compliance_agent
from agents.strategy_agent import StrategyAgent, create_strategy_agent
from agents.drafting_agent import DraftingAgent, create_drafting_agent
from agents.red_team_agent import RedTeamAgent, create_red_team_agent
from tools.document_tools import DocumentLoader, ComplianceMatrixExporter


# ============== Pydantic Models ==============

class CreateProposalRequest(BaseModel):
    """Request to create a new proposal"""
    client_name: str
    opportunity_name: str
    solicitation_number: str
    due_date: Optional[str] = None


class ProposalResponse(BaseModel):
    """Response with proposal details"""
    proposal_id: str
    client_name: str
    opportunity_name: str
    solicitation_number: str
    current_phase: str
    created_at: str
    updated_at: str


class HumanFeedbackRequest(BaseModel):
    """Request to submit human feedback"""
    section_id: str
    feedback_type: str = Field(..., description="'edit', 'reject', 'approve', 'comment'")
    corrected_content: Optional[str] = None
    correction_reason: Optional[str] = None


class ShredResultResponse(BaseModel):
    """Response from shredding operation"""
    status: str
    requirements_count: int
    instructions_count: int
    evaluation_criteria_count: int
    sections_found: List[str]


class StrategyResultResponse(BaseModel):
    """Response from strategy generation"""
    status: str
    win_themes_count: int
    volumes_count: int


class RedTeamResultResponse(BaseModel):
    """Response from red team evaluation"""
    status: str
    overall_score: str
    numeric_score: float
    recommendation: str
    deficiencies_count: int


# ============== In-Memory State Store ==============
# In production, this would be PostgreSQL with LangGraph checkpointing

class ProposalStore:
    """In-memory store for proposal states"""
    
    def __init__(self):
        self.proposals: Dict[str, ProposalState] = {}
    
    def create(self, proposal: ProposalState) -> str:
        """Create a new proposal"""
        proposal_id = proposal["proposal_id"]
        self.proposals[proposal_id] = proposal
        return proposal_id
    
    def get(self, proposal_id: str) -> Optional[ProposalState]:
        """Get a proposal by ID"""
        return self.proposals.get(proposal_id)
    
    def update(self, proposal_id: str, updates: Dict[str, Any]) -> ProposalState:
        """Update a proposal"""
        if proposal_id not in self.proposals:
            raise KeyError(f"Proposal not found: {proposal_id}")
        
        proposal = self.proposals[proposal_id]
        
        # Merge updates
        for key, value in updates.items():
            if key in proposal:
                # Handle list appending (for things like agent_trace_log)
                if isinstance(value, list) and isinstance(proposal.get(key), list):
                    proposal[key] = proposal[key] + value
                else:
                    proposal[key] = value
        
        proposal["updated_at"] = datetime.now().isoformat()
        return proposal
    
    def list(self) -> List[ProposalState]:
        """List all proposals"""
        return list(self.proposals.values())
    
    def delete(self, proposal_id: str) -> bool:
        """Delete a proposal"""
        if proposal_id in self.proposals:
            del self.proposals[proposal_id]
            return True
        return False


# ============== Global Instances ==============

store = ProposalStore()
doc_loader = DocumentLoader()
compliance_exporter = ComplianceMatrixExporter()

# Create agents
compliance_agent = create_compliance_agent()
strategy_agent = create_strategy_agent()
drafting_agent = create_drafting_agent()
red_team_agent = create_red_team_agent()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("🚀 PropelAI APOS starting up...")
    print("📋 Agents initialized: Compliance, Strategy, Drafting, Red Team")
    yield
    print("👋 PropelAI APOS shutting down...")


app = FastAPI(
    title="PropelAI APOS API",
    description="Autonomous Proposal Operating System - AI-powered government proposal generation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Health Check ==============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "agents": {
            "compliance": "ready",
            "strategy": "ready",
            "drafting": "ready",
            "red_team": "ready"
        }
    }


# ============== Proposal CRUD ==============

@app.post("/proposals", response_model=ProposalResponse)
async def create_proposal(request: CreateProposalRequest):
    """Create a new proposal"""
    proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
    
    proposal = create_initial_state(
        proposal_id=proposal_id,
        client_name=request.client_name,
        opportunity_name=request.opportunity_name,
        solicitation_number=request.solicitation_number,
        due_date=request.due_date
    )
    
    store.create(proposal)
    
    return ProposalResponse(
        proposal_id=proposal_id,
        client_name=proposal["client_name"],
        opportunity_name=proposal["opportunity_name"],
        solicitation_number=proposal["solicitation_number"],
        current_phase=proposal["current_phase"],
        created_at=proposal["created_at"],
        updated_at=proposal["updated_at"]
    )


@app.get("/proposals")
async def list_proposals():
    """List all proposals"""
    proposals = store.list()
    return {
        "count": len(proposals),
        "proposals": [
            {
                "proposal_id": p["proposal_id"],
                "client_name": p["client_name"],
                "opportunity_name": p["opportunity_name"],
                "current_phase": p["current_phase"],
                "updated_at": p["updated_at"]
            }
            for p in proposals
        ]
    }


@app.get("/proposals/{proposal_id}")
async def get_proposal(proposal_id: str):
    """Get a specific proposal with full state"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    return {
        "proposal_id": proposal["proposal_id"],
        "client_name": proposal["client_name"],
        "opportunity_name": proposal["opportunity_name"],
        "solicitation_number": proposal["solicitation_number"],
        "current_phase": proposal["current_phase"],
        "due_date": proposal.get("due_date"),
        "created_at": proposal["created_at"],
        "updated_at": proposal["updated_at"],
        "statistics": {
            "requirements_count": len(proposal.get("requirements", [])),
            "win_themes_count": len(proposal.get("win_themes", [])),
            "draft_sections_count": len(proposal.get("draft_sections", {})),
            "red_team_evaluations": len(proposal.get("red_team_feedback", [])),
            "human_feedback_count": len(proposal.get("human_feedback", []))
        },
        "pending_human_review": proposal.get("pending_human_review", False),
        "error_state": proposal.get("error_state")
    }


# ============== Document Upload ==============

@app.post("/proposals/{proposal_id}/upload")
async def upload_rfp(proposal_id: str, file: UploadFile = File(...)):
    """Upload an RFP document for processing"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    # Read file content
    content = await file.read()
    
    # Parse document
    try:
        parsed = doc_loader.load_from_bytes(
            content=content,
            file_name=file.filename,
            file_type=file.filename.split(".")[-1]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse document: {str(e)}")
    
    # Update proposal state
    file_path = f"/uploads/{proposal_id}/{file.filename}"
    
    store.update(proposal_id, {
        "rfp_raw_text": parsed.raw_text,
        "rfp_file_paths": proposal.get("rfp_file_paths", []) + [file_path],
        "rfp_metadata": {
            **proposal.get("rfp_metadata", {}),
            "uploaded_files": proposal.get("rfp_metadata", {}).get("uploaded_files", []) + [{
                "file_name": file.filename,
                "file_type": parsed.file_type,
                "total_pages": parsed.total_pages,
                "total_chars": parsed.total_chars,
                "uploaded_at": datetime.now().isoformat()
            }],
            "structure": parsed.structure
        }
    })
    
    return {
        "status": "uploaded",
        "file_name": file.filename,
        "file_type": parsed.file_type,
        "pages": parsed.total_pages,
        "characters": parsed.total_chars,
        "structure": parsed.structure
    }


# ============== Agent Execution Endpoints ==============

@app.post("/proposals/{proposal_id}/shred", response_model=ShredResultResponse)
async def run_shredding(proposal_id: str, background_tasks: BackgroundTasks):
    """Run the Compliance Agent to shred the RFP"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    if not proposal.get("rfp_raw_text"):
        raise HTTPException(status_code=400, detail="No RFP document uploaded")
    
    # Execute compliance agent
    result = compliance_agent(proposal)
    
    # Update state
    updated = store.update(proposal_id, result)
    
    return ShredResultResponse(
        status="completed",
        requirements_count=len(result.get("requirements", [])),
        instructions_count=len(result.get("instructions", [])),
        evaluation_criteria_count=len(result.get("evaluation_criteria", [])),
        sections_found=result.get("rfp_metadata", {}).get("sections_found", [])
    )


@app.post("/proposals/{proposal_id}/strategy", response_model=StrategyResultResponse)
async def run_strategy(proposal_id: str):
    """Run the Strategy Agent to develop win themes"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    if not proposal.get("evaluation_criteria"):
        raise HTTPException(status_code=400, detail="Run shredding first to extract evaluation criteria")
    
    # Execute strategy agent
    result = strategy_agent(proposal)
    
    # Update state
    updated = store.update(proposal_id, result)
    
    outline = result.get("annotated_outline", {})
    volumes_count = len(outline.get("volumes", {}))
    
    return StrategyResultResponse(
        status="completed",
        win_themes_count=len(result.get("win_themes", [])),
        volumes_count=volumes_count
    )


@app.post("/proposals/{proposal_id}/draft")
async def run_drafting(proposal_id: str):
    """Run the Drafting Agent to generate proposal content"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    if not proposal.get("annotated_outline"):
        raise HTTPException(status_code=400, detail="Run strategy first to generate outline")
    
    # Execute drafting agent
    result = drafting_agent(proposal)
    
    # Update state
    updated = store.update(proposal_id, result)
    
    draft_sections = result.get("draft_sections", {})
    total_words = sum(d.get("word_count", 0) for d in draft_sections.values())
    uncited_count = sum(len(d.get("uncited_claims", [])) for d in draft_sections.values())
    
    return {
        "status": "completed",
        "sections_drafted": len(draft_sections),
        "total_words": total_words,
        "uncited_claims": uncited_count,
        "warning": "Review uncited claims before proceeding" if uncited_count > 0 else None
    }


@app.post("/proposals/{proposal_id}/redteam", response_model=RedTeamResultResponse)
async def run_red_team(proposal_id: str):
    """Run the Red Team Agent to evaluate the proposal"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    if not proposal.get("draft_sections"):
        raise HTTPException(status_code=400, detail="Run drafting first to generate content")
    
    # Execute red team agent
    result = red_team_agent(proposal)
    
    # Update state
    updated = store.update(proposal_id, result)
    
    feedback = result.get("red_team_feedback", [{}])[-1]
    
    return RedTeamResultResponse(
        status="completed",
        overall_score=feedback.get("overall_score", "unknown"),
        numeric_score=feedback.get("overall_numeric", 0),
        recommendation=feedback.get("recommendation", "unknown"),
        deficiencies_count=len(feedback.get("critical_deficiencies", []))
    )


# ============== Human-in-the-Loop ==============

@app.post("/proposals/{proposal_id}/feedback")
async def submit_feedback(proposal_id: str, feedback: HumanFeedbackRequest):
    """Submit human feedback on a draft section"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    # Create feedback record
    feedback_record = {
        "timestamp": datetime.now().isoformat(),
        "section_id": feedback.section_id,
        "feedback_type": feedback.feedback_type,
        "original_content": proposal.get("draft_sections", {}).get(feedback.section_id, {}).get("content", ""),
        "corrected_content": feedback.corrected_content,
        "correction_reason": feedback.correction_reason,
        "user_id": "human_reviewer"  # Would come from auth in production
    }
    
    # Update proposal with feedback
    updates = {
        "human_feedback": [feedback_record],
        "pending_human_review": False
    }
    
    # If edit, update the draft section
    if feedback.feedback_type == "edit" and feedback.corrected_content:
        draft_sections = dict(proposal.get("draft_sections", {}))
        if feedback.section_id in draft_sections:
            draft_sections[feedback.section_id]["content"] = feedback.corrected_content
            draft_sections[feedback.section_id]["last_modified"] = datetime.now().isoformat()
            draft_sections[feedback.section_id]["modified_by"] = "human_reviewer"
            draft_sections[feedback.section_id]["version"] = draft_sections[feedback.section_id].get("version", 1) + 1
        updates["draft_sections"] = draft_sections
    
    store.update(proposal_id, updates)
    
    return {
        "status": "feedback_recorded",
        "feedback_type": feedback.feedback_type,
        "section_id": feedback.section_id,
        "timestamp": feedback_record["timestamp"]
    }


# ============== Export ==============

@app.get("/proposals/{proposal_id}/export/compliance-matrix")
async def export_compliance_matrix(proposal_id: str):
    """Export the compliance matrix as Excel"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    compliance_matrix = proposal.get("compliance_matrix", [])
    if not compliance_matrix:
        raise HTTPException(status_code=400, detail="No compliance matrix generated")
    
    # Export to temp file
    import tempfile
    output_path = os.path.join(tempfile.gettempdir(), f"{proposal_id}_compliance_matrix.xlsx")
    
    result_path = compliance_exporter.export(compliance_matrix, output_path)
    
    return FileResponse(
        result_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{proposal_id}_compliance_matrix.xlsx"
    )


@app.get("/proposals/{proposal_id}/export/evaluation")
async def export_evaluation(proposal_id: str):
    """Export the red team evaluation report"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    red_team_feedback = proposal.get("red_team_feedback", [])
    if not red_team_feedback:
        raise HTTPException(status_code=400, detail="No red team evaluation available")
    
    latest = red_team_feedback[-1]
    
    return {
        "proposal_id": proposal_id,
        "evaluation_id": latest.get("evaluation_id"),
        "evaluated_at": latest.get("evaluated_at"),
        "overall_score": latest.get("overall_score"),
        "numeric_score": latest.get("overall_numeric"),
        "recommendation": latest.get("recommendation"),
        "narrative": latest.get("narrative"),
        "section_scores": latest.get("section_scores", []),
        "findings": latest.get("findings", []),
        "remediation_plan": latest.get("remediation_plan", [])
    }


@app.get("/proposals/{proposal_id}/audit-log")
async def get_audit_log(proposal_id: str):
    """Get the full audit log (Agent-Trace) for governance"""
    proposal = store.get(proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")
    
    return {
        "proposal_id": proposal_id,
        "total_entries": len(proposal.get("agent_trace_log", [])),
        "entries": proposal.get("agent_trace_log", [])
    }


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
