"""
PropelAI API v2.3
FastAPI backend with Enhanced Compliance Agent integration

Endpoints:
- POST /api/rfp/upload - Upload RFP documents
- POST /api/rfp/process - Process uploaded documents
- GET /api/rfp/{rfp_id} - Get RFP details and requirements
- GET /api/rfp/{rfp_id}/requirements - Get all requirements
- GET /api/rfp/{rfp_id}/export - Export to Excel
- POST /api/rfp/{rfp_id}/amendments - Upload amendment
- GET /api/rfp/{rfp_id}/amendments - Get amendment history
"""

import os
import sys
import uuid
import json
import shutil
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_compliance import (
    EnhancedComplianceAgent,
    AmendmentProcessor,
    export_to_excel
)


# ============== Configuration ==============

UPLOAD_DIR = Path(tempfile.gettempdir()) / "propelai_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path(tempfile.gettempdir()) / "propelai_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============== Pydantic Models ==============

class RFPCreate(BaseModel):
    """Request to create a new RFP"""
    name: str
    solicitation_number: Optional[str] = None
    agency: Optional[str] = None
    due_date: Optional[str] = None


class RFPResponse(BaseModel):
    """Response with RFP details"""
    id: str
    name: str
    solicitation_number: Optional[str]
    agency: Optional[str]
    status: str
    files: List[str]
    requirements_count: int
    created_at: str
    updated_at: str


class RequirementResponse(BaseModel):
    """Single requirement"""
    id: str
    text: str
    section: str
    type: str
    priority: str
    confidence: float
    source_page: Optional[int]


class ProcessingStatus(BaseModel):
    """Processing status"""
    status: str
    progress: int
    message: str
    requirements_count: Optional[int] = None


class AmendmentUpload(BaseModel):
    """Amendment upload request"""
    amendment_number: int
    amendment_date: Optional[str] = None


# ============== In-Memory Store ==============

class RFPStore:
    """In-memory store for RFP data"""
    
    def __init__(self):
        self.rfps: Dict[str, Dict] = {}
        self.processing_status: Dict[str, ProcessingStatus] = {}
    
    def create(self, rfp_id: str, data: Dict) -> Dict:
        """Create a new RFP entry"""
        self.rfps[rfp_id] = {
            "id": rfp_id,
            "name": data.get("name", "Untitled RFP"),
            "solicitation_number": data.get("solicitation_number"),
            "agency": data.get("agency"),
            "due_date": data.get("due_date"),
            "status": "created",
            "files": [],
            "file_paths": [],
            "requirements": [],
            "requirements_graph": None,
            "stats": None,
            "amendments": [],
            "amendment_processor": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        return self.rfps[rfp_id]
    
    def get(self, rfp_id: str) -> Optional[Dict]:
        """Get RFP by ID"""
        return self.rfps.get(rfp_id)
    
    def update(self, rfp_id: str, updates: Dict) -> Dict:
        """Update RFP"""
        if rfp_id not in self.rfps:
            raise KeyError(f"RFP not found: {rfp_id}")
        
        self.rfps[rfp_id].update(updates)
        self.rfps[rfp_id]["updated_at"] = datetime.now().isoformat()
        return self.rfps[rfp_id]
    
    def list_all(self) -> List[Dict]:
        """List all RFPs"""
        return list(self.rfps.values())
    
    def delete(self, rfp_id: str) -> bool:
        """Delete RFP"""
        if rfp_id in self.rfps:
            del self.rfps[rfp_id]
            return True
        return False
    
    def set_status(self, rfp_id: str, status: str, progress: int, message: str, req_count: int = None):
        """Set processing status"""
        self.processing_status[rfp_id] = ProcessingStatus(
            status=status,
            progress=progress,
            message=message,
            requirements_count=req_count
        )
    
    def get_status(self, rfp_id: str) -> Optional[ProcessingStatus]:
        """Get processing status"""
        return self.processing_status.get(rfp_id)


# ============== Global Instances ==============

store = RFPStore()
agent = EnhancedComplianceAgent()


# ============== FastAPI App ==============

app = FastAPI(
    title="PropelAI API",
    description="RFP Intelligence Platform - Extract requirements, track amendments, generate compliance matrices",
    version="2.3.0"
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Root Route (Web UI) ==============

@app.get("/")
async def root():
    """Serve the PropelAI web interface"""
    # Try to serve index.html
    web_paths = [
        Path("/app/web/index.html"),
        Path(__file__).parent.parent / "web" / "index.html",
    ]
    
    for path in web_paths:
        if path.exists():
            return HTMLResponse(content=path.read_text(), status_code=200)
    
    # Fallback HTML
    return HTMLResponse(content="""<!DOCTYPE html>
<html><head><title>PropelAI</title></head>
<body style="background:#0a0a0f;color:#fff;font-family:sans-serif;padding:40px;text-align:center;">
<h1>PropelAI API</h1>
<p>Web UI files not found. API is running.</p>
<p><a href="/docs" style="color:#4f8cff">API Documentation</a></p>
</body></html>""", status_code=200)


# ============== Health Check ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.3.0",
        "components": {
            "enhanced_compliance_agent": "ready",
            "amendment_processor": "ready",
            "excel_export": "ready"
        }
    }


# ============== RFP Management ==============

@app.post("/api/rfp", response_model=RFPResponse)
async def create_rfp(rfp: RFPCreate):
    """Create a new RFP project"""
    rfp_id = f"RFP-{uuid.uuid4().hex[:8].upper()}"
    
    data = store.create(rfp_id, rfp.dict())
    
    return RFPResponse(
        id=data["id"],
        name=data["name"],
        solicitation_number=data["solicitation_number"],
        agency=data["agency"],
        status=data["status"],
        files=data["files"],
        requirements_count=len(data["requirements"]),
        created_at=data["created_at"],
        updated_at=data["updated_at"]
    )


@app.get("/api/rfp")
async def list_rfps():
    """List all RFPs"""
    rfps = store.list_all()
    return {
        "count": len(rfps),
        "rfps": [
            {
                "id": r["id"],
                "name": r["name"],
                "solicitation_number": r["solicitation_number"],
                "status": r["status"],
                "files_count": len(r["files"]),
                "requirements_count": len(r["requirements"]),
                "created_at": r["created_at"]
            }
            for r in rfps
        ]
    }


@app.get("/api/rfp/{rfp_id}")
async def get_rfp(rfp_id: str):
    """Get RFP details"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    return {
        "id": rfp["id"],
        "name": rfp["name"],
        "solicitation_number": rfp["solicitation_number"],
        "agency": rfp["agency"],
        "due_date": rfp["due_date"],
        "status": rfp["status"],
        "files": rfp["files"],
        "requirements_count": len(rfp["requirements"]),
        "stats": rfp["stats"],
        "amendments_count": len(rfp["amendments"]),
        "created_at": rfp["created_at"],
        "updated_at": rfp["updated_at"]
    }


@app.delete("/api/rfp/{rfp_id}")
async def delete_rfp(rfp_id: str):
    """Delete an RFP"""
    if not store.delete(rfp_id):
        raise HTTPException(status_code=404, detail="RFP not found")
    
    # Clean up files
    rfp_dir = UPLOAD_DIR / rfp_id
    if rfp_dir.exists():
        shutil.rmtree(rfp_dir)
    
    return {"status": "deleted", "id": rfp_id}


# ============== File Upload ==============

@app.post("/api/rfp/{rfp_id}/upload")
async def upload_files(
    rfp_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload RFP documents"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    # Create upload directory for this RFP
    rfp_dir = UPLOAD_DIR / rfp_id
    rfp_dir.mkdir(exist_ok=True)
    
    uploaded = []
    file_paths = list(rfp["file_paths"])
    file_names = list(rfp["files"])
    
    for file in files:
        # Validate file type
        ext = Path(file.filename).suffix.lower()
        if ext not in [".pdf", ".docx", ".xlsx", ".doc", ".xls"]:
            continue
        
        # Save file
        file_path = rfp_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_paths.append(str(file_path))
        file_names.append(file.filename)
        
        uploaded.append({
            "name": file.filename,
            "size": len(content),
            "type": ext[1:].upper()
        })
    
    # Update store
    store.update(rfp_id, {
        "files": file_names,
        "file_paths": file_paths,
        "status": "files_uploaded"
    })
    
    return {
        "status": "uploaded",
        "files": uploaded,
        "total_files": len(file_names)
    }


# ============== Processing ==============

def process_rfp_background(rfp_id: str):
    """Background task to process RFP"""
    rfp = store.get(rfp_id)
    if not rfp:
        return
    
    try:
        # Update status
        store.set_status(rfp_id, "processing", 10, "Parsing documents...")
        store.update(rfp_id, {"status": "processing"})
        
        # Get file paths
        file_paths = rfp["file_paths"]
        if not file_paths:
            store.set_status(rfp_id, "error", 0, "No files to process")
            store.update(rfp_id, {"status": "error"})
            return
        
        # Process with Enhanced Compliance Agent
        store.set_status(rfp_id, "processing", 30, "Extracting requirements...")
        
        result = agent.process_files(file_paths)
        
        store.set_status(rfp_id, "processing", 70, "Classifying and prioritizing...")
        
        # Convert requirements graph to list
        requirements = []
        for req_id, req in result.requirements_graph.items():
            requirements.append({
                "id": req.id,
                "text": req.text,
                "section": req.section,
                "type": req.req_type,
                "priority": req.priority,
                "confidence": req.confidence,
                "source_page": req.source_page,
                "source_doc": req.source_doc,
                "keywords": req.keywords
            })
        
        # Build stats
        stats = {
            "total": len(requirements),
            "by_type": result.stats.get("by_type", {}),
            "by_priority": {
                "high": len([r for r in requirements if r["priority"] == "high"]),
                "medium": len([r for r in requirements if r["priority"] == "medium"]),
                "low": len([r for r in requirements if r["priority"] == "low"])
            },
            "by_section": {},
            "processing_time": result.duration_seconds,
            "pages_processed": result.stats.get("pages_processed", 0)
        }
        
        # Count by section
        for req in requirements:
            sec = req["section"]
            stats["by_section"][sec] = stats["by_section"].get(sec, 0) + 1
        
        store.set_status(rfp_id, "processing", 90, "Finalizing...")
        
        # Initialize amendment processor
        amendment_processor = AmendmentProcessor()
        amendment_processor.load_base_requirements(result.requirements_graph)
        
        # Update store
        store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "requirements_graph": result.requirements_graph,
            "stats": stats,
            "amendment_processor": amendment_processor
        })
        
        store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))
        
    except Exception as e:
        store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        store.update(rfp_id, {"status": "error"})


@app.post("/api/rfp/{rfp_id}/process")
async def process_rfp(rfp_id: str, background_tasks: BackgroundTasks):
    """Start processing RFP documents"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp["file_paths"]:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Start background processing
    store.set_status(rfp_id, "starting", 0, "Starting processing...")
    background_tasks.add_task(process_rfp_background, rfp_id)
    
    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"])
    }


@app.get("/api/rfp/{rfp_id}/status")
async def get_processing_status(rfp_id: str):
    """Get processing status"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    status = store.get_status(rfp_id)
    if status:
        return status.dict()
    
    return {
        "status": rfp["status"],
        "progress": 100 if rfp["status"] == "completed" else 0,
        "message": rfp["status"],
        "requirements_count": len(rfp["requirements"])
    }


# ============== Requirements ==============

@app.get("/api/rfp/{rfp_id}/requirements")
async def get_requirements(
    rfp_id: str,
    type: Optional[str] = None,
    priority: Optional[str] = None,
    section: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 500,
    offset: int = 0
):
    """Get requirements with optional filters"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    requirements = rfp["requirements"]
    
    # Apply filters
    if type and type != "all":
        requirements = [r for r in requirements if r["type"] == type]
    
    if priority and priority != "all":
        requirements = [r for r in requirements if r["priority"] == priority]
    
    if section and section != "all":
        requirements = [r for r in requirements if r["section"] == section]
    
    if search:
        search_lower = search.lower()
        requirements = [
            r for r in requirements 
            if search_lower in r["text"].lower() or search_lower in r["id"].lower()
        ]
    
    # Paginate
    total = len(requirements)
    requirements = requirements[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "requirements": requirements
    }


@app.get("/api/rfp/{rfp_id}/requirements/{req_id}")
async def get_requirement(rfp_id: str, req_id: str):
    """Get a single requirement by ID"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    for req in rfp["requirements"]:
        if req["id"] == req_id:
            return req
    
    raise HTTPException(status_code=404, detail="Requirement not found")


# ============== Export ==============

@app.get("/api/rfp/{rfp_id}/export")
async def export_rfp(rfp_id: str, format: str = "xlsx"):
    """Export RFP to Excel"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp["requirements"]:
        raise HTTPException(status_code=400, detail="No requirements to export")
    
    if format != "xlsx":
        raise HTTPException(status_code=400, detail="Only xlsx format supported")
    
    # Create a result-like object for export
    class ExportResult:
        def __init__(self, reqs, stats):
            self.requirements_graph = reqs
            self.stats = stats
            self.duration_seconds = stats.get("processing_time", 0)
    
    result = ExportResult(rfp["requirements_graph"], rfp["stats"])
    
    # Export
    output_path = OUTPUT_DIR / f"{rfp_id}_ComplianceMatrix.xlsx"
    export_to_excel(
        result,
        str(output_path),
        solicitation_number=rfp.get("solicitation_number", rfp_id),
        title=rfp.get("name", "RFP Analysis")
    )
    
    return FileResponse(
        str(output_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{rfp_id}_ComplianceMatrix.xlsx"
    )


# ============== Amendments ==============

@app.post("/api/rfp/{rfp_id}/amendments")
async def upload_amendment(
    rfp_id: str,
    file: UploadFile = File(...),
    amendment_number: int = Form(...),
    amendment_date: Optional[str] = Form(None)
):
    """Upload an amendment document"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp.get("amendment_processor"):
        raise HTTPException(status_code=400, detail="Process base RFP first")
    
    # Save amendment file
    rfp_dir = UPLOAD_DIR / rfp_id / "amendments"
    rfp_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = rfp_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process amendment
    processor = rfp["amendment_processor"]
    
    try:
        result = processor.process_amendment(
            str(file_path),
            amendment_number=amendment_number,
            amendment_date=amendment_date
        )
        
        # Build amendment summary
        amendment = {
            "number": amendment_number,
            "date": amendment_date or datetime.now().strftime("%Y-%m-%d"),
            "file": file.filename,
            "type": result.amendment_type.value,
            "changes": {
                "added": result.requirements_added,
                "modified": result.requirements_modified,
                "deleted": result.requirements_deleted,
                "clarified": result.requirements_clarified
            },
            "qa_count": result.total_questions,
            "mod_count": result.total_modifications,
            "conflicts": len(result.conflicts)
        }
        
        # Update amendments list
        amendments = list(rfp["amendments"])
        amendments.append(amendment)
        
        # Get updated requirements
        updated_reqs = []
        for req_id, req in processor.get_updated_requirements().items():
            updated_reqs.append({
                "id": req.id,
                "text": req.text,
                "section": req.section,
                "type": req.req_type,
                "priority": req.priority,
                "confidence": req.confidence,
                "source_page": req.source_page,
                "source_doc": req.source_doc,
                "keywords": req.keywords
            })
        
        store.update(rfp_id, {
            "amendments": amendments,
            "requirements": updated_reqs
        })
        
        return {
            "status": "processed",
            "amendment": amendment
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing amendment: {str(e)}")


@app.get("/api/rfp/{rfp_id}/amendments")
async def get_amendments(rfp_id: str):
    """Get amendment history"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    processor = rfp.get("amendment_processor")
    
    return {
        "amendments": rfp["amendments"],
        "total_changes": len(processor.get_all_changes()) if processor else 0,
        "change_history": [
            {
                "requirement_id": c.requirement_id,
                "change_type": c.change_type.value,
                "change_source": c.change_source,
                "amendment_number": c.amendment_number,
                "notes": c.notes[:100] if c.notes else None
            }
            for c in (processor.get_all_changes() if processor else [])
        ][:50]  # Limit to 50 for API response
    }


@app.get("/api/rfp/{rfp_id}/amendments/report")
async def get_amendment_report(rfp_id: str):
    """Get amendment change report"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    processor = rfp.get("amendment_processor")
    if not processor:
        raise HTTPException(status_code=400, detail="No amendments processed")
    
    report = processor.generate_change_report()
    
    return {
        "report": report,
        "format": "markdown"
    }


# ============== Stats ==============

@app.get("/api/rfp/{rfp_id}/stats")
async def get_stats(rfp_id: str):
    """Get detailed statistics"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    stats = rfp.get("stats", {})
    requirements = rfp.get("requirements", [])
    
    # Build detailed stats
    return {
        "total_requirements": len(requirements),
        "by_type": stats.get("by_type", {}),
        "by_priority": stats.get("by_priority", {}),
        "by_section": stats.get("by_section", {}),
        "processing_time": stats.get("processing_time", 0),
        "pages_processed": stats.get("pages_processed", 0),
        "files_processed": len(rfp.get("files", [])),
        "amendments_count": len(rfp.get("amendments", [])),
        "high_confidence": len([r for r in requirements if r.get("confidence", 0) >= 0.8]),
        "medium_confidence": len([r for r in requirements if 0.5 <= r.get("confidence", 0) < 0.8]),
        "low_confidence": len([r for r in requirements if r.get("confidence", 0) < 0.5])
    }


# ============== Proposal Outline ==============

@app.post("/api/rfp/{rfp_id}/outline")
async def generate_outline(rfp_id: str):
    """Generate proposal outline from RFP"""
    from agents.enhanced_compliance import OutlineGenerator
    
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp["file_paths"]:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        generator = OutlineGenerator()
        
        # Process first file (main RFP document)
        result = generator.process_rfp(rfp["file_paths"][0])
        
        # Store outline
        outline_data = generator.generate_json_outline(result)
        store.update(rfp_id, {"outline": outline_data})
        
        return {
            "status": "generated",
            "outline": outline_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outline generation failed: {str(e)}")


@app.get("/api/rfp/{rfp_id}/outline")
async def get_outline(rfp_id: str, format: str = "json"):
    """Get proposal outline"""
    from agents.enhanced_compliance import OutlineGenerator
    
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    outline = rfp.get("outline")
    
    if not outline:
        # Generate if not exists
        if not rfp["file_paths"]:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        generator = OutlineGenerator()
        result = generator.process_rfp(rfp["file_paths"][0])
        outline = generator.generate_json_outline(result)
        store.update(rfp_id, {"outline": outline})
    
    if format == "markdown":
        # Regenerate markdown from stored data
        generator = OutlineGenerator()
        result = generator.process_rfp(rfp["file_paths"][0])
        markdown = generator.generate_markdown_outline(result)
        return {"format": "markdown", "content": markdown}
    
    return {"format": "json", "outline": outline}


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
