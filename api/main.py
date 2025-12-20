"""
PropelAI API v4.0
FastAPI backend with Enhanced Compliance Agent + Trust Gate integration

v4.0 Endpoints (Trust Gate):
- GET /api/requirements/{req_id}/source - Get source coordinates for a requirement
- GET /api/documents/{doc_id}/page/{page_num}/image - Get PDF page as image
- POST /api/rfp/{rfp_id}/strategy - Generate win themes and strategy

Core Endpoints:
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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Response
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

# v2.8: Import semantic extractor
try:
    from agents.enhanced_compliance import (
        SEMANTIC_AVAILABLE,
        SemanticRequirementExtractor,
        SemanticCTMExporter,
        SemanticExtractionResult,
    )
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticRequirementExtractor = None
    SemanticCTMExporter = None
    SemanticExtractionResult = None

# v2.9: Import best practices CTM components
try:
    from agents.enhanced_compliance import (
        BEST_PRACTICES_AVAILABLE,
        SectionAwareExtractor,
        BestPracticesCTMExporter,
        analyze_rfp_structure,
        extract_requirements_structured,
    )
except ImportError:
    BEST_PRACTICES_AVAILABLE = False
    SectionAwareExtractor = None
    BestPracticesCTMExporter = None
    analyze_rfp_structure = None
    extract_requirements_structured = None


# v2.11: Annotated Outline Exporter
try:
    from agents.enhanced_compliance.annotated_outline_exporter import (
        AnnotatedOutlineExporter,
        AnnotatedOutlineConfig
    )
    ANNOTATED_OUTLINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Annotated outline exporter not available: {e}")
    ANNOTATED_OUTLINE_AVAILABLE = False

# v3.0: Resilient Extraction Pipeline (Extract-First Architecture)
try:
    from agents.enhanced_compliance.resilient_extractor import (
        ResilientExtractor,
        create_resilient_extractor,
    )
    from agents.enhanced_compliance.extraction_models import (
        ExtractionResult as ResilientExtractionResult,
        ConfidenceLevel,
    )
    from agents.enhanced_compliance.extraction_validator import (
        ExtractionValidator,
        ReproducibilityTester,
    )
    from agents.enhanced_compliance.resilient_extractor import PIPELINE_VERSION
    from agents.enhanced_compliance.universal_extractor import EXTRACTOR_VERSION
    RESILIENT_EXTRACTION_AVAILABLE = True
    resilient_extractor = create_resilient_extractor()
    print(f"=== PropelAI Resilient Extraction v{PIPELINE_VERSION} (extractor v{EXTRACTOR_VERSION}) ===")
except ImportError as e:
    print(f"Warning: Resilient extraction not available: {e}")
    RESILIENT_EXTRACTION_AVAILABLE = False
    resilient_extractor = None

# v3.2: Guided Document Upload
try:
    from agents.enhanced_compliance.document_types import (
        DocumentType,
        DocumentSlot,
        UPLOAD_SLOTS,
        SKIP_DOCUMENTS,
        get_slot_by_id,
        classify_document_by_filename,
        get_ui_config,
    )
    GUIDED_UPLOAD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Guided upload not available: {e}")
    GUIDED_UPLOAD_AVAILABLE = False

# v4.0: Trust Gate - PDF Coordinate Extraction
try:
    from agents.enhanced_compliance import (
        TRUST_GATE_AVAILABLE,
        BoundingBox,
        SourceCoordinate,
        PDFCoordinateExtractor,
        get_coordinate_extractor,
    )
    if TRUST_GATE_AVAILABLE:
        coordinate_extractor = get_coordinate_extractor()
        print("=== PropelAI Trust Gate v4.0 (PDF coordinate extraction) ===")
    else:
        coordinate_extractor = None
except ImportError as e:
    print(f"Warning: Trust Gate not available: {e}")
    TRUST_GATE_AVAILABLE = False
    coordinate_extractor = None

# v4.0: Strategy Agent + Competitive Analysis
try:
    from agents.strategy_agent import (
        StrategyAgent,
        CompetitorAnalyzer,
        GhostingLanguageGenerator,
    )
    STRATEGY_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Strategy Agent not available: {e}")
    STRATEGY_AGENT_AVAILABLE = False
    StrategyAgent = None
    CompetitorAnalyzer = None
    GhostingLanguageGenerator = None

# v4.0: Drafting Workflow
try:
    from agents.drafting_workflow import (
        run_drafting_workflow,
        build_drafting_graph,
        LANGGRAPH_AVAILABLE,
    )
    DRAFTING_WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Drafting workflow not available: {e}")
    DRAFTING_WORKFLOW_AVAILABLE = False
    run_drafting_workflow = None
    LANGGRAPH_AVAILABLE = False

# v4.0: Vector Store (imported later, default to False)
VECTOR_STORE_AVAILABLE = False


# ============== Configuration ==============

# v4.1: Use persistent storage on Render Disk if available, otherwise temp directory
PERSISTENT_DATA_DIR = Path("/data")
if PERSISTENT_DATA_DIR.exists() and os.access(PERSISTENT_DATA_DIR, os.W_OK):
    UPLOAD_DIR = PERSISTENT_DATA_DIR / "uploads"
    OUTPUT_DIR = PERSISTENT_DATA_DIR / "outputs"
    print(f"[Storage] Using persistent disk at /data")
else:
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "propelai_uploads"
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "propelai_outputs"
    print(f"[Storage] Using temporary storage (no persistent disk)")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# v4.1: Database configuration
try:
    from api.database import init_db, is_db_available, db_store, DatabaseStore
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    is_db_available = lambda: False
    db_store = None
    print("[DB] Database module not available")


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


# ============== Hybrid Store (In-Memory + Database) ==============

class RFPStore:
    """
    Hybrid store for RFP data (v4.1).

    Uses in-memory storage for fast access during processing,
    with async database persistence for durability.

    - Processing status: in-memory only (transient)
    - RFP data: in-memory + async sync to database
    """

    def __init__(self):
        self.rfps: Dict[str, Dict] = {}
        self.processing_status: Dict[str, ProcessingStatus] = {}
        self._db_available = False
        self._db_initialized = False

    async def init_database(self):
        """Initialize database connection and load existing data"""
        if not DATABASE_AVAILABLE or not is_db_available():
            print("[Store] Running in memory-only mode (no database)")
            return

        try:
            success = await init_db()
            if success:
                self._db_available = True
                self._db_initialized = True
                # Load existing RFPs from database
                await self._load_from_db()
                print(f"[Store] Database initialized, loaded {len(self.rfps)} RFPs")
            else:
                print("[Store] Database init failed, using memory-only mode")
        except Exception as e:
            print(f"[Store] Database error: {e}, using memory-only mode")

    async def _load_from_db(self):
        """Load all RFPs from database into memory"""
        if not self._db_available:
            return
        try:
            rfps = await db_store.list_all()
            for rfp_data in rfps:
                self.rfps[rfp_data["id"]] = self._db_to_memory(rfp_data)
        except Exception as e:
            print(f"[Store] Error loading from DB: {e}")

    def _db_to_memory(self, db_data: Dict) -> Dict:
        """Convert database format to in-memory format"""
        # Add fields that are only in memory
        result = dict(db_data)
        result.setdefault("requirements_graph", None)
        result.setdefault("amendment_processor", None)
        result.setdefault("best_practices_result", None)
        result.setdefault("semantic_result", None)
        result.setdefault("resilient_result", None)
        return result

    async def _sync_to_db(self, rfp_id: str):
        """Sync an RFP to the database"""
        if not self._db_available:
            return
        try:
            rfp = self.rfps.get(rfp_id)
            if not rfp:
                return

            # Filter out non-serializable fields
            db_data = {k: v for k, v in rfp.items()
                      if k not in ["requirements_graph", "amendment_processor",
                                   "best_practices_result", "semantic_result", "resilient_result"]}

            # Check if exists
            existing = await db_store.get(rfp_id)
            if existing:
                await db_store.update(rfp_id, db_data)
            else:
                await db_store.create(rfp_id, db_data)
                # Update with remaining fields
                await db_store.update(rfp_id, db_data)
        except Exception as e:
            print(f"[Store] DB sync error for {rfp_id}: {e}")

    def _schedule_db_sync(self, rfp_id: str):
        """Schedule async database sync (non-blocking)"""
        if not self._db_available:
            return
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._sync_to_db(rfp_id))
            else:
                asyncio.run(self._sync_to_db(rfp_id))
        except RuntimeError:
            # No event loop, try to create one
            try:
                asyncio.run(self._sync_to_db(rfp_id))
            except Exception as e:
                print(f"[Store] Could not sync to DB: {e}")

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
            "document_metadata": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._schedule_db_sync(rfp_id)
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
        self._schedule_db_sync(rfp_id)
        return self.rfps[rfp_id]

    def list_all(self) -> List[Dict]:
        """List all RFPs"""
        return list(self.rfps.values())

    def delete(self, rfp_id: str) -> bool:
        """Delete RFP"""
        if rfp_id in self.rfps:
            del self.rfps[rfp_id]
            # Also delete from database
            if self._db_available:
                import asyncio
                try:
                    asyncio.run(db_store.delete(rfp_id))
                except Exception as e:
                    print(f"[Store] DB delete error: {e}")
            return True
        return False

    def set_status(self, rfp_id: str, status: str, progress: int, message: str, req_count: int = None):
        """Set processing status (in-memory only)"""
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

# v2.8: Initialize semantic extractor if available
semantic_extractor = None
semantic_ctm_exporter = None
if SEMANTIC_AVAILABLE:
    try:
        semantic_extractor = SemanticRequirementExtractor(use_llm=False)  # Start without LLM
        semantic_ctm_exporter = SemanticCTMExporter()
    except Exception as e:
        print(f"Warning: Could not initialize semantic extractor: {e}")

# v2.9: Initialize best practices extractor if available
best_practices_extractor = None
best_practices_exporter = None
if BEST_PRACTICES_AVAILABLE:
    try:
        best_practices_extractor = SectionAwareExtractor()
        best_practices_exporter = BestPracticesCTMExporter()
    except Exception as e:
        print(f"Warning: Could not initialize best practices extractor: {e}")


# ============== FastAPI App ==============

app = FastAPI(
    title="PropelAI API",
    description="RFP Intelligence Platform - Extract requirements, track amendments, generate compliance matrices. v4.0 adds Trust Gate source traceability.",
    version="4.0.0"
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Startup Event ==============

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    print("[Startup] PropelAI v4.1 starting...")
    print(f"[Startup] Upload directory: {UPLOAD_DIR}")
    print(f"[Startup] Database available: {DATABASE_AVAILABLE and is_db_available()}")

    # Initialize database and load existing RFPs
    await store.init_database()

    print("[Startup] Ready to serve requests")


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
    # v4.1: Check storage type
    storage_type = "persistent" if PERSISTENT_DATA_DIR.exists() else "temporary"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.1.0",
        "storage": {
            "type": storage_type,
            "upload_dir": str(UPLOAD_DIR),
            "database": "connected" if store._db_available else "not configured",
            "rfps_loaded": len(store.rfps),
        },
        "components": {
            "enhanced_compliance_agent": "ready",
            "amendment_processor": "ready",
            "excel_export": "ready",
            "semantic_extractor": "ready" if semantic_extractor else "not available",
            "semantic_ctm_export": "ready" if semantic_ctm_exporter else "not available",
            "best_practices_extractor": "ready" if best_practices_extractor else "not available",
            "best_practices_ctm_export": "ready" if best_practices_exporter else "not available",
            # v4.0 components
            "trust_gate": "ready" if TRUST_GATE_AVAILABLE and coordinate_extractor else "not available",
            "strategy_agent": "ready" if STRATEGY_AGENT_AVAILABLE else "not available",
            "drafting_workflow": "ready" if DRAFTING_WORKFLOW_AVAILABLE else "not available",
            "langgraph": "available" if LANGGRAPH_AVAILABLE else "not installed",
            "vector_store": "ready" if VECTOR_STORE_AVAILABLE else "not available",
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


# ============== Guided Document Upload (v3.2) ==============

@app.get("/api/upload-config")
async def get_upload_config():
    """
    Get the guided upload UI configuration.

    Returns the upload slots, skip documents list, and tips
    for the guided document upload interface.
    """
    if not GUIDED_UPLOAD_AVAILABLE:
        # Return a basic config if module not available
        return {
            "upload_slots": [
                {
                    "id": "all",
                    "doc_type": "auto_detect",
                    "title": "RFP Documents",
                    "description": "Upload all RFP documents",
                    "help_text": "Upload your RFP files here",
                    "common_names": [],
                    "required": True,
                    "allows_multiple": True,
                    "show_not_applicable": False,
                    "order": 1,
                    "icon": "📄",
                    "color": "#4A5568"
                }
            ],
            "skip_documents": [],
            "tips": [],
            "guided_mode_available": False
        }

    config = get_ui_config()
    config["guided_mode_available"] = True
    return config


@app.post("/api/rfp/{rfp_id}/upload-guided")
async def upload_files_guided(
    rfp_id: str,
    files: List[UploadFile] = File(...),
    doc_types: str = Form(default="")  # Comma-separated doc_type values matching file order
):
    """
    Upload RFP documents with explicit document type tags.

    This endpoint supports the guided upload UI where users specify
    what type of document each file is (SOW, Section L, Section M, etc.)

    Args:
        rfp_id: The RFP identifier
        files: List of files to upload
        doc_types: Comma-separated document types matching file order
                   e.g., "sow,instructions,evaluation"
                   Use "auto_detect" or empty for automatic classification

    Returns:
        Upload status with classified document info
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Parse document types
    doc_type_list = [dt.strip() for dt in doc_types.split(",")] if doc_types else []

    # Create upload directory
    rfp_dir = UPLOAD_DIR / rfp_id
    rfp_dir.mkdir(exist_ok=True)

    uploaded = []
    file_paths = list(rfp.get("file_paths", []))
    file_names = list(rfp.get("files", []))

    # Initialize document metadata if not present
    doc_metadata = rfp.get("document_metadata", {})

    for i, file in enumerate(files):
        # Validate file type
        ext = Path(file.filename).suffix.lower()
        if ext not in [".pdf", ".docx", ".xlsx", ".doc", ".xls"]:
            continue

        # Get document type (from form or auto-detect)
        if i < len(doc_type_list) and doc_type_list[i] and doc_type_list[i] != "auto_detect":
            doc_type = doc_type_list[i]
        elif GUIDED_UPLOAD_AVAILABLE:
            doc_type = classify_document_by_filename(file.filename).value
        else:
            doc_type = "auto_detect"

        # Save file
        file_path = rfp_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        file_paths.append(str(file_path))
        file_names.append(file.filename)

        # Store document metadata
        doc_metadata[file.filename] = {
            "doc_type": doc_type,
            "file_path": str(file_path),
            "size": len(content),
            "uploaded_at": datetime.now().isoformat()
        }

        uploaded.append({
            "name": file.filename,
            "size": len(content),
            "type": ext[1:].upper(),
            "doc_type": doc_type,
            "doc_type_label": _get_doc_type_label(doc_type)
        })

    # Update store with metadata
    store.update(rfp_id, {
        "files": file_names,
        "file_paths": file_paths,
        "document_metadata": doc_metadata,
        "status": "files_uploaded"
    })

    return {
        "status": "uploaded",
        "files": uploaded,
        "total_files": len(file_names),
        "document_summary": _summarize_documents(doc_metadata)
    }


def _get_doc_type_label(doc_type: str) -> str:
    """Get human-readable label for document type"""
    labels = {
        "sow": "Statement of Work (SOW)",
        "instructions": "Proposal Instructions (Section L)",
        "evaluation": "Evaluation Criteria (Section M)",
        "combined_lm": "Combined L & M",
        "solicitation": "Main Solicitation",
        "amendment": "Amendment",
        "attachment": "Attachment",
        "auto_detect": "Auto-Detected",
        "not_applicable": "Not Applicable"
    }
    return labels.get(doc_type, doc_type.title())


def _summarize_documents(doc_metadata: Dict) -> Dict:
    """Summarize uploaded documents by type"""
    summary = {
        "has_sow": False,
        "has_instructions": False,
        "has_evaluation": False,
        "by_type": {}
    }

    for filename, meta in doc_metadata.items():
        doc_type = meta.get("doc_type", "auto_detect")

        if doc_type == "sow":
            summary["has_sow"] = True
        elif doc_type == "instructions":
            summary["has_instructions"] = True
        elif doc_type == "evaluation":
            summary["has_evaluation"] = True
        elif doc_type == "combined_lm":
            summary["has_instructions"] = True
            summary["has_evaluation"] = True

        if doc_type not in summary["by_type"]:
            summary["by_type"][doc_type] = []
        summary["by_type"][doc_type].append(filename)

    # Add warnings for missing required documents
    summary["warnings"] = []
    if not summary["has_sow"]:
        summary["warnings"].append(
            "No Statement of Work (SOW) identified. "
            "Technical requirements extraction may be incomplete."
        )
    if not summary["has_instructions"]:
        summary["warnings"].append(
            "No Section L identified. "
            "Proposal structure and page limits may not be detected."
        )

    return summary


@app.get("/api/rfp/{rfp_id}/documents")
async def get_rfp_documents(rfp_id: str):
    """
    Get detailed information about uploaded documents for an RFP.

    Returns document metadata including types, sizes, and classification.
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    doc_metadata = rfp.get("document_metadata", {})

    # Build response with all documents
    documents = []
    for filename in rfp.get("files", []):
        meta = doc_metadata.get(filename, {})
        documents.append({
            "filename": filename,
            "doc_type": meta.get("doc_type", "auto_detect"),
            "doc_type_label": _get_doc_type_label(meta.get("doc_type", "auto_detect")),
            "size": meta.get("size", 0),
            "uploaded_at": meta.get("uploaded_at")
        })

    return {
        "rfp_id": rfp_id,
        "documents": documents,
        "summary": _summarize_documents(doc_metadata)
    }


@app.put("/api/rfp/{rfp_id}/documents/{filename}/type")
async def update_document_type(rfp_id: str, filename: str, doc_type: str = Form(...)):
    """
    Update the document type for a specific file.

    Allows users to correct misclassified documents.
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    if filename not in rfp.get("files", []):
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate document type
    valid_types = ["sow", "instructions", "evaluation", "combined_lm",
                   "solicitation", "amendment", "attachment", "auto_detect", "not_applicable"]
    if doc_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Must be one of: {', '.join(valid_types)}"
        )

    # Update metadata
    doc_metadata = rfp.get("document_metadata", {})
    if filename not in doc_metadata:
        doc_metadata[filename] = {}

    doc_metadata[filename]["doc_type"] = doc_type
    doc_metadata[filename]["manually_classified"] = True

    store.update(rfp_id, {"document_metadata": doc_metadata})

    return {
        "status": "updated",
        "filename": filename,
        "doc_type": doc_type,
        "doc_type_label": _get_doc_type_label(doc_type)
    }


# ============== Processing ==============

def process_rfp_background(rfp_id: str):
    """Background task to process RFP - uses resilient extraction as default (v3.0)"""
    rfp = store.get(rfp_id)
    if not rfp:
        return

    # Use resilient extraction if available (v3.0) - this is now the default
    if RESILIENT_EXTRACTION_AVAILABLE and resilient_extractor:
        process_rfp_resilient_background(rfp_id)
        return

    # Fall back to best practices extraction (v2.10)
    if BEST_PRACTICES_AVAILABLE and best_practices_extractor:
        process_rfp_best_practices_background(rfp_id)
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
            # Safely extract attributes from RequirementNode
            section = ""
            source_page = None
            source_doc = ""
            
            if hasattr(req, 'source') and req.source:
                section = getattr(req.source, 'section_id', '') or ''
                source_page = getattr(req.source, 'page_number', None)
                source_doc = getattr(req.source, 'document_name', '') or ''
            
            # Get type - could be requirement_type or req_type
            req_type = "performance"
            if hasattr(req, 'requirement_type'):
                req_type = req.requirement_type.value if hasattr(req.requirement_type, 'value') else str(req.requirement_type)
            elif hasattr(req, 'req_type'):
                req_type = req.req_type
            
            # Get confidence
            confidence = 0.7
            if hasattr(req, 'confidence'):
                if hasattr(req.confidence, 'value'):
                    # It's an enum - map to float
                    conf_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
                    confidence = conf_map.get(str(req.confidence.value).lower(), 0.7)
                else:
                    confidence = float(req.confidence) if req.confidence else 0.7
            
            # Get priority (may not exist)
            priority = "medium"
            if hasattr(req, 'priority'):
                priority = req.priority
            elif req_type in ['compliance', 'prohibition']:
                priority = "high"
            
            requirements.append({
                "id": req.id,
                "text": req.text,
                "section": section,
                "type": req_type,
                "priority": priority,
                "confidence": confidence,
                "source_page": source_page,
                "source_doc": source_doc,
                "keywords": getattr(req, 'keywords', []) or []
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
        # v4.0 FIX: Clear cached outline when reprocessing to prevent stale data
        store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "requirements_graph": result.requirements_graph,
            "stats": stats,
            "amendment_processor": amendment_processor,
            "outline": None  # Clear cached outline - will be regenerated from new requirements
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


# ============== v2.8: Semantic Processing ==============

def process_rfp_semantic_background(rfp_id: str):
    """Background task to process RFP with semantic extraction (v2.8)"""
    rfp = store.get(rfp_id)
    if not rfp:
        return
    
    if not semantic_extractor:
        store.set_status(rfp_id, "error", 0, "Semantic extractor not available")
        store.update(rfp_id, {"status": "error"})
        return
    
    try:
        import time
        start_time = time.time()
        
        # Update status
        store.set_status(rfp_id, "processing", 10, "Reading documents...")
        store.update(rfp_id, {"status": "processing"})
        
        # Get file paths
        file_paths = rfp["file_paths"]
        if not file_paths:
            store.set_status(rfp_id, "error", 0, "No files to process")
            store.update(rfp_id, {"status": "error"})
            return
        
        # Parse documents into text
        store.set_status(rfp_id, "processing", 20, "Parsing documents...")
        
        from agents.enhanced_compliance import MultiFormatParser, DocumentType
        parser = MultiFormatParser()
        
        def infer_doc_type(filename: str) -> DocumentType:
            """Infer document type from filename"""
            fname_lower = filename.lower()
            if 'amendment' in fname_lower:
                return DocumentType.AMENDMENT
            elif 'attachment' in fname_lower or 'exhibit' in fname_lower:
                return DocumentType.ATTACHMENT
            elif 'sow' in fname_lower or 'statement of work' in fname_lower:
                return DocumentType.STATEMENT_OF_WORK
            elif 'pws' in fname_lower or 'performance work statement' in fname_lower:
                return DocumentType.STATEMENT_OF_WORK
            elif 'section_l' in fname_lower or 'instructions' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            elif 'section_m' in fname_lower or 'evaluation' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            elif 'rfp' in fname_lower or 'solicitation' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            else:
                return DocumentType.ATTACHMENT
        
        documents = []
        for file_path in file_paths:
            try:
                import os
                filename = os.path.basename(file_path)
                doc_type = infer_doc_type(filename)
                parsed = parser.parse_file(file_path, doc_type)
                if parsed:
                    documents.append({
                        'text': parsed.full_text,
                        'filename': parsed.filename,
                        'pages': parsed.pages if parsed.pages else [parsed.full_text],
                    })
            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")
        
        if not documents:
            store.set_status(rfp_id, "error", 0, "No documents could be parsed")
            store.update(rfp_id, {"status": "error"})
            return
        
        # Run semantic extraction
        store.set_status(rfp_id, "processing", 40, "Extracting requirements semantically...")
        
        result = semantic_extractor.extract(documents, strict_mode=True)
        
        store.set_status(rfp_id, "processing", 70, "Classifying and scoring...")
        
        # Convert to API format
        requirements = []
        for req in result.requirements:
            requirements.append({
                "id": req.id,
                "text": req.text,
                "raw_text": req.raw_text,
                "section": req.rfp_section.value,
                "section_ref": req.section_reference,
                "type": req.requirement_type.value,
                "priority": req.priority,
                "confidence": req.confidence_score,
                "source_page": req.page_number,
                "source_doc": req.source_document,
                "is_mandatory": req.is_mandatory,
                "binding_keyword": req.binding_keyword,
                "action_verb": req.action_verb,
                "actor": req.actor,
                "subject": req.subject,
                "constraints": req.constraints,
                "references_sections": req.references_sections,
                "references_attachments": req.references_attachments,
                "evaluation_factor": req.related_evaluation_factor,
            })
        
        # Build stats
        duration = time.time() - start_time
        stats = {
            "total": result.stats.get('total', len(requirements)),
            "by_type": result.stats.get('by_type', {}),
            "by_priority": result.stats.get('by_priority', {}),
            "by_section": result.stats.get('by_section', {}),
            "mandatory": result.stats.get('mandatory', 0),
            "desirable": result.stats.get('desirable', 0),
            "processing_time": duration,
            "extractor_version": "semantic_v2.8"
        }
        
        store.set_status(rfp_id, "processing", 90, "Finalizing...")
        
        # Store semantic results
        # v4.0 FIX: Clear cached outline when reprocessing to prevent stale data
        store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "semantic_result": result,  # Keep full semantic result for export
            "stats": stats,
            "evaluation_factors": result.evaluation_factors,
            "warnings": result.warnings,
            "extraction_mode": "semantic",
            "outline": None  # Clear cached outline - will be regenerated from new requirements
        })
        
        store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        store.update(rfp_id, {"status": "error"})


@app.post("/api/rfp/{rfp_id}/process-semantic")
async def process_rfp_semantic(rfp_id: str, background_tasks: BackgroundTasks):
    """
    Start semantic processing of RFP documents (v2.8)
    
    Uses the new SemanticRequirementExtractor for:
    - Better requirement classification (PERFORMANCE vs PROPOSAL INSTRUCTION)
    - Aggressive garbage filtering
    - Proper RFP section mapping (L, M, C, PWS, SOW)
    - Action verb and actor extraction
    - Cross-reference detection
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp["file_paths"]:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if not semantic_extractor:
        raise HTTPException(
            status_code=501, 
            detail="Semantic extraction not available. Use /process endpoint instead."
        )
    
    # Start background processing
    store.set_status(rfp_id, "starting", 0, "Starting semantic processing...")
    background_tasks.add_task(process_rfp_semantic_background, rfp_id)
    
    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"]),
        "mode": "semantic"
    }


# ============== v2.9: Best Practices CTM Processing ==============

def process_rfp_best_practices_background(rfp_id: str):
    """
    Background task to process RFP with Best Practices CTM extraction (v2.9).
    
    Per best practices:
    - Analyzes document structure BEFORE extraction
    - Preserves RFP's own numbering (L.4.B.2, C.3.1)
    - Creates separate L/M/C matrices
    - Extracts complete paragraphs, not sentence fragments
    """
    rfp = store.get(rfp_id)
    if not rfp:
        return
    
    if not best_practices_extractor:
        store.set_status(rfp_id, "error", 0, "Best practices extractor not available")
        store.update(rfp_id, {"status": "error"})
        return
    
    try:
        import time
        start_time = time.time()
        
        # Update status
        store.set_status(rfp_id, "processing", 10, "Analyzing document structure...")
        store.update(rfp_id, {"status": "processing"})
        
        # Get file paths
        file_paths = rfp["file_paths"]
        if not file_paths:
            store.set_status(rfp_id, "error", 0, "No files to process")
            store.update(rfp_id, {"status": "error"})
            return
        
        # Parse documents into text
        store.set_status(rfp_id, "processing", 20, "Parsing documents...")
        
        from agents.enhanced_compliance import MultiFormatParser, DocumentType
        parser = MultiFormatParser()
        
        def infer_doc_type(filename: str) -> DocumentType:
            """Infer document type from filename"""
            fname_lower = filename.lower()
            if 'amendment' in fname_lower:
                return DocumentType.AMENDMENT
            elif 'attachment' in fname_lower or 'exhibit' in fname_lower:
                return DocumentType.ATTACHMENT
            elif 'sow' in fname_lower or 'statement of work' in fname_lower:
                return DocumentType.STATEMENT_OF_WORK
            elif 'pws' in fname_lower or 'performance work statement' in fname_lower:
                return DocumentType.STATEMENT_OF_WORK
            elif 'rfp' in fname_lower or 'solicitation' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            else:
                return DocumentType.ATTACHMENT
        
        documents = []
        for file_path in file_paths:
            try:
                import os
                filename = os.path.basename(file_path)
                doc_type = infer_doc_type(filename)
                parsed = parser.parse_file(file_path, doc_type)
                if parsed:
                    documents.append({
                        'text': parsed.full_text,
                        'filename': parsed.filename,
                        'pages': parsed.pages if parsed.pages else [parsed.full_text],
                    })
            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")
        
        if not documents:
            store.set_status(rfp_id, "error", 0, "No documents could be parsed")
            store.update(rfp_id, {"status": "error"})
            return
        
        # Analyze document structure
        store.set_status(rfp_id, "processing", 40, "Analyzing RFP structure (L/M/C sections)...")
        structure = analyze_rfp_structure(documents)
        
        # Extract requirements with structure awareness
        store.set_status(rfp_id, "processing", 60, "Extracting requirements by section...")
        result = best_practices_extractor.extract(documents, structure)
        
        store.set_status(rfp_id, "processing", 80, "Building compliance matrices...")
        
        # Convert to API format
        requirements = []
        for req in result.all_requirements:
            # Derive priority from binding level (matches CTM export logic)
            priority = "medium"
            if req.binding_level.value in ["Mandatory", "Required"]:
                priority = "high"
            elif req.binding_level.value in ["Desirable", "Informational", "Reference"]:
                priority = "low"
            
            requirements.append({
                "id": req.generated_id,
                "rfp_reference": req.rfp_reference,
                "text": req.full_text,
                "category": req.category.value,
                "type": req.category.value,  # Alias for UI compatibility
                "section": req.source_section.value,
                "subsection": req.source_subsection,
                "binding_level": req.binding_level.value,
                "binding_keyword": req.binding_keyword,
                "page": req.page_number,
                "source_doc": req.source_document,
                "parent_title": req.parent_title,
                "cross_references": req.references_to,
                "priority": priority,  # For UI compatibility
            })
        
        # Build stats
        duration = time.time() - start_time
        
        # Count priorities for UI compatibility
        priority_counts = {"high": 0, "medium": 0, "low": 0}
        for req in requirements:
            priority_counts[req["priority"]] = priority_counts.get(req["priority"], 0) + 1
        
        # Detect if this is a non-UCF RFP (GSA/BPA)
        is_non_ucf = len(result.section_l_requirements) == 0 and (
            len(result.technical_requirements) > 0 or len(result.evaluation_requirements) > 0
        )
        
        stats = {
            "total": len(result.all_requirements),
            "section_l": len(result.section_l_requirements),
            "technical": len(result.technical_requirements),
            "evaluation": len(result.evaluation_requirements),
            "attachment": len(result.attachment_requirements),
            "by_binding_level": result.stats.get('by_binding_level', {}),
            "by_priority": priority_counts,  # For UI compatibility
            "sections_found": result.stats.get('sections_found', []),
            "sow_location": result.stats.get('sow_location'),
            "processing_time": duration,
            "extractor_version": "best_practices_v2.10",
            "is_non_ucf_format": is_non_ucf,  # Flag for UI to show guidance
            "rfp_format_note": (
                "This appears to be a GSA Schedule, BPA, or non-standard RFP. "
                "Submission instructions are in Section M Alignment; "
                "Technical requirements are from PWS/SOW."
            ) if is_non_ucf else None
        }
        
        store.set_status(rfp_id, "processing", 95, "Finalizing...")
        
        # Store results
        # v4.0 FIX: Clear cached outline when reprocessing to prevent stale data
        store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "best_practices_result": result,  # Keep full result for export
            "stats": stats,
            "extraction_mode": "best_practices",
            "outline": None  # Clear cached outline - will be regenerated from new requirements
        })
        
        store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        store.update(rfp_id, {"status": "error"})


@app.post("/api/rfp/{rfp_id}/process-best-practices")
async def process_rfp_best_practices(rfp_id: str, background_tasks: BackgroundTasks):
    """
    Start Best Practices CTM processing of RFP documents (v2.9)
    
    Per federal proposal best practices:
    - Analyzes document structure FIRST (identifies Section L, M, C boundaries)
    - Preserves RFP's own numbering scheme (L.4.B.2, C.3.1.a)
    - Creates THREE distinct matrices: L Compliance, Technical (C/PWS), M Alignment
    - Extracts complete requirement paragraphs (never fragments)
    - Maintains evaluator-friendly formatting
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp["file_paths"]:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if not best_practices_extractor:
        raise HTTPException(
            status_code=501, 
            detail="Best practices extraction not available. Use /process-semantic or /process endpoint instead."
        )
    
    # Start background processing
    store.set_status(rfp_id, "starting", 0, "Starting best practices extraction...")
    background_tasks.add_task(process_rfp_best_practices_background, rfp_id)
    
    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"]),
        "mode": "best_practices"
    }


# ============== v3.0: Resilient Extraction Pipeline ==============

def process_rfp_resilient_background(rfp_id: str):
    """
    Background task to process RFP with Resilient Extraction Pipeline (v3.0).

    Key features:
    - Extract First, Classify Later architecture
    - Multi-layer SOW detection (never misses due to typos)
    - Graduated confidence scores (never binary found/not-found)
    - Quality metrics and anomaly detection
    - Review queue for low-confidence items
    """
    rfp = store.get(rfp_id)
    if not rfp:
        return

    if not resilient_extractor:
        store.set_status(rfp_id, "error", 0, "Resilient extractor not available")
        store.update(rfp_id, {"status": "error"})
        return

    try:
        import time
        start_time = time.time()

        # Log version for deployment verification
        from agents.enhanced_compliance.resilient_extractor import PIPELINE_VERSION
        from agents.enhanced_compliance.universal_extractor import EXTRACTOR_VERSION
        print(f"Processing {rfp_id} with Resilient Extraction v{PIPELINE_VERSION} (extractor v{EXTRACTOR_VERSION})")

        store.set_status(rfp_id, "processing", 10, "Initializing resilient extraction...")
        store.update(rfp_id, {"status": "processing"})

        file_paths = rfp["file_paths"]
        if not file_paths:
            store.set_status(rfp_id, "error", 0, "No files to process")
            store.update(rfp_id, {"status": "error"})
            return

        # Get document metadata from guided upload (if available)
        document_metadata = rfp.get("document_metadata", {})
        print(f"[DEBUG] Guided upload document_metadata: {document_metadata}")

        # Parse documents
        store.set_status(rfp_id, "processing", 20, "Parsing documents...")
        from agents.enhanced_compliance import MultiFormatParser, DocumentType
        parser = MultiFormatParser()

        def get_doc_type_from_metadata(filename: str) -> tuple:
            """
            Get document type from guided upload metadata.
            Returns (DocumentType, content_type) where content_type is used for categorization.
            """
            meta = document_metadata.get(filename, {})
            guided_doc_type = meta.get("doc_type", "auto_detect")

            # Map guided upload doc_type to DocumentType and content categorization
            if guided_doc_type == "sow":
                return (DocumentType.STATEMENT_OF_WORK, "technical")
            elif guided_doc_type in ["instructions", "instructions_only"]:
                return (DocumentType.MAIN_SOLICITATION, "section_l")
            elif guided_doc_type in ["evaluation", "evaluation_only"]:
                return (DocumentType.MAIN_SOLICITATION, "section_m")
            elif guided_doc_type in ["combined_lm", "instructions_evaluation"]:
                return (DocumentType.MAIN_SOLICITATION, "section_lm")
            elif guided_doc_type == "combined_rfp":
                return (DocumentType.MAIN_SOLICITATION, "combined")
            elif guided_doc_type == "amendments":
                return (DocumentType.AMENDMENT, "amendment")
            elif guided_doc_type == "solicitation":
                return (DocumentType.MAIN_SOLICITATION, "cover")
            else:
                return (None, None)  # Fall back to inference

        def infer_doc_type(filename: str) -> DocumentType:
            """Infer document type from filename (with typo tolerance for SOW)"""
            fname_lower = filename.lower()
            if 'amendment' in fname_lower:
                return DocumentType.AMENDMENT
            elif 'attachment' in fname_lower or 'exhibit' in fname_lower:
                return DocumentType.ATTACHMENT
            # SOW detection with typo tolerance
            elif any(x in fname_lower for x in ['sow', 'statement of work', 'stament of work', 'statment of work']):
                return DocumentType.STATEMENT_OF_WORK
            elif any(x in fname_lower for x in ['pws', 'performance work statement', 'performace work']):
                return DocumentType.STATEMENT_OF_WORK
            elif 'section_l' in fname_lower or 'instructions' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            elif 'section_m' in fname_lower or 'evaluation' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            elif 'rfp' in fname_lower or 'solicitation' in fname_lower:
                return DocumentType.MAIN_SOLICITATION
            else:
                return DocumentType.ATTACHMENT

        documents = []
        for file_path in file_paths:
            try:
                import os
                filename = os.path.basename(file_path)

                # First try guided upload metadata, then fall back to inference
                doc_type, content_type = get_doc_type_from_metadata(filename)
                if doc_type is None:
                    doc_type = infer_doc_type(filename)
                    # v3.1: Infer content_type from filename patterns
                    fname_lower = filename.lower()
                    if any(x in fname_lower for x in ['attachment 2', 'attachment_2', 'placement', 'procedure']):
                        content_type = "section_lm"  # Attachment 2 typically contains L/M content
                    elif any(x in fname_lower for x in ['attachment 1', 'attachment_1', 'sow', 'statement']):
                        content_type = "technical"  # Attachment 1 typically contains SOW
                    else:
                        content_type = "auto"

                parsed = parser.parse_file(file_path, doc_type)
                if parsed:
                    documents.append({
                        'text': parsed.full_text,
                        'filename': parsed.filename,
                        'pages': parsed.pages if parsed.pages else [parsed.full_text],
                        'content_type': content_type,  # Pass to extractor for categorization
                    })
            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")

        if not documents:
            store.set_status(rfp_id, "error", 0, "No documents could be parsed")
            store.update(rfp_id, {"status": "error"})
            return

        # Build filename to content_type mapping for post-processing
        doc_content_types = {doc['filename']: doc.get('content_type', 'auto') for doc in documents}

        # Run resilient extraction
        store.set_status(rfp_id, "processing", 40, "Extracting requirements (resilient mode)...")
        result = resilient_extractor.extract_from_parsed(documents, rfp_id)

        store.set_status(rfp_id, "processing", 70, "Building compliance data...")

        # Convert to API format with category assignment based on source document type
        requirements = []
        for req in result.requirements:
            priority = "medium"
            if req.binding_level == "SHALL":
                priority = "high"
            elif req.binding_level == "MAY":
                priority = "low"

            # Determine category based on source document's content_type (from guided upload)
            source_content_type = doc_content_types.get(req.source_document, "auto")
            category = req.category or "general"

            # Override category based on guided upload document type
            if source_content_type == "section_l":
                category = "L_COMPLIANCE"
            elif source_content_type == "section_m":
                category = "EVALUATION"
            elif source_content_type == "section_lm":
                # For combined L+M docs, check text for clues
                text_lower = req.text.lower()
                if any(kw in text_lower for kw in ['evaluat', 'factor', 'rating', 'score', 'criterion', 'assessed']):
                    category = "EVALUATION"
                elif any(kw in text_lower for kw in ['submit', 'page limit', 'font', 'format', 'volume', 'shall include']):
                    category = "L_COMPLIANCE"
                else:
                    category = "L_COMPLIANCE"  # Default to L for combined docs
            elif source_content_type == "technical":
                category = "TECHNICAL"
            elif source_content_type == "combined":
                # For all-in-one RFPs, keep original category or infer
                pass  # Use extractor's category
            elif source_content_type == "auto":
                # Fallback: infer from source document filename when no guided upload metadata
                source_lower = (req.source_document or "").lower()
                # v3.1: Added 'attachment 2' - typically contains Section L/M (Placement Procedures)
                if any(kw in source_lower for kw in ['placement', 'procedure', 'section l', 'section_l', 'instruction', 'attachment 2', 'attachment_2']):
                    # Likely Section L content - check text for L vs M distinction
                    text_lower = req.text.lower()
                    if any(kw in text_lower for kw in ['evaluat', 'factor', 'rating', 'score', 'assessed', 'criteria']):
                        category = "EVALUATION"
                    else:
                        category = "L_COMPLIANCE"
                elif any(kw in source_lower for kw in ['section m', 'section_m', 'evaluation']):
                    category = "EVALUATION"

            requirements.append({
                "id": req.id,
                "text": req.text,
                "section": req.assigned_section or "UNASSIGNED",
                "type": req.category or "general",
                "category": category,  # Add explicit category for outline generator
                "source_content_type": source_content_type,  # v3.1: Save for fallback in outline generation
                "priority": priority,
                "confidence": req.confidence_score,
                "confidence_level": req.confidence.value,
                "source_page": req.source_page,
                "source_doc": req.source_document,
                "rfp_reference": req.rfp_reference,
                "binding_level": req.binding_level,
                "needs_review": req.needs_review,
                "review_reasons": req.review_reasons,
            })

        # Build stats with quality metrics
        metrics = result.quality_metrics

        # Count requirements by category (for outline generator)
        by_category = {}
        for r in requirements:
            cat = r.get("category", "general")
            by_category[cat] = by_category.get(cat, 0) + 1

        print(f"[DEBUG] Requirements by category: {by_category}")

        stats = {
            "total": len(requirements),
            "by_section": metrics.section_counts,
            "by_category": by_category,  # Track L_COMPLIANCE, EVALUATION, TECHNICAL
            "by_confidence": {
                "high": metrics.high_confidence_count,
                "medium": metrics.medium_confidence_count,
                "low": metrics.low_confidence_count,
                "uncertain": metrics.uncertain_count,
            },
            "by_priority": {
                "high": len([r for r in requirements if r["priority"] == "high"]),
                "medium": len([r for r in requirements if r["priority"] == "medium"]),
                "low": len([r for r in requirements if r["priority"] == "low"])
            },
            "processing_time": round(time.time() - start_time, 2),
            "pages_processed": metrics.total_pages,
            "documents_processed": metrics.total_documents,
            "sow_detected": metrics.sow_detected,
            "sow_source": metrics.sow_source,
            "review_queue_size": len(result.review_queue),
            "anomalies": metrics.anomalies,
            "warnings": metrics.warnings,
            "guided_upload_used": bool(document_metadata),  # Track if guided upload was used
        }

        store.set_status(rfp_id, "processing", 95, "Finalizing...")

        # Store results
        # v4.0 FIX: Clear cached outline when reprocessing to prevent stale data
        store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "resilient_result": result.to_dict(),
            "stats": stats,
            "extraction_mode": "resilient_v3",
            "outline": None  # Clear cached outline - will be regenerated from new requirements
        })

        # Log quality summary
        if metrics.anomalies:
            print(f"RFP {rfp_id} ANOMALIES: {metrics.anomalies}")
        if metrics.warnings:
            print(f"RFP {rfp_id} WARNINGS: {metrics.warnings}")

        store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))

    except Exception as e:
        import traceback
        traceback.print_exc()
        store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        store.update(rfp_id, {"status": "error"})


@app.post("/api/rfp/{rfp_id}/process-resilient")
async def process_rfp_resilient(rfp_id: str, background_tasks: BackgroundTasks):
    """
    Start Resilient Extraction processing of RFP documents (v3.0)

    This uses the Extract-First architecture that:
    - Never loses requirements due to classification failures
    - Provides graduated confidence scores
    - Detects quality anomalies automatically
    - Flags low-confidence items for review
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    if not rfp["file_paths"]:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if not RESILIENT_EXTRACTION_AVAILABLE or not resilient_extractor:
        raise HTTPException(
            status_code=501,
            detail="Resilient extraction not available. Use /process endpoint instead."
        )

    background_tasks.add_task(process_rfp_resilient_background, rfp_id)

    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"]),
        "mode": "resilient_v3"
    }


@app.get("/api/rfp/{rfp_id}/quality")
async def get_extraction_quality(rfp_id: str):
    """Get extraction quality metrics and anomalies"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    stats = rfp.get("stats", {})

    return {
        "rfp_id": rfp_id,
        "extraction_mode": rfp.get("extraction_mode", "unknown"),
        "sow_detected": stats.get("sow_detected", False),
        "sow_source": stats.get("sow_source"),
        "section_counts": stats.get("by_section", {}),
        "confidence_distribution": stats.get("by_confidence", {}),
        "review_queue_size": stats.get("review_queue_size", 0),
        "anomalies": stats.get("anomalies", []),
        "warnings": stats.get("warnings", []),
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


# ============== v4.0: Trust Gate - Source Coordinates ==============

@app.get("/api/rfp/{rfp_id}/requirements/{req_id}/source")
async def get_requirement_source(rfp_id: str, req_id: str):
    """
    v4.0 Trust Gate: Get source coordinates for a requirement.

    Returns the exact PDF location (page, bounding box) where this
    requirement was extracted from, enabling one-click source verification.

    Response includes:
    - document_id: PDF file hash
    - page_number: 1-indexed page
    - bounding_box: CSS-ready coordinates for highlighting
    - text_snippet: Original text at that location
    - confidence: Extraction confidence score
    """
    if not TRUST_GATE_AVAILABLE or not coordinate_extractor:
        raise HTTPException(
            status_code=501,
            detail="Trust Gate not available. Install pdfplumber: pip install pdfplumber>=0.10.0"
        )

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Find the requirement
    requirement = None
    for req in rfp["requirements"]:
        if req["id"] == req_id:
            requirement = req
            break

    if not requirement:
        raise HTTPException(status_code=404, detail="Requirement not found")

    # Check if we have pre-computed source coordinates
    if requirement.get("source_coordinates"):
        return {
            "requirement_id": req_id,
            "source": requirement["source_coordinates"],
            "cached": True
        }

    # Otherwise, search the source document for this requirement's text
    source_doc = requirement.get("source_doc", "")
    req_text = requirement.get("text", "")

    if not source_doc or not req_text:
        raise HTTPException(
            status_code=404,
            detail="Cannot locate source: missing source document or requirement text"
        )

    # Find the PDF file path
    file_paths = rfp.get("file_paths", [])
    pdf_path = None
    for fp in file_paths:
        if source_doc in fp and fp.lower().endswith(".pdf"):
            pdf_path = fp
            break

    if not pdf_path:
        # Try fuzzy matching on filename
        for fp in file_paths:
            if fp.lower().endswith(".pdf"):
                pdf_path = fp
                break

    if not pdf_path:
        raise HTTPException(
            status_code=404,
            detail=f"Source PDF not found for document: {source_doc}"
        )

    try:
        # Find requirement location in PDF
        source_coord = coordinate_extractor.find_requirement_location(
            pdf_path,
            req_text,
            context_words=10
        )

        if source_coord:
            return {
                "requirement_id": req_id,
                "source": source_coord.to_dict(),
                "cached": False,
                "highlight_data": {
                    "page_number": source_coord.page_number,
                    "css_position": source_coord.bounding_box.to_css_percent(),
                }
            }
        else:
            # Text not found - return page info if available
            source_page = requirement.get("source_page")
            return {
                "requirement_id": req_id,
                "source": None,
                "fallback": {
                    "page_number": source_page,
                    "document": source_doc,
                    "message": "Exact location not found. Navigate to page manually."
                }
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting source coordinates: {str(e)}"
        )


@app.get("/api/documents/{doc_id}/page/{page_num}/image")
async def get_page_image(doc_id: str, page_num: int, dpi: int = 150):
    """
    v4.0 Trust Gate: Get a rendered PDF page as an image.

    Returns a PNG image of the specified page for display in the
    PDF viewer with requirement highlighting overlay.

    Args:
        doc_id: Document hash (from extraction) or RFP ID
        page_num: 1-indexed page number
        dpi: Resolution (default 150, max 300)

    Returns:
        PNG image of the page
    """
    # Limit DPI for performance
    dpi = min(dpi, 300)

    # Find the document - doc_id could be RFP ID or document hash
    rfp = store.get(doc_id)
    pdf_path = None

    if rfp:
        # doc_id is RFP ID - get first PDF
        for fp in rfp.get("file_paths", []):
            if fp.lower().endswith(".pdf"):
                pdf_path = fp
                break
    else:
        # doc_id might be a document hash - search all RFPs
        for rfp_data in store.list_all():
            for fp in rfp_data.get("file_paths", []):
                if fp.lower().endswith(".pdf"):
                    # Check if this file's hash matches
                    try:
                        import hashlib
                        with open(fp, "rb") as f:
                            file_hash = hashlib.md5()
                            for chunk in iter(lambda: f.read(8192), b""):
                                file_hash.update(chunk)
                            if file_hash.hexdigest()[:16] == doc_id:
                                pdf_path = fp
                                break
                    except Exception:
                        pass
            if pdf_path:
                break

    if not pdf_path:
        raise HTTPException(status_code=404, detail="Document not found")

    if not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF file no longer exists")

    try:
        # Try using pdf2image if available
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                pdf_path,
                first_page=page_num,
                last_page=page_num,
                dpi=dpi
            )
            if not images:
                raise HTTPException(status_code=404, detail=f"Page {page_num} not found")

            # Convert to PNG bytes
            import io
            img_buffer = io.BytesIO()
            images[0].save(img_buffer, format="PNG")
            img_buffer.seek(0)

            return Response(
                content=img_buffer.getvalue(),
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename=page_{page_num}.png"}
            )

        except ImportError:
            # Fallback: use pdfplumber for basic rendering
            if not TRUST_GATE_AVAILABLE:
                raise HTTPException(
                    status_code=501,
                    detail="PDF rendering requires pdf2image or pdfplumber. Install with: pip install pdf2image"
                )

            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    raise HTTPException(status_code=404, detail=f"Page {page_num} not found")

                page = pdf.pages[page_num - 1]
                img = page.to_image(resolution=dpi)

                import io
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                return Response(
                    content=img_buffer.getvalue(),
                    media_type="image/png",
                    headers={"Content-Disposition": f"inline; filename=page_{page_num}.png"}
                )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering page: {str(e)}")


@app.post("/api/rfp/{rfp_id}/highlight")
async def get_requirement_highlight(rfp_id: str, requirement_text: str = Form(...)):
    """
    v4.0 Trust Gate: Get highlight data for requirement text in RFP PDFs.

    Searches all PDF documents in the RFP for the given text and
    returns coordinates for highlighting.

    Args:
        rfp_id: RFP identifier
        requirement_text: Text to find and highlight

    Returns:
        Highlight data including page number and CSS positioning
    """
    if not TRUST_GATE_AVAILABLE or not coordinate_extractor:
        raise HTTPException(
            status_code=501,
            detail="Trust Gate not available"
        )

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Search all PDFs for this text
    results = []
    for fp in rfp.get("file_paths", []):
        if not fp.lower().endswith(".pdf"):
            continue

        try:
            highlight_data = coordinate_extractor.highlight_requirement(
                fp,
                requirement_text
            )
            if highlight_data:
                highlight_data["document"] = Path(fp).name
                results.append(highlight_data)
        except Exception as e:
            # Continue searching other documents
            pass

    if results:
        return {
            "found": True,
            "results": results,
            "primary": results[0]  # First match is primary
        }
    else:
        return {
            "found": False,
            "message": "Text not found in any RFP documents"
        }


# ============== v4.0: Strategy Engine ==============

@app.post("/api/rfp/{rfp_id}/strategy")
async def generate_strategy(rfp_id: str):
    """
    v4.0 Strategy Engine: Generate win themes and strategy for RFP.

    Analyzes evaluation factors (Section M) and requirements to produce:
    - Ranked win themes with discriminators
    - Suggested page allocations per theme
    - Ghosting language library
    - Storyboard outline

    Requires processed RFP with requirements extracted.
    """
    if not STRATEGY_AGENT_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Strategy Agent not available"
        )

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    requirements = rfp.get("requirements", [])
    if not requirements:
        raise HTTPException(
            status_code=400,
            detail="No requirements found. Process RFP first."
        )

    try:
        # Initialize strategy agent
        strategy_agent = StrategyAgent()

        # Get evaluation criteria (Section M requirements)
        eval_requirements = [
            r for r in requirements
            if r.get("category") == "EVALUATION" or r.get("section", "").upper() == "M"
        ]

        # Get technical requirements for capability mapping
        tech_requirements = [
            r for r in requirements
            if r.get("category") == "TECHNICAL" or r.get("section", "").upper() in ["C", "SOW", "PWS"]
        ]

        # Run strategy analysis
        rfp_data = {
            "id": rfp_id,
            "name": rfp.get("name", ""),
            "evaluation_requirements": eval_requirements,
            "technical_requirements": tech_requirements,
            "all_requirements": requirements,
        }

        strategy_result = strategy_agent.analyze(rfp_data)

        # Store strategy in RFP
        store.update(rfp_id, {"strategy": strategy_result})

        return {
            "status": "generated",
            "rfp_id": rfp_id,
            "strategy": strategy_result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Strategy generation failed: {str(e)}"
        )


@app.get("/api/rfp/{rfp_id}/strategy")
async def get_strategy(rfp_id: str):
    """Get previously generated strategy for RFP"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    strategy = rfp.get("strategy")
    if not strategy:
        raise HTTPException(
            status_code=404,
            detail="No strategy generated. Use POST /api/rfp/{rfp_id}/strategy first."
        )

    return {
        "rfp_id": rfp_id,
        "strategy": strategy
    }


@app.post("/api/rfp/{rfp_id}/competitive-analysis")
async def analyze_competitors(
    rfp_id: str,
    competitors: Optional[str] = Form(None)
):
    """
    v4.0 Strategy Engine: Analyze competitive landscape.

    Performs competitor analysis including:
    - Competitor profiling (strengths, weaknesses)
    - Theme prediction for competitors
    - Ghosting opportunity identification
    - Win probability factors

    Args:
        rfp_id: RFP identifier
        competitors: Optional JSON string of known competitors
            Format: [{"name": "...", "is_incumbent": true, "strengths": [...], "weaknesses": [...]}]

    Returns:
        Comprehensive competitive analysis with ghosting library
    """
    if not STRATEGY_AGENT_AVAILABLE or not CompetitorAnalyzer:
        raise HTTPException(
            status_code=501,
            detail="Competitive analysis not available"
        )

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    requirements = rfp.get("requirements", [])

    # Parse competitors JSON if provided
    known_competitors = []
    if competitors:
        try:
            known_competitors = json.loads(competitors)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid competitors JSON")

    try:
        analyzer = CompetitorAnalyzer(use_llm=True)

        # Build RFP data for analysis
        rfp_data = {
            "id": rfp_id,
            "requirements": requirements,
            "evaluation_criteria": [
                r for r in requirements
                if r.get("category") == "EVALUATION" or r.get("section", "").upper() == "M"
            ]
        }

        analysis = analyzer.analyze_competitive_landscape(
            rfp_data,
            known_competitors=known_competitors
        )

        # Store analysis in RFP
        store.update(rfp_id, {"competitive_analysis": analysis})

        return {
            "status": "analyzed",
            "rfp_id": rfp_id,
            "analysis": analysis
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Competitive analysis failed: {str(e)}"
        )


@app.get("/api/rfp/{rfp_id}/ghosting-library")
async def get_ghosting_library(rfp_id: str):
    """
    v4.0: Get ghosting language library for proposal writing.

    Returns ready-to-use ghosting statements aligned with
    evaluation criteria and competitor weaknesses.
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Check if competitive analysis exists
    comp_analysis = rfp.get("competitive_analysis")
    if comp_analysis and comp_analysis.get("ghosting_library"):
        return {
            "rfp_id": rfp_id,
            "ghosting_library": comp_analysis["ghosting_library"],
            "source": "competitive_analysis"
        }

    # Check if strategy has ghosting language
    strategy = rfp.get("strategy")
    if strategy:
        win_themes = strategy.get("win_themes", [])
        ghosting_from_themes = []
        for theme in win_themes:
            if theme.get("ghosting_language"):
                ghosting_from_themes.append({
                    "theme_id": theme.get("id"),
                    "language": theme["ghosting_language"],
                    "eval_criteria_link": theme.get("linked_eval_criteria", [])
                })
        if ghosting_from_themes:
            return {
                "rfp_id": rfp_id,
                "ghosting_library": ghosting_from_themes,
                "source": "win_themes"
            }

    raise HTTPException(
        status_code=404,
        detail="No ghosting library available. Run competitive analysis or strategy generation first."
    )


# ============== v4.0: Drafting Workflow ==============

@app.post("/api/rfp/{rfp_id}/draft")
async def start_drafting(
    rfp_id: str,
    requirement_id: str = Form(...),
    target_word_count: int = Form(250),
    background_tasks: BackgroundTasks = None
):
    """
    v4.0 Drafting Agent: Generate proposal content for a requirement.

    Uses the LangGraph F-B-P workflow to:
    1. Research company library for evidence
    2. Structure Feature-Benefit-Proof blocks
    3. Generate narrative prose
    4. Quality check and revise

    Args:
        rfp_id: RFP identifier
        requirement_id: Specific requirement to draft for
        target_word_count: Target length for the draft

    Returns:
        Draft content with quality scores and F-B-P structure
    """
    if not DRAFTING_WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Drafting workflow not available. Install langgraph: pip install langgraph"
        )

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Find the requirement
    requirements = rfp.get("requirements", [])
    requirement = None
    for req in requirements:
        if req.get("id") == requirement_id:
            requirement = req
            break

    if not requirement:
        raise HTTPException(status_code=404, detail="Requirement not found")

    # Get strategy (win themes) if available
    strategy = rfp.get("strategy", {})
    win_themes = strategy.get("win_themes", [])
    primary_theme = win_themes[0] if win_themes else None

    try:
        # Run the drafting workflow
        result = run_drafting_workflow(
            requirement=requirement,
            win_theme=primary_theme,
            target_word_count=target_word_count,
        )

        # Store draft in RFP
        drafts = rfp.get("drafts", {})
        drafts[requirement_id] = {
            "draft_text": result.get("draft_text", ""),
            "quality_scores": result.get("quality_scores", {}),
            "fbp_blocks": result.get("fbp_blocks", []),
            "revision_count": result.get("revision_count", 0),
            "word_count": result.get("draft_word_count", 0),
            "generated_at": datetime.now().isoformat()
        }
        store.update(rfp_id, {"drafts": drafts})

        return {
            "status": "completed",
            "requirement_id": requirement_id,
            "draft": result.get("draft_text", ""),
            "word_count": result.get("draft_word_count", 0),
            "quality_scores": result.get("quality_scores", {}),
            "fbp_blocks": result.get("fbp_blocks", []),
            "approved": result.get("approved", False),
            "langgraph_used": LANGGRAPH_AVAILABLE
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Drafting failed: {str(e)}"
        )


@app.get("/api/rfp/{rfp_id}/drafts")
async def get_drafts(rfp_id: str):
    """Get all generated drafts for an RFP"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    return {
        "rfp_id": rfp_id,
        "drafts": rfp.get("drafts", {}),
        "count": len(rfp.get("drafts", {}))
    }


@app.get("/api/rfp/{rfp_id}/drafts/{requirement_id}")
async def get_draft(rfp_id: str, requirement_id: str):
    """Get a specific draft by requirement ID"""
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    drafts = rfp.get("drafts", {})
    draft = drafts.get(requirement_id)

    if not draft:
        raise HTTPException(
            status_code=404,
            detail="No draft found for this requirement. Use POST /api/rfp/{rfp_id}/draft first."
        )

    return {
        "rfp_id": rfp_id,
        "requirement_id": requirement_id,
        "draft": draft
    }


@app.post("/api/rfp/{rfp_id}/drafts/{requirement_id}/feedback")
async def submit_draft_feedback(
    rfp_id: str,
    requirement_id: str,
    feedback: str = Form(...),
    approved: bool = Form(False)
):
    """
    Submit feedback on a draft and trigger revision.

    This is the human-in-the-loop endpoint that allows users to
    provide feedback that gets incorporated into the next revision.
    """
    if not DRAFTING_WORKFLOW_AVAILABLE:
        raise HTTPException(status_code=501, detail="Drafting workflow not available")

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    drafts = rfp.get("drafts", {})
    draft = drafts.get(requirement_id)

    if not draft:
        raise HTTPException(status_code=404, detail="No draft found for this requirement")

    if approved:
        # Mark as approved
        draft["approved"] = True
        draft["feedback_history"] = draft.get("feedback_history", [])
        draft["feedback_history"].append({
            "feedback": feedback,
            "approved": True,
            "timestamp": datetime.now().isoformat()
        })
        store.update(rfp_id, {"drafts": drafts})

        return {
            "status": "approved",
            "requirement_id": requirement_id,
            "draft": draft
        }

    # Find the requirement
    requirement = None
    for req in rfp.get("requirements", []):
        if req.get("id") == requirement_id:
            requirement = req
            break

    if not requirement:
        raise HTTPException(status_code=404, detail="Requirement not found")

    # Get strategy
    strategy = rfp.get("strategy", {})
    win_themes = strategy.get("win_themes", [])
    primary_theme = win_themes[0] if win_themes else None

    try:
        # Re-run workflow with feedback
        result = run_drafting_workflow(
            requirement=requirement,
            win_theme=primary_theme,
            target_word_count=draft.get("word_count", 250),
        )

        # Update draft
        draft["draft_text"] = result.get("draft_text", "")
        draft["quality_scores"] = result.get("quality_scores", {})
        draft["revision_count"] = draft.get("revision_count", 0) + 1
        draft["word_count"] = result.get("draft_word_count", 0)
        draft["revised_at"] = datetime.now().isoformat()
        draft["feedback_history"] = draft.get("feedback_history", [])
        draft["feedback_history"].append({
            "feedback": feedback,
            "approved": False,
            "timestamp": datetime.now().isoformat()
        })

        store.update(rfp_id, {"drafts": drafts})

        return {
            "status": "revised",
            "requirement_id": requirement_id,
            "draft": draft,
            "revision_count": draft["revision_count"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Revision failed: {str(e)}"
        )


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
    
    output_path = OUTPUT_DIR / f"{rfp_id}_ComplianceMatrix.xlsx"
    
    # Check extraction mode and use appropriate exporter
    extraction_mode = rfp.get("extraction_mode", "legacy")
    
    if extraction_mode == "best_practices" and rfp.get("best_practices_result") and best_practices_exporter:
        # v2.9: Use Best Practices CTM Exporter
        best_practices_exporter.export(
            rfp["best_practices_result"],
            str(output_path),
            solicitation_number=rfp.get("solicitation_number", rfp_id),
            title=rfp.get("name", "RFP Analysis")
        )
    elif extraction_mode == "semantic" and rfp.get("semantic_result") and semantic_ctm_exporter:
        # v2.8: Use semantic CTM exporter
        semantic_ctm_exporter.export(
            rfp["semantic_result"],
            str(output_path),
            solicitation_number=rfp.get("solicitation_number", rfp_id),
            title=rfp.get("name", "RFP Analysis")
        )
    else:
        # Legacy export path
        # Create a result-like object for export with all required attributes
        class ExportResult:
            def __init__(self, reqs, reqs_graph, stats):
                self.requirements_graph = reqs_graph or {}
                self.compliance_matrix = []  # Will build from requirements
                self.stats = stats or {}
                self.duration_seconds = stats.get("processing_time", 0) if stats else 0
                self.cross_reference_count = 0
                self.extraction_coverage = stats.get("coverage", 0.7) if stats else 0.7
                
                # Build compliance matrix rows from requirements
                for req in (reqs or []):
                    class MatrixRow:
                        pass
                    row = MatrixRow()
                    row.requirement_id = req.get("id", "")
                    row.requirement_text = req.get("text", "")
                    row.section_reference = req.get("section", "")
                    row.section_type = "C"  # Default to section C
                    row.requirement_type = req.get("type", "performance")
                    row.priority = req.get("priority", "Medium").capitalize()
                    row.compliance_status = "Not Started"
                    row.response_text = ""
                    row.proposal_section = ""
                    row.assigned_owner = ""
                    row.evidence_required = []
                    row.related_requirements = []
                    row.evaluation_factor = None
                    row.risk_if_non_compliant = ""
                    row.notes = ""
                    self.compliance_matrix.append(row)
        
        result = ExportResult(rfp["requirements"], rfp.get("requirements_graph", {}), rfp["stats"])
        
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
            # Safely extract attributes from RequirementNode
            section = ""
            source_page = None
            source_doc = ""
            
            if hasattr(req, 'source') and req.source:
                section = getattr(req.source, 'section_id', '') or ''
                source_page = getattr(req.source, 'page_number', None)
                source_doc = getattr(req.source, 'document_name', '') or ''
            
            req_type = "performance"
            if hasattr(req, 'requirement_type'):
                req_type = req.requirement_type.value if hasattr(req.requirement_type, 'value') else str(req.requirement_type)
            elif hasattr(req, 'req_type'):
                req_type = req.req_type
            
            confidence = 0.7
            if hasattr(req, 'confidence'):
                if hasattr(req.confidence, 'value'):
                    conf_map = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
                    confidence = conf_map.get(str(req.confidence.value).lower(), 0.7)
                else:
                    confidence = float(req.confidence) if req.confidence else 0.7
            
            priority = "medium"
            if hasattr(req, 'priority'):
                priority = req.priority
            elif req_type in ['compliance', 'prohibition']:
                priority = "high"
            
            updated_reqs.append({
                "id": req.id,
                "text": req.text,
                "section": section,
                "type": req_type,
                "priority": priority,
                "confidence": confidence,
                "source_page": source_page,
                "source_doc": source_doc,
                "keywords": getattr(req, 'keywords', []) or []
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
    """
    Generate proposal outline from RFP.
    
    v2.10: Uses SmartOutlineGenerator which leverages already-extracted
    compliance matrix data for better accuracy.
    """
    from agents.enhanced_compliance.smart_outline_generator import SmartOutlineGenerator
    
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    # Check if we have compliance matrix data
    requirements = rfp.get("requirements", [])
    stats = rfp.get("stats", {})
    
    if not requirements:
        raise HTTPException(
            status_code=400, 
            detail="No compliance matrix data. Process RFP first using /process-best-practices"
        )
    
    try:
        generator = SmartOutlineGenerator()
        
        # Separate requirements by category
        section_l = [r for r in requirements if r.get("category") == "L_COMPLIANCE" 
                     or r.get("section", "").upper() == "L"]
        section_m = [r for r in requirements if r.get("category") == "EVALUATION"
                     or r.get("section", "").upper() == "M"]
        technical = [r for r in requirements if r.get("category") == "TECHNICAL"
                     or r.get("section", "").upper() in ["C", "PWS", "SOW"]]
        
        # If section_l is empty, treat section_m as containing submission instructions (GSA/BPA)
        if not section_l and section_m:
            section_l = section_m  # GSA/BPA format has instructions in eval section
        
        # Generate outline from compliance matrix data
        outline = generator.generate_from_compliance_matrix(
            section_l_requirements=section_l,
            section_m_requirements=section_m,
            technical_requirements=technical,
            stats=stats
        )
        
        # Convert to JSON
        outline_data = generator.to_json(outline)
        
        # Store outline
        store.update(rfp_id, {"outline": outline_data})
        
        return {
            "status": "generated",
            "outline": outline_data
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Outline generation failed: {str(e)}")


@app.get("/api/rfp/{rfp_id}/outline")
async def get_outline(rfp_id: str, format: str = "json"):
    """Get proposal outline"""
    from agents.enhanced_compliance.smart_outline_generator import SmartOutlineGenerator
    
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    outline = rfp.get("outline")
    
    if not outline:
        # Generate if not exists - need compliance matrix data
        requirements = rfp.get("requirements", [])
        stats = rfp.get("stats", {})
        
        if not requirements:
            raise HTTPException(
                status_code=400, 
                detail="No compliance matrix data. Process RFP first using /process-best-practices"
            )
        
        generator = SmartOutlineGenerator()
        
        # Separate requirements by category
        section_l = [r for r in requirements if r.get("category") == "L_COMPLIANCE" 
                     or r.get("section", "").upper() == "L"]
        section_m = [r for r in requirements if r.get("category") == "EVALUATION"
                     or r.get("section", "").upper() == "M"]
        technical = [r for r in requirements if r.get("category") == "TECHNICAL"
                     or r.get("section", "").upper() in ["C", "PWS", "SOW"]]
        
        if not section_l and section_m:
            section_l = section_m
        
        outline_obj = generator.generate_from_compliance_matrix(
            section_l_requirements=section_l,
            section_m_requirements=section_m,
            technical_requirements=technical,
            stats=stats
        )
        outline = generator.to_json(outline_obj)
        store.update(rfp_id, {"outline": outline})
    
    return {"format": "json", "outline": outline}


@app.get("/api/rfp/{rfp_id}/outline/export")
async def export_annotated_outline(rfp_id: str):
    """Export annotated proposal outline as Word document."""
    from agents.enhanced_compliance.smart_outline_generator import SmartOutlineGenerator
    
    if not ANNOTATED_OUTLINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Annotated outline export not available. Install Node.js and docx package."
        )
    
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # v4.0: Debug logging for data isolation tracing
    print(f"[DEBUG] export_annotated_outline called for RFP: {rfp_id}")
    print(f"[DEBUG] RFP name: {rfp.get('name')}, solicitation: {rfp.get('solicitation_number')}")
    print(f"[DEBUG] Requirements count in store: {len(rfp.get('requirements', []))}")

    outline = rfp.get("outline")
    if outline:
        print(f"[DEBUG] Using CACHED outline (already generated)")
    if not outline:
        generator = SmartOutlineGenerator()
        # v3.1 FIX: Use category field (set by extraction), not section field
        # The category field contains: L_COMPLIANCE, EVALUATION, TECHNICAL
        # The section field contains SOW references like "2.1" which never start with L/M
        section_l = [r for r in rfp.get("requirements", []) if r.get("category") == "L_COMPLIANCE"]
        section_m = [r for r in rfp.get("requirements", []) if r.get("category") == "EVALUATION"]

        # Fallback: also include requirements from documents categorized as section_lm
        if not section_l:
            section_l = [r for r in rfp.get("requirements", [])
                        if r.get("source_content_type") in ["section_l", "section_lm"]]
        if not section_m:
            section_m = [r for r in rfp.get("requirements", [])
                        if r.get("source_content_type") in ["section_m", "section_lm"]]

        # v4.0 FIX: Get technical requirements (not section L or M)
        technical = [r for r in rfp.get("requirements", [])
                    if r.get("category") not in ["L_COMPLIANCE", "EVALUATION"]]
        stats = {"is_non_ucf_format": len(section_l) == 0}

        print(f"[DEBUG] REGENERATING outline from {len(rfp.get('requirements', []))} requirements")
        print(f"[DEBUG] Outline generation - section_l count: {len(section_l)}, section_m count: {len(section_m)}, technical count: {len(technical)}")

        # Log sample requirement to verify data isolation
        if section_l:
            print(f"[DEBUG] Sample section_l requirement: {section_l[0].get('text', '')[:100]}...")

        outline_obj = generator.generate_from_compliance_matrix(
            section_l_requirements=section_l,
            section_m_requirements=section_m,
            technical_requirements=technical,
            stats=stats
        )
        outline = generator.to_json(outline_obj)
        store.update(rfp_id, {"outline": outline})
    
    requirements = rfp.get("requirements", [])

    # v3.2: Use proper proposal title, not source document filename
    # The RFP name might be "Attachment 1. Stament Of Work" which is wrong
    # v4.0 FIX: Use 'or' to handle None values (not just missing keys)
    solicitation_number = rfp.get("solicitation_number") or rfp_id
    rfp_name = rfp.get("name") or ""

    # Don't use source document names as title
    if any(x in rfp_name.lower() for x in ["attachment", "sow", "statement of work", "stament"]):
        rfp_title = f"Proposal Response to {solicitation_number}"
    else:
        rfp_title = rfp_name or f"Proposal Response to {solicitation_number}"

    config = AnnotatedOutlineConfig(
        rfp_title=rfp_title,
        solicitation_number=solicitation_number,
        due_date=outline.get("submission", {}).get("due_date", "TBD"),
        submission_method=outline.get("submission", {}).get("method", "Not Specified"),
        total_pages=outline.get("total_pages"),
        company_name="[Your Company Name]"
    )
    
    try:
        exporter = AnnotatedOutlineExporter()
        # v4.0 FIX: Explicitly handle None format_requirements
        format_reqs = outline.get("format_requirements") or {}

        # Debug: Log what we're passing to the exporter
        print(f"[DEBUG] export_annotated_outline: outline keys = {list(outline.keys()) if outline else 'None'}")
        print(f"[DEBUG] export_annotated_outline: requirements count = {len(requirements) if requirements else 'None'}")
        print(f"[DEBUG] export_annotated_outline: format_reqs = {format_reqs}")
        print(f"[DEBUG] export_annotated_outline: volumes = {outline.get('volumes', 'MISSING')}")

        doc_bytes = exporter.export(outline, requirements, format_reqs, config)
        
        # v3.2: Use solicitation number for filename, not source document name
        safe_name = "".join(c for c in solicitation_number if c.isalnum() or c in " -_")[:50]
        filename = f"{safe_name}_Annotated_Outline.docx"
        
        return Response(
            content=doc_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate annotated outline: {str(e)}")



# ============== Company Library API ==============

# Import company library components
try:
    from agents.enhanced_compliance.company_library import (
        CompanyLibrary,
        CompanyLibraryParser,
        DocumentType,
    )
    COMPANY_LIBRARY_AVAILABLE = True
except ImportError:
    COMPANY_LIBRARY_AVAILABLE = False

# Initialize company library
if COMPANY_LIBRARY_AVAILABLE:
    company_library = CompanyLibrary(str(OUTPUT_DIR / "company_library"))
else:
    company_library = None


@app.get("/api/library")
async def get_library_status():
    """Get company library status and summary"""
    if not COMPANY_LIBRARY_AVAILABLE:
        return {"available": False, "error": "Company library not available"}
    
    documents = company_library.list_documents()
    profile = company_library.get_profile()
    
    return {
        "available": True,
        "document_count": len(documents),
        "documents": documents,
        "profile_summary": {
            "company_name": profile.get("company_name", ""),
            "capabilities_count": len(profile.get("capabilities", [])),
            "differentiators_count": len(profile.get("differentiators", [])),
            "past_performance_count": len(profile.get("past_performance", [])),
            "key_personnel_count": len(profile.get("key_personnel", [])),
        }
    }


@app.get("/api/library/profile")
async def get_company_profile():
    """Get full company profile aggregated from all documents"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    return company_library.get_profile()


@app.post("/api/library/upload")
async def upload_library_document(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None)
):
    """
    Upload a document to the company library
    
    Supported types: capabilities, past_performance, resume, technical_approach, corporate_info
    """
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    # Validate file type
    allowed_extensions = [".docx", ".doc", ".pdf", ".txt", ".md", ".rtf"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_path = UPLOAD_DIR / f"lib_{uuid.uuid4().hex[:8]}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse document type
        doc_type = None
        if document_type:
            try:
                doc_type = DocumentType(document_type)
            except ValueError:
                pass  # Let parser auto-detect
        
        # Add to library
        parsed_doc = company_library.add_document(str(temp_path), doc_type)
        
        return {
            "success": True,
            "document": parsed_doc.to_dict(),
            "message": f"Document '{file.filename}' added to library as {parsed_doc.document_type.value}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)


@app.get("/api/library/documents")
async def list_library_documents():
    """List all documents in the company library"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    return {"documents": company_library.list_documents()}


@app.get("/api/library/documents/{doc_id}")
async def get_library_document(doc_id: str):
    """Get details of a specific document"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    doc = company_library.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return doc.to_dict()


@app.delete("/api/library/documents/{doc_id}")
async def delete_library_document(doc_id: str):
    """Remove a document from the library"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    if company_library.remove_document(doc_id):
        return {"success": True, "message": f"Document {doc_id} removed"}
    else:
        raise HTTPException(status_code=404, detail="Document not found")


@app.get("/api/library/search")
async def search_library(query: str):
    """
    Search the company library for relevant content
    
    Returns matching capabilities, past performance, key personnel, etc.
    """
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    results = company_library.search(query)
    return {"query": query, "results": results}


@app.get("/api/library/capabilities")
async def get_capabilities():
    """Get all extracted capabilities"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    profile = company_library.get_profile()
    return {"capabilities": profile.get("capabilities", [])}


@app.get("/api/library/differentiators")
async def get_differentiators():
    """Get all extracted differentiators"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    profile = company_library.get_profile()
    return {"differentiators": profile.get("differentiators", [])}


@app.get("/api/library/past-performance")
async def get_past_performance():
    """Get all extracted past performance records"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")
    
    profile = company_library.get_profile()
    return {"past_performance": profile.get("past_performance", [])}


@app.get("/api/library/key-personnel")
async def get_key_personnel():
    """Get all extracted key personnel"""
    if not COMPANY_LIBRARY_AVAILABLE:
        raise HTTPException(status_code=500, detail="Company library not available")

    profile = company_library.get_profile()
    return {"key_personnel": profile.get("key_personnel", [])}


# ============== v4.0: Vector Search (pgvector) ==============

# Import vector store and update global flag
try:
    from api.vector_store import (
        VectorStore, get_vector_store, SearchResult,
        VECTOR_STORE_AVAILABLE as _VS_AVAILABLE,
    )
    VECTOR_STORE_AVAILABLE = _VS_AVAILABLE
except ImportError:
    # VECTOR_STORE_AVAILABLE already False from line 176
    get_vector_store = None


@app.get("/api/library/vector-search")
async def vector_search_library(
    query: str,
    top_k: int = 10,
    types: Optional[str] = None,  # Comma-separated: capability,past_performance,key_personnel,differentiator
):
    """
    v4.0: Semantic search across Company Library using pgvector.

    Uses embedding similarity to find the most relevant content for proposal drafting.

    Args:
        query: Natural language search query (e.g., "cloud migration experience for federal agencies")
        top_k: Number of results to return (default: 10)
        types: Optional filter for content types (comma-separated)

    Returns:
        List of search results with similarity scores
    """
    if not VECTOR_STORE_AVAILABLE:
        # Fall back to keyword search
        if COMPANY_LIBRARY_AVAILABLE:
            results = company_library.search(query)
            return {
                "query": query,
                "results": results,
                "search_type": "keyword",
                "message": "Vector search not available, using keyword search"
            }
        raise HTTPException(
            status_code=501,
            detail="Vector search not available. Configure DATABASE_URL and OPENAI_API_KEY."
        )

    store = await get_vector_store()

    # Parse type filter
    include_types = None
    if types:
        include_types = [t.strip() for t in types.split(",")]

    results = await store.search_all(query, top_k=top_k, include_types=include_types)

    return {
        "query": query,
        "results": [
            {
                "id": r.id,
                "type": r.content_type,
                "name": r.name,
                "description": r.description,
                "similarity": round(r.similarity_score, 4),
            }
            for r in results
        ],
        "search_type": "vector",
        "top_k": top_k,
    }


@app.post("/api/library/vector/capabilities")
async def add_capability_vector(
    name: str = Form(...),
    description: str = Form(...),
    category: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),  # Comma-separated
    company_id: Optional[str] = Form(None),
):
    """
    v4.0: Add a capability to the vector store with auto-generated embedding.

    The capability will be searchable via semantic similarity.
    """
    if not VECTOR_STORE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Vector store not available")

    store = await get_vector_store()

    # Parse keywords
    keyword_list = None
    if keywords:
        keyword_list = [k.strip() for k in keywords.split(",")]

    # Use default company_id if not provided
    cid = company_id or "default"

    cap_id = await store.add_capability(
        company_id=cid,
        name=name,
        description=description,
        category=category,
        keywords=keyword_list,
    )

    if cap_id:
        return {
            "status": "created",
            "id": cap_id,
            "name": name,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to add capability")


@app.post("/api/library/vector/past-performance")
async def add_past_performance_vector(
    project_name: str = Form(...),
    description: str = Form(...),
    client_name: Optional[str] = Form(None),
    client_agency: Optional[str] = Form(None),
    contract_number: Optional[str] = Form(None),
    contract_value: Optional[float] = Form(None),
    period_of_performance: Optional[str] = Form(None),
    company_id: Optional[str] = Form(None),
):
    """
    v4.0: Add a past performance record to the vector store.

    Enables semantic search for relevant project experience during proposal drafting.
    """
    if not VECTOR_STORE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Vector store not available")

    store = await get_vector_store()
    cid = company_id or "default"

    pp_id = await store.add_past_performance(
        company_id=cid,
        project_name=project_name,
        description=description,
        client_name=client_name,
        client_agency=client_agency,
        contract_number=contract_number,
        contract_value=contract_value,
        period_of_performance=period_of_performance,
    )

    if pp_id:
        return {
            "status": "created",
            "id": pp_id,
            "project_name": project_name,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to add past performance")


@app.post("/api/library/vector/key-personnel")
async def add_key_personnel_vector(
    name: str = Form(...),
    bio: str = Form(...),
    title: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    years_experience: Optional[int] = Form(None),
    clearance_level: Optional[str] = Form(None),
    certifications: Optional[str] = Form(None),  # Comma-separated
    expertise_areas: Optional[str] = Form(None),  # Comma-separated
    company_id: Optional[str] = Form(None),
):
    """
    v4.0: Add key personnel to the vector store.

    Enables semantic search for relevant team expertise during proposal staffing.
    """
    if not VECTOR_STORE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Vector store not available")

    store = await get_vector_store()
    cid = company_id or "default"

    # Parse arrays
    cert_list = [c.strip() for c in certifications.split(",")] if certifications else None
    expertise_list = [e.strip() for e in expertise_areas.split(",")] if expertise_areas else None

    kp_id = await store.add_key_personnel(
        company_id=cid,
        name=name,
        bio=bio,
        title=title,
        role=role,
        years_experience=years_experience,
        clearance_level=clearance_level,
        certifications=cert_list,
        expertise_areas=expertise_list,
    )

    if kp_id:
        return {
            "status": "created",
            "id": kp_id,
            "name": name,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to add key personnel")


@app.post("/api/library/vector/differentiators")
async def add_differentiator_vector(
    title: str = Form(...),
    description: str = Form(...),
    category: Optional[str] = Form(None),
    proof_points: Optional[str] = Form(None),  # Comma-separated
    competitor_comparison: Optional[str] = Form(None),
    company_id: Optional[str] = Form(None),
):
    """
    v4.0: Add a differentiator/discriminator to the vector store.

    Enables semantic search for competitive advantages during theme development.
    """
    if not VECTOR_STORE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Vector store not available")

    store = await get_vector_store()
    cid = company_id or "default"

    # Parse proof points
    proof_list = [p.strip() for p in proof_points.split(",")] if proof_points else None

    diff_id = await store.add_differentiator(
        company_id=cid,
        title=title,
        description=description,
        category=category,
        proof_points=proof_list,
        competitor_comparison=competitor_comparison,
    )

    if diff_id:
        return {
            "status": "created",
            "id": diff_id,
            "title": title,
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to add differentiator")


# ============== v4.1: Team Workspaces & Role-Based Access ==============

from api.database import (
    UserModel, TeamModel, TeamMembershipModel, ActivityLogModel, APIKeyModel, UserRole
)
import uuid
import hashlib
import secrets


def generate_id() -> str:
    """Generate a short unique ID"""
    return str(uuid.uuid4())[:8].upper()


def hash_password(password: str) -> str:
    """Simple password hashing (use bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash


# Current user context (simplified - use JWT in production)
_current_user: Optional[Dict] = None


def get_current_user() -> Optional[Dict]:
    """Get current user from context"""
    return _current_user


def require_role(required_role: str, team_id: str = None):
    """Decorator to require a specific role"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")
            # In production, check user's role for the team
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@app.post("/api/auth/register")
async def register_user(
    email: str = Form(...),
    name: str = Form(...),
    password: str = Form(...),
):
    """Register a new user"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        # Check if email exists
        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = generate_id()
        user = UserModel(
            id=user_id,
            email=email,
            name=name,
            password_hash=hash_password(password),
        )
        session.add(user)
        await session.flush()

        return {
            "success": True,
            "user": user.to_dict(),
            "message": "Registration successful",
        }


@app.post("/api/auth/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...),
):
    """Login user (simplified - use JWT in production)"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()

        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user.last_login = datetime.utcnow()
        await session.flush()

        # In production, return JWT token
        return {
            "success": True,
            "user": user.to_dict(),
            "token": f"demo-token-{user.id}",  # Replace with JWT
        }


@app.get("/api/users/me")
async def get_current_user_info():
    """Get current user info"""
    user = get_current_user()
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.post("/api/teams")
async def create_team(
    name: str = Form(...),
    slug: str = Form(None),
    description: str = Form(None),
):
    """Create a new team workspace"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        # Generate slug if not provided
        if not slug:
            slug = name.lower().replace(" ", "-").replace("_", "-")
            slug = "".join(c for c in slug if c.isalnum() or c == "-")

        # Check if slug exists
        result = await session.execute(
            select(TeamModel).where(TeamModel.slug == slug)
        )
        if result.scalar_one_or_none():
            slug = f"{slug}-{generate_id().lower()}"

        team_id = generate_id()
        team = TeamModel(
            id=team_id,
            name=name,
            slug=slug,
            description=description,
        )
        session.add(team)
        await session.flush()

        return {
            "success": True,
            "team": team.to_dict(),
        }


@app.get("/api/teams")
async def list_teams():
    """List all teams (for demo - in production, filter by user)"""
    async with get_db_session() as session:
        if session is None:
            return {"teams": []}

        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(TeamModel)
            .options(selectinload(TeamModel.memberships))
            .order_by(TeamModel.created_at.desc())
        )
        teams = result.scalars().all()
        return {"teams": [t.to_dict() for t in teams]}


@app.get("/api/teams/{team_id}")
async def get_team(team_id: str):
    """Get team details"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(TeamModel)
            .options(selectinload(TeamModel.memberships).selectinload(TeamMembershipModel.user))
            .where(TeamModel.id == team_id)
        )
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        team_dict = team.to_dict()
        team_dict["members"] = [m.to_dict() for m in team.memberships]
        return team_dict


@app.put("/api/teams/{team_id}")
async def update_team(
    team_id: str,
    name: str = Form(None),
    description: str = Form(None),
    settings: str = Form(None),  # JSON string
):
    """Update team settings (admin only)"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(TeamModel).where(TeamModel.id == team_id)
        )
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        if name:
            team.name = name
        if description:
            team.description = description
        if settings:
            import json
            team.settings = json.loads(settings)

        await session.flush()
        return {"success": True, "team": team.to_dict()}


@app.delete("/api/teams/{team_id}")
async def delete_team(team_id: str):
    """Delete team (admin only)"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(TeamModel).where(TeamModel.id == team_id)
        )
        team = result.scalar_one_or_none()

        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        await session.delete(team)
        return {"success": True, "message": f"Team {team_id} deleted"}


@app.post("/api/teams/{team_id}/members")
async def add_team_member(
    team_id: str,
    email: str = Form(...),
    role: str = Form("viewer"),  # admin, contributor, viewer
):
    """Add a member to the team"""
    if role not in ["admin", "contributor", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        # Find user by email
        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check if already a member
        result = await session.execute(
            select(TeamMembershipModel).where(
                TeamMembershipModel.team_id == team_id,
                TeamMembershipModel.user_id == user.id
            )
        )
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="User already a member")

        membership = TeamMembershipModel(
            id=generate_id(),
            team_id=team_id,
            user_id=user.id,
            role=role,
        )
        session.add(membership)
        await session.flush()

        return {
            "success": True,
            "membership": membership.to_dict(),
        }


@app.put("/api/teams/{team_id}/members/{user_id}")
async def update_member_role(
    team_id: str,
    user_id: str,
    role: str = Form(...),
):
    """Update member's role (admin only)"""
    if role not in ["admin", "contributor", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(TeamMembershipModel).where(
                TeamMembershipModel.team_id == team_id,
                TeamMembershipModel.user_id == user_id
            )
        )
        membership = result.scalar_one_or_none()

        if not membership:
            raise HTTPException(status_code=404, detail="Membership not found")

        membership.role = role
        await session.flush()

        return {"success": True, "membership": membership.to_dict()}


@app.delete("/api/teams/{team_id}/members/{user_id}")
async def remove_team_member(team_id: str, user_id: str):
    """Remove member from team (admin only)"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(TeamMembershipModel).where(
                TeamMembershipModel.team_id == team_id,
                TeamMembershipModel.user_id == user_id
            )
        )
        membership = result.scalar_one_or_none()

        if not membership:
            raise HTTPException(status_code=404, detail="Membership not found")

        await session.delete(membership)
        return {"success": True, "message": "Member removed"}


@app.get("/api/teams/{team_id}/activity")
async def get_team_activity(
    team_id: str,
    limit: int = 50,
    offset: int = 0,
):
    """Get team activity log"""
    async with get_db_session() as session:
        if session is None:
            return {"activities": []}

        from sqlalchemy import select

        result = await session.execute(
            select(ActivityLogModel)
            .where(ActivityLogModel.team_id == team_id)
            .order_by(ActivityLogModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        activities = result.scalars().all()
        return {"activities": [a.to_dict() for a in activities]}


async def log_activity(
    team_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str = None,
    details: Dict = None,
):
    """Log an activity for audit trail"""
    try:
        async with get_db_session() as session:
            if session is None:
                return

            activity = ActivityLogModel(
                id=generate_id(),
                team_id=team_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details or {},
            )
            session.add(activity)
            await session.flush()
    except Exception as e:
        print(f"[Activity Log] Error: {e}")


# ============== API Key Management ==============

def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key. Returns (full_key, key_hash, key_prefix)"""
    # Generate a secure random key with prefix
    key_bytes = secrets.token_bytes(32)
    full_key = f"pk_{secrets.token_urlsafe(32)}"
    key_prefix = full_key[:10]
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, key_hash, key_prefix


def verify_api_key(key: str, key_hash: str) -> bool:
    """Verify an API key against its hash"""
    return hashlib.sha256(key.encode()).hexdigest() == key_hash


@app.post("/api/teams/{team_id}/api-keys")
async def create_api_key(
    team_id: str,
    name: str = Form(...),
    permissions: str = Form("read"),  # Comma-separated: "read,write"
    expires_days: int = Form(None),  # Optional expiry in days
    user_id: str = Form(None),  # Creator's user ID
):
    """Create a new API key for a team"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select
        from datetime import timedelta

        # Verify team exists
        result = await session.execute(
            select(TeamModel).where(TeamModel.id == team_id)
        )
        team = result.scalar_one_or_none()
        if not team:
            raise HTTPException(404, "Team not found")

        # Generate the API key
        full_key, key_hash, key_prefix = generate_api_key()

        # Parse permissions
        perm_list = [p.strip() for p in permissions.split(",") if p.strip()]
        valid_perms = ["read", "write", "admin"]
        perm_list = [p for p in perm_list if p in valid_perms]
        if not perm_list:
            perm_list = ["read"]

        # Calculate expiry
        expires_at = None
        if expires_days and expires_days > 0:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        # Create the API key record
        api_key = APIKeyModel(
            id=generate_id(),
            team_id=team_id,
            user_id=user_id or "system",
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permissions=perm_list,
            expires_at=expires_at,
        )
        session.add(api_key)
        await session.flush()

        # Log activity
        await log_activity(
            team_id=team_id,
            user_id=user_id or "system",
            action="create",
            resource_type="api_key",
            resource_id=api_key.id,
            details={"name": name, "permissions": perm_list}
        )

        # Return the full key ONLY on creation (never stored/returned again)
        return {
            "id": api_key.id,
            "name": api_key.name,
            "key": full_key,  # Only shown once!
            "key_prefix": key_prefix,
            "permissions": perm_list,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
            "message": "Store this key securely - it will not be shown again!"
        }


@app.get("/api/teams/{team_id}/api-keys")
async def list_api_keys(team_id: str):
    """List all API keys for a team (without the actual key values)"""
    async with get_db_session() as session:
        if session is None:
            return {"api_keys": []}

        from sqlalchemy import select

        result = await session.execute(
            select(APIKeyModel)
            .where(APIKeyModel.team_id == team_id)
            .order_by(APIKeyModel.created_at.desc())
        )
        api_keys = result.scalars().all()

        return {
            "api_keys": [
                {
                    "id": key.id,
                    "name": key.name,
                    "key_prefix": key.key_prefix,
                    "permissions": key.permissions or ["read"],
                    "last_used": key.last_used.isoformat() if key.last_used else None,
                    "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                    "created_at": key.created_at.isoformat() if key.created_at else None,
                    "is_expired": key.expires_at and key.expires_at < datetime.utcnow(),
                }
                for key in api_keys
            ]
        }


@app.delete("/api/teams/{team_id}/api-keys/{key_id}")
async def revoke_api_key(
    team_id: str,
    key_id: str,
    user_id: str = None,
):
    """Revoke (delete) an API key"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select, delete

        # Find the key
        result = await session.execute(
            select(APIKeyModel)
            .where(APIKeyModel.id == key_id)
            .where(APIKeyModel.team_id == team_id)
        )
        api_key = result.scalar_one_or_none()
        if not api_key:
            raise HTTPException(404, "API key not found")

        key_name = api_key.name

        # Delete the key
        await session.execute(
            delete(APIKeyModel).where(APIKeyModel.id == key_id)
        )
        await session.flush()

        # Log activity
        await log_activity(
            team_id=team_id,
            user_id=user_id or "system",
            action="delete",
            resource_type="api_key",
            resource_id=key_id,
            details={"name": key_name}
        )

        return {"success": True, "message": f"API key '{key_name}' revoked"}


@app.post("/api/auth/verify-key")
async def verify_api_key_endpoint(
    api_key: str = Form(...),
):
    """Verify an API key and return its permissions"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        # Extract prefix to find the key
        key_prefix = api_key[:10] if len(api_key) >= 10 else api_key

        result = await session.execute(
            select(APIKeyModel)
            .where(APIKeyModel.key_prefix == key_prefix)
        )
        stored_key = result.scalar_one_or_none()

        if not stored_key:
            raise HTTPException(401, "Invalid API key")

        # Verify the hash
        if not verify_api_key(api_key, stored_key.key_hash):
            raise HTTPException(401, "Invalid API key")

        # Check expiry
        if stored_key.expires_at and stored_key.expires_at < datetime.utcnow():
            raise HTTPException(401, "API key has expired")

        # Update last used timestamp
        stored_key.last_used = datetime.utcnow()
        await session.flush()

        return {
            "valid": True,
            "team_id": stored_key.team_id,
            "permissions": stored_key.permissions or ["read"],
            "name": stored_key.name,
        }


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
