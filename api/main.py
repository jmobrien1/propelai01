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
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Phase 5: Load environment variables and MongoDB layer
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Ensure MONGO_URL is set
if not os.getenv('MONGO_URL'):
    os.environ['MONGO_URL'] = 'mongodb://localhost:27017/propelai'

from api.db import db

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

# v2.12: RFP Chat Agent
try:
    from agents.chat import RFPChatAgent, ChatMessage, DocumentChunk
    RFP_CHAT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RFP Chat agent not available: {e}")
    RFP_CHAT_AVAILABLE = False
    RFPChatAgent = None
    ChatMessage = None
    DocumentChunk = None


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


class ChatRequest(BaseModel):
    """Request to chat with RFP"""
    message: str
    include_sources: bool = True


class ChatResponse(BaseModel):
    """Response from RFP chat"""
    answer: str
    sources: List[Dict[str, Any]] = []
    timestamp: str


# ============== Phase 5: MongoDB Store (Replaces In-Memory) ==============

class RFPStore:
    """
    MongoDB-backed store for RFP data
    
    Phase 5: Replaced in-memory dict with persistent MongoDB storage.
    All operations are now async and use Motor for non-blocking I/O.
    """
    
    async def create(self, rfp_id: str, data: Dict) -> Dict:
        """Create a new RFP entry in MongoDB"""
        rfp_doc = {
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
            "document_chunks": [],
            "chat_history": [],
            "rfp_type": "unknown",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        await db.rfps.insert_one(rfp_doc)
        return db.serialize_doc(rfp_doc)
    
    async def get(self, rfp_id: str) -> Optional[Dict]:
        """Get RFP by ID from MongoDB"""
        doc = await db.rfps.find_one({"id": rfp_id}, {"_id": 0})
        return doc
    
    async def update(self, rfp_id: str, updates: Dict) -> Dict:
        """Update RFP in MongoDB"""
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        result = await db.rfps.update_one(
            {"id": rfp_id},
            {"$set": updates}
        )
        
        if result.matched_count == 0:
            raise KeyError(f"RFP not found: {rfp_id}")
        
        return await self.get(rfp_id)
    
    async def list_all(self) -> List[Dict]:
        """List all RFPs from MongoDB"""
        cursor = db.rfps.find({}, {"_id": 0})
        docs = await cursor.to_list(length=1000)
        return docs
    
    async def delete(self, rfp_id: str) -> bool:
        """Delete RFP from MongoDB"""
        result = await db.rfps.delete_one({"id": rfp_id})
        return result.deleted_count > 0
    
    async def set_status(self, rfp_id: str, status: str, progress: int, message: str, req_count: int = None):
        """Set processing status in RFP document"""
        status_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "requirements_count": req_count
        }
        await db.rfps.update_one(
            {"id": rfp_id},
            {"$set": {"processing_status": status_data}}
        )
    
    async def get_status(self, rfp_id: str) -> Optional[ProcessingStatus]:
        """Get processing status from RFP document"""
        doc = await db.rfps.find_one({"id": rfp_id}, {"processing_status": 1, "_id": 0})
        if doc and "processing_status" in doc:
            status_data = doc["processing_status"]
            return ProcessingStatus(**status_data)
        return None


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

# v3.1: Import and Initialize Company Library FIRST (needed for Chat Agent RAG)
company_library = None
try:
    from agents.enhanced_compliance.company_library import (
        CompanyLibrary,
        CompanyLibraryParser,
        DocumentType,
    )
    company_library = CompanyLibrary(str(OUTPUT_DIR / "company_library"))
    print("[STARTUP] Company Library initialized")
except ImportError as e:
    print(f"[STARTUP] Warning: Company Library not available: {e}")
    company_library = None
except Exception as e:
    print(f"[STARTUP] Warning: Could not initialize Company Library: {e}")
    company_library = None

# Initialize RFP Chat Agent (v2.12+ / v3.1 with Library integration)
rfp_chat_agent = None
try:
    if RFP_CHAT_AVAILABLE:
        # v3.1: Pass company_library for RAG integration
        rfp_chat_agent = RFPChatAgent(company_library=company_library)
        print("[STARTUP] RFP Chat Agent initialized with Company Library integration")
except Exception as e:
    print(f"[STARTUP] Warning: Could not initialize RFP Chat Agent: {e}")
    rfp_chat_agent = None


# ============== FastAPI App ==============

app = FastAPI(
    title="PropelAI API",
    description="RFP Intelligence Platform - Extract requirements, track amendments, generate compliance matrices",
    version="4.0.0"  # Phase 5: Production Hardening with MongoDB
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Phase 5: Startup & Shutdown Events ==============

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup"""
    try:
        await db.connect()
        print("[STARTUP] PropelAI v4.0 - MongoDB connected successfully")
    except Exception as e:
        print(f"[STARTUP] ERROR: Failed to connect to MongoDB: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown"""
    await db.close()
    print("[SHUTDOWN] MongoDB connection closed")


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
        "version": "2.9.0",
        "components": {
            "enhanced_compliance_agent": "ready",
            "amendment_processor": "ready",
            "excel_export": "ready",
            "semantic_extractor": "ready" if semantic_extractor else "not available",
            "semantic_ctm_export": "ready" if semantic_ctm_exporter else "not available",
            "best_practices_extractor": "ready" if best_practices_extractor else "not available",
            "best_practices_ctm_export": "ready" if best_practices_exporter else "not available",
        }
    }


# ============== RFP Management ==============

@app.post("/api/rfp", response_model=RFPResponse)
async def create_rfp(rfp: RFPCreate):
    """Create a new RFP project"""
    rfp_id = f"RFP-{uuid.uuid4().hex[:8].upper()}"
    
    data = await store.create(rfp_id, rfp.dict())
    
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
    rfps = await store.list_all()
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
    rfp = await store.get(rfp_id)
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
    deleted = await store.delete(rfp_id)
    if not deleted:
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
    rfp = await store.get(rfp_id)
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
    await store.update(rfp_id, {
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

async def process_rfp_background(rfp_id: str):
    """Background task to process RFP - uses best practices extractor when available"""
    rfp = await store.get(rfp_id)
    if not rfp:
        return
    
    # Use best practices extraction if available (v2.10+)
    if BEST_PRACTICES_AVAILABLE and best_practices_extractor:
        await process_rfp_best_practices_background(rfp_id)
        return
    
    try:
        # Update status
        await store.set_status(rfp_id, "processing", 10, "Parsing documents...")
        await store.update(rfp_id, {"status": "processing"})
        
        # Get file paths
        file_paths = rfp["file_paths"]
        if not file_paths:
            await store.set_status(rfp_id, "error", 0, "No files to process")
            await store.update(rfp_id, {"status": "error"})
            return
        
        # Process with Enhanced Compliance Agent
        await store.set_status(rfp_id, "processing", 30, "Extracting requirements...")
        
        result = agent.process_files(file_paths)
        
        await store.set_status(rfp_id, "processing", 70, "Classifying and prioritizing...")
        
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
        
        await store.set_status(rfp_id, "processing", 90, "Finalizing...")
        
        # Initialize amendment processor
        amendment_processor = AmendmentProcessor()
        amendment_processor.load_base_requirements(result.requirements_graph)
        
        # Update store
        await store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "requirements_graph": result.requirements_graph,
            "stats": stats,
            "amendment_processor": amendment_processor
        })
        
        await store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))
        
    except Exception as e:
        await store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        await store.update(rfp_id, {"status": "error"})


@app.post("/api/rfp/{rfp_id}/process")
async def process_rfp(rfp_id: str, background_tasks: BackgroundTasks):
    """Start processing RFP documents"""
    rfp = await store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    if not rfp["file_paths"]:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Start background processing
    await store.set_status(rfp_id, "starting", 0, "Starting processing...")
    background_tasks.add_task(process_rfp_background, rfp_id)
    
    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"])
    }


# ============== v2.8: Semantic Processing ==============

async def process_rfp_semantic_background(rfp_id: str):
    """Background task to process RFP with semantic extraction (v2.8)"""
    rfp = await store.get(rfp_id)
    if not rfp:
        return
    
    if not semantic_extractor:
        await store.set_status(rfp_id, "error", 0, "Semantic extractor not available")
        await store.update(rfp_id, {"status": "error"})
        return
    
    try:
        import time
        start_time = time.time()
        
        # Update status
        await store.set_status(rfp_id, "processing", 10, "Reading documents...")
        await store.update(rfp_id, {"status": "processing"})
        
        # Get file paths
        file_paths = rfp["file_paths"]
        if not file_paths:
            await store.set_status(rfp_id, "error", 0, "No files to process")
            await store.update(rfp_id, {"status": "error"})
            return
        
        # Parse documents into text
        await store.set_status(rfp_id, "processing", 20, "Parsing documents...")
        
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
            await store.set_status(rfp_id, "error", 0, "No documents could be parsed")
            await store.update(rfp_id, {"status": "error"})
            return
        
        # Run semantic extraction
        await store.set_status(rfp_id, "processing", 40, "Extracting requirements semantically...")
        
        result = semantic_extractor.extract(documents, strict_mode=True)
        
        await store.set_status(rfp_id, "processing", 70, "Classifying and scoring...")
        
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
        
        await store.set_status(rfp_id, "processing", 90, "Finalizing...")
        
        # Store semantic results
        await store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "semantic_result": result,  # Keep full semantic result for export
            "stats": stats,
            "evaluation_factors": result.evaluation_factors,
            "warnings": result.warnings,
            "extraction_mode": "semantic"
        })
        
        await store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        await store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        await store.update(rfp_id, {"status": "error"})


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
    rfp = await store.get(rfp_id)
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
    await store.set_status(rfp_id, "starting", 0, "Starting semantic processing...")
    background_tasks.add_task(process_rfp_semantic_background, rfp_id)
    
    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"]),
        "mode": "semantic"
    }


# ============== v2.9: Best Practices CTM Processing ==============

async def process_rfp_best_practices_background(rfp_id: str):
    """
    Background task to process RFP with Best Practices CTM extraction (v2.9).
    
    Per best practices:
    - Analyzes document structure BEFORE extraction
    - Preserves RFP's own numbering (L.4.B.2, C.3.1)
    - Creates separate L/M/C matrices
    - Extracts complete paragraphs, not sentence fragments
    """
    rfp = await store.get(rfp_id)
    if not rfp:
        return
    
    if not best_practices_extractor:
        await store.set_status(rfp_id, "error", 0, "Best practices extractor not available")
        await store.update(rfp_id, {"status": "error"})
        return
    
    try:
        import time
        start_time = time.time()
        
        # Update status
        await store.set_status(rfp_id, "processing", 10, "Analyzing document structure...")
        await store.update(rfp_id, {"status": "processing"})
        
        # Get file paths
        file_paths = rfp["file_paths"]
        if not file_paths:
            await store.set_status(rfp_id, "error", 0, "No files to process")
            await store.update(rfp_id, {"status": "error"})
            return
        
        # Parse documents into text
        await store.set_status(rfp_id, "processing", 20, "Parsing documents...")
        
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
            await store.set_status(rfp_id, "error", 0, "No documents could be parsed")
            await store.update(rfp_id, {"status": "error"})
            return
        
        # Analyze document structure
        await store.set_status(rfp_id, "processing", 40, "Analyzing RFP structure (L/M/C sections)...")
        structure = analyze_rfp_structure(documents)
        
        # Extract requirements with structure awareness
        await store.set_status(rfp_id, "processing", 60, "Extracting requirements by section...")
        result = best_practices_extractor.extract(documents, structure)
        
        await store.set_status(rfp_id, "processing", 80, "Building compliance matrices...")
        
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
        
        await store.set_status(rfp_id, "processing", 95, "Finalizing...")
        
        # Store results
        await store.update(rfp_id, {
            "status": "completed",
            "requirements": requirements,
            "best_practices_result": result,  # Keep full result for export
            "stats": stats,
            "extraction_mode": "best_practices"
        })
        
        await store.set_status(rfp_id, "completed", 100, "Processing complete", len(requirements))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        await store.set_status(rfp_id, "error", 0, f"Error: {str(e)}")
        await store.update(rfp_id, {"status": "error"})


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
    rfp = await store.get(rfp_id)
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
    await store.set_status(rfp_id, "starting", 0, "Starting best practices extraction...")
    background_tasks.add_task(process_rfp_best_practices_background, rfp_id)
    
    return {
        "status": "processing_started",
        "rfp_id": rfp_id,
        "files_count": len(rfp["file_paths"]),
        "mode": "best_practices"
    }


@app.get("/api/rfp/{rfp_id}/status")
async def get_processing_status(rfp_id: str):
    """Get processing status"""
    rfp = await store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    status = await store.get_status(rfp_id)
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
    rfp = await store.get(rfp_id)
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
    rfp = await store.get(rfp_id)
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
    rfp = await store.get(rfp_id)
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
    rfp = await store.get(rfp_id)
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
        
        await store.update(rfp_id, {
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
    rfp = await store.get(rfp_id)
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
    rfp = await store.get(rfp_id)
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
    rfp = await store.get(rfp_id)
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
    
    rfp = await store.get(rfp_id)
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
        await store.update(rfp_id, {"outline": outline_data})
        
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
    
    rfp = await store.get(rfp_id)
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
        await store.update(rfp_id, {"outline": outline})
    
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
    
    rfp = await store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")
    
    outline = rfp.get("outline")
    if not outline:
        generator = SmartOutlineGenerator()
        section_l = [r for r in rfp.get("requirements", []) if r.get("section", "").upper().startswith("L")]
        section_m = [r for r in rfp.get("requirements", []) if r.get("section", "").upper().startswith("M")]
        outline_obj = generator.generate_from_compliance_matrix(section_l, section_m, rfp.get("documents", []))
        outline = generator.to_json(outline_obj)
        await store.update(rfp_id, {"outline": outline})
    
    requirements = rfp.get("requirements", [])
    
    config = AnnotatedOutlineConfig(
        rfp_title=rfp.get("name", rfp.get("title", "RFP")),
        solicitation_number=rfp.get("solicitation_number", rfp_id),
        due_date=outline.get("submission", {}).get("due_date", "TBD"),
        submission_method=outline.get("submission", {}).get("method", "Not Specified"),
        total_pages=outline.get("total_pages"),
        company_name="[Your Company Name]"
    )
    
    try:
        exporter = AnnotatedOutlineExporter()
        doc_bytes = exporter.export(outline, requirements, outline.get("format_requirements", {}), config)
        
        safe_name = "".join(c for c in rfp.get("name", rfp_id) if c.isalnum() or c in " -_")[:50]
        filename = f"{safe_name}_Annotated_Outline.docx"
        
        return Response(
            content=doc_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate annotated outline: {str(e)}")



# ============== Company Library API ==============

# Company library already imported and initialized earlier (before chat agent)
COMPANY_LIBRARY_AVAILABLE = company_library is not None


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
    document_type: Optional[str] = Form(None),
    tag: Optional[str] = Form(None)
):
    """
    Upload a document to the company library
    
    v4.0: Added duplicate detection and tagging support.
    
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
        
        # Add to library (v4.0 returns dict with status)
        result = company_library.add_document(str(temp_path), doc_type, tag)
        
        if result['status'] == 'success':
            return {
                "success": True,
                "document": result['document'].to_dict(),
                "message": result['message'],
                "filename": result['filename']
            }
        elif result['status'] == 'duplicate':
            return {
                "success": False,
                "duplicate": True,
                "message": result['message'],
                "existing_file": result['existing_file']
            }
        else:  # error
            raise HTTPException(status_code=400, detail=result['message'])
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
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


# ============== RFP Chat Endpoints (v2.12) ==============

@app.post("/api/rfp/{rfp_id}/chat")
async def chat_with_rfp(rfp_id: str, request: ChatRequest):
    """
    Chat with RFP documents - ask questions and get answers based on uploaded content.
    
    Similar to NotebookLM - uses RAG (Retrieval Augmented Generation) to answer
    questions based solely on the uploaded RFP documents.
    """
    if not RFP_CHAT_AVAILABLE or not rfp_chat_agent:
        raise HTTPException(
            status_code=503, 
            detail="Chat functionality not available. Please ensure anthropic package is installed."
        )
    
    rfp = await store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail=f"RFP {rfp_id} not found")
    
    # Ensure document chunks exist
    if not rfp.get("document_chunks"):
        # Create chunks on first chat
        try:
            chunks = rfp_chat_agent.chunk_rfp_documents(rfp)
            rfp["document_chunks"] = [chunk.to_dict() for chunk in chunks]
            # v3.0: Store detected RFP type
            rfp["rfp_type"] = rfp_chat_agent.detected_rfp_type.value
            print(f"[CHAT] Created {len(chunks)} document chunks for RFP {rfp_id}")
            print(f"[CHAT] Detected RFP Type: {rfp['rfp_type']}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error creating document chunks: {str(e)}"
            )
    
    # Convert stored chunks back to DocumentChunk objects
    document_chunks = [
        DocumentChunk(**chunk_dict) 
        for chunk_dict in rfp.get("document_chunks", [])
    ]
    
    # Get chat history
    chat_history = [
        ChatMessage(**msg_dict)
        for msg_dict in rfp.get("chat_history", [])
    ]
    
    # Generate response
    try:
        # Add user message to history
        user_message = ChatMessage(
            role="user",
            content=request.message,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        chat_history.append(user_message)
        
        # Get AI response
        assistant_message = rfp_chat_agent.chat(
            question=request.message,
            document_chunks=document_chunks,
            chat_history=chat_history[:-1]  # Don't include the current user message
        )
        
        # Add to history
        chat_history.append(assistant_message)
        
        # Save history (keep last 50 messages)
        rfp["chat_history"] = [msg.to_dict() for msg in chat_history[-50:]]
        rfp["updated_at"] = datetime.now().isoformat()
        
        # Return response
        return ChatResponse(
            answer=assistant_message.content,
            sources=assistant_message.sources if request.include_sources else [],
            timestamp=assistant_message.timestamp
        )
        
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@app.get("/api/rfp/{rfp_id}/chat/history")
async def get_chat_history(rfp_id: str, limit: int = 50):
    """Get chat history for an RFP"""
    if not RFP_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chat functionality not available")
    
    rfp = await store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail=f"RFP {rfp_id} not found")
    
    history = rfp.get("chat_history", [])
    
    # Return last N messages
    return {
        "rfp_id": rfp_id,
        "history": history[-limit:],
        "total_messages": len(history)
    }


@app.delete("/api/rfp/{rfp_id}/chat/history")
async def clear_chat_history(rfp_id: str):
    """Clear chat history for an RFP"""
    if not RFP_CHAT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chat functionality not available")
    
    rfp = await store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail=f"RFP {rfp_id} not found")
    
    rfp["chat_history"] = []
    rfp["updated_at"] = datetime.now().isoformat()
    
    return {"message": "Chat history cleared", "rfp_id": rfp_id}


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
