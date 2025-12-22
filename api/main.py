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
import time
import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Response, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Password hashing with bcrypt (secure)
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    pwd_context = None


# ============== Structured Logging ==============

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging in production"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "status_code"):
            log_data["status_code"] = record.status_code
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        return json.dumps(log_data)


def setup_logging():
    """Configure logging based on environment"""
    log_format = os.environ.get("LOG_FORMAT", "text")  # "json" or "text"
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Get root logger
    logger = logging.getLogger("propelai")
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Remove existing handlers
    logger.handlers = []

    # Create console handler
    handler = logging.StreamHandler()

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

    logger.addHandler(handler)
    return logger


# Initialize logger
logger = setup_logging()


# ============== Input Sanitization ==============

def sanitize_html(text: str) -> str:
    """
    Remove potentially dangerous HTML/script content from user input.
    Used to prevent XSS attacks when displaying user-generated content.
    """
    if not text:
        return text

    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove event handlers (onclick, onerror, etc.)
    text = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+on\w+\s*=\s*\S+', '', text, flags=re.IGNORECASE)

    # Remove javascript: URLs
    text = re.sub(r'javascript\s*:', '', text, flags=re.IGNORECASE)

    # Remove data: URLs that could contain scripts
    text = re.sub(r'data\s*:\s*text/html', '', text, flags=re.IGNORECASE)

    # Remove style tags (can be used for CSS injection)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove iframe, embed, object tags
    text = re.sub(r'<(iframe|embed|object)[^>]*>.*?</\1>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<(iframe|embed|object)[^>]*/>', '', text, flags=re.IGNORECASE)

    return text.strip()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other attacks.
    """
    if not filename:
        return "unnamed_file"

    # Remove path components
    filename = os.path.basename(filename)

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    # Ensure it's not empty
    if not filename or filename in ('.', '..'):
        filename = "unnamed_file"

    return filename


# ============== File Upload Security ==============

# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed file extensions and their MIME types
ALLOWED_EXTENSIONS = {
    ".pdf": ["application/pdf"],
    ".docx": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    ".doc": ["application/msword"],
    ".xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
    ".xls": ["application/vnd.ms-excel"],
}

# Magic bytes signatures for file type validation
FILE_SIGNATURES = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],  # ZIP-based format
    ".doc": [b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"],  # OLE Compound Document
    ".xlsx": [b"PK\x03\x04"],  # ZIP-based format
    ".xls": [b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"],  # OLE Compound Document
}


class FileValidationError(Exception):
    """Exception raised when file validation fails"""
    pass


async def validate_uploaded_file(file: UploadFile) -> tuple[bytes, str]:
    """
    Validate an uploaded file for security.

    Returns:
        Tuple of (file_content, sanitized_filename)

    Raises:
        FileValidationError: If validation fails
    """
    # Sanitize filename
    original_filename = file.filename or "unnamed_file"
    safe_filename = sanitize_filename(original_filename)

    # Get extension
    ext = Path(safe_filename).suffix.lower()

    # Check allowed extension
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"File type '{ext}' is not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS.keys())}"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise FileValidationError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed size ({max_mb:.0f} MB)"
        )

    # Check for empty files
    if len(content) == 0:
        raise FileValidationError("Empty files are not allowed")

    # Validate magic bytes
    valid_signature = False
    signatures = FILE_SIGNATURES.get(ext, [])
    for sig in signatures:
        if content.startswith(sig):
            valid_signature = True
            break

    if signatures and not valid_signature:
        raise FileValidationError(
            f"File content does not match expected format for '{ext}' files"
        )

    # Check MIME type if provided by client (optional check)
    if file.content_type:
        allowed_mimes = ALLOWED_EXTENSIONS.get(ext, [])
        # Be lenient with MIME types as browsers can be inconsistent
        # Just log a warning if mismatch
        if allowed_mimes and file.content_type not in allowed_mimes:
            logger.warning(
                f"MIME type mismatch: got {file.content_type}, expected one of {allowed_mimes}",
                extra={"filename": safe_filename}
            )

    # Additional security: scan for embedded scripts in documents
    # Check for common malicious patterns in file content
    malicious_patterns = [
        b"<script",
        b"javascript:",
        b"vbscript:",
        b"powershell",
        b"/bin/bash",
        b"/bin/sh",
        b"cmd.exe",
    ]

    # Only check text-readable portions (first and last 10KB) to avoid false positives
    # in binary content
    check_content = content[:10240] + content[-10240:] if len(content) > 20480 else content
    check_lower = check_content.lower()

    for pattern in malicious_patterns:
        if pattern.lower() in check_lower:
            logger.warning(
                f"Suspicious pattern detected in file: {safe_filename}",
                extra={"pattern": pattern.decode('utf-8', errors='ignore')}
            )
            # Don't block, just log - could be legitimate content
            break

    # Reset file position for any subsequent reads
    await file.seek(0)

    return content, safe_filename


def validate_file_extension(filename: str) -> bool:
    """Quick check if file extension is allowed"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def get_max_file_size_mb() -> float:
    """Get maximum file size in MB"""
    return MAX_FILE_SIZE / (1024 * 1024)


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
    from api.database import init_db, is_db_available, db_store, DatabaseStore, get_db_session
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    is_db_available = lambda: False
    db_store = None
    get_db_session = None
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


# ============== Pagination ==============

class PaginationParams(BaseModel):
    """Standard pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")

    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Alias for page_size"""
        return self.page_size


class PaginatedResponse(BaseModel):
    """Standard paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def paginate(
    items: List[Any],
    page: int = 1,
    page_size: int = 20,
    total: Optional[int] = None
) -> PaginatedResponse:
    """
    Create a paginated response from a list of items.

    Args:
        items: List of items (already sliced for current page)
        page: Current page number (1-indexed)
        page_size: Items per page
        total: Total number of items (if None, uses len(items))

    Returns:
        PaginatedResponse with metadata
    """
    if total is None:
        total = len(items)

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )


def get_pagination_params(
    page: int = 1,
    page_size: int = 20
) -> tuple[int, int]:
    """
    Validate and return pagination parameters.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page

    Returns:
        Tuple of (offset, limit)
    """
    page = max(1, page)
    page_size = max(1, min(100, page_size))  # Clamp between 1 and 100
    offset = (page - 1) * page_size
    return offset, page_size


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

    def get(self, rfp_id: str, include_deleted: bool = False) -> Optional[Dict]:
        """
        Get RFP by ID.

        Args:
            rfp_id: The RFP identifier
            include_deleted: If True, also return soft-deleted RFPs
        """
        rfp = self.rfps.get(rfp_id)
        if rfp is None:
            return None

        # Check if deleted
        if not include_deleted and rfp.get("is_deleted"):
            return None

        return rfp

    def update(self, rfp_id: str, updates: Dict) -> Dict:
        """Update RFP"""
        if rfp_id not in self.rfps:
            raise KeyError(f"RFP not found: {rfp_id}")

        self.rfps[rfp_id].update(updates)
        self.rfps[rfp_id]["updated_at"] = datetime.now().isoformat()
        self._schedule_db_sync(rfp_id)
        return self.rfps[rfp_id]

    def list_all(self, include_deleted: bool = False) -> List[Dict]:
        """
        List all RFPs.

        Args:
            include_deleted: If True, include soft-deleted RFPs
        """
        if include_deleted:
            return list(self.rfps.values())
        else:
            return [r for r in self.rfps.values() if not r.get("is_deleted")]

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

# OpenAPI Tags for documentation organization
OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Health checks and monitoring endpoints",
    },
    {
        "name": "Authentication",
        "description": "User authentication, registration, and session management",
    },
    {
        "name": "Users",
        "description": "User profile management",
    },
    {
        "name": "Teams",
        "description": "Team workspaces and member management",
    },
    {
        "name": "RFP",
        "description": "RFP document upload, processing, and management",
    },
    {
        "name": "Requirements",
        "description": "Requirements extraction and tracking",
    },
    {
        "name": "Strategy",
        "description": "Win themes, competitive analysis, and Iron Triangle strategy",
    },
    {
        "name": "Drafting",
        "description": "AI-powered proposal drafting workflow",
    },
    {
        "name": "Library",
        "description": "Company Library with vector search capabilities",
    },
    {
        "name": "Webhooks",
        "description": "Webhook management for event notifications",
    },
    {
        "name": "Admin",
        "description": "Administrative operations and bulk actions",
    },
]

app = FastAPI(
    title="PropelAI API",
    description="""
# PropelAI - Autonomous Proposal Operating System

PropelAI is an AI-powered RFP intelligence platform that helps organizations win more proposals through:

- **Trust Gate**: Source traceability with visual overlays proving extraction accuracy
- **Iron Triangle**: Strategic analysis linking Section L, M, and C requirements
- **Drafting Agent**: AI-powered proposal generation using Feature-Benefit-Proof framework
- **Vector Search**: Semantic search across your Company Library

## Authentication

Most endpoints require authentication via JWT Bearer token. Obtain a token by:
1. Register: `POST /api/auth/register`
2. Login: `POST /api/auth/login`
3. Include token in header: `Authorization: Bearer <token>`

## Rate Limiting

Authentication endpoints are rate-limited to prevent abuse:
- Login: 5 attempts per minute
- Register: 3 attempts per minute
- Forgot Password: 3 attempts per 5 minutes
- General API: 100 requests per minute

## Versioning

API version is returned in the `X-API-Version` response header.
Current version: **4.1.0**
    """,
    version="4.1.0",
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "PropelAI Support",
        "url": "https://github.com/propelai/propelai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS Configuration - Use environment variable in production
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
if CORS_ORIGINS == "*":
    # Development mode - allow all origins (with warning)
    _cors_origins = ["*"]
    _cors_credentials = False  # Cannot use credentials with wildcard origin
else:
    # Production mode - restrict to specified origins
    _cors_origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-API-Version"],
)


# Security Headers Middleware
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Strict Transport Security (HTTPS only)
        # Only enable in production with HTTPS
        if os.environ.get("ENABLE_HSTS", "").lower() == "true":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (relaxed for development)
        # In production, tighten this significantly
        if os.environ.get("ENABLE_CSP", "").lower() == "true":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self' https:; "
                "frame-ancestors 'none';"
            )

        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        )

        return response


app.add_middleware(SecurityHeadersMiddleware)


# Request ID Middleware for tracing
import contextvars

# Context variable to store request ID
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


# ============== API Versioning ==============

API_VERSION = "4.1.0"
API_VERSION_MAJOR = 4
API_VERSION_MINOR = 1
API_VERSION_PATCH = 0


class APIVersionMiddleware(BaseHTTPMiddleware):
    """Add API version headers to all responses"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Add API version headers
        response.headers["X-API-Version"] = API_VERSION
        response.headers["X-API-Deprecated"] = "false"

        # Check for version in request header
        requested_version = request.headers.get("X-API-Version")
        if requested_version and requested_version != API_VERSION:
            # Log version mismatch (client might need to update)
            major = int(requested_version.split(".")[0]) if "." in requested_version else 0
            if major < API_VERSION_MAJOR:
                response.headers["X-API-Deprecated"] = "true"
                response.headers["X-API-Upgrade-Message"] = f"Please upgrade to API v{API_VERSION}"

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests for tracing and logging"""

    async def dispatch(self, request, call_next):
        # Check for existing request ID in header (from load balancer/gateway)
        request_id = request.headers.get("X-Request-ID") or secrets.token_hex(8)

        # Store in context variable for use in logging
        request_id_var.set(request_id)

        # Add request ID to request state for easy access
        request.state.request_id = request_id

        # Log the incoming request
        print(f"[{request_id}] {request.method} {request.url.path}")

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


def get_request_id() -> str:
    """Get the current request ID from context"""
    return request_id_var.get()


def get_api_version() -> dict:
    """Get current API version info"""
    return {
        "version": API_VERSION,
        "major": API_VERSION_MAJOR,
        "minor": API_VERSION_MINOR,
        "patch": API_VERSION_PATCH
    }


app.add_middleware(APIVersionMiddleware)
app.add_middleware(RequestIDMiddleware)


# ============== Startup/Shutdown Events ==============

# Background task for rate limiter cleanup
_rate_limiter_cleanup_task: Optional[asyncio.Task] = None
_shutdown_event = asyncio.Event()


async def _rate_limiter_cleanup_loop():
    """Background task to periodically clean up old rate limiter entries"""
    while not _shutdown_event.is_set():
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            await rate_limiter.cleanup_old_entries(max_age_seconds=3600)
            logger.debug("Rate limiter cleanup completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Rate limiter cleanup error: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize database and background tasks on startup"""
    global _rate_limiter_cleanup_task

    logger.info("[Startup] PropelAI v4.1 starting...")
    logger.info(f"[Startup] Environment: {PROPELAI_ENV}")
    logger.info(f"[Startup] Upload directory: {UPLOAD_DIR}")
    logger.info(f"[Startup] Database available: {DATABASE_AVAILABLE and is_db_available()}")
    logger.info(f"[Startup] Bcrypt available: {BCRYPT_AVAILABLE}")

    # Initialize database and load existing RFPs
    await store.init_database()

    # Start rate limiter cleanup background task
    _rate_limiter_cleanup_task = asyncio.create_task(_rate_limiter_cleanup_loop())

    logger.info("[Startup] Ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown - cleanup resources"""
    global _rate_limiter_cleanup_task

    logger.info("[Shutdown] PropelAI shutting down...")

    # Signal shutdown to background tasks
    _shutdown_event.set()

    # Cancel rate limiter cleanup task
    if _rate_limiter_cleanup_task:
        _rate_limiter_cleanup_task.cancel()
        try:
            await _rate_limiter_cleanup_task
        except asyncio.CancelledError:
            pass

    # Close database connections
    if DATABASE_AVAILABLE:
        try:
            from api.database import _get_engine
            engine = _get_engine()
            if engine:
                await engine.dispose()
                logger.info("[Shutdown] Database connections closed")
        except Exception as e:
            logger.error(f"[Shutdown] Error closing database: {e}")

    # Close Redis connection if available
    if REDIS_AVAILABLE and redis_client:
        try:
            await redis_client.close()
            logger.info("[Shutdown] Redis connection closed")
        except Exception as e:
            logger.error(f"[Shutdown] Error closing Redis: {e}")

    logger.info("[Shutdown] Cleanup complete")


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


# ============== API Version ==============

@app.get("/api/version")
async def get_version():
    """
    Get API version information.

    Returns current API version, changelog summary, and deprecation warnings.
    """
    return {
        "version": API_VERSION,
        "major": API_VERSION_MAJOR,
        "minor": API_VERSION_MINOR,
        "patch": API_VERSION_PATCH,
        "changelog": {
            "4.1.0": [
                "Added team workspaces with RBAC",
                "Added two-factor authentication (2FA)",
                "Added session management",
                "Added rate limiting (Redis + in-memory)",
                "Added email service integration",
                "Added file upload security validation",
                "Added standard pagination for list endpoints",
                "Added security headers middleware",
                "Added GDPR data export/delete",
                "Added structured JSON logging",
            ],
            "4.0.0": [
                "Trust Gate with source traceability",
                "Iron Triangle logic engine",
                "LangGraph drafting workflow",
                "PostgreSQL + pgvector integration",
            ]
        },
        "deprecations": [],
        "links": {
            "documentation": "/docs",
            "health": "/api/health"
        }
    }


# ============== Health Check ==============

@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns detailed status of all system components including:
    - Storage configuration and database connectivity
    - AI/ML component availability (Trust Gate, Strategy Agent, Drafting)
    - External services (Redis, Email)
    """
    # v4.1: Check storage type
    storage_type = "persistent" if PERSISTENT_DATA_DIR.exists() else "temporary"

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION,
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
        },
        "services": {
            "redis": "connected" if REDIS_AVAILABLE else "not configured",
            "email": "configured" if EMAIL_SERVICE_AVAILABLE else "console only",
        }
    }


@app.get("/api/health/live", tags=["Health"])
async def liveness_probe():
    """
    Kubernetes liveness probe.

    Returns 200 if the application is running.
    Used by Kubernetes to determine if the container should be restarted.
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/api/health/ready", tags=["Health"])
async def readiness_probe():
    """
    Kubernetes readiness probe.

    Returns 200 if the application is ready to receive traffic.
    Checks critical dependencies like database connectivity and storage access.
    """
    checks = {
        "database": False,
        "storage": False,
    }
    errors = []

    # Check database connectivity
    try:
        async with get_db_session() as session:
            if session is not None:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
                checks["database"] = True
            else:
                errors.append("Database session not available")
    except Exception as e:
        errors.append(f"Database error: {str(e)}")

    # Check storage directory
    if UPLOAD_DIR.exists() and os.access(UPLOAD_DIR, os.W_OK):
        checks["storage"] = True
    else:
        errors.append("Upload directory not writable")

    # Determine overall status
    is_ready = all(checks.values())

    if is_ready:
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
        }
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "timestamp": datetime.now().isoformat(),
                "checks": checks,
                "errors": errors,
            }
        )


@app.get("/api/metrics")
async def metrics():
    """
    Basic metrics endpoint for monitoring.
    Returns counts and statistics about the application.
    """
    # Count RFPs
    rfp_count = len(store.rfps)

    # Count users and teams (if database available)
    user_count = 0
    team_count = 0
    session_count = 0

    try:
        async with get_db_session() as session:
            if session is not None:
                from sqlalchemy import select, func

                # Count users
                result = await session.execute(
                    select(func.count()).select_from(UserModel)
                )
                user_count = result.scalar() or 0

                # Count teams
                result = await session.execute(
                    select(func.count()).select_from(TeamModel)
                )
                team_count = result.scalar() or 0

                # Count active sessions
                result = await session.execute(
                    select(func.count()).select_from(UserSessionModel)
                    .where(UserSessionModel.revoked_at == None)
                    .where(UserSessionModel.expires_at > datetime.utcnow())
                )
                session_count = result.scalar() or 0
    except Exception:
        pass

    return {
        "timestamp": datetime.now().isoformat(),
        "rfps": rfp_count,
        "users": user_count,
        "teams": team_count,
        "active_sessions": session_count,
        "uptime_seconds": int((datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()),
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
async def list_rfps(
    page: int = 1,
    page_size: int = 20
):
    """
    List all RFPs with pagination.

    Query Parameters:
    - page: Page number (1-indexed, default 1)
    - page_size: Items per page (1-100, default 20)
    """
    offset, limit = get_pagination_params(page, page_size)

    # Get all RFPs and apply pagination
    all_rfps = store.list_all()
    total = len(all_rfps)

    # Sort by created_at descending (newest first)
    all_rfps.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Apply pagination
    paginated_rfps = all_rfps[offset:offset + limit]

    items = [
        {
            "id": r["id"],
            "name": r["name"],
            "solicitation_number": r["solicitation_number"],
            "status": r["status"],
            "files_count": len(r["files"]),
            "requirements_count": len(r["requirements"]),
            "created_at": r["created_at"]
        }
        for r in paginated_rfps
    ]

    return paginate(items, page, limit, total).model_dump()


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


# Default data retention period (30 days)
DEFAULT_RETENTION_DAYS = 30


@app.delete("/api/rfp/{rfp_id}")
async def delete_rfp(
    rfp_id: str,
    permanent: bool = False,
    reason: str = None,
    authorization: str = Header(None),
):
    """
    Soft delete an RFP (moves to trash).

    Query Parameters:
    - permanent: If true, permanently deletes the RFP (skips trash)
    - reason: Optional reason for deletion (for audit trail)
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Get user ID from token if available
    user_id = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            try:
                payload = verify_jwt_token(parts[1])
                user_id = payload.get("sub")
            except ValueError:
                pass

    if permanent:
        # Permanent delete - remove files and data
        if not store.delete(rfp_id):
            raise HTTPException(status_code=404, detail="RFP not found")

        # Clean up files
        rfp_dir = UPLOAD_DIR / rfp_id
        if rfp_dir.exists():
            shutil.rmtree(rfp_dir)

        logger.info(f"RFP permanently deleted: {rfp_id}", extra={"user_id": user_id})

        return {"status": "permanently_deleted", "id": rfp_id}
    else:
        # Soft delete - mark as deleted
        permanent_delete_at = datetime.now() + timedelta(days=DEFAULT_RETENTION_DAYS)

        store.update(rfp_id, {
            "is_deleted": True,
            "deleted_at": datetime.now().isoformat(),
            "deleted_by": user_id,
            "delete_reason": reason,
            "permanent_delete_at": permanent_delete_at.isoformat(),
            "status": "deleted"
        })

        logger.info(f"RFP soft deleted: {rfp_id}", extra={"user_id": user_id})

        return {
            "status": "deleted",
            "id": rfp_id,
            "can_restore_until": permanent_delete_at.isoformat(),
            "message": f"RFP moved to trash. It will be permanently deleted after {DEFAULT_RETENTION_DAYS} days."
        }


@app.post("/api/rfp/{rfp_id}/restore")
async def restore_rfp(
    rfp_id: str,
    authorization: str = Header(None),
):
    """
    Restore a soft-deleted RFP from trash.
    """
    rfp = store.get(rfp_id, include_deleted=True)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    if not rfp.get("is_deleted"):
        raise HTTPException(status_code=400, detail="RFP is not deleted")

    # Check if past permanent delete date
    permanent_delete_at = rfp.get("permanent_delete_at")
    if permanent_delete_at:
        if isinstance(permanent_delete_at, str):
            permanent_delete_at = datetime.fromisoformat(permanent_delete_at)
        if datetime.now() > permanent_delete_at:
            raise HTTPException(
                status_code=400,
                detail="RFP cannot be restored - retention period has expired"
            )

    # Restore the RFP
    store.update(rfp_id, {
        "is_deleted": False,
        "deleted_at": None,
        "deleted_by": None,
        "delete_reason": None,
        "permanent_delete_at": None,
        "status": rfp.get("_previous_status", "created")
    })

    logger.info(f"RFP restored: {rfp_id}")

    return {
        "status": "restored",
        "id": rfp_id,
        "message": "RFP has been restored from trash"
    }


@app.get("/api/rfp/trash")
async def list_deleted_rfps(
    page: int = 1,
    page_size: int = 20,
):
    """
    List all soft-deleted RFPs (trash).

    Query Parameters:
    - page: Page number (1-indexed, default 1)
    - page_size: Items per page (1-100, default 20)
    """
    offset, limit = get_pagination_params(page, page_size)

    # Get deleted RFPs
    all_rfps = store.list_all(include_deleted=True)
    deleted_rfps = [r for r in all_rfps if r.get("is_deleted")]
    total = len(deleted_rfps)

    # Sort by deleted_at descending
    deleted_rfps.sort(key=lambda x: x.get("deleted_at", ""), reverse=True)

    # Apply pagination
    paginated = deleted_rfps[offset:offset + limit]

    items = [
        {
            "id": r["id"],
            "name": r["name"],
            "solicitation_number": r.get("solicitation_number"),
            "deleted_at": r.get("deleted_at"),
            "deleted_by": r.get("deleted_by"),
            "delete_reason": r.get("delete_reason"),
            "permanent_delete_at": r.get("permanent_delete_at"),
            "files_count": len(r.get("files", [])),
            "requirements_count": len(r.get("requirements", [])),
        }
        for r in paginated
    ]

    return paginate(items, page, limit, total).model_dump()


@app.delete("/api/rfp/trash/empty")
async def empty_trash(
    authorization: str = Header(None),
):
    """
    Permanently delete all RFPs in trash.
    """
    # Get user ID from token
    user_id = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            try:
                payload = verify_jwt_token(parts[1])
                user_id = payload.get("sub")
            except ValueError:
                pass

    # Get deleted RFPs
    all_rfps = store.list_all(include_deleted=True)
    deleted_rfps = [r for r in all_rfps if r.get("is_deleted")]

    deleted_count = 0
    for rfp in deleted_rfps:
        rfp_id = rfp["id"]

        # Delete from store
        store.delete(rfp_id)

        # Clean up files
        rfp_dir = UPLOAD_DIR / rfp_id
        if rfp_dir.exists():
            shutil.rmtree(rfp_dir)

        deleted_count += 1

    logger.info(f"Trash emptied: {deleted_count} RFPs permanently deleted", extra={"user_id": user_id})

    return {
        "status": "trash_emptied",
        "deleted_count": deleted_count,
        "message": f"{deleted_count} RFPs permanently deleted"
    }


@app.get("/api/retention-policy")
async def get_retention_policy():
    """
    Get data retention policy settings.
    """
    return {
        "default_retention_days": DEFAULT_RETENTION_DAYS,
        "description": f"Deleted RFPs are kept for {DEFAULT_RETENTION_DAYS} days before permanent deletion",
        "policies": {
            "rfp": {
                "retention_days": DEFAULT_RETENTION_DAYS,
                "soft_delete": True,
                "auto_purge": True
            },
            "webhook_deliveries": {
                "retention_days": 30,
                "description": "Webhook delivery logs are kept for 30 days"
            },
            "activity_logs": {
                "retention_days": 90,
                "description": "Activity logs are kept for 90 days"
            },
            "sessions": {
                "retention_days": 7,
                "description": "Expired sessions are purged after 7 days"
            }
        }
    }


# ============== Bulk Operations ==============

class BulkOperationRequest(BaseModel):
    """Request for bulk operations"""
    ids: List[str] = Field(..., description="List of RFP IDs to operate on")
    reason: Optional[str] = Field(None, description="Reason for the operation (for audit)")


class BulkOperationResult(BaseModel):
    """Result of a bulk operation"""
    success_count: int
    failure_count: int
    results: List[Dict[str, Any]]


@app.post("/api/rfp/bulk/delete")
async def bulk_delete_rfps(
    request: BulkOperationRequest,
    permanent: bool = False,
    authorization: str = Header(None),
):
    """
    Bulk delete multiple RFPs.

    Query Parameters:
    - permanent: If true, permanently deletes (skips trash)
    """
    # Get user ID from token
    user_id = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            try:
                payload = verify_jwt_token(parts[1])
                user_id = payload.get("sub")
            except ValueError:
                pass

    results = []
    success_count = 0
    failure_count = 0

    for rfp_id in request.ids:
        try:
            rfp = store.get(rfp_id)
            if not rfp:
                results.append({"id": rfp_id, "success": False, "error": "RFP not found"})
                failure_count += 1
                continue

            if permanent:
                store.delete(rfp_id)
                rfp_dir = UPLOAD_DIR / rfp_id
                if rfp_dir.exists():
                    shutil.rmtree(rfp_dir)
                results.append({"id": rfp_id, "success": True, "action": "permanently_deleted"})
            else:
                permanent_delete_at = datetime.now() + timedelta(days=DEFAULT_RETENTION_DAYS)
                store.update(rfp_id, {
                    "is_deleted": True,
                    "deleted_at": datetime.now().isoformat(),
                    "deleted_by": user_id,
                    "delete_reason": request.reason,
                    "permanent_delete_at": permanent_delete_at.isoformat(),
                    "status": "deleted"
                })
                results.append({"id": rfp_id, "success": True, "action": "soft_deleted"})

            success_count += 1

        except Exception as e:
            results.append({"id": rfp_id, "success": False, "error": str(e)})
            failure_count += 1

    logger.info(
        f"Bulk delete: {success_count} succeeded, {failure_count} failed",
        extra={"user_id": user_id, "permanent": permanent}
    )

    return BulkOperationResult(
        success_count=success_count,
        failure_count=failure_count,
        results=results
    )


@app.post("/api/rfp/bulk/restore")
async def bulk_restore_rfps(
    request: BulkOperationRequest,
    authorization: str = Header(None),
):
    """
    Bulk restore multiple RFPs from trash.
    """
    results = []
    success_count = 0
    failure_count = 0

    for rfp_id in request.ids:
        try:
            rfp = store.get(rfp_id, include_deleted=True)
            if not rfp:
                results.append({"id": rfp_id, "success": False, "error": "RFP not found"})
                failure_count += 1
                continue

            if not rfp.get("is_deleted"):
                results.append({"id": rfp_id, "success": False, "error": "RFP is not deleted"})
                failure_count += 1
                continue

            # Check retention period
            permanent_delete_at = rfp.get("permanent_delete_at")
            if permanent_delete_at:
                if isinstance(permanent_delete_at, str):
                    permanent_delete_at = datetime.fromisoformat(permanent_delete_at)
                if datetime.now() > permanent_delete_at:
                    results.append({"id": rfp_id, "success": False, "error": "Retention period expired"})
                    failure_count += 1
                    continue

            store.update(rfp_id, {
                "is_deleted": False,
                "deleted_at": None,
                "deleted_by": None,
                "delete_reason": None,
                "permanent_delete_at": None,
                "status": rfp.get("_previous_status", "created")
            })

            results.append({"id": rfp_id, "success": True, "action": "restored"})
            success_count += 1

        except Exception as e:
            results.append({"id": rfp_id, "success": False, "error": str(e)})
            failure_count += 1

    logger.info(f"Bulk restore: {success_count} succeeded, {failure_count} failed")

    return BulkOperationResult(
        success_count=success_count,
        failure_count=failure_count,
        results=results
    )


@app.post("/api/rfp/bulk/export")
async def bulk_export_rfps(
    request: BulkOperationRequest,
    format: str = "json",
):
    """
    Bulk export multiple RFPs.

    Query Parameters:
    - format: Export format (json, csv)
    """
    export_data = []

    for rfp_id in request.ids:
        rfp = store.get(rfp_id)
        if rfp:
            export_data.append({
                "id": rfp["id"],
                "name": rfp["name"],
                "solicitation_number": rfp.get("solicitation_number"),
                "agency": rfp.get("agency"),
                "status": rfp["status"],
                "files_count": len(rfp.get("files", [])),
                "requirements_count": len(rfp.get("requirements", [])),
                "created_at": rfp.get("created_at"),
                "updated_at": rfp.get("updated_at"),
                "requirements": rfp.get("requirements", [])
            })

    if format == "csv":
        # Convert to CSV
        import io
        import csv

        output = io.StringIO()
        if export_data:
            # Flatten requirements for CSV
            flat_data = []
            for rfp in export_data:
                base = {k: v for k, v in rfp.items() if k != "requirements"}
                if rfp.get("requirements"):
                    for req in rfp["requirements"]:
                        flat_data.append({**base, **{"req_" + k: v for k, v in req.items()}})
                else:
                    flat_data.append(base)

            if flat_data:
                writer = csv.DictWriter(output, fieldnames=flat_data[0].keys())
                writer.writeheader()
                writer.writerows(flat_data)

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=rfp_export.csv"}
        )

    return {
        "format": "json",
        "count": len(export_data),
        "data": export_data
    }


@app.post("/api/rfp/bulk/update-status")
async def bulk_update_status(
    ids: List[str],
    status: str,
    authorization: str = Header(None),
):
    """
    Bulk update status for multiple RFPs.
    """
    valid_statuses = ["created", "files_uploaded", "processing", "processed", "reviewed", "archived"]
    if status not in valid_statuses:
        raise HTTPException(400, f"Invalid status. Must be one of: {', '.join(valid_statuses)}")

    results = []
    success_count = 0
    failure_count = 0

    for rfp_id in ids:
        try:
            rfp = store.get(rfp_id)
            if not rfp:
                results.append({"id": rfp_id, "success": False, "error": "RFP not found"})
                failure_count += 1
                continue

            store.update(rfp_id, {"status": status})
            results.append({"id": rfp_id, "success": True, "new_status": status})
            success_count += 1

        except Exception as e:
            results.append({"id": rfp_id, "success": False, "error": str(e)})
            failure_count += 1

    logger.info(f"Bulk status update to '{status}': {success_count} succeeded, {failure_count} failed")

    return BulkOperationResult(
        success_count=success_count,
        failure_count=failure_count,
        results=results
    )


# ============== File Upload ==============

@app.post("/api/rfp/{rfp_id}/upload")
async def upload_files(
    rfp_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload RFP documents with security validation.

    Validates:
    - File extension (PDF, DOCX, DOC, XLSX, XLS)
    - File size (max 50 MB)
    - File content (magic bytes verification)
    - Filename sanitization
    """
    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    # Create upload directory for this RFP
    rfp_dir = UPLOAD_DIR / rfp_id
    rfp_dir.mkdir(exist_ok=True)

    uploaded = []
    skipped = []
    file_paths = list(rfp["file_paths"])
    file_names = list(rfp["files"])

    for file in files:
        try:
            # Validate file security
            content, safe_filename = await validate_uploaded_file(file)

            # Get extension for response
            ext = Path(safe_filename).suffix.lower()

            # Save file with sanitized name
            file_path = rfp_dir / safe_filename
            with open(file_path, "wb") as f:
                f.write(content)

            file_paths.append(str(file_path))
            file_names.append(safe_filename)

            uploaded.append({
                "name": safe_filename,
                "original_name": file.filename,
                "size": len(content),
                "type": ext[1:].upper()
            })

            logger.info(
                f"File uploaded successfully: {safe_filename}",
                extra={"rfp_id": rfp_id, "file_size": len(content)}
            )

        except FileValidationError as e:
            skipped.append({
                "name": file.filename,
                "reason": str(e)
            })
            logger.warning(
                f"File upload rejected: {file.filename} - {str(e)}",
                extra={"rfp_id": rfp_id}
            )

    # Update store
    store.update(rfp_id, {
        "files": file_names,
        "file_paths": file_paths,
        "status": "files_uploaded"
    })

    return {
        "status": "uploaded",
        "files": uploaded,
        "skipped": skipped,
        "total_files": len(file_names),
        "max_file_size_mb": get_max_file_size_mb()
    }


@app.get("/api/upload-constraints")
async def get_upload_constraints():
    """
    Get file upload constraints for client-side validation.

    Returns allowed file types, maximum file size, and other constraints.
    """
    return {
        "max_file_size_bytes": MAX_FILE_SIZE,
        "max_file_size_mb": get_max_file_size_mb(),
        "allowed_extensions": list(ALLOWED_EXTENSIONS.keys()),
        "allowed_mime_types": {
            ext: mimes for ext, mimes in ALLOWED_EXTENSIONS.items()
        },
        "constraints": {
            "max_filename_length": 255,
            "empty_files_allowed": False,
            "content_validation": True,  # Magic bytes verification
            "filename_sanitization": True
        }
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
    Upload RFP documents with explicit document type tags and security validation.

    This endpoint supports the guided upload UI where users specify
    what type of document each file is (SOW, Section L, Section M, etc.)

    Validates:
    - File extension (PDF, DOCX, DOC, XLSX, XLS)
    - File size (max 50 MB)
    - File content (magic bytes verification)
    - Filename sanitization

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
    skipped = []
    file_paths = list(rfp.get("file_paths", []))
    file_names = list(rfp.get("files", []))

    # Initialize document metadata if not present
    doc_metadata = rfp.get("document_metadata", {})

    for i, file in enumerate(files):
        try:
            # Validate file security
            content, safe_filename = await validate_uploaded_file(file)

            # Get extension for response
            ext = Path(safe_filename).suffix.lower()

            # Get document type (from form or auto-detect)
            if i < len(doc_type_list) and doc_type_list[i] and doc_type_list[i] != "auto_detect":
                doc_type = doc_type_list[i]
            elif GUIDED_UPLOAD_AVAILABLE:
                doc_type = classify_document_by_filename(safe_filename).value
            else:
                doc_type = "auto_detect"

            # Save file with sanitized name
            file_path = rfp_dir / safe_filename
            with open(file_path, "wb") as f:
                f.write(content)

            file_paths.append(str(file_path))
            file_names.append(safe_filename)

            # Store document metadata
            doc_metadata[safe_filename] = {
                "doc_type": doc_type,
                "file_path": str(file_path),
                "size": len(content),
                "uploaded_at": datetime.now().isoformat(),
                "original_name": file.filename
            }

            uploaded.append({
                "name": safe_filename,
                "original_name": file.filename,
                "size": len(content),
                "type": ext[1:].upper(),
                "doc_type": doc_type,
                "doc_type_label": _get_doc_type_label(doc_type)
            })

            logger.info(
                f"File uploaded successfully: {safe_filename}",
                extra={"rfp_id": rfp_id, "doc_type": doc_type, "file_size": len(content)}
            )

        except FileValidationError as e:
            skipped.append({
                "name": file.filename,
                "reason": str(e)
            })
            logger.warning(
                f"File upload rejected: {file.filename} - {str(e)}",
                extra={"rfp_id": rfp_id}
            )

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
        "skipped": skipped,
        "total_files": len(file_names),
        "document_summary": _summarize_documents(doc_metadata),
        "max_file_size_mb": get_max_file_size_mb()
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
    page: int = 1,
    page_size: int = 50
):
    """
    Get requirements with optional filters and pagination.

    Query Parameters:
    - type: Filter by requirement type
    - priority: Filter by priority level
    - section: Filter by document section
    - search: Search in requirement text and ID
    - page: Page number (1-indexed, default 1)
    - page_size: Items per page (1-100, default 50)
    """
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

    total = len(requirements)

    # If page_size is 0, return all requirements (no pagination)
    if page_size == 0:
        return {
            "items": requirements,
            "total": total,
            "page": 1,
            "page_size": total,
            "total_pages": 1,
            "has_next": False,
            "has_prev": False
        }

    # Apply pagination
    offset, limit = get_pagination_params(page, page_size)
    paginated = requirements[offset:offset + limit]

    return paginate(paginated, page, limit, total).model_dump()


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
        # v4.2: Include Company Library data for win theme generation
        company_library_data = None
        if COMPANY_LIBRARY_AVAILABLE and company_library:
            try:
                company_library_data = company_library.get_profile()
            except Exception as e:
                print(f"[WARN] Could not fetch Company Library: {e}")

        outline = generator.generate_from_compliance_matrix(
            section_l_requirements=section_l,
            section_m_requirements=section_m,
            technical_requirements=technical,
            stats=stats,
            company_library_data=company_library_data
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

        # v4.2: Include Company Library data for win theme generation
        company_library_data = None
        if COMPANY_LIBRARY_AVAILABLE and company_library:
            try:
                company_library_data = company_library.get_profile()
            except Exception:
                pass

        outline_obj = generator.generate_from_compliance_matrix(
            section_l_requirements=section_l,
            section_m_requirements=section_m,
            technical_requirements=technical,
            stats=stats,
            company_library_data=company_library_data
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

        # v4.2: Include Company Library data for win theme generation
        company_library_data = None
        if COMPANY_LIBRARY_AVAILABLE and company_library:
            try:
                company_library_data = company_library.get_profile()
                print(f"[DEBUG] Company Library loaded with {len(company_library_data.get('differentiators', []))} differentiators")
            except Exception as e:
                print(f"[WARN] Could not fetch Company Library: {e}")

        outline_obj = generator.generate_from_compliance_matrix(
            section_l_requirements=section_l,
            section_m_requirements=section_m,
            technical_requirements=technical,
            stats=stats,
            company_library_data=company_library_data
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
    UserModel, TeamModel, TeamMembershipModel, ActivityLogModel, APIKeyModel,
    TeamInvitationModel, UserRole, UserSessionModel, WebhookModel, WebhookDeliveryModel
)
import uuid
import hashlib
import secrets
from datetime import timedelta

# Email service for password reset and invitations
try:
    from api.email_service import email_service, EmailService
    EMAIL_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Email service not available: {e}")
    EMAIL_SERVICE_AVAILABLE = False
    email_service = None

# Redis for distributed rate limiting and session storage
try:
    import redis.asyncio as redis
    REDIS_URL = os.environ.get("REDIS_URL", "")
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        REDIS_AVAILABLE = True
        print(f"[Redis] Connected to Redis for rate limiting")
    else:
        redis_client = None
        REDIS_AVAILABLE = False
except ImportError:
    redis_client = None
    REDIS_AVAILABLE = False
except Exception as e:
    print(f"Warning: Redis not available: {e}")
    redis_client = None
    REDIS_AVAILABLE = False

# JWT Authentication
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

# JWT Configuration
_jwt_secret_env = os.environ.get("JWT_SECRET")
PROPELAI_ENV = os.environ.get("PROPELAI_ENV", "development")

# Fail loudly in production if JWT_SECRET is not set
if PROPELAI_ENV == "production" and not _jwt_secret_env:
    raise RuntimeError(
        "CRITICAL: JWT_SECRET environment variable must be set in production. "
        "Generate a secure secret with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

# Warn in development if using default secret
if not _jwt_secret_env:
    import warnings
    warnings.warn(
        "Using default JWT_SECRET for development. Set JWT_SECRET env var for production.",
        UserWarning
    )

JWT_SECRET = _jwt_secret_env or "propelai-dev-secret-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24 * 7  # 7 days


def generate_id() -> str:
    """Generate a short unique ID"""
    return str(uuid.uuid4())[:8].upper()


def hash_password(password: str) -> str:
    """Hash password using bcrypt (production-grade security)"""
    if BCRYPT_AVAILABLE and pwd_context:
        return pwd_context.hash(password)
    # Fallback to SHA256 only if bcrypt unavailable (NOT recommended for production)
    import hashlib
    logger.warning("Using SHA256 fallback for password hashing - install passlib[bcrypt] for production")
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash using bcrypt"""
    if BCRYPT_AVAILABLE and pwd_context:
        try:
            return pwd_context.verify(password, password_hash)
        except Exception:
            # Handle legacy SHA256 hashes during migration
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest() == password_hash
    # Fallback for SHA256
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest() == password_hash


def create_jwt_token(user_id: str, email: str, name: str) -> str:
    """Create a JWT token for a user"""
    if not JWT_AVAILABLE:
        return f"demo-token-{user_id}"

    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> dict:
    """Verify and decode a JWT token. Returns payload or raises exception."""
    if not JWT_AVAILABLE:
        # Demo mode - extract user_id from demo token
        if token.startswith("demo-token-"):
            return {"sub": token.replace("demo-token-", ""), "demo": True}
        raise ValueError("Invalid token")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


async def get_user_from_token(token: str) -> Optional[Dict]:
    """Get the current user from a JWT token"""
    try:
        payload = verify_jwt_token(token)
        user_id = payload.get("sub")

        async with get_db_session() as session:
            if session is None:
                return None

            from sqlalchemy import select
            result = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            user = result.scalar_one_or_none()
            return user.to_dict() if user else None
    except ValueError:
        return None


# Current user context (for backwards compatibility)
_current_user: Optional[Dict] = None


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Hybrid rate limiter using sliding window algorithm.
    Uses Redis when available (for distributed systems), falls back to in-memory.
    """

    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def _redis_is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """Redis-based rate limiting using sorted sets"""
        try:
            now = time.time()
            window_start = now - window_seconds
            redis_key = f"rate_limit:{key}"

            # Use pipeline for atomic operations
            pipe = redis_client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, window_start)

            # Count current entries
            pipe.zcard(redis_key)

            # Add current request
            pipe.zadd(redis_key, {str(now): now})

            # Set expiry on the key
            pipe.expire(redis_key, window_seconds + 60)

            results = await pipe.execute()
            current_count = results[1]

            if current_count >= max_requests:
                # Get oldest entry to calculate retry-after
                oldest = await redis_client.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = float(oldest[0][1])
                    retry_after = int(oldest_time + window_seconds - now) + 1
                    return True, max(retry_after, 1)
                return True, window_seconds

            return False, 0

        except Exception as e:
            print(f"[RateLimiter] Redis error, falling back to memory: {e}")
            return await self._memory_is_rate_limited(key, max_requests, window_seconds)

    async def _memory_is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """In-memory rate limiting using sliding window"""
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds

            # Get existing requests for this key
            if key not in self._requests:
                self._requests[key] = []

            # Remove expired requests (outside the window)
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            # Check if over limit
            if len(self._requests[key]) >= max_requests:
                # Calculate retry-after time
                oldest_request = min(self._requests[key])
                retry_after = int(oldest_request + window_seconds - now) + 1
                return True, max(retry_after, 1)

            # Record this request
            self._requests[key].append(now)
            return False, 0

    async def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int = 60
    ) -> tuple[bool, int]:
        """
        Check if a key is rate limited.

        Returns:
            tuple: (is_limited, retry_after_seconds)
        """
        if REDIS_AVAILABLE and redis_client:
            return await self._redis_is_rate_limited(key, max_requests, window_seconds)
        return await self._memory_is_rate_limited(key, max_requests, window_seconds)

    async def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Remove entries older than max_age to prevent memory leaks (in-memory only)"""
        async with self._lock:
            cutoff = time.time() - max_age_seconds
            keys_to_remove = []

            for key, timestamps in self._requests.items():
                self._requests[key] = [ts for ts in timestamps if ts > cutoff]
                if not self._requests[key]:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._requests[key]


# Global rate limiter instance
rate_limiter = RateLimiter()

# Rate limit configurations (requests per minute)
RATE_LIMITS = {
    "login": {"max_requests": 5, "window_seconds": 60},         # 5 attempts per minute
    "register": {"max_requests": 3, "window_seconds": 60},      # 3 registrations per minute
    "forgot_password": {"max_requests": 3, "window_seconds": 300},  # 3 requests per 5 minutes
    "api_general": {"max_requests": 100, "window_seconds": 60}, # 100 requests per minute
}


async def check_rate_limit(request, limit_type: str) -> None:
    """
    Check rate limit for a request. Raises HTTPException if rate limited.

    Args:
        request: FastAPI Request object
        limit_type: Key from RATE_LIMITS dict
    """
    # Get client IP (handle proxies)
    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    # Build rate limit key
    key = f"{limit_type}:{client_ip}"

    # Get limit config
    config = RATE_LIMITS.get(limit_type, RATE_LIMITS["api_general"])

    # Check rate limit
    is_limited, retry_after = await rate_limiter.is_rate_limited(
        key,
        config["max_requests"],
        config["window_seconds"]
    )

    if is_limited:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Too many requests",
                "message": f"Rate limit exceeded. Please try again in {retry_after} seconds.",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )


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


# Password strength requirements
MIN_PASSWORD_LENGTH = 8
REQUIRE_UPPERCASE = True
REQUIRE_LOWERCASE = True
REQUIRE_DIGIT = True
REQUIRE_SPECIAL = False  # Optional for now


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets strength requirements.
    Returns (is_valid, error_message).
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters"

    if REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if REQUIRE_LOWERCASE and not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if REQUIRE_DIGIT and not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"

    if REQUIRE_SPECIAL and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain at least one special character"

    return True, ""


# Email verification configuration
EMAIL_VERIFICATION_EXPIRY_HOURS = 24
REQUIRE_EMAIL_VERIFICATION = os.environ.get("REQUIRE_EMAIL_VERIFICATION", "false").lower() == "true"


@app.post("/api/auth/register", tags=["Authentication"])
async def register_user(
    request: Request,
    email: str = Form(...),
    name: str = Form(...),
    password: str = Form(...),
):
    """
    Register a new user account.

    Creates a new user with email, name, and password. Returns a JWT token
    for immediate authentication. Rate limited to 3 registrations per minute.
    """
    # Rate limit: 3 registrations per minute
    await check_rate_limit(request, "register")

    # Validate password strength
    is_valid, error_msg = validate_password_strength(password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

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
        verification_token = secrets.token_urlsafe(32)

        user = UserModel(
            id=user_id,
            email=email,
            name=name,
            password_hash=hash_password(password),
            email_verified=not REQUIRE_EMAIL_VERIFICATION,  # Auto-verify if not required
            email_verification_token=verification_token if REQUIRE_EMAIL_VERIFICATION else None,
            email_verification_sent_at=datetime.utcnow() if REQUIRE_EMAIL_VERIFICATION else None,
        )
        session.add(user)
        await session.flush()

        # Send verification email if required
        email_sent = False
        if REQUIRE_EMAIL_VERIFICATION and EMAIL_SERVICE_AVAILABLE and email_service:
            email_sent = await email_service.send_email_verification(
                to_email=email,
                verification_token=verification_token,
                user_name=name
            )

        # Generate JWT token for immediate login after registration
        token = create_jwt_token(user.id, user.email, user.name)
        token_expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)

        # Create session record
        await create_session(
            user_id=user.id,
            token=token,
            request=request,
            expires_at=token_expiry
        )

        response = {
            "success": True,
            "user": user.to_dict(),
            "token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRY_HOURS * 3600,
            "message": "Registration successful",
            "email_verification_required": REQUIRE_EMAIL_VERIFICATION,
        }

        if REQUIRE_EMAIL_VERIFICATION:
            response["email_sent"] = email_sent
            if not email_sent:
                response["verification_token"] = verification_token
                response["note"] = "Email not configured - token provided directly"

        return response


@app.post("/api/auth/verify-email")
async def verify_email(
    token: str = Form(...),
):
    """Verify email address using verification token"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.email_verification_token == token)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=400, detail="Invalid verification token")

        if user.email_verified:
            return {"success": True, "message": "Email already verified"}

        # Check if token expired
        if user.email_verification_sent_at:
            expiry = user.email_verification_sent_at + timedelta(hours=EMAIL_VERIFICATION_EXPIRY_HOURS)
            if datetime.utcnow() > expiry:
                raise HTTPException(status_code=400, detail="Verification token has expired. Please request a new one.")

        # Mark email as verified
        user.email_verified = True
        user.email_verification_token = None
        await session.flush()

        return {"success": True, "message": "Email verified successfully"}


@app.post("/api/auth/resend-verification")
async def resend_verification_email(
    request: Request,
    authorization: str = Header(None),
):
    """Resend email verification email"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(404, "User not found")

        if user.email_verified:
            return {"success": True, "message": "Email already verified"}

        # Generate new token
        verification_token = secrets.token_urlsafe(32)
        user.email_verification_token = verification_token
        user.email_verification_sent_at = datetime.utcnow()
        await session.flush()

        # Send verification email
        email_sent = False
        if EMAIL_SERVICE_AVAILABLE and email_service:
            email_sent = await email_service.send_email_verification(
                to_email=user.email,
                verification_token=verification_token,
                user_name=user.name
            )

        response = {
            "success": True,
            "message": "Verification email sent",
            "email_sent": email_sent,
        }

        if not email_sent:
            response["verification_token"] = verification_token
            response["note"] = "Email not configured - token provided directly"

        return response


# Account lockout configuration
ACCOUNT_LOCKOUT_THRESHOLD = 5  # Lock after 5 failed attempts
ACCOUNT_LOCKOUT_DURATION_MINUTES = 15  # Lock for 15 minutes


@app.post("/api/auth/login", tags=["Authentication"])
async def login_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    """
    Login user and return JWT token.

    Authenticates with email and password. If 2FA is enabled, returns a
    challenge requiring TOTP verification. Rate limited to 5 attempts per minute.
    Account locks after 5 failed attempts for 15 minutes.
    """
    # Rate limit: 5 attempts per minute
    await check_rate_limit(request, "login")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()

        # Check if account is locked
        if user and user.locked_until:
            if datetime.utcnow() < user.locked_until:
                remaining_minutes = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
                raise HTTPException(
                    status_code=403,
                    detail=f"Account is locked. Try again in {remaining_minutes} minute(s)."
                )
            else:
                # Lockout expired, reset
                user.locked_until = None
                user.failed_login_attempts = 0

        if not user or not verify_password(password, user.password_hash):
            # Track failed attempt if user exists
            if user:
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1

                if user.failed_login_attempts >= ACCOUNT_LOCKOUT_THRESHOLD:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=ACCOUNT_LOCKOUT_DURATION_MINUTES)
                    await session.flush()
                    raise HTTPException(
                        status_code=403,
                        detail=f"Account locked due to too many failed attempts. Try again in {ACCOUNT_LOCKOUT_DURATION_MINUTES} minutes."
                    )

                await session.flush()

            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Reset failed attempts on successful login
        if user.failed_login_attempts:
            user.failed_login_attempts = 0
            user.locked_until = None

        # Check if 2FA is enabled
        if user.totp_enabled:
            # Return 2FA challenge instead of logging in directly
            return {
                "success": True,
                "requires_2fa": True,
                "user_id": user.id,
                "message": "Please enter your 2FA code",
            }

        user.last_login = datetime.utcnow()
        await session.flush()

        # Generate JWT token
        token = create_jwt_token(user.id, user.email, user.name)
        token_expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)

        # Create session record for session management
        await create_session(
            user_id=user.id,
            token=token,
            request=request,
            expires_at=token_expiry
        )

        return {
            "success": True,
            "requires_2fa": False,
            "user": user.to_dict(),
            "token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRY_HOURS * 3600,  # seconds
        }


@app.get("/api/users/me")
async def get_current_user_info(authorization: str = Header(None)):
    """Get current user info from JWT token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Extract token from "Bearer <token>" format
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = parts[1]
    user = await get_user_from_token(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user


@app.put("/api/users/me")
async def update_user_profile(
    authorization: str = Header(None),
    name: str = Form(None),
):
    """Update current user's profile"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = parts[1]

    try:
        payload = verify_jwt_token(token)
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update fields if provided
        if name is not None and name.strip():
            user.name = name.strip()

        await session.flush()

        # Generate new token with updated info
        new_token = create_jwt_token(user.id, user.email, user.name)

        return {
            "success": True,
            "user": user.to_dict(),
            "token": new_token,
            "message": "Profile updated successfully",
        }


@app.post("/api/users/me/change-password")
async def change_password(
    authorization: str = Header(None),
    current_password: str = Form(...),
    new_password: str = Form(...),
):
    """Change current user's password"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = parts[1]

    try:
        payload = verify_jwt_token(token)
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Verify current password
        if not verify_password(current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Update password
        user.password_hash = hash_password(new_password)
        await session.flush()

        return {
            "success": True,
            "message": "Password changed successfully",
        }


@app.get("/api/users/me/export")
async def export_user_data(
    authorization: str = Header(None),
):
    """
    Export all user data (GDPR compliance).
    Returns a JSON file with all data associated with the user.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        # Get user data
        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Build export data
        export_data = {
            "export_date": datetime.utcnow().isoformat() + "Z",
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "avatar_url": user.avatar_url,
                "is_active": user.is_active,
                "email_verified": user.email_verified,
                "totp_enabled": user.totp_enabled,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None,
            },
            "team_memberships": [],
            "sessions": [],
            "activity": [],
        }

        # Get team memberships
        result = await session.execute(
            select(TeamMembershipModel).where(TeamMembershipModel.user_id == user_id)
        )
        memberships = result.scalars().all()

        for membership in memberships:
            # Get team info
            team_result = await session.execute(
                select(TeamModel).where(TeamModel.id == membership.team_id)
            )
            team = team_result.scalar_one_or_none()

            export_data["team_memberships"].append({
                "team_id": membership.team_id,
                "team_name": team.name if team else "Unknown",
                "role": membership.role,
                "joined_at": membership.created_at.isoformat() if membership.created_at else None,
            })

        # Get sessions
        result = await session.execute(
            select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        )
        sessions = result.scalars().all()

        for sess in sessions:
            export_data["sessions"].append({
                "id": sess.id,
                "device_info": sess.device_info,
                "ip_address": sess.ip_address,
                "is_active": sess.revoked_at is None and datetime.utcnow() < sess.expires_at,
                "created_at": sess.created_at.isoformat() if sess.created_at else None,
                "last_active": sess.last_active.isoformat() if sess.last_active else None,
            })

        # Get activity logs where user was the actor
        result = await session.execute(
            select(ActivityLogModel)
            .where(ActivityLogModel.user_id == user_id)
            .order_by(ActivityLogModel.created_at.desc())
            .limit(100)
        )
        activities = result.scalars().all()

        for activity in activities:
            export_data["activity"].append({
                "action": activity.action,
                "resource_type": activity.resource_type,
                "resource_id": activity.resource_id,
                "details": activity.details,
                "created_at": activity.created_at.isoformat() if activity.created_at else None,
            })

        # Return as downloadable JSON file
        return Response(
            content=json.dumps(export_data, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="propelai_export_{user_id}_{datetime.utcnow().strftime("%Y%m%d")}.json"'
            }
        )


@app.delete("/api/users/me")
async def delete_user_account(
    authorization: str = Header(None),
    password: str = Form(...),
):
    """
    Delete user account and all associated data (GDPR compliance).
    This action is irreversible.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select, delete

        # Get user
        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Verify password
        if not verify_password(password, user.password_hash):
            raise HTTPException(status_code=400, detail="Incorrect password")

        # Delete all user sessions
        await session.execute(
            delete(UserSessionModel).where(UserSessionModel.user_id == user_id)
        )

        # Delete team memberships (teams themselves are not deleted)
        await session.execute(
            delete(TeamMembershipModel).where(TeamMembershipModel.user_id == user_id)
        )

        # Delete the user (cascades to related data)
        await session.delete(user)
        await session.flush()

        logger.info(f"User account deleted: {user_id}")

        return {
            "success": True,
            "message": "Account deleted successfully. All your data has been removed.",
        }


@app.post("/api/auth/verify")
async def verify_token(authorization: str = Header(None)):
    """Verify a JWT token and return user info"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = parts[1]

    try:
        payload = verify_jwt_token(token)
        user = await get_user_from_token(token)

        return {
            "valid": True,
            "user": user,
            "expires_at": payload.get("exp"),
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/refresh")
async def refresh_token(authorization: str = Header(None)):
    """Refresh a JWT token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = parts[1]

    try:
        payload = verify_jwt_token(token)
        user_id = payload.get("sub")
        email = payload.get("email")
        name = payload.get("name")

        # Create a new token
        new_token = create_jwt_token(user_id, email, name)

        return {
            "success": True,
            "token": new_token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRY_HOURS * 3600,
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


# Password Reset Token Storage (use Redis or database in production)
_password_reset_tokens: Dict[str, Dict] = {}
PASSWORD_RESET_EXPIRY_HOURS = 1  # Reset tokens expire in 1 hour


def generate_reset_token() -> str:
    """Generate a secure password reset token"""
    return secrets.token_urlsafe(32)


@app.post("/api/auth/forgot-password")
async def forgot_password(
    request: Request,
    email: str = Form(...),
):
    """Request a password reset token"""
    # Rate limit: 3 requests per 5 minutes
    await check_rate_limit(request, "forgot_password")

    async with get_db_session() as session:
        if session is None:
            # Still return success to prevent email enumeration
            return {"success": True, "message": "If an account exists, a reset link has been sent"}

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()

        if user:
            # Generate reset token
            reset_token = generate_reset_token()
            expiry = datetime.utcnow() + timedelta(hours=PASSWORD_RESET_EXPIRY_HOURS)

            # Store token
            _password_reset_tokens[reset_token] = {
                "user_id": user.id,
                "email": user.email,
                "expires_at": expiry,
            }

            # Send password reset email
            email_sent = False
            if EMAIL_SERVICE_AVAILABLE and email_service:
                email_sent = await email_service.send_password_reset(
                    to_email=user.email,
                    reset_token=reset_token,
                    user_name=user.name
                )

            # In console mode or if email fails, still return the token for development
            response = {
                "success": True,
                "message": "If an account exists, a reset link has been sent",
                "email_sent": email_sent,
            }

            # Only include token in development (console email mode)
            if not email_sent:
                response["reset_token"] = reset_token
                response["note"] = "Email not configured - token provided directly"

            return response

        # Return same response to prevent email enumeration
        return {"success": True, "message": "If an account exists, a reset link has been sent"}


@app.post("/api/auth/reset-password")
async def reset_password(
    token: str = Form(...),
    new_password: str = Form(...),
):
    """Reset password using a reset token"""
    # Check if token exists and is valid
    token_data = _password_reset_tokens.get(token)

    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if datetime.utcnow() > token_data["expires_at"]:
        # Remove expired token
        del _password_reset_tokens[token]
        raise HTTPException(status_code=400, detail="Reset token has expired")

    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(status_code=500, detail="Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == token_data["user_id"])
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=400, detail="User not found")

        # Update password
        user.password_hash = hash_password(new_password)
        user.updated_at = datetime.utcnow()
        await session.flush()

        # Remove used token
        del _password_reset_tokens[token]

        # Generate new JWT token for immediate login
        new_token = create_jwt_token(user.id, user.email, user.name)

        return {
            "success": True,
            "message": "Password reset successful",
            "user": user.to_dict(),
            "token": new_token,
            "token_type": "bearer",
        }


# ============== TWO-FACTOR AUTHENTICATION ==============

# Try to import pyotp for TOTP
try:
    import pyotp
    import base64
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    pyotp = None


def generate_backup_codes(count: int = 8) -> list:
    """Generate a list of backup codes for 2FA recovery"""
    return [secrets.token_hex(4).upper() for _ in range(count)]


@app.post("/api/auth/2fa/setup")
async def setup_2fa(
    authorization: str = Header(None),
):
    """Begin 2FA setup - generates TOTP secret and returns QR code data"""
    if not TOTP_AVAILABLE:
        raise HTTPException(500, "2FA is not available (pyotp not installed)")

    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
        user_email = payload.get("email")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(404, "User not found")

        if user.totp_enabled:
            raise HTTPException(400, "2FA is already enabled. Disable it first to reconfigure.")

        # Generate new TOTP secret
        secret = pyotp.random_base32()

        # Store secret (not enabled yet - requires verification)
        user.totp_secret = secret
        await session.flush()

        # Generate provisioning URI for QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_email,
            issuer_name="PropelAI"
        )

        return {
            "success": True,
            "secret": secret,  # User can manually enter this in their app
            "provisioning_uri": provisioning_uri,  # For QR code generation
            "message": "Scan the QR code with your authenticator app, then verify with a code",
        }


@app.post("/api/auth/2fa/verify-setup")
async def verify_2fa_setup(
    code: str = Form(...),
    authorization: str = Header(None),
):
    """Verify 2FA setup with first code and enable 2FA"""
    if not TOTP_AVAILABLE:
        raise HTTPException(500, "2FA is not available")

    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(404, "User not found")

        if not user.totp_secret:
            raise HTTPException(400, "2FA setup not started. Call /api/auth/2fa/setup first.")

        if user.totp_enabled:
            raise HTTPException(400, "2FA is already enabled")

        # Verify the code
        totp = pyotp.TOTP(user.totp_secret)
        if not totp.verify(code, valid_window=1):
            raise HTTPException(400, "Invalid verification code")

        # Generate backup codes
        backup_codes = generate_backup_codes()

        # Enable 2FA
        user.totp_enabled = True
        user.totp_backup_codes = backup_codes
        await session.flush()

        return {
            "success": True,
            "message": "Two-factor authentication enabled successfully",
            "backup_codes": backup_codes,  # Show only once!
            "warning": "Save these backup codes in a secure location. They will not be shown again.",
        }


@app.post("/api/auth/2fa/disable")
async def disable_2fa(
    password: str = Form(...),
    authorization: str = Header(None),
):
    """Disable 2FA (requires password confirmation)"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(404, "User not found")

        if not user.totp_enabled:
            raise HTTPException(400, "2FA is not enabled")

        # Verify password
        if not verify_password(password, user.password_hash):
            raise HTTPException(400, "Invalid password")

        # Disable 2FA
        user.totp_enabled = False
        user.totp_secret = None
        user.totp_backup_codes = []
        await session.flush()

        return {
            "success": True,
            "message": "Two-factor authentication disabled",
        }


@app.post("/api/auth/2fa/verify")
async def verify_2fa_code(
    request: Request,
    user_id: str = Form(...),
    code: str = Form(...),
):
    """Verify 2FA code during login (called after password verification)"""
    if not TOTP_AVAILABLE:
        raise HTTPException(500, "2FA is not available")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(404, "User not found")

        if not user.totp_enabled:
            raise HTTPException(400, "2FA is not enabled for this user")

        # First, check if it's a backup code
        backup_codes = user.totp_backup_codes or []
        code_upper = code.upper().replace("-", "").replace(" ", "")

        if code_upper in backup_codes:
            # Use and remove backup code
            backup_codes.remove(code_upper)
            user.totp_backup_codes = backup_codes
            user.last_login = datetime.utcnow()
            await session.flush()

            # Generate JWT token
            token = create_jwt_token(user.id, user.email, user.name)
            token_expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)

            # Create session record for session management
            await create_session(
                user_id=user.id,
                token=token,
                request=request,
                expires_at=token_expiry
            )

            return {
                "success": True,
                "user": user.to_dict(),
                "token": token,
                "token_type": "bearer",
                "expires_in": JWT_EXPIRY_HOURS * 3600,
                "backup_code_used": True,
                "backup_codes_remaining": len(backup_codes),
            }

        # Check TOTP code
        totp = pyotp.TOTP(user.totp_secret)
        if not totp.verify(code, valid_window=1):
            raise HTTPException(400, "Invalid verification code")

        user.last_login = datetime.utcnow()
        await session.flush()

        # Generate JWT token
        token = create_jwt_token(user.id, user.email, user.name)
        token_expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)

        # Create session record for session management
        await create_session(
            user_id=user.id,
            token=token,
            request=request,
            expires_at=token_expiry
        )

        return {
            "success": True,
            "user": user.to_dict(),
            "token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRY_HOURS * 3600,
        }


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
    page: int = 1,
    page_size: int = 50,
):
    """
    Get team activity log with pagination.

    Query Parameters:
    - page: Page number (1-indexed, default 1)
    - page_size: Items per page (1-100, default 50)
    """
    offset, limit = get_pagination_params(page, page_size)

    async with get_db_session() as session:
        if session is None:
            return paginate([], page, limit, 0).model_dump()

        from sqlalchemy import select, func

        # Get total count
        count_result = await session.execute(
            select(func.count()).select_from(ActivityLogModel)
            .where(ActivityLogModel.team_id == team_id)
        )
        total = count_result.scalar() or 0

        # Get paginated results
        result = await session.execute(
            select(ActivityLogModel)
            .where(ActivityLogModel.team_id == team_id)
            .order_by(ActivityLogModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        activities = result.scalars().all()

        return paginate([a.to_dict() for a in activities], page, limit, total).model_dump()


async def log_activity(
    team_id: str,
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str = None,
    details: Dict = None,
    ip_address: str = None,
    user_agent: str = None,
    request_id: str = None,
):
    """
    Log an activity for audit trail with enhanced metadata.

    Args:
        team_id: Team identifier
        user_id: User who performed the action
        action: Action performed (create, update, delete, etc.)
        resource_type: Type of resource affected
        resource_id: ID of the affected resource
        details: Additional details about the action
        ip_address: Client IP address
        user_agent: Client user agent
        request_id: Correlation ID for request tracing
    """
    try:
        async with get_db_session() as session:
            if session is None:
                return

            # Get request ID from context if not provided
            if request_id is None:
                try:
                    request_id = get_request_id()
                except Exception:
                    pass

            activity = ActivityLogModel(
                id=generate_id(),
                team_id=team_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent[:500] if user_agent and len(user_agent) > 500 else user_agent,
                request_id=request_id,
            )
            session.add(activity)
            await session.flush()

            # Also log to structured logger
            logger.info(
                f"Activity: {action} on {resource_type}",
                extra={
                    "activity_id": activity.id,
                    "team_id": team_id,
                    "user_id": user_id,
                    "action": action,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "request_id": request_id,
                }
            )
    except Exception as e:
        logger.error(f"[Activity Log] Error: {e}")


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


# ============== WEBHOOKS ==============

# Supported webhook event types
WEBHOOK_EVENTS = [
    "rfp.created",
    "rfp.updated",
    "rfp.deleted",
    "rfp.processed",
    "requirement.extracted",
    "draft.started",
    "draft.completed",
    "draft.feedback_received",
    "team.member_added",
    "team.member_removed",
    "library.item_added",
    "library.item_updated",
]


async def trigger_webhook_event(
    team_id: str,
    event_type: str,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks = None,
):
    """
    Trigger webhooks for a specific event.

    This queues webhook deliveries to run in the background.
    """
    async with get_db_session() as session:
        if session is None:
            return

        from sqlalchemy import select

        # Find active webhooks subscribed to this event
        result = await session.execute(
            select(WebhookModel)
            .where(
                WebhookModel.team_id == team_id,
                WebhookModel.is_active == True
            )
        )
        webhooks = result.scalars().all()

        for webhook in webhooks:
            # Check if webhook is subscribed to this event
            if webhook.events and event_type not in webhook.events:
                continue

            # Queue the delivery
            if background_tasks:
                background_tasks.add_task(
                    deliver_webhook,
                    webhook.id,
                    event_type,
                    payload
                )
            else:
                # If no background tasks, run synchronously
                asyncio.create_task(deliver_webhook(webhook.id, event_type, payload))


async def deliver_webhook(webhook_id: str, event_type: str, payload: Dict[str, Any]):
    """
    Deliver a webhook with retry logic.
    """
    import aiohttp
    import hmac
    import hashlib

    async with get_db_session() as session:
        if session is None:
            return

        from sqlalchemy import select

        # Get webhook
        result = await session.execute(
            select(WebhookModel).where(WebhookModel.id == webhook_id)
        )
        webhook = result.scalar_one_or_none()
        if not webhook:
            return

        # Prepare payload
        delivery_payload = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": payload
        }
        payload_json = json.dumps(delivery_payload)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PropelAI-Webhook/4.1",
            "X-Webhook-Event": event_type,
            "X-Webhook-Delivery": generate_id(),
        }

        # Add signature if secret is set
        if webhook.secret:
            signature = hmac.new(
                webhook.secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        # Add custom headers
        if webhook.headers:
            headers.update(webhook.headers)

        # Attempt delivery with retries
        max_retries = webhook.retry_count or 3
        timeout = aiohttp.ClientTimeout(total=webhook.timeout_seconds or 30)

        response_status = None
        response_body = None
        error_message = None
        duration_ms = 0
        attempt_count = 0

        for attempt in range(max_retries):
            attempt_count = attempt + 1
            start_time = time.time()

            try:
                async with aiohttp.ClientSession(timeout=timeout) as client:
                    async with client.post(
                        webhook.url,
                        data=payload_json,
                        headers=headers
                    ) as response:
                        response_status = response.status
                        response_body = await response.text()
                        duration_ms = int((time.time() - start_time) * 1000)

                        # Success - exit retry loop
                        if 200 <= response_status < 300:
                            break

            except asyncio.TimeoutError:
                error_message = f"Timeout after {webhook.timeout_seconds}s"
                duration_ms = int((time.time() - start_time) * 1000)
            except aiohttp.ClientError as e:
                error_message = str(e)
                duration_ms = int((time.time() - start_time) * 1000)
            except Exception as e:
                error_message = f"Unexpected error: {str(e)}"
                duration_ms = int((time.time() - start_time) * 1000)

            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        # Record delivery
        delivery = WebhookDeliveryModel(
            id=generate_id(),
            webhook_id=webhook_id,
            event_type=event_type,
            payload=delivery_payload,
            response_status=response_status,
            response_body=response_body[:5000] if response_body else None,
            error_message=error_message,
            attempt_count=attempt_count,
            duration_ms=duration_ms,
        )
        session.add(delivery)

        # Update webhook stats
        webhook.last_triggered = datetime.utcnow()
        if response_status and 200 <= response_status < 300:
            webhook.last_success = datetime.utcnow()
            webhook.success_count = (webhook.success_count or 0) + 1
        else:
            webhook.last_failure = datetime.utcnow()
            webhook.failure_count = (webhook.failure_count or 0) + 1

        await session.flush()


@app.get("/api/webhooks/events")
async def list_webhook_events():
    """List all available webhook event types"""
    return {
        "events": WEBHOOK_EVENTS,
        "categories": {
            "rfp": ["rfp.created", "rfp.updated", "rfp.deleted", "rfp.processed"],
            "requirement": ["requirement.extracted"],
            "draft": ["draft.started", "draft.completed", "draft.feedback_received"],
            "team": ["team.member_added", "team.member_removed"],
            "library": ["library.item_added", "library.item_updated"],
        }
    }


@app.post("/api/teams/{team_id}/webhooks")
async def create_webhook(
    team_id: str,
    name: str = Form(...),
    url: str = Form(...),
    events: str = Form(""),  # Comma-separated event types
    secret: str = Form(None),
    authorization: str = Header(None),
):
    """Create a webhook subscription for a team"""
    # Validate user
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    # Validate URL
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL must start with http:// or https://")

    # Parse events
    event_list = [e.strip() for e in events.split(",") if e.strip()] if events else []

    # Validate events
    for event in event_list:
        if event not in WEBHOOK_EVENTS:
            raise HTTPException(400, f"Invalid event type: {event}")

    # Generate secret if not provided
    if not secret:
        secret = secrets.token_hex(32)

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        # Create webhook
        webhook = WebhookModel(
            id=generate_id(),
            team_id=team_id,
            user_id=user_id,
            name=name,
            url=url,
            secret=secret,
            events=event_list if event_list else None,  # None means all events
            is_active=True,
        )
        session.add(webhook)
        await session.flush()

        # Log activity
        await log_activity(
            team_id=team_id,
            user_id=user_id,
            action="create",
            resource_type="webhook",
            resource_id=webhook.id,
            details={"name": name, "url": url, "events": event_list}
        )

        return {
            "id": webhook.id,
            "name": webhook.name,
            "url": webhook.url,
            "events": webhook.events or WEBHOOK_EVENTS,
            "secret": secret,  # Only shown once!
            "is_active": webhook.is_active,
            "message": "Store the secret securely - it will not be shown again!"
        }


@app.get("/api/teams/{team_id}/webhooks")
async def list_webhooks(team_id: str):
    """List all webhooks for a team"""
    async with get_db_session() as session:
        if session is None:
            return {"webhooks": []}

        from sqlalchemy import select

        result = await session.execute(
            select(WebhookModel)
            .where(WebhookModel.team_id == team_id)
            .order_by(WebhookModel.created_at.desc())
        )
        webhooks = result.scalars().all()

        return {
            "webhooks": [w.to_dict() for w in webhooks]
        }


@app.get("/api/teams/{team_id}/webhooks/{webhook_id}")
async def get_webhook(team_id: str, webhook_id: str):
    """Get webhook details"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(WebhookModel)
            .where(
                WebhookModel.id == webhook_id,
                WebhookModel.team_id == team_id
            )
        )
        webhook = result.scalar_one_or_none()

        if not webhook:
            raise HTTPException(404, "Webhook not found")

        return webhook.to_dict()


@app.put("/api/teams/{team_id}/webhooks/{webhook_id}")
async def update_webhook(
    team_id: str,
    webhook_id: str,
    name: str = Form(None),
    url: str = Form(None),
    events: str = Form(None),
    is_active: bool = Form(None),
    authorization: str = Header(None),
):
    """Update a webhook"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(WebhookModel)
            .where(
                WebhookModel.id == webhook_id,
                WebhookModel.team_id == team_id
            )
        )
        webhook = result.scalar_one_or_none()

        if not webhook:
            raise HTTPException(404, "Webhook not found")

        # Update fields
        if name is not None:
            webhook.name = name
        if url is not None:
            if not url.startswith(("http://", "https://")):
                raise HTTPException(400, "URL must start with http:// or https://")
            webhook.url = url
        if events is not None:
            event_list = [e.strip() for e in events.split(",") if e.strip()]
            for event in event_list:
                if event not in WEBHOOK_EVENTS:
                    raise HTTPException(400, f"Invalid event type: {event}")
            webhook.events = event_list if event_list else None
        if is_active is not None:
            webhook.is_active = is_active

        await session.flush()

        return webhook.to_dict()


@app.delete("/api/teams/{team_id}/webhooks/{webhook_id}")
async def delete_webhook(
    team_id: str,
    webhook_id: str,
    authorization: str = Header(None),
):
    """Delete a webhook"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select, delete

        # Get webhook for logging
        result = await session.execute(
            select(WebhookModel)
            .where(
                WebhookModel.id == webhook_id,
                WebhookModel.team_id == team_id
            )
        )
        webhook = result.scalar_one_or_none()

        if not webhook:
            raise HTTPException(404, "Webhook not found")

        webhook_name = webhook.name

        # Delete
        await session.execute(
            delete(WebhookModel).where(WebhookModel.id == webhook_id)
        )
        await session.flush()

        # Log activity
        await log_activity(
            team_id=team_id,
            user_id=user_id,
            action="delete",
            resource_type="webhook",
            resource_id=webhook_id,
            details={"name": webhook_name}
        )

        return {"success": True, "message": f"Webhook '{webhook_name}' deleted"}


@app.get("/api/teams/{team_id}/webhooks/{webhook_id}/deliveries")
async def list_webhook_deliveries(
    team_id: str,
    webhook_id: str,
    page: int = 1,
    page_size: int = 20,
):
    """List recent webhook deliveries"""
    offset, limit = get_pagination_params(page, page_size)

    async with get_db_session() as session:
        if session is None:
            return paginate([], page, limit, 0).model_dump()

        from sqlalchemy import select, func

        # Verify webhook exists and belongs to team
        result = await session.execute(
            select(WebhookModel)
            .where(
                WebhookModel.id == webhook_id,
                WebhookModel.team_id == team_id
            )
        )
        webhook = result.scalar_one_or_none()
        if not webhook:
            raise HTTPException(404, "Webhook not found")

        # Get total count
        count_result = await session.execute(
            select(func.count()).select_from(WebhookDeliveryModel)
            .where(WebhookDeliveryModel.webhook_id == webhook_id)
        )
        total = count_result.scalar() or 0

        # Get deliveries
        result = await session.execute(
            select(WebhookDeliveryModel)
            .where(WebhookDeliveryModel.webhook_id == webhook_id)
            .order_by(WebhookDeliveryModel.delivered_at.desc())
            .offset(offset)
            .limit(limit)
        )
        deliveries = result.scalars().all()

        return paginate([d.to_dict() for d in deliveries], page, limit, total).model_dump()


@app.post("/api/teams/{team_id}/webhooks/{webhook_id}/test")
async def test_webhook(
    team_id: str,
    webhook_id: str,
    authorization: str = Header(None),
):
    """Send a test event to a webhook"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(WebhookModel)
            .where(
                WebhookModel.id == webhook_id,
                WebhookModel.team_id == team_id
            )
        )
        webhook = result.scalar_one_or_none()

        if not webhook:
            raise HTTPException(404, "Webhook not found")

        # Send test event
        test_payload = {
            "test": True,
            "message": "This is a test webhook delivery from PropelAI",
            "webhook_id": webhook_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        await deliver_webhook(webhook_id, "test.ping", test_payload)

        return {
            "success": True,
            "message": "Test webhook sent. Check the deliveries endpoint for results."
        }


# ============== TEAM INVITATIONS ==============

INVITATION_EXPIRY_DAYS = 7


@app.post("/api/teams/{team_id}/invitations")
async def create_invitation(
    team_id: str,
    email: str = Form(...),
    role: str = Form("viewer"),
    authorization: str = Header(None),
):
    """Create a team invitation for a user (who may not have an account yet)"""
    # Get current user from token
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        inviter_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    if role not in ["admin", "contributor", "viewer"]:
        raise HTTPException(400, "Invalid role. Must be admin, contributor, or viewer")

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        # Check if team exists
        result = await session.execute(
            select(TeamModel).where(TeamModel.id == team_id)
        )
        team = result.scalar_one_or_none()
        if not team:
            raise HTTPException(404, "Team not found")

        # Check if user is already a member
        result = await session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            result = await session.execute(
                select(TeamMembershipModel)
                .where(TeamMembershipModel.team_id == team_id)
                .where(TeamMembershipModel.user_id == existing_user.id)
            )
            if result.scalar_one_or_none():
                raise HTTPException(400, "User is already a team member")

        # Check for existing pending invitation
        result = await session.execute(
            select(TeamInvitationModel)
            .where(TeamInvitationModel.team_id == team_id)
            .where(TeamInvitationModel.email == email)
            .where(TeamInvitationModel.status == "pending")
        )
        existing_invite = result.scalar_one_or_none()
        if existing_invite:
            raise HTTPException(400, "An invitation is already pending for this email")

        # Create invitation
        invitation_id = generate_id()
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=INVITATION_EXPIRY_DAYS)

        invitation = TeamInvitationModel(
            id=invitation_id,
            team_id=team_id,
            email=email,
            role=role,
            token=token,
            invited_by=inviter_id,
            status="pending",
            expires_at=expires_at,
        )
        session.add(invitation)
        await session.flush()

        # Get inviter name for email
        inviter = await session.execute(
            select(UserModel).where(UserModel.id == inviter_id)
        )
        inviter_user = inviter.scalar_one_or_none()
        inviter_name = inviter_user.name if inviter_user else "A team member"

        # Log activity
        await log_activity(
            team_id=team_id,
            user_id=inviter_id,
            action="create",
            resource_type="invitation",
            resource_id=invitation_id,
            details={"email": email, "role": role}
        )

        # Send invitation email
        email_sent = False
        if EMAIL_SERVICE_AVAILABLE and email_service:
            email_sent = await email_service.send_team_invitation(
                to_email=email,
                team_name=team.name,
                inviter_name=inviter_name,
                invitation_token=token,
                role=role
            )

        response = {
            "success": True,
            "invitation": invitation.to_dict(),
            "message": f"Invitation sent to {email}",
            "email_sent": email_sent,
        }

        # Only include token in development (console email mode)
        if not email_sent:
            response["invitation_token"] = token
            response["invitation_url"] = f"/accept-invite?token={token}"
            response["note"] = "Email not configured - token provided directly"

        return response


@app.get("/api/teams/{team_id}/invitations")
async def list_invitations(
    team_id: str,
    status: str = None,
):
    """List all invitations for a team"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        query = (
            select(TeamInvitationModel)
            .options(selectinload(TeamInvitationModel.inviter))
            .where(TeamInvitationModel.team_id == team_id)
            .order_by(TeamInvitationModel.created_at.desc())
        )

        if status:
            query = query.where(TeamInvitationModel.status == status)

        result = await session.execute(query)
        invitations = result.scalars().all()

        return {
            "invitations": [inv.to_dict() for inv in invitations],
        }


@app.get("/api/invitations/{token}")
async def get_invitation_by_token(token: str):
    """Get invitation details by token (for accept page)"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(TeamInvitationModel)
            .options(
                selectinload(TeamInvitationModel.team),
                selectinload(TeamInvitationModel.inviter)
            )
            .where(TeamInvitationModel.token == token)
        )
        invitation = result.scalar_one_or_none()

        if not invitation:
            raise HTTPException(404, "Invitation not found")

        if invitation.status != "pending":
            raise HTTPException(400, f"Invitation has been {invitation.status}")

        if datetime.utcnow() > invitation.expires_at:
            raise HTTPException(400, "Invitation has expired")

        return {
            "invitation": invitation.to_dict(),
            "team_name": invitation.team.name if invitation.team else None,
            "inviter_name": invitation.inviter.name if invitation.inviter else None,
        }


@app.post("/api/invitations/{token}/accept")
async def accept_invitation(
    token: str,
    authorization: str = Header(None),
):
    """Accept a team invitation (user must be logged in)"""
    if not authorization:
        raise HTTPException(401, "You must be logged in to accept an invitation")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
        user_email = payload.get("email")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        # Find the invitation
        result = await session.execute(
            select(TeamInvitationModel)
            .where(TeamInvitationModel.token == token)
        )
        invitation = result.scalar_one_or_none()

        if not invitation:
            raise HTTPException(404, "Invitation not found")

        if invitation.status != "pending":
            raise HTTPException(400, f"Invitation has been {invitation.status}")

        if datetime.utcnow() > invitation.expires_at:
            invitation.status = "expired"
            await session.flush()
            raise HTTPException(400, "Invitation has expired")

        # Verify email matches (optional - can be removed to allow any logged-in user)
        if invitation.email.lower() != user_email.lower():
            raise HTTPException(403, "This invitation was sent to a different email address")

        # Check if already a member
        result = await session.execute(
            select(TeamMembershipModel)
            .where(TeamMembershipModel.team_id == invitation.team_id)
            .where(TeamMembershipModel.user_id == user_id)
        )
        if result.scalar_one_or_none():
            invitation.status = "accepted"
            await session.flush()
            return {"success": True, "message": "You are already a member of this team"}

        # Create membership
        membership = TeamMembershipModel(
            id=generate_id(),
            team_id=invitation.team_id,
            user_id=user_id,
            role=invitation.role,
        )
        session.add(membership)

        # Update invitation status
        invitation.status = "accepted"
        invitation.accepted_at = datetime.utcnow()
        await session.flush()

        # Log activity
        await log_activity(
            team_id=invitation.team_id,
            user_id=user_id,
            action="accept",
            resource_type="invitation",
            resource_id=invitation.id,
            details={"role": invitation.role}
        )

        return {
            "success": True,
            "team_id": invitation.team_id,
            "role": invitation.role,
            "message": "Successfully joined the team",
        }


@app.delete("/api/teams/{team_id}/invitations/{invitation_id}")
async def cancel_invitation(
    team_id: str,
    invitation_id: str,
):
    """Cancel a pending invitation"""
    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(TeamInvitationModel)
            .where(TeamInvitationModel.id == invitation_id)
            .where(TeamInvitationModel.team_id == team_id)
        )
        invitation = result.scalar_one_or_none()

        if not invitation:
            raise HTTPException(404, "Invitation not found")

        if invitation.status != "pending":
            raise HTTPException(400, "Can only cancel pending invitations")

        invitation.status = "cancelled"
        await session.flush()

        return {"success": True, "message": "Invitation cancelled"}


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


# ============== SESSION MANAGEMENT ==============


def get_device_info(request: Request) -> str:
    """Extract device info from request headers"""
    user_agent = request.headers.get("user-agent", "Unknown")
    # Simplify user agent for display
    if "Mobile" in user_agent:
        if "iPhone" in user_agent:
            return "iPhone"
        elif "Android" in user_agent:
            return "Android"
        else:
            return "Mobile Device"
    elif "Chrome" in user_agent:
        return "Chrome Browser"
    elif "Firefox" in user_agent:
        return "Firefox Browser"
    elif "Safari" in user_agent:
        return "Safari Browser"
    elif "Edge" in user_agent:
        return "Edge Browser"
    else:
        return user_agent[:50] if user_agent else "Unknown"


def get_client_ip(request: Request) -> str:
    """Get client IP from request, handling proxies"""
    # Check X-Forwarded-For header first (for reverse proxies)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # Check X-Real-IP header
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    # Fall back to direct client
    return request.client.host if request.client else "Unknown"


async def create_session(
    user_id: str,
    token: str,
    request: Request,
    expires_at: datetime
) -> Optional[str]:
    """Create a new session record for a user"""
    async with get_db_session() as session:
        if session is None:
            return None

        # Hash the token for storage
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        session_record = UserSessionModel(
            id=generate_id(),
            user_id=user_id,
            token_hash=token_hash,
            device_info=get_device_info(request),
            ip_address=get_client_ip(request),
            is_current=True,
            last_active=datetime.utcnow(),
            expires_at=expires_at,
        )
        session.add(session_record)
        await session.flush()
        return session_record.id


async def update_session_activity(token: str) -> None:
    """Update the last_active timestamp for a session"""
    async with get_db_session() as session:
        if session is None:
            return

        from sqlalchemy import select, update

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        await session.execute(
            update(UserSessionModel)
            .where(UserSessionModel.token_hash == token_hash)
            .values(last_active=datetime.utcnow())
        )


@app.get("/api/sessions")
async def list_sessions(
    request: Request,
    authorization: str = Header(None),
):
    """List all active sessions for the current user"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    # Get current token hash to mark current session
    current_token_hash = hashlib.sha256(parts[1].encode()).hexdigest()

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserSessionModel)
            .where(UserSessionModel.user_id == user_id)
            .where(UserSessionModel.revoked_at == None)
            .where(UserSessionModel.expires_at > datetime.utcnow())
            .order_by(UserSessionModel.last_active.desc())
        )
        sessions = result.scalars().all()

        return {
            "sessions": [
                {
                    **s.to_dict(),
                    "is_current": s.token_hash == current_token_hash,
                }
                for s in sessions
            ]
        }


@app.delete("/api/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    authorization: str = Header(None),
):
    """Revoke a specific session"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select

        result = await session.execute(
            select(UserSessionModel)
            .where(UserSessionModel.id == session_id)
            .where(UserSessionModel.user_id == user_id)
        )
        session_record = result.scalar_one_or_none()

        if not session_record:
            raise HTTPException(404, "Session not found")

        if session_record.revoked_at:
            raise HTTPException(400, "Session already revoked")

        session_record.revoked_at = datetime.utcnow()
        await session.flush()

        return {"success": True, "message": "Session revoked"}


@app.post("/api/sessions/revoke-all")
async def revoke_all_sessions(
    request: Request,
    keep_current: bool = Form(True),
    authorization: str = Header(None),
):
    """Revoke all sessions for the current user (optionally keep current)"""
    if not authorization:
        raise HTTPException(401, "Authorization required")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization format")

    try:
        payload = verify_jwt_token(parts[1])
        user_id = payload.get("sub")
    except ValueError as e:
        raise HTTPException(401, str(e))

    current_token_hash = hashlib.sha256(parts[1].encode()).hexdigest()

    async with get_db_session() as session:
        if session is None:
            raise HTTPException(500, "Database not available")

        from sqlalchemy import select, update

        # Build the update query
        query = (
            update(UserSessionModel)
            .where(UserSessionModel.user_id == user_id)
            .where(UserSessionModel.revoked_at == None)
            .values(revoked_at=datetime.utcnow())
        )

        # Optionally exclude current session
        if keep_current:
            query = query.where(UserSessionModel.token_hash != current_token_hash)

        result = await session.execute(query)
        revoked_count = result.rowcount

        return {
            "success": True,
            "revoked_count": revoked_count,
            "message": f"Revoked {revoked_count} session(s)",
        }


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
