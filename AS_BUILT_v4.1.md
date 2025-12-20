# PropelAI v4.1 As-Built Technical Document

**Version:** 4.1.0
**Date:** December 2024
**Classification:** Technical Architecture Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [Agent Architecture](#5-agent-architecture)
6. [Extraction Pipeline](#6-extraction-pipeline)
7. [API Reference](#7-api-reference)
8. [Data Models](#8-data-models)
9. [Database Schema](#9-database-schema)
10. [Frontend Architecture](#10-frontend-architecture)
11. [Deployment Configuration](#11-deployment-configuration)
12. [Security & Governance](#12-security--governance)
13. [Version History](#13-version-history)
14. [Team Workspaces (v4.1)](#14-team-workspaces-v41)

---

## 1. Executive Summary

### 1.1 Purpose

PropelAI is an AI-powered federal proposal automation platform that extracts requirements from RFP documents, generates compliance matrices, and assists in proposal development. The system achieves **95%+ requirement extraction accuracy** through multi-pattern semantic analysis.

### 1.2 Key Capabilities

| Capability | Description | Version |
|------------|-------------|---------|
| Requirement Extraction | Multi-pattern extraction from PDF/DOCX/XLSX | v1.0+ |
| Compliance Matrix Generation | Automated CTM with 19+ columns | v2.8+ |
| Section-Aware Processing | Section L/M/C boundary detection | v2.9+ |
| Resilient Extraction | Extract-first architecture with validation | v3.0+ |
| Trust Gate | PDF coordinate extraction for source verification | v4.0 |
| Strategy Generation | Win themes and discriminators | v4.0 |
| Persistent Storage | PostgreSQL + Render Disk | v4.1 |

### 1.3 Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Database | PostgreSQL 16+, SQLAlchemy 2.0, asyncpg |
| Document Processing | pypdf, pdfplumber, python-docx, openpyxl |
| Orchestration | LangGraph (optional) |
| Frontend | React 18 (CDN), Lucide Icons |
| File Storage | Render Disk (/data) or ephemeral temp |
| Deployment | Render.com (Web Service + PostgreSQL + Disk) |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    React SPA (web/index.html)                   ││
│  │  - Document Upload (drag & drop, guided classification)        ││
│  │  - Requirements Table (search, filter, sort)                   ││
│  │  - Proposal Outline Viewer                                      ││
│  │  - Export Controls (Excel, Word)                                ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP/REST
┌─────────────────────────────────────────────────────────────────────┐
│                           API LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                 FastAPI Application (api/main.py)               ││
│  │  - 40+ REST Endpoints                                           ││
│  │  - Hybrid Store (In-Memory + PostgreSQL)                        ││
│  │  - Background Task Processing                                   ││
│  │  - CORS, Error Handling, Logging                                ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
│   EXTRACTION LAYER    │ │  STORAGE LAYER  │ │    AGENT LAYER       │
│                       │ │                 │ │                      │
│ EnhancedCompliance    │ │ PostgreSQL      │ │ StrategyAgent        │
│   Agent               │ │  - RFPs         │ │ DraftingAgent        │
│                       │ │  - Requirements │ │ RedTeamAgent         │
│ Extractors:           │ │  - Amendments   │ │                      │
│  - Resilient (v3.0)   │ │                 │ │ LangGraph Workflow   │
│  - BestPractices(v2.9)│ │ Render Disk     │ │  (optional)          │
│  - Semantic (v2.8)    │ │  - /data/uploads│ │                      │
│  - Legacy (v1.0)      │ │  - /data/outputs│ │                      │
└───────────────────────┘ └─────────────────┘ └──────────────────────┘
```

### 2.2 Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  RFP Upload  │────▶│Bundle Detect │────▶│Multi-Format  │
│  (PDF/DOCX)  │     │              │     │   Parser     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Compliance  │◀────│Cross-Ref     │◀────│ Requirement  │
│   Matrix     │     │  Resolver    │     │  Extractor   │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Outline    │────▶│   Export     │────▶│  PostgreSQL  │
│  Generator   │     │ (Excel/Word) │     │   + Disk     │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 2.3 Storage Architecture (v4.1)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HYBRID STORAGE MODEL                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐          ┌─────────────────────┐           │
│  │   IN-MEMORY CACHE   │◀────────▶│     POSTGRESQL      │           │
│  │                     │  async   │                     │           │
│  │  - Fast read/write  │   sync   │  - RFPModel         │           │
│  │  - Processing state │          │  - RequirementModel │           │
│  │  - Transient data   │          │  - AmendmentModel   │           │
│  └─────────────────────┘          └─────────────────────┘           │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      RENDER DISK (/data)                        ││
│  │  /data/uploads/{rfp_id}/     - Uploaded RFP documents           ││
│  │  /data/outputs/               - Generated exports               ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
│  Fallback: tempfile.gettempdir() if /data not writable              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
propelai01/
├── agents/                              # AI Agent Modules
│   ├── enhanced_compliance/             # Core Extraction Engine (29 modules)
│   │   ├── __init__.py                  # Module exports
│   │   ├── agent.py                     # EnhancedComplianceAgent orchestrator
│   │   ├── models.py                    # Core data models
│   │   ├── extractor.py                 # RequirementExtractor
│   │   ├── parser.py                    # MultiFormatParser (PDF/DOCX/XLSX)
│   │   ├── bundle_detector.py           # Document classification
│   │   ├── resolver.py                  # CrossReferenceResolver
│   │   ├── amendment_processor.py       # Amendment tracking
│   │   ├── outline_generator.py         # Basic outline generation
│   │   ├── smart_outline_generator.py   # LLM-enhanced outlines
│   │   ├── excel_export.py              # CTM Excel export
│   │   ├── excel_parser.py              # Parse existing CTMs
│   │   ├── semantic_extractor.py        # v2.8 Semantic extraction
│   │   ├── semantic_ctm_export.py       # 19-column CTM
│   │   ├── section_aware_extractor.py   # v2.9 Section-aware
│   │   ├── best_practices_ctm.py        # v2.9 Best practices CTM
│   │   ├── universal_extractor.py       # v3.0 Universal patterns
│   │   ├── resilient_extractor.py       # v3.0 Extract-first
│   │   ├── extraction_models.py         # v3.0 Result models
│   │   ├── extraction_validator.py      # v3.0 Validation
│   │   ├── pdf_coordinate_extractor.py  # v4.0 Trust Gate
│   │   ├── annotated_outline_exporter.py# v4.0 Word export
│   │   ├── document_structure.py        # RFP structure analysis
│   │   ├── document_types.py            # Upload slot definitions
│   │   ├── section_classifier.py        # Section classification
│   │   ├── ctm_data_models.py           # CTM enums
│   │   ├── ctm_extractor.py             # CTM extraction
│   │   ├── ctm_integration.py           # CTM integration
│   │   └── company_library.py           # Company knowledge base
│   │
│   ├── strategy_agent.py                # Win themes & discriminators
│   ├── drafting_agent.py                # Proposal writing
│   ├── drafting_workflow.py             # LangGraph workflow
│   ├── red_team_agent.py                # Evaluation simulation
│   ├── compliance_agent.py              # Legacy v1.0 agent
│   └── __init__.py
│
├── api/                                 # FastAPI Backend
│   ├── __init__.py
│   ├── main.py                          # 40+ REST endpoints
│   └── database.py                      # v4.1 PostgreSQL ORM
│
├── core/                                # Orchestration Layer
│   ├── __init__.py
│   ├── orchestrator.py                  # LangGraph supervisor
│   ├── state.py                         # ProposalState schema
│   └── config.py                        # Configuration management
│
├── web/                                 # Frontend
│   └── index.html                       # React SPA (CDN-based)
│
├── validation/                          # Quality Assurance
│   ├── schemas.py                       # Annotation schemas
│   ├── metrics.py                       # Accuracy metrics
│   ├── matching.py                      # Requirement matching
│   ├── annotation_tool/                 # Annotation UI
│   └── ground_truth/                    # Test datasets
│
├── tests/                               # Test Suite
│   ├── test_agents.py
│   ├── test_accuracy.py
│   ├── test_resilient_extraction.py
│   └── test_structure_determinism.py
│
├── tools/                               # Utilities
│   └── document_tools.py
│
├── scripts/                             # Automation Scripts
│   └── check_accuracy_regression.py
│
├── docs/                                # Documentation
│
├── requirements.txt                     # Python dependencies
├── requirements-prod.txt                # Production dependencies
├── package.json                         # Node.js (docx)
├── render.yaml                          # Render deployment
├── docker-compose.yml                   # Docker setup
├── init.sql                             # Database schema
├── Dockerfile                           # Container build
├── Procfile                             # Heroku deployment
├── .env.example                         # Environment template
├── README.md                            # Getting started
├── HANDOFF_DOCUMENT.md                  # Architecture guide
└── AS_BUILT_v4.1.md                     # This document
```

---

## 4. Core Components

### 4.1 RFPStore (Hybrid Storage)

**Location:** `api/main.py` lines 251-420

The RFPStore implements a hybrid storage pattern combining in-memory caching with asynchronous PostgreSQL persistence.

```python
class RFPStore:
    """
    Hybrid store for RFP data (v4.1).

    - Processing status: in-memory only (transient)
    - RFP data: in-memory + async sync to database
    """

    def __init__(self):
        self.rfps: Dict[str, Dict] = {}           # In-memory cache
        self.processing_status: Dict[str, ProcessingStatus] = {}
        self._db_available = False
        self._db_initialized = False

    async def init_database(self):
        """Initialize database and load existing RFPs on startup"""

    def create(self, rfp_id: str, data: Dict) -> Dict:
        """Create RFP in memory, schedule async DB sync"""

    def update(self, rfp_id: str, updates: Dict) -> Dict:
        """Update RFP in memory, schedule async DB sync"""

    def _schedule_db_sync(self, rfp_id: str):
        """Non-blocking async database synchronization"""
```

**Key Features:**
- Fast in-memory reads during processing
- Async background sync to PostgreSQL
- Automatic data loading on startup
- Graceful fallback if database unavailable

### 4.2 MultiFormatParser

**Location:** `agents/enhanced_compliance/parser.py`

Handles document parsing across multiple formats with quality assessment.

```python
class MultiFormatParser:
    """Parse PDF, DOCX, XLSX documents with section detection"""

    SUPPORTED_FORMATS = ['.pdf', '.docx', '.xlsx', '.doc', '.xls']

    def parse_file(self, filepath: str, doc_type: DocumentType) -> ParsedDocument:
        """
        Parse document and extract:
        - Full text content
        - Page-by-page breakdown
        - Section boundaries
        - Tables (as structured data)
        - Metadata (title, author, dates)
        """

    def _parse_pdf(self, filepath, filename, doc_type) -> ParsedDocument:
        """PDF parsing using pdfplumber + pypdf"""

    def _parse_docx(self, filepath, filename, doc_type) -> ParsedDocument:
        """DOCX parsing using python-docx"""

    def _parse_xlsx(self, filepath, filename, doc_type) -> ParsedDocument:
        """Excel parsing using openpyxl"""
```

**PDF Extraction Quality Assessment:**
```python
def _assess_extraction_quality(self, text: str, page_count: int) -> float:
    """
    Score extraction quality 0.0-1.0 based on:
    - Characters per page (expected: 2000-4000)
    - Word density
    - Presence of structured content
    - OCR artifacts detection
    """
```

### 4.3 RequirementExtractor

**Location:** `agents/enhanced_compliance/extractor.py`

Multi-pattern extraction with semantic classification.

```python
class RequirementExtractor:
    """Extract requirements using pattern-based analysis"""

    MANDATORY_PATTERNS = [
        r'\bshall\b', r'\bmust\b', r'\brequired\b', r'\bmandatory\b',
        r'\bwill\b(?!\s+be\s+evaluated)', r'\bneeds?\s+to\b'
    ]

    CONDITIONAL_PATTERNS = [
        r'\bshould\b', r'\bmay\b', r'\bcan\b', r'\brecommended\b'
    ]

    PROHIBITION_PATTERNS = [
        r'\bshall\s+not\b', r'\bmust\s+not\b', r'\bprohibited\b',
        r'\bforbidden\b', r'\bnot\s+permitted\b'
    ]

    NOISE_PATTERNS = [
        r'^table\s+of\s+contents',
        r'^\d+\.\s*$',  # Section numbers only
        r'^page\s+\d+',
        r'^attachment\s+\d+\s*$'
    ]

    MIN_REQUIREMENT_LENGTH = 100  # characters
    MIN_WORD_COUNT = 15

    def extract_from_document(self, doc: ParsedDocument) -> List[RequirementNode]:
        """Extract all requirements with confidence scoring"""
```

### 4.4 CrossReferenceResolver

**Location:** `agents/enhanced_compliance/resolver.py`

Builds requirement graph with cross-document linking.

```python
class CrossReferenceResolver:
    """Resolve cross-references between requirements"""

    def resolve_references(
        self,
        requirements: List[RequirementNode],
        documents: List[ParsedDocument]
    ) -> Dict[str, List[str]]:
        """
        Build adjacency list mapping requirement IDs to related requirements.

        Links:
        - Section L instructions → Section C requirements
        - Section M criteria → Section C requirements
        - CDRL references → SOW tasks
        - Amendment changes → Original requirements
        """

    def find_gaps(self, graph: Dict) -> List[str]:
        """Identify requirements with no cross-references (potential gaps)"""
```

---

## 5. Agent Architecture

### 5.1 EnhancedComplianceAgent

**Location:** `agents/enhanced_compliance/agent.py`

The primary extraction orchestrator achieving 95%+ accuracy.

```python
class EnhancedComplianceAgent:
    """
    Main RFP processing agent.

    Pipeline:
    1. Bundle Detection - Classify document types
    2. Multi-Format Parsing - Extract text from all formats
    3. Requirement Extraction - Pattern-based with semantic classification
    4. Cross-Reference Resolution - Build requirements graph
    5. Compliance Matrix Generation - Create CTM structure
    """

    def __init__(self, llm_client=None, use_llm_enhancement=False):
        self.bundle_detector = BundleDetector()
        self.parser = MultiFormatParser()
        self.extractor = RequirementExtractor()
        self.resolver = CrossReferenceResolver()

    def process_files(self, file_paths: List[str]) -> ProcessingResult:
        """Process RFP document bundle"""

    def process_folder(self, folder_path: str) -> ProcessingResult:
        """Process all documents in folder"""
```

**Processing Result:**
```python
@dataclass
class ProcessingResult:
    requirements_graph: Dict[str, RequirementNode]
    compliance_matrix: List[ComplianceRow]
    stats: Dict[str, Any]
    duration_seconds: float
    warnings: List[str]
```

### 5.2 StrategyAgent

**Location:** `agents/strategy_agent.py`

Generates win themes and competitive positioning.

```python
class StrategyAgent:
    """Generate win strategy from RFP analysis"""

    def analyze_evaluation_factors(
        self,
        section_m: List[Dict]
    ) -> List[EvaluationFactor]:
        """Parse Section M evaluation criteria with weights"""

    def generate_win_themes(
        self,
        requirements: List[Dict],
        evaluation_factors: List[EvaluationFactor],
        company_capabilities: Dict
    ) -> List[WinTheme]:
        """Generate 3-5 win themes aligned to evaluation criteria"""

    def create_discriminators(
        self,
        win_themes: List[WinTheme],
        competitor_intel: Dict
    ) -> List[Discriminator]:
        """Create unique differentiators with proof points"""

@dataclass
class WinTheme:
    theme: str
    evaluation_factor: str
    proof_points: List[str]
    discriminators: List[str]

@dataclass
class Discriminator:
    type: str  # technical, management, past_performance, price, innovation
    statement: str
    evidence: List[str]
    competitors_lacking: List[str]
```

### 5.3 SmartOutlineGenerator

**Location:** `agents/enhanced_compliance/smart_outline_generator.py`

Generates proposal outlines from compliance matrix data.

```python
class SmartOutlineGenerator:
    """Generate proposal outline from Section L/M analysis"""

    # RFP types that should NOT use UCF defaults
    NO_UCF_TYPES = {'TASK_ORDER', 'RFE', 'RFQ', 'BPA', 'GSA_SCHEDULE'}

    def generate_from_compliance_matrix(
        self,
        section_l_requirements: List[Dict],
        section_m_requirements: List[Dict],
        technical_requirements: List[Dict],
        stats: Dict
    ) -> ProposalOutline:
        """
        Generate outline structure:
        1. Detect RFP format (UCF vs non-standard)
        2. Extract volumes from Section L
        3. Map evaluation factors from Section M
        4. Apply page limits
        5. Populate sections with requirements
        """

    def to_json(self, outline: ProposalOutline) -> Dict:
        """Convert outline to JSON for API/export"""

@dataclass
class ProposalOutline:
    volumes: List[ProposalVolume]
    eval_factors: List[EvaluationFactor]
    format_requirements: FormatRequirements
    submission: SubmissionInfo
    total_pages: Optional[int]
    warnings: List[str]
```

### 5.4 PDF Coordinate Extractor (Trust Gate)

**Location:** `agents/enhanced_compliance/pdf_coordinate_extractor.py`

Enables one-click source verification by extracting exact PDF locations.

```python
class PDFCoordinateExtractor:
    """Extract PDF coordinates for requirement highlighting (v4.0 Trust Gate)"""

    def find_requirement_location(
        self,
        requirement_text: str,
        pdf_path: str
    ) -> Optional[SourceCoordinate]:
        """
        Find exact location of requirement in PDF.
        Returns bounding box coordinates for highlighting.
        """

    def get_page_image(
        self,
        pdf_path: str,
        page_num: int,
        highlight_box: Optional[BoundingBox] = None
    ) -> bytes:
        """Render PDF page as image with optional highlighting"""

@dataclass
class SourceCoordinate:
    document_path: str
    page_number: int
    bounding_box: BoundingBox
    confidence: float
    extraction_method: str  # 'pdfplumber', 'pypdf', 'ocr'

@dataclass
class BoundingBox:
    x0: float  # Left (PDF points)
    y0: float  # Top
    x1: float  # Right
    y1: float  # Bottom
    page_width: float
    page_height: float

    def to_css_percent(self) -> Dict[str, str]:
        """Convert to CSS percentage for web overlay"""
```

---

## 6. Extraction Pipeline

### 6.1 Pipeline Versions

| Version | Class | Description | Accuracy |
|---------|-------|-------------|----------|
| v1.0 | `EnhancedComplianceAgent` | Basic pattern matching | ~40% |
| v2.8 | `SemanticRequirementExtractor` | Semantic classification | ~75% |
| v2.9 | `SectionAwareExtractor` | Section boundary detection | ~85% |
| v3.0 | `ResilientExtractor` | Extract-first architecture | ~90% |
| v4.0+ | Combined with Trust Gate | PDF coordinates | ~95% |

### 6.2 Resilient Extraction Pipeline (v3.0)

**Location:** `agents/enhanced_compliance/resilient_extractor.py`

```python
class ResilientExtractionPipeline:
    """
    Extract-First Architecture:
    1. Extract ALL potential requirements (high recall)
    2. Classify and score each candidate
    3. Filter by confidence threshold
    4. Validate against reproducibility tests
    """

    def __init__(self, enable_coordinates=True):
        self.parser = MultiFormatParser()
        self.universal_extractor = UniversalExtractor()
        self.section_classifier = SectionClassifier()
        self.coordinate_extractor = PDFCoordinateExtractor()

    def extract(self, file_paths: List[str]) -> ExtractionResult:
        """Full extraction pipeline"""

    def extract_from_parsed(
        self,
        documents: List[Dict],
        rfp_id: str
    ) -> ExtractionResult:
        """Extract from pre-parsed documents"""

@dataclass
class ExtractionResult:
    requirements: List[RequirementCandidate]
    quality_metrics: QualityMetrics
    review_queue: List[RequirementCandidate]  # Low-confidence items
    document_hashes: List[str]  # For reproducibility

@dataclass
class QualityMetrics:
    total_documents: int
    total_pages: int
    total_requirements: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    uncertain_count: int
    sow_detected: bool
    sow_source: Optional[str]
    section_counts: Dict[str, int]
    anomalies: List[str]
    warnings: List[str]
```

### 6.3 Universal Extractor

**Location:** `agents/enhanced_compliance/universal_extractor.py`

```python
class UniversalExtractor:
    """Pattern-agnostic requirement extraction"""

    BINDING_PATTERNS = {
        'SHALL': [r'\bshall\b', r'\bmust\b', r'\brequired\b'],
        'SHOULD': [r'\bshould\b', r'\brecommended\b'],
        'MAY': [r'\bmay\b', r'\bcan\b', r'\boptional\b'],
        'WILL': [r'\bwill\b(?!\s+be\s+evaluated)']
    }

    def extract_all(self, documents: List[Dict]) -> List[RequirementCandidate]:
        """
        Extract all potential requirements:
        1. Split into paragraphs
        2. Check for binding language
        3. Filter noise patterns
        4. Score confidence
        5. Deduplicate
        """
        self.seen_hashes = set()  # Reset per extraction
        self.req_counter = 0

        for doc in documents:
            # Process each page
            for page_num, page_text in enumerate(doc['pages']):
                candidates = self._extract_from_page(page_text, ...)
```

### 6.4 Section Classifier

**Location:** `agents/enhanced_compliance/section_classifier.py`

```python
class SectionClassifier:
    """Classify requirements by RFP section"""

    SECTION_INDICATORS = {
        'SECTION_L': ['instructions to offerors', 'proposal preparation'],
        'SECTION_M': ['evaluation factors', 'evaluation criteria'],
        'SECTION_C': ['statement of work', 'performance work statement'],
        'SOW': ['scope of work', 'task order', 'deliverables'],
    }

    def detect_sow_documents(
        self,
        documents: List[Dict]
    ) -> Dict[str, Any]:
        """Identify which documents contain SOW content"""

    def classify_requirements(
        self,
        candidates: List[RequirementCandidate],
        documents: List[Dict]
    ) -> List[RequirementCandidate]:
        """Assign section classification to each requirement"""
```

---

## 7. API Reference

### 7.1 Endpoint Summary

**Base URL:** `/api`

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Health | 1 | System health and component status |
| RFP Management | 4 | Create, read, update, delete RFPs |
| Upload | 4 | Document upload with classification |
| Processing | 4 | Extraction pipelines (v1-v3) |
| Requirements | 4 | Query and analyze requirements |
| Export | 3 | Excel/Word export |
| Outline | 3 | Proposal outline generation |
| Strategy | 3 | Win themes and analysis |
| Amendments | 3 | Amendment tracking |
| Library | 9 | Company knowledge base |

### 7.2 Core Endpoints

#### Health Check
```
GET /api/health

Response:
{
  "status": "healthy",
  "version": "4.1.0",
  "timestamp": "2024-12-20T...",
  "storage": {
    "type": "persistent",          // or "temporary"
    "upload_dir": "/data/uploads",
    "database": "connected",       // or "not configured"
    "rfps_loaded": 5
  },
  "components": {
    "enhanced_compliance_agent": "ready",
    "best_practices_extractor": "ready",
    "trust_gate": "ready",
    ...
  }
}
```

#### Create RFP
```
POST /api/rfp
Content-Type: application/json

{
  "name": "NIH Research Grant RFP",
  "solicitation_number": "75N96025R00004",
  "agency": "NIH",
  "due_date": "2025-03-15"
}

Response:
{
  "id": "RFP-A1B2C3D4",
  "name": "NIH Research Grant RFP",
  "solicitation_number": "75N96025R00004",
  "status": "created",
  "files": [],
  "requirements_count": 0,
  "created_at": "2024-12-20T...",
  "updated_at": "2024-12-20T..."
}
```

#### Upload Documents
```
POST /api/rfp/{rfp_id}/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx, ...]

Response:
{
  "status": "uploaded",
  "files": [
    {"name": "file1.pdf", "size": 1234567, "type": "PDF"},
    {"name": "file2.docx", "size": 234567, "type": "DOCX"}
  ],
  "total_files": 2
}
```

#### Guided Upload (with Classification)
```
POST /api/rfp/{rfp_id}/upload-guided
Content-Type: multipart/form-data

files: [sow.pdf, section_lm.pdf]
doc_types: "sow,combined_lm"

Response:
{
  "status": "uploaded",
  "files": [...],
  "document_metadata": {
    "sow.pdf": {"doc_type": "sow"},
    "section_lm.pdf": {"doc_type": "combined_lm"}
  }
}
```

#### Process RFP (Best Practices - Recommended)
```
POST /api/rfp/{rfp_id}/process-best-practices

Response:
{
  "status": "processing_started",
  "rfp_id": "RFP-A1B2C3D4",
  "files_count": 2,
  "mode": "best_practices"
}
```

#### Get Processing Status
```
GET /api/rfp/{rfp_id}/status

Response:
{
  "status": "processing",      // or "completed", "error"
  "progress": 65,
  "message": "Extracting requirements by section...",
  "requirements_count": null   // Set when complete
}
```

#### Get Requirements
```
GET /api/rfp/{rfp_id}/requirements?category=L_COMPLIANCE&priority=high

Response:
{
  "requirements": [
    {
      "id": "TW-L-0001",
      "text": "The offeror shall provide...",
      "category": "L_COMPLIANCE",
      "type": "PROPOSAL_INSTRUCTION",
      "section": "L.4.2",
      "priority": "high",
      "binding_level": "SHALL",
      "confidence": 0.95,
      "source_doc": "RFP.pdf",
      "source_page": 42
    },
    ...
  ],
  "total": 156,
  "stats": {...}
}
```

#### Export Compliance Matrix
```
GET /api/rfp/{rfp_id}/export?format=xlsx

Response: Binary Excel file
Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
```

#### Get/Generate Outline
```
GET /api/rfp/{rfp_id}/outline

Response:
{
  "format": "json",
  "outline": {
    "volumes": [
      {
        "id": "vol-1",
        "name": "Technical Volume",
        "type": "technical",
        "page_limit": 50,
        "sections": [...]
      }
    ],
    "submission": {
      "due_date": "2025-03-15",
      "method": "Electronic via SAM.gov"
    },
    "format_requirements": {
      "font": "Times New Roman",
      "font_size": "12pt",
      "margins": "1 inch"
    }
  }
}
```

#### Export Outline as Word
```
GET /api/rfp/{rfp_id}/outline/export

Response: Binary DOCX file
Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document
```

---

## 8. Data Models

### 8.1 RFP Model

```python
# In-memory representation (api/main.py)
RFP = {
    "id": "RFP-A1B2C3D4",
    "name": "NIH Research Grant RFP",
    "solicitation_number": "75N96025R00004",
    "agency": "NIH",
    "due_date": "2025-03-15",
    "status": "completed",  # created, files_uploaded, processing, completed, error
    "extraction_mode": "best_practices",  # semantic, resilient, best_practices

    # Files
    "files": ["RFP.pdf", "SOW.docx"],
    "file_paths": ["/data/uploads/RFP-A1B2C3D4/RFP.pdf", ...],
    "document_metadata": {
        "RFP.pdf": {"doc_type": "combined_rfp"},
        "SOW.docx": {"doc_type": "sow"}
    },

    # Extracted data
    "requirements": [...],  # List of requirement dicts
    "stats": {
        "total": 156,
        "section_l": 45,
        "technical": 89,
        "evaluation": 22,
        "by_priority": {"high": 78, "medium": 56, "low": 22},
        "processing_time": 12.5
    },
    "outline": {...},  # Generated proposal outline

    # Amendments
    "amendments": [...],

    # Timestamps
    "created_at": "2024-12-20T10:00:00",
    "updated_at": "2024-12-20T10:15:00"
}
```

### 8.2 Requirement Model

```python
Requirement = {
    "id": "TW-L-0001",                    # Generated ID
    "rfp_reference": "L.4.2.a",           # Original RFP reference
    "text": "The offeror shall provide a detailed technical approach...",

    # Classification
    "category": "L_COMPLIANCE",            # L_COMPLIANCE, EVALUATION, TECHNICAL
    "type": "PROPOSAL_INSTRUCTION",        # Requirement type
    "section": "L",
    "subsection": "4.2.a",

    # Binding
    "priority": "high",                    # high, medium, low
    "binding_level": "SHALL",              # SHALL, SHOULD, MAY, WILL
    "binding_keyword": "shall",

    # Confidence
    "confidence": 0.95,
    "confidence_level": "HIGH",            # HIGH, MEDIUM, LOW, UNCERTAIN
    "needs_review": False,
    "review_reasons": [],

    # Source
    "source_page": 42,
    "source_doc": "RFP.pdf",
    "source_content_type": "section_l",    # section_l, section_m, technical, etc.
    "parent_title": "Technical Approach Requirements",

    # Cross-references
    "cross_references": ["C.3.1", "M.2.a"]
}
```

### 8.3 Proposal Outline Model

```python
ProposalOutline = {
    "volumes": [
        {
            "id": "vol-1",
            "name": "Technical Volume",
            "order": 1,
            "type": "technical",
            "page_limit": 50,
            "eval_factors": ["Technical Approach", "Understanding"],
            "confidence": "HIGH",
            "evidence_source": "SECTION_L",
            "sections": [
                {
                    "id": "sec-1.1",
                    "name": "Technical Approach",
                    "title": "Technical Approach",
                    "page_limit": 20,
                    "requirements": [...],
                    "eval_criteria": [...],
                    "subsections": [...]
                }
            ]
        }
    ],
    "eval_factors": [
        {
            "name": "Technical Approach",
            "weight": "Most Important",
            "description": "...",
            "color_rating": True
        }
    ],
    "format_requirements": {
        "font": "Times New Roman",
        "font_size": "12pt",
        "margins": "1 inch",
        "line_spacing": "Single"
    },
    "submission": {
        "due_date": "2025-03-15T14:00:00",
        "method": "Electronic via SAM.gov",
        "copies": 1
    },
    "total_pages": 150,
    "total_page_limit": 150,
    "rfp_format": "STANDARD_UCF",
    "warnings": []
}
```

---

## 9. Database Schema

### 9.1 PostgreSQL Models (v4.1)

**Location:** `api/database.py`

```python
class RFPModel(Base):
    __tablename__ = "rfps"

    id = Column(String(50), primary_key=True)
    name = Column(String(500), nullable=False)
    solicitation_number = Column(String(200), nullable=True)
    agency = Column(String(500), nullable=True)
    due_date = Column(String(100), nullable=True)
    status = Column(String(50), default="created")
    extraction_mode = Column(String(50), nullable=True)

    # JSON fields
    files = Column(JSONB, default=list)
    file_paths = Column(JSONB, default=list)
    document_metadata = Column(JSONB, default=dict)
    stats = Column(JSONB, nullable=True)
    outline = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    requirements = relationship("RequirementModel", back_populates="rfp",
                               cascade="all, delete-orphan")
    amendments = relationship("AmendmentModel", back_populates="rfp",
                             cascade="all, delete-orphan")


class RequirementModel(Base):
    __tablename__ = "requirements"

    id = Column(String(100), primary_key=True)
    rfp_id = Column(String(50), ForeignKey("rfps.id", ondelete="CASCADE"))

    text = Column(Text, nullable=False)
    rfp_reference = Column(String(200), nullable=True)
    category = Column(String(100), nullable=True)
    type = Column(String(100), nullable=True)
    section = Column(String(100), nullable=True)
    subsection = Column(String(200), nullable=True)
    priority = Column(String(20), default="medium")
    binding_level = Column(String(50), nullable=True)
    binding_keyword = Column(String(50), nullable=True)
    confidence = Column(Float, default=0.7)
    confidence_level = Column(String(20), nullable=True)
    needs_review = Column(Boolean, default=False)
    review_reasons = Column(JSONB, default=list)
    source_page = Column(Integer, nullable=True)
    source_doc = Column(String(500), nullable=True)
    source_content_type = Column(String(50), nullable=True)
    parent_title = Column(String(500), nullable=True)
    cross_references = Column(JSONB, default=list)

    # Indexes
    __table_args__ = (
        Index('idx_requirements_rfp_id', 'rfp_id'),
        Index('idx_requirements_category', 'category'),
    )


class AmendmentModel(Base):
    __tablename__ = "amendments"

    id = Column(String(100), primary_key=True)
    rfp_id = Column(String(50), ForeignKey("rfps.id", ondelete="CASCADE"))
    amendment_number = Column(Integer, nullable=False)
    amendment_date = Column(String(100), nullable=True)
    filename = Column(String(500), nullable=True)
    file_path = Column(String(1000), nullable=True)
    changes = Column(JSONB, default=list)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 9.2 Database Connection

```python
# Environment variable (auto-set by Render)
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Connection handling
async def init_db():
    """Create tables on startup"""
    engine = create_async_engine(ASYNC_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@asynccontextmanager
async def get_db_session():
    """Async context manager for database sessions"""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

---

## 10. Frontend Architecture

### 10.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | React 18 (via CDN, no build step) |
| Styling | Custom CSS with CSS Variables |
| Icons | Lucide Icons (SVG) |
| Fonts | DM Sans (UI), JetBrains Mono (code) |
| State | React useState/useEffect hooks |

### 10.2 Design System

**Color Palette:**
```css
:root {
    --bg-primary: #0a0a0f;      /* Main background */
    --bg-secondary: #12121a;     /* Card background */
    --bg-tertiary: #1a1a24;      /* Input background */
    --text-primary: #f0f0f5;     /* Main text */
    --text-secondary: #a0a0b0;   /* Secondary text */
    --text-muted: #6b6b7b;       /* Muted text */
    --border-color: #2a2a3a;     /* Borders */
    --accent-blue: #4f8cff;      /* Primary accent */
    --accent-green: #34d399;     /* Success */
    --accent-amber: #fbbf24;     /* Warning */
    --accent-red: #f87171;       /* Error */
    --accent-purple: #a78bfa;    /* Info */
}
```

### 10.3 Component Structure

```
App
├── Sidebar
│   ├── Logo
│   ├── API Status Indicator
│   └── Navigation
│       ├── Upload RFP (view: upload)
│       ├── Compliance Matrix (view: matrix)
│       ├── Amendments (view: amendments)
│       ├── Proposal Outline (view: outline)
│       └── Company Library (view: library)
│
├── MainContent
│   ├── UploadView
│   │   ├── FileDropzone
│   │   ├── GuidedUpload (document classification)
│   │   ├── ExtractionModeSelector
│   │   └── ProcessButton
│   │
│   ├── MatrixView
│   │   ├── StatsGrid (requirement counts by category)
│   │   ├── SearchBar
│   │   ├── FilterControls
│   │   ├── RequirementsTable
│   │   │   ├── TableHeader (sortable columns)
│   │   │   └── TableRows (expandable details)
│   │   └── ExportButton
│   │
│   ├── OutlineView
│   │   ├── SubmissionInfo (due date, method)
│   │   ├── FormatRequirements
│   │   ├── VolumeNavigator
│   │   ├── SectionDetails
│   │   └── ExportOutlineButton
│   │
│   ├── AmendmentsView
│   │   ├── AmendmentUploader
│   │   └── AmendmentList
│   │
│   └── LibraryView
│       ├── ProfileCard
│       ├── DocumentUploader
│       └── DocumentList
│
└── ProcessingOverlay (modal during extraction)
    ├── ProgressBar
    └── StatusMessage
```

### 10.4 API Integration

```javascript
const api = {
    async createRFP(data) {
        const res = await fetch(`${API_BASE}/rfp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return res.json();
    },

    async uploadFiles(rfpId, files) {
        const formData = new FormData();
        files.forEach(f => formData.append('files', f));
        const res = await fetch(`${API_BASE}/rfp/${rfpId}/upload`, {
            method: 'POST',
            body: formData
        });
        return res.json();
    },

    async processRFPBestPractices(rfpId) {
        const res = await fetch(`${API_BASE}/rfp/${rfpId}/process-best-practices`,
            { method: 'POST' });
        return res.json();
    },

    async getStatus(rfpId) {
        const res = await fetch(`${API_BASE}/rfp/${rfpId}/status`);
        return res.json();
    },

    // ... additional methods
};
```

---

## 11. Deployment Configuration

### 11.1 Render.com Setup

**Services Required:**
1. **Web Service** - Python application
2. **PostgreSQL Database** - Persistent data storage
3. **Disk** - File storage at `/data`

**render.yaml:**
```yaml
services:
  - type: web
    name: propelai
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt && npm install
    startCommand: uvicorn api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
    disk:
      name: propelai-data
      mountPath: /data
      sizeGB: 1

databases:
  - name: propelai-db
    plan: starter
    databaseName: propelai
```

### 11.2 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Auto (Render) |
| `PORT` | Server port | Auto (Render) |
| `GOOGLE_API_KEY` | Gemini API key | Optional |
| `ANTHROPIC_API_KEY` | Claude API key | Optional |
| `OPENAI_API_KEY` | OpenAI API key | Optional |

### 11.3 Docker Deployment

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://propelai:password@postgres:5432/propelai
    volumes:
      - ./data:/data
    depends_on:
      - postgres
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: propelai
      POSTGRES_PASSWORD: password
      POSTGRES_DB: propelai
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### 11.4 Startup Sequence

```
1. Render starts container
2. uvicorn loads api.main:app
3. FastAPI triggers startup_event()
4. init_database() called:
   a. Check DATABASE_URL exists
   b. Create async engine
   c. Run Base.metadata.create_all()
   d. Load existing RFPs into memory
5. Application ready to serve requests
```

**Startup Logs:**
```
[Startup] PropelAI v4.1 starting...
[Storage] Using persistent disk at /data
[Startup] Upload directory: /data/uploads
[Startup] Database available: True
[DB] Database initialized successfully
[Store] Database initialized, loaded 5 RFPs
[Startup] Ready to serve requests
INFO:     Uvicorn running on http://0.0.0.0:10000
```

---

## 12. Security & Governance

### 12.1 Security Measures

| Area | Implementation |
|------|----------------|
| CORS | Configured for allowed origins |
| Input Validation | Pydantic models for all requests |
| File Upload | Extension whitelist (pdf, docx, xlsx) |
| SQL Injection | SQLAlchemy ORM with parameterized queries |
| XSS | Content-Type headers, no user HTML rendering |
| Rate Limiting | Configurable (not enforced in current version) |

### 12.2 Data Handling

- **File Storage:** Uploaded files stored in `/data/uploads/{rfp_id}/`
- **Database:** PostgreSQL with connection pooling
- **Memory:** In-memory cache cleared on restart (except DB-persisted data)
- **Logs:** Structured logging to stdout

### 12.3 Audit Trail

```python
# Agent trace logging
LogEntry = {
    "timestamp": "2024-12-20T10:15:00",
    "agent_name": "compliance_agent",
    "action": "extract_requirements",
    "input_summary": "Processed 3 documents, 228 pages",
    "output_summary": "Extracted 156 requirements",
    "duration_ms": 12500,
    "token_count": 0  # Non-LLM extraction
}
```

---

## 13. Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0 | 2024 | Initial release: basic extraction, single document |
| v2.0 | 2024 | Multi-document bundle handling |
| v2.8 | 2024 | Semantic classification, 19-column CTM |
| v2.9 | 2024 | Section-aware extraction, best practices |
| v3.0 | 2024 | Extract-first architecture, validation framework |
| v3.1 | 2024 | Guided upload, document classification |
| v3.2 | 2024 | Improved outline generation |
| v4.0 | 2024 | Trust Gate (PDF coordinates), strategy agent, drafting workflow |
| v4.1 | 2024-12 | **Persistent storage: PostgreSQL + Render Disk** |

### v4.1 Changes (Current)

1. **PostgreSQL Integration**
   - SQLAlchemy 2.0 async ORM
   - RFPModel, RequirementModel, AmendmentModel
   - Automatic table creation on startup

2. **Hybrid Storage Model**
   - In-memory cache for fast processing
   - Async background sync to database
   - Data persistence across restarts

3. **Render Disk Support**
   - File uploads at `/data/uploads`
   - Automatic fallback to temp directory

4. **Outline Cache Fix**
   - Clear cached outline when reprocessing
   - Prevents stale data contamination

5. **Enhanced Health Check**
   - Storage type reporting
   - Database connection status
   - Loaded RFP count

6. **Team Workspaces & RBAC** (v4.1.1)
   - User authentication (register/login)
   - Team creation and management
   - Role-based access control (admin, contributor, viewer)
   - Activity logging for audit trail
   - Vector search UI with semantic filtering

---

## 14. Team Workspaces (v4.1)

### 14.1 Overview

Team Workspaces enable multi-user collaboration on Company Library content with role-based access control.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TEAM WORKSPACE MODEL                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐│
│  │      USER       │────▶│   MEMBERSHIP    │◀────│      TEAM       ││
│  │                 │     │                 │     │                 ││
│  │  - id           │     │  - user_id      │     │  - id           ││
│  │  - email        │     │  - team_id      │     │  - name         ││
│  │  - name         │     │  - role         │     │  - slug         ││
│  │  - password_hash│     │  - joined_at    │     │  - settings     ││
│  └─────────────────┘     └─────────────────┘     └─────────────────┘│
│                                 │                                    │
│                                 ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                         ROLES                                    ││
│  │  admin       - Full access: manage team, members, all content   ││
│  │  contributor - Can add and edit library content                 ││
│  │  viewer      - Read-only access to library content              ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Database Models

**Location:** `api/database.py`

```python
class UserRole(str, enum.Enum):
    ADMIN = "admin"
    CONTRIBUTOR = "contributor"
    VIEWER = "viewer"


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(50), primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    memberships = relationship("TeamMembershipModel", back_populates="user")


class TeamModel(Base):
    __tablename__ = "teams"

    id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    settings = Column(JSONB, default=dict)
    created_by = Column(String(50), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    memberships = relationship("TeamMembershipModel", back_populates="team")
    activity_logs = relationship("ActivityLogModel", back_populates="team")


class TeamMembershipModel(Base):
    __tablename__ = "team_memberships"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id", ondelete="CASCADE"))
    user_id = Column(String(50), ForeignKey("users.id", ondelete="CASCADE"))
    role = Column(String(20), nullable=False, default="viewer")
    invited_by = Column(String(50), ForeignKey("users.id"), nullable=True)
    joined_at = Column(DateTime, default=datetime.utcnow)


class ActivityLogModel(Base):
    __tablename__ = "activity_log"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id", ondelete="CASCADE"))
    user_id = Column(String(50), ForeignKey("users.id", ondelete="SET NULL"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(50), nullable=True)
    details = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 14.3 API Endpoints

#### Authentication

```
POST /api/auth/register
Content-Type: application/x-www-form-urlencoded

email=user@example.com&name=John+Doe&password=secret123

Response:
{
  "id": "USR-A1B2C3D4",
  "email": "user@example.com",
  "name": "John Doe",
  "is_active": true,
  "created_at": "2024-12-20T10:00:00"
}
```

```
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded

email=user@example.com&password=secret123

Response:
{
  "user": {
    "id": "USR-A1B2C3D4",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "message": "Login successful"
}
```

#### Team Management

```
POST /api/teams
Content-Type: application/x-www-form-urlencoded

name=Proposal+Team&description=Main+proposal+development+team

Response:
{
  "id": "TEAM-A1B2C3D4",
  "name": "Proposal Team",
  "slug": "proposal-team",
  "description": "Main proposal development team",
  "created_at": "2024-12-20T10:00:00"
}
```

```
GET /api/teams

Response:
{
  "teams": [
    {
      "id": "TEAM-A1B2C3D4",
      "name": "Proposal Team",
      "slug": "proposal-team",
      "member_count": 5,
      "my_role": "admin"
    }
  ]
}
```

```
GET /api/teams/{team_id}

Response:
{
  "id": "TEAM-A1B2C3D4",
  "name": "Proposal Team",
  "slug": "proposal-team",
  "description": "Main proposal development team",
  "members": [
    {
      "user_id": "USR-A1B2C3D4",
      "email": "user@example.com",
      "name": "John Doe",
      "role": "admin",
      "joined_at": "2024-12-20T10:00:00"
    }
  ]
}
```

#### Member Management

```
POST /api/teams/{team_id}/members
Content-Type: application/x-www-form-urlencoded

email=newmember@example.com&role=contributor

Response:
{
  "message": "Member added",
  "membership": {
    "user_id": "USR-X1Y2Z3W4",
    "team_id": "TEAM-A1B2C3D4",
    "role": "contributor"
  }
}
```

```
PUT /api/teams/{team_id}/members/{user_id}
Content-Type: application/x-www-form-urlencoded

role=admin

Response:
{
  "message": "Role updated",
  "new_role": "admin"
}
```

```
DELETE /api/teams/{team_id}/members/{user_id}

Response:
{
  "message": "Member removed"
}
```

#### Activity Log

```
GET /api/teams/{team_id}/activity?limit=50

Response:
{
  "activities": [
    {
      "id": "ACT-A1B2C3D4",
      "action": "create",
      "resource_type": "capability",
      "user_name": "John Doe",
      "details": {"name": "Cloud Migration"},
      "created_at": "2024-12-20T10:15:00"
    }
  ]
}
```

### 14.4 Vector Search UI

**Location:** `web/index.html` - VectorSearchPanel component

The Vector Search UI provides AI-powered semantic search across Company Library content.

```javascript
function VectorSearchPanel() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [filters, setFilters] = useState({
        types: ['capability', 'past_performance', 'key_personnel', 'differentiator'],
        topK: 10
    });

    const handleSearch = async () => {
        const response = await fetch(
            `${API_BASE}/library/vector-search?` +
            `query=${encodeURIComponent(query)}&` +
            `types=${filters.types.join(',')}&` +
            `top_k=${filters.topK}`
        );
        const data = await response.json();
        setResults(data.results);
    };

    // Renders search input, type filters, and results with similarity scores
}
```

**Features:**
- Natural language query input
- Type filtering (Capabilities, Past Performance, Key Personnel, Differentiators)
- Configurable result count (Top 5/10/20)
- Similarity score display (0-100%)
- Real-time search feedback

### 14.5 Frontend Components

**TeamsView** - Team workspace management

```javascript
function TeamsView() {
    const [teams, setTeams] = useState([]);
    const [selectedTeam, setSelectedTeam] = useState(null);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showAddMemberModal, setShowAddMemberModal] = useState(false);

    // Team list with member counts
    // Team detail view with member management
    // Create team modal
    // Add member modal with role selection
}
```

**Role-based UI elements:**
- Admin: Can manage team settings, add/remove members, change roles
- Contributor: Can add/edit library content, view team
- Viewer: Read-only access to library content

### 14.6 API Keys for Programmatic Access

**Location:** `api/main.py`

API keys enable programmatic access to PropelAI for integrations and automation.

**Key Features:**
- Secure key generation with SHA256 hashing
- Only the full key is shown once at creation time
- Keys identified by prefix for fast lookup
- Permission levels: read, write, admin
- Optional expiration dates
- Activity logging for all key operations

**Database Model:**

```python
class APIKeyModel(Base):
    __tablename__ = "api_keys"

    id = Column(String(50), primary_key=True)
    team_id = Column(String(50), ForeignKey("teams.id"))
    user_id = Column(String(50), ForeignKey("users.id"))
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False)  # SHA256 hash
    key_prefix = Column(String(10), nullable=False)  # First 10 chars
    permissions = Column(JSONB, default=list)  # ["read", "write", "admin"]
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**API Endpoints:**

```
POST /api/teams/{team_id}/api-keys
Content-Type: application/x-www-form-urlencoded

name=CI+Pipeline&permissions=read,write&expires_days=90

Response:
{
  "id": "KEY-A1B2C3D4",
  "name": "CI Pipeline",
  "key": "pk_abc123...",  // Only shown once!
  "key_prefix": "pk_abc123",
  "permissions": ["read", "write"],
  "expires_at": "2025-03-20T10:00:00",
  "message": "Store this key securely - it will not be shown again!"
}
```

```
GET /api/teams/{team_id}/api-keys

Response:
{
  "api_keys": [
    {
      "id": "KEY-A1B2C3D4",
      "name": "CI Pipeline",
      "key_prefix": "pk_abc123",
      "permissions": ["read", "write"],
      "last_used": "2024-12-20T15:30:00",
      "expires_at": "2025-03-20T10:00:00",
      "is_expired": false
    }
  ]
}
```

```
DELETE /api/teams/{team_id}/api-keys/{key_id}

Response:
{
  "success": true,
  "message": "API key 'CI Pipeline' revoked"
}
```

```
POST /api/auth/verify-key
Content-Type: application/x-www-form-urlencoded

api_key=pk_abc123...

Response:
{
  "valid": true,
  "team_id": "TEAM-A1B2C3D4",
  "permissions": ["read", "write"],
  "name": "CI Pipeline"
}
```

**Security Considerations:**
- Keys are never stored in plain text (only SHA256 hash)
- Full key is only returned at creation time
- Expired keys are rejected at verification
- All key operations are logged to activity log

---

## Appendix A: Dependencies

**requirements.txt:**
```
# API Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Document Processing
pypdf>=3.0.0
pdfplumber>=0.10.0
python-docx>=0.8.11
openpyxl>=3.0.10

# Agentic Workflow
langgraph>=0.0.40

# Database (v4.1)
sqlalchemy>=2.0.0
asyncpg>=0.29.0
psycopg2-binary>=2.9.0

# Utils
httpx>=0.24.0
```

**package.json (Node.js):**
```json
{
  "dependencies": {
    "docx": "^8.0.0"
  }
}
```

---

## Appendix B: API Quick Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| POST | `/api/rfp` | Create RFP |
| GET | `/api/rfp` | List RFPs |
| GET | `/api/rfp/{id}` | Get RFP |
| DELETE | `/api/rfp/{id}` | Delete RFP |
| POST | `/api/rfp/{id}/upload` | Upload files |
| POST | `/api/rfp/{id}/upload-guided` | Guided upload |
| POST | `/api/rfp/{id}/process-best-practices` | Process RFP |
| GET | `/api/rfp/{id}/status` | Get status |
| GET | `/api/rfp/{id}/requirements` | Get requirements |
| GET | `/api/rfp/{id}/export` | Export Excel |
| GET | `/api/rfp/{id}/outline` | Get outline |
| POST | `/api/rfp/{id}/outline` | Generate outline |
| GET | `/api/rfp/{id}/outline/export` | Export Word |

---

*Document generated: December 2024*
*PropelAI v4.1.0*
