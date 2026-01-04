# PropelAI v5.0 As-Built Technical Document

**Version:** 5.0.7
**Date:** January 2025
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
15. [Master Architect Workflow (v4.2)](#15-master-architect-workflow-v42)
16. [Iron Triangle Graph & Validation (v5.0)](#16-iron-triangle-graph--validation-v50)
    - [16.7 Click-to-Verify UI](#167-click-to-verify-ui-fr-12)
    - [16.8 War Room Dashboard](#168-war-room-dashboard-section-41)
    - [16.9 Word Integration API](#169-word-integration-api-section-42)
    - [16.10 Word API Semantic Search (v5.0.1)](#1610-word-api-semantic-search-v501)
    - [16.11 Force-Directed Graph Layout (v5.0.2)](#1611-force-directed-graph-layout-v502)
    - [16.12 Agent Trace Log (NFR-2.3)](#1612-agent-trace-log-nfr-23-data-flywheel)
17. [QA Test Infrastructure (v5.0.3)](#17-qa-test-infrastructure-v503)
    - [17.1 Test Suite Overview](#171-test-suite-overview)
    - [17.2 Test Fixtures](#172-test-fixtures)
    - [17.3 CI/CD Pipeline](#173-cicd-pipeline)
18. [Decoupled Outline v3.0 (v5.0.4)](#18-decoupled-outline-v30-v504)
    - [18.1 Production Bug Fixes](#181-production-bug-fixes)
    - [18.2 Volume Title Parsing](#182-volume-title-parsing)
    - [18.3 Frontend Compatibility](#183-frontend-compatibility)
19. [Outline Remediation v5.0.5](#19-outline-remediation-v505)
    - [19.1 Root Cause Analysis](#191-root-cause-analysis)
    - [19.2 Phase 1: Validation Gate](#192-phase-1-validation-gate)
    - [19.3 Phase 2: Data Source Fix](#193-phase-2-data-source-fix)
    - [19.4 Phase 3: Fallback Removal](#194-phase-3-fallback-removal)
    - [19.5 Iron Triangle Validation](#195-iron-triangle-validation)
    - [19.6 Breaking Changes](#196-breaking-changes)
    - [19.7 Test Results](#197-test-results)
    - [19.8 Endpoint Consolidation (v5.0.5-hotfix)](#198-endpoint-consolidation-v505-hotfix)
    - [19.9 Complete Test Results (v5.0.5-hotfix)](#199-complete-test-results-v505-hotfix)
20. [Parsing Remediation v5.0.6](#20-parsing-remediation-v506)
    - [20.1 Root Cause Analysis](#201-root-cause-analysis)
    - [20.2 Fix 1: Table-First Page Limit Extraction](#202-fix-1-table-first-page-limit-extraction)
    - [20.3 Fix 2: Volume Promotion for Administrative Sections](#203-fix-2-volume-promotion-for-administrative-sections)
    - [20.4 Fix 3: Content-First Solicitation Extraction](#204-fix-3-content-first-solicitation-extraction)
    - [20.5 Updated Parser Flow](#205-updated-parser-flow)
    - [20.6 Solicitation Extraction Flow](#206-solicitation-extraction-flow)
21. [Iron Triangle Enforcement v5.0.7](#21-iron-triangle-enforcement-v507)
    - [21.1 Root Cause Analysis](#211-root-cause-analysis)
    - [21.2 Fix 1: Volume Classification](#212-fix-1-volume-classification)
    - [21.3 Fix 2: Section Header Validation](#213-fix-2-section-header-validation)
    - [21.4 Semantic Match Update](#214-semantic-match-update)
    - [21.5 Expected Behavior](#215-expected-behavior)

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
| Master Architect | 6-phase proposal development workflow | v4.2 |
| F-B-P Drafting | Feature-Benefit-Proof content generation | v4.2 |
| Red Team Review | Government evaluator simulation with color ratings | v4.2 |
| NetworkX DAG | Iron Triangle dependency graph with orphan detection | v5.0 |
| Multi-Page Spanning | SourceCoordinate with visual_rects for spanning requirements | v5.0 |
| Validation Engine | Deterministic L-M-C consistency checking | v5.0 |
| Click-to-Verify UI | Split-screen PDF viewer with multi-page highlights | v5.0 |
| War Room Dashboard | Iron Triangle visualization with CCS score and orphan panel | v5.0 |
| Word Integration API | Context awareness endpoint for Word Add-in | v5.0 |
| Semantic Search | pgvector embeddings for Word API requirement matching | v5.0.1 |
| Force-Directed Graph | Physics-based layout for Iron Triangle visualization | v5.0.2 |
| Agent Trace Log | Data Flywheel foundation (Input→Output→Correction) | v5.0.2 |
| QA Test Infrastructure | 114 tests with GoldenRFP fixtures and CI/CD pipeline | v5.0.3 |
| Decoupled Outline v3.0 | StrictStructureBuilder + ContentInjector + UI/Export fixes | v5.0.4 |
| Outline Remediation | Validation gate, data source fix, fallback removal | v5.0.5 |
| Parsing Remediation | Table-first extraction, volume promotion, content-first solicitation | v5.0.6 |
| Iron Triangle Enforcement | Volume classification, header validation, Section C blocking | v5.0.7 |

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
| v4.1 | 2024-12 | Persistent storage: PostgreSQL + Render Disk |
| v4.2 | 2024-12 | Master Architect Workflow: F-B-P Drafting + Red Team Review |
| v5.0 | 2024-12 | Iron Triangle + Click-to-Verify + War Room + Word API |
| v5.0.1 | 2024-12 | Word API Semantic Search (pgvector embeddings) |
| v5.0.2 | 2024-12 | Force-Directed Graph + Agent Trace Log (NFR-2.3) |
| v5.0.3 | 2024-12 | **QA Test Infrastructure: 114 tests + CI/CD pipeline** |
| v5.0.4 | 2024-12 | **Decoupled Outline v3.0: Fixed production deployment issues** |

### v5.0 Changes

**Phase 1: Iron Triangle Backend**

1. **NetworkX Requirements DAG (FR-2.2)**
   - Directed acyclic graph for requirements dependencies
   - Iron Triangle edges: C ↔ L ↔ M relationships
   - Automatic edge building based on text similarity
   - Orphan detection with suggestions

2. **Multi-Page Spanning (FR-1.3)**
   - SourceCoordinate extended with visual_rects list
   - Support for requirements spanning multiple pages
   - Backwards compatible with single bounding box

3. **Validation Engine (FR-2.3)**
   - Deterministic L-M-C consistency checking
   - Section placement validation
   - Volume restriction enforcement
   - Duplicate detection
   - Compliance score calculation (0-100)

4. **New API Endpoints**
   - `GET /api/rfp/{id}/graph` - Get Iron Triangle graph
   - `GET /api/rfp/{id}/graph/orphans` - Get orphan requirements
   - `POST /api/rfp/{id}/validate` - Validate all requirements
   - `POST /api/rfp/{id}/validate/requirement` - Validate single requirement

5. **New Dependencies**
   - NetworkX >= 3.0 for graph operations

**Phase 2: Click-to-Verify UI (FR-1.2)**

6. **Split-Screen PDF Viewer**
   - Compliance matrix on left, PDF viewer on right
   - Click "Source" to open split-screen view
   - Shift+Click for popup modal mode
   - Selected requirement row highlighted

7. **Multi-Page Highlight Support**
   - Page indicators (dots) for spanning requirements
   - Click dot to navigate to that page
   - Amber highlights for multi-rect requirements
   - Blue highlights for single-rect requirements

8. **PDFViewerModal Enhancements**
   - `mode` prop: "modal" or "split"
   - `highlightPages` computed from visual_rects
   - `getCurrentPageHighlights()` for multi-rect rendering
   - "Spans X pages" badge indicator

9. **MatrixView Updates**
   - `splitScreenMode` state for layout switching
   - `MatrixContent` extracted for reuse
   - Selected row visual indicator
   - Dynamic table height in split mode

**Phase 3: War Room Dashboard (Section 4.1)**

10. **War Room View Component**
    - Iron Triangle visualization with SVG graph
    - C-L-M node positioning in triangular layout
    - Edge rendering with type-based styling
    - Interactive node selection

11. **Compliance Certainty Score (CCS)**
    - Real-time score calculation from graph coverage
    - Color-coded score display (high/medium/low)
    - Breakdown metrics: C→L, C→M, L→M coverage
    - Node and edge count display

12. **Orphan Requirements Panel**
    - Sidebar listing orphan requirements
    - Section-based color coding
    - Suggestion display for resolution
    - Click to view source integration

13. **Graph Visualization**
    - SVG-based node graph
    - Section C (blue), L (green), M (amber) coloring
    - Orphan nodes highlighted in red
    - Edge types: instructs, evaluates, references

14. **Navigation Integration**
    - "War Room" tab in sidebar navigation
    - Triangle icon for menu item
    - Enabled after RFP processing

**Phase 4: Word Integration API (Section 4.2)**

15. **POST /api/word/context Endpoint**
    - Context awareness for Word Add-in
    - Keyword-based requirement matching (Jaccard similarity)
    - Section context lookup from outline
    - Compliance status calculation
    - Intelligent suggestions generation

16. **GET /api/word/rfps Endpoint**
    - List available RFPs for Word Add-in
    - Returns RFP metadata and requirements count

17. **WordContextRequest/Response Models**
    - Pydantic models for type-safe API
    - Configurable max_results (1-20)
    - Optional section_heading and document_context

### v5.0.1 Changes (Semantic Search)

18. **Word API Semantic Search Upgrade**
    - Upgraded from Jaccard similarity to pgvector embeddings
    - EmbeddingGenerator integration (Voyage AI / OpenAI / fallback)
    - Requirement embeddings cached in `rfp["_requirement_embeddings"]`
    - Cosine similarity with 0.3 threshold
    - Automatic fallback to Jaccard if embedding fails

19. **New Request/Response Fields**
    - `use_semantic_search: bool = True` request parameter
    - `search_method: str` response field ("semantic" or "jaccard")

### v5.0.2 Changes (Graph + Trace)

20. **Force-Directed Graph Layout**
    - Replaced random positioning with physics simulation
    - 100 iterations with simulated annealing (alpha cooling)
    - Forces: repulsion (800), attraction (0.05), section clustering (0.15)
    - Velocity damping (0.9) for stable convergence
    - Bounds constraint to keep nodes within canvas

21. **Agent Trace Log (NFR-2.3)**
    - AgentTraceLogModel database schema for action logging
    - Fields: agent_name, action, input_data, output_data, confidence_score
    - Human correction support: correction_type, correction_reason, corrected_by
    - Execution metadata: duration_ms, model_name, token_count
    - 5 performance indexes for query optimization

22. **Agent Trace Log API Endpoints**
    - `POST /api/trace-logs` - Create trace entry
    - `GET /api/trace-logs` - List with filters
    - `GET /api/trace-logs/{id}` - Get specific log
    - `POST /api/trace-logs/{id}/correct` - Submit correction
    - `GET /api/trace-logs/stats/summary` - Correction statistics

### v4.2 Changes

1. **Master Architect Workflow**
   - 6-phase proposal development orchestration
   - Volume cloning fix for outline re-processing
   - Requirement injection into outline sections

2. **Win Theme Generation (P1)**
   - Automatic matching of Company Library to sections
   - Capability → section keyword matching
   - Past performance → proof point extraction
   - Differentiator → discriminator mapping

3. **Page Allocation from Section M (P1)**
   - Weight parsing (percentage, points, qualitative)
   - Proportional page distribution by factor importance
   - Support for "Most Important", "Significant", etc.

4. **F-B-P Drafting Workflow (P3)**
   - Feature-Benefit-Proof structured drafting
   - LangGraph workflow with human-in-the-loop
   - Quality scoring on 5 dimensions
   - Draft revision with feedback integration

5. **Red Team Review (P4)**
   - Government SSEB simulation
   - Color rating system (Blue/Green/Yellow/Red)
   - Finding generation with severity levels
   - Prioritized remediation planning

6. **Draft & Review UI**
   - Section drafts table with quality progress bars
   - Red Team results with color-coded scores
   - Findings and remediation plan display

### v4.1 Changes

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

All authentication endpoints now return JWT tokens for session management.

```
POST /api/auth/register
Content-Type: application/x-www-form-urlencoded

email=user@example.com&name=John+Doe&password=secret123

Response:
{
  "user": {
    "id": "USR-A1B2C3D4",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "message": "Registration successful"
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
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "message": "Login successful"
}
```

```
POST /api/auth/verify
Content-Type: application/json
Authorization: Bearer <token>

Response:
{
  "valid": true,
  "user": {
    "id": "USR-A1B2C3D4",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

```
POST /api/auth/refresh
Content-Type: application/json
Authorization: Bearer <token>

Response:
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {...}
}
```

#### Password Reset

```
POST /api/auth/forgot-password
Content-Type: application/x-www-form-urlencoded

email=user@example.com

Response:
{
  "message": "If an account exists with this email, a password reset link has been sent",
  "reset_token": "abc123..."  // Only in dev mode
}
```

```
POST /api/auth/reset-password
Content-Type: application/x-www-form-urlencoded

token=abc123...&new_password=newSecret123

Response:
{
  "user": {
    "id": "USR-A1B2C3D4",
    "email": "user@example.com",
    "name": "John Doe"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "message": "Password reset successful"
}
```

#### Rate Limiting

All authentication endpoints are rate-limited to prevent abuse:

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/auth/login` | 5 requests | 60 seconds |
| `/api/auth/register` | 3 requests | 60 seconds |
| `/api/auth/forgot-password` | 3 requests | 300 seconds |

Rate-limited responses return HTTP 429 with a Retry-After header:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 45
Content-Type: application/json

{
  "detail": {
    "error": "Too many requests",
    "message": "Rate limit exceeded. Please try again in 45 seconds.",
    "retry_after": 45
  }
}
```

**Implementation Notes:**
- Uses in-memory sliding window algorithm
- Per-IP address tracking (respects X-Forwarded-For header)
- For production, consider Redis-based rate limiting for distributed deployments

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

## 15. Master Architect Workflow (v4.2)

### 15.1 Overview

The Master Architect is a 6-phase proposal development workflow that orchestrates the entire proposal creation process from RFP analysis through final quality review.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MASTER ARCHITECT WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │  P0     │──▶│  P1     │──▶│  P2     │──▶│  P3     │──▶│  P4     │       │
│  │ Outline │   │Strategy │   │Architect│   │ F-B-P   │   │Red Team │       │
│  │  Gen    │   │  Gen    │   │  API    │   │Drafting │   │ Review  │       │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
│       │             │             │             │             │              │
│       ▼             ▼             ▼             ▼             ▼              │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│  │ Volume  │   │Win Theme│   │Require- │   │ Draft   │   │ Color   │       │
│  │Cloning  │   │  from   │   │  ment   │   │ with    │   │ Score   │       │
│  │  Fix    │   │ Library │   │Injection│   │Quality  │   │(B/G/Y/R)│       │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15.2 Phase Implementation

#### Phase 0: Outline Generation Fix

**Location:** `agents/enhanced_compliance/smart_outline_generator.py`

Fixed volume cloning issue that caused outline corruption when re-processing RFPs.

```python
class SmartOutlineGenerator:
    """Generate proposal outline from Section L/M analysis"""

    def _clone_volumes_deep(self, volumes: List[ProposalVolume]) -> List[ProposalVolume]:
        """Deep clone volumes to prevent reference issues"""
        return [
            ProposalVolume(
                id=vol.id,
                name=vol.name,
                volume_type=vol.volume_type,
                page_limit=vol.page_limit,
                order=vol.order,
                eval_factors=vol.eval_factors.copy(),
                sections=[
                    ProposalSection(
                        id=sec.id,
                        name=sec.name,
                        page_limit=sec.page_limit,
                        requirements=sec.requirements.copy(),
                        eval_criteria=sec.eval_criteria.copy(),
                        subsections=sec.subsections.copy()
                    )
                    for sec in vol.sections
                ]
            )
            for vol in volumes
        ]
```

#### Phase 1: Strategy Generation

**Location:** `agents/enhanced_compliance/smart_outline_generator.py`

**Win Theme Generation from Company Library:**

```python
def _generate_win_themes_for_sections(
    self,
    volumes: List[ProposalVolume],
    company_library_data: Optional[Dict],
    eval_factors: List[EvaluationFactor]
) -> None:
    """
    Generate win themes by matching Company Library data to sections.

    Matches:
    - Capabilities to section requirements
    - Past performance to relevant sections
    - Differentiators to hot buttons
    - Key personnel to staffing sections
    """
```

**Page Allocation from Section M Weights:**

```python
def _calculate_page_allocations(
    self,
    volumes: List[ProposalVolume],
    eval_factors: List[EvaluationFactor],
    total_pages: Optional[int]
) -> None:
    """
    Calculate page allocations based on Section M evaluation weights.

    Formula: section_pages = total_pages * (factor_weight / total_weight)

    Weight interpretation:
    - "Most Important" / "Primary" = 0.35
    - "Important" / "Significant" = 0.25
    - "Less Important" / "Secondary" = 0.15
    - Numeric weights (e.g., "30%", "300 points") are normalized
    """
```

**Data Model Extensions:**

```python
@dataclass
class ProposalSection:
    """A section within a proposal volume"""
    id: str
    name: str
    page_limit: Optional[int] = None
    page_allocation: Optional[int] = None  # Calculated from M weights
    requirements: List[str] = field(default_factory=list)
    eval_criteria: List[str] = field(default_factory=list)
    win_themes: List[str] = field(default_factory=list)  # From Company Library
    proof_points: List[str] = field(default_factory=list)  # Evidence
    subsections: List['ProposalSection'] = field(default_factory=list)
```

#### Phase 2: Requirement Injector

**Location:** `agents/enhanced_compliance/requirement_injector.py`

The RequirementInjector maps extracted requirements to outline sections based on semantic matching.

```python
class RequirementInjector:
    """Inject extracted requirements into proposal outline sections"""

    def inject(
        self,
        outline: Dict,
        requirements: List[Dict],
        eval_factors: List[Dict]
    ) -> Dict:
        """
        Inject requirements into outline sections.

        Mapping Strategy:
        - Section L requirements → Compliance/format sections
        - Section M criteria → Evaluation factor sections
        - Section C/SOW requirements → Technical approach sections
        """

    def _find_best_section_match(
        self,
        requirement: Dict,
        sections: List[Dict]
    ) -> Optional[str]:
        """Find best matching section for a requirement using keyword overlap"""
```

**API Endpoint:**

```
POST /api/rfp/{rfp_id}/master-architect

Request Body:
{
  "phases": ["outline", "inject", "allocate", "themes"]
}

Response:
{
  "status": "success",
  "phases_completed": ["outline", "inject", "allocate", "themes"],
  "outline": {
    "volumes": [...],
    "total_pages": 150,
    "warnings": []
  },
  "stats": {
    "sections_with_requirements": 12,
    "sections_with_win_themes": 8,
    "total_page_allocation": 145
  }
}
```

### 15.3 F-B-P Drafting Workflow

**Location:** `agents/drafting_workflow.py`

The Feature-Benefit-Proof (F-B-P) framework ensures all drafted content follows a structured format that connects capabilities to customer benefits with evidence.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        F-B-P DRAFTING WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Research │──▶│Structure │──▶│  Draft   │──▶│ Quality  │──▶│ Human    │  │
│  │   Node   │   │FBP Node  │   │   Node   │   │  Check   │   │ Review   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       │              │              │              │              │         │
│       ▼              ▼              ▼              ▼              ▼         │
│  Query Company  Build F-B-P    Generate      Score Draft    Pause for     │
│  Library for    blocks from    narrative     on 5 dims      feedback      │
│  evidence       requirements   prose                                       │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Quality Scoring Dimensions                        │  │
│  │  compliance (0-100): Addresses all requirements                      │  │
│  │  clarity (0-100): Clear, well-structured prose                       │  │
│  │  citation_coverage (0-100): Evidence properly cited                  │  │
│  │  word_count_ratio (0-100): Fits within page allocation              │  │
│  │  theme_alignment (0-100): Aligns with win themes                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**LangGraph State:**

```python
class DraftingState(TypedDict):
    """State for drafting workflow"""
    rfp_id: str
    section_id: str
    requirements: List[Dict]
    company_library: Dict
    win_themes: List[str]

    # F-B-P blocks
    fbp_blocks: List[Dict]  # [{feature, benefit, proof}]

    # Draft content
    draft_text: str
    word_count: int

    # Quality scores
    quality_scores: Dict[str, float]
    overall_score: float

    # Human review
    feedback: Optional[str]
    revision_count: int
```

**API Endpoints:**

```
POST /api/rfp/{rfp_id}/draft/section/{section_id}

Request Body:
{
  "requirements": ["REQ-001", "REQ-002"],
  "win_themes": ["theme-1"],
  "page_limit": 5
}

Response:
{
  "status": "drafting_started",
  "draft_id": "DRAFT-A1B2C3D4",
  "section_id": "sec-1.1",
  "estimated_time": 30
}
```

```
POST /api/rfp/{rfp_id}/draft/section/{section_id}/feedback

Request Body:
{
  "feedback": "Add more specific metrics on cost savings",
  "action": "revise"  // or "approve", "reject"
}

Response:
{
  "status": "revision_started",
  "revision_number": 2
}
```

```
GET /api/rfp/{rfp_id}/drafts

Response:
{
  "drafts": [
    {
      "section_id": "sec-1.1",
      "section_name": "Technical Approach",
      "status": "completed",
      "quality_scores": {
        "compliance": 92,
        "clarity": 88,
        "citation_coverage": 95,
        "word_count_ratio": 85,
        "theme_alignment": 90
      },
      "overall_score": 90,
      "word_count": 2450,
      "revision_count": 1,
      "updated_at": "2024-12-20T15:30:00"
    }
  ],
  "stats": {
    "total_sections": 12,
    "drafted": 8,
    "pending": 4,
    "average_score": 87
  }
}
```

### 15.4 Red Team Review

**Location:** `agents/red_team_agent.py`

The Red Team Agent simulates a government Source Selection Evaluation Board (SSEB) to identify weaknesses before submission.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RED TEAM REVIEW SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        COLOR RATING SYSTEM                               ││
│  │                                                                          ││
│  │  🔵 BLUE (90-100)   Exceptional - Exceeds requirements                  ││
│  │  🟢 GREEN (70-89)   Strong - Meets requirements with strengths          ││
│  │  🟡 YELLOW (50-69)  Adequate - Meets minimum requirements               ││
│  │  🔴 RED (0-49)      Needs Work - Does not meet requirements             ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        EVALUATION PROCESS                                ││
│  │                                                                          ││
│  │  1. Compliance Check                                                     ││
│  │     - Does section address all L requirements?                          ││
│  │     - Are page limits observed?                                          ││
│  │     - Are format requirements followed?                                  ││
│  │                                                                          ││
│  │  2. Technical Evaluation                                                 ││
│  │     - Does approach address C/SOW requirements?                         ││
│  │     - Is the solution technically sound?                                 ││
│  │     - Are risks identified and mitigated?                                ││
│  │                                                                          ││
│  │  3. Scoring Alignment                                                    ││
│  │     - Does content address M evaluation factors?                        ││
│  │     - Are discriminators highlighted?                                    ││
│  │     - Is evidence sufficient for claimed strengths?                     ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Red Team Agent:**

```python
class RedTeamAgent:
    """Simulate government evaluation board review"""

    def review_proposal(
        self,
        sections: List[Dict],
        requirements: Dict,
        eval_factors: List[Dict]
    ) -> RedTeamResult:
        """
        Perform comprehensive Red Team review.

        Returns:
        - Overall color rating
        - Per-section scores
        - Findings (strengths, weaknesses, deficiencies)
        - Remediation plan
        """

    def _evaluate_section(
        self,
        section: Dict,
        requirements: List[Dict],
        eval_factor: Optional[Dict]
    ) -> SectionScore:
        """Evaluate single section against requirements and criteria"""

    def _generate_findings(
        self,
        section_scores: List[SectionScore]
    ) -> List[Finding]:
        """Generate findings from section evaluations"""

    def _create_remediation_plan(
        self,
        findings: List[Finding]
    ) -> List[RemediationItem]:
        """Create prioritized remediation plan"""
```

**Data Models:**

```python
@dataclass
class Finding:
    """Red Team finding"""
    id: str
    section_id: str
    severity: str  # "critical", "major", "minor"
    category: str  # "compliance", "technical", "scoring"
    description: str
    recommendation: str
    effort_estimate: str  # "low", "medium", "high"

@dataclass
class SectionScore:
    """Section evaluation score"""
    section_id: str
    section_name: str
    score: int  # 0-100
    color: str  # "blue", "green", "yellow", "red"
    strengths: List[str]
    weaknesses: List[str]
    compliance_gaps: List[str]

@dataclass
class RedTeamResult:
    """Complete Red Team review result"""
    overall_score: int
    overall_color: str
    section_scores: List[SectionScore]
    findings: List[Finding]
    remediation_plan: List[Dict]
    summary: str
    review_date: str
```

**API Endpoints:**

```
POST /api/rfp/{rfp_id}/red-team

Request Body:
{
  "sections": ["all"],  // or specific section IDs
  "evaluation_mode": "full"  // or "quick"
}

Response:
{
  "status": "review_complete",
  "result": {
    "overall_score": 78,
    "overall_color": "green",
    "section_scores": [
      {
        "section_id": "sec-1.1",
        "section_name": "Technical Approach",
        "score": 85,
        "color": "green",
        "strengths": ["Clear methodology", "Strong past performance"],
        "weaknesses": ["Risk section could be more detailed"]
      }
    ],
    "findings": [
      {
        "id": "F-001",
        "severity": "major",
        "category": "compliance",
        "description": "Section L.4.2 page limit exceeded by 2 pages",
        "recommendation": "Condense risk discussion and remove redundant content"
      }
    ],
    "remediation_plan": [
      {
        "priority": 1,
        "finding_id": "F-001",
        "action": "Reduce Technical Approach by 2 pages",
        "owner": "Technical Lead",
        "effort": "medium"
      }
    ]
  }
}
```

```
GET /api/rfp/{rfp_id}/red-team/history

Response:
{
  "reviews": [
    {
      "id": "RT-001",
      "date": "2024-12-20T10:00:00",
      "overall_score": 72,
      "overall_color": "green",
      "findings_count": 8
    },
    {
      "id": "RT-002",
      "date": "2024-12-21T14:00:00",
      "overall_score": 85,
      "overall_color": "green",
      "findings_count": 3
    }
  ]
}
```

### 15.5 Draft & Review UI

**Location:** `web/index.html` - DraftingView component

The Draft & Review UI provides a comprehensive interface for managing the drafting and review workflow.

```javascript
function DraftingView({ rfpId }) {
    const [drafts, setDrafts] = useState([]);
    const [redTeamResult, setRedTeamResult] = useState(null);
    const [activeTab, setActiveTab] = useState('drafts'); // 'drafts' | 'redteam'

    // Section Drafts Tab
    // - Table with section name, status, quality scores
    // - Progress bars for each quality dimension
    // - Draft/Redraft action buttons

    // Red Team Results Tab
    // - Overall score card with color rating
    // - Section-by-section breakdown
    // - Findings list with severity badges
    // - Remediation plan display
}
```

**UI Components:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Draft & Review                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │ 📝 Drafted: 8   │ │ ⏳ Pending: 4   │ │ 🎯 Red Team: 78 │                │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ [Section Drafts] [Red Team Results]                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  SECTION DRAFTS TAB:                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Section          │ Status    │ Quality Score        │ Actions        │   │
│  │──────────────────│───────────│──────────────────────│────────────────│   │
│  │ Tech Approach    │ ✓ Done    │ ████████░░ 85%       │ [Redraft]     │   │
│  │ Management       │ ✓ Done    │ ███████░░░ 72%       │ [Redraft]     │   │
│  │ Past Performance │ ⏳ Pending │ ░░░░░░░░░░ --        │ [Draft]       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  RED TEAM RESULTS TAB:                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  OVERALL SCORE                                                  │  │   │
│  │  │  🟢 78 - GREEN                                                 │  │   │
│  │  │  "Strong proposal with minor improvements needed"               │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                       │   │
│  │  Section Scores:                                                      │   │
│  │  🔵 Technical Approach      92                                        │   │
│  │  🟢 Management Plan         78                                        │   │
│  │  🟡 Past Performance        65                                        │   │
│  │                                                                       │   │
│  │  Findings (3):                                                        │   │
│  │  ⚠️ MAJOR: Page limit exceeded in Technical section                  │   │
│  │  ⚠️ MAJOR: Missing risk mitigation details                           │   │
│  │  📝 MINOR: Inconsistent formatting in tables                         │   │
│  │                                                                       │   │
│  │  Remediation Plan:                                                    │   │
│  │  1. Reduce Technical section by 2 pages                               │   │
│  │  2. Add risk register to Management section                           │   │
│  │  3. Standardize table formatting                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Color Score Display:**

```javascript
function getColorClass(score) {
    if (score >= 90) return 'score-blue';      // 🔵 Exceptional
    if (score >= 70) return 'score-green';     // 🟢 Strong
    if (score >= 50) return 'score-yellow';    // 🟡 Adequate
    return 'score-red';                         // 🔴 Needs Work
}

function getColorLabel(score) {
    if (score >= 90) return 'Blue - Exceptional';
    if (score >= 70) return 'Green - Strong';
    if (score >= 50) return 'Yellow - Adequate';
    return 'Red - Needs Work';
}
```

### 15.6 Annotated Outline Export

**Location:** `agents/enhanced_compliance/annotated_outline_exporter.js`

The annotated outline exporter generates Word documents with color-coded annotations for each requirement type.

**Color Coding:**

| Annotation Type | Color | Hex Code |
|-----------------|-------|----------|
| Section L (Compliance) | Red | #FF0000 |
| Section M (Evaluation) | Blue | #0000FF |
| Section C (Technical) | Purple | #800080 |
| Win Themes | Green | #008000 |
| Proof Points | Orange | #FF8C00 |

**Document Structure:**

```javascript
function buildSectionContent(section, data, sectionNum) {
    // Section heading
    children.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        text: `${sectionNum}. ${section.name}`
    }));

    // Page allocation (from M weights OR RFP limit)
    if (section.page_allocation || section.page_limit) {
        children.push(createPageAllocationBlock(section));
    }

    // Section L requirements (RED)
    children.push(createAnnotationBlock(
        "SECTION L - COMPLIANCE REQUIREMENTS",
        section.requirements,
        COLORS.SECTION_L
    ));

    // Section M criteria (BLUE)
    children.push(createAnnotationBlock(
        "SECTION M - EVALUATION CRITERIA",
        section.eval_criteria,
        COLORS.SECTION_M
    ));

    // Win themes (GREEN) - From Company Library
    children.push(createAnnotationBlock(
        "WIN THEMES & DISCRIMINATORS",
        section.win_themes,
        COLORS.WIN_THEME
    ));

    // Proof points (ORANGE) - Evidence from past performance
    children.push(createAnnotationBlock(
        "PROOF POINTS REQUIRED",
        section.proof_points,
        COLORS.PROOF_POINT
    ));

    // Graphics placeholder
    children.push(createGraphicsPlaceholder());

    // Boilerplate guidance
    children.push(createBoilerplateGuidance());
}
```

### 15.7 API Endpoint Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/rfp/{id}/master-architect` | Run Master Architect workflow |
| POST | `/api/rfp/{id}/draft/section/{sec_id}` | Start section draft |
| POST | `/api/rfp/{id}/draft/section/{sec_id}/feedback` | Submit draft feedback |
| GET | `/api/rfp/{id}/drafts` | List all section drafts |
| POST | `/api/rfp/{id}/red-team` | Run Red Team review |
| GET | `/api/rfp/{id}/red-team/history` | Get review history |

---

## 16. Iron Triangle Graph & Validation (v5.0)

### 16.1 Overview

The v5.0 release implements PRD requirements for deterministic compliance through the Iron Triangle dependency graph and validation engine.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IRON TRIANGLE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              Section M                                       │
│                            (Evaluation)                                      │
│                           ╱           ╲                                      │
│                          ╱   EVALUATES  ╲                                    │
│                         ╱                 ╲                                  │
│                        ▼                   ▼                                 │
│                   Section L ◄───────── Section C                             │
│                 (Instructions)  INSTRUCTS  (Performance)                     │
│                                                                              │
│   Legend:                                                                    │
│   • Section C: What contractor must DO (SOW/PWS requirements)               │
│   • Section L: What offeror must WRITE (proposal instructions)              │
│   • Section M: How government will EVALUATE (evaluation criteria)           │
│                                                                              │
│   A complete requirement should have:                                        │
│   • C linked to L (how to write about it)                                   │
│   • C linked to M (how it will be scored)                                   │
│   • L linked to M (what the instruction addresses)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 16.2 NetworkX Requirements DAG

**Location:** `agents/enhanced_compliance/requirements_graph.py`

The RequirementsDAG class provides a directed acyclic graph for requirements dependency mapping.

```python
from agents.enhanced_compliance.requirements_graph import (
    RequirementsDAG,
    EdgeType,
    NodeSection,
    OrphanReport,
    GraphAnalysis,
)

# Create DAG from requirements
dag = RequirementsDAG.from_requirements(
    requirements=requirement_nodes,
    auto_link=True,
    similarity_threshold=0.3
)

# Get orphan requirements
orphans = dag.find_orphans()

# Get graph analysis
analysis = dag.analyze()
print(f"Coverage: {analysis.iron_triangle_coverage}")
```

**Edge Types:**

| EdgeType | Description |
|----------|-------------|
| `REFERENCES` | Generic reference between requirements |
| `INSTRUCTS` | L instructs how to address C |
| `EVALUATES` | M evaluates C or L content |
| `DELIVERS` | Deliverable fulfills requirement |
| `PARENT_OF` | Hierarchical parent relationship |
| `AMENDS` | Amendment modifies original |
| `CONFLICTS` | Conflicting requirements |

**Orphan Detection:**

```python
@dataclass
class OrphanReport:
    orphan_id: str
    section: NodeSection  # C, L, M, or OTHER
    requirement_type: RequirementType
    reason: str  # Why it's an orphan
    suggestion: str  # How to fix
```

### 16.3 Multi-Page Spanning (FR-1.3)

**Location:** `agents/enhanced_compliance/models.py`

SourceCoordinate now supports requirements that span multiple pages.

```python
@dataclass
class VisualRect:
    """A single visual rectangle on a specific page"""
    page_number: int
    bounding_box: BoundingBox

@dataclass
class SourceCoordinate:
    """Links requirement to exact PDF location(s)"""
    document_id: str
    page_number: int  # Primary page
    bounding_box: BoundingBox  # Primary box (backwards compatible)
    visual_rects: List[VisualRect] = []  # All rectangles
    spans_pages: bool = False

    def get_all_pages(self) -> List[int]:
        """Get list of all pages this coordinate spans"""

    def get_rects_for_page(self, page_number: int) -> List[BoundingBox]:
        """Get all bounding boxes for a specific page"""
```

### 16.4 Validation Engine (FR-2.3)

**Location:** `agents/enhanced_compliance/validation_engine.py`

The ValidationEngine enforces Iron Triangle consistency rules.

```python
from agents.enhanced_compliance.validation_engine import (
    ValidationEngine,
    ValidationResult,
    ViolationType,
    Severity,
    validate_requirements,
)

# Full validation
engine = ValidationEngine()
result = engine.validate(
    requirements=requirement_nodes,
    graph=graph_data,  # From RequirementsDAG.to_dict()
    outline=outline_dict  # For volume validation
)

print(f"Valid: {result.is_valid}")
print(f"Compliance Score: {result.compliance_score}")
print(f"Critical Violations: {result.critical_count}")
```

**Violation Types:**

| ViolationType | Severity | Description |
|---------------|----------|-------------|
| `WRONG_SECTION` | WARNING | Content in wrong section |
| `ORPHAN_PERFORMANCE` | CRITICAL | C without L or M links |
| `ORPHAN_INSTRUCTION` | WARNING | L without M link |
| `ORPHAN_EVALUATION` | WARNING | M without targets |
| `DUPLICATE_REQUIREMENT` | WARNING | Same text in multiple places |
| `VOLUME_RESTRICTION` | CRITICAL | Content in wrong volume |
| `CIRCULAR_REFERENCE` | CRITICAL | A -> B -> A dependency |

**Volume Placement Rules:**

```python
VOLUME_SECTION_RULES = {
    "technical": ["C", "SOW", "PWS"],
    "management": ["C", "SOW"],
    "past_performance": ["L"],
    "cost": ["B", "PRICING"],
    "administrative": ["K", "L"],
}
```

### 16.5 API Endpoints

#### Get Requirements Graph

```
GET /api/rfp/{rfp_id}/graph?auto_link=true&similarity_threshold=0.3

Response:
{
  "rfp_id": "RFP-A1B2C3D4",
  "nodes": [
    {
      "id": "REQ-001",
      "text": "The contractor shall...",
      "section": "C",
      "req_type": "performance",
      "confidence": "high"
    }
  ],
  "edges": [
    {
      "source": "REQ-M-001",
      "target": "REQ-001",
      "edge_type": "evaluates",
      "weight": 0.85
    }
  ],
  "analysis": {
    "total_nodes": 156,
    "total_edges": 234,
    "orphan_count": 12,
    "section_counts": {"C": 89, "L": 45, "M": 22},
    "connected_components": 3,
    "iron_triangle_coverage": {
      "c_with_l": 0.82,
      "c_with_m": 0.91,
      "l_with_m": 0.78,
      "overall": 0.84
    },
    "dependency_depth": 4
  },
  "orphans": [...]
}
```

#### Get Orphan Requirements

```
GET /api/rfp/{rfp_id}/graph/orphans

Response:
{
  "rfp_id": "RFP-A1B2C3D4",
  "orphan_count": 12,
  "orphans": [
    {
      "id": "REQ-C-023",
      "section": "C",
      "type": "performance",
      "reason": "Section C requirement has no Section L instruction link",
      "suggestion": "Add proposal instruction that addresses this requirement"
    }
  ],
  "iron_triangle_coverage": {...}
}
```

#### Validate Requirements

```
POST /api/rfp/{rfp_id}/validate

Response:
{
  "rfp_id": "RFP-A1B2C3D4",
  "is_valid": false,
  "total_violations": 8,
  "critical_count": 2,
  "warning_count": 5,
  "info_count": 1,
  "compliance_score": 78.5,
  "violations": [
    {
      "id": "VIO-0001",
      "type": "volume_restriction",
      "severity": "critical",
      "requirement_id": "REQ-C-045",
      "message": "Section C content cannot be placed in past_performance volume",
      "suggestion": "Move to technical volume"
    }
  ]
}
```

#### Validate Single Requirement

```
POST /api/rfp/{rfp_id}/validate/requirement
    ?requirement_id=REQ-001
    &target_section=L.4.2
    &target_volume=technical

Response:
{
  "requirement_id": "REQ-001",
  "is_valid": false,
  "violations": [
    {
      "type": "wrong_section",
      "severity": "critical",
      "message": "Cannot place performance requirement in Section L",
      "suggestion": "This type belongs in: C, SOW"
    }
  ]
}
```

### 16.6 Dependencies

```
networkx>=3.0  # v5.0: Requirements dependency graph
```

### 16.7 Click-to-Verify UI (FR-1.2)

**Location:** `web/index.html` - PDFViewerModal, MatrixView

The Click-to-Verify UI provides visual source verification through split-screen PDF viewing with highlight overlays.

#### Split-Screen Mode

```javascript
// User clicks "Source" button in Compliance Matrix
<button onClick={() => onViewSource(req, true)}>
    Source
</button>

// MatrixView renders split-screen layout
if (splitScreenMode && selectedRequirement) {
    return (
        <div className="split-screen-container">
            <div className="split-screen-left">
                <MatrixContent ... />
            </div>
            <PDFViewerModal
                rfpId={rfpId}
                requirement={selectedRequirement}
                sourceData={sourceData}
                mode="split"
            />
        </div>
    );
}
```

#### Multi-Page Highlight Support

```javascript
// PDFViewerModal handles visual_rects from v5.0 API
const highlightPages = useMemo(() => {
    if (sourceData?.visual_rects) {
        return [...new Set(sourceData.visual_rects.map(vr => vr.page_number))];
    }
    return sourceData?.page_number ? [sourceData.page_number] : [];
}, [sourceData]);

// Render all highlights for current page
const getCurrentPageHighlights = () => {
    return sourceData.visual_rects
        .filter(vr => vr.page_number === currentPage)
        .map(vr => ({
            style: {
                left: `${vr.bounding_box.x0 * 100}%`,
                top: `${vr.bounding_box.y0 * 100}%`,
                width: `${(vr.bounding_box.x1 - vr.bounding_box.x0) * 100}%`,
                height: `${(vr.bounding_box.y1 - vr.bounding_box.y0) * 100}%`
            }
        }));
};
```

#### Page Indicators

The UI shows page indicators (dots) for requirements spanning multiple pages:

```jsx
{highlightPages.map(pageNum => (
    <div
        className={`page-dot has-highlight ${currentPage === pageNum ? 'current' : ''}`}
        onClick={() => setCurrentPage(pageNum)}
    />
))}
```

#### CSS Classes

| Class | Description |
|-------|-------------|
| `.split-screen-container` | Flexbox container for split layout |
| `.split-screen-left` | Matrix table half (50%) |
| `.split-screen-right` | PDF viewer half (50%) |
| `.pdf-highlight` | Blue highlight (single rect) |
| `.pdf-highlight-multi` | Amber highlight (multi-rect) |
| `.page-dot` | Page indicator dot |
| `.page-dot.has-highlight` | Amber dot for pages with content |
| `.spans-badge` | "Multi-page" indicator badge |

#### User Interaction

| Action | Result |
|--------|--------|
| Click "Source" | Open split-screen with PDF viewer |
| Shift+Click "Source" | Open popup modal instead |
| Click page dot | Navigate to that page |
| Click "Close Viewer" | Exit split-screen mode |

### 16.8 War Room Dashboard (Section 4.1)

**Location:** `web/index.html` - WarRoomView

The War Room provides a unified dashboard for Iron Triangle dependency analysis and compliance verification.

#### Component Structure

```javascript
function WarRoomView({ rfpId, onViewSource }) {
    // Fetch graph from /api/rfp/{rfpId}/graph
    // Fetch orphans from /api/rfp/{rfpId}/graph/orphans
    // Calculate CCS from graph coverage metrics

    return (
        <div className="war-room-container">
            {/* CCS Header - Compliance Certainty Score */}
            {/* Iron Triangle Graph - SVG visualization */}
            {/* Orphan Panel - Sidebar with orphan requirements */}
        </div>
    );
}
```

#### Compliance Certainty Score (CCS)

The CCS is calculated from Iron Triangle coverage metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| C → L Coverage | Section C requirements linked to Section L instructions | >85% |
| C → M Coverage | Section C requirements linked to Section M evaluation | >85% |
| L → M Coverage | Section L instructions linked to Section M criteria | >85% |
| Overall CCS | Weighted average of all coverage metrics | >85% |

**Score Levels:**
- **High (85%+):** Green - Excellent coverage
- **Medium (60-84%):** Amber - Needs attention
- **Low (<60%):** Red - Critical gaps

#### Graph Visualization

SVG-based visualization with triangular layout:

```
            Section M (Evaluation)
                    ⚪
                   / \
                  /   \
                 /     \
         Section C ——— Section L
       (Performance)  (Instructions)
```

**Node Colors:**
| Section | Color | Hex Code |
|---------|-------|----------|
| C (SOW/PWS) | Blue | #3b82f6 |
| L | Green | #22c55e |
| M | Amber | #f59e0b |
| Orphan | Red | #ef4444 |

**Edge Types:**
| Type | Style | Description |
|------|-------|-------------|
| instructs | Solid green | L instructs how to address C |
| evaluates | Dashed amber | M evaluates C or L content |
| references | Dotted gray | Generic reference |

#### Orphan Panel

The orphan panel displays requirements without proper Iron Triangle links:

```jsx
<div className="orphan-panel">
    <div className="orphan-header">
        <h3>Orphan Requirements</h3>
        <span className="orphan-count">{count}</span>
    </div>
    <div className="orphan-list">
        {orphans.map(orphan => (
            <div className="orphan-item">
                <span className="orphan-item-id">{orphan.id}</span>
                <span className="orphan-item-section">{orphan.section}</span>
                <div className="orphan-item-text">{orphan.reason}</div>
                <div className="orphan-item-suggestion">{orphan.suggestion}</div>
            </div>
        ))}
    </div>
</div>
```

#### CSS Classes

| Class | Description |
|-------|-------------|
| `.war-room-container` | Grid layout for dashboard |
| `.ccs-header` | Score header spanning full width |
| `.ccs-score-value` | Large score display |
| `.ccs-breakdown` | Coverage metrics row |
| `.iron-triangle-graph` | Main graph container |
| `.graph-legend` | Node type legend |
| `.graph-canvas` | SVG graph viewport |
| `.orphan-panel` | Sidebar for orphans |
| `.orphan-item` | Individual orphan card |

### 16.9 Word Integration API (Section 4.2)

**Location:** `api/main.py`

The Word Integration API provides context awareness for a future Microsoft Word Add-in, enabling real-time compliance verification while writing proposals.

#### POST /api/word/context

Query the Compliance Matrix based on current document context.

**Request:**
```json
{
    "rfp_id": "RFP-A1B2C3D4",
    "current_text": "The contractor shall provide experienced project managers...",
    "section_heading": "Management Approach",
    "document_context": "Technical Volume",
    "max_results": 5
}
```

**Response:**
```json
{
    "rfp_id": "RFP-A1B2C3D4",
    "matching_requirements": [
        {
            "id": "REQ-C-042",
            "text": "The contractor shall provide qualified project management...",
            "type": "performance",
            "section": "C.3.2",
            "priority": "high",
            "similarity_score": 0.421
        }
    ],
    "section_context": {
        "volume": "Technical Volume",
        "section_id": "SEC-2.1",
        "section_title": "Management Approach",
        "page_limit": 15,
        "requirements": ["REQ-C-042", "REQ-C-043"]
    },
    "compliance_status": {
        "requirements_found": 3,
        "addressed_count": 2,
        "compliance_rate": 66.7
    },
    "suggestions": [
        "Address 1 high-priority requirements in this section",
        "Include evidence addressing evaluation criteria"
    ]
}
```

#### GET /api/word/rfps

List available RFPs for Word Add-in integration.

**Response:**
```json
{
    "rfps": [
        {
            "id": "RFP-A1B2C3D4",
            "title": "IT Services RFP",
            "requirements_count": 156,
            "created_at": "2024-12-28T10:00:00Z"
        }
    ],
    "count": 1
}
```

#### Matching Algorithm (Legacy)

The original implementation used Jaccard similarity. See Section 16.10 for the upgraded semantic search.

#### Use Cases

1. **Real-time Compliance Check**: Show relevant requirements as user types
2. **Section Navigation**: Jump to specific RFP requirements from Word
3. **Compliance Status**: Visual indicator of addressed vs. unaddressed requirements
4. **Suggestion Generation**: Context-aware writing suggestions

### 16.10 Word API Semantic Search (v5.0.1)

**Location:** `api/main.py`

The Word API was upgraded from keyword-based Jaccard similarity to pgvector-powered semantic search, significantly improving requirement matching accuracy.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEMANTIC SEARCH FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Query Text ──▶ EmbeddingGenerator ──▶ Query Vector (1536 dims)              │
│                         │                                                    │
│                         ▼                                                    │
│  Requirements ──▶ Cached Embeddings ──▶ Cosine Similarity ──▶ Top-K Results │
│                         │                                                    │
│                         └── rfp["_requirement_embeddings"] (cache)          │
│                                                                              │
│  Fallback: If embedding fails ──▶ Jaccard Similarity                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Implementation

**Helper Functions:**

```python
# Lazy-initialized embedding generator
_word_embedding_generator = None

def _get_embedding_generator():
    """Get or create the embedding generator for Word API"""
    global _word_embedding_generator
    if _word_embedding_generator is None:
        from api.vector_store import EmbeddingGenerator
        _word_embedding_generator = EmbeddingGenerator()
    return _word_embedding_generator

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (magnitude1 * magnitude2)

async def _get_or_create_requirement_embeddings(rfp: Dict, requirements: List[Dict]):
    """Get cached requirement embeddings or generate them."""
    cached = rfp.get("_requirement_embeddings")
    if cached:
        return cached

    # Batch generate embeddings
    generator = _get_embedding_generator()
    embeddings = {}
    for req in requirements:
        embedding = await generator.generate(req.get("text", ""))
        if embedding:
            embeddings[req.get("id")] = embedding

    # Cache in RFP
    rfp["_requirement_embeddings"] = embeddings
    return embeddings
```

#### API Request Enhancement

**New Request Field:**
```json
{
    "rfp_id": "RFP-A1B2C3D4",
    "current_text": "experienced project managers",
    "use_semantic_search": true  // Default: true
}
```

**Enhanced Response:**
```json
{
    "matching_requirements": [...],
    "search_method": "semantic",  // "semantic" or "jaccard"
    ...
}
```

#### Embedding Providers (Priority Order)

| Provider | Environment Variable | Dimension |
|----------|---------------------|-----------|
| Voyage AI | `VOYAGE_API_KEY` | 1536 |
| OpenAI | `OPENAI_API_KEY` | 1536 |
| Simple Hash | (fallback) | 256 |

#### Benefits Over Jaccard

| Aspect | Jaccard | Semantic |
|--------|---------|----------|
| Synonym Matching | ✗ | ✓ (e.g., "personnel" = "staffing") |
| Paraphrasing | ✗ | ✓ |
| Context Understanding | ✗ | ✓ |
| Similarity Threshold | 0.1 | 0.3 |
| Accuracy | ~60% | ~90%+ |

### 16.11 Force-Directed Graph Layout (v5.0.2)

**Location:** `web/index.html` - WarRoomView

The Iron Triangle graph visualization was upgraded from random positioning to a physics-based force-directed simulation.

#### Algorithm

```javascript
// Force-directed simulation parameters
const iterations = 100;
const repulsionStrength = 800;   // Node-node repulsion
const attractionStrength = 0.05; // Edge attraction
const sectionPull = 0.15;        // Pull toward section target
const damping = 0.9;             // Velocity damping
const minDistance = 25;          // Minimum node distance

// Section target positions (Iron Triangle layout)
const sectionTargets = {
    'C': { x: 150, y: 300 },  // Performance - Bottom Left
    'L': { x: 450, y: 300 },  // Instructions - Bottom Right
    'M': { x: 300, y: 100 },  // Evaluation - Top Center
    'OTHER': { x: 300, y: 380 }
};
```

#### Force Components

**1. Repulsion (Inverse Square Law):**
```javascript
// Nodes push apart to prevent overlap
const force = (repulsionStrength * alpha) / (dist * dist);
nodeA.vx -= fx;
nodeB.vx += fx;
```

**2. Attraction (Edge Spring):**
```javascript
// Connected nodes pull together
const force = dist * attractionStrength * alpha;
source.vx += fx;
target.vx -= fx;
```

**3. Section Clustering:**
```javascript
// Nodes gravitate toward their section target
node.vx += dx * sectionPull * alpha;
node.vy += dy * sectionPull * alpha;
```

**4. Simulated Annealing:**
```javascript
// Cooling factor reduces over iterations
const alpha = 1 - iter / iterations;
```

#### Visual Results

- **Before (Random):** Overlapping nodes, no visual hierarchy
- **After (Force-Directed):** Clear C/L/M clustering, visible edge relationships

### 16.12 Agent Trace Log (NFR-2.3 Data Flywheel)

**Location:** `api/database.py`, `api/main.py`

The Agent Trace Log implements NFR-2.3 from the PRD, creating the foundation for the "Data Flywheel" - a system that logs agent actions and human corrections to enable model improvement.

#### Database Schema

```python
class AgentTraceLogModel(Base):
    """Agent Trace Log (NFR-2.3) - Foundation for Data Flywheel"""
    __tablename__ = "agent_trace_logs"

    id = Column(String(50), primary_key=True)           # trace-{uuid}
    rfp_id = Column(String(50), ForeignKey("rfps.id"))  # Associated RFP
    user_id = Column(String(50), ForeignKey("users.id"))

    # Agent identification
    agent_name = Column(String(100), nullable=False)    # e.g., "ComplianceAgent"
    action = Column(String(100), nullable=False)        # e.g., "extract_requirements"

    # Input/Output (core trace data)
    input_data = Column(JSONB, nullable=False)          # What was given to agent
    output_data = Column(JSONB, nullable=True)          # What agent produced
    confidence_score = Column(Float, nullable=True)     # 0.0-1.0

    # Human correction (Data Flywheel)
    human_correction = Column(JSONB, nullable=True)     # Corrected output
    correction_type = Column(String(50), nullable=True) # accepted/modified/rejected
    correction_reason = Column(Text, nullable=True)     # Why corrected
    corrected_by = Column(String(50), ForeignKey("users.id"))
    corrected_at = Column(DateTime, nullable=True)

    # Execution metadata
    duration_ms = Column(Integer, nullable=True)        # Agent execution time
    model_name = Column(String(100), nullable=True)     # LLM model used
    token_count = Column(Integer, nullable=True)        # Tokens consumed

    # Status
    status = Column(String(20), default="completed")    # pending/completed/failed/corrected

    # Indexes for query performance
    __table_args__ = (
        Index('idx_agent_trace_logs_rfp_id', 'rfp_id'),
        Index('idx_agent_trace_logs_agent_name', 'agent_name'),
        Index('idx_agent_trace_logs_action', 'action'),
        Index('idx_agent_trace_logs_created_at', 'created_at'),
        Index('idx_agent_trace_logs_status', 'status'),
    )
```

#### API Endpoints

**POST /api/trace-logs** - Create trace log entry

```json
{
    "rfp_id": "RFP-A1B2C3D4",
    "agent_name": "ComplianceAgent",
    "action": "extract_requirements",
    "input_data": {"document": "section_c.pdf", "page": 5},
    "output_data": {"requirements": [...]},
    "confidence_score": 0.92,
    "duration_ms": 1250,
    "model_name": "gpt-4o-mini"
}
```

**GET /api/trace-logs** - List logs with filters

```
GET /api/trace-logs?rfp_id=RFP-A1B2C3D4&agent_name=ComplianceAgent&status=completed
```

**POST /api/trace-logs/{trace_id}/correct** - Submit human correction

```json
{
    "human_correction": {"requirements": [/* corrected list */]},
    "correction_type": "modified",
    "correction_reason": "Missed 2 SHALL statements in paragraph 3"
}
```

**GET /api/trace-logs/stats/summary** - Correction rate statistics

```json
{
    "total_logs": 1250,
    "corrected_logs": 87,
    "correction_rate": 6.96,
    "by_agent": {
        "ComplianceAgent": 800,
        "StrategyAgent": 300,
        "OutlineAgent": 150
    },
    "by_status": {
        "completed": 1163,
        "corrected": 87
    }
}
```

#### Use Cases

1. **Time-Travel Debugging**: Replay agent decisions to understand failures
2. **Human-in-the-Loop**: Capture corrections as training signals
3. **Performance Monitoring**: Track agent accuracy and correction rates
4. **Model Improvement**: Export corrected data for fine-tuning

#### Data Flywheel Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLYWHEEL                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent Action ──▶ Trace Log ──▶ Human Review ──▶ Correction                 │
│       │                              │                 │                     │
│       │                              ▼                 │                     │
│       │                     [User accepts/modifies]    │                     │
│       │                              │                 │                     │
│       ▼                              ▼                 ▼                     │
│  output_data            correction_type      human_correction               │
│                                                        │                     │
│                                                        ▼                     │
│                                              Training Data Export            │
│                                                        │                     │
│                                                        ▼                     │
│                                              Model Fine-Tuning               │
│                                                        │                     │
│                                                        ▼                     │
│  ◀────────────────────── Improved Agent ◀──────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 17. QA Test Infrastructure (v5.0.3)

**Location:** `tests/`, `requirements-test.txt`, `.github/workflows/test.yml`

The QA Test Infrastructure provides comprehensive automated testing for all PropelAI components, ensuring reliability and enabling confident refactoring.

### 17.1 Test Suite Overview

**114 tests** organized across three tiers:

| Category | Count | Location | Purpose |
|----------|-------|----------|---------|
| Unit | 38 | `tests/unit/` | Graph logic, validation, models |
| Integration | 33 | `tests/integration/` | Word API, semantic search |
| E2E | 24 | `tests/e2e/` | Agent trajectory, trace logs |
| Agent | 19 | `tests/test_agents.py` | All 4 agents + full workflow |

**Test Categories:**

```
tests/
├── conftest.py                      # Shared fixtures (GoldenRFP, MockEmbedding)
├── test_agents.py                   # Agent unit tests (19 tests)
├── unit/
│   └── test_graph_logic.py          # Iron Triangle DAG tests (38 tests)
├── integration/
│   └── test_word_semantic_search.py # Word API tests (33 tests)
└── e2e/
    └── test_agent_trajectory.py     # Trace log tests (24 tests)
```

**Run Tests:**

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agents --cov=api --cov-report=html

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

### 17.2 Test Fixtures

#### GoldenRFP Fixture

A realistic RFP with C/L/M sections for deterministic testing:

```python
@pytest.fixture
def golden_rfp() -> GoldenRFP:
    """
    Provides:
    - 12 requirements across C/L/M sections
    - Realistic SHALL statements
    - Cross-section references
    - 1 intentional orphan (REQ-C-005)
    """
```

**Sections:**
- **Section C (5 reqs):** Performance, compliance, deliverables
- **Section L (4 reqs):** Proposal instructions, format rules
- **Section M (3 reqs):** Evaluation criteria

#### MockEmbeddingGenerator

Deterministic embedding generator for semantic search tests:

```python
class MockEmbeddingGenerator:
    """
    - Returns 1536-dim vectors from text hash
    - Consistent embeddings for same input
    - Configurable failure patterns
    - Tracks call count for verification
    """

    async def generate(self, text: str) -> List[float]:
        # MD5 hash → deterministic random vector
        # Normalized to unit length
```

**Usage:**

```python
def test_semantic_search(mock_embedding_generator):
    # Generator returns consistent vectors
    vec1 = await mock_embedding_generator.generate("cloud services")
    vec2 = await mock_embedding_generator.generate("cloud services")
    assert vec1 == vec2  # Deterministic
```

#### Agent State Fixtures

```python
@pytest.fixture
def initial_state() -> ProposalState:
    """Clean proposal state for agent testing"""

@pytest.fixture
def compliance_complete_state(initial_state, golden_rfp) -> ProposalState:
    """State with requirements extracted"""

@pytest.fixture
def strategy_complete_state(compliance_complete_state) -> ProposalState:
    """State with win themes generated"""
```

### 17.3 CI/CD Pipeline

**Location:** `.github/workflows/test.yml`

GitHub Actions workflow for automated testing:

```yaml
name: Tests

on:
  push:
    branches: [main, develop, 'feature/**', 'claude/**']
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements-test.txt
      - run: pytest tests/ -v --tb=short
      - run: pytest tests/ --cov=agents --cov=api --cov-report=xml
```

**Pipeline Features:**

| Feature | Description |
|---------|-------------|
| Matrix Testing | Python 3.10 + 3.11 |
| Import Verification | Catches missing deps early |
| Coverage Reports | XML for Codecov integration |
| Lint Check | Ruff (non-blocking) |

#### Test Dependencies

**requirements-test.txt:**

```
# Testing framework
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-cov>=4.0.0

# Core dependencies
fastapi>=0.109.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
httpx>=0.24.0
networkx>=3.0

# Security/Auth
bcrypt>=4.0.0
passlib[bcrypt]>=1.7.4
PyJWT>=2.8.0
pyotp>=2.9.0

# Document processing
pypdf>=3.0.0
python-docx>=0.8.11
openpyxl>=3.0.10
```

### 17.4 Agent Trace Log Accumulation Fix

**Issue:** Agents were replacing trace logs instead of accumulating them.

**Root Cause:** Each agent returned `{"agent_trace_log": [trace_log]}` which overwrote previous entries when using `state.update()`.

**Fix:** All agents now accumulate trace logs:

```python
# Before (broken)
return {"agent_trace_log": [trace_log]}

# After (fixed)
existing_trace = state.get("agent_trace_log", [])
return {"agent_trace_log": existing_trace + [trace_log]}
```

**Files Modified:**
- `agents/compliance_agent.py`
- `agents/strategy_agent.py`
- `agents/drafting_agent.py`
- `agents/red_team_agent.py`
- `agents/enhanced_compliance/agent.py`

**Test Coverage:**

```python
def test_trace_log_completeness(self, initial_state, golden_rfp):
    """Verify all agents contribute to trace log"""
    # Run all agents in sequence
    # Assert trace log has entries from each agent
    assert len(trace_log) == 4  # compliance, strategy, drafting, red_team
    agent_names = [t["agent_name"] for t in trace_log]
    assert "ComplianceAgent" in agent_names
    assert "StrategyAgent" in agent_names
    # ...
```

### 17.5 Test Design Principles

1. **Trajectory-First Testing**: Tests validate agent execution order and state transitions
2. **Deterministic Fixtures**: GoldenRFP and MockEmbedding ensure reproducible results
3. **Isolation**: Each test starts with clean state, no shared mutable state
4. **Fast Feedback**: Full suite runs in <5 seconds locally
5. **CI Integration**: Automated testing on every push/PR

---

## 18. Decoupled Outline v3.0 (v5.0.4)

**Location:** `api/main.py`, `agents/enhanced_compliance/section_l_parser.py`, `web/index.html`, `agents/enhanced_compliance/annotated_outline_exporter.js`

The v3.0 Decoupled Outline architecture (StrictStructureBuilder + ContentInjector + OutlineOrchestrator) was implemented in v5.0.3 but had production deployment issues. v5.0.4 fixes these issues.

### 18.1 Production Bug Fixes

**Root Cause:** The v3.0 outline generation failed in production due to multiple issues in the data pipeline.

| Issue | Root Cause | Fix |
|-------|------------|-----|
| `parse_file() missing 1 required positional argument` | `MultiFormatParser.parse_file()` requires `doc_type` parameter | Added `ParserDocType.ATTACHMENT` as second argument |
| `ParsedDocument has no 'text' attribute` | Wrong attribute name - dataclass uses `full_text` | Changed `hasattr(parsed, 'text')` to `hasattr(parsed, 'full_text')` |
| Only 1 volume detected instead of 3 | Was passing extracted requirements text instead of full PDF | Read full document text from `instructions_evaluation` PDF |

**Code Changes (api/main.py):**

```python
# Before (broken)
parsed = parser.parse_file(file_path)
if parsed and hasattr(parsed, 'text'):
    section_l_text = parsed.text

# After (fixed)
parsed = parser.parse_file(file_path, ParserDocType.ATTACHMENT)
if parsed and hasattr(parsed, 'full_text') and parsed.full_text:
    section_l_text = parsed.full_text
```

### 18.2 Volume Title Parsing

**Root Cause:** Volume titles were malformed with trailing content like `'Technical Proposal)'` or `'Cost Proposal), which is linked to all tabs...'`.

**Issue:** The regex pattern `([^\n]+)` captured everything until newline, including parenthetical content. The cleanup only removed `.,:;` not parentheses.

**Fix (section_l_parser.py):**

```python
# Comprehensive title cleanup:
# 1. Remove parenthetical content (page limits, notes)
title = re.sub(r'\s*\([^)]*\).*$', '', title)
# 2. Remove content after comma/semicolon
title = re.sub(r'\s*[,;].*$', '', title)
# 3. Remove trailing punctuation including ')'
title = re.sub(r'[\.\,\;\:\)]+$', '', title).strip()
# 4. Truncate overly long titles with smart break points
if len(title) > 80:
    # Find natural break point
    for sep in [' - ', ' – ', ': ', ' ']:
        if sep in title[:80]:
            title = title[:title.rfind(sep, 0, 80)]
            break
```

### 18.3 Frontend Compatibility

**Issue 1: OutlineView Crash**

```
TypeError: req.slice is not a function
```

**Root Cause:** v3.0 outline returns requirements as objects `{id, text, confidence, ...}` but frontend called `.slice()` assuming strings.

**Fix (web/index.html):**

```javascript
// Before (broken)
• {req.slice(0, 150)}

// After (fixed)
const reqText = typeof req === 'string' ? req : (req?.text || req?.description || '');
• {reqText.slice(0, 150)}
```

**Issue 2: Word Export Crash**

```
TypeError: (factor.criteria || []).join is not a function
```

**Root Cause:** v3.0 returns `factor.criteria` as a string, but exporter called `.join()` expecting array.

**Fix (annotated_outline_exporter.js):**

```javascript
// Before (broken)
text: (factor.criteria || []).join("; ")

// After (fixed)
text: (typeof factor.criteria === 'string'
    ? factor.criteria
    : (Array.isArray(factor.criteria) ? factor.criteria.join("; ") : ""))
```

### 18.4 API Enhancement

**Added `regenerate` parameter to GET outline endpoint:**

```python
@app.get("/api/rfp/{rfp_id}/outline")
async def get_outline(rfp_id: str, format: str = "json", regenerate: bool = False):
    """Get proposal outline. Generates using v3.0 if not cached or regenerate=true."""
    outline = rfp.get("outline") if not regenerate else None
    # ...
```

**Usage:** `GET /api/rfp/{id}/outline?regenerate=true` bypasses cache and forces regeneration.

### 18.5 Debug Logging

Comprehensive debug logging was added to trace the outline generation pipeline:

```
[v3.0 Outline] document_metadata has 4 documents
[v3.0 Outline] Attempting to parse: /data/uploads/RFP-XXX/Attachment.pdf
[v3.0 Outline] Read full text from Attachment.pdf: 25434 chars
[v3.0 Parser] Found 2 volumes
[v3.0 Parser] Volume titles: ['Technical Proposal', 'Cost Proposal']
[v3.0 Outline] annotated_outline.volumes count: 2
[v3.0 Outline] SUCCESS - Generated 2 volumes
[GET Outline] Returning outline with 2 volumes
```

---

## 19. Outline Remediation v5.0.5

**Location:** `api/main.py`, `agents/enhanced_compliance/section_l_parser.py`, `agents/enhanced_compliance/validation_engine.py`, `agents/enhanced_compliance/strict_structure_builder.py`

The v5.0.4 fixes addressed immediate bugs but didn't resolve the root cause of outline hallucination. v5.0.5 implements a comprehensive three-phase remediation based on expert analysis.

### 19.1 Root Cause Analysis

**The Problem:** The outline generator was producing incorrect volumes (e.g., 4 volumes when RFP specified 3) because of three architectural failures:

| Failure Mode | Description | Impact |
|--------------|-------------|--------|
| **Data Source** | System passed extracted requirements (flat list) instead of full PDF text | Volume headers like "Volume I: Technical" were lost |
| **Lenient Mode** | `strict_mode=False` allowed validation errors to be ignored | Invalid outlines reached the UI |
| **Fallback Chain** | SmartOutlineGenerator defaults created hardcoded volumes | Hallucinated Past Performance, Management volumes |

**The Iron Triangle Principle:**

```
Section L → STRUCTURE (mandatory, non-negotiable)
Section M → EVALUATION WEIGHTS (scoring criteria)
Section C → CONTENT (requirements to inject)
```

Government procurement requires exact compliance with Section L structure. Submitting the wrong number of volumes results in **immediate disqualification** as non-responsive.

### 19.2 Phase 1: Validation Gate

**Goal:** Ensure invalid outlines never reach the UI.

**Changes to `/api/rfp/{rfp_id}/outline` endpoint:**

```python
# Changed: strict_mode now defaults to True
async def generate_outline(rfp_id: str, strict_mode: bool = True):
    ...
    # Pass strict_mode to orchestrator
    orchestrator = OutlineOrchestrator(strict_mode=strict_mode)
```

**Three Validation Gates:**

| Gate | Condition | Action |
|------|-----------|--------|
| Gate 1 | `volumes_count == 0` | HTTP 422: "No volumes found in Section L" |
| Gate 2 | `skeleton_valid == False` | HTTP 422: "Structure validation failed" |
| Gate 3 | `stated_volume_count != generated_count` | HTTP 422: "Iron Triangle validation failed" |

**Lenient Mode Response:**

When `strict_mode=False`, validation failures return with `requires_manual_review: True` instead of blocking.

### 19.3 Phase 2: Data Source Fix

**Goal:** Ensure StrictStructureBuilder receives full PDF text, not concatenated requirements.

**Change 1: Store full text during processing**

```python
# In process_rfp_resilient_background():
section_l_full_text = None  # Capture during parsing

for file_path in file_paths:
    parsed = parser.parse_file(file_path, doc_type)
    if parsed:
        # Capture Section L full text for outline generation
        guided_doc_type = document_metadata.get(filename, {}).get("doc_type", "")
        if guided_doc_type in ["instructions_evaluation", "combined_lm", "instructions"]:
            section_l_full_text = parsed.full_text

# Store with RFP data
if section_l_full_text:
    update_data["section_l_full_text"] = section_l_full_text
```

**Change 2: Remove dangerous fallback**

```python
# REMOVED: The fallback that concatenated requirements
# if not section_l_text:
#     section_l_text = "\n\n".join([r.get("text") for r in section_l])

# NEW: Fail with clear error
if not section_l_text:
    raise HTTPException(
        status_code=422,
        detail={
            "error": "Cannot determine proposal structure",
            "reason": "Full Section L document text is not available.",
            "action": "Please re-upload the Instructions/Evaluation document."
        }
    )
```

**Data Flow (Fixed):**

```
PDF Upload → parse_file() → full_text with "VOLUME I:" headers
                              ↓
              Store as rfp["section_l_full_text"]
                              ↓
         Outline Generation reads full text
                              ↓
         SectionLParser finds explicit volumes
                              ↓
         Correct outline generated ✓
```

### 19.4 Phase 3: Fallback Removal

**Goal:** Eliminate all paths that can create hallucinated volumes.

**Change 1: Remove SmartOutlineGenerator from API**

The entire fallback chain to SmartOutlineGenerator was removed:

```python
# REMOVED from api/main.py:
# except Exception:
#     generator = SmartOutlineGenerator()
#     outline = generator.generate_from_compliance_matrix(...)
```

Now, if v3.0 fails, the API returns an error instead of silently using legacy defaults.

**Change 2: Deprecate mention-based extraction**

```python
# In section_l_parser.py - REMOVED the call:
# if not volumes:
#     volumes = self._extract_volumes_from_mentions(full_text, warnings)

# NEW: Just warn if no volumes found
if not volumes:
    warnings.append(
        "No explicit volume declarations found in Section L "
        "(e.g., 'Volume I: Technical Approach'). "
        "Please verify the document contains proposal structure instructions."
    )
```

The `_extract_volumes_from_mentions()` method is now deprecated and returns empty list if called.

**Why Mention-Based Extraction Was Dangerous:**

| RFP Text | Mention-Based Result | Correct Result |
|----------|---------------------|----------------|
| "Submit a technical proposal and cost proposal" | 2 volumes (Technical, Cost) | Depends on Section L structure |
| "Past performance may be discussed in Section 1.3" | 3 volumes (includes Past Performance) | Wrong if PP isn't a separate volume |
| No explicit "Volume I:" patterns | Infers from keywords | Should fail and ask user |

### 19.5 Iron Triangle Validation

**Goal:** Cross-validate generated outline against RFP specifications.

**New Method in ValidationEngine:**

```python
def validate_outline_volume_count(
    self,
    generated_volumes: int,
    stated_volume_count: Optional[int],
    compliance_matrix_volumes: Optional[int] = None,
) -> List[ValidationViolation]:
    """
    Validate that generated outline volume count matches RFP specifications.
    """
    violations = []

    # Check against stated volume count from Section L
    if stated_volume_count is not None and generated_volumes != stated_volume_count:
        violations.append(ValidationViolation(
            violation_type=ViolationType.VOLUME_COUNT_MISMATCH,
            severity=Severity.CRITICAL,  # Non-negotiable
            message=f"Generated {generated_volumes} volumes but Section L specifies {stated_volume_count}",
        ))

    # Cross-check against Compliance Matrix
    if compliance_matrix_volumes is not None and generated_volumes != compliance_matrix_volumes:
        violations.append(ValidationViolation(
            violation_type=ViolationType.VOLUME_COUNT_MISMATCH,
            severity=Severity.WARNING,
            message=f"Generated {generated_volumes} volumes but Compliance Matrix shows {compliance_matrix_volumes}",
        ))

    return violations
```

**New Field in ProposalSkeleton:**

```python
@dataclass
class ProposalSkeleton:
    rfp_number: str
    rfp_title: str
    volumes: List[SkeletonVolume]
    total_page_limit: Optional[int]
    format_rules: Dict[str, Any]
    submission_rules: Dict[str, Any]
    stated_volume_count: Optional[int] = None  # NEW: For Iron Triangle validation
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
```

**API Response Enhancement:**

```json
{
  "status": "generated",
  "version": "3.0",
  "skeleton_valid": true,
  "volumes_count": 3,
  "stated_volume_count": 3,
  "validation": {
    "volume_check_passed": true,
    "violations": [],
    "errors": [],
    "warnings": []
  },
  "outline": {...}
}
```

### 19.6 Breaking Changes

| Change | Impact | Migration |
|--------|--------|-----------|
| `strict_mode=True` default | Invalid outlines return HTTP 422 | Use `?strict_mode=false` for debugging |
| No SmartOutlineGenerator fallback | Legacy mode no longer available | Ensure full PDF text is available |
| Requires full PDF text | Can't generate from requirements only | Re-upload Section L if needed |

### 19.7 Test Results

```
Test 1 - Parser with no volumes:
  Volumes found: 0
  Warnings: ["No explicit volume declarations found..."]
  ✓ PASSED - No hallucinated volumes

Test 2 - Parser with explicit volumes:
  Volumes found: 3
  Stated count: 3
  ✓ PASSED - Found all 3 volumes

Test 3 - Skeleton has stated_volume_count:
  stated_volume_count in skeleton: 3
  ✓ PASSED

Test 4 - Volume count mismatch validation:
  Violations: 1
  Message: Generated 2 volumes but Section L specifies 3
  ✓ PASSED - Mismatch detected as CRITICAL
```

### 19.8 Endpoint Consolidation (v5.0.5-hotfix)

**Problem Discovered:** Despite the v5.0.5 three-phase fix, phantom volumes (e.g., "Past Performance", "Staffing Plan") were still appearing. Expert analysis revealed that the fix only covered ONE of FIVE outline generation paths.

**Root Cause:** Five different API endpoints could generate outlines, but only `POST /outline` was using the v3.0 pipeline correctly:

| Endpoint | Before v5.0.5-hotfix | Status |
|----------|---------------------|--------|
| `POST /api/rfp/{rfp_id}/outline` | v3.0 pipeline | ✓ Fixed in v5.0.5 |
| `GET /api/rfp/{rfp_id}/outline` | Wrong parameter name | ✗ Broken |
| `GET /api/rfp/{rfp_id}/outline/export` | SmartOutlineGenerator | ✗ Broken |
| `POST /api/rfp/{rfp_id}/master-architect` | SmartOutlineGenerator (Phase 2) | ✗ Broken |
| `POST /api/rfp/{rfp_id}/outline/v3` | Concatenated requirements | ✗ Broken |

**Fix 1: GET /outline Parameter Name**

```python
# BEFORE (broken - use_v3 is not a valid parameter)
result = await generate_outline(rfp_id, use_v3=True)

# AFTER (correct - matches POST endpoint signature)
result = await generate_outline(rfp_id, strict_mode=True)
```

**Fix 2: GET /outline/export - Complete Replacement**

The entire SmartOutlineGenerator block (~89 lines) was replaced with v3.0 pipeline:

```python
@app.get("/api/rfp/{rfp_id}/outline/export")
async def export_annotated_outline(rfp_id: str, regenerate: bool = False):
    """
    Export annotated proposal outline as Word document.
    v5.0.5: Uses validated v3.0 pipeline only. No SmartOutlineGenerator fallback.
    """
    rfp = await get_rfp(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    outline = rfp.get("outline")

    # v5.0.5: Use validated v3.0 pipeline instead of SmartOutlineGenerator
    if not outline or regenerate:
        try:
            result = await generate_outline(rfp_id, strict_mode=True)
            outline = result.get("outline")
            if not outline:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Outline generation failed",
                        "message": "v3.0 pipeline returned no outline",
                        "action": "Check Section L document for explicit volume declarations"
                    }
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Outline generation failed",
                    "message": str(e),
                    "action": "Ensure Section L document is uploaded with full text"
                }
            )

    # ... rest of export logic ...
```

**Fix 3: POST /master-architect Phase 2 - SmartOutlineGenerator Removal**

Phase 2 ("Generate RFP-Specific Annotated Outline") was using SmartOutlineGenerator directly. Now uses v3.0 pipeline:

```python
# PHASE 2: Generate RFP-Specific Annotated Outline
# v5.0.5: Uses v3.0 pipeline with Iron Triangle validation
if phase is None or phase == 2:
    outline = rfp.get("outline")

    if not outline:
        try:
            gen_result = await generate_outline(rfp_id, strict_mode=True)
            outline = gen_result.get("outline")
            if not outline:
                results["warnings"].append(
                    "Outline generation failed - check Section L for volume declarations"
                )
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else e.detail.get("error", str(e.detail))
            results["warnings"].append(f"Outline generation failed: {detail}")
            outline = {}
        except Exception as e:
            results["warnings"].append(f"Outline generation error: {str(e)}")
            outline = {}

    phase2_result = {
        "phase": 2,
        "name": "Annotated Outline Generation",
        "volumes_created": len(outline.get("volumes", [])),
        "validation": {
            "used_v3_pipeline": True,
            "skeleton_valid": rfp.get("proposal_skeleton", {}).get("is_valid", False),
        }
    }
```

**Fix 4: POST /outline/v3 - Data Source Fix**

The `/outline/v3` endpoint (ironically named "v3") was building `section_l_text` by concatenating extracted requirements, losing volume headers:

```python
# BEFORE (broken - loses "Volume I: Technical" headers)
section_l_text = "\n\n".join([
    r.get("text", "") or r.get("full_text", "") or r.get("requirement_text", "")
    for r in instructions
])

# AFTER (correct - uses full PDF text)
# Try stored full text first
if rfp.get("section_l_full_text"):
    section_l_text = rfp["section_l_full_text"]

# Fallback: read from instructions_evaluation document
if not section_l_text:
    for doc_name, doc_info in document_metadata.items():
        if doc_info.get("doc_type") == "instructions_evaluation":
            parser = MultiFormatParser()
            parsed = parser.parse_file(doc_info["file_path"], ...)
            section_l_text = parsed.full_text

# NO fallback to concatenation - fail with HTTP 422
if not section_l_text:
    raise HTTPException(status_code=422, detail={
        "error": "Cannot determine proposal structure",
        "action": "Re-upload Instructions/Evaluation document"
    })
```

**Result: Single Source of Truth**

All five endpoints now route through the same validated v3.0 pipeline:

```
POST /outline       ──┐
GET /outline        ──┤
GET /outline/export ──┼──→ generate_outline(strict_mode=True)
POST /master-architect│          ↓
POST /outline/v3    ──┘    OutlineOrchestrator
                                 ↓
                         StrictStructureBuilder
                                 ↓
                           ContentInjector
                                 ↓
                         Validated Outline ✓
```

**SmartOutlineGenerator Status:** Fully deprecated. No production code path references it.

### 19.9 Complete Test Results (v5.0.5-hotfix)

```
TEST 1: Module imports
✓ PASSED - All modules imported successfully

TEST 2: SectionLParser with no volumes (hallucination check)
Volumes found: 0
Warnings: ['No explicit volume declarations found...']
✓ PASSED - No hallucinated volumes

TEST 3: SectionLParser with explicit volumes
Volumes found: 3 (Technical Approach, Cost Proposal, Past Performance)
Stated count: 3
✓ PASSED - Found all 3 declared volumes

TEST 4: StrictStructureBuilder skeleton validation
Skeleton is_valid: True (may show page limit warning)
stated_volume_count in skeleton: 3
✓ PASSED

TEST 5: ValidationEngine volume count mismatch
Violations: 1
Message: Generated 2 volumes but Section L specifies 3
Severity: CRITICAL
✓ PASSED - Mismatch detected

TEST 6: Full OutlineOrchestrator pipeline
Skeleton valid: True
Volumes generated: 2 (Technical Approach, Cost Proposal)
Requirements mapped: 45
✓ PASSED - Generated exactly 2 volumes (no hallucination)
```

---

## 20. Parsing Remediation v5.0.6

**Location:** `agents/enhanced_compliance/section_l_parser.py`, `agents/enhanced_compliance/bundle_detector.py`, `api/main.py`

Despite v5.0.5 fixing phantom volume hallucination, forensic analysis of RFP-C5B27A6A revealed three new parsing failures that would cause proposal disqualification.

### 20.1 Root Cause Analysis

| Bug | Symptom | Impact |
|-----|---------|--------|
| **Table Blindness** | "Page Limit: No limit specified" when table says "8 Pages" | Proposal exceeds page limit → disqualified |
| **Section Nesting** | Contract Documentation nested under Volume 1 | SF1449/DD254 forms count against Tech page limit |
| **Wrong Solicitation** | Cover shows "RFP-C5B27A6A" instead of "FA880625RB003" | Administrative non-compliance |

### 20.2 Fix 1: Table-First Page Limit Extraction

**Problem:** The parser read text flow but ignored tabular associations between Volume and Page Limit columns.

**New Method:** `_extract_page_limits_from_table()` in `section_l_parser.py`

```python
def _extract_page_limits_from_table(
    self,
    text: str,
    warnings: List[str]
) -> Dict[str, Tuple[str, int]]:
    """
    v5.0.6: Extract volume page limits from table structures.

    Tables in Section L often define volume structure like:
    | Volume | Title | Page Limit | Copies |
    |   1    | Technical | 8 Pages | 1 |
    |   2    | Cost/Price | No Limit | 1 |
    """
    # Detect table-like structures
    table_header_patterns = [
        r"(?:volume|vol\.?)\s*[|\t].*?(?:page\s*limit|pages?)[|\t\n]",
        r"(?:title|description)\s*[|\t].*?(?:page\s*limit|pages?)[|\t\n]",
    ]

    # Extract rows with: Volume# | Title | ## Pages
    table_row_patterns = [
        r"(?:Volume\s*)?([1-3IVX]+)\s*[|\t]+\s*([A-Za-z][^|\t\n]{3,50})\s*[|\t]+\s*(\d+)\s*(?:pages?)?",
    ]
```

**Priority Chain:**
1. Table data (highest priority)
2. Partial match in table (e.g., "Technical" → "Technical Proposal")
3. Nearby text extraction (fallback)

### 20.3 Fix 2: Volume Promotion for Administrative Sections

**Problem:** "Section 3: Contract Documentation" was nested under Volume 1 instead of being a separate volume.

**New Method:** `_promote_sections_to_volumes()` in `section_l_parser.py`

```python
def _promote_sections_to_volumes(
    self,
    text: str,
    existing_volumes: List[VolumeInstruction],
    warnings: List[str]
) -> List[VolumeInstruction]:
    """
    v5.0.6: Promote certain sections to root-level volumes.
    """
    volume_promotion_keywords = [
        'contract documentation',
        'representations and certifications',
        'administrative volume',
        'contractual documents',
        'required forms',
        'attachments volume',
    ]
```

**Behavior:**
- Scans for promotion keywords not already in volume list
- Creates new volume with next available number
- Adds warning for user verification

### 20.4 Fix 3: Content-First Solicitation Extraction

**Problem:** System used internal job ID (`RFP-C5B27A6A`) instead of official solicitation number (`FA880625RB003`).

**Root Cause:** Filename extraction had higher priority than document content.

**New Method:** `extract_solicitation_from_content()` in `bundle_detector.py`

```python
def extract_solicitation_from_content(self, text: str) -> Optional[str]:
    """
    v5.0.6: Extract solicitation number from document content.
    Prioritizes header/footer patterns and official format patterns.
    """
    # Labeled patterns (highest confidence)
    label_patterns = [
        r"Solicitation\s*(?:No\.?|Number|#)[:\s]+([A-Z0-9][-A-Z0-9]+)",
        r"RFP\s*(?:No\.?|Number|#)?[:\s]+([A-Z0-9][-A-Z0-9]+)",
    ]
```

**New Agency Patterns Added:**

| Agency | Pattern | Example |
|--------|---------|---------|
| Air Force | `FA\d{4}[-]?\d{2}[-]?[RQ][-]?[A-Z0-9]+` | FA880625RB003 |
| Army | `W\d{3}[A-Z]{2}[-]?\d{2}[-]?[RQ][-]?\d+` | W912DY-25-R-0001 |
| GSA | `\d{2}[A-Z]{4}\d{2}[RQ]\d+` | 47QFCA25R0001 |
| Generic DoD | `[A-Z]{3,5}\d[A-Z0-9][-]?\d{2}[-]?[RQ][-]?\d+` | SPE4A6-25-R-0001 |

**Priority Change in `api/main.py`:**

```python
# v5.0.6: CONTENT-FIRST strategy
# 1. Document content (most reliable - official headers/footers)
# 2. Filename patterns (may contain internal IDs)
# 3. Files array

# Skip internal ID patterns
if filename.startswith("RFP-") and len(filename.split("-")[1]) == 8:
    continue  # Skip internal UUID-like IDs
```

### 20.5 Updated Parser Flow

```
Section L Text
      ↓
┌─────────────────────────────────────┐
│ _extract_page_limits_from_table()   │ ← NEW: Table detection
│ Returns: {title: (title, limit)}    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ _extract_volumes()                  │
│ - Uses table data for page limits   │ ← UPDATED: Table-first
│ - Falls back to nearby text         │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ _promote_sections_to_volumes()      │ ← NEW: Volume promotion
│ - Contract Documentation → Volume 3 │
│ - Reps & Certs → Volume N           │
└─────────────────────────────────────┘
      ↓
Validated Volume List
```

### 20.6 Solicitation Extraction Flow

```
Document Processing
      ↓
┌─────────────────────────────────────┐
│ extract_solicitation_from_content() │ ← Priority 1 (NEW)
│ - Check labeled patterns            │
│ - Check agency-specific formats     │
└─────────────────────────────────────┘
      ↓ (if not found)
┌─────────────────────────────────────┐
│ _extract_solicitation_number()      │ ← Priority 2
│ - Skip RFP-XXXXXXXX internal IDs    │
│ - Check filename patterns           │
└─────────────────────────────────────┘
      ↓
Official Solicitation Number
```

---

## 21. Iron Triangle Enforcement v5.0.7

**Location:** `agents/enhanced_compliance/content_injector.py`, `agents/enhanced_compliance/section_l_parser.py`

Forensic analysis of RFP-239FA6F8 revealed that despite v5.0.6 fixes, technical requirements were being nested in administrative volumes due to keyword matching (e.g., "Personnel" matching both "Key Personnel" and "Personnel Security Questionnaire").

### 21.1 Root Cause Analysis

| Bug | Symptom | Impact |
|-----|---------|--------|
| **Volume Nesting** | Key Personnel requirements in Contract Documentation | Technical content in wrong volume |
| **Header Hallucination** | "505.. This is a competitive..." promoted to section | Invalid section structure |
| **Keyword Overlap** | "Personnel" matches any section with that word | Wrong section mapping |

### 21.2 Fix 1: Iron Triangle Validation in ContentInjector

**New Classification Methods:**

```python
def _classify_volume_type(self, vol_title: str) -> str:
    """Classify volume as 'technical', 'cost', or 'admin'."""

def _classify_requirement_type(self, requirement: Dict) -> str:
    """Classify requirement as 'section_c', 'admin_form', or 'other'."""

def _is_valid_iron_triangle_mapping(self, req_type: str, vol_type: str) -> Tuple[bool, str]:
    """
    Validate mapping against Iron Triangle rules.
    Section C (PWS/SOW) requirements → Technical Volume ONLY
    """
```

**Volume Type Indicators:**

| Volume Type | Indicators |
|-------------|------------|
| Technical | technical, approach, solution, methodology, management, staffing |
| Cost | cost, price, pricing, budget, financial |
| Admin | contract documentation, administrative, representations, certifications |

**Requirement Type Indicators:**

| Requirement Type | Indicators |
|-----------------|------------|
| Section C | shall, must, contractor, offeror shall, provide, deliver |
| Admin Form | sf1449, dd254, certification, representation, disclosure |

**Blocking Rule:**
```python
if req_type == 'section_c' and vol_type == 'admin':
    return False, "Section C/PWS requirements cannot go in Administrative volumes"
```

### 21.3 Fix 2: Header Regex Hardening in SectionLParser

**Problem:** Greedy regex captured "505.. This is a competitive solicitation..." as a section header.

**Pattern Change:**
```python
# BEFORE (greedy - matched 505)
r"(\d+\.\d*)\s+([A-Z][^\n]{4,60})"

# AFTER (max 2 digits)
r"(\d{1,2}\.\d*)\s+([A-Z][^\n]{4,60})"
```

**New Validation Method:**

```python
def _is_valid_section_header(self, sec_id: str, sec_title: str) -> bool:
    """
    v5.0.7: Validate that a section header match is legitimate.

    Rejects:
    - Section numbers > 50 (unrealistic)
    - Titles containing "shall" (requirement prose)
    - Titles matching invalid_header_patterns
    - Titles with > 15 words (too long for header)
    """
```

**Invalid Header Patterns:**
```python
invalid_header_patterns = [
    r'^\d{3,}',              # Starts with 3+ digits (e.g., "505")
    r'^this\s+is\s+',        # Sentence "This is..."
    r'^the\s+\w+\s+shall',   # Requirement "The contractor shall..."
    r'^\w+\s+shall\s+',      # Requirement "... shall ..."
    r'^solicitation',        # "Solicitation" is not a header
    r'^competitive',         # "Competitive" is not a header
    r'^offeror',            # Requirement about offeror
]
```

### 21.4 Semantic Match Update

The `_semantic_match` method now applies Iron Triangle validation before scoring:

```python
for vol in skeleton.get('volumes', []):
    vol_type = self._classify_volume_type(vol['title'])

    # Check Iron Triangle validity BEFORE scoring
    is_valid, violation_reason = self._is_valid_iron_triangle_mapping(
        iron_req_type, vol_type
    )
    if not is_valid:
        print(f"[v5.0.7] Iron Triangle block: {violation_reason}")
        continue  # Skip this volume entirely
```

**Score Boost for Correct Mappings:**
```python
# Boost for correct Iron Triangle mapping
if iron_req_type == 'section_c' and vol_type == 'technical':
    score += 25  # Strong boost
    match_reasons.append("Iron Triangle: Section C → Technical")
```

### 21.5 Expected Behavior

| Requirement | Before v5.0.7 | After v5.0.7 |
|-------------|--------------|--------------|
| "Key Personnel 100% assigned" | Volume 3: Personnel Security | Volume 1: Technical (Staffing) |
| "Infrastructure approach" | Volume 3 (keyword match) | Volume 1: Technical |
| "SF1449 form" | Volume 1 (incorrect) | Volume 3: Contract Documentation |
| "505.. This is competitive..." | Section header | Rejected (invalid) |

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
*PropelAI v5.0.0*
