# PropelAI As-Built Technical Design Document (TDD)

**Document Version:** 1.0
**Last Updated:** December 14, 2024
**System Version:** 2.11.x
**Status:** Production (Beta)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Component Specifications](#4-component-specifications)
5. [Data Models & Schema](#5-data-models--schema)
6. [Processing Pipeline](#6-processing-pipeline)
7. [API Specification](#7-api-specification)
8. [Supported Formats & Agencies](#8-supported-formats--agencies)
9. [Quality Controls & Filters](#9-quality-controls--filters)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Performance Characteristics](#11-performance-characteristics)
12. [Known Limitations](#12-known-limitations)
13. [Security Considerations](#13-security-considerations)
14. [Testing Strategy](#14-testing-strategy)
15. [Operational Procedures](#15-operational-procedures)

---

## 1. Executive Summary

### 1.1 What PropelAI Is

PropelAI is an AI-powered **RFP (Request for Proposal) Intelligence Platform** designed for government contractors. The system automates the extraction of requirements from complex federal RFP documents and generates **Compliance Traceability Matrices (CTMs)** following Shipley Process methodology.

### 1.2 Core Value Proposition

The system transforms the manual "RFP shredding" process—typically requiring **2-5 days of analyst work**—into automated processing completed in **minutes**:

| Capability | Manual Process | PropelAI |
|------------|----------------|----------|
| Document parsing | 4-8 hours | 2-3 minutes |
| Requirement extraction | 2-3 days | 5-10 minutes |
| CTM generation | 4-8 hours | Automatic |
| Amendment reconciliation | 2-4 hours | 15 minutes |

### 1.3 Target Users

- **Primary:** Government contractors responding to federal RFPs
- **Segment:** Mid-tier contractors ($25M-$100M revenue)
- **Roles:** Capture Managers, Proposal Managers, Compliance Analysts

### 1.4 Current Capabilities (Phase 1 Complete)

| Feature | Status | Accuracy |
|---------|--------|----------|
| RFP Document Parsing (PDF/DOCX/XLSX) | ✅ Complete | 95%+ |
| Section L/M/C Extraction | ✅ Complete | 85%+ |
| Compliance Matrix Generation | ✅ Complete | 90%+ |
| Annotated Outline Generation | ✅ Complete | 85%+ |
| Amendment Processing | ✅ Complete | 80%+ |
| Multi-document Bundle Handling | ✅ Complete | 90%+ |
| Agency-specific Detection | ✅ Complete | 95%+ |

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PropelAI Platform                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                    ┌─────────────────────────────────┐│
│  │   Web Frontend  │                    │      Processing Engine          ││
│  │  (React SPA)    │                    │                                 ││
│  │                 │                    │  ┌───────────┐  ┌───────────┐   ││
│  │  - Upload UI    │                    │  │ Document  │  │ Structure │   ││
│  │  - Requirements │◄──────────────────►│  │  Parser   │─►│  Analyzer │   ││
│  │    Table        │    REST API        │  └───────────┘  └───────────┘   ││
│  │  - CTM Export   │                    │        │              │         ││
│  │  - Outline View │                    │        ▼              ▼         ││
│  └─────────────────┘                    │  ┌───────────┐  ┌───────────┐   ││
│                                         │  │Requirement│  │   CTM     │   ││
│  ┌─────────────────┐                    │  │ Extractor │─►│ Exporter  │   ││
│  │  FastAPI Server │                    │  └───────────┘  └───────────┘   ││
│  │                 │                    │        │              │         ││
│  │  /api/rfp/*     │────────────────────│        ▼              ▼         ││
│  │                 │                    │  ┌───────────┐  ┌───────────┐   ││
│  └─────────────────┘                    │  │  Outline  │  │ Amendment │   ││
│           │                             │  │ Generator │  │ Processor │   ││
│           ▼                             │  └───────────┘  └───────────┘   ││
│  ┌─────────────────┐                    └─────────────────────────────────┘│
│  │  In-Memory      │                                                       │
│  │  Storage        │                                                       │
│  │  (RFPStore)     │                                                       │
│  └─────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Frontend** | React | 18.x (CDN) | Single-page application |
| **API** | FastAPI | 0.104+ | REST API with async support |
| **Runtime** | Python | 3.11+ | Backend processing |
| **Document Processing** | PyMuPDF | 1.23+ | PDF text extraction |
| | python-docx | 1.1+ | Word document parsing |
| | openpyxl | 3.1+ | Excel file handling |
| **Storage** | In-memory (dict) | - | RFP project storage |
| **File Storage** | Temp directory | - | `/tmp/propelai_uploads/` |
| **Deployment** | Render.com | Free tier | Cloud hosting |

### 2.3 File Structure

```
/propelai01/
├── api/
│   └── main.py                          # FastAPI application (2000+ lines)
│
├── agents/
│   ├── enhanced_compliance/             # Core extraction engine
│   │   ├── __init__.py                  # Module exports
│   │   ├── agent.py                     # EnhancedComplianceAgent orchestrator
│   │   ├── parser.py                    # MultiFormatParser
│   │   ├── models.py                    # Data structures
│   │   ├── bundle_detector.py           # Document type classifier
│   │   ├── document_structure.py        # RFP structure analyzer (v2.9)
│   │   ├── section_aware_extractor.py   # Best practices extractor (v2.9)
│   │   ├── best_practices_ctm.py        # CTM Excel generator (v2.9)
│   │   ├── semantic_extractor.py        # Semantic classification (v2.8)
│   │   ├── semantic_ctm_export.py       # 19-column CTM (v2.8)
│   │   ├── resolver.py                  # Cross-reference resolution
│   │   ├── amendment_processor.py       # Amendment tracking
│   │   ├── outline_generator.py         # Proposal outline generation
│   │   ├── smart_outline_generator.py   # Enhanced outline generation
│   │   ├── annotated_outline_exporter.py # Word/HTML export (v2.11)
│   │   ├── excel_export.py              # Legacy Excel export
│   │   └── excel_parser.py              # Parse existing CTMs
│   │
│   ├── strategy_agent.py                # Win theme development (stub)
│   ├── drafting_agent.py                # Content generation (stub)
│   └── red_team_agent.py                # Quality evaluation (stub)
│
├── core/
│   ├── config.py                        # Configuration management
│   ├── state.py                         # Proposal state schema
│   └── orchestrator.py                  # LangGraph workflow (stub)
│
├── tools/
│   └── document_tools.py                # Document utilities
│
├── web/
│   └── index.html                       # React single-page application
│
├── tests/
│   └── test_agents.py                   # Unit tests
│
├── cli.py                               # Command-line interface
├── requirements.txt                     # Python dependencies
├── requirements-prod.txt                # Production dependencies
├── Procfile                             # Render deployment config
├── docker-compose.yml                   # Local Docker setup
├── HANDOFF_DOCUMENT.md                  # Developer handoff guide
└── README.md                            # Project overview
```

---

## 3. Architecture Deep Dive

### 3.1 Request Flow

```
User Upload Request
        │
        ▼
┌───────────────────┐
│ FastAPI Endpoint  │  POST /api/rfp/{id}/upload
│ (api/main.py)     │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ File Validation   │  Check extension, size, count
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Store Files       │  /tmp/propelai_uploads/{rfp_id}/
└───────┬───────────┘
        │
        ▼
Processing Request (POST /api/rfp/{id}/process-best-practices)
        │
        ▼
┌───────────────────┐
│ MultiFormatParser │  Extract text from PDF/DOCX/XLSX
│ (parser.py)       │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ RFPStructureParser│  Identify sections (L, M, C, etc.)
│ (document_        │  Find SOW/PWS location
│  structure.py)    │  Parse subsections
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ SectionAware      │  Extract requirements by section
│ Extractor         │  Preserve RFP IDs (L.4.B.2)
│ (section_aware_   │  Classify binding level
│  extractor.py)    │  Detect cross-references
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ BestPracticesCTM  │  Generate multi-sheet Excel
│ Exporter          │  Cover, L, Technical, M, All, Cross-Ref
│ (best_practices_  │
│  ctm.py)          │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Store Results     │  In RFPStore memory
└───────┬───────────┘
        │
        ▼
Export Request (GET /api/rfp/{id}/export)
        │
        ▼
┌───────────────────┐
│ FileResponse      │  Stream Excel file to client
└───────────────────┘
```

### 3.2 Extraction Modes

PropelAI supports three extraction modes, selectable via UI dropdown:

| Mode | Endpoint | Use Case | Accuracy |
|------|----------|----------|----------|
| **Legacy** | `/process` | Quick testing | ~40% |
| **Semantic** | `/process-semantic` | Simple RFPs | ~70% |
| **Best Practices** | `/process-best-practices` | Production | ~85% |

**Best Practices Mode** (Default) implements:
1. Structure analysis BEFORE extraction
2. Preservation of RFP's own requirement IDs
3. Separate L/M/C matrices
4. Verbatim extraction (never summarize)
5. Cross-reference tracking

### 3.3 Agent Architecture (Future)

The codebase includes stub implementations for a multi-agent system:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Proposal Orchestrator                        │
│                     (core/orchestrator.py)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │   Compliance  │  │   Strategy    │  │   Drafting    │        │
│  │     Agent     │  │     Agent     │  │     Agent     │        │
│  │               │  │               │  │               │        │
│  │ - RFP Shred   │  │ - Win Themes  │  │ - Section     │        │
│  │ - CTM Gen     │  │ - PWin Calc   │  │   Drafts      │        │
│  │ - Gap Analysis│  │ - Competitive │  │ - Response    │        │
│  │               │  │   Intel       │  │   Generation  │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│          │                  │                  │                 │
│          └──────────────────┼──────────────────┘                 │
│                             ▼                                    │
│                    ┌───────────────┐                            │
│                    │   Red Team    │                            │
│                    │     Agent     │                            │
│                    │               │                            │
│                    │ - Compliance  │                            │
│                    │   Review      │                            │
│                    │ - Quality     │                            │
│                    │   Scoring     │                            │
│                    └───────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Current Status:** Only Compliance Agent is fully implemented. Others are stubs.

---

## 4. Component Specifications

### 4.1 MultiFormatParser (`parser.py`)

**Purpose:** Extract text content from uploaded documents.

**Supported Formats:**

| Format | Library | Extraction Method |
|--------|---------|-------------------|
| PDF | PyMuPDF (fitz) | `page.get_text()` |
| DOCX | python-docx | Paragraph iteration |
| XLSX | openpyxl | Cell-by-cell extraction |
| DOC | python-docx | Auto-convert to DOCX |
| XLS | openpyxl | Auto-convert to XLSX |

**Output Model:**
```python
@dataclass
class ParsedDocument:
    filename: str           # Original filename
    full_text: str          # Complete extracted text
    pages: List[str]        # Text by page (PDF) or sheet (XLSX)
    page_count: int         # Total pages/sheets
    document_type: DocumentType  # Classification
    sections: Dict[str, str]     # Detected section content
    metadata: Dict[str, Any]     # File metadata
```

**Quality Score Calculation:**
- Based on text-to-binary ratio
- Characters per page metric
- Detection of extraction errors

### 4.2 RFPStructureParser (`document_structure.py`)

**Purpose:** Identify UCF sections and document structure before extraction.

**Detection Patterns:**
```python
SECTION_PATTERNS = {
    UCFSection.SECTION_L: [
        r"SECTION\s+L[\s:\-–—]+",
        r"INSTRUCTIONS.*(?:CONDITIONS|NOTICES).*OFFERORS",
        r"L\s*[-–—\.]\s*INSTRUCTIONS",
    ],
    UCFSection.SECTION_M: [
        r"SECTION\s+M[\s:\-–—]+",
        r"EVALUATION\s+(?:FACTORS|CRITERIA)",
        r"M\s*[-–—\.]\s*EVALUATION",
    ],
    UCFSection.SECTION_C: [
        r"SECTION\s+C[\s:\-–—]+",
        r"STATEMENT\s+OF\s+(?:WORK|OBJECTIVES)",
        r"PERFORMANCE\s+WORK\s+STATEMENT",
    ],
}
```

**SOW/PWS Location Detection:**
```python
SOW_LOCATION_PATTERNS = [
    ("SECTION_C", r"(?:Section\s+)?C.*(?:SOW|PWS|Statement)"),
    ("ATTACHMENT_2", r"Attachment\s+(?:2|II).*SOW"),
    ("ATTACHMENT_1", r"Attachment\s+(?:1|I).*PWS"),
    ("EXHIBIT_A", r"Exhibit\s+A.*(?:Work|Performance)"),
]
```

**Output:**
```python
@dataclass
class DocumentStructure:
    solicitation_number: str
    title: str
    agency: str
    sections: Dict[UCFSection, SectionBoundary]
    attachments: Dict[str, AttachmentInfo]
    sow_location: Optional[str]
    pws_location: Optional[str]
    has_section_l: bool
    has_section_m: bool
    is_ucf_format: bool
    total_pages: int
```

### 4.3 SectionAwareExtractor (`section_aware_extractor.py`)

**Purpose:** Extract requirements while respecting document structure.

**Key Principles:**
1. NEVER rename RFP's own requirement references
2. Extract COMPLETE requirement text (paragraphs, not fragments)
3. Maintain clear separation of L/M/C requirements
4. Track cross-references for compliance mapping

**Binding Level Detection:**
```python
MANDATORY_KEYWORDS = [
    r'\bshall\b', r'\bmust\b',
    r'\brequired\s+to\b', r'\bis\s+required\b',
    r'\bshall\s+not\b', r'\bmust\s+not\b',
]

SHOULD_KEYWORDS = [
    r'\bshould\b', r'\bis\s+expected\b',
]

MAY_KEYWORDS = [
    r'\bmay\b', r'\bcan\b', r'\bis\s+encouraged\b',
]
```

**Category Classification:**

| Category | Source Sections | Purpose |
|----------|-----------------|---------|
| `L_COMPLIANCE` | Section L | Submission instructions |
| `TECHNICAL` | Section C, PWS, SOW, Attachments | Performance requirements |
| `EVALUATION` | Section M | Scoring criteria |
| `ADMINISTRATIVE` | Sections B, F, G, H | Contract admin |
| `ATTACHMENT` | J-Attachments | Technical specs |

### 4.4 BestPracticesCTMExporter (`best_practices_ctm.py`)

**Purpose:** Generate evaluator-ready Excel compliance matrices.

**Output Sheets:**

| Sheet | Purpose | Key Columns |
|-------|---------|-------------|
| **Cover** | Summary & navigation | RFP info, counts, legend |
| **Section L Compliance** | Submission checklist | RFP Ref, Text, Page, Binding, Compliance Status |
| **Technical Requirements** | PWS/SOW requirements | Req ID, Text, Source, Binding, How We Meet This |
| **Section M Alignment** | Evaluation factors | Factor, Criterion, Weight, Our Strength, Proof Points |
| **All Requirements** | Complete list | All columns, all categories |
| **Cross-References** | L→M→C linkages | Section structure, attachment index |

**Formatting:**
- Header row: Blue background, white text, frozen
- Mandatory rows: Light orange background
- Alternating row colors for readability
- Auto-adjusted column widths
- Data validation dropdowns for status fields

### 4.5 AnnotatedOutlineExporter (`annotated_outline_exporter.py`)

**Purpose:** Generate proposal outlines with requirements mapped to sections.

**Supported Outputs:**
- Word document (.docx)
- HTML for web display

**Outline Structure:**
```
1.0 Technical Approach
    1.1 Understanding of Requirements
        [Requirement: L.4.1 - Demonstrate understanding...]
        [Requirement: M.2.a - Technical approach will be evaluated...]
    1.2 Technical Solution
        [Requirement: C.3.1 - The contractor shall provide...]
2.0 Management Approach
    2.1 Project Management
        ...
```

---

## 5. Data Models & Schema

### 5.1 Core Enumerations

```python
class UCFSection(Enum):
    """Uniform Contract Format sections per FAR 15.204-1"""
    SECTION_A = "A"   # Solicitation/Contract Form
    SECTION_B = "B"   # Supplies or Services and Prices
    SECTION_C = "C"   # Description/SOW/PWS
    SECTION_D = "D"   # Packaging and Marking
    SECTION_E = "E"   # Inspection and Acceptance
    SECTION_F = "F"   # Deliveries or Performance
    SECTION_G = "G"   # Contract Administration
    SECTION_H = "H"   # Special Contract Requirements
    SECTION_I = "I"   # Contract Clauses
    SECTION_J = "J"   # List of Attachments
    SECTION_K = "K"   # Representations/Certifications
    SECTION_L = "L"   # Instructions to Offerors
    SECTION_M = "M"   # Evaluation Factors

class RequirementCategory(Enum):
    SECTION_L_COMPLIANCE = "L_COMPLIANCE"
    TECHNICAL_REQUIREMENT = "TECHNICAL"
    EVALUATION_FACTOR = "EVALUATION"
    ADMINISTRATIVE = "ADMINISTRATIVE"
    ATTACHMENT_REQUIREMENT = "ATTACHMENT"

class BindingLevel(Enum):
    MANDATORY = "Mandatory"        # SHALL, MUST
    HIGHLY_DESIRABLE = "Highly Desirable"  # SHOULD
    DESIRABLE = "Desirable"        # MAY, CAN
    INFORMATIONAL = "Informational"

class DocumentType(Enum):
    MAIN_SOLICITATION = "main_solicitation"
    STATEMENT_OF_WORK = "statement_of_work"
    PERFORMANCE_WORK_STATEMENT = "performance_work_statement"
    RESEARCH_OUTLINE = "research_outline"  # NIH-specific
    CDRL = "cdrl"                          # DoD
    AMENDMENT = "amendment"
    ATTACHMENT = "attachment"
    BUDGET_TEMPLATE = "budget_template"
    SECURITY = "security"                  # DD254
    FORM = "form"                          # SF33, etc.
    QA_RESPONSE = "qa_response"
```

### 5.2 StructuredRequirement Model

```python
@dataclass
class StructuredRequirement:
    # Identification
    rfp_reference: str        # RFP's own ID (L.4.B.2) - PRESERVED
    generated_id: str         # Backup ID (TW-L-001)

    # Content
    full_text: str            # Complete requirement text (VERBATIM)

    # Classification
    category: RequirementCategory
    binding_level: BindingLevel
    binding_keyword: str      # The actual "shall", "must", etc.

    # Source Location
    source_section: UCFSection
    source_subsection: Optional[str]  # e.g., "L.4.B"
    page_number: int
    source_document: str

    # Context
    parent_title: str         # Subsection title
    evaluation_factor: Optional[str]  # If linked to Section M

    # Cross-references
    references_to: List[str]  # Other sections/attachments referenced

    # Compliance Gate flag
    is_compliance_gate: bool  # Pass/fail requirement

    # Deduplication
    text_hash: str            # MD5 of normalized text
```

### 5.3 RFP Store Schema

```python
# In-memory structure for each RFP project
{
    "id": "RFP-a1b2c3d4",
    "name": "NIH Cancer Research RFP",
    "solicitation_number": "75N96025R00004",
    "agency": "NIH",
    "due_date": "2025-03-15",
    "status": "processed",  # created, uploaded, processing, processed, error
    "files": ["solicitation.pdf", "sow.pdf"],
    "file_paths": ["/tmp/.../solicitation.pdf", "/tmp/.../sow.pdf"],
    "requirements": [StructuredRequirement, ...],
    "requirements_graph": {...},  # Cross-reference graph
    "stats": {
        "total": 156,
        "section_l": 23,
        "technical": 98,
        "evaluation": 35,
        "mandatory": 89,
        "by_binding": {...}
    },
    "amendments": [AmendmentData, ...],
    "created_at": "2025-01-15T10:30:00",
    "updated_at": "2025-01-15T11:45:00"
}
```

---

## 6. Processing Pipeline

### 6.1 Document Upload Flow

```
1. POST /api/rfp/{rfp_id}/upload
   │
   ├── Validate file extension (PDF, DOCX, XLSX, DOC, XLS)
   ├── Check file size (max 15MB per file)
   ├── Check total upload size (max 50MB)
   │
   ├── Create directory: /tmp/propelai_uploads/{rfp_id}/
   ├── Save files to disk
   │
   └── Return: {"uploaded": ["file1.pdf", ...], "count": 3}
```

### 6.2 Best Practices Processing Flow

```
1. POST /api/rfp/{rfp_id}/process-best-practices
   │
   ├── STAGE 1: Document Parsing
   │   ├── For each file in /tmp/propelai_uploads/{rfp_id}/
   │   │   ├── Detect format (PDF/DOCX/XLSX)
   │   │   ├── Extract text with page boundaries
   │   │   ├── Calculate quality score
   │   │   └── Return ParsedDocument
   │   └── Concatenate all documents
   │
   ├── STAGE 2: Structure Analysis
   │   ├── Detect UCF sections (A-M)
   │   ├── Identify SOW/PWS location
   │   ├── Parse subsection hierarchy
   │   ├── Catalog attachments
   │   ├── Detect solicitation number
   │   ├── Identify agency
   │   └── Return DocumentStructure
   │
   ├── STAGE 3: Requirement Extraction
   │   ├── For each section in structure:
   │   │   ├── Extract paragraphs with binding language
   │   │   ├── Preserve RFP's own IDs
   │   │   ├── Classify binding level
   │   │   ├── Detect cross-references
   │   │   ├── Filter noise (TOC, headers, boilerplate)
   │   │   └── Add to category list
   │   ├── Deduplicate by text hash
   │   └── Return ExtractionResult
   │
   ├── STAGE 4: CTM Generation
   │   ├── Create Excel workbook
   │   ├── Generate Cover Sheet
   │   ├── Generate Section L Compliance
   │   ├── Generate Technical Requirements
   │   ├── Generate Section M Alignment
   │   ├── Generate All Requirements
   │   ├── Generate Cross-References
   │   ├── Apply formatting
   │   └── Save to /tmp/propelai_outputs/{rfp_id}.xlsx
   │
   └── STAGE 5: Store Results
       ├── Update RFP status to "processed"
       ├── Store requirements in memory
       └── Return processing stats
```

### 6.3 Amendment Processing Flow

```
1. POST /api/rfp/{rfp_id}/amendments
   │
   ├── Upload amendment file(s)
   ├── Parse amendment content
   │
   ├── Detect change types:
   │   ├── ADDITION - New requirements
   │   ├── DELETION - Removed requirements
   │   ├── MODIFICATION - Changed text
   │   └── DATE_CHANGE - Schedule updates
   │
   ├── Match changes to original requirements
   │   ├── By section reference (L.4.B.2)
   │   ├── By text similarity (fuzzy match)
   │   └── By page reference
   │
   ├── Update requirements list
   │   ├── Mark deleted items
   │   ├── Add new items with amendment source
   │   └── Update modified items with change tracking
   │
   └── Regenerate CTM with amendment column
```

---

## 7. API Specification

### 7.1 Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check, version info |
| `/api/rfp` | GET | List all RFP projects |
| `/api/rfp` | POST | Create new RFP project |
| `/api/rfp/{id}` | GET | Get RFP details |
| `/api/rfp/{id}` | DELETE | Delete RFP project |
| `/api/rfp/{id}/upload` | POST | Upload documents |
| `/api/rfp/{id}/process` | POST | Legacy extraction |
| `/api/rfp/{id}/process-semantic` | POST | Semantic extraction |
| `/api/rfp/{id}/process-best-practices` | POST | Best practices extraction |
| `/api/rfp/{id}/status` | GET | Processing status |
| `/api/rfp/{id}/requirements` | GET | Get requirements |
| `/api/rfp/{id}/export` | GET | Download CTM Excel |
| `/api/rfp/{id}/amendments` | POST | Upload amendment |
| `/api/rfp/{id}/amendments` | GET | Amendment history |
| `/api/rfp/{id}/outline` | POST | Generate outline |
| `/api/rfp/{id}/outline` | GET | Get outline |
| `/api/rfp/{id}/outline/export` | GET | Export outline (Word) |

### 7.2 Request/Response Examples

**Create RFP:**
```http
POST /api/rfp
Content-Type: application/json

{
  "name": "NIH Cancer Research RFP",
  "solicitation_number": "75N96025R00004",
  "agency": "NIH",
  "due_date": "2025-03-15"
}

Response 201:
{
  "id": "RFP-a1b2c3d4",
  "name": "NIH Cancer Research RFP",
  "status": "created",
  "created_at": "2025-01-15T10:30:00"
}
```

**Upload Documents:**
```http
POST /api/rfp/{id}/upload
Content-Type: multipart/form-data

files: [solicitation.pdf, sow.pdf, amendment1.pdf]

Response 200:
{
  "uploaded": ["solicitation.pdf", "sow.pdf", "amendment1.pdf"],
  "count": 3
}
```

**Check Status:**
```http
GET /api/rfp/{id}/status

Response 200:
{
  "status": "completed",
  "progress": 100,
  "message": "Extraction complete",
  "stats": {
    "total": 156,
    "section_l": 23,
    "technical": 98,
    "evaluation": 35,
    "mandatory": 89
  }
}
```

---

## 8. Supported Formats & Agencies

### 8.1 Agency Detection

| Agency | Pattern | Solicitation Format |
|--------|---------|---------------------|
| **NIH** | `^75N` | 75N96025R00004 |
| **Navy** | `^N\d{5}` | N0017826R30020003 |
| **Army** | `^W` | W912HQ-24-R-0001 |
| **Air Force** | `^FA` | FA8730-24-R-0001 |
| **GSA** | `^GS` | GS-00F-12345 |
| **VA** | `^36C` | 36C25724R0001 |
| **DHS** | `^70\|^HSC` | 70CDCR24R00001 |
| **HHS/CMS** | `^75` (non-NIH) | 75P00124R00001 |

### 8.2 RFP Format Types

| Type | Characteristics | Section L | Section M |
|------|-----------------|-----------|-----------|
| **UCF (Standard)** | FAR 15.204-1 format | Full Section L | Full Section M |
| **GSA Schedule** | Task order format | Often in M or SOW | In main body |
| **BPA Call** | Simplified | Minimal | Minimal |
| **IDIQ TO** | Pre-competed | Reference Master | Abbreviated |
| **Commercial (FAR 12)** | Streamlined | Combined | Combined |

### 8.3 NIH-Specific Handling

| Feature | Detection | Handling |
|---------|-----------|----------|
| Research Outlines | `RO\s*[IVX]+` | Extract as technical requirements |
| Attachment 2 = SOW | Filename pattern | Mark as SOW source |
| Institute Codes | NIEHS, NCI, NIAID | Agency sub-identification |
| 75N Format | `75N\d{5}[A-Z]\d{5}` | NIH agency assignment |

### 8.4 DoD-Specific Handling

| Feature | Detection | Handling |
|---------|-----------|----------|
| DFARS Clauses | `DFARS\s*\d+\.\d+` | Extract as administrative |
| DD254 | Filename | Security requirement attachment |
| CDRL | Document type | Extract deliverable requirements |
| J-Attachments | `J[-_.]?\d+` | Attachment indexing |
| SF33 | Form detection | Main solicitation identifier |

---

## 9. Quality Controls & Filters

### 9.1 Requirement Filters

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Minimum length | 40 characters | Remove fragments |
| Maximum length | 2,000 characters | Prevent full sections |
| Minimum words | 10 words | Remove headers |
| Alpha ratio | 50% alphabetic | Filter tables/numbers |
| TOC detection | Pattern match | Remove table of contents |
| Boilerplate detection | 67 patterns | Remove standard clauses |

### 9.2 Boilerplate Patterns (Sample)

```python
BOILERPLATE_PATTERNS = [
    r"contractor shall comply with all applicable",
    r"in accordance with FAR",
    r"pursuant to the terms",
    r"government furnished",
    r"subject to availability",
    r"standard form",
    # ... 67 total patterns
]
```

### 9.3 SF30 Amendment Form Filter

Amendment forms are filtered to prevent extraction of form fields as requirements:

```python
SF30_PATTERNS = [
    r'amendment of solicitation/modification of contract',
    r'contract id code',
    r'item no\.\s*supplies/services\s*quantity\s*unit',
    r'name of offeror or contractor',
    # ... form field patterns
]
```

### 9.4 Excel Formula Protection

Cell values that could be interpreted as Excel formulas are escaped:

```python
def _safe_cell_value(self, value: str) -> str:
    """Escape values starting with =, +, -, @ or tab"""
    if value.strip().startswith(('=', '+', '-', '@', '\t')):
        return "'" + value
    return value
```

---

## 10. Deployment Architecture

### 10.1 Render.com Configuration

**Procfile:**
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | 8000 | Server port (set by Render) |
| `ANTHROPIC_API_KEY` | No | None | Claude API (optional) |
| `GOOGLE_API_KEY` | No | None | Gemini API (optional) |

### 10.2 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 512MB | 1GB |
| Storage | 1GB | 5GB |
| CPU | 0.5 vCPU | 1 vCPU |

### 10.3 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/
```

---

## 11. Performance Characteristics

### 11.1 Processing Times (Benchmarks)

| RFP Size | Documents | Pages | Processing Time |
|----------|-----------|-------|-----------------|
| Small | 1-2 | <50 | 30-60 seconds |
| Medium | 3-5 | 50-200 | 2-5 minutes |
| Large | 5-10 | 200-500 | 5-10 minutes |
| Very Large | 10+ | 500+ | 10-15 minutes |

### 11.2 Memory Usage

| Phase | Memory Usage |
|-------|--------------|
| Idle | ~100MB |
| Parsing | ~200-500MB |
| Extraction | ~300-600MB |
| Export | ~200-400MB |

### 11.3 Rate Limits

| Limit | Value |
|-------|-------|
| Max concurrent requests | 1 (not designed for concurrency) |
| Max file size | 15MB |
| Max total upload | 50MB |
| Max text length | 2,000,000 characters |

---

## 12. Known Limitations

### 12.1 Critical Limitations

| Limitation | Impact | Severity |
|------------|--------|----------|
| No OCR support | Scanned PDFs produce no text | HIGH |
| No persistent storage | Data lost on restart | HIGH |
| No authentication | Shared instance | HIGH |
| Single-threaded | No concurrency | MEDIUM |

### 12.2 Extraction Limitations

| Issue | Current State | Impact |
|-------|---------------|--------|
| Section detection | ~43% map to "UNK" | Missing context |
| Table extraction | Text only, no structure | Lost formatting |
| Complex layouts | May merge incorrectly | Accuracy loss |
| Non-English | English only | No i18n support |

### 12.3 Format-Specific Issues

| Format | Issue | Workaround |
|--------|-------|------------|
| Scanned PDF | No text extraction | Use text-based PDF |
| Encrypted PDF | Cannot process | Remove password |
| Password XLSX | Cannot process | Remove password |
| DOC (legacy) | Limited support | Convert to DOCX |

---

## 13. Security Considerations

### 13.1 Current Security Model

| Aspect | Status | Risk |
|--------|--------|------|
| Authentication | None | HIGH |
| Authorization | None | HIGH |
| File validation | Extension only | MEDIUM |
| Input sanitization | Basic | MEDIUM |
| Data encryption | None | MEDIUM |

### 13.2 Data Handling

- Files stored in `/tmp/` (ephemeral)
- No encryption at rest
- No audit logging
- In-memory storage (lost on restart)

### 13.3 Compliance Gaps

| Regulation | Status | Required For |
|------------|--------|--------------|
| FedRAMP | Not certified | Federal cloud |
| DoD IL4/IL5 | Not certified | Defense contracts |
| SOC 2 | Not certified | Enterprise |
| HIPAA | Not compliant | Healthcare RFPs |

---

## 14. Testing Strategy

### 14.1 Test Coverage

| Component | Coverage | Test Type |
|-----------|----------|-----------|
| Parser | ~70% | Unit tests |
| Extractor | ~60% | Unit tests |
| CTM Export | ~50% | Integration |
| API | ~40% | Manual |

### 14.2 Test RFPs Used

| RFP | Agency | Pages | Requirements | Notes |
|-----|--------|-------|--------------|-------|
| 75N96025R00004 | NIH | 400+ | 629 | Research services |
| FA880625RB003 | Air Force | 150+ | 496 | NOC support |
| 70LGLY26QSSB00001 | DHS/FLETC | 100+ | 245 | IT software |
| Illinois IDES | State | 200+ | 773 | UI Portal |

### 14.3 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=agents --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v
```

---

## 15. Operational Procedures

### 15.1 Deployment Checklist

```
[ ] Verify requirements.txt is current
[ ] Run local tests
[ ] Commit changes to GitHub
[ ] Verify Render auto-deploy triggers
[ ] Check Render logs for startup errors
[ ] Test /api/health endpoint
[ ] Test upload/process/export flow
```

### 15.2 Troubleshooting Guide

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| 502 Gateway Error | App crashed | Check Render logs |
| Slow processing | Large file | Wait or reduce file size |
| Missing requirements | Section detection | Check document format |
| Export fails | Memory pressure | Restart app |
| Empty CTM | Parsing failed | Check file format |

### 15.3 Log Monitoring

```bash
# Render logs
# Via dashboard: https://dashboard.render.com

# Local logs
uvicorn api.main:app --log-level debug
```

---

## Appendix A: Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.11 | Dec 2024 | Annotated outline export, semantic mapping |
| 2.10 | Nov 2024 | Agency detection, bundled attachment fixes |
| 2.9 | Nov 2024 | Best practices CTM, structure-aware extraction |
| 2.8 | Nov 2024 | Semantic extraction, 19-column CTM |
| 2.7 | Oct 2024 | Amendment processing, outline generation |
| 2.0 | Q2 2024 | Multi-document bundle processing |
| 1.0 | Q1 2024 | Initial release, keyword extraction |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **CTM** | Compliance Traceability Matrix |
| **UCF** | Uniform Contract Format (FAR 15.204-1) |
| **SOW** | Statement of Work |
| **PWS** | Performance Work Statement |
| **CDRL** | Contract Data Requirements List |
| **Section L** | Instructions to Offerors |
| **Section M** | Evaluation Factors for Award |
| **Section C** | Description/Specifications/Statement of Work |
| **SF30** | Standard Form 30 (Amendment form) |
| **SF33** | Standard Form 33 (Solicitation form) |

---

*End of As-Built Technical Design Document*
