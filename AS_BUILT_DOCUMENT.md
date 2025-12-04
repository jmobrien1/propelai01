# PropelAI As-Built Technical Document

**Version:** 2.10.0
**Last Updated:** December 2024
**Platform:** FastAPI (Python 3.11+) / React
**Deployment:** Render.com

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Supported RFP Types & Agencies](#3-supported-rfp-types--agencies)
4. [Document Processing Pipeline](#4-document-processing-pipeline)
5. [Requirement Extraction Engine](#5-requirement-extraction-engine)
6. [Compliance Matrix Generation](#6-compliance-matrix-generation)
7. [API Reference](#7-api-reference)
8. [Data Models](#8-data-models)
9. [Configuration](#9-configuration)
10. [Known Limitations & Shortcomings](#10-known-limitations--shortcomings)
11. [Future Roadmap](#11-future-roadmap)

---

## 1. Executive Summary

### 1.1 What is PropelAI?

PropelAI is an AI-powered RFP (Request for Proposal) intelligence platform designed for government contractors. It automates the extraction of requirements from complex federal RFP documents and generates Compliance Traceability Matrices (CTMs) following Shipley Process methodology.

### 1.2 Core Value Proposition

The system transforms the manual "RFP shredding" process—typically requiring 2-5 days of analyst work—into automated processing completed in minutes. PropelAI:

- Parses multi-document RFP bundles (PDFs, Word documents, Excel files)
- Identifies and extracts all contractual requirements
- Classifies requirements by type (Technical, Proposal Instructions, Evaluation Criteria)
- Generates evaluator-ready Compliance Traceability Matrices
- Tracks amendments and requirement changes across RFP versions

### 1.3 Version History

| Version | Release | Key Features |
|---------|---------|--------------|
| v1.0 | Initial | Keyword-based extraction (~40% accuracy) |
| v2.0 | Q2 2024 | Multi-document bundle processing |
| v2.8 | Q3 2024 | Semantic extraction with 19-column CTM |
| v2.9 | Q4 2024 | Structure-aware extraction, L/M/C separation |
| v2.10 | Dec 2024 | Agency detection, bundled attachment fixes |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PropelAI Platform                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   React UI  │───▶│  FastAPI    │───▶│  Processing Engine  │  │
│  │  (SPA)      │    │  Backend    │    │                     │  │
│  └─────────────┘    └─────────────┘    │  ┌───────────────┐  │  │
│                           │            │  │ Document      │  │  │
│                           │            │  │ Parser        │  │  │
│                           │            │  └───────────────┘  │  │
│                           │            │  ┌───────────────┐  │  │
│                           ▼            │  │ Structure     │  │  │
│                     ┌───────────┐      │  │ Analyzer      │  │  │
│                     │ In-Memory │      │  └───────────────┘  │  │
│                     │ Storage   │      │  ┌───────────────┐  │  │
│                     │ (RFPStore)│      │  │ Requirement   │  │  │
│                     └───────────┘      │  │ Extractor     │  │  │
│                                        │  └───────────────┘  │  │
│                                        │  ┌───────────────┐  │  │
│                                        │  │ CTM Exporter  │  │  │
│                                        │  └───────────────┘  │  │
│                                        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
/propelai01/
├── api/
│   └── main.py                    # FastAPI application (1,834 lines)
│
├── agents/
│   ├── enhanced_compliance/       # Core extraction engine
│   │   ├── agent.py               # Orchestrator
│   │   ├── parser.py              # Multi-format document parser
│   │   ├── document_structure.py  # RFP structure analyzer
│   │   ├── section_aware_extractor.py  # Best practices extractor
│   │   ├── best_practices_ctm.py  # CTM Excel generator
│   │   ├── semantic_extractor.py  # Semantic classification
│   │   ├── bundle_detector.py     # Document type classifier
│   │   ├── resolver.py            # Cross-reference resolution
│   │   ├── amendment_processor.py # Amendment tracking
│   │   └── models.py              # Data structures
│   │
│   ├── strategy_agent.py          # Win theme development
│   ├── drafting_agent.py          # Content generation
│   └── red_team_agent.py          # Quality evaluation
│
├── core/
│   ├── config.py                  # Configuration management
│   ├── state.py                   # Proposal state schema
│   └── orchestrator.py            # LangGraph workflow
│
├── tools/
│   └── document_tools.py          # Document utilities
│
├── web/
│   └── index.html                 # React single-page application
│
├── cli.py                         # Command-line interface
├── requirements.txt               # Python dependencies
└── Procfile                       # Render deployment config
```

### 2.3 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React 18 (CDN) | Single-page web application |
| API | FastAPI | REST API with async support |
| Document Processing | PyMuPDF, python-docx, openpyxl | PDF/DOCX/XLSX parsing |
| Storage | In-memory (dict) | RFP project storage |
| File Storage | `/tmp/propelai_uploads/` | Temporary file storage |
| Deployment | Render.com | Cloud hosting (free tier) |

---

## 3. Supported RFP Types & Agencies

### 3.1 Agency Detection

PropelAI automatically detects the issuing agency from solicitation number format:

| Agency | Solicitation Format | Example | Detection Pattern |
|--------|---------------------|---------|-------------------|
| **NIH** | 75N + 5 digits + letter + 5 digits | `75N96025R00004` | `^75N` |
| **Navy** | N + 5 digits + 2 digits + R/Q + digits | `N0017826R30020003` | `^N\d{5}` |
| **Army** | W + contract code | `W912HQ-24-R-0001` | `^W` |
| **Air Force** | FA + digits | `FA8730-24-R-0001` | `^FA` |
| **GSA** | GS- prefix | `GS-00F-12345` | `^GS` |
| **VA** | 36C prefix | `36C25724R0001` | `^36C` |
| **DHS** | 70CDCR or HSC prefix | `70CDCR24R00001` | `^70\|^HSC` |
| **HHS/CMS** | 75 prefix (non-NIH) | `75P00124R00001` | `^75` |

### 3.2 Document Format Support

#### 3.2.1 UCF (Uniform Contract Format) RFPs

Standard federal RFPs following FAR 15.204-1:

| Section | Name | Extraction Focus |
|---------|------|------------------|
| **A** | Solicitation/Contract Form | SF33, SF1449 identification |
| **B** | Supplies/Services and Prices | Contract value, CLINs |
| **C** | Description/SOW/PWS | Technical requirements |
| **D** | Packaging and Marking | Delivery requirements |
| **E** | Inspection and Acceptance | QA requirements |
| **F** | Deliveries or Performance | Schedule requirements |
| **G** | Contract Administration | Admin requirements |
| **H** | Special Contract Requirements | Custom clauses |
| **I** | Contract Clauses | FAR/DFARS clauses |
| **J** | List of Attachments | Attachment index |
| **K** | Representations/Certifications | Certifications |
| **L** | Instructions to Offerors | **Submission format** |
| **M** | Evaluation Factors | **Scoring criteria** |

#### 3.2.2 Non-UCF RFPs

PropelAI also handles non-standard formats:

| Type | Characteristics | Handling |
|------|-----------------|----------|
| **GSA Schedule** | Task order format, no Section L | Section M treated as submission instructions |
| **BPA Call** | Blanket Purchase Agreement orders | PWS/SOW as primary requirement source |
| **IDIQ Task Orders** | Simplified format | Technical requirements from SOW |
| **Commercial Item** | FAR Part 12 format | Streamlined extraction |

### 3.3 NIH-Specific Features

NIH RFPs have unique characteristics that PropelAI handles:

| Feature | Description | Detection |
|---------|-------------|-----------|
| **Research Outlines** | RO-I, RO-II, RO-III structure | `RO\s*[IVX]+` pattern |
| **Attachment 2 = SOW** | NIH convention for SOW location | Filename pattern matching |
| **Institute Codes** | NIEHS, NCI, NIAID, NICHD | Agency identifier patterns |
| **75N Format** | Standard NIH solicitation number | `75N\d{5}[A-Z]\d{5}` |

### 3.4 DoD-Specific Features

Department of Defense RFPs include additional requirements:

| Feature | Description | Detection |
|---------|-------------|-----------|
| **DFARS Clauses** | Defense-specific FAR supplements | `DFARS\s*\d+\.\d+` |
| **DD254** | Security classification requirements | Filename detection |
| **CDRL** | Contract Data Requirements List | Document type classification |
| **J-Attachments** | Standard attachment format (J.1, J.2, etc.) | `J[-_.]?\d+` pattern |
| **SF33** | Standard Form 33 solicitation | Main document identifier |

### 3.5 Supported File Formats

| Format | Extension | Parser | Notes |
|--------|-----------|--------|-------|
| PDF | `.pdf` | PyMuPDF | Text-based only (no OCR) |
| Word | `.docx` | python-docx | Full support |
| Word (Legacy) | `.doc` | python-docx | Converted to DOCX |
| Excel | `.xlsx` | openpyxl | Full support |
| Excel (Legacy) | `.xls` | openpyxl | Converted to XLSX |

---

## 4. Document Processing Pipeline

### 4.1 Processing Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Upload     │────▶│    Parse     │────▶│   Analyze    │
│   Files      │     │   Documents  │     │   Structure  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Export     │◀────│   Generate   │◀────│   Extract    │
│   CTM        │     │   Matrix     │     │ Requirements │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 4.2 Stage 1: Document Upload

**Endpoint:** `POST /api/rfp/{rfp_id}/upload`

**Process:**
1. Receive multipart form data with files
2. Validate file extensions (PDF, DOCX, XLSX, DOC, XLS)
3. Check file size limits (15MB per file, 50MB total)
4. Store files in `/tmp/propelai_uploads/{rfp_id}/`
5. Return upload confirmation with file list

**Limits:**
- Maximum single file: 15 MB
- Maximum total upload: 50 MB
- Maximum text length: 2,000,000 characters per document

### 4.3 Stage 2: Document Parsing

**Component:** `MultiFormatParser` (`parser.py`)

**Process:**
1. Detect file format from extension
2. Extract text using format-specific parser
3. Preserve page structure and boundaries
4. Identify section headers and boundaries
5. Return `ParsedDocument` with metadata

**Output Structure:**
```python
@dataclass
class ParsedDocument:
    filename: str
    text: str
    pages: List[str]
    page_count: int
    document_type: DocumentType
    sections: Dict[str, str]
    quality_score: float  # 0.0-1.0
```

### 4.4 Stage 3: Structure Analysis

**Component:** `RFPStructureParser` (`document_structure.py`)

**Process:**
1. Identify UCF sections (A through M)
2. Detect section boundaries (start/end page, character offsets)
3. Locate SOW/PWS (Section C or attachment)
4. Parse subsection structure (L.4.B.2, C.3.1.a)
5. Catalog attachments and their types

**Output Structure:**
```python
@dataclass
class DocumentStructure:
    solicitation_number: str
    title: str
    sections: Dict[UCFSection, SectionBoundary]
    attachments: Dict[str, AttachmentInfo]
    sow_location: Optional[str]  # "SECTION_C" or "Attachment 2"
    pws_location: Optional[str]
    has_section_l: bool
    has_section_m: bool
```

### 4.5 Stage 4: Requirement Extraction

**Component:** `SectionAwareExtractor` (`section_aware_extractor.py`)

**Process:**
1. Process each UCF section independently
2. Extract complete requirement paragraphs (never summarize)
3. Preserve RFP's own reference numbers (L.4.B.2)
4. Classify binding level (Mandatory, Highly Desirable, Desirable)
5. Detect cross-references between sections

**Extraction Categories:**

| Category | Source Sections | Focus |
|----------|-----------------|-------|
| `L_COMPLIANCE` | Section L | Submission format, page limits, fonts |
| `TECHNICAL` | Section C, PWS, SOW, Attachments | Performance requirements |
| `EVALUATION` | Section M | Scoring criteria, evaluation factors |
| `ADMINISTRATIVE` | Sections B, F, G, H | Contract administration |
| `ATTACHMENT` | Section J attachments | Technical specifications |

### 4.6 Stage 5: Matrix Generation

**Component:** `BestPracticesCTMExporter` (`best_practices_ctm.py`)

**Output:** Multi-sheet Excel workbook with:

| Sheet | Purpose | Audience |
|-------|---------|----------|
| **Cover** | Summary statistics, navigation | All |
| **Section L Compliance** | Submission format checklist | Internal |
| **Technical Requirements** | C/PWS/SOW requirements | Evaluators |
| **Section M Alignment** | Evaluation factor mapping | Evaluators |
| **All Requirements** | Complete reference list | Internal |
| **Cross-References** | L→M→C linkages | Internal |

---

## 5. Requirement Extraction Engine

### 5.1 Extraction Methods

PropelAI supports three extraction modes:

| Mode | Endpoint | Accuracy | Speed | Recommended For |
|------|----------|----------|-------|-----------------|
| **Legacy** | `/process` | ~40% | Fast | Testing only |
| **Semantic** | `/process-semantic` | ~70% | Medium | Simple RFPs |
| **Best Practices** | `/process-best-practices` | ~85% | Slower | Production |

### 5.2 Binding Level Detection

Requirements are classified by binding strength:

| Level | Keywords | Priority | Color |
|-------|----------|----------|-------|
| **Mandatory** | shall, must, required, will [verb] | HIGH | Orange |
| **Highly Desirable** | should, expected | MEDIUM | Yellow |
| **Desirable** | may, can, encouraged | LOW | Green |
| **Informational** | No binding language | INFO | Gray |

### 5.3 Requirement Classification

**Semantic Types:**

| Type | Description | Example Keywords |
|------|-------------|------------------|
| `PERFORMANCE` | What contractor must DO | "contractor shall provide" |
| `PROPOSAL_INSTRUCTION` | What to WRITE in proposal | "offeror shall submit" |
| `EVALUATION_CRITERION` | How proposal is SCORED | "will be evaluated" |
| `DELIVERABLE` | Tangible outputs required | "shall deliver" |
| `QUALIFICATION` | Experience/capability requirements | "must demonstrate" |
| `COMPLIANCE` | Regulatory requirements | "FAR", "DFARS" |
| `FORMAT` | Document formatting requirements | "page limit", "font" |

### 5.4 Cross-Reference Resolution

PropelAI tracks relationships between requirements:

| Relationship | Example | Detection |
|--------------|---------|-----------|
| **L→M** | "Proposal section L.4 will be evaluated under M.2" | Pattern matching |
| **M→C** | "Technical approach per PWS Section 3.1" | Pattern matching |
| **Attachment→Main** | "See Attachment J.1 for specifications" | Explicit references |
| **Amendment→Original** | "Paragraph L.5.2 is hereby deleted" | Amendment processor |

### 5.5 Quality Filters

Requirements are filtered to remove noise:

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Minimum length | 40 characters | Remove fragments |
| Maximum length | 2,000 characters | Remove full sections |
| Minimum words | 10 words | Remove headers |
| Alpha ratio | 50% alphabetic | Remove tables/numbers |
| TOC detection | Pattern matching | Remove table of contents |
| Boilerplate detection | 67 patterns | Remove standard clauses |

---

## 6. Compliance Matrix Generation

### 6.1 Excel Output Structure

**Filename Format:** `{SolicitationNumber}_{Agency}_ComplianceMatrix.xlsx`

Example: `75N96025R00004_NIH_ComplianceMatrix.xlsx`

### 6.2 Section L Compliance Matrix

| Column | Description | Width |
|--------|-------------|-------|
| RFP Reference | L.4.B.2, L.5.1.a | 15 |
| Requirement Text | Full verbatim text | 60 |
| Source Page | Page number | 10 |
| Binding Level | Mandatory/Desirable | 15 |
| Volume/Section | Proposal location | 20 |
| Compliance Status | Compliant/Non-Compliant/Partial | 15 |
| Compliance Response | How we comply | 40 |
| Evidence/Notes | Supporting documentation | 30 |

### 6.3 Technical Requirements Matrix

| Column | Description | Width |
|--------|-------------|-------|
| Req ID | RFP reference or generated ID | 15 |
| Requirement Text | Full verbatim text | 60 |
| Source | Document filename | 20 |
| Page | Page number | 8 |
| Binding | Mandatory/Desirable | 12 |
| Proposal Section | Where addressed | 20 |
| Compliance Status | Dropdown | 15 |
| How We Meet This | Technical approach | 40 |
| Evidence Required | Proof points | 25 |
| Owner | Responsible person | 15 |

### 6.4 Section M Alignment Matrix

| Column | Description | Width |
|--------|-------------|-------|
| Evaluation Factor | M.2.a, M.3.1 | 15 |
| Criterion Text | Full evaluation text | 50 |
| Page | Page number | 8 |
| Weight | Importance indicator | 10 |
| Proposal Location | Section mapping | 20 |
| Our Strength | Discriminating feature | 40 |
| Discriminator | Yes/No/Partial | 12 |
| Proof Points | Evidence | 30 |
| Risk/Gap | Identified weaknesses | 25 |

---

## 7. API Reference

### 7.1 Core Endpoints

#### Create RFP Project
```http
POST /api/rfp
Content-Type: application/json

{
  "name": "NIH Cancer Research RFP",
  "description": "75N96025R00004"
}

Response: { "id": "RFP-a1b2c3d4", "name": "...", "created_at": "..." }
```

#### Upload Documents
```http
POST /api/rfp/{rfp_id}/upload
Content-Type: multipart/form-data

files: [file1.pdf, file2.docx, ...]

Response: { "uploaded": ["file1.pdf", "file2.docx"], "count": 2 }
```

#### Process RFP (Best Practices)
```http
POST /api/rfp/{rfp_id}/process-best-practices

Response: { "status": "processing", "message": "Processing started" }
```

#### Check Status
```http
GET /api/rfp/{rfp_id}/status

Response: {
  "status": "completed",
  "progress": 100,
  "message": "Extraction complete",
  "stats": {
    "total": 156,
    "section_l": 23,
    "technical": 98,
    "evaluation": 35
  }
}
```

#### Export Compliance Matrix
```http
GET /api/rfp/{rfp_id}/export?format=xlsx

Response: [Excel file download]
```

### 7.2 Additional Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/rfp` | GET | List all RFP projects |
| `/api/rfp/{id}` | GET | Get RFP details |
| `/api/rfp/{id}` | DELETE | Delete RFP project |
| `/api/rfp/{id}/requirements` | GET | Get extracted requirements |
| `/api/rfp/{id}/requirements/{req_id}` | GET | Get single requirement |
| `/api/rfp/{id}/stats` | GET | Get detailed statistics |
| `/api/rfp/{id}/amendments` | POST | Upload amendment |
| `/api/rfp/{id}/amendments` | GET | Get amendment history |
| `/api/rfp/{id}/outline` | POST | Generate proposal outline |
| `/api/rfp/{id}/outline/export` | GET | Export outline as Word |

---

## 8. Data Models

### 8.1 Core Models

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
    """Requirement classification categories"""
    SECTION_L_COMPLIANCE = "L_COMPLIANCE"
    TECHNICAL_REQUIREMENT = "TECHNICAL"
    EVALUATION_FACTOR = "EVALUATION"
    ADMINISTRATIVE = "ADMINISTRATIVE"
    ATTACHMENT_REQUIREMENT = "ATTACHMENT"

class BindingLevel(Enum):
    """How binding is this requirement?"""
    MANDATORY = "Mandatory"
    HIGHLY_DESIRABLE = "Highly Desirable"
    DESIRABLE = "Desirable"
    INFORMATIONAL = "Informational"

@dataclass
class StructuredRequirement:
    rfp_reference: str           # RFP's own ID (L.4.B.2)
    generated_id: str            # Backup ID (TW-L-001)
    full_text: str               # Complete requirement text
    category: RequirementCategory
    binding_level: BindingLevel
    binding_keyword: str         # "shall", "must", "should"
    source_section: UCFSection
    source_subsection: Optional[str]
    page_number: int
    source_document: str
    references_to: List[str]     # Cross-references
```

### 8.2 Document Type Classification

```python
class DocumentType(Enum):
    """Classification of documents in an RFP bundle"""
    MAIN_SOLICITATION = "main_solicitation"
    STATEMENT_OF_WORK = "statement_of_work"
    RESEARCH_OUTLINE = "research_outline"    # NIH-specific
    CDRL = "cdrl"                             # DoD Contract Data Requirements
    AMENDMENT = "amendment"
    ATTACHMENT = "attachment"
    BUDGET_TEMPLATE = "budget_template"
    SECURITY = "security"                    # DD254
    FORM = "form"                            # Standard forms
    QA_RESPONSE = "qa_response"
```

---

## 9. Configuration

### 9.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Yes | 8000 | Server port (set by Render) |
| `ANTHROPIC_API_KEY` | No | None | Claude API for enhanced extraction |
| `GOOGLE_API_KEY` | No | None | Gemini API integration |

### 9.2 Hardcoded Limits

| Limit | Value | Location |
|-------|-------|----------|
| Max single file | 15 MB | `api/main.py:796` |
| Max total upload | 50 MB | `api/main.py:795` |
| Max text length | 2,000,000 chars | `api/main.py:833` |
| Min requirement length | 40 chars | `section_aware_extractor.py:167` |
| Max requirement length | 2,000 chars | `section_aware_extractor.py:168` |
| Excel sheet name max | 31 chars | `excel_parser.py:645` |

### 9.3 Deployment Configuration

**Procfile (Render.com):**
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

**Local Development:**
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 10. Known Limitations & Shortcomings

### 10.1 Critical Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **No OCR Support** | Scanned PDFs produce no text | Use text-based PDFs only |
| **No Persistent Storage** | Data lost on server restart | Export CTMs immediately |
| **No Authentication** | All users share instance | Single-user deployment |
| **In-Memory Storage** | Limited scalability | Restart if memory issues |

### 10.2 Document Processing Limitations

| Issue | Details | Severity |
|-------|---------|----------|
| **Table Extraction** | Tables converted to text, structure lost | HIGH |
| **Image Content** | Images in PDFs not processed | MEDIUM |
| **Complex Formatting** | Multi-column layouts may merge incorrectly | MEDIUM |
| **Encrypted PDFs** | Cannot process password-protected files | HIGH |
| **Non-English** | English only | HIGH |

### 10.3 Extraction Accuracy Issues

| Issue | Current State | Impact |
|-------|---------------|--------|
| **Section Detection** | ~43% map to "UNK" section | Requirements lack context |
| **Priority Scoring** | No LOW priority assigned | All HIGH/MEDIUM skews review |
| **Subsection Detection** | L.4.B.2 patterns not always found | Missing RFP references |
| **GSA/BPA Format** | Non-UCF RFPs need special handling | May miss instructions |

### 10.4 Agency-Specific Gaps

| Agency | Known Issues |
|--------|--------------|
| **NIH** | Research Outline detection needs improvement |
| **DoD** | CDRL table extraction incomplete |
| **GSA** | Task order format varies significantly |
| **State/Local** | Not designed for state/local government RFPs |

### 10.5 Performance Constraints

| Constraint | Limit | Symptom |
|------------|-------|---------|
| Large files | >15MB per file | Slow processing, potential timeout |
| Many files | >20 files per RFP | Memory pressure |
| Long documents | >2M characters | Text truncation |
| Concurrent users | Not supported | Data collision risk |

### 10.6 API Design Issues

| Issue | Description | Risk |
|-------|-------------|------|
| **No Request Locking** | Concurrent requests may corrupt state | HIGH |
| **No Rate Limiting** | Vulnerable to overload | MEDIUM |
| **Error Message Truncation** | Debug info lost | LOW |
| **No Request Validation Middleware** | Large uploads consume resources | MEDIUM |

### 10.7 Missing Features

| Feature | Status | Priority |
|---------|--------|----------|
| User authentication | Not implemented | HIGH |
| Persistent database | Not implemented | HIGH |
| Team collaboration | Not implemented | MEDIUM |
| Version control for CTMs | Not implemented | MEDIUM |
| Automated testing | Minimal coverage | MEDIUM |
| API documentation (Swagger) | Not exposed | LOW |

---

## 11. Future Roadmap

### 11.1 Short-Term Improvements

- [ ] Improve section detection (ML-based classification)
- [ ] Add LOW priority scoring tier
- [ ] Better subsection linking (L.4.B.2 → specific content)
- [ ] Table structure extraction
- [ ] Enhanced GSA/BPA support

### 11.2 Medium-Term Features

- [ ] Claude API integration for semantic enhancement
- [ ] Requirement consolidation (group related requirements)
- [ ] Win theme integration with Section M
- [ ] Team collaboration features
- [ ] Persistent database storage (PostgreSQL)

### 11.3 Long-Term Vision

- [ ] Full proposal generation (draft compliant responses)
- [ ] Past performance auto-matching
- [ ] Price volume integration
- [ ] Competitive intelligence analysis
- [ ] Multi-user collaboration platform
- [ ] OCR support for scanned documents
- [ ] FedRAMP compliance for government deployment

---

## Appendix A: File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `api/main.py` | 1,834 | FastAPI backend |
| `agents/enhanced_compliance/agent.py` | 516 | Orchestrator |
| `agents/enhanced_compliance/parser.py` | 679 | Document parsing |
| `agents/enhanced_compliance/document_structure.py` | 724 | Structure analysis |
| `agents/enhanced_compliance/section_aware_extractor.py` | 541 | Requirement extraction |
| `agents/enhanced_compliance/best_practices_ctm.py` | 628 | CTM generation |
| `agents/enhanced_compliance/semantic_extractor.py` | 1,180 | Semantic classification |
| `agents/enhanced_compliance/extractor.py` | 1,729 | Legacy extraction |
| `agents/enhanced_compliance/amendment_processor.py` | 805 | Amendment tracking |
| `core/config.py` | 237 | Configuration |
| `core/state.py` | 257 | State schema |
| `core/orchestrator.py` | 331 | Workflow orchestration |
| `web/index.html` | ~2,000 | React UI |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **CTM** | Compliance Traceability Matrix - Excel document mapping requirements to proposal sections |
| **UCF** | Uniform Contract Format - Standard federal RFP structure (Sections A-M) |
| **SOW** | Statement of Work - Technical requirements document |
| **PWS** | Performance Work Statement - DoD variant of SOW |
| **FAR** | Federal Acquisition Regulation - Government contracting rules |
| **DFARS** | Defense FAR Supplement - DoD-specific acquisition rules |
| **SF33** | Standard Form 33 - Solicitation/Contract form |
| **SF1449** | Standard Form 1449 - Commercial items contract form |
| **CLIN** | Contract Line Item Number - Pricing breakdown |
| **CDRL** | Contract Data Requirements List - DoD deliverables specification |

---

*Document generated by PropelAI v2.10.0*
