# PropelAI As-Built Document v3.3

**Version:** 3.3 (Strict Constructionist Architecture)
**Date:** December 19, 2025
**Classification:** Technical Reference Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Core Components](#4-core-components)
5. [Enhanced Compliance Module](#5-enhanced-compliance-module)
6. [Smart Outline Generator v3.3](#6-smart-outline-generator-v33)
7. [API Reference](#7-api-reference)
8. [Data Models](#8-data-models)
9. [Processing Pipelines](#9-processing-pipelines)
10. [Frontend Application](#10-frontend-application)
11. [Database Schema](#11-database-schema)
12. [Deployment Configuration](#12-deployment-configuration)
13. [Version History](#13-version-history)
14. [Known Issues & Limitations](#14-known-issues--limitations)

---

## 1. Executive Summary

### 1.1 Purpose

PropelAI is an autonomous RFP (Request for Proposal) analysis and proposal development platform designed for government contractors. It automates the extraction of requirements from complex solicitation documents, generates compliance matrices, and produces annotated proposal outlines.

### 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| **Document Parsing** | Multi-format support (PDF, DOCX, XLSX) with OCR fallback |
| **Requirement Extraction** | Resilient extraction pipeline with confidence scoring |
| **Compliance Matrix** | Automated CTM generation with Section L/M/C categorization |
| **Proposal Outline** | Smart outline generation following Section L structure |
| **Amendment Tracking** | Lifecycle management for requirement changes |
| **Company Library** | Organizational knowledge base for reuse |
| **Quality Metrics** | Real-time extraction quality assessment |

### 1.3 Current Version Highlights (v3.3)

- **Strict Constructionist Architecture**: Eliminates phantom volume hallucination
- **Hierarchy of Authority**: 4-level priority system for volume detection
- **Evidence Tracking**: All volumes tagged with source and confidence level
- **No UCF Defaults**: Task Orders, RFEs, RFQs use explicit structure only

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PROPELAI PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Frontend   │    │   REST API   │    │   Agents     │               │
│  │   (React)    │───▶│   (FastAPI)  │───▶│  (LangGraph) │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         │                   ▼                   ▼                        │
│         │            ┌──────────────┐    ┌──────────────┐               │
│         │            │   Storage    │    │    LLM       │               │
│         │            │  (PostgreSQL)│    │  (Gemini/    │               │
│         │            │              │    │   Claude)    │               │
│         │            └──────────────┘    └──────────────┘               │
│         │                                                                │
│         └────────────────────────────────────────────────────────────── │
│                              Real-time Status Updates                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
User Upload → Document Parser → Requirement Extractor → Categorization
                                        │
                                        ▼
                              Compliance Matrix Builder
                                        │
                                        ▼
                              Smart Outline Generator
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              Excel Export      Word Outline Export    JSON API Response
```

### 2.3 Directory Structure

```
propelai01/
├── api/
│   └── main.py                    # FastAPI application (2,337 lines)
├── agents/
│   ├── compliance_agent.py        # RFP analysis agent
│   ├── drafting_agent.py          # Proposal writing agent
│   ├── strategy_agent.py          # Win theme development
│   ├── red_team_agent.py          # Quality evaluation
│   └── enhanced_compliance/       # Core extraction module (28 files)
│       ├── __init__.py            # Module exports & availability flags
│       ├── smart_outline_generator.py  # v3.3 Strict Constructionist
│       ├── resilient_extractor.py      # v3.0 Extract-first pipeline
│       ├── parser.py              # Multi-format document parser
│       ├── models.py              # Core data models
│       └── [24 more modules]
├── core/
│   ├── config.py                  # Configuration management
│   ├── orchestrator.py            # LangGraph workflow engine
│   └── state.py                   # Global proposal state schema
├── web/
│   └── index.html                 # React SPA frontend
├── tests/
│   ├── test_agents.py
│   ├── test_accuracy.py
│   ├── test_resilient_extraction.py
│   └── test_structure_determinism.py  # v3.3 tests
├── docs/
│   └── AS_BUILT_v3.3.md           # This document
├── docker-compose.yml
├── requirements.txt
└── init.sql
```

---

## 3. Technology Stack

### 3.1 Backend

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Web Framework | FastAPI | 0.100.0+ | REST API, async support |
| ASGI Server | Uvicorn | 0.23.0+ | Production server |
| Validation | Pydantic | 2.0+ | Data validation, serialization |
| Workflow | LangGraph | Latest | Stateful agent orchestration |
| PDF Parsing | pypdf | 3.0+ | PDF text extraction |
| Word Parsing | python-docx | 0.8.11+ | DOCX processing |
| Excel | openpyxl | 3.0.10+ | XLSX read/write |
| HTTP Client | httpx | 0.24.0+ | Async HTTP requests |

### 3.2 LLM Providers (Cascading)

| Priority | Provider | Model | Use Case |
|----------|----------|-------|----------|
| 1 | Google | Gemini 1.5 Flash/Pro | Primary extraction |
| 2 | Anthropic | Claude 3.5 Sonnet | Complex analysis |
| 3 | OpenAI | GPT-4 Turbo | Fallback |

### 3.3 Data Storage

| Component | Technology | Purpose |
|-----------|------------|---------|
| Primary DB | PostgreSQL 15+ | State checkpointing, audit logs |
| Vector Store | Chroma/Pinecone | Semantic search, embeddings |
| File Storage | Local filesystem | Uploaded documents |

### 3.4 Frontend

| Technology | Purpose |
|------------|---------|
| React 18 | UI framework (via CDN) |
| Tailwind CSS | Styling |
| Lucide Icons | Iconography |
| Babel | JSX compilation |

---

## 4. Core Components

### 4.1 Orchestrator (`core/orchestrator.py`)

The `ProposalOrchestrator` class implements a supervisor pattern using LangGraph for stateful workflow execution.

**Workflow Phases:**
```
INTAKE → SHRED → STRATEGY → OUTLINE → DRAFTING → REVIEW → FINALIZE → SUBMITTED
```

**Agent Nodes:**
- `supervisor` - Task routing based on current phase
- `compliance_agent` - RFP extraction and categorization
- `strategy_agent` - Win theme development
- `drafting_agent` - Content generation
- `research_agent` - Evidence gathering
- `red_team_agent` - Quality evaluation
- `human_review` - Human-in-the-loop checkpoint

### 4.2 State Management (`core/state.py`)

**`ProposalState` TypedDict:**

```python
ProposalState = TypedDict('ProposalState', {
    # Identity
    'proposal_id': str,
    'client_name': str,
    'solicitation_number': str,
    'due_date': str,

    # RFP Data
    'rfp_raw_text': str,
    'rfp_file_paths': List[str],
    'rfp_metadata': Dict,

    # The "Iron Triangle"
    'requirements': List[Dict],           # Section C (SOW)
    'instructions': List[Dict],           # Section L
    'evaluation_criteria': List[Dict],    # Section M

    # Graph Structure
    'requirements_graph': Dict,           # Dependency mapping

    # Generated Content
    'annotated_outline': Dict,
    'draft_sections': List[Dict],

    # Quality
    'red_team_feedback': List[Dict],
    'proposal_quality_score': float,

    # Audit
    'agent_trace_log': List[Dict],
    'human_feedback': List[Dict],
})
```

### 4.3 Configuration (`core/config.py`)

**Environment Variables:**

```bash
# Core
PROPELAI_ENV=development|staging|production
API_HOST=0.0.0.0
API_PORT=8000

# LLM Providers
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=propelai
POSTGRES_USER=propelai
POSTGRES_PASSWORD=...

# Vector Store
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=propelai-proposals

# Features
ENABLE_OCR=true
ENABLE_AUDIT_LOG=true
```

---

## 5. Enhanced Compliance Module

### 5.1 Module Overview

Located at `/agents/enhanced_compliance/`, this module contains 28 Python files totaling ~20,000 lines of code.

### 5.2 Submodule Inventory

| Module | Lines | Purpose |
|--------|-------|---------|
| `smart_outline_generator.py` | 1,713 | v3.3 Strict Constructionist outline generation |
| `resilient_extractor.py` | 800+ | v3.0 Extract-first pipeline |
| `semantic_extractor.py` | 1,016 | LLM-enhanced extraction |
| `section_aware_extractor.py` | 600+ | Section L/M/C detection |
| `best_practices_ctm.py` | 500+ | Best practices CTM export |
| `universal_extractor.py` | 662 | Unified extraction layer |
| `parser.py` | 400+ | Multi-format document parsing |
| `document_structure.py` | 500+ | UCF section boundary detection |
| `bundle_detector.py` | 300+ | RFP bundle structure analysis |
| `excel_export.py` | 400+ | Excel CTM generation |
| `annotated_outline_exporter.py` | 300+ | Word outline export |
| `company_library.py` | 500+ | Organizational knowledge base |
| `amendment_processor.py` | 400+ | Amendment lifecycle tracking |
| `extraction_validator.py` | 300+ | Quality validation |
| `models.py` | 300+ | Core data models |
| `ctm_data_models.py` | 400+ | Compliance matrix models |
| `extraction_models.py` | 200+ | Extraction result models |
| `document_types.py` | 100+ | Document classification (v3.2) |

### 5.3 Availability Flags

The module uses graceful degradation with availability flags:

```python
# From __init__.py
MODELS_AVAILABLE = True/False
SEMANTIC_AVAILABLE = True/False
OUTLINE_GENERATOR_AVAILABLE = True/False
COMPANY_LIBRARY_AVAILABLE = True/False
DOCUMENT_STRUCTURE_AVAILABLE = True/False
BEST_PRACTICES_AVAILABLE = True/False
RESILIENT_EXTRACTION_AVAILABLE = True/False
```

### 5.4 Extraction Engine Hierarchy

```
v3.0+ Processing Decision Tree:

if RESILIENT_EXTRACTION_AVAILABLE:
    → ResilientExtractor (extract-first, classify-later)
elif BEST_PRACTICES_AVAILABLE:
    → SectionAwareExtractor + BestPracticesCTMExporter
elif SEMANTIC_AVAILABLE:
    → SemanticRequirementExtractor + SemanticCTMExporter
else:
    → EnhancedComplianceAgent (legacy)
```

---

## 6. Smart Outline Generator v3.3

### 6.1 Architecture

The Smart Outline Generator underwent a major refactor in v3.3 to implement "Strict Constructionist" architecture, eliminating phantom volume hallucination.

### 6.2 New Components

#### 6.2.1 Evidence Tracking Enums

```python
class EvidenceSource(Enum):
    TABLE = "table"           # Explicit table in Section L (highest)
    LIST = "list"             # Numbered list "Volume 1:", "Volume 2:"
    FACTOR = "factor"         # Mapped from Section M evaluation factors
    FILENAME = "filename"     # Inferred from uploaded file names
    KEYWORD = "keyword"       # Keyword match (low confidence)
    DEFAULT = "default"       # No evidence found, using fallback

class ConfidenceLevel(Enum):
    HIGH = "high"             # Explicit table or numbered list
    MEDIUM = "medium"         # Factor mapping or filename inference
    LOW = "low"               # Keyword match or default
```

#### 6.2.2 ProposalStructureParser Class

```python
class ProposalStructureParser:
    """
    v3.3: Strict Constructionist parser for proposal volume structure.

    HIERARCHY OF AUTHORITY:
    1. PRIMARY: Explicit tables in Section L with Volume/Pages columns
    2. SECONDARY: Numbered lists "Volume 1:", "Volume 2:" in Section L
    3. TERTIARY: Evaluation Factors from Section M mapped to volumes
    4. QUATERNARY: File names of uploaded documents
    5. FALLBACK: Single generic volume (NOT 4-volume UCF default)
    """

    # RFP types that should NEVER use UCF defaults
    NO_UCF_TYPES = {
        "task_order", "delivery_order", "idiq_order",
        "request_for_estimate", "rfe", "rac",
        "request_for_quote", "rfq", "bpa_call"
    }

    def parse_section_l_structure(self, text) -> Tuple[List[Dict], EvidenceSource, ConfidenceLevel]
    def parse_section_m_factors(self, text) -> Tuple[List[Dict], EvidenceSource, ConfidenceLevel]
    def infer_from_filenames(self, filenames) -> Tuple[List[Dict], EvidenceSource, ConfidenceLevel]
    def detect_rfp_type(self, text) -> str
    def should_use_ucf_defaults(self, rfp_type) -> bool
```

#### 6.2.3 Updated ProposalVolume

```python
@dataclass
class ProposalVolume:
    id: str
    name: str
    volume_type: VolumeType
    page_limit: Optional[int] = None
    sections: List[ProposalSection] = field(default_factory=list)
    eval_factors: List[str] = field(default_factory=list)
    order: int = 0
    # v3.3: Evidence tracking
    evidence_source: EvidenceSource = EvidenceSource.DEFAULT
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
```

### 6.3 Hierarchy of Authority

```
┌─────────────────────────────────────────────────────────────────┐
│                  HIERARCHY OF AUTHORITY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: EXPLICIT VOLUME TABLES (Section L)                    │
│  ─────────────────────────────────────────────                  │
│  Pattern: "Volume 1: Technical... 8 pages"                      │
│  Pattern: Table with columns [Volume | Description | Pages]     │
│  Confidence: HIGH                                                │
│  If found → THIS IS LAW. Stop searching.                        │
│                                                                  │
│  Level 2: EVALUATION FACTORS (Section M)                        │
│  ─────────────────────────────────────────                      │
│  Pattern: "Factor 1: Technical Approach"                        │
│  Confidence: MEDIUM                                              │
│  If Section L silent → Map Factors to Volumes                   │
│                                                                  │
│  Level 3: FILE NAME INFERENCE                                   │
│  ────────────────────────────                                   │
│  Pattern: "Cost_Volume.pdf", "Technical_Proposal.docx"          │
│  Confidence: MEDIUM                                              │
│  If L and M silent → Infer from what was uploaded               │
│                                                                  │
│  Level 4: MINIMAL FALLBACK                                      │
│  ─────────────────────────                                      │
│  Single "Proposal Response" volume                              │
│  Confidence: LOW                                                 │
│  NEVER default to 4-volume UCF structure                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Key Method: _extract_volumes

```python
def _extract_volumes(
    self,
    section_l: List[Dict],
    section_m: List[Dict],
    rfp_format: str,
    filenames: List[str] = None
) -> Tuple[List[ProposalVolume], List[str]]:
    """
    v3.3: Extract proposal volumes using STRICT CONSTRUCTIONIST approach.

    Returns:
        (volumes, warnings)
    """
    # Level 1: Try Section L explicit structure
    vol_dicts, evidence_source, confidence = self.structure_parser.parse_section_l_structure(section_l_text)
    if vol_dicts:
        return self._convert_vol_dicts_to_volumes(vol_dicts, evidence_source, confidence), []

    # Level 2: Try Section M factors
    vol_dicts, evidence_source, confidence = self.structure_parser.parse_section_m_factors(section_m_text)
    if vol_dicts:
        return self._convert_vol_dicts_to_volumes(vol_dicts, evidence_source, confidence), []

    # Level 3: Try file names
    if filenames:
        vol_dicts, evidence_source, confidence = self.structure_parser.infer_from_filenames(filenames)
        if vol_dicts:
            return self._convert_vol_dicts_to_volumes(vol_dicts, evidence_source, confidence), warnings

    # Level 4: FALLBACK - Single generic volume
    return [
        ProposalVolume(
            id="VOL-1",
            name="Proposal Response",
            volume_type=VolumeType.TECHNICAL,
            evidence_source=EvidenceSource.DEFAULT,
            confidence=ConfidenceLevel.LOW
        )
    ], ["WARNING: No explicit volume structure found..."]
```

### 6.5 Deprecated: _create_default_volumes

```python
def _create_default_volumes(self, rfp_format: str, section_m: List[Dict]) -> List[ProposalVolume]:
    """
    DEPRECATED in v3.3 - Strict Constructionist approach.
    Returns empty list to prevent phantom volume creation.
    """
    return []
```

---

## 7. API Reference

### 7.1 Base URL

```
http://localhost:8000/api
```

### 7.2 Endpoints

#### 7.2.1 RFP Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rfp` | Create new RFP project |
| `GET` | `/rfp` | List all RFPs |
| `GET` | `/rfp/{rfp_id}` | Get RFP details |
| `DELETE` | `/rfp/{rfp_id}` | Delete RFP |

#### 7.2.2 Document Upload

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rfp/{rfp_id}/upload` | Upload documents |
| `GET` | `/upload-config` | Get guided upload UI config |
| `POST` | `/rfp/{rfp_id}/upload-guided` | Upload with document type tags |
| `GET` | `/rfp/{rfp_id}/documents` | Get document metadata |
| `PUT` | `/rfp/{rfp_id}/documents/{filename}/type` | Update document type |

#### 7.2.3 Processing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rfp/{rfp_id}/process` | Process (auto-selects extractor) |
| `POST` | `/rfp/{rfp_id}/process-resilient` | Force resilient extraction |
| `POST` | `/rfp/{rfp_id}/process-best-practices` | Force best practices mode |
| `POST` | `/rfp/{rfp_id}/process-semantic` | Force semantic extraction |
| `GET` | `/rfp/{rfp_id}/status` | Get processing status |
| `GET` | `/rfp/{rfp_id}/quality` | Get quality metrics |

#### 7.2.4 Requirements & Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/rfp/{rfp_id}/requirements` | Get all requirements (paginated) |
| `GET` | `/rfp/{rfp_id}/requirements/{req_id}` | Get single requirement |
| `GET` | `/rfp/{rfp_id}/export` | Export to Excel |
| `GET` | `/rfp/{rfp_id}/stats` | Get detailed statistics |

#### 7.2.5 Proposal Outline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rfp/{rfp_id}/outline` | Generate outline |
| `GET` | `/rfp/{rfp_id}/outline` | Get outline as JSON |
| `GET` | `/rfp/{rfp_id}/outline/export` | Export as DOCX |

#### 7.2.6 Amendments

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rfp/{rfp_id}/amendments` | Upload amendment |
| `GET` | `/rfp/{rfp_id}/amendments` | Get amendment history |
| `GET` | `/rfp/{rfp_id}/amendments/report` | Generate change report |

#### 7.2.7 Company Library

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/library` | Get library status |
| `GET` | `/library/profile` | Get full company profile |
| `POST` | `/library/upload` | Upload company document |
| `GET` | `/library/documents` | List documents |
| `GET` | `/library/{doc_id}` | Get document details |
| `DELETE` | `/library/{doc_id}` | Remove document |
| `GET` | `/library/search` | Search library |
| `GET` | `/library/capabilities` | Get capabilities |
| `GET` | `/library/differentiators` | Get differentiators |
| `GET` | `/library/past-performance` | Get past performance |
| `GET` | `/library/key-personnel` | Get key personnel |

### 7.3 Request/Response Examples

#### Create RFP

```http
POST /api/rfp
Content-Type: application/json

{
  "name": "DOD NOC Support Services",
  "solicitation_number": "FA880625RB003"
}
```

**Response:**
```json
{
  "id": "rfp_abc123",
  "name": "DOD NOC Support Services",
  "solicitation_number": "FA880625RB003",
  "status": "created",
  "created_at": "2025-12-19T10:30:00Z"
}
```

#### Get Outline with Evidence Tracking (v3.3)

```http
GET /api/rfp/{rfp_id}/outline
```

**Response:**
```json
{
  "rfp_format": "DOD_UCF",
  "total_pages": 8,
  "volumes": [
    {
      "id": "VOL-1",
      "name": "Technical",
      "type": "technical",
      "page_limit": 8,
      "order": 0,
      "evidence_source": "list",
      "confidence": "high",
      "sections": [...]
    },
    {
      "id": "VOL-2",
      "name": "Cost/Price",
      "type": "cost_price",
      "page_limit": null,
      "order": 1,
      "evidence_source": "list",
      "confidence": "high",
      "sections": [...]
    }
  ],
  "evaluation_factors": [...],
  "warnings": []
}
```

---

## 8. Data Models

### 8.1 Core Requirement Model

```python
@dataclass
class RequirementNode:
    # Identity
    id: str                           # REQ-001, REQ-C-001
    text: str                         # Full requirement text
    text_hash: str                    # For deduplication

    # Classification
    requirement_type: RequirementType # PERFORMANCE, PROPOSAL_INSTRUCTION, etc.
    confidence: ConfidenceLevel       # HIGH, MEDIUM, LOW
    status: RequirementStatus         # ACTIVE, MODIFIED, DELETED

    # Source Tracking
    source: SourceLocation            # page, section, paragraph
    context_before: str
    context_after: str
    extraction_method: str            # regex, llm, hybrid

    # Graph Edges
    references_to: List[str]          # Cross-references
    referenced_by: List[str]
    parent_requirement: Optional[str]  # Hierarchy
    evaluated_by: List[str]           # Section M links
    instructed_by: List[str]          # Section L links

    # Amendment Tracking
    version: int
    modified_by_amendment: Optional[str]
    previous_text: Optional[str]
    modification_reason: Optional[str]
```

### 8.2 Extraction Result (v3.0)

```python
@dataclass
class ExtractionResult:
    requirements: List[RequirementCandidate]
    quality_metrics: ExtractionQualityMetrics
    review_queue: List[RequirementCandidate]  # Low confidence
    section_candidates: Dict[str, SectionCandidate]
    sow_location: Optional[str]
```

### 8.3 Quality Metrics

```python
@dataclass
class ExtractionQualityMetrics:
    # Counts
    total_documents: int
    total_pages: int
    total_requirements: int

    # Confidence Distribution
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    uncertain_count: int

    # Section Counts
    section_counts: Dict[str, int]

    # Anomalies & Warnings
    anomalies: List[str]
    warnings: List[str]

    # Performance
    requirements_per_page: float
    sow_detected: bool
    sow_source: Optional[str]
```

### 8.4 Proposal Outline (v3.3)

```python
@dataclass
class ProposalOutline:
    rfp_format: str                        # NIH, GSA_BPA, DOD_UCF, etc.
    volumes: List[ProposalVolume]
    eval_factors: List[EvaluationFactor]
    format_requirements: FormatRequirements
    submission_info: SubmissionInfo
    warnings: List[str]
    total_pages: Optional[int]

@dataclass
class ProposalVolume:
    id: str
    name: str
    volume_type: VolumeType
    page_limit: Optional[int]
    sections: List[ProposalSection]
    eval_factors: List[str]
    order: int
    evidence_source: EvidenceSource        # v3.3
    confidence: ConfidenceLevel            # v3.3

@dataclass
class ProposalSection:
    id: str
    name: str
    page_limit: Optional[int]
    requirements: List[str]
    eval_criteria: List[str]
    subsections: List['ProposalSection']
```

---

## 9. Processing Pipelines

### 9.1 Document Processing Flow

```
                    ┌─────────────────┐
                    │   File Upload   │
                    │  (PDF/DOCX/XLS) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ MultiFormatParser│
                    │  - PDF: pypdf   │
                    │  - DOCX: docx   │
                    │  - XLSX: openpyxl│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Document Type   │
                    │ Classification  │
                    │ (v3.2 Guided)   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │    SOW    │    │ Section L │    │ Section M │
    │(Attachment│    │(Placement │    │(Evaluation│
    │    1)     │    │Procedures)│    │ Factors)  │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  ┌─────────────────┐
                  │ResilientExtractor│
                  │   (v3.0)        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Categorization │
                  │  L_COMPLIANCE   │
                  │  EVALUATION     │
                  │  TECHNICAL      │
                  └────────┬────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │   Excel   │  │   JSON    │  │   Word    │
    │   Export  │  │   API     │  │  Outline  │
    └───────────┘  └───────────┘  └───────────┘
```

### 9.2 Outline Generation Flow (v3.3)

```
                    ┌─────────────────┐
                    │  Requirements   │
                    │ (Categorized)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │SmartOutlineGen  │
                    │     v3.3        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ProposalStructure│
                    │    Parser       │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Level 1: TABLE│   │ Level 2: LIST │   │ Level 3: FACTOR│
│   (HIGH)      │   │   (HIGH)      │   │   (MEDIUM)    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    Found?  │
                   ┌────────┴────────┐
                   │ YES             │ NO
                   ▼                 ▼
          ┌───────────────┐  ┌───────────────┐
          │ Build Volumes │  │ Level 4:      │
          │ with Evidence │  │ FALLBACK      │
          │ Tracking      │  │ Single Volume │
          └───────────────┘  └───────────────┘
                   │                 │
                   └────────┬────────┘
                            ▼
                   ┌───────────────┐
                   │ Populate      │
                   │ Sections      │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ Apply Page    │
                   │ Limits        │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │ Export        │
                   │ (JSON/DOCX)   │
                   └───────────────┘
```

---

## 10. Frontend Application

### 10.1 Architecture

Single-page React application embedded in `/web/index.html`.

**Technology:**
- React 18 (via CDN)
- Tailwind CSS for styling
- Lucide Icons
- Babel for JSX compilation

### 10.2 Key Components

| Component | Purpose |
|-----------|---------|
| `App` | Root component, state management |
| `FileUploader` | Drag-and-drop document upload |
| `GuidedUpload` | Document type selector (v3.2) |
| `ProcessingStatus` | Real-time progress display |
| `RequirementsTable` | Interactive requirements list |
| `StatsPanel` | Extraction statistics |
| `ExportButtons` | Excel/Word export triggers |

### 10.3 Design System

```css
/* Color Palette */
--bg-primary: #0a0a0f;
--bg-secondary: #1a1a2e;
--accent-blue: #4f8cff;
--accent-green: #34d399;
--text-primary: #ffffff;
--text-secondary: #94a3b8;

/* Typography */
--font-primary: 'DM Sans', sans-serif;
--font-mono: 'JetBrains Mono', monospace;
```

---

## 11. Database Schema

### 11.1 Tables

```sql
-- LangGraph state persistence
CREATE TABLE checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    checkpoint_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- RFP project metadata
CREATE TABLE proposals (
    proposal_id VARCHAR(255) PRIMARY KEY,
    client_name VARCHAR(255),
    solicitation_number VARCHAR(255),
    current_phase VARCHAR(50),
    quality_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit trail
CREATE TABLE agent_trace_log (
    id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(255) REFERENCES proposals(proposal_id),
    agent_name VARCHAR(100) NOT NULL,
    action VARCHAR(255) NOT NULL,
    reasoning_trace TEXT,
    duration_ms INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Human corrections (data flywheel)
CREATE TABLE human_feedback (
    id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(255) REFERENCES proposals(proposal_id),
    section_id VARCHAR(255),
    feedback_type VARCHAR(50),
    original_content TEXT,
    corrected_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 11.2 Indexes

```sql
CREATE INDEX idx_checkpoints_thread ON checkpoints(thread_id, created_at DESC);
CREATE INDEX idx_agent_trace_proposal ON agent_trace_log(proposal_id, agent_name);
CREATE INDEX idx_feedback_proposal ON human_feedback(proposal_id, feedback_type);
```

---

## 12. Deployment Configuration

### 12.1 Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PROPELAI_ENV=production
      - POSTGRES_HOST=postgres
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=propelai
      - POSTGRES_USER=propelai
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  postgres_data:
  chroma_data:
```

### 12.2 Startup Commands

```bash
# Docker Compose
docker-compose up -d

# Direct Python
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Development with reload
python -m uvicorn api.main:app --reload
```

---

## 13. Version History

### v3.3 (December 19, 2025) - Strict Constructionist Architecture

**Major Changes:**
- New `ProposalStructureParser` class for dedicated Section L parsing
- `EvidenceSource` and `ConfidenceLevel` enums for tracking
- Hierarchy of Authority for volume detection
- Deprecated `_create_default_volumes` (returns empty list)
- No more UCF defaults for Task Orders, RFEs, RFQs

**Bug Fixes:**
- Eliminated phantom volume hallucination (SENTRA RFE, DOD NOC)
- Fixed SF1449 being parsed as "Sub Factor 1449"
- Fixed "1 page" limit from volume number matching
- Fixed outline title using source filename

### v3.2 (December 19, 2025) - Guided Upload & Bug Fixes

**Features:**
- Guided document upload UI with document type slots
- Document type classification (SOW, Instructions, Evaluation)
- "Attachment 2" pattern recognition for Section L/M content

**Bug Fixes:**
- Fixed section_l/section_m filter using category field
- Fixed phantom volumes from aggressive keyword matching
- Fixed page limit extraction for table formats

### v3.1 (December 19, 2025) - Section L Structure

**Features:**
- Outline follows Section L hierarchy, not SOW structure
- Sub-factor extraction from Section L
- Page limit extraction from tables
- Evaluation weight extraction from Section M

### v3.0 (December 17, 2025) - Resilient Extraction

**Major Changes:**
- "Extract-First" architecture
- Multi-layer SOW detection
- Confidence scoring for all requirements
- Quality metrics and anomaly detection
- Never drops requirements (flags for review instead)

### v2.9 (December 15, 2025) - Best Practices CTM

**Features:**
- Section-aware extraction
- Best practices CTM export format
- Improved L/M/C categorization

### v2.8 (December 2025) - Semantic Extraction

**Features:**
- LLM-enhanced extraction
- Action verb identification
- Actor and constraint extraction

---

## 14. Known Issues & Limitations

### 14.1 Current Limitations

| Issue | Status | Workaround |
|-------|--------|------------|
| Page limits in complex tables | Partial | Manual review recommended |
| Multi-volume RFPs with shared sections | Known | Not yet supported |
| Scanned PDF OCR | Partial | ENABLE_OCR=true required |
| Very large documents (>500 pages) | Performance | Consider splitting |

### 14.2 Planned Improvements

1. **Phase 2 Agents**: Strategy, Drafting, Red Team integration
2. **Vector Search**: Semantic requirement matching
3. **Amendment Differencing**: Visual diff for changes
4. **Multi-tenant Support**: Organization-level isolation

---

## Appendix A: Test Coverage

### Test Files

| File | Purpose | Key Tests |
|------|---------|-----------|
| `test_structure_determinism.py` | v3.3 Strict Constructionist | SENTRA RFE, DOD NOC volume counts |
| `test_resilient_extraction.py` | v3.0 Pipeline | Confidence scoring, anomaly detection |
| `test_accuracy.py` | Extraction quality | Recall, precision, F1 scores |
| `test_agents.py` | Agent interfaces | Orchestration flow |

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_structure_determinism.py -v

# With coverage
pytest tests/ --cov=agents --cov-report=html
```

---

## Appendix B: Debug Logging

### Key Debug Messages (v3.3)

```
[DEBUG] StructureParser: Found N volumes from TABLE
[DEBUG] StructureParser: Found N volumes from LIST
[DEBUG] StructureParser: Detected RFP type 'request_for_estimate'
[DEBUG] _extract_volumes v3.3: section_l=X, section_m=Y
[DEBUG] _extract_volumes v3.3: rfp_type=X, allow_ucf_defaults=False
[DEBUG] Volume 'Technical': source=list, confidence=high
```

### Enabling Debug Logs

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**Document End**

*Generated: December 19, 2025*
*PropelAI Version: 3.3 (Strict Constructionist)*
