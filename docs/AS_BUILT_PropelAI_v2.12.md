# PropelAI As-Built Documentation v2.12

**Document Version:** 2.12
**Date:** December 14, 2024
**System:** PropelAI - Autonomous Proposal Operating System (APOS)
**Status:** Production

---

## 1. Executive Summary

PropelAI is an AI-powered **Autonomous Proposal Operating System (APOS)** designed for government contractors. It automates the complex RFP analysis process—traditionally called "RFP shredding"—that normally takes 2-5 days of manual analyst work and completes it in minutes.

### Core Capabilities
- **Automated RFP Shredding** - Extracts requirements, instructions, and evaluation criteria
- **Compliance Traceability Matrix (CTM)** - 19-column Shipley-methodology export
- **Multi-Document Bundle Processing** - SOW, Amendments, CDRLs, Attachments
- **Amendment Tracking** - Version control for requirement changes
- **Annotated Outline Generation** - Proposal structure with page allocations
- **Win Strategy Development** - Themes, discriminators, ghosting language
- **Content Drafting** - Zero-hallucination proposal sections with citations
- **Red Team Evaluation** - Color-coded scoring and remediation guidance

### Technology Stack
| Component | Technology |
|-----------|------------|
| Backend | FastAPI (Python 3.11+) |
| Frontend | React 18 (Single-file SPA) |
| Database | PostgreSQL with pgvector |
| Vector Store | Chroma / Pinecone |
| Document Processing | PyMuPDF, python-docx, openpyxl |
| OCR | Tensorlake (Gemini 3-powered) |
| LLM Providers | Google Gemini, Anthropic Claude, OpenAI |
| Orchestration | LangGraph |
| Deployment | Docker / Render |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PropelAI APOS Platform                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Frontend Layer (React SPA)                     │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ File       │  │ Processing │  │ CTM        │  │ Red Team   │     │  │
│  │  │ Upload     │  │ Dashboard  │  │ Viewer     │  │ Scorecard  │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         API Layer (FastAPI)                           │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ RFP        │  │ Processing │  │ Export     │  │ OASIS+     │     │  │
│  │  │ Routes     │  │ Routes     │  │ Routes     │  │ Routes     │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    Orchestration Layer (LangGraph)                    │  │
│  │                                                                       │  │
│  │   ┌─────────────────────────────────────────────────────────────┐    │  │
│  │   │                   ProposalOrchestrator                       │    │  │
│  │   │    (Supervisor Agent - Routes tasks, manages checkpoints)    │    │  │
│  │   └─────────────────────────────────────────────────────────────┘    │  │
│  │                                │                                      │  │
│  │        ┌───────────┬───────────┼───────────┬───────────┐             │  │
│  │        ▼           ▼           ▼           ▼           ▼             │  │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │
│  │   │Compliance│ │Strategy │ │Drafting │ │Red Team │ │ OASIS+  │       │  │
│  │   │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │ │ Module  │       │  │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Enhanced Compliance Engine                        │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ Document   │  │ Section-   │  │ CTM        │  │ Outline    │     │  │
│  │  │ Parser     │  │ Aware      │  │ Exporter   │  │ Generator  │     │  │
│  │  │            │  │ Extractor  │  │            │  │            │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Storage Layer                                  │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ PostgreSQL │  │ Chroma     │  │ File       │  │ In-Memory  │     │  │
│  │  │ (State)    │  │ (Vectors)  │  │ Storage    │  │ (Session)  │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
propelai01/
├── agents/                          # Multi-Agent Framework
│   ├── compliance_agent.py          # Legacy compliance agent
│   ├── strategy_agent.py            # Win strategy generation
│   ├── drafting_agent.py            # Content generation
│   ├── red_team_agent.py            # Evaluation scoring
│   ├── enhanced_compliance/         # Core extraction engine (22 modules)
│   │   ├── agent.py                 # Enhanced compliance orchestrator
│   │   ├── models.py                # Data models & enums
│   │   ├── parser.py                # Multi-format document parser
│   │   ├── document_structure.py    # RFP structure analysis
│   │   ├── extractor.py             # Legacy regex extraction
│   │   ├── section_aware_extractor.py  # Best practices extraction
│   │   ├── semantic_extractor.py    # LLM-based classification
│   │   ├── ctm_extractor.py         # 19-column CTM extraction
│   │   ├── ctm_data_models.py       # CTM schema definitions
│   │   ├── best_practices_ctm.py    # Shipley methodology export
│   │   ├── bundle_detector.py       # Document type classifier
│   │   ├── amendment_processor.py   # Amendment tracking
│   │   ├── outline_generator.py     # Section L/M outline parsing
│   │   ├── smart_outline_generator.py  # CTM-based outline generation
│   │   ├── annotated_outline_exporter.js  # Word document export
│   │   ├── excel_export.py          # Legacy Excel export
│   │   ├── excel_parser.py          # CTM import
│   │   ├── semantic_ctm_export.py   # 19-column semantic export
│   │   ├── company_library.py       # Past performance parsing
│   │   └── resolver.py              # Cross-reference resolution
│   ├── oasis_plus/                  # GSA OASIS+ specialized module
│   │   ├── __init__.py              # Module exports
│   │   └── database.py              # PostgreSQL schema
│   └── integrations/
│       └── tensorlake_processor.py  # OCR integration
├── api/                             # REST API Layer
│   ├── main.py                      # FastAPI application (1,834 lines)
│   └── oasis_routes.py              # OASIS+ endpoints (480+ lines)
├── core/                            # Core Orchestration
│   ├── state.py                     # ProposalState schema (30+ fields)
│   ├── config.py                    # Environment configuration
│   └── orchestrator.py              # LangGraph workflow engine
├── web/
│   └── index.html                   # React SPA frontend
├── tools/
│   └── document_tools.py            # CLI utilities
├── tests/
│   └── test_agents.py               # Pytest suite
├── docs/                            # Documentation
├── cli.py                           # Command-line interface
├── demo.py                          # Workflow demonstration
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Container orchestration
└── init.sql                         # Database schema
```

---

## 3. Multi-Agent System

### 3.1 Agent Overview

| Agent | Role | Input | Output | LLM |
|-------|------|-------|--------|-----|
| **Compliance Agent** | RFP Shredding | Raw documents | Requirements, CTM | Gemini Flash |
| **Strategy Agent** | Win Strategy | Section M, history | Themes, ghosting | Gemini Pro/Claude |
| **Drafting Agent** | Content Generation | Outline, themes | Proposal sections | Gemini Pro |
| **Red Team Agent** | Evaluation | Draft, Section M | Scores, findings | Gemini Pro/Claude |
| **Orchestrator** | Workflow Control | All agents | State routing | Logic-based |

### 3.2 Compliance Agent (The Paralegal)

**Location:** `agents/enhanced_compliance/agent.py`

**Responsibilities:**
- Parse RFP documents (PDF, DOCX, XLSX)
- Detect document bundle types
- Extract requirements with confidence scoring
- Build cross-reference graph
- Track amendments and version changes
- Generate compliance matrix

**Three-Tier Extraction Pipeline:**

| Tier | Version | Method | Accuracy |
|------|---------|--------|----------|
| Legacy | v2.7 | Regex patterns | ~40% |
| Semantic | v2.8 | LLM classification | ~65% |
| Best Practices | v2.9+ | Structure-aware + LLM | ~85%+ |

### 3.3 Strategy Agent (The Capture Manager)

**Location:** `agents/strategy_agent.py`

**Responsibilities:**
- Analyze Section M evaluation factors
- Generate win themes (max 7)
- Create competitor ghosting language
- Map discriminators to evaluation criteria
- Produce annotated proposal outlines with page allocations

### 3.4 Drafting Agent (The Writer)

**Location:** `agents/drafting_agent.py`

**Responsibilities:**
- Generate compliant proposal narrative
- Embed win themes naturally
- Track all citations
- Flag uncited claims as HIGH RISK
- Enforce zero-hallucination policy

### 3.5 Red Team Agent (The Evaluator)

**Location:** `agents/red_team_agent.py`

**Responsibilities:**
- Simulate government evaluator perspective
- Apply color-based scoring
- Identify compliance gaps
- Provide remediation guidance
- Maintain immutable audit log

**Scoring Colors:**
| Color | Rating | Meaning |
|-------|--------|---------|
| BLUE | Exceptional | Exceeds requirements significantly |
| GREEN | Acceptable | Meets requirements |
| YELLOW | Marginal | Minor deficiencies |
| RED | Unacceptable | Major deficiencies |

---

## 4. Enhanced Compliance Engine

### 4.1 Module Inventory

The enhanced compliance engine contains 22 modules with 17,566+ lines of code:

| Module | Lines | Purpose |
|--------|-------|---------|
| `agent.py` | 800+ | Orchestrator for multi-document processing |
| `models.py` | 600+ | Data models, enums, dataclasses |
| `parser.py` | 500+ | PDF/DOCX/XLSX parsing |
| `document_structure.py` | 400+ | UCF section detection |
| `extractor.py` | 600+ | Legacy regex extraction |
| `section_aware_extractor.py` | 1,200+ | Best practices extraction |
| `semantic_extractor.py` | 800+ | LLM-based classification |
| `ctm_extractor.py` | 700+ | 19-column CTM extraction |
| `ctm_data_models.py` | 300+ | CTM schema definitions |
| `best_practices_ctm.py` | 900+ | Shipley methodology export |
| `bundle_detector.py` | 400+ | Document type classification |
| `amendment_processor.py` | 500+ | Amendment tracking |
| `outline_generator.py` | 600+ | Section L/M parsing |
| `smart_outline_generator.py` | 1,500+ | CTM-based outline generation |
| `annotated_outline_exporter.js` | 700+ | Word document export |
| `excel_export.py` | 400+ | Legacy Excel export |
| `excel_parser.py` | 300+ | CTM import |
| `semantic_ctm_export.py` | 500+ | 19-column semantic export |
| `company_library.py` | 400+ | Past performance parsing |
| `resolver.py` | 300+ | Cross-reference resolution |

### 4.2 Data Models

#### RequirementNode (Graph-Based)
```python
@dataclass
class RequirementNode:
    id: str                          # REQ-001
    text: str                        # Full requirement text
    text_hash: str                   # MD5 for deduplication
    requirement_type: RequirementType
    confidence: ConfidenceLevel      # HIGH, MEDIUM, LOW
    confidence_score: float          # 0.0-1.0
    status: RequirementStatus        # ACTIVE, MODIFIED, DELETED
    binding_level: str               # MANDATORY, HIGHLY_DESIRABLE, OPTIONAL
    category: str                    # Grouping category
    source: SourceLocation           # Document, page, section

    # Graph edges
    references_to: List[str]         # Outbound cross-references
    referenced_by: List[str]         # Inbound references
    parent_requirement: Optional[str]
    child_requirements: List[str]

    # Iron Triangle mapping
    evaluated_by: List[str]          # Section M factors
    instructed_by: List[str]         # Section L instructions
    deliverable_for: Optional[str]   # CDRL reference

    # Amendment tracking
    version: int
    modified_by_amendment: Optional[str]
    previous_text: Optional[str]
```

#### RequirementType Enum
```python
class RequirementType(Enum):
    PERFORMANCE = "performance"           # "Shall provide..."
    PROPOSAL_INSTRUCTION = "proposal_instruction"  # "Shall describe..."
    EVALUATION_CRITERION = "evaluation_criterion"  # "Will evaluate..."
    PERFORMANCE_METRIC = "performance_metric"      # KPIs, SLAs
    DELIVERABLE = "deliverable"           # "Submit report..."
    LABOR_REQUIREMENT = "labor_requirement"        # Hours, staffing
    QUALIFICATION = "qualification"       # "Must be SB..."
    COMPLIANCE = "compliance"             # "FAR 52.xxx applies"
    FORMAT = "format"                     # "12-point font..."
    PROHIBITION = "prohibition"           # "Shall not..."
```

#### ComplianceMatrixRow (19-Column CTM)
```python
@dataclass
class ComplianceMatrixRow:
    requirement_id: str
    requirement_text: str
    section_reference: str           # C.3.1, L.4, M.2
    section_type: str                # C, L, M, attachment
    requirement_type: str
    binding_level: str               # MANDATORY, HIGHLY_DESIRABLE
    scoring_type: str                # PASS_FAIL, WEIGHTED, QUALITATIVE
    max_points: Optional[int]
    response_format: str             # NARRATIVE, TABLE, FORM
    proposed_response: str
    compliance_status: str           # Compliant, Partial, N/A
    assigned_owner: str
    proposal_section: str
    related_requirements: List[str]
    evaluation_factor: Optional[str]
    priority: str                    # High, Medium, Low
    risk_if_non_compliant: str
    evidence_required: List[str]
    notes: str
```

### 4.3 Document Processing Pipeline

```
RFP Upload → Bundle Detector → MultiFormatParser
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
               PDF Parser      DOCX Parser      XLSX Parser
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                         Document Structure Parser
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              Section A-K      Section L        Section M
             (Technical)      (Instructions)   (Evaluation)
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                         Section-Aware Extractor
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              Requirements    P0 Constraints    Eval Factors
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                         Cross-Reference Resolver
                                     │
                                     ▼
                         Best Practices CTM Export
```

### 4.4 P0 Constraint Extraction

**Location:** `smart_outline_generator.py:_extract_p0_constraints()`

P0 constraints are pass/fail formatting requirements that cause immediate disqualification if violated:

| Constraint Type | Description | Example |
|----------------|-------------|---------|
| PAGE_LIMIT | Maximum page count | "Not to exceed 25 pages" |
| FONT_SIZE | Required font size | "12-point minimum" |
| FONT_FAMILY | Required typeface | "Times New Roman" |
| MARGIN | Page margins | "1-inch margins" |
| LINE_SPACING | Text spacing | "Double-spaced" |
| FILE_FORMAT | Submission format | "PDF only" |
| SUBMISSION_METHOD | Delivery method | "via SAM.gov" |
| DEADLINE | Due date/time | "4:00 PM EST" |
| NAMING_CONVENTION | File naming | "CompanyName_Vol1.pdf" |

**Implementation Fix (v2.12):**
```python
def get_field(obj, field: str, default: str = '', alt_field: str = None) -> str:
    """Get field from either dict or object, with optional alternate field name"""
    if isinstance(obj, dict):
        value = obj.get(field) or (obj.get(alt_field) if alt_field else None)
        return value if value else default
    value = getattr(obj, field, None) or (getattr(obj, alt_field, None) if alt_field else None)
    return value if value else default

# Usage - handles both JSON serialized ('type') and Python objects ('constraint_type')
ctype = get_field(c, 'constraint_type', 'UNKNOWN', alt_field='type')
```

### 4.5 Binding Level Detection

**Location:** `section_aware_extractor.py`

Binding levels indicate requirement strength:

| Level | Keywords | Meaning |
|-------|----------|---------|
| MANDATORY | shall, must, required, will | Must comply |
| HIGHLY_DESIRABLE | should, expected, strongly | Important but not disqualifying |
| DESIRABLE | may, can, preferred | Nice to have |
| INFORMATIONAL | for information, note | Background only |

**Section M Fallback:** Items from Section M without explicit binding language default to HIGHLY_DESIRABLE since evaluation factors carry significant weight.

---

## 5. API Layer

### 5.1 Core Endpoints

**Location:** `api/main.py`

#### RFP Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rfp` | Create new RFP project |
| GET | `/api/rfp` | List all RFPs |
| GET | `/api/rfp/{id}` | Get RFP details |
| DELETE | `/api/rfp/{id}` | Delete RFP |

#### Document Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rfp/{id}/upload` | Upload documents (batch) |
| GET | `/api/rfp/{id}/status` | Processing status |

#### Requirement Extraction
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rfp/{id}/process` | v2.7 legacy extraction |
| POST | `/api/rfp/{id}/process-semantic` | v2.8 semantic extraction |
| POST | `/api/rfp/{id}/process-best-practices` | v2.9 best practices |
| GET | `/api/rfp/{id}/requirements` | Get requirements |
| GET | `/api/rfp/{id}/stats` | Extraction statistics |

#### Export & Artifacts
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/rfp/{id}/export` | Download CTM (Excel) |
| POST | `/api/rfp/{id}/outline` | Generate outline |
| GET | `/api/rfp/{id}/outline/export` | Export outline |

#### Amendment Tracking
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/rfp/{id}/amendments` | Upload amendment |
| GET | `/api/rfp/{id}/amendments` | Amendment history |
| GET | `/api/rfp/{id}/amendments/report` | Change summary |

### 5.2 OASIS+ Endpoints

**Location:** `api/oasis_routes.py`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/oasis/proposals` | Create OASIS+ proposal |
| GET | `/api/oasis/proposals` | List proposals |
| POST | `/api/oasis/proposals/{id}/jp1` | Upload J.P-1 Matrix |
| POST | `/api/oasis/proposals/{id}/projects` | Add project |
| POST | `/api/oasis/proposals/{id}/score` | Score for domain |
| POST | `/api/oasis/proposals/{id}/optimize` | Optimize selection |
| POST | `/api/oasis/proposals/{id}/artifacts` | Generate artifacts |
| GET | `/api/oasis/proposals/{id}/scorecard` | Domain scorecard |
| GET | `/api/oasis/proposals/{id}/export` | Symphony bundle |

---

## 6. Proposal State Management

### 6.1 ProposalState Schema

**Location:** `core/state.py`

```python
class ProposalState(TypedDict):
    # Metadata
    proposal_id: str
    client_name: str
    opportunity_name: str
    solicitation_number: str
    due_date: str
    current_phase: ProposalPhase

    # RFP Data
    rfp_raw_text: str
    rfp_file_paths: List[str]
    rfp_metadata: Dict[str, Any]

    # Iron Triangle
    requirements: List[Dict]          # Section C
    instructions: List[Dict]          # Section L
    evaluation_criteria: List[Dict]   # Section M

    # Requirements Graph
    requirements_graph: Dict[str, RequirementNode]

    # Compliance Matrix
    compliance_matrix: List[ComplianceMatrixRow]

    # Strategy
    win_themes: List[WinTheme]
    competitor_analysis: List[Dict]
    price_to_win: Optional[Dict]

    # Drafts
    draft_sections: Dict[str, DraftSection]

    # Red Team
    red_team_findings: List[Dict]
    overall_score: Optional[float]

    # Audit Trail
    agent_trace_log: List[Dict]
    human_feedback: List[Dict]
```

### 6.2 Proposal Phases

```python
class ProposalPhase(Enum):
    INTAKE = "intake"           # Documents uploaded
    SHRED = "shred"             # Extracting requirements
    STRATEGY = "strategy"       # Developing win themes
    OUTLINE = "outline"         # Creating structure
    DRAFTING = "drafting"       # Writing content
    REVIEW = "review"           # Red team evaluation
    FINALIZE = "finalize"       # Final edits
    SUBMITTED = "submitted"     # Complete
```

---

## 7. External Integrations

### 7.1 Tensorlake OCR

**Location:** `agents/integrations/tensorlake_processor.py`

**Purpose:** Gemini 3-powered OCR for complex documents

**Use Cases:**
- Scanned PDFs (image-based)
- Complex tables with merged cells
- Multi-column layouts
- Charts and diagrams with text

**Configuration:**
```python
TENSORLAKE_API_KEY=xxx
output_format: markdown | json | chunks
table_extraction: true
image_extraction: true
chunk_size: 1000
chunk_overlap: 100
```

### 7.2 LLM Providers

| Provider | Model | Use Case |
|----------|-------|----------|
| Google | Gemini 1.5 Flash | Fast extraction |
| Google | Gemini 1.5 Pro | Complex reasoning (2M context) |
| Anthropic | Claude 3.5 Sonnet | Strategy & drafting |
| OpenAI | GPT-4 Turbo | Fallback |

### 7.3 Vector Database

| Environment | Technology | Purpose |
|-------------|------------|---------|
| Development | Chroma (embedded) | Semantic search |
| Production | Pinecone (cloud) | Scalable search |

---

## 8. Database Schema

### 8.1 PostgreSQL Schema

```sql
-- Proposal State Checkpointing
CREATE TABLE proposal_checkpoints (
    checkpoint_id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(50) NOT NULL,
    phase VARCHAR(50) NOT NULL,
    state_json JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- OASIS+ Proposals
CREATE TABLE oasis_proposals (
    proposal_id VARCHAR(50) PRIMARY KEY,
    contractor_name VARCHAR(255),
    contractor_cage VARCHAR(20),
    business_size VARCHAR(50),
    status VARCHAR(50),
    target_domains JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Document Chunks (with embeddings)
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(50),
    content TEXT,
    embedding vector(1536),
    source_document VARCHAR(255),
    page_number INT
);

-- Project Claims
CREATE TABLE project_claims (
    id SERIAL PRIMARY KEY,
    project_id VARCHAR(50),
    claim_text TEXT,
    verification_status VARCHAR(50),
    supporting_evidence JSONB
);
```

---

## 9. CLI Interface

**Location:** `cli.py`

```bash
# Workflow Commands
propelai new "Name" --client "Client" --solicitation "SOL-001"
propelai upload PROP-123 /path/to/rfp.pdf
propelai shred PROP-123
propelai strategy PROP-123
propelai draft PROP-123
propelai redteam PROP-123

# Status & Export
propelai status PROP-123
propelai export PROP-123 --output ctm.xlsx
propelai list

# Server
propelai serve --host 0.0.0.0 --port 8000
```

---

## 10. Agency Detection

**Location:** `smart_outline_generator.py:_detect_agency()`

| Agency | Pattern | Example |
|--------|---------|---------|
| NIH | `^75N` | 75N96025R00004 |
| Navy | `^N\d{5}` | N0017826R30020003 |
| Air Force | `^FA` | FA8730-24-R-0001 |
| Army | `^W` | W912HQ-24-R-0001 |
| GSA | `^GS` | GS-00F-12345 |
| VA | `^36C` | 36C25724R0001 |

---

## 11. Annotated Outline Generator

### 11.1 Architecture

**Location:** `agents/enhanced_compliance/smart_outline_generator.py`

```
Compliance Matrix → Structure Analyzer → Volume Detector
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                   Technical Vol        Management Vol       Cost/Price Vol
                         │                    │                    │
                         └────────────────────┼────────────────────┘
                                              ▼
                              Page Budget Calculator
                                              │
                                              ▼
                              Evaluation Factor Mapper
                                              │
                                              ▼
                              Content Populator (L/M/C)
                                              │
                                              ▼
                              Annotated Outline JSON
                                              │
                                              ▼
                              Word Document Exporter
```

### 11.2 Volume Types

```python
class VolumeType(Enum):
    TECHNICAL = "technical"
    MANAGEMENT = "management"
    PAST_PERFORMANCE = "past_performance"
    COST_PRICE = "cost_price"
    SMALL_BUSINESS = "small_business"
    ORAL_PRESENTATION = "oral_presentation"
    EXECUTIVE_SUMMARY = "executive_summary"
    STAFFING = "staffing"
    TRANSITION = "transition"
    SECURITY = "security"
```

### 11.3 Content Color Coding

| Color | Source | Content Type |
|-------|--------|--------------|
| Blue | Section L | Instructions |
| Purple | Section M | Evaluation Criteria |
| Green | Section C | Technical Requirements |
| Red | P0 | Pass/Fail Constraints |

---

## 12. Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0 | Initial | Keyword regex extraction |
| v2.0 | Q2 2024 | Multi-document bundles |
| v2.7 | Q3 2024 | Amendment tracking |
| v2.8 | Q3 2024 | Semantic LLM extraction |
| v2.9 | Q4 2024 | Best practices, structure-aware |
| v2.10 | Dec 2024 | Agency detection, constraints |
| v2.11 | Dec 2024 | Annotated outline exporter |
| v2.12 | Dec 2024 | Smart outline, P0 fix |

---

## 13. Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Max file size 100MB | Large RFPs may fail | Split documents |
| English only | Non-English RFPs unsupported | Manual translation |
| OCR quality varies | Scanned docs may have errors | Use clean PDFs |
| Processing time 5-60min | Complex RFPs take longer | Background processing |
| Legacy accuracy ~40% | Use best practices pipeline | Upgrade to v2.9+ |

---

## 14. Testing

**Location:** `tests/test_agents.py`

```bash
# Run all tests
pytest tests/test_agents.py -v

# Run specific test class
pytest tests/test_agents.py::TestProposalState -v

# Run with coverage
pytest tests/test_agents.py --cov=agents --cov-report=html
```

---

## 15. Deployment

### 15.1 Docker

```bash
docker-compose up
```

Services:
- `propelai-api` (FastAPI on :8000)
- `propelai-postgres` (PostgreSQL on :5432)
- `propelai-chroma` (Vector DB on :8001)

### 15.2 Render.com

```bash
# Start command
python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT

# Environment variables
GOOGLE_API_KEY=xxx
POSTGRES_HOST=xxx
POSTGRES_DB=propelai
```

---

## 16. Support & Maintenance

### Monitoring
- API health: `GET /api/health`
- Processing status: `GET /api/rfp/{id}/status`
- Statistics: `GET /api/rfp/{id}/stats`

### Logs
- Application logs: stdout/stderr
- Audit trail: `proposal_state.agent_trace_log`
- Human feedback: `proposal_state.human_feedback`

---

**Document Status:** Production Reference
**Last Updated:** December 14, 2024
