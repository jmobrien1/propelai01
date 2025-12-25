# PropelAI As-Built Documentation

**Version:** 2.11.0
**Document Date:** December 2024
**System:** Autonomous Proposal Operating System (APOS)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Core Components](#4-core-components)
5. [Agent System](#5-agent-system)
6. [Enhanced Compliance Engine](#6-enhanced-compliance-engine)
7. [API Reference](#7-api-reference)
8. [Data Models](#8-data-models)
9. [Processing Pipelines](#9-processing-pipelines)
10. [Database Schema](#10-database-schema)
11. [Frontend Architecture](#11-frontend-architecture)
12. [Integration Points](#12-integration-points)
13. [Configuration Reference](#13-configuration-reference)
14. [Deployment](#14-deployment)
15. [Technical Specifications](#15-technical-specifications)

---

## 1. Executive Summary

### 1.1 Purpose

PropelAI is an AI-powered Request for Proposal (RFP) analysis and compliance matrix generation platform designed for government contractors. The system automates the extraction of requirements from complex federal solicitation documents and generates professional Compliance Traceability Matrices (CTM).

### 1.2 Key Capabilities

- **Multi-format document parsing**: PDF, DOCX, XLSX with page-level traceability
- **Intelligent requirement extraction**: 69+ pattern matching with semantic classification
- **Requirements graph**: Cross-document relationship mapping (C↔L↔M Iron Triangle)
- **Amendment tracking**: SF30, Q&A, and modification lifecycle management
- **Compliance matrix generation**: Professional Excel export with evaluation factor alignment
- **Proposal outline generation**: Annotated DOCX with color-coded requirements
- **Company library**: Capability and past performance knowledge base
- **Multi-agent orchestration**: LangGraph-based workflow with human-in-the-loop

### 1.3 Technology Stack

| Layer | Technology |
|-------|------------|
| Backend API | FastAPI 0.100+, Python 3.11+ |
| Frontend | React 18 (single-file, unbundled) |
| Database | PostgreSQL 15 (LangGraph checkpointing) |
| Vector Store | Chroma (dev) / Pinecone (prod) |
| LLM Primary | Google Gemini 1.5 Flash/Pro |
| LLM Fallback | Anthropic Claude 3.5, OpenAI GPT-4 |
| Orchestration | LangGraph StateGraph |
| Document Gen | Node.js + docx library |
| Containerization | Docker Compose |

### 1.4 Metrics

- **Codebase Size**: ~5.3MB, 40 Python files, 21,781 lines of Python
- **API Endpoints**: 47 REST endpoints
- **Extraction Patterns**: 69+ requirement patterns, 68 noise filters
- **Supported Document Types**: 10 (MAIN_SOLICITATION, SOW, AMENDMENT, ATTACHMENT, etc.)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PROPELAI APOS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────────────────────────────────────────┐    │
│  │   React     │    │                  FastAPI Backend                 │    │
│  │  Frontend   │◄──►│                   (api/main.py)                  │    │
│  │  (83KB SPA) │    │                   47 Endpoints                   │    │
│  └─────────────┘    └──────────────────────┬──────────────────────────┘    │
│                                             │                               │
│  ┌──────────────────────────────────────────┴──────────────────────────┐   │
│  │                        AGENT LAYER                                   │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │   │
│  │  │ Compliance │ │  Strategy  │ │  Drafting  │ │  Red Team  │       │   │
│  │  │   Agent    │ │   Agent    │ │   Agent    │ │   Agent    │       │   │
│  │  │ "Paralegal"│ │ "Capture   │ │ "Writer"   │ │ "Evaluator"│       │   │
│  │  │            │ │  Manager"  │ │            │ │            │       │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                             │                               │
│  ┌──────────────────────────────────────────┴──────────────────────────┐   │
│  │                  ENHANCED COMPLIANCE ENGINE v3.0                     │   │
│  │  ┌─────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ │   │
│  │  │ Parser  │ │ Extractor │ │ Resolver │ │ Amendment │ │  Export  │ │   │
│  │  │         │ │           │ │          │ │ Processor │ │          │ │   │
│  │  └─────────┘ └───────────┘ └──────────┘ └───────────┘ └──────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                             │                               │
│  ┌──────────────────────────────────────────┴──────────────────────────┐   │
│  │                         CORE LAYER                                   │   │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           │   │
│  │  │   config.py    │ │    state.py    │ │ orchestrator.py│           │   │
│  │  │  APOSConfig    │ │ ProposalState  │ │   LangGraph    │           │   │
│  │  └────────────────┘ └────────────────┘ └────────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                             │                               │
│  ┌──────────────────────────────────────────┴──────────────────────────┐   │
│  │                      INFRASTRUCTURE LAYER                            │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │   │
│  │  │ PostgreSQL │ │   Chroma/  │ │  Gemini/   │ │   File     │        │   │
│  │  │ Checkpoint │ │  Pinecone  │ │  Claude    │ │  Storage   │        │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User Upload RFP Documents
         │
         ▼
┌─────────────────────────────────┐
│  api/main.py: upload_files()   │
│  Save to /uploads/{rfp_id}/    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  BundleDetector.detect_bundle()│
│  Classify document types       │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  MultiFormatParser.parse()     │
│  Extract text with page nums   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  RequirementExtractor.extract()│
│  Pattern + Semantic matching   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  CrossReferenceResolver        │
│  Build Requirements Graph      │
│  Link C ↔ L ↔ M sections       │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  RFPStore (in-memory)          │
│  Store requirements + graph    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ExcelExporter / OutlineGen    │
│  Generate CTM.xlsx / Outline   │
└─────────────────────────────────┘
```

### 2.3 Processing Modes

| Mode | Version | Description | Use Case |
|------|---------|-------------|----------|
| Legacy | v2.7 | Keyword-based extraction | Quick scans |
| Semantic | v2.8 | LLM-based classification | Accurate categorization |
| Best Practices | v2.9 | Structure-aware, three-matrix | Production CTMs |

---

## 3. Directory Structure

```
/home/user/propelai01/
├── api/                              # FastAPI Backend
│   └── main.py                       # 1,679 lines - All 47 endpoints
│
├── agents/                           # AI Agent System (746KB)
│   ├── __init__.py                   # Conditional imports
│   ├── compliance_agent.py           # 22KB - RFP ingestion
│   ├── strategy_agent.py             # 27KB - Win theme development
│   ├── drafting_agent.py             # 23KB - Content generation
│   ├── red_team_agent.py             # 22KB - Evaluation/scoring
│   │
│   └── enhanced_compliance/          # v3.0 Compliance Engine
│       ├── __init__.py               # Module exports
│       ├── models.py                 # 11KB - Core data models
│       ├── agent.py                  # 20KB - Main orchestrator
│       ├── parser.py                 # 17KB - PDF/DOCX/XLSX parsing
│       ├── extractor.py              # 28KB - Requirement extraction
│       ├── bundle_detector.py        # 13KB - Document classification
│       ├── resolver.py               # 17KB - Cross-reference resolution
│       ├── amendment_processor.py    # 31KB - Amendment tracking
│       ├── excel_export.py           # 30KB - Excel CTM generation
│       ├── ctm_data_models.py        # 27KB - Enhanced CTM models
│       ├── ctm_extractor.py          # 30KB - CTM-specific extraction
│       ├── ctm_integration.py        # 16KB - Integration helpers
│       ├── semantic_extractor.py     # 40KB - v2.8 LLM extraction
│       ├── semantic_ctm_export.py    # 24KB - Semantic export
│       ├── section_aware_extractor.py# 22KB - v2.9 structure-aware
│       ├── best_practices_ctm.py     # 26KB - v2.9 three-matrix
│       ├── document_structure.py     # 26KB - Section detection
│       ├── smart_outline_generator.py# 31KB - v2.10 outline gen
│       ├── annotated_outline_exporter.py  # 9KB - Python wrapper
│       ├── annotated_outline_exporter.js  # Node.js DOCX generator
│       └── company_library.py        # 38KB - Company knowledge base
│
├── core/                             # Orchestration & State (34KB)
│   ├── config.py                     # 7.7KB - Configuration management
│   ├── state.py                      # 8.2KB - ProposalState schema
│   └── orchestrator.py               # 12.7KB - LangGraph orchestrator
│
├── web/                              # React Frontend
│   └── index.html                    # 83KB - Single-file SPA
│
├── tools/                            # Utility modules
├── tests/                            # Test suite
│
├── docker-compose.yml                # Multi-container deployment
├── init.sql                          # PostgreSQL schema
├── requirements.txt                  # Python dependencies
├── package.json                      # Node.js dependencies
├── .env.example                      # Environment template
├── start.sh                          # Startup script
├── cli.py                            # Command-line interface
└── HANDOFF_DOCUMENT.md               # Architecture documentation
```

---

## 4. Core Components

### 4.1 Configuration (core/config.py)

The configuration system uses dataclasses with environment variable overrides.

```python
@dataclass
class APOSConfig:
    """Master configuration for APOS system."""
    llm: LLMConfig
    database: DatabaseConfig
    vector_store: VectorStoreConfig
    security: SecurityConfig
    agents: AgentConfig
```

#### LLM Configuration

| Provider | Model | Purpose | Context Window |
|----------|-------|---------|----------------|
| Google Gemini | gemini-1.5-flash | Extraction (cost-efficient) | 2M tokens |
| Google Gemini | gemini-1.5-pro | Reasoning (powerful) | 2M tokens |
| Anthropic | claude-3-5-sonnet | Complex reasoning | 200K tokens |
| OpenAI | gpt-4-turbo-preview | Fallback | 128K tokens |

#### Database Configuration

```python
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "propelai"
    user: str = "propelai"
    password: str = ""  # Required for production
```

### 4.2 State Management (core/state.py)

The system uses a TypedDict-based state schema for LangGraph compatibility.

#### ProposalState Schema

```python
class ProposalState(TypedDict, total=False):
    # Identifiers
    proposal_id: str
    client_name: str
    opportunity_name: str
    solicitation_number: str

    # Lifecycle
    current_phase: ProposalPhase
    due_date: Optional[str]

    # Core Data
    requirements: Annotated[List[Requirement], operator.add]
    evaluation_criteria: Annotated[List[EvaluationCriterion], operator.add]
    win_themes: Annotated[List[WinTheme], operator.add]
    draft_sections: Annotated[List[DraftSection], operator.add]

    # Quality
    red_team_feedback: Annotated[List[RedTeamFeedback], operator.add]
    quality_score: Optional[float]

    # Governance
    audit_log: Annotated[List[Dict[str, Any]], operator.add]
```

#### Proposal Phases

```python
class ProposalPhase(str, Enum):
    INTAKE = "intake"           # Initial RFP upload
    SHRED = "shred"             # Requirement extraction
    STRATEGY = "strategy"       # Win theme development
    OUTLINE = "outline"         # Structure planning
    DRAFTING = "drafting"       # Content generation
    REVIEW = "review"           # Red team evaluation
    FINALIZE = "finalize"       # Final polish
    SUBMITTED = "submitted"     # Delivered
```

#### Compliance Status

```python
class ComplianceStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DRAFT = "draft"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
```

#### Score Colors (Government Evaluation)

```python
class ScoreColor(str, Enum):
    BLUE = "blue"       # Exceptional - exceeds requirements
    GREEN = "green"     # Acceptable - meets requirements
    YELLOW = "yellow"   # Marginal - partially meets
    RED = "red"         # Unacceptable - fails to meet
```

### 4.3 Orchestrator (core/orchestrator.py)

LangGraph-based supervisor pattern for multi-agent coordination.

```python
class ProposalOrchestrator:
    """Supervisor pattern implementation using LangGraph StateGraph."""

    def __init__(self, config: APOSConfig):
        self.graph = StateGraph(ProposalState)
        self._build_graph()

    def _build_graph(self):
        # Add agent nodes
        self.graph.add_node("compliance", self.compliance_agent)
        self.graph.add_node("strategy", self.strategy_agent)
        self.graph.add_node("drafting", self.drafting_agent)
        self.graph.add_node("red_team", self.red_team_agent)
        self.graph.add_node("human_review", self.human_review_node)

        # Add conditional edges for routing
        self.graph.add_conditional_edges(
            "supervisor",
            self._route_to_agent,
            {
                "compliance": "compliance",
                "strategy": "strategy",
                "drafting": "drafting",
                "red_team": "red_team",
                "human_review": "human_review",
                "end": END
            }
        )
```

#### Agent Routing Logic

```
START → Compliance Agent → Strategy Agent → Drafting Agent → Red Team → Human Review → END
           │                    │                │              │            │
           └────────────────────┴────────────────┴──────────────┴────────────┘
                                    (iterative refinement)
```

---

## 5. Agent System

### 5.1 Compliance Agent ("The Paralegal")

**File:** `agents/compliance_agent.py` (22KB)

**Purpose:** RFP ingestion and requirement extraction

**Responsibilities:**
- Extract "Iron Triangle": Requirements (C), Instructions (L), Evaluation (M)
- Build Requirements Traceability Matrix (RTM)
- Classify requirement types and priorities
- Track source document and page numbers

**LLM Strategy:** Uses Gemini Flash for cost-efficient extraction (model cascading)

```python
class ComplianceAgent:
    """Extracts requirements from RFP documents."""

    def extract_requirements(self, documents: List[Document]) -> List[Requirement]:
        # Parse documents
        # Extract requirements using patterns
        # Classify by type (PERFORMANCE, INSTRUCTION, EVALUATION)
        # Build cross-references
        # Return structured requirements
```

### 5.2 Strategy Agent ("The Capture Manager")

**File:** `agents/strategy_agent.py` (27KB)

**Purpose:** Win theme development and competitive positioning

**Responsibilities:**
- Analyze Section M evaluation factors
- Query past bid strategies from database
- Perform competitor ghosting analysis
- Generate win themes and discriminators
- Create annotated outline with page allocations

**LLM Strategy:** Uses Gemini Pro for strategic reasoning

```python
class StrategyAgent:
    """Develops win themes and capture strategy."""

    def develop_strategy(self, state: ProposalState) -> StrategyOutput:
        # Analyze evaluation criteria weights
        # Identify discriminators
        # Generate win themes per section
        # Map themes to requirements
        # Allocate page budgets
```

### 5.3 Drafting Agent ("The Writer")

**File:** `agents/drafting_agent.py` (23KB)

**Purpose:** Citation-backed content generation

**Critical Policy: ZERO HALLUCINATION**
- Every claim MUST have hyperlinked citation to source
- Uncited claims flagged as "High Risk"
- Works with Research Agent for evidence retrieval

**Voice Styles:**

```python
class VoiceStyle(str, Enum):
    TECHNICAL = "technical"     # Engineering precision
    PERSUASIVE = "persuasive"   # Sales-oriented
    PLAIN = "plain"             # Government plain language
    FORMAL = "formal"           # Legal/contractual
    EXECUTIVE = "executive"     # C-suite summaries
```

**Claim Status Tracking:**

```python
class ClaimStatus(str, Enum):
    CITED = "cited"         # Has source citation
    UNCITED = "uncited"     # Needs citation
    VERIFIED = "verified"   # Citation confirmed
    REJECTED = "rejected"   # Citation invalid
```

### 5.4 Red Team Agent ("The Evaluator")

**File:** `agents/red_team_agent.py` (22KB)

**Purpose:** Government-style evaluation simulation

**Responsibilities:**
- Simulate government evaluator perspective
- Apply color scoring (BLUE/GREEN/YELLOW/RED)
- Identify compliance gaps
- Provide remediation feedback
- Maintain audit log for governance

**Evaluation Categories:**

```python
class EvaluationCategory(str, Enum):
    COMPLIANCE = "compliance"
    TECHNICAL_MERIT = "technical_merit"
    PAST_PERFORMANCE = "past_performance"
    MANAGEMENT = "management"
    PRICE_REASONABLENESS = "price_reasonableness"
```

---

## 6. Enhanced Compliance Engine

### 6.1 Module Overview

The Enhanced Compliance Engine v3.0 (`agents/enhanced_compliance/`) is the core RFP processing system.

| Module | Size | Purpose |
|--------|------|---------|
| `models.py` | 11KB | Core data models and enums |
| `agent.py` | 20KB | Main orchestrator |
| `parser.py` | 17KB | Multi-format document parsing |
| `extractor.py` | 28KB | Requirement extraction |
| `bundle_detector.py` | 13KB | Document type classification |
| `resolver.py` | 17KB | Cross-reference resolution |
| `amendment_processor.py` | 31KB | Amendment tracking |
| `excel_export.py` | 30KB | Excel CTM generation |
| `semantic_extractor.py` | 40KB | LLM-based extraction |
| `section_aware_extractor.py` | 22KB | Structure-aware extraction |
| `best_practices_ctm.py` | 26KB | Three-matrix export |
| `document_structure.py` | 26KB | Section detection |
| `smart_outline_generator.py` | 31KB | Outline generation |
| `company_library.py` | 38KB | Company knowledge base |

### 6.2 Data Models (models.py)

#### Document Types

```python
class DocumentType(str, Enum):
    MAIN_SOLICITATION = "main_solicitation"
    SOW = "sow"                          # Statement of Work
    PWS = "pws"                          # Performance Work Statement
    AMENDMENT = "amendment"
    ATTACHMENT = "attachment"
    BUDGET_TEMPLATE = "budget_template"
    QA_RESPONSE = "qa_response"
    CDRL = "cdrl"                        # Contract Data Requirements List
    PRICING = "pricing"
    INSTRUCTIONS = "instructions"
```

#### Requirement Types

```python
class RequirementType(str, Enum):
    PERFORMANCE = "performance"                  # Section C requirements
    PROPOSAL_INSTRUCTION = "proposal_instruction" # Section L instructions
    EVALUATION_CRITERION = "evaluation_criterion" # Section M criteria
    DELIVERABLE = "deliverable"                  # CDRL items
    LABOR_REQUIREMENT = "labor_requirement"      # Key personnel
    QUALIFICATION = "qualification"              # Certifications
    COMPLIANCE = "compliance"                    # Regulatory
    PROHIBITION = "prohibition"                  # Restrictions
```

#### Requirement Node

```python
@dataclass
class RequirementNode:
    """Core requirement entity with cross-document relationships."""
    id: str
    text: str
    requirement_type: RequirementType
    source_document: str
    source_section: str
    source_page: int
    confidence: float  # 0.0 - 1.0
    priority: str      # HIGH, MEDIUM, LOW

    # Cross-references
    related_requirements: List[str]    # IDs of related requirements
    evaluation_factors: List[str]      # Linked Section M factors
    instruction_refs: List[str]        # Linked Section L instructions

    # Extracted entities
    keywords: List[str]
    actors: List[str]                  # Who must perform
    subjects: List[str]                # What must be done
    constraints: List[str]             # Limitations

    # Amendment tracking
    amendment_history: List[Dict]
    lifecycle_status: str              # ADDED, MODIFIED, DELETED, CLARIFIED
```

#### RFP Bundle

```python
@dataclass
class RFPBundle:
    """Complete RFP package with all related documents."""
    bundle_id: str
    main_solicitation: ParsedDocument
    sow: Optional[ParsedDocument]
    amendments: List[ParsedDocument]
    attachments: List[ParsedDocument]
    qa_responses: List[ParsedDocument]

    # Extracted data
    requirements_graph: Dict[str, RequirementNode]
    cross_reference_edges: List[Tuple[str, str, str]]  # (from, to, relationship)
```

### 6.3 Document Parser (parser.py)

**Multi-format parsing with page-level traceability.**

```python
class MultiFormatParser:
    """Parses PDF, DOCX, and XLSX with section detection."""

    def parse(self, file_path: str) -> ParsedDocument:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)
        elif ext == '.xlsx':
            return self._parse_xlsx(file_path)
```

**Section Detection Patterns (FAR Uniform Contract Format):**

```python
SECTION_PATTERNS = {
    'A': r'(?:SECTION\s*A|PART\s*I).*?SOLICITATION',
    'B': r'(?:SECTION\s*B|SUPPLIES.*?SERVICES)',
    'C': r'(?:SECTION\s*C|DESCRIPTION.*?SPECIFICATIONS|STATEMENT\s*OF\s*WORK)',
    'D': r'(?:SECTION\s*D|PACKAGING.*?MARKING)',
    'E': r'(?:SECTION\s*E|INSPECTION.*?ACCEPTANCE)',
    'F': r'(?:SECTION\s*F|DELIVERIES.*?PERFORMANCE)',
    'G': r'(?:SECTION\s*G|CONTRACT\s*ADMINISTRATION)',
    'H': r'(?:SECTION\s*H|SPECIAL\s*CONTRACT\s*REQUIREMENTS)',
    'I': r'(?:SECTION\s*I|CONTRACT\s*CLAUSES)',
    'J': r'(?:SECTION\s*J|LIST\s*OF\s*ATTACHMENTS)',
    'K': r'(?:SECTION\s*K|REPRESENTATIONS.*?CERTIFICATIONS)',
    'L': r'(?:SECTION\s*L|INSTRUCTIONS.*?CONDITIONS.*?NOTICES)',
    'M': r'(?:SECTION\s*M|EVALUATION\s*FACTORS)',
}
```

### 6.4 Requirement Extractor (extractor.py)

**69+ extraction patterns for federal RFP requirements.**

#### Keyword Patterns

```python
OBLIGATION_KEYWORDS = [
    r'\bshall\b',
    r'\bmust\b',
    r'\brequired\s+to\b',
    r'\bwill\s+be\s+required\b',
    r'\bis\s+required\b',
    r'\bare\s+required\b',
    r'\bmandatory\b',
    r'\bshall\s+not\b',
    r'\bmust\s+not\b',
    r'\bprohibited\b',
]
```

#### Quality Filters

```python
# Minimum/maximum length thresholds
MIN_REQUIREMENT_LENGTH = 100  # characters
MAX_REQUIREMENT_LENGTH = 1000 # characters

# 68 noise patterns to exclude
NOISE_PATTERNS = [
    r'^table\s*of\s*contents',
    r'^page\s*\d+\s*of\s*\d+',
    r'^section\s+[a-m]\s*$',
    r'^\d+\.\d+\s*$',
    r'^continued\s*on\s*next\s*page',
    # ... 63 more patterns
]
```

#### Confidence Scoring

| Level | Score | Criteria |
|-------|-------|----------|
| HIGH | 0.9 | Contains "shall" or "must" + specific deliverable |
| MEDIUM | 0.7 | Contains obligation keyword + context |
| LOW | 0.5 | Implied requirement, needs verification |

### 6.5 Cross-Reference Resolver (resolver.py)

**Builds Requirements Graph with relationship edges.**

```python
class CrossReferenceResolver:
    """Links requirements across sections and documents."""

    RELATIONSHIP_TYPES = {
        'C_TO_L': 'performance_to_instruction',
        'C_TO_M': 'performance_to_evaluation',
        'L_TO_M': 'instruction_to_evaluation',
        'CDRL_TO_C': 'deliverable_to_performance',
        'AMENDMENT_TO_ORIGINAL': 'modifies',
    }

    def build_graph(self, requirements: List[RequirementNode]) -> nx.DiGraph:
        """Build directed graph of requirement relationships."""
        graph = nx.DiGraph()

        for req in requirements:
            graph.add_node(req.id, **asdict(req))

        # Add Iron Triangle edges (C ↔ L ↔ M)
        self._link_sections(graph, 'C', 'L', 'C_TO_L')
        self._link_sections(graph, 'C', 'M', 'C_TO_M')
        self._link_sections(graph, 'L', 'M', 'L_TO_M')

        return graph
```

### 6.6 Amendment Processor (amendment_processor.py)

**Tracks requirement lifecycle through amendments.**

```python
class AmendmentProcessor:
    """Processes SF30 amendments and Q&A responses."""

    class ChangeType(str, Enum):
        ADDED = "added"
        MODIFIED = "modified"
        DELETED = "deleted"
        CLARIFIED = "clarified"

    def process_amendment(self, amendment: ParsedDocument) -> AmendmentResult:
        # Detect amendment format (SF30, NIH, DoD)
        # Extract modification instructions
        # Match to original requirements
        # Track change type and effective date
        # Detect conflicts
```

**Supported Amendment Formats:**

| Format | Agency | Detection Pattern |
|--------|--------|-------------------|
| SF30 | General | `AMENDMENT OF SOLICITATION/MODIFICATION` |
| NIH | HHS | `NIH-\d+` amendment numbering |
| DoD | Defense | `DFARS` clause references |

### 6.7 Excel Export (excel_export.py)

**Generates professional Compliance Traceability Matrix workbooks.**

```python
class ExcelExporter:
    """Creates CTM Excel workbooks with multiple sheets."""

    def export(self, bundle: RFPBundle, output_path: str) -> str:
        workbook = Workbook()

        # Sheet 1: Executive Summary
        self._create_summary_sheet(workbook, bundle)

        # Sheet 2: Full Compliance Matrix
        self._create_full_matrix_sheet(workbook, bundle)

        # Sheet 3: High Priority Requirements
        self._create_priority_sheet(workbook, bundle)

        # Sheet 4: Section L Instructions
        self._create_instructions_sheet(workbook, bundle)

        # Sheet 5: Section M Evaluation
        self._create_evaluation_sheet(workbook, bundle)

        workbook.save(output_path)
        return output_path
```

**CTM Columns:**

| Column | Description |
|--------|-------------|
| Req ID | Unique requirement identifier |
| Section | Source section (A-M) |
| Page | Source page number |
| Requirement Text | Full requirement text |
| Type | PERFORMANCE, INSTRUCTION, EVALUATION |
| Priority | HIGH, MEDIUM, LOW |
| Compliance Status | Dropdown: Compliant, Partial, Non-Compliant |
| Proposal Reference | Cross-reference to proposal section |
| Win Theme | Strategic theme alignment |
| Discriminator | Competitive differentiator |
| Proof Points | Evidence citations |
| Notes | Evaluator comments |

### 6.8 Semantic Extractor (semantic_extractor.py)

**v2.8 LLM-based requirement classification.**

```python
class SemanticRequirementExtractor:
    """Uses LLM for intelligent requirement classification."""

    def extract_with_llm(self, text: str) -> List[SemanticRequirement]:
        prompt = """
        Analyze this RFP text and extract requirements.

        For each requirement, classify as:
        - PERFORMANCE_REQUIREMENT: Technical work to be performed (Section C)
        - PROPOSAL_INSTRUCTION: How to format/submit proposal (Section L)
        - EVALUATION_CRITERION: How proposals will be scored (Section M)

        Return JSON array with:
        - text: Exact requirement text
        - type: Classification
        - confidence: 0.0-1.0
        - rationale: Why this classification
        """

        response = self.llm.generate(prompt + text)
        return self._parse_response(response)
```

### 6.9 Document Structure Analyzer (document_structure.py)

**v2.9 Structure-aware section detection.**

```python
class DocumentStructureAnalyzer:
    """Analyzes RFP structure before extraction."""

    class UCFSection(str, Enum):
        SECTION_A = "A"  # Solicitation/Contract Form
        SECTION_B = "B"  # Supplies/Services and Prices
        SECTION_C = "C"  # Description/Specs/SOW
        SECTION_D = "D"  # Packaging and Marking
        SECTION_E = "E"  # Inspection and Acceptance
        SECTION_F = "F"  # Deliveries or Performance
        SECTION_G = "G"  # Contract Administration
        SECTION_H = "H"  # Special Contract Requirements
        SECTION_I = "I"  # Contract Clauses
        SECTION_J = "J"  # List of Attachments
        SECTION_K = "K"  # Representations/Certifications
        SECTION_L = "L"  # Instructions/Conditions
        SECTION_M = "M"  # Evaluation Factors

    def analyze(self, document: ParsedDocument) -> DocumentStructure:
        # Detect section boundaries
        # Map page ranges
        # Identify SOW/PWS location
        # Detect non-UCF format (GSA, BPA)
```

### 6.10 Proposal Outline Generator (smart_outline_generator.py)

**v2.10 Intelligent outline generation from CTM.**

```python
class SmartOutlineGenerator:
    """Generates proposal outlines from compliance matrix."""

    class VolumeType(str, Enum):
        TECHNICAL = "technical"
        MANAGEMENT = "management"
        PAST_PERFORMANCE = "past_performance"
        COST_PRICE = "cost_price"
        SMALL_BUSINESS = "small_business"
        ORAL_PRESENTATION = "oral_presentation"

    def generate(self, bundle: RFPBundle) -> ProposalOutline:
        outline = ProposalOutline()

        # Detect required volumes from Section L
        volumes = self._detect_volumes(bundle)

        for volume in volumes:
            # Create section hierarchy
            sections = self._create_sections(volume, bundle)

            # Allocate pages based on evaluation weights
            self._allocate_pages(sections, bundle.evaluation_factors)

            # Map requirements to sections
            self._map_requirements(sections, bundle.requirements_graph)

            outline.volumes.append(volume)

        return outline
```

### 6.11 Annotated Outline Exporter

**v2.11 Color-coded DOCX outline generation.**

**Python Wrapper:** `annotated_outline_exporter.py`
**Node.js Generator:** `annotated_outline_exporter.js`

```python
class AnnotatedOutlineExporter:
    """Exports outline to DOCX with color-coded requirements."""

    COLOR_CODING = {
        'L': '#FF0000',  # Red - Section L instructions
        'M': '#0000FF',  # Blue - Section M evaluation
        'C': '#800080',  # Purple - Section C requirements
    }

    def export(self, outline: ProposalOutline, output_path: str) -> str:
        # Prepare data for Node.js
        outline_json = json.dumps(asdict(outline))

        # Call Node.js generator
        result = subprocess.run(
            ['node', 'annotated_outline_exporter.js', outline_json, output_path],
            capture_output=True,
            text=True
        )

        return output_path
```

### 6.12 Company Library (company_library.py)

**Company knowledge base for proposal content.**

```python
class CompanyLibrary:
    """Manages company capabilities and past performance."""

    @dataclass
    class CompanyProfile:
        name: str
        tagline: str
        summary: str
        cage_code: str
        duns_number: str
        certifications: List[str]
        naics_codes: List[str]

    @dataclass
    class PastPerformance:
        contract_name: str
        agency: str
        contract_number: str
        period_of_performance: str
        contract_value: float
        description: str
        relevance_keywords: List[str]

    @dataclass
    class KeyPersonnel:
        name: str
        title: str
        clearance: str
        certifications: List[str]
        experience_years: int
        bio: str
```

---

## 7. API Reference

### 7.1 Endpoint Overview

**Base URL:** `http://localhost:8000/api`

| Category | Endpoints | Description |
|----------|-----------|-------------|
| RFP Management | 9 | Create, list, upload, process |
| Requirements | 2 | Query and filter requirements |
| Export | 4 | Excel and DOCX generation |
| Amendments | 3 | Amendment tracking |
| Company Library | 10 | Knowledge base management |
| System | 1 | Health check |

### 7.2 RFP Management Endpoints

#### Create RFP Project

```http
POST /api/rfp
Content-Type: application/json

{
    "name": "FY25 IT Support Services",
    "solicitation_number": "W911QY-25-R-0001",
    "agency": "Army",
    "due_date": "2025-03-15"
}

Response: 201 Created
{
    "id": "rfp_abc123",
    "name": "FY25 IT Support Services",
    "solicitation_number": "W911QY-25-R-0001",
    "agency": "Army",
    "status": "created",
    "files": [],
    "requirements_count": 0,
    "created_at": "2024-12-25T10:00:00Z",
    "updated_at": "2024-12-25T10:00:00Z"
}
```

#### List RFPs

```http
GET /api/rfp

Response: 200 OK
{
    "rfps": [
        {
            "id": "rfp_abc123",
            "name": "FY25 IT Support Services",
            "status": "processed",
            "requirements_count": 247
        }
    ],
    "total": 1
}
```

#### Upload Documents

```http
POST /api/rfp/{rfp_id}/upload
Content-Type: multipart/form-data

files: [solicitation.pdf, sow.docx, attachment_j.xlsx]

Response: 200 OK
{
    "uploaded": 3,
    "files": [
        {"name": "solicitation.pdf", "size": 2456789, "type": "pdf"},
        {"name": "sow.docx", "size": 345678, "type": "docx"},
        {"name": "attachment_j.xlsx", "size": 123456, "type": "xlsx"}
    ]
}
```

#### Process RFP (Legacy)

```http
POST /api/rfp/{rfp_id}/process

Response: 202 Accepted
{
    "status": "processing",
    "message": "RFP processing started"
}
```

#### Process RFP (Semantic v2.8)

```http
POST /api/rfp/{rfp_id}/process-semantic

Response: 202 Accepted
{
    "status": "processing",
    "mode": "semantic",
    "message": "Semantic extraction started"
}
```

#### Process RFP (Best Practices v2.9)

```http
POST /api/rfp/{rfp_id}/process-best-practices

Response: 202 Accepted
{
    "status": "processing",
    "mode": "best_practices",
    "message": "Best practices extraction started"
}
```

#### Get Processing Status

```http
GET /api/rfp/{rfp_id}/status

Response: 200 OK
{
    "status": "completed",
    "progress": 100,
    "message": "Extraction complete",
    "requirements_count": 247,
    "stats": {
        "by_type": {
            "performance": 89,
            "proposal_instruction": 67,
            "evaluation_criterion": 23,
            "deliverable": 45,
            "other": 23
        },
        "by_priority": {
            "high": 112,
            "medium": 98,
            "low": 37
        }
    }
}
```

### 7.3 Requirements Endpoints

#### Get Requirements (with filters)

```http
GET /api/rfp/{rfp_id}/requirements?type=performance&priority=high&section=C&page=1&limit=50

Response: 200 OK
{
    "requirements": [
        {
            "id": "req_001",
            "text": "The contractor shall provide 24/7 help desk support...",
            "section": "C.3.1",
            "type": "performance",
            "priority": "high",
            "confidence": 0.95,
            "source_page": 42,
            "keywords": ["help desk", "24/7", "support"],
            "related_requirements": ["req_045", "req_089"]
        }
    ],
    "total": 89,
    "page": 1,
    "limit": 50
}
```

### 7.4 Export Endpoints

#### Export to Excel

```http
GET /api/rfp/{rfp_id}/export

Response: 200 OK (application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)
[Binary Excel file]
```

#### Generate Proposal Outline

```http
POST /api/rfp/{rfp_id}/outline

Response: 200 OK
{
    "outline_id": "outline_xyz789",
    "volumes": [
        {
            "type": "technical",
            "title": "Volume I: Technical Approach",
            "page_limit": 50,
            "sections": [...]
        }
    ]
}
```

#### Export Outline to DOCX

```http
GET /api/rfp/{rfp_id}/outline/export

Response: 200 OK (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
[Binary DOCX file]
```

### 7.5 Amendment Endpoints

#### Upload Amendment

```http
POST /api/rfp/{rfp_id}/amendments
Content-Type: multipart/form-data

file: amendment_001.pdf
amendment_number: 1
amendment_date: 2025-01-15

Response: 200 OK
{
    "amendment_id": "amd_001",
    "changes": {
        "added": 5,
        "modified": 12,
        "deleted": 2,
        "clarified": 8
    }
}
```

#### Get Amendment Report

```http
GET /api/rfp/{rfp_id}/amendments/report

Response: 200 OK
{
    "amendments": [
        {
            "number": 1,
            "date": "2025-01-15",
            "changes": [
                {
                    "requirement_id": "req_045",
                    "change_type": "modified",
                    "original_text": "...",
                    "new_text": "...",
                    "rationale": "Clarified delivery schedule"
                }
            ]
        }
    ]
}
```

### 7.6 Company Library Endpoints

#### Get Library Overview

```http
GET /api/library

Response: 200 OK
{
    "profile": {...},
    "documents_count": 45,
    "capabilities_count": 23,
    "past_performance_count": 12,
    "key_personnel_count": 8
}
```

#### Search Library

```http
GET /api/library/search?q=cybersecurity&type=capability

Response: 200 OK
{
    "results": [
        {
            "type": "capability",
            "title": "Cybersecurity Operations",
            "relevance": 0.95,
            "source_document": "capability_statement_2024.pdf"
        }
    ]
}
```

### 7.7 Health Check

```http
GET /api/health

Response: 200 OK
{
    "status": "healthy",
    "version": "2.11.0",
    "components": {
        "api": "ok",
        "database": "ok",
        "vector_store": "ok",
        "llm": "ok"
    },
    "uptime": 86400
}
```

---

## 8. Data Models

### 8.1 Request Models

```python
class RFPCreate(BaseModel):
    name: str
    solicitation_number: Optional[str] = None
    agency: Optional[str] = None
    due_date: Optional[str] = None

class AmendmentUpload(BaseModel):
    amendment_number: int
    amendment_date: str

class LibraryUpload(BaseModel):
    document_type: str  # capability, past_performance, personnel
    tags: List[str] = []
```

### 8.2 Response Models

```python
class RFPResponse(BaseModel):
    id: str
    name: str
    solicitation_number: Optional[str]
    agency: Optional[str]
    status: str
    files: List[FileInfo]
    requirements_count: int
    created_at: datetime
    updated_at: datetime

class RequirementResponse(BaseModel):
    id: str
    text: str
    section: str
    type: str
    priority: str
    confidence: float
    source_page: int
    keywords: List[str]
    related_requirements: List[str]

class ProcessingStatus(BaseModel):
    status: str
    progress: int
    message: str
    requirements_count: int
    stats: Optional[Dict]
```

---

## 9. Processing Pipelines

### 9.1 Legacy Pipeline (v2.7)

```
Upload → Parse → Keyword Extract → Dedupe → Store
           │
           └── Simple pattern matching
               "shall", "must", "required"
```

### 9.2 Semantic Pipeline (v2.8)

```
Upload → Parse → LLM Classification → Filter → Store
           │            │
           │            └── Gemini Flash
           │                - PERFORMANCE vs INSTRUCTION
           │                - Confidence scoring
           │                - Entity extraction
           │
           └── Aggressive garbage filtering
               - 68 noise patterns
               - Length thresholds
               - Boilerplate detection
```

### 9.3 Best Practices Pipeline (v2.9)

```
Upload → Structure Analysis → Section-Aware Extract → Three-Matrix Export
              │                      │
              │                      └── Preserves RFP numbering
              │                          L.4.B.2, C.3.1, etc.
              │
              └── UCF Section Detection
                  - Identify A-M boundaries
                  - Detect SOW/PWS location
                  - Handle non-UCF (GSA, BPA)
```

### 9.4 Background Processing

```python
async def process_rfp_background(rfp_id: str, mode: str = "legacy"):
    """Background task for RFP processing."""
    try:
        store.set_status(rfp_id, "processing", 0, "Starting extraction...")

        # Load documents
        documents = load_documents(rfp_id)
        store.set_status(rfp_id, "processing", 10, "Documents loaded")

        # Detect bundle structure
        bundle = BundleDetector().detect_bundle(documents)
        store.set_status(rfp_id, "processing", 20, "Bundle detected")

        # Parse documents
        parsed = MultiFormatParser().parse_bundle(bundle)
        store.set_status(rfp_id, "processing", 40, "Documents parsed")

        # Extract requirements (mode-dependent)
        if mode == "semantic":
            requirements = SemanticExtractor().extract(parsed)
        elif mode == "best_practices":
            requirements = SectionAwareExtractor().extract(parsed)
        else:
            requirements = RequirementExtractor().extract(parsed)
        store.set_status(rfp_id, "processing", 70, "Requirements extracted")

        # Resolve cross-references
        graph = CrossReferenceResolver().resolve(requirements)
        store.set_status(rfp_id, "processing", 90, "Cross-references resolved")

        # Store results
        store.update(rfp_id, {
            "requirements": requirements,
            "requirements_graph": graph,
            "stats": calculate_stats(requirements)
        })

        store.set_status(rfp_id, "completed", 100, "Extraction complete")

    except Exception as e:
        store.set_status(rfp_id, "error", 0, str(e))
```

---

## 10. Database Schema

### 10.1 PostgreSQL Tables

```sql
-- LangGraph state persistence
CREATE TABLE checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    checkpoint_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX idx_checkpoints_created ON checkpoints(created_at);

-- Proposal metadata
CREATE TABLE proposals (
    proposal_id VARCHAR(50) PRIMARY KEY,
    client_name VARCHAR(255),
    opportunity_name VARCHAR(500),
    solicitation_number VARCHAR(100),
    due_date DATE,
    current_phase VARCHAR(50) DEFAULT 'intake',
    quality_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent audit trail
CREATE TABLE agent_trace_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id VARCHAR(50) REFERENCES proposals(proposal_id),
    agent_name VARCHAR(100) NOT NULL,
    action VARCHAR(255) NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    reasoning_trace TEXT,
    duration_ms INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trace_proposal ON agent_trace_log(proposal_id);
CREATE INDEX idx_trace_agent ON agent_trace_log(agent_name);
CREATE INDEX idx_trace_created ON agent_trace_log(created_at);

-- Human feedback for data flywheel
CREATE TABLE human_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    proposal_id VARCHAR(50) REFERENCES proposals(proposal_id),
    section_id VARCHAR(100),
    feedback_type VARCHAR(50) NOT NULL, -- edit, reject, approve, comment
    original_content TEXT,
    corrected_content TEXT,
    correction_reason TEXT,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_proposal ON human_feedback(proposal_id);
CREATE INDEX idx_feedback_type ON human_feedback(feedback_type);
```

### 10.2 In-Memory Store (Development)

```python
class RFPStore:
    """In-memory storage for RFP projects."""

    def __init__(self):
        self._rfps: Dict[str, Dict] = {}
        self._status: Dict[str, ProcessingStatus] = {}

    def create(self, rfp_id: str, data: Dict) -> Dict:
        self._rfps[rfp_id] = {
            "id": rfp_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            **data
        }
        return self._rfps[rfp_id]

    def get(self, rfp_id: str) -> Optional[Dict]:
        return self._rfps.get(rfp_id)

    def update(self, rfp_id: str, data: Dict) -> Dict:
        if rfp_id in self._rfps:
            self._rfps[rfp_id].update(data)
            self._rfps[rfp_id]["updated_at"] = datetime.utcnow()
        return self._rfps.get(rfp_id)

    def set_status(self, rfp_id: str, status: str, progress: int, message: str):
        self._status[rfp_id] = ProcessingStatus(
            status=status,
            progress=progress,
            message=message
        )

    def get_status(self, rfp_id: str) -> Optional[ProcessingStatus]:
        return self._status.get(rfp_id)
```

---

## 11. Frontend Architecture

### 11.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | React 18 |
| Build | Unbundled (Babel in-browser) |
| Styling | CSS Variables, Custom CSS |
| HTTP Client | Fetch API |
| State | React Hooks (useState, useEffect) |

### 11.2 Application Structure

**File:** `web/index.html` (83KB single-file SPA)

```html
<!DOCTYPE html>
<html>
<head>
    <title>PropelAI - RFP Analysis</title>
    <style>
        /* CSS Variables */
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --accent-blue: #4f8cff;
            --accent-green: #34d399;
            --accent-amber: #fbbf24;
            --accent-red: #f87171;
            --accent-purple: #a78bfa;
        }
        /* ... styles ... */
    </style>
</head>
<body>
    <div id="root"></div>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script type="text/babel">
        // React Application
        function App() {
            const [currentPage, setCurrentPage] = useState('rfp-list');
            const [rfps, setRfps] = useState([]);
            // ... application logic
        }

        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>
```

### 11.3 Key Components

| Component | Purpose |
|-----------|---------|
| `App` | Root component, routing |
| `Sidebar` | Navigation menu |
| `RFPList` | List of RFP projects |
| `UploadZone` | Drag-and-drop file upload |
| `RequirementsTable` | Paginated, filterable requirements |
| `ProcessingStatus` | Progress indicator |
| `ExportControls` | Excel/DOCX download buttons |
| `AmendmentManager` | Amendment upload and tracking |

### 11.4 Color Scheme

```css
/* Dark Theme */
--bg-primary: #0a0a0f;
--bg-secondary: #12121a;
--bg-tertiary: #1a1a24;

/* Accent Colors */
--accent-blue: #4f8cff;      /* Primary actions */
--accent-green: #34d399;     /* Success states */
--accent-amber: #fbbf24;     /* Warnings */
--accent-red: #f87171;       /* Errors, Section L */
--accent-purple: #a78bfa;    /* Section C */

/* Gradient */
--gradient-primary: linear-gradient(135deg, #4f8cff 0%, #7c3aed 100%);
```

---

## 12. Integration Points

### 12.1 LLM Providers

#### Google Gemini (Primary)

```python
# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")

# Models
GEMINI_FLASH = "gemini-1.5-flash"  # Extraction (cost-efficient)
GEMINI_PRO = "gemini-1.5-pro"      # Reasoning (powerful)

# Context Window: 2M tokens
```

#### Anthropic Claude (Alternative)

```python
# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model
CLAUDE_SONNET = "claude-3-5-sonnet-20241022"

# Context Window: 200K tokens
```

#### OpenAI (Fallback)

```python
# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model
GPT4_TURBO = "gpt-4-turbo-preview"

# Context Window: 128K tokens
```

### 12.2 Vector Stores

#### Chroma (Development)

```python
# Configuration
CHROMA_PERSIST_DIR = "./data/chroma"

# Usage
from chromadb import Client
client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_PERSIST_DIR
))
```

#### Pinecone (Production)

```python
# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "propelai-proposals"

# Usage
import pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)
```

### 12.3 Document Processing Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `pypdf` | 3.0+ | PDF text extraction with page tracking |
| `python-docx` | 0.8.11+ | DOCX reading and writing |
| `openpyxl` | 3.0.10+ | XLSX parsing and generation |
| `markitdown` | latest | Fallback for complex layouts |

### 12.4 Node.js Integration

**DOCX Generation via subprocess:**

```python
# annotated_outline_exporter.py
def export_outline(outline: ProposalOutline, output_path: str) -> str:
    outline_json = json.dumps(asdict(outline))

    result = subprocess.run(
        ['node', 'annotated_outline_exporter.js', outline_json, output_path],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        raise RuntimeError(f"DOCX generation failed: {result.stderr}")

    return output_path
```

**Node.js package.json:**

```json
{
    "dependencies": {
        "docx": "^8.5.0"
    }
}
```

---

## 13. Configuration Reference

### 13.1 Environment Variables

```bash
# Environment
PROPELAI_ENV=development  # development, staging, production

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# LLM Providers
GOOGLE_API_KEY=your_google_api_key
GOOGLE_PROJECT_ID=your_project_id
ANTHROPIC_API_KEY=your_anthropic_key  # Optional
OPENAI_API_KEY=your_openai_key        # Optional

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=propelai
POSTGRES_USER=propelai
POSTGRES_PASSWORD=secure_password

# Vector Store
PINECONE_API_KEY=your_pinecone_key    # Optional
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=propelai-proposals

# Security
API_SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret

# Feature Flags
ENABLE_OCR=true
ENABLE_AUDIT_LOG=true
ENABLE_COMPETITOR_GHOSTING=true
```

### 13.2 Configuration Classes

```python
@dataclass
class APOSConfig:
    llm: LLMConfig
    database: DatabaseConfig
    vector_store: VectorStoreConfig
    security: SecurityConfig
    agents: AgentConfig

@dataclass
class LLMConfig:
    primary_provider: str = "google"
    primary_model: str = "gemini-1.5-flash"
    reasoning_model: str = "gemini-1.5-pro"
    fallback_provider: str = "anthropic"
    fallback_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.1
    max_tokens: int = 8192

@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "propelai"
    user: str = "propelai"
    password: str = ""

@dataclass
class VectorStoreConfig:
    provider: str = "chroma"  # chroma or pinecone
    chroma_persist_directory: str = "./data/chroma"
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "propelai-proposals"

@dataclass
class SecurityConfig:
    enable_fedramp: bool = False
    enable_encryption: bool = True
    enable_audit_log: bool = True
    api_secret_key: str = ""
    jwt_secret: str = ""

@dataclass
class AgentConfig:
    enable_compliance_agent: bool = True
    enable_strategy_agent: bool = True
    enable_drafting_agent: bool = True
    enable_red_team_agent: bool = True
    max_iterations: int = 10
    human_in_loop: bool = True
```

---

## 14. Deployment

### 14.1 Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: propelai-api
    ports:
      - "8000:8000"
    environment:
      - PROPELAI_ENV=production
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=propelai
      - POSTGRES_USER=propelai
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./data/uploads:/app/data/uploads
      - ./data/outputs:/app/data/outputs
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    container_name: propelai-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=propelai
      - POSTGRES_USER=propelai
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U propelai"]
      interval: 10s
      timeout: 5s
      retries: 5

  chroma:
    image: chromadb/chroma:latest
    container_name: propelai-chroma
    ports:
      - "8001:8000"
    volumes:
      - ./data/chroma:/chroma/chroma

volumes:
  postgres_data:
```

### 14.2 Startup Script

```bash
#!/bin/bash
# start.sh

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Install dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Run database migrations (if using PostgreSQL)
# python -m alembic upgrade head

# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 14.3 Production Deployment

**Recommended Platforms:**
- Render.com (PaaS)
- Railway.com (PaaS)
- AWS ECS/Fargate
- Google Cloud Run

**Production Checklist:**
- [ ] Set `PROPELAI_ENV=production`
- [ ] Configure PostgreSQL with SSL
- [ ] Set up Pinecone vector store
- [ ] Configure CORS for specific domains
- [ ] Enable audit logging
- [ ] Set up monitoring (Datadog, New Relic)
- [ ] Configure backup for PostgreSQL
- [ ] Set up CI/CD pipeline

---

## 15. Technical Specifications

### 15.1 Performance Metrics

| Metric | Value |
|--------|-------|
| Max upload size | 50MB per file |
| Supported file types | PDF, DOCX, XLSX |
| Max pages per document | 500 |
| Avg extraction time | 30-60 seconds per 100 pages |
| Max concurrent users | 50 (development), 500+ (production) |
| API response time | < 200ms (non-processing endpoints) |

### 15.2 System Requirements

#### Development

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| Node.js | 18+ |
| Memory | 4GB RAM |
| Storage | 10GB |

#### Production

| Component | Requirement |
|-----------|-------------|
| Python | 3.11+ |
| Node.js | 18+ |
| Memory | 8GB RAM |
| Storage | 50GB+ |
| PostgreSQL | 15+ |
| Workers | 4+ uvicorn workers |

### 15.3 Security Considerations

- **Authentication**: JWT-based API authentication
- **Authorization**: Role-based access control
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **FedRAMP**: Optional compliance mode
- **Audit Logging**: All agent actions logged
- **Data Isolation**: Per-tenant data segregation

### 15.4 Scalability

- **Horizontal Scaling**: Stateless API, multiple instances behind load balancer
- **Database Scaling**: PostgreSQL read replicas
- **Vector Store**: Pinecone scales automatically
- **Background Processing**: Celery/Redis for distributed task queue (planned)
- **Caching**: Redis for API response caching (planned)

---

## Appendix A: Extraction Patterns

### A.1 Obligation Keywords

```python
OBLIGATION_PATTERNS = [
    r'\bshall\b',
    r'\bmust\b',
    r'\brequired\s+to\b',
    r'\bwill\s+be\s+required\b',
    r'\bis\s+required\b',
    r'\bare\s+required\b',
    r'\bmandatory\b',
    r'\bshall\s+not\b',
    r'\bmust\s+not\b',
    r'\bprohibited\b',
]
```

### A.2 Semantic Requirement Patterns

```python
SEMANTIC_PATTERNS = {
    'PERFORMANCE': [
        r'contractor\s+shall\s+(?:provide|perform|deliver|maintain)',
        r'(?:services|work)\s+(?:shall|must)\s+(?:include|consist)',
    ],
    'PROPOSAL_INSTRUCTION': [
        r'offeror\s+shall\s+(?:submit|provide|include|demonstrate)',
        r'proposal\s+(?:shall|must)\s+(?:include|contain|address)',
    ],
    'EVALUATION_CRITERION': [
        r'will\s+be\s+evaluated\s+(?:on|based\s+on)',
        r'evaluation\s+(?:factor|criterion|criteria)',
    ],
    'DELIVERABLE': [
        r'(?:deliver|submit|provide)\s+(?:the\s+)?following',
        r'CDRL\s+[A-Z]\d+',
    ],
}
```

---

## Appendix B: FAR Uniform Contract Format

| Section | Title | Content |
|---------|-------|---------|
| A | Solicitation/Contract Form | SF33, basic info |
| B | Supplies/Services and Prices | CLINs, pricing |
| C | Description/Specs/SOW | Technical requirements |
| D | Packaging and Marking | Shipping requirements |
| E | Inspection and Acceptance | QA requirements |
| F | Deliveries or Performance | Schedule, milestones |
| G | Contract Administration | COR, invoicing |
| H | Special Contract Requirements | Custom clauses |
| I | Contract Clauses | FAR/DFARS clauses |
| J | List of Attachments | CDRLs, exhibits |
| K | Representations/Certifications | SAM, certs |
| L | Instructions/Conditions | Proposal format |
| M | Evaluation Factors | Scoring criteria |

---

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.7 | 2024-10 | Legacy keyword extraction |
| 2.8 | 2024-11 | Semantic LLM classification |
| 2.9 | 2024-11 | Structure-aware extraction, best practices |
| 2.10 | 2024-12 | Smart outline generation |
| 2.11 | 2024-12 | Annotated outline export, color coding |

---

*Document generated: December 2024*
*PropelAI APOS v2.11.0*
