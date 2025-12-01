# PropelAI - Technical As-Built Documentation

**Version:** 4.0  
**Last Updated:** December 2024  
**Document Purpose:** Complete technical reference for development team onboarding

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Directory Structure](#directory-structure)
5. [Data Layer](#data-layer)
6. [Backend Services](#backend-services)
7. [Frontend Application](#frontend-application)
8. [AI Agents & Prompts](#ai-agents--prompts)
9. [API Reference](#api-reference)
10. [Key Workflows](#key-workflows)
11. [Deployment](#deployment)
12. [Environment Configuration](#environment-configuration)
13. [Testing](#testing)
14. [Known Limitations](#known-limitations)
15. [Future Roadmap](#future-roadmap)

---

## System Overview

### What is PropelAI?

PropelAI is an **AI-powered proposal intelligence platform** for government contractors. It automates the analysis of complex Request for Proposals (RFPs), extracts requirements, generates compliance matrices, and provides intelligent chat-based Q&A with dual-context RAG (RFP documents + company capability library).

### Core Value Proposition

1. **Requirement Extraction**: Automated extraction from 500+ page RFPs in 60 seconds
2. **Dual-Context RAG**: Chat that understands BOTH the RFP AND your company's capabilities
3. **Compliance Matrix Generation**: Export production-ready Excel files (L-M-C matrices)
4. **Intelligent Routing**: Automatically detects RFP type (FAR 15, GSA, OTA, SBIR, Spreadsheet, RFI) and applies appropriate analysis logic
5. **Amendment Tracking**: Cross-reference amendments and detect conflicts

### Key Differentiators

- **Not Generic AI**: Specialized prompts trained on FAR/DFARS regulations
- **Company-Specific Analysis**: "Can WE win this RFP?" vs generic document Q&A
- **Mode-Adaptive**: 6 different analysis modes based on RFP type
- **Source Citations**: All answers include page numbers and section references

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│                   (React Single-Page App)                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    FASTAPI BACKEND                           │
│                   (Python 3.11+)                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  API Routes  │  │  RFP Store   │  │  File Storage   │  │
│  │  (main.py)   │  │  (db.py)     │  │  (/outputs/)    │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  AI AGENT LAYER                       │  │
│  │  ┌───────────┐ ┌────────────┐ ┌──────────────────┐  │  │
│  │  │ RFP Chat  │ │ Compliance │ │ Company Library  │  │  │
│  │  │   Agent   │ │   Agent    │ │      Agent       │  │  │
│  │  └───────────┘ └────────────┘ └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌────▼────┐  ┌─────▼──────┐
    │  MongoDB  │  │ File    │  │  Anthropic │
    │  (Local)  │  │ Storage │  │  Claude API│
    │  OR       │  │ (JSON)  │  │  (External)│
    │  JSON DB  │  │         │  │            │
    └───────────┘  └─────────┘  └────────────┘
```

### Data Flow

1. **Upload**: User uploads RFP (PDF/DOCX) → Saved to `/outputs/{rfp_id}/`
2. **Processing**: Backend extracts text → Chunks documents → Stores in DB
3. **Chat**: User asks question → Agent queries DB + Company Library → Claude API → Response with citations
4. **Export**: User requests matrix → Agent generates Excel → Download

### Deployment Model

- **Current**: Single-tenant on Render.com
- **Container**: Docker (FastAPI + React served from same process)
- **Storage**: File-based JSON (persistent disk mount at `/outputs/`)
- **Future**: Multi-tenant SaaS with database backend

---

## Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | FastAPI | 0.100+ | REST API, async support |
| **Runtime** | Python | 3.11+ | Core language |
| **Database** | JSON Files (Phase 5) | N/A | RFP data, chat history |
| **Alternative DB** | MongoDB (Optional) | 4.x+ | Production scaling |
| **Document Parsing** | pypdf, python-docx, openpyxl | Latest | PDF/DOCX/Excel extraction |
| **LLM Integration** | Anthropic Claude | API | Chat & analysis |
| **Excel Generation** | openpyxl, pandas | Latest | Compliance matrix export |
| **Web Server** | Uvicorn | 0.23+ | ASGI server |

### Frontend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | React | 18.x | UI framework |
| **Build** | None (vanilla) | N/A | Single HTML file |
| **State Management** | React Hooks | Native | useState, useEffect |
| **HTTP Client** | Fetch API | Native | API calls |
| **Styling** | CSS (custom) | N/A | Dark theme |
| **Icons** | Lucide React | Latest | UI icons |

### AI/ML

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Claude (Anthropic) | Text generation, analysis |
| **Models** | claude-sonnet-4-5, claude-haiku-4-5 | Primary models |
| **RAG** | Custom implementation | Document chunking & retrieval |
| **Embedding** | Keyword-based (no vectors yet) | Fast retrieval |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Hosting** | Render.com | Cloud platform |
| **Process Manager** | Supervisor | Service management |
| **Storage** | Persistent Disk | File storage |
| **Logging** | Supervisor logs | Application logging |

---

## Directory Structure

```
/app/
├── api/                          # Backend API layer
│   ├── main.py                   # FastAPI app, endpoints (1859 lines)
│   ├── server.py                 # Supervisor startup wrapper
│   ├── db.py                     # JSON database layer (Phase 5)
│   └── .env                      # Environment variables (not in repo)
│
├── agents/                       # AI Agent implementations
│   ├── chat/
│   │   ├── rfp_chat_agent.py    # Main chat agent (1500+ lines)
│   │   │                         # - v4.0 Omni-Federal system prompt
│   │   │                         # - 6-mode router
│   │   │                         # - RAG implementation
│   │   │                         # - Company Library integration
│   │   └── __init__.py
│   │
│   ├── enhanced_compliance/      # Compliance & extraction agents
│   │   ├── agent.py              # Main compliance agent
│   │   ├── extractor.py          # Basic requirement extractor
│   │   ├── section_aware_extractor.py  # Mode-aware extractor
│   │   ├── semantic_extractor.py # Semantic analysis (optional)
│   │   ├── best_practices_ctm.py # Best practices extractor
│   │   ├── smart_outline_generator.py  # Outline generation
│   │   ├── company_library.py    # Company doc management
│   │   ├── amendment_processor.py # Amendment tracking
│   │   ├── excel_export.py       # Excel matrix export
│   │   ├── parser.py             # Document parsing
│   │   ├── excel_parser.py       # Excel RFP parsing
│   │   └── models.py             # Pydantic data models
│   │
│   ├── compliance_agent.py       # Legacy compliance agent
│   ├── strategy_agent.py         # Strategy analysis
│   └── red_team_agent.py         # Proposal review
│
├── web/                          # Frontend application
│   ├── index.html                # Monolithic React SPA (3000+ lines)
│   │                             # - All components inline
│   │                             # - React + Lucide icons via CDN
│   │                             # - Custom dark theme CSS
│   └── demo.html                 # Investor demo landing page
│
├── outputs/                      # File storage directory
│   ├── data/                     # JSON database files (Phase 5)
│   │   ├── rfps.json            # RFP metadata & requirements
│   │   ├── chat_history.json   # Chat message logs
│   │   └── library.json         # Company library index
│   │
│   ├── company_library/          # Company documents
│   │   ├── documents/           # Uploaded company files
│   │   └── index.json           # Library metadata (legacy)
│   │
│   └── {rfp_id}/                # Per-RFP file storage
│       ├── original_files/      # Uploaded RFP documents
│       └── exports/             # Generated Excel files
│
├── tests/                        # Test files
│   ├── test_agents.py
│   └── test_result.md           # Testing protocol document
│
├── requirements.txt              # Python dependencies
├── package.json                  # Node dependencies (minimal)
├── supervisord.conf              # Process manager config
├── Dockerfile                    # Container definition
├── render.yaml                   # Render deployment config
│
└── Documentation/
    ├── INVESTOR_DEMO_SCRIPT.md
    ├── QUICK_START_CARD.md
    ├── DEPLOYMENT_GUIDE.md
    └── TECHNICAL_AS_BUILT.md    # This document
```

---

## Data Layer

### Phase 5: JSON File Database

**Location**: `/app/outputs/data/`

**Implementation**: `/app/api/db.py`

#### Design Philosophy

- **Simplicity**: Pure Python, no external database dependencies
- **Stability**: File-based storage eliminates connection issues
- **Performance**: Adequate for single-tenant use cases
- **Persistence**: Data survives container restarts (with persistent disk)

#### Database Class

```python
class JSONFileDB:
    """
    Simple file-based JSON storage with immediate persistence.
    
    Storage:
    - rfps.json: RFP metadata and requirements
    - chat_history.json: Chat messages by RFP ID
    - library.json: Company library metadata
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        # Smart path resolution: project root or /tmp fallback
        # Thread-safe with locking
        # Atomic writes with temp file + rename
```

#### Data Files

**1. `rfps.json`**
```json
{
  "RFP-ABC12345": {
    "id": "RFP-ABC12345",
    "name": "Coast Guard Intelligence Support",
    "solicitation_number": "HQ-2024-001",
    "agency": "US Coast Guard",
    "due_date": "2024-12-31",
    "status": "processed",
    "files": ["rfp_base.pdf", "amendment_1.pdf"],
    "file_paths": ["/outputs/RFP-ABC12345/..."],
    "requirements": [...],  // Extracted requirements
    "requirements_graph": {...},  // Graph structure
    "stats": {...},  // Statistics
    "amendments": [...],  // Amendment history
    "document_chunks": [...],  // For chat RAG
    "chat_history": [...],  // Chat messages
    "rfp_type": "federal_standard",  // Router classification
    "created_at": "2024-11-30T12:00:00Z",
    "updated_at": "2024-11-30T13:30:00Z"
  }
}
```

**2. `chat_history.json`**
```json
{
  "RFP-ABC12345": [
    {
      "role": "user",
      "content": "What are the evaluation factors?",
      "timestamp": "2024-11-30T14:00:00Z"
    },
    {
      "role": "assistant",
      "content": "Based on Section M...",
      "timestamp": "2024-11-30T14:00:05Z",
      "sources": [...]
    }
  ]
}
```

**3. `library.json`**
```json
{
  "DOC-UUID-1": {
    "id": "DOC-UUID-1",
    "filename": "capabilities_statement.pdf",
    "hash": "sha256:abc...",
    "type": "capabilities",
    "size": 1024000,
    "parse_date": "2024-11-30T10:00:00Z",
    "content": {...},  // Parsed content
    "version": 1
  }
}
```

#### Key Methods

```python
# RFP Operations
db.rfps_insert_one(document)      # Create RFP
db.rfps_find_one({"id": rfp_id})  # Read RFP
db.rfps_update_one(query, update) # Update RFP
db.rfps_delete_one({"id": rfp_id})# Delete RFP
db.rfps_find()                     # List all RFPs

# Chat History
db.chat_history_insert_one(message)
db.chat_history_find({"rfp_id": rfp_id})

# Company Library
db.library_insert_one(document)
db.library_find()
db.library_delete_one({"id": doc_id})
```

#### Migration Path (Future)

For multi-tenant production, replace with:
- **MongoDB**: Full-featured database with indexing
- **PostgreSQL**: Relational option with JSON support
- **Redis**: Fast cache layer

File paths remain unchanged; only `db.py` needs refactoring.

---

## Backend Services

### Main API (`/app/api/main.py`)

**Size**: 1859 lines  
**Framework**: FastAPI  
**Purpose**: REST API, business logic, agent orchestration

#### Key Components

**1. RFPStore Class** (Line 167-247)
```python
class RFPStore:
    """File-based JSON store for RFP data"""
    
    def create(self, rfp_id: str, data: Dict) -> Dict
    def get(self, rfp_id: str) -> Optional[Dict]
    def update(self, rfp_id: str, updates: Dict) -> Dict
    def list_all(self) -> List[Dict]
    def delete(self, rfp_id: str) -> bool
    def set_status(...)  # Processing status
    def get_status(...)  # Get status
```

**2. Global Instances** (Line 262-310)
```python
store = RFPStore()                    # Data store
compliance_agent = EnhancedComplianceAgent()  # Compliance
amendment_processor = AmendmentProcessor()    # Amendments
best_practices_extractor = ...        # Best practices
company_library = CompanyLibrary()    # Company docs
rfp_chat_agent = RFPChatAgent()      # Chat with RAG
```

**3. Background Processing Functions**
```python
def process_rfp_background(rfp_id: str)
    # Standard extraction

def process_rfp_semantic_background(rfp_id: str)
    # Semantic extraction (experimental)

def process_rfp_best_practices_background(rfp_id: str)
    # Best practices extraction (recommended)
```

#### API Endpoints

**RFP Management**
```python
POST   /api/rfp                       # Create RFP
GET    /api/rfp                       # List all RFPs
GET    /api/rfp/{rfp_id}             # Get RFP details
DELETE /api/rfp/{rfp_id}             # Delete RFP
POST   /api/rfp/{rfp_id}/upload      # Upload files
POST   /api/rfp/{rfp_id}/process     # Start extraction
GET    /api/rfp/{rfp_id}/status      # Processing status
```

**Requirements & Analysis**
```python
GET    /api/rfp/{rfp_id}/requirements       # Get requirements
GET    /api/rfp/{rfp_id}/requirements-graph # Graph data
POST   /api/rfp/{rfp_id}/outline           # Generate outline
POST   /api/rfp/{rfp_id}/compliance-matrix # Generate matrix
```

**Chat (RAG)**
```python
POST   /api/rfp/{rfp_id}/chat              # Chat with RFP
```

**Exports**
```python
GET    /api/rfp/{rfp_id}/export            # Excel download
POST   /api/rfp/{rfp_id}/export-semantic   # Semantic export
POST   /api/rfp/{rfp_id}/export-best-practices  # Best practices
```

**Amendments**
```python
POST   /api/rfp/{rfp_id}/amendments        # Upload amendment
GET    /api/rfp/{rfp_id}/amendments        # Get amendments
POST   /api/rfp/{rfp_id}/amendments/resolve # Resolve conflicts
```

**Company Library**
```python
POST   /api/library/upload                 # Upload company docs
GET    /api/library                        # Get library status
DELETE /api/library/documents/{doc_id}     # Delete document
GET    /api/library/search                 # Search library
```

**Utilities**
```python
GET    /api/health                         # Health check
GET    /                                   # Serve frontend
```

---

## Frontend Application

### Architecture

**Type**: Single-Page Application (SPA)  
**File**: `/app/web/index.html` (3000+ lines)  
**Framework**: React 18 (via CDN)  
**Styling**: Custom CSS (dark theme)  
**Icons**: Lucide React

### Component Structure (All in One File)

```javascript
// Main Application
function App() {
  // Root component
  // Manages global state, routing
}

// Core Views
function HomeView() { }           // Upload & dashboard
function RFPDetailView() { }      // RFP analysis view
function ChatView() { }           // Chat interface
function LibraryView() { }        // Company library (v4.0)

// Sub-Components
function RFPCard() { }            // RFP list item
function RequirementCard() { }    // Requirement display
function ChatMessage() { }        // Chat bubble
function FileUploader() { }       // Drag-and-drop
function Icon() { }               // Lucide icon wrapper
```

### State Management

```javascript
// Global State (in App component)
const [rfps, setRfps] = useState([])                    // All RFPs
const [currentRfp, setCurrentRfp] = useState(null)      // Selected RFP
const [requirements, setRequirements] = useState([])    // Requirements
const [chatHistory, setChatHistory] = useState([])      // Chat messages
const [currentView, setCurrentView] = useState('home')  // View routing
```

### Key Features

**1. Drag-and-Drop Upload**
- Supports PDF, DOCX, Excel
- Visual feedback during upload
- Progress indication
- Multi-file support

**2. Real-Time Processing Status**
- Polls `/api/rfp/{id}/status` every 2 seconds
- Progress bar (0-100%)
- Status messages
- Completion notification

**3. Requirements Grid**
- Filterable by section (L, M, C, PWS)
- Color-coded by priority
- Expandable details
- Source citations

**4. Chat Interface (v4.0)**
- Message history
- Streaming responses
- Source citations with page numbers
- Company Library integration
- "Starter chips" for common questions

**5. Company Library (v4.0)**
- Drag-and-drop multi-file upload
- File staging with category dropdowns
- Document table with actions
- Search functionality

### Styling

**Theme**: Dark mode with purple/blue accent  
**Colors**:
- Primary: `#667eea` (purple-blue)
- Background: `#0f0f1e`
- Card: `#1a1a2e`
- Text: `#e0e0e0`
- Muted: `#a0a0a0`

**Typography**:
- System fonts: `-apple-system, BlinkMacSystemFont, 'Segoe UI'`
- Monospace for code: `'Courier New', monospace`

---

## AI Agents & Prompts

### 1. RFP Chat Agent (`/app/agents/chat/rfp_chat_agent.py`)

**Size**: 1500+ lines  
**Purpose**: Intelligent chat with RFP documents + Company Library

#### Core Features

**a) v4.0 "Omni-Federal" System Prompt** (Line 972-1100)

The heart of PropelAI's intelligence. 1,000+ line prompt containing:

```python
"""
# SYSTEM PROMPT: PROPELAI PROPOSAL ARCHITECT (v4.0)

## 1. IDENTITY & MISSION
You are the PropelAI Proposal Architect, an elite capture strategist.

## 2. PHASE I: CLASSIFICATION (The "Super-Router")
Upon receiving a document, classify into one of 6 Federal Modes:

MODE A: STANDARD FAR 15 (The "Iron Triangle")
- Triggers: Sections A-M, "Section L", "Section M"
- Protocol: Enforce L-M-C alignment

MODE B: GSA / IDIQ TASK ORDER (The "Agile Order")
- Triggers: "RFQ", "GSA Schedule", "BPA Call"
- Protocol: Cover Letter Supremacy

MODE C: OTA / CSO (The "Innovation Pitch")
- Triggers: "Other Transaction Authority", "CSO"
- Protocol: Merit over Compliance

MODE D: R&D / SBIR / BAA (The "Scientific Method")
- Triggers: "SBIR", "Phase I", "BAA"
- Protocol: Rigorous Science

MODE E: SPREADSHEET / QUESTIONNAIRE (The "Data Entry")
- Triggers: Excel files, "J.2", "Questionnaire"
- Protocol: Cell-Constraint

MODE F: MARKET RESEARCH / RFI (The "Soft Sell")
- Triggers: "Sources Sought", "RFI"
- Protocol: Influence Strategy

## 3. PHASE II: OPERATIONAL PROTOCOLS
- The "Shall" Detective: Mandatory vs. Preference
- The "Buried Limit" Search: Find hidden page limits
- The "Gap Analysis": Flag unrewarded compliance
- The "Forensic Scan": Trace requirements to source
- Amendment Conflict Detection
"""
```

**b) 6-Mode Router** (Line 195-350)

```python
class RFPType(Enum):
    FEDERAL_STANDARD = "federal_standard"    # MODE A
    SLED_STATE = "sled_state"               # MODE B (GSA)
    DOD_ATTACHMENT = "dod_attachment"        # MODE C (OTA)
    SPREADSHEET = "spreadsheet"              # MODE E
    MARKET_RESEARCH = "market_research"      # MODE F (RFI)
    UNKNOWN = "unknown"

def detect_rfp_type(self, text: str) -> RFPType:
    """
    Detect RFP type using keyword analysis.
    
    Returns one of 6 modes to determine analysis protocol.
    """
```

**c) Company Library RAG Integration** (Line 105-250)

```python
def _detect_library_intent(self, question: str) -> bool:
    """
    Detect if user is asking about company capabilities.
    
    Triggers:
    - Pronouns: "we", "our", "us"
    - Keywords: "experience", "capability", "past performance"
    """

def _query_company_library(self, query: str, top_k: int = 3) -> List[Dict]:
    """
    Query company library for relevant content.
    
    Returns matching capabilities, past performance, resumes.
    """

def _format_library_context(self, results: List[Dict]) -> str:
    """
    Format library results for LLM context.
    
    Adds section like:
    === CONTEXT FROM COMPANY LIBRARY ===
    [Source: Company Capabilities]
    Capability: Cloud Modernization
    Description: ...
    """
```

**d) Chat Method with Dual-RAG** (Line 1402-1490)

```python
def chat(
    self,
    question: str,
    document_chunks: List[DocumentChunk],
    chat_history: Optional[List[ChatMessage]] = None
) -> ChatMessage:
    """
    Main chat function with Company Library integration.
    
    Flow:
    1. Detect if asking about company (library intent)
    2. If yes: Query company_library AND RFP chunks
    3. Combine contexts
    4. Send to Claude with specialized prompt
    5. Return answer with sources
    """
```

**e) Specialized Query Handlers**

```python
def handle_cross_reference_query(...)
    # Section L vs. M cross-reference analysis

def handle_contradiction_detection(...)
    # Find conflicts between sections/amendments

def handle_formatting_query(...)
    # Extract Section L formatting rules
```

**f) Document Chunking**

```python
def chunk_rfp_documents(self, rfp: Dict) -> List[DocumentChunk]:
    """
    Split documents into overlapping chunks for RAG.
    
    Config:
    - chunk_size: 1000 chars
    - chunk_overlap: 200 chars
    - Preserves section context
    """
```

**g) Retrieval**

```python
def retrieve_relevant_chunks(
    self,
    question: str,
    document_chunks: List[DocumentChunk],
    top_k: int = 20
) -> List[Tuple[DocumentChunk, float]]:
    """
    Keyword-based retrieval (no embeddings yet).
    
    Scoring:
    - TF-IDF style keyword matching
    - Section prioritization
    - Returns top K chunks
    """
```

### 2. Enhanced Compliance Agent (`/app/agents/enhanced_compliance/agent.py`)

**Purpose**: Extract requirements from RFP documents

#### Extractors

**a) Basic Extractor** (`extractor.py`)
- Rule-based requirement extraction
- Keyword matching ("shall", "must", "will")
- Section detection

**b) Section-Aware Extractor** (`section_aware_extractor.py`)
- Mode-aware extraction (uses router)
- L-M-C section mapping
- Context preservation

**c) Best Practices Extractor** (`best_practices_ctm.py`)
- Document structure analysis FIRST
- Preserves RFP numbering scheme
- Generates L-M-C matrices
- Excel export with formulas

**d) Semantic Extractor** (`semantic_extractor.py`) [Optional]
- Advanced NLP analysis
- Requirement classification
- Actor/action extraction
- Cross-reference detection

#### Processing Flow

```python
1. Upload RFP files → /outputs/{rfp_id}/original_files/

2. Parse documents:
   - PDF: pypdf
   - DOCX: python-docx
   - Excel: openpyxl

3. Detect RFP type (router logic)

4. Extract text by section (L, M, C, PWS, SOW)

5. Apply mode-specific extraction rules

6. Generate requirements list with metadata:
   {
     "id": "REQ-001",
     "text": "The contractor shall...",
     "section": "L.4.2.1",
     "page": 45,
     "priority": "mandatory",
     "type": "technical",
     "evaluation_factor": "Factor 1"
   }

7. Store in db.rfps
```

### 3. Company Library Agent (`/app/agents/enhanced_compliance/company_library.py`)

**Purpose**: Manage company document library for RAG

#### Features

**a) Document Parsing**
```python
def add_document(self, file_path: str, doc_type: str, metadata: Dict):
    """
    Parse and index company document.
    
    Supported:
    - PDF: Extract text, identify capabilities
    - DOCX: Parse structured content
    - TXT/MD: Direct text extraction
    
    Content types:
    - capabilities: Company capabilities/services
    - past_performance: Project summaries
    - resume: Staff qualifications
    - technical_approach: White papers
    """
```

**b) Content Extraction**

Parses different document types into structured format:

```python
# Capabilities Document
{
  "type": "capability",
  "content": {
    "name": "Cloud Modernization",
    "description": "...",
    "use_cases": [...],
    "certifications": [...]
  }
}

# Past Performance
{
  "type": "past_performance",
  "content": {
    "project_name": "USCG Intelligence Modernization",
    "client": "US Coast Guard",
    "contract_value": "$5M",
    "period": "2020-2023",
    "scope": "...",
    "outcomes": [...]
  }
}

# Resume
{
  "type": "resume",
  "content": {
    "name": "John Doe",
    "clearance": "TS/SCI",
    "certifications": ["PMP", "CISSP"],
    "skills": [...],
    "experience": [...]
  }
}
```

**c) SHA-256 Hashing**
```python
def _calculate_file_hash(self, file_path: str) -> str:
    """
    Calculate SHA-256 hash of file content.
    
    Purpose: Duplicate detection
    """
```

**d) Versioning**
```python
# If file with same name but different content:
# - Keep original as "doc.pdf" (version 1)
# - Save new as "doc_v2.pdf" (version 2)
```

**e) Search**
```python
def search(self, query: str) -> List[Dict]:
    """
    Keyword-based search across company library.
    
    Returns ranked results with content snippets.
    """
```

### 4. Amendment Processor (`/app/agents/enhanced_compliance/amendment_processor.py`)

**Purpose**: Track and analyze RFP amendments

#### Features

```python
def process_amendment(self, base_rfp: Dict, amendment_file: str):
    """
    Compare amendment to base RFP.
    
    Detects:
    - Changed requirements
    - New requirements
    - Deleted requirements
    - Modified deadlines
    - Page limit changes
    """

def detect_conflicts(self, amendments: List):
    """
    Cross-reference all amendments.
    
    Flags:
    - Contradictions between amendments
    - Requirements that changed multiple times
    - Critical vs. minor changes
    """
```

### 5. Smart Outline Generator (`/app/agents/enhanced_compliance/smart_outline_generator.py`)

**Purpose**: Generate proposal outlines based on RFP

#### Features

```python
def generate_outline(self, rfp: Dict, mode: str) -> Dict:
    """
    Generate proposal outline using mode-specific logic.
    
    Modes:
    - MODE A (FAR 15): L-M-C aligned outline
    - MODE B (GSA): Task Order format
    - MODE C (OTA): Innovation brief structure
    - MODE D (SBIR): Technical volume format
    """
```

---

## API Reference

### Authentication

**Current**: None (single-tenant)  
**Future**: JWT tokens for multi-tenant

### Request/Response Format

**Content-Type**: `application/json`  
**File Uploads**: `multipart/form-data`

### Common Headers

```
Content-Type: application/json
Accept: application/json
```

### Error Responses

```json
{
  "detail": "Error message here"
}
```

**Status Codes**:
- `200`: Success
- `400`: Bad request
- `404`: Not found
- `500`: Server error
- `503`: Service unavailable (e.g., chat agent not initialized)

### Endpoint Details

#### POST /api/rfp

Create new RFP project.

**Request**:
```json
{
  "name": "Coast Guard Intelligence Support",
  "solicitation_number": "HQ-2024-001",
  "agency": "US Coast Guard",
  "due_date": "2024-12-31"
}
```

**Response**:
```json
{
  "id": "RFP-ABC12345",
  "name": "Coast Guard Intelligence Support",
  "solicitation_number": "HQ-2024-001",
  "agency": "US Coast Guard",
  "status": "created",
  "files": [],
  "requirements_count": 0,
  "created_at": "2024-11-30T12:00:00Z",
  "updated_at": "2024-11-30T12:00:00Z"
}
```

#### POST /api/rfp/{rfp_id}/upload

Upload RFP documents.

**Request**: `multipart/form-data`
```
files: [File, File, ...]
```

**Response**:
```json
{
  "rfp_id": "RFP-ABC12345",
  "uploaded_files": ["rfp_base.pdf", "attachment_j2.xlsx"],
  "file_count": 2
}
```

#### POST /api/rfp/{rfp_id}/process

Start requirement extraction.

**Request**: Empty body or:
```json
{
  "mode": "best_practices"  // or "standard", "semantic"
}
```

**Response**:
```json
{
  "status": "processing_started",
  "rfp_id": "RFP-ABC12345",
  "files_count": 2
}
```

#### GET /api/rfp/{rfp_id}/status

Get processing status.

**Response**:
```json
{
  "status": "processing",  // or "complete", "error"
  "progress": 75,
  "message": "Extracting requirements from Section C...",
  "requirements_count": 42
}
```

#### POST /api/rfp/{rfp_id}/chat

Chat with RFP (RAG).

**Request**:
```json
{
  "message": "Do our capabilities match this RFP?",
  "include_sources": true
}
```

**Response**:
```json
{
  "answer": "Based on the RFP requirements and your company library...",
  "sources": [
    {
      "id": "chunk-1",
      "text": "Section L.4.2.1 requires...",
      "section": "L.4.2.1",
      "page": 45,
      "file": "rfp_base.pdf"
    }
  ],
  "timestamp": "2024-11-30T14:05:00Z"
}
```

#### POST /api/library/upload

Upload company documents.

**Request**: `multipart/form-data`
```
file: File
tag: "capabilities" | "past_performance" | "resume" | "technical_approach" | "context_rfi"
```

**Response**:
```json
{
  "success": true,
  "document_id": "DOC-UUID-1",
  "filename": "capabilities_statement.pdf",
  "hash": "sha256:abc...",
  "duplicate": false,
  "version": 1
}
```

---

## Key Workflows

### Workflow 1: Upload & Process RFP

```
1. User drops PDF into upload area

2. Frontend: POST /api/rfp
   → Creates RFP record
   → Returns rfp_id

3. Frontend: POST /api/rfp/{rfp_id}/upload
   → Uploads file(s)
   → Saves to /outputs/{rfp_id}/

4. Frontend: POST /api/rfp/{rfp_id}/process
   → Triggers background processing

5. Backend: process_rfp_best_practices_background(rfp_id)
   a. Parse PDF → extract text
   b. Detect RFP type (router)
   c. Extract requirements (mode-specific)
   d. Store in db.rfps

6. Frontend: Polls GET /api/rfp/{rfp_id}/status
   → Shows progress bar
   → Updates UI when complete

7. User views requirements grid
```

### Workflow 2: Chat with RFP + Company Library

```
1. User uploads company docs to Library
   → POST /api/library/upload (multiple times)
   → CompanyLibrary parses and indexes docs

2. User uploads RFP
   → Same as Workflow 1

3. User types: "Do our capabilities match this RFP?"

4. Frontend: POST /api/rfp/{rfp_id}/chat
   {
     "message": "Do our capabilities match this RFP?"
   }

5. Backend: rfp_chat_agent.chat()
   a. Detect library intent → TRUE
   b. Query company_library.search("capabilities")
      → Returns matching docs
   c. Retrieve RFP chunks (keyword-based)
   d. Combine contexts:
      - RFP requirements
      - Company capabilities
   e. Send to Claude with v4.0 prompt
   f. Parse response

6. Frontend: Display answer with sources
   - RFP citations (page numbers)
   - Company library citations
```

### Workflow 3: Generate Compliance Matrix

```
1. RFP already processed (has requirements)

2. User clicks "Generate Compliance Matrix"

3. Frontend: POST /api/rfp/{rfp_id}/compliance-matrix
   OR
   Frontend: GET /api/rfp/{rfp_id}/export

4. Backend: best_practices_ctm.export_to_excel()
   a. Load requirements from db
   b. Group by section (L, M, C)
   c. Create Excel workbook with:
      - Sheet 1: Section L Compliance
      - Sheet 2: Technical Requirements (C/PWS)
      - Sheet 3: Section M Evaluation
   d. Add formulas, formatting
   e. Save to /outputs/{rfp_id}/exports/

5. Backend: Return FileResponse

6. Frontend: Downloads Excel file
```

### Workflow 4: Amendment Processing

```
1. User uploads amendment document

2. Frontend: POST /api/rfp/{rfp_id}/amendments

3. Backend: amendment_processor.process_amendment()
   a. Parse amendment document
   b. Extract changed requirements
   c. Compare to base RFP
   d. Detect:
      - New requirements
      - Modified requirements
      - Deleted requirements
      - Conflicting changes
   e. Store amendment history

4. Frontend: Display amendment analysis
   - Red: Deleted/Changed
   - Green: New
   - Yellow: Conflicts
```

---

## Deployment

### Current: Render.com

**Service Type**: Web Service  
**Region**: US East  
**Plan**: Starter (upgradeable)

#### Build Configuration

**Build Command**:
```bash
pip install -r requirements.txt
```

**Start Command**:
```bash
supervisord -c supervisord.conf
```

#### Persistent Disk

**Mount Path**: `/app/outputs`  
**Size**: 1GB (recommended minimum)  
**Purpose**: Store RFP files, database, exports

**Without persistent disk**: Data uses `/tmp/propelai_data/` (ephemeral)

#### Environment Variables

```bash
# Optional: API key for chat functionality
ANTHROPIC_API_KEY=sk-ant-...

# Auto-set by db.py if not present
MONGO_URL=mongodb://localhost:27017/propelai  # If using MongoDB

# Port (auto-set by Render)
PORT=8001
```

#### Supervisor Configuration

**File**: `/app/supervisord.conf`

```ini
[supervisord]
nodaemon=true

[program:backend]
command=uvicorn api.server:app --host 0.0.0.0 --port 8001
directory=/app
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/backend.out.log
stderr_logfile=/var/log/supervisor/backend.err.log

[program:frontend]
# Frontend served by FastAPI (static files)
# No separate process needed
```

#### Health Check

**Endpoint**: `GET /api/health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-30T12:00:00Z",
  "version": "4.0.0",
  "components": {
    "enhanced_compliance_agent": "ready",
    "amendment_processor": "ready",
    "excel_export": "ready",
    "semantic_extractor": "not available",  // If not configured
    "best_practices_extractor": "ready",
    "rfp_chat_agent": "ready"
  }
}
```

### Docker Deployment (Alternative)

**Dockerfile**: `/app/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create outputs directory
RUN mkdir -p /app/outputs/data

# Expose port
EXPOSE 8001

# Start supervisord
CMD ["supervisord", "-c", "supervisord.conf"]
```

**Build**:
```bash
docker build -t propelai:4.0 .
```

**Run**:
```bash
docker run -p 8001:8001 \
  -v $(pwd)/outputs:/app/outputs \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  propelai:4.0
```

---

## Environment Configuration

### Required Files

**1. `/app/requirements.txt`**
```
# API Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0
python-dotenv>=1.0.0

# Document Processing
pypdf>=3.0.0
python-docx>=0.8.11
openpyxl>=3.0.10

# Utils
httpx>=0.24.0

# AI/LLM Integration
anthropic>=0.21.0
pandas>=2.0.0
```

**2. `/app/api/.env` (Optional)**
```bash
# Anthropic API Key (for chat)
ANTHROPIC_API_KEY=sk-ant-...

# Database URL (if using MongoDB)
# MONGO_URL=mongodb://localhost:27017/propelai
```

### Configuration Precedence

1. Environment variables (system)
2. `.env` file
3. Defaults in code

### Feature Flags

**In Code** (`/app/api/main.py`):

```python
# Chat availability
RFP_CHAT_AVAILABLE = os.getenv("ANTHROPIC_API_KEY") is not None

# Semantic extraction (experimental)
SEMANTIC_AVAILABLE = False  # Manually enable

# Best practices extraction (recommended)
BEST_PRACTICES_AVAILABLE = True
```

---

## Testing

### Test Framework

**Current**: Manual testing  
**Future**: pytest + playwright

### Test Protocol

**Location**: `/app/tests/test_result.md`

**Process**:
1. Upload sample RFP
2. Process with all 3 extractors
3. Verify requirements extracted
4. Test chat with sample questions
5. Upload company docs to library
6. Test capability matching questions
7. Generate compliance matrix
8. Upload amendment, verify conflict detection

### Sample Test Cases

**Test 1: Basic Upload**
```bash
curl -X POST http://localhost:8001/api/rfp/{id}/upload \
  -F "files=@sample_rfp.pdf"
```

**Test 2: Chat**
```bash
curl -X POST http://localhost:8001/api/rfp/{id}/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the evaluation factors?"}'
```

**Test 3: Library Search**
```bash
curl http://localhost:8001/api/library/search?query=cloud
```

### Known Test Data

**Sample RFPs**:
- GSA Schedule RFQ
- SBIR Phase I
- Standard FAR 15 solicitation
- Excel questionnaire
- RFI/Sources Sought

---

## Known Limitations

### Technical Debt

1. **Monolithic Frontend** (3000+ line HTML file)
   - Should be refactored into React components
   - Hard to maintain and debug
   - No build process

2. **File-Based Database**
   - Not suitable for multi-tenant
   - No transactions or rollback
   - Manual locking required

3. **Keyword-Based Retrieval**
   - No vector embeddings yet
   - Less accurate than semantic search
   - Scalability limits

4. **Single-Tenant Architecture**
   - No user authentication
   - No data isolation
   - One company at a time

### Functional Limitations

1. **Chat Requires API Key**
   - Users must provide Anthropic key
   - No fallback model
   - Cost visibility unclear

2. **No Real-Time Collaboration**
   - Single user per RFP
   - No comments or annotations
   - No team workflows

3. **Limited Export Formats**
   - Excel only (no Word)
   - Fixed template
   - No customization

4. **No Email/Notifications**
   - Processing completion not notified
   - No amendment alerts
   - Manual checking required

### Performance Constraints

1. **Large Files (>100MB)**
   - Slow parsing
   - High memory usage
   - May timeout

2. **Complex Excel Files**
   - Formula errors possible
   - Merged cell issues
   - Format loss

3. **Long Chat History**
   - Context window limits (200K tokens)
   - Old messages not summarized
   - Performance degradation

---

## Future Roadmap

### Phase 6: Production Multi-Tenant (Q1 2025)

**Database Migration**:
- [ ] Replace JSON files with PostgreSQL or MongoDB
- [ ] Add user authentication (JWT)
- [ ] Implement data isolation (tenant_id)
- [ ] Add audit logging

**User Management**:
- [ ] Sign up / login
- [ ] Team invitations
- [ ] Role-based access (Admin, Analyst, Reviewer)
- [ ] Usage quotas

### Phase 7: Advanced Features (Q2 2025)

**Vector Search**:
- [ ] Implement embedding models (OpenAI, Cohere)
- [ ] Vector database (Pinecone, Weaviate)
- [ ] Semantic retrieval
- [ ] Improved relevance

**Collaboration**:
- [ ] Real-time co-editing
- [ ] Comments and annotations
- [ ] Task assignments
- [ ] Version control for proposals

**Integrations**:
- [ ] Deltek integration (opportunity feed)
- [ ] GovWin API
- [ ] Microsoft 365 (SharePoint, Teams)
- [ ] Box.com document sync
- [ ] SAM.gov direct feed

### Phase 8: Advanced AI (Q3 2025)

**Auto-Shred**:
- [ ] Automatically assign RFP sections to writers
- [ ] Generate writing instructions
- [ ] Pre-populate outline with content suggestions

**Proposal Generation**:
- [ ] Generate proposal text (not just analysis)
- [ ] Compliance checking
- [ ] Style and tone matching
- [ ] Automated red team review

**Predictive Analytics**:
- [ ] Win probability scoring
- [ ] Competitive intelligence
- [ ] Historical win theme analysis
- [ ] Pricing recommendations

### Phase 9: Enterprise Features (Q4 2025)

**Security**:
- [ ] FedRAMP certification
- [ ] On-prem deployment option
- [ ] Air-gapped mode (no internet)
- [ ] Classified document handling

**Reporting**:
- [ ] Pipeline dashboards
- [ ] Win rate analytics
- [ ] Team productivity metrics
- [ ] Cost per proposal tracking

**Customization**:
- [ ] Custom compliance templates
- [ ] Configurable workflows
- [ ] White-label branding
- [ ] API for third-party integrations

---

## Appendix A: Key File Locations

```
/app/api/main.py                          # Main API (1859 lines)
/app/api/db.py                            # JSON database layer
/app/api/server.py                        # Supervisor wrapper
/app/agents/chat/rfp_chat_agent.py       # Chat agent (1500+ lines)
/app/agents/enhanced_compliance/best_practices_ctm.py  # Best extractor
/app/agents/enhanced_compliance/company_library.py     # Company library
/app/web/index.html                       # Frontend SPA (3000+ lines)
/app/outputs/data/rfps.json              # RFP database
/app/outputs/data/chat_history.json      # Chat logs
/app/outputs/data/library.json           # Company library index
/app/requirements.txt                     # Python dependencies
/app/supervisord.conf                     # Process manager config
/app/Dockerfile                           # Container definition
/app/DEPLOYMENT_GUIDE.md                  # Deployment instructions
/app/INVESTOR_DEMO_SCRIPT.md             # Demo guide
```

---

## Appendix B: Critical Code Snippets

### Router Logic (Detect RFP Type)

```python
# From /app/agents/chat/rfp_chat_agent.py
def detect_rfp_type(self, text: str) -> RFPType:
    # MODE A: Standard FAR 15
    if 'section l' in text_lower and 'section m' in text_lower:
        return RFPType.FEDERAL_STANDARD
    
    # MODE B: GSA/Task Order
    if any(kw in text_lower for kw in ['gsaschedule', 'bpa call', 'task order']):
        return RFPType.SLED_STATE
    
    # MODE C: OTA
    if 'other transaction' in text_lower or 'cso' in text_lower:
        return RFPType.DOD_ATTACHMENT
    
    # MODE E: Spreadsheet
    if '.xlsx' in text_lower or 'questionnaire' in text_lower:
        return RFPType.SPREADSHEET
    
    # MODE F: RFI
    if 'sources sought' in text_lower or 'rfi' in text_lower:
        return RFPType.MARKET_RESEARCH
    
    return RFPType.UNKNOWN
```

### Company Library Intent Detection

```python
# From /app/agents/chat/rfp_chat_agent.py
def _detect_library_intent(self, question: str) -> bool:
    question_lower = question.lower()
    
    # Company pronouns
    company_pronouns = ['we', 'our', 'us', 'have we', 'do we']
    
    # Capability keywords
    capability_keywords = [
        'experience', 'capability', 'past performance',
        'proof', 'resume', 'personnel', 'team',
        'competitive advantage', 'differentiator'
    ]
    
    has_pronoun = any(p in question_lower for p in company_pronouns)
    has_keyword = any(k in question_lower for k in capability_keywords)
    
    return has_pronoun or has_keyword
```

### JSON Database Operations

```python
# From /app/api/db.py
class JSONFileDB:
    def rfps_insert_one(self, document: Dict) -> Dict:
        with self._rfps_lock:
            rfp_id = document.get('id')
            self._rfps_data[rfp_id] = document
            self._save_json(self.rfps_file, self._rfps_data)
            return document
    
    def _save_json(self, file_path: Path, data: Any):
        temp_file = file_path.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.replace(file_path)  # Atomic rename
```

---

## Appendix C: Common Debugging Commands

### Check Service Status
```bash
sudo supervisorctl status
```

### View Backend Logs
```bash
tail -f /var/log/supervisor/backend.out.log
tail -f /var/log/supervisor/backend.err.log
```

### Test API
```bash
# Health check
curl http://localhost:8001/api/health

# List RFPs
curl http://localhost:8001/api/rfp

# Check library
curl http://localhost:8001/api/library
```

### Restart Services
```bash
sudo supervisorctl restart backend
```

### Check Database Files
```bash
ls -lh /app/outputs/data/
cat /app/outputs/data/rfps.json | jq
```

### Python Console Debug
```python
import sys
sys.path.insert(0, '/app')
from api.db import db
db.get_stats()
```

---

## Appendix D: Dependencies Detail

### Python Packages

```
fastapi==0.104.1           # REST framework
uvicorn==0.24.0           # ASGI server
pydantic==2.5.0           # Data validation
python-multipart==0.0.6   # File uploads
python-dotenv==1.0.0      # Environment config
pypdf==3.17.0             # PDF parsing
python-docx==1.1.0        # DOCX parsing
openpyxl==3.1.2           # Excel read/write
pandas==2.1.3             # Data manipulation
anthropic==0.21.3         # Claude API client
httpx==0.25.2             # HTTP client
```

### CDN Resources (Frontend)

```html
<!-- React -->
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>

<!-- Lucide Icons -->
<script src="https://unpkg.com/lucide@latest"></script>

<!-- Babel (JSX) -->
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
```

---

## Appendix E: Glossary

**Terms**:

- **RFP**: Request for Proposal - Government solicitation document
- **FAR**: Federal Acquisition Regulation - Rules governing US Gov contracting
- **Section L**: Proposal instructions section in FAR 15 RFPs
- **Section M**: Evaluation criteria section in FAR 15 RFPs
- **Section C**: Statement of Work (SOW) or Performance Work Statement (PWS)
- **Compliance Matrix**: Table mapping RFP requirements to proposal sections
- **RAG**: Retrieval-Augmented Generation - AI technique for document Q&A
- **CTM**: Compliance Traceability Matrix - Another term for compliance matrix
- **GSA**: General Services Administration - Federal procurement agency
- **OTA**: Other Transaction Authority - Non-FAR contract vehicle
- **SBIR**: Small Business Innovation Research - R&D grant program
- **RFI**: Request for Information - Pre-solicitation market research
- **Amendment**: Modification to original RFP after release

**Acronyms**:

- **SPA**: Single-Page Application
- **API**: Application Programming Interface
- **LLM**: Large Language Model
- **NLP**: Natural Language Processing
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **JWT**: JSON Web Token
- **CORS**: Cross-Origin Resource Sharing
- **ASGI**: Asynchronous Server Gateway Interface

---

## Document Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2024-11-30 | 1.0 | Initial as-built documentation |

---

**END OF DOCUMENT**

For questions or clarifications, refer to:
- Code comments in source files
- `/app/DEPLOYMENT_GUIDE.md` for deployment specifics
- `/app/INVESTOR_DEMO_SCRIPT.md` for feature demonstrations
- Git commit history for implementation timeline
