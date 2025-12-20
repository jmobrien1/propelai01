# PropelAI: Autonomous Proposal Operating System
**Current Status:** v4.1 - Stateful Agentic Architecture with pgvector semantic search.
**All Core Phases Complete:** Trust Gate ✓ | Iron Triangle ✓ | Drafting Agent ✓ | Persistence ✓

## 1. Architecture & Tech Stack
- **Backend:** Python 3.10+ / FastAPI / Uvicorn.
- **Frontend:** React (Single Page App via CDN). *Note: No npm build step. We edit `web/index.html` directly.*
- **Orchestration:** LangGraph (moving away from linear chains).
- **Storage:** PostgreSQL with `pgvector` (replacing SQLite/In-Memory).
- **PDF Engine:** Dual-Pipeline.
  - Tier 1: `pypdf` for digital-native files (Fast/Free).
  - Tier 2: `pdfplumber` for coordinate extraction (bounding boxes).
  - Tier 3: `Tensorlake` for scanned/complex files (Cost/Accurate).

## 2. Core Commands
- **Start Server:** `python -m uvicorn api.main:app --reload --port 8000`
- **Run Extraction Test:** `python agents/enhanced_compliance/section_aware_extractor.py`
- **Docker Build:** `docker-compose up --build`
- **Deploy:** `git push origin main` (Triggers Render.com auto-deploy).

## 3. Development Guidelines (The "Iron Triangle")
1. **Trust First:** Never generate text without a source citation.
2. **Traceability:** All extracted requirements must map to a `SourceCoordinate` (page/x/y).
3. **Immutability:** Do not modify `AS_BUILT_TDD.md` unless architecture changes. It is the source of truth.

## 4. Key Files
- `api/main.py`: Primary API entry point (3500+ lines).
- `api/vector_store.py`: pgvector semantic search for Company Library.
- `api/database.py`: PostgreSQL ORM with SQLAlchemy 2.0.
- `agents/strategy_agent.py`: Iron Triangle logic engine (StrategyAgent, CompetitorAnalyzer).
- `agents/drafting_workflow.py`: LangGraph drafting workflow (F-B-P framework).
- `agents/enhanced_compliance/pdf_coordinate_extractor.py`: Trust Gate coordinate extraction.
- `agents/enhanced_compliance/document_types.py`: Guided upload document classification.
- `web/index.html`: React SPA with guided upload wizard (v4.1 dark theme).
- `init.sql`: PostgreSQL schema with pgvector extension.

## 5. v4.0 Architecture Phases

### Phase 1: Trust Gate (Source Traceability Engine) - COMPLETE ✓
**Goal:** Visual overlays proving extraction accuracy via `SourceCoordinate`.

**Data Models** (`document_structure.py`):
- `BoundingBox`: Normalized coordinates (x, y, width, height) 0.0-1.0
- `SourceCoordinate`: Links requirements to visual locations (document_id, page_index, visual_rects)

**Parser Upgrade** (`parser.py`):
- Primary: `pypdf` for fast text extraction
- Secondary: `pdfplumber` for character-level bounding box extraction
- Normalization: PDF coords (bottom-left origin, points) → Web coords (top-left origin, percentage)

**API Endpoint**:
- `GET /api/rfp/{rfp_id}/requirements/{req_id}/source` - Returns bounding boxes for frontend overlay

### Phase 2: Iron Triangle Logic Engine - COMPLETE ✓
**Goal:** Model dependencies between Section L, M, and C. Move from "Shredding" to "Reasoning".

**Data Models** (`document_structure.py`):
- `WinTheme`: Discriminators and proof points for competitive advantage
- `CompetitorProfile`: Known weaknesses and likely approaches for ghosting strategies
- `EvaluationFactor`: Section M scoring weights and sub-factors
- `StructureConflict`: Detected conflicts between L/M/C sections

**Strategy Agent** (`agents/strategy_agent.py`):
- `StrategyAgent`: L-M-C analysis and win theme generation
- `CompetitorAnalyzer`: Competitive landscape analysis
- `GhostingLanguageGenerator`: Competitive differentiation language

**API Endpoints**:
- `GET /api/rfp/{rfp_id}/strategy` - Returns L-M-C analysis
- `POST /api/rfp/{rfp_id}/strategy` - Generate strategy
- `POST /api/rfp/{rfp_id}/competitive-analysis` - Analyze competitors

### Phase 3: Drafting Agent (F-B-P Framework) - COMPLETE ✓
**Goal:** Prevent LLM hallucination via structured planning.

**LangGraph Workflow** (`agents/drafting_workflow.py`):
- `research_node`: Query Company Library for evidence
- `structure_fbp_node`: Build Feature-Benefit-Proof blocks
- `draft_node`: Generate narrative prose
- `quality_check_node`: Score draft on compliance, clarity, citations
- `human_review_node`: Pause for human feedback
- `revise_node`: Incorporate feedback

**API Endpoints**:
- `POST /api/rfp/{rfp_id}/draft` - Start drafting workflow
- `POST /api/rfp/{rfp_id}/draft/{req_id}/feedback` - Submit feedback

### Phase 4: Persistence Layer - COMPLETE ✓
**Goal:** Enable long-running workflows and semantic search.

**PostgreSQL + pgvector** (`api/vector_store.py`, `init.sql`):
- Company Library tables with vector(1536) embeddings
- IVFFlat indexes for fast similarity search
- OpenAI text-embedding-3-small integration

**Vector Search API**:
- `GET /api/library/vector-search` - Semantic search across all content
- `POST /api/library/vector/capabilities` - Add capability with embedding
- `POST /api/library/vector/past-performance` - Add past performance
- `POST /api/library/vector/key-personnel` - Add key personnel
- `POST /api/library/vector/differentiators` - Add differentiator

**LangGraph Checkpointing** (`init.sql`):
- `checkpoints` table for workflow state persistence
- Enables pause/resume of long-running drafting workflows

## 6. Coordinate System Notes
- **PDF Origin:** Bottom-left, measured in points (72 pts/inch)
- **Web Origin:** Top-left, measured in percentage (0.0-1.0)
- **Conversion:** `web_y = 1.0 - (pdf_y + height) / page_height`

## 7. Iron Triangle Concepts
The "Iron Triangle" links three critical RFP sections:
- **Section L (Instructions):** How to format/submit the proposal
- **Section M (Evaluation):** How the government scores proposals
- **Section C (SOW/PWS):** What work must be performed

A winning proposal ensures:
1. Every L instruction is followed (compliance)
2. Every M factor is addressed with evidence (scoring)
3. Every C requirement is mapped to proposal content (traceability)

## 8. Reference Documents
- `docs/TECHNICAL_SPECIFICATION_v4.md`: Original v4.0 architecture specification (all phases complete)
- `AS_BUILT_v4.1.md`: Comprehensive technical documentation
- `HANDOFF_DOCUMENT.md`: Legacy v2.9 documentation (Shipley methodology)
