# PropelAI: Autonomous Proposal Operating System
**Current Status:** Moving from v3.3 (Linear Script) to v4.0 (Stateful Agentic Architecture).
**Strategic Priority:** Phase 1 - "The Trust Gate" (Source Traceability & Compliance Accuracy).

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
- `api/main.py`: Primary API entry point.
- `agents/enhanced_compliance/parser.py`: PDF ingestion logic with coordinate extraction.
- `agents/enhanced_compliance/document_structure.py`: Data models (ComplianceMatrix, SourceCoordinate, BoundingBox).
- `PRODUCT_ROADMAP_PRD.md`: The plan for the next 3 sprints.

## 5. v4.0 Architecture Phases

### Phase 1: Trust Gate (Source Traceability Engine) - IN PROGRESS
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

### Phase 2: Iron Triangle Logic Engine
**Goal:** Model dependencies between Section L, M, and C.
- `StrategyAgent`: Cross-walk L→M→C, detect conflicts (page limits, structure)
- New models: `WinTheme`, `CompetitorProfile`

### Phase 3: Drafting Agent (PEARL Framework)
**Goal:** Prevent LLM hallucination via structured planning.
- LangGraph StateGraph with 4 nodes:
  1. Action Mining (decompose task)
  2. Plan Generation (pseudocode)
  3. Plan Execution (F-B-P framework)
  4. Red Team Critique (score < 80 → loop)

### Phase 4: Persistence Layer
**Goal:** Enable long-running, interruptible workflows.
- PostgreSQL + pgvector (semantic search for Company Library)
- LangGraph checkpointing for pause/resume

## 6. Coordinate System Notes
- **PDF Origin:** Bottom-left, measured in points (72 pts/inch)
- **Web Origin:** Top-left, measured in percentage (0.0-1.0)
- **Conversion:** `web_y = 1.0 - (pdf_y + height) / page_height`
