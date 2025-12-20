# PropelAI: Autonomous Proposal Operating System
**Current Status:** Moving from v3.3 (Linear Script) to v4.0 (Stateful Agentic Architecture).
**Strategic Priority:** Phase 2 - "The Trust Gate" (Source Traceability & Compliance Accuracy).

## 1. Architecture & Tech Stack
- **Backend:** Python 3.10+ / FastAPI / Uvicorn.
- **Frontend:** React (Single Page App via CDN). *Note: No npm build step. We edit `web/index.html` directly.*
- **Orchestration:** LangGraph (moving away from linear chains).
- **Storage:** PostgreSQL with `pgvector` (replacing SQLite/In-Memory).
- **PDF Engine:** Dual-Pipeline.
  - Tier 1: `pypdf` for digital-native files (Fast/Free).
  - Tier 2: `Tensorlake` for scanned/complex files (Cost/Accurate).

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
- `agents/enhanced_compliance/parser.py`: PDF ingestion logic.
- `agents/enhanced_compliance/document_structure.py`: Data models (ComplianceMatrix, SourceCoordinate).
- `PRODUCT_ROADMAP_PRD.md`: The plan for the next 3 sprints.
