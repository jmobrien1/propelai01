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

## 8. v4.1 Features: Team Workspaces & Vector Search UI

### Team Workspaces (Role-Based Access Control)
**Goal:** Enable team collaboration with shared Company Library content.

**Database Models** (`api/database.py`, `init.sql`):
- `UserModel`: User accounts with email/password authentication
- `TeamModel`: Team workspaces with slug identifiers
- `TeamMembershipModel`: Links users to teams with roles
- `ActivityLogModel`: Audit trail for team actions

**Roles:**
- `admin`: Full access - manage team, members, and all content
- `contributor`: Can add and edit library content
- `viewer`: Read-only access to library content

**API Endpoints**:
- `POST /api/auth/register` - Register new user (returns JWT token)
- `POST /api/auth/login` - Login user (returns JWT token)
- `POST /api/auth/verify` - Verify JWT token
- `POST /api/auth/refresh` - Refresh JWT token
- `POST /api/auth/forgot-password` - Request password reset token
- `POST /api/auth/reset-password` - Reset password with token
- `POST /api/auth/verify-key` - Verify API key
- `PUT /api/users/me` - Update user profile (name)
- `POST /api/users/me/change-password` - Change password
- `POST /api/teams` - Create team
- `GET /api/teams` - List teams
- `GET /api/teams/{team_id}` - Get team details with members
- `POST /api/teams/{team_id}/members` - Add team member
- `PUT /api/teams/{team_id}/members/{user_id}` - Update member role
- `DELETE /api/teams/{team_id}/members/{user_id}` - Remove member
- `GET /api/teams/{team_id}/activity` - Get team activity log
- `POST /api/teams/{team_id}/api-keys` - Create API key
- `GET /api/teams/{team_id}/api-keys` - List API keys
- `DELETE /api/teams/{team_id}/api-keys/{key_id}` - Revoke API key
- `POST /api/teams/{team_id}/invitations` - Create team invitation
- `GET /api/teams/{team_id}/invitations` - List pending invitations
- `DELETE /api/teams/{team_id}/invitations/{id}` - Cancel invitation
- `GET /api/invitations/{token}` - Get invitation details
- `POST /api/invitations/{token}/accept` - Accept invitation

### Vector Search UI
**Goal:** AI-powered semantic search interface for Company Library.

**Features** (`web/index.html`):
- Natural language search with similarity scores
- Type filters (Capabilities, Past Performance, Key Personnel, Differentiators)
- Configurable result count (Top 5/10/20)
- Real-time search with visual feedback

**Frontend Components**:
- `VectorSearchPanel`: AI search tab in Library view
- `TeamsView`: Team workspace management UI with API key management, activity log, invitations
- `AuthModal`: Login/register/forgot-password/reset-password UI
- `ProfileModal`: User profile editing and password change
- User profile section with JWT-based session management
- Modal dialogs for creating teams, adding members, and sending invitations

### Rate Limiting
**Goal:** Prevent API abuse on authentication endpoints.

**Implementation** (`api/main.py`):
- In-memory sliding window rate limiter
- Per-IP tracking with configurable limits
- Returns 429 Too Many Requests with Retry-After header

**Rate Limits:**
- Login: 5 attempts per minute
- Register: 3 attempts per minute
- Forgot Password: 3 attempts per 5 minutes
- General API: 100 requests per minute

### Two-Factor Authentication (2FA)
**Goal:** Add TOTP-based two-factor authentication for enhanced security.

**Implementation** (`api/main.py`, `api/database.py`):
- TOTP (Time-based One-Time Password) using pyotp library
- Backup codes for account recovery
- QR code generation for authenticator app setup

**API Endpoints:**
- `POST /api/auth/2fa/setup` - Generate TOTP secret and provisioning URI
- `POST /api/auth/2fa/verify-setup` - Verify first code and enable 2FA
- `POST /api/auth/2fa/disable` - Disable 2FA (requires password)
- `POST /api/auth/2fa/verify` - Verify 2FA code during login

**UI Components** (`web/index.html`):
- 2FA setup wizard in ProfileModal Security tab
- QR code display for authenticator app scanning
- Backup codes display with copy functionality
- 2FA challenge during login flow in AuthModal

## 9. Reference Documents
- `docs/TECHNICAL_SPECIFICATION_v4.md`: Original v4.0 architecture specification (all phases complete)
- `AS_BUILT_v4.1.md`: Comprehensive technical documentation
- `HANDOFF_DOCUMENT.md`: Legacy v2.9 documentation (Shipley methodology)
