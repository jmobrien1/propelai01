# PropelAI: Autonomous Proposal Operating System
**Current Status:** v5.0.3 - Trust Gate + Strategy Engine + QA Infrastructure
**Phase 1 Complete:** Iron Triangle DAG ✓ | Click-to-Verify ✓ | War Room ✓ | Word API ✓ | QA Tests ✓
**Phase 2 Planned:** Strategy Engine | Annotated Outline | Win Theme Generator

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
- **Run Tests:** `pytest tests/ -v` (114 tests, <5s)
- **Run Tests with Coverage:** `pytest tests/ --cov=agents --cov=api --cov-report=html`
- **Run Extraction Test:** `python agents/enhanced_compliance/section_aware_extractor.py`
- **Docker Build:** `docker-compose up --build`
- **Deploy:** `git push origin main` (Triggers Render.com auto-deploy).

## 3. Development Guidelines (The "Iron Triangle")
1. **Trust First:** Never generate text without a source citation.
2. **Traceability:** All extracted requirements must map to a `SourceCoordinate` (page/x/y).
3. **Immutability:** Do not modify `AS_BUILT_TDD.md` unless architecture changes. It is the source of truth.

## 4. Key Files
- `api/main.py`: Primary API entry point (6700+ lines).
- `api/vector_store.py`: pgvector semantic search for Company Library.
- `api/database.py`: PostgreSQL ORM with SQLAlchemy 2.0.
- `api/email_service.py`: Email abstraction with SMTP/SendGrid support.
- `agents/strategy_agent.py`: Iron Triangle logic engine (StrategyAgent, CompetitorAnalyzer).
- `agents/drafting_workflow.py`: LangGraph drafting workflow (F-B-P framework).
- `agents/enhanced_compliance/pdf_coordinate_extractor.py`: Trust Gate coordinate extraction.
- `agents/enhanced_compliance/document_types.py`: Guided upload document classification.
- `web/index.html`: React SPA with guided upload wizard (v4.1 dark theme).
- `init.sql`: PostgreSQL schema with pgvector extension.
- `tests/conftest.py`: Shared test fixtures (GoldenRFP, MockEmbeddingGenerator).
- `tests/test_agents.py`: Agent unit tests (19 tests).
- `tests/unit/test_graph_logic.py`: Iron Triangle DAG tests (38 tests).
- `tests/integration/test_word_semantic_search.py`: Word API semantic search tests (33 tests).
- `tests/e2e/test_agent_trajectory.py`: Agent trace log tests (24 tests).
- `.github/workflows/test.yml`: CI/CD pipeline (Python 3.10/3.11 matrix).

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
- Hybrid rate limiter: Uses Redis when available, falls back to in-memory
- Sliding window algorithm with per-IP tracking
- Returns 429 Too Many Requests with Retry-After header

**Redis Configuration:**
- Set `REDIS_URL` environment variable to enable distributed rate limiting
- Example: `REDIS_URL=redis://localhost:6379`

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

### Email Service
**Goal:** Provide email sending abstraction for password reset and team invitations.

**Implementation** (`api/email_service.py`):
- Pluggable provider architecture (Console, SMTP, SendGrid)
- HTML email templates with professional styling
- Automatic fallback to console output in development

**Providers:**
- `console`: Prints emails to console (default, for development)
- `smtp`: Standard SMTP with TLS support
- `sendgrid`: SendGrid API integration

**Configuration (Environment Variables):**
- `EMAIL_PROVIDER`: "console" | "smtp" | "sendgrid"
- `EMAIL_FROM`: Sender email address
- `EMAIL_FROM_NAME`: Sender display name
- `APP_BASE_URL`: Base URL for links in emails

**SMTP Settings:**
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_USE_TLS`

**SendGrid Settings:**
- `SENDGRID_API_KEY`: SendGrid API key

**Email Types:**
- Password reset emails with secure tokens
- Team invitation emails with role information
- Welcome emails for new users

### Session Management
**Goal:** Allow users to view and revoke active sessions across devices.

**Implementation** (`api/main.py`, `api/database.py`):
- UserSessionModel tracks active sessions
- Token hash storage for secure session identification
- Device info and IP address tracking

**API Endpoints:**
- `GET /api/sessions` - List all active sessions for current user
- `DELETE /api/sessions/{session_id}` - Revoke a specific session
- `POST /api/sessions/revoke-all` - Revoke all sessions (optionally keep current)

**Session Data:**
- Device info (browser/mobile detection)
- IP address (with proxy support)
- Last active timestamp
- Expiration tracking

**UI** (`web/index.html`):
- Sessions tab in ProfileModal
- View all active sessions with device info
- Revoke individual sessions
- Sign out of all other devices

### Account Lockout
**Goal:** Prevent brute force attacks by locking accounts after failed attempts.

**Implementation** (`api/main.py`, `api/database.py`):
- Track failed login attempts per user
- Lock account after 5 failed attempts
- Auto-unlock after 15 minutes

**UserModel Fields:**
- `failed_login_attempts`: Counter for failed attempts
- `locked_until`: DateTime when lockout expires

### Security Headers
**Goal:** Add security headers to protect against common web vulnerabilities.

**Implementation** (`api/main.py`):
- SecurityHeadersMiddleware adds headers to all responses
- Configurable for development vs production

**Headers Added:**
- `X-Content-Type-Options: nosniff` - Prevent MIME sniffing
- `X-Frame-Options: DENY` - Prevent clickjacking
- `X-XSS-Protection: 1; mode=block` - XSS protection
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy` - Disable unused browser features

**Optional (Environment Variables):**
- `ENABLE_HSTS=true` - Enable Strict-Transport-Security
- `ENABLE_CSP=true` - Enable Content-Security-Policy

### Email Verification
**Goal:** Verify new user email addresses before granting full access.

**Implementation** (`api/main.py`, `api/database.py`):
- Optional email verification on registration
- 24-hour token expiration
- Resend verification email endpoint

**UserModel Fields:**
- `email_verified`: Boolean verification status
- `email_verification_token`: Secure token for verification
- `email_verification_sent_at`: Timestamp for expiration

**API Endpoints:**
- `POST /api/auth/verify-email` - Verify email with token
- `POST /api/auth/resend-verification` - Resend verification email

**Configuration:**
- `REQUIRE_EMAIL_VERIFICATION=true` - Enable email verification requirement

### Password Strength Validation
**Goal:** Enforce strong passwords to improve security.

**Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number

### Health Check Endpoints
**Goal:** Provide endpoints for monitoring and Kubernetes deployments.

**Endpoints:**
- `GET /api/health` - Comprehensive health status with component info
- `GET /api/health/live` - Kubernetes liveness probe (is app running?)
- `GET /api/health/ready` - Kubernetes readiness probe (is app ready for traffic?)
- `GET /api/metrics` - Basic metrics (user count, team count, etc.)

### Request ID Tracing
**Goal:** Enable request tracing for debugging and logging.

**Implementation** (`api/main.py`):
- RequestIDMiddleware adds unique ID to each request
- Preserves X-Request-ID header from load balancers
- Returns X-Request-ID in response headers
- Logs requests with their ID for correlation

### File Upload Security
**Goal:** Prevent malicious file uploads and ensure data integrity.

**Implementation** (`api/main.py`):
- File size validation (max 50 MB)
- File extension whitelist (PDF, DOCX, DOC, XLSX, XLS)
- Magic bytes verification (content matches extension)
- Filename sanitization (prevent path traversal)
- MIME type validation
- Suspicious pattern detection

**Validation Functions:**
- `validate_uploaded_file()`: Async file validation with all security checks
- `sanitize_filename()`: Remove path components and dangerous characters
- `validate_file_extension()`: Quick extension check

**API Endpoint:**
- `GET /api/upload-constraints` - Get allowed file types and size limits

**Constants:**
- `MAX_FILE_SIZE`: 50 MB
- `ALLOWED_EXTENSIONS`: [.pdf, .docx, .doc, .xlsx, .xls]
- `FILE_SIGNATURES`: Magic bytes for each file type

### Standard Pagination
**Goal:** Provide consistent pagination across all list endpoints.

**Implementation** (`api/main.py`):
- `PaginationParams` model with page/page_size
- `PaginatedResponse` model with items, total, has_next, has_prev
- `paginate()` helper function
- `get_pagination_params()` with bounds checking

**Response Format:**
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "total_pages": 5,
  "has_next": true,
  "has_prev": false
}
```

**Paginated Endpoints:**
- `GET /api/rfp` - List RFPs
- `GET /api/rfp/{rfp_id}/requirements` - List requirements
- `GET /api/teams/{team_id}/activity` - List activity log

**Query Parameters:**
- `page`: Page number (1-indexed, default 1)
- `page_size`: Items per page (1-100, default 20)

### API Versioning
**Goal:** Enable API version tracking and deprecation management.

**Implementation** (`api/main.py`):
- APIVersionMiddleware adds version headers to all responses
- Version constants for semantic versioning
- Deprecation warnings for old API versions

**Response Headers:**
- `X-API-Version`: Current API version (e.g., "4.1.0")
- `X-API-Deprecated`: "true" if client version is outdated
- `X-API-Upgrade-Message`: Upgrade instructions if deprecated

**API Endpoint:**
- `GET /api/version` - Get version info and changelog

**Version Constants:**
- `API_VERSION`: "4.1.0"
- `API_VERSION_MAJOR`: 4
- `API_VERSION_MINOR`: 1
- `API_VERSION_PATCH`: 0

### Webhook System
**Goal:** Enable integrations by sending event notifications to external services.

**Database Models** (`api/database.py`):
- `WebhookModel`: Webhook subscriptions with URL, secret, events
- `WebhookDeliveryModel`: Delivery history for debugging

**Event Types:**
- RFP events: `rfp.created`, `rfp.updated`, `rfp.deleted`, `rfp.processed`
- Requirement events: `requirement.extracted`
- Draft events: `draft.started`, `draft.completed`, `draft.feedback_received`
- Team events: `team.member_added`, `team.member_removed`
- Library events: `library.item_added`, `library.item_updated`

**API Endpoints:**
- `GET /api/webhooks/events` - List available event types
- `POST /api/teams/{team_id}/webhooks` - Create webhook
- `GET /api/teams/{team_id}/webhooks` - List webhooks
- `GET /api/teams/{team_id}/webhooks/{webhook_id}` - Get webhook details
- `PUT /api/teams/{team_id}/webhooks/{webhook_id}` - Update webhook
- `DELETE /api/teams/{team_id}/webhooks/{webhook_id}` - Delete webhook
- `GET /api/teams/{team_id}/webhooks/{webhook_id}/deliveries` - View delivery history
- `POST /api/teams/{team_id}/webhooks/{webhook_id}/test` - Send test event

**Security:**
- HMAC-SHA256 signature in `X-Webhook-Signature` header
- Configurable retry count and timeout
- Automatic exponential backoff on failures

### Soft Delete & Data Retention
**Goal:** Enable recovery of deleted data and comply with data retention policies.

**Implementation** (`api/main.py`, `api/database.py`):
- Soft delete moves RFPs to trash instead of permanent deletion
- Configurable retention period (default 30 days)
- Auto-purge after retention period expires

**RFPModel Fields:**
- `is_deleted`: Soft delete flag
- `deleted_at`: Deletion timestamp
- `deleted_by`: User who deleted
- `delete_reason`: Optional deletion reason
- `permanent_delete_at`: Scheduled permanent deletion date

**API Endpoints:**
- `DELETE /api/rfp/{rfp_id}` - Soft delete (use `?permanent=true` for hard delete)
- `POST /api/rfp/{rfp_id}/restore` - Restore from trash
- `GET /api/rfp/trash` - List deleted RFPs
- `DELETE /api/rfp/trash/empty` - Empty trash (permanent delete all)
- `GET /api/retention-policy` - View retention policy settings

### Bulk Operations
**Goal:** Perform operations on multiple RFPs efficiently.

**Request Model:**
```json
{
  "ids": ["rfp-001", "rfp-002", "rfp-003"],
  "reason": "Optional reason for audit"
}
```

**API Endpoints:**
- `POST /api/rfp/bulk/delete` - Bulk delete RFPs
- `POST /api/rfp/bulk/restore` - Bulk restore from trash
- `POST /api/rfp/bulk/export` - Bulk export (JSON or CSV)
- `POST /api/rfp/bulk/update-status` - Bulk status update

**Response Format:**
```json
{
  "success_count": 2,
  "failure_count": 1,
  "results": [
    {"id": "rfp-001", "success": true, "action": "soft_deleted"},
    {"id": "rfp-002", "success": true, "action": "soft_deleted"},
    {"id": "rfp-003", "success": false, "error": "RFP not found"}
  ]
}
```

### Enhanced Audit Logging
**Goal:** Provide comprehensive audit trail for compliance and debugging.

**ActivityLogModel Fields:**
- `ip_address`: Client IP address
- `user_agent`: Client browser/app info
- `request_id`: Correlation ID for request tracing

**Features:**
- Automatic request ID correlation
- Structured JSON logging integration
- Indexed by action type for efficient queries

## 9. v4.2 Infrastructure Improvements

### Security Hardening
**Critical security fixes for production deployment:**

1. **Password Hashing (bcrypt)**
   - Replaced SHA256 with bcrypt via passlib
   - Backward-compatible with legacy hashes during migration
   - Location: `api/main.py:hash_password()`, `verify_password()`

2. **JWT Secret Validation**
   - Fails loudly in production if `JWT_SECRET` not set
   - Development warning when using default secret
   - Environment: `PROPELAI_ENV=production` triggers validation

3. **CORS Configuration**
   - Environment-based origin control via `CORS_ORIGINS`
   - Wildcard disables credentials for security
   - Example: `CORS_ORIGINS=https://app.propelai.com,https://propelai.com`

### Infrastructure Files
**New deployment and development artifacts:**

- **Dockerfile** - Multi-stage production build with security best practices
- **alembic.ini** - Database migration configuration
- **migrations/** - Alembic migrations directory with initial schema
- **.dockerignore** - Optimized Docker build context

### Database Migrations (Alembic)
**Version-controlled schema evolution:**

```bash
# Apply migrations
make db-upgrade
# or: alembic upgrade head

# Create new migration
make db-migrate
# or: alembic revision --autogenerate -m "Description"

# Rollback
make db-downgrade
# or: alembic downgrade -1
```

### Modular Architecture (In Progress)
**New module structure for maintainability:**

```
api/
├── main.py            # FastAPI app entry point
├── config.py          # Centralized configuration
├── database.py        # SQLAlchemy models
├── middleware/        # HTTP middleware
│   ├── security.py    # Security headers
│   ├── versioning.py  # API version headers
│   └── tracing.py     # Request ID tracing
├── routers/           # API route modules (planned)
│   ├── auth.py        # Authentication endpoints
│   ├── rfp.py         # RFP management
│   ├── teams.py       # Team workspaces
│   ├── library.py     # Company Library
│   └── webhooks.py    # Webhook management
└── schemas/           # Pydantic models (planned)
```

### Graceful Shutdown
**Proper resource cleanup on shutdown:**

- Cancels background tasks (rate limiter cleanup)
- Closes database connection pool
- Closes Redis connection
- Logs shutdown progress

### OpenAPI Documentation
**Enhanced API documentation at `/docs`:**

- Organized endpoints by tags (Health, Authentication, RFP, etc.)
- Detailed endpoint descriptions
- Authentication instructions
- Rate limiting documentation

### Performance Optimizations
**N+1 Query Prevention:**

- Eager loading with `selectinload` for RFP relationships
- Configurable relationship loading in `list_all()`
- Optimized database queries

## 10. Environment Variables Reference

### Required in Production
```bash
JWT_SECRET=<32+ character secret>
DATABASE_URL=postgresql://user:pass@host:5432/propelai
PROPELAI_ENV=production
```

### Recommended in Production
```bash
CORS_ORIGINS=https://yourdomain.com
ENABLE_HSTS=true
ENABLE_CSP=true
REDIS_URL=redis://localhost:6379
```

### Optional Configuration
```bash
# Email
EMAIL_PROVIDER=smtp|sendgrid|console
EMAIL_FROM=noreply@propelai.com
SMTP_HOST=smtp.example.com
SENDGRID_API_KEY=SG.xxx

# Features
REQUIRE_EMAIL_VERIFICATION=true

# LLM APIs
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
```

## 11. Reference Documents
- `docs/TECHNICAL_SPECIFICATION_v4.md`: Original v4.0 architecture specification (all phases complete)
- `AS_BUILT_v4.1.md`: Comprehensive technical documentation
- `AS_BUILT_v5.0.md`: v5.0.3 Trust Gate, Strategy Engine, and QA Infrastructure documentation
- `HANDOFF_DOCUMENT.md`: Legacy v2.9 documentation (Shipley methodology)
- `prd5.rtf`: PRD v5.0 Phase 1 - Trust Gate specification
- `prd12.rtf`: PRD v5.0 Phase 2 - Strategy Engine specification
- `requirements-test.txt`: Test dependencies for CI/CD pipeline

## 12. v5.0 Phase 1: Trust Gate (IMPLEMENTED)
**Status:** Complete - Deterministic compliance verification

### Features Implemented:
- **Iron Triangle DAG**: NetworkX-based dependency graph linking C ↔ L ↔ M sections
- **Multi-Page Spanning**: `visual_rects` array for requirements crossing page boundaries
- **Click-to-Verify UI**: Split-screen PDF viewer with yellow highlight overlays
- **War Room Dashboard**: CCS score, Iron Triangle graph visualization, orphan panel
- **Word Integration API**: Context awareness endpoints for future Word Add-in

### API Endpoints:
- `GET /api/rfp/{rfp_id}/graph` - Iron Triangle DAG with analysis
- `GET /api/rfp/{rfp_id}/graph/orphans` - Unlinked requirements
- `POST /api/rfp/{rfp_id}/graph/link` - Create C↔L↔M links
- `DELETE /api/rfp/{rfp_id}/graph/link/{link_id}` - Remove links
- `POST /api/word/context` - Word Add-in context awareness (with semantic search)
- `GET /api/word/rfps` - List RFPs for Word Add-in

### Word API Semantic Search (v5.0.1)
**Goal:** Upgrade from keyword-based Jaccard similarity to meaning-based semantic search.

**Implementation:**
- Uses `EmbeddingGenerator` from `api/vector_store.py` (Voyage AI / OpenAI / fallback)
- Requirement embeddings cached in `rfp["_requirement_embeddings"]` for performance
- Cosine similarity matching with 0.3 threshold
- Automatic fallback to Jaccard similarity if semantic unavailable

**Request Parameters:**
- `use_semantic_search: bool = True` - Enable/disable semantic search

**Response Fields:**
- `search_method: str` - "semantic" or "jaccard" indicating which method was used

**Benefits over Jaccard:**
- Understands synonyms (e.g., "personnel" matches "staffing requirements")
- Handles paraphrasing and context
- Better accuracy for proposal compliance checking

### Frontend Components:
- `PDFViewerModal`: Supports `mode="split"` for split-screen, multi-page highlights
- `WarRoomView`: CCS header, SVG Iron Triangle graph, orphan sidebar
- `MatrixView`: Split-screen mode toggle with `splitScreenMode` state

### Force-Directed Graph Layout (v5.0.2)
**Goal:** Replace random node positioning with physics-based simulation for better visualization.

**Algorithm:**
- 100 iterations with cooling factor (simulated annealing)
- **Repulsion force:** Nodes push apart (inverse square law)
- **Attraction force:** Connected nodes pull together along edges
- **Section clustering:** Nodes gravitate toward their section target (C/L/M triangle)
- **Bounds constraint:** Keeps nodes within canvas

**Parameters:**
```javascript
repulsionStrength: 800   // Node-node repulsion
attractionStrength: 0.05 // Edge attraction
sectionPull: 0.15        // Pull toward section target
damping: 0.9             // Velocity damping
minDistance: 25          // Minimum node distance
```

### Agent Trace Log (NFR-2.3 Data Flywheel)
**Goal:** Log every agent action (Input → Output → Human Correction) for debugging and training.

**Database Model** (`api/database.py`):
```python
class AgentTraceLogModel(Base):
    id: str                    # trace-{uuid}
    rfp_id: str               # Associated RFP
    agent_name: str           # e.g., "ComplianceAgent"
    action: str               # e.g., "extract_requirements"
    input_data: JSONB         # What was given to agent
    output_data: JSONB        # What agent produced
    confidence_score: float   # 0.0-1.0
    human_correction: JSONB   # Corrected output if any
    correction_type: str      # "accepted", "modified", "rejected"
    status: str               # pending, completed, failed, corrected
```

**API Endpoints:**
- `POST /api/trace-logs` - Create trace log entry
- `GET /api/trace-logs` - List logs with filters (rfp_id, agent_name, action, status)
- `GET /api/trace-logs/{trace_id}` - Get specific log
- `POST /api/trace-logs/{trace_id}/correct` - Submit human correction
- `GET /api/trace-logs/stats/summary` - Get correction rate statistics

**Use Cases:**
- Time-travel debugging (replay agent decisions)
- Human-in-the-loop correction feedback
- Training data collection for model improvement (Data Flywheel)

## 13. v5.0 Phase 2: The Strategy Engine (ROADMAP)
**Status:** Planned - Transform Compliance Matrix into strategic Annotated Outline
**Target Users:** "Charles" (Executive Strategist) & "Brenda" (Proposal Manager)

### Epic 1: Iron Triangle Logic Graph Enhancement

**FR-1.1: Section M Decomposition**
Parse Section M (Evaluation Factors) to extract scoring weights and identify discriminators.

**FR-1.2: Dependency Graphing**
Extend NetworkX graph so Section M factors become parent nodes to Section L instructions.
- *Constraint:* If Section L is silent, default to Section M structure for evaluator scoring.

### Epic 2: Automated Annotated Outline Generation

**FR-2.1: Outline Skeleton**
Auto-generate hierarchical outline (Vol 1, Section 1.1, 1.2, etc.) based on Section L instructions.

**FR-2.2: Requirement Injection**
Inject "Writing Guidance" tables with "Shall" statements from Compliance Matrix per section.

**FR-2.3: Annotation Injection**
Each section includes:
- **Page Budget:** Calculated from Section M weight
- **Win Theme:** Strategic message placeholder
- **Proof Points:** Required evidence (e.g., "Insert Past Performance Citation")

**FR-2.4: Export to DOCX**
Generate downloadable `.docx` matching RFP formatting (fonts, margins).

### Epic 3: Strategy Agent (Win Themes & Ghosting)

**FR-3.1: Hot Button Analysis**
Infer agency pain points from PWS/SOW (e.g., "24/7 uptime → previous outages").

**FR-3.2: Win Theme Generator**
Using F-B-P framework, generate 3 draft Win Themes per volume:
- *Format:* "Feature [X] delivers Benefit [Y], proven by [Evidence Z]"

**FR-3.3: Black Hat / Ghosting**
If competitor identified, suggest ghosting statements:
- *Example:* Incumbent with legacy tech → themes on "Modernization" and "No Technical Debt"

### Technical Architecture

**LangGraph Plan-and-Execute Pattern:**
```
Node 1: Strategy Agent    → Reads Section M + PWS → StrategicPlan JSON
Node 2: Outline Agent     → Reads Section L + StrategicPlan → OutlineStructure JSON
Node 3: Requirements Injector → Maps extraction IDs into OutlineStructure
```

**State Persistence:**
PostgreSQL `ProposalState` updated with `strategic_plan` and `outline_tree` columns.

**Data Models:**
```python
class WinTheme(TypedDict):
    feature: str
    benefit: str
    proof_point_id: str  # Link to Company Library
    ghosting_strategy: Optional[str]

class OutlineNode(TypedDict):
    section_number: str
    title: str
    page_allocation: int
    assigned_requirements: List[str]  # Requirement IDs
    assigned_win_theme: WinTheme
```

**RAG Strategy: GraphRAG**
Use GraphRAG (not simple vector search) to traverse knowledge graph for proof points.
- *Example:* Theme about "Rapid Staffing" → find "Time-to-Fill" metrics in Company Library.

### UX: The War Room (Phase 2 Enhancements)

**Strategy Dashboard:**
- Input: Strategy Configuration modal (Incumbent, Our Strengths)
- Output: Split-view with Section M factors (left) and Win Themes (right)
- Actions: Accept / Reject / Edit AI-suggested themes

**Outline Builder:**
- Visual drag-and-drop tree for proposal volumes
- Drag "Orphaned Requirements" from sidebar into sections
- Click-to-Verify integration (Phase 1 PDF overlay)

### Planned API Endpoints:
- `POST /api/rfp/{rfp_id}/strategy/configure` - Set incumbent/strengths
- `POST /api/rfp/{rfp_id}/strategy/generate` - Generate win themes
- `GET /api/rfp/{rfp_id}/outline` - Get outline structure
- `PUT /api/rfp/{rfp_id}/outline` - Update outline (drag-drop)
- `POST /api/rfp/{rfp_id}/outline/export` - Export to DOCX
- `POST /api/rfp/{rfp_id}/outline/{section}/requirements` - Assign requirements

### Success Metrics:
- **Outline Generation Time:** 4 days (manual) → <1 hour (AI)
- **Strategy Alignment Score:** % of Section M factors mapped to outline sections
- **Requirement Coverage:** 100% of "Shall" statements assigned (no orphans)
