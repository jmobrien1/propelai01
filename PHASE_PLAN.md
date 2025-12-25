# PropelAI Next Phase Implementation Plan

**Target Version:** v5.0 "Foundation & Trust"
**Current Version:** v2.11
**Document Date:** December 2024

---

## Executive Summary

PropelAI v2.11 successfully validated core capabilities: RFP parsing, requirement extraction, CTM generation, and proposal outline export. However, the current architecture cannot support paid enterprise customers due to:

1. **Data Volatility** - `/tmp` storage loses data on redeployment
2. **No Authentication** - No user identity or multi-tenancy
3. **Trust Gap** - Users cannot verify AI extraction accuracy
4. **Passive Library** - Company documents are stored but not queryable

**v5.0 "Foundation & Trust"** addresses these critical gaps to enable the first paid customers.

---

## Gap Analysis: v2.11 → v5.0

| Gap | Current State (v2.11) | Target State (v5.0) | Business Impact |
|-----|----------------------|---------------------|-----------------|
| **Storage** | `/tmp` ephemeral storage | S3 + PostgreSQL persistent | Data survives deployments |
| **Auth** | None | OAuth2/OIDC + RBAC | Multi-tenant isolation |
| **Traceability** | Page numbers only | Click-to-verify PDF highlighting | Builds user trust |
| **Library** | File upload only | Vector search + RAG | Enables content reuse |
| **Checkpointing** | None | LangGraph + PostgreSQL | Resumable workflows |

---

## Phase 1: Enterprise Infrastructure (Weeks 1-3)

### 1.1 Authentication & Multi-Tenancy

**Objective:** Establish user identity and data isolation

**Implementation:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Authentication Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User ──► Auth0/Supabase ──► JWT Token ──► FastAPI Middleware  │
│                                    │                             │
│                                    ▼                             │
│                          ┌─────────────────┐                    │
│                          │  tenant_id      │                    │
│                          │  user_id        │                    │
│                          │  role           │                    │
│                          └─────────────────┘                    │
│                                    │                             │
│                    ┌───────────────┼───────────────┐            │
│                    ▼               ▼               ▼            │
│               PostgreSQL      S3 Bucket      Vector Store       │
│               (RLS Policy)    (Prefix)       (Namespace)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 1.1.1 | Set up Auth0 or Supabase Auth tenant | 2 days |
| 1.1.2 | Create FastAPI authentication middleware | 2 days |
| 1.1.3 | Add `tenant_id` to all database tables | 1 day |
| 1.1.4 | Implement Row-Level Security (RLS) policies | 2 days |
| 1.1.5 | Create user management endpoints | 2 days |
| 1.1.6 | Update React frontend with login flow | 2 days |

**Database Schema:**

```sql
-- Core tenant/user tables
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan_tier VARCHAR(50) DEFAULT 'free', -- free, pro, enterprise
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) DEFAULT 'member', -- admin, member, viewer
    auth_provider_id VARCHAR(255), -- Auth0/Supabase user ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Row-Level Security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
```

**Deliverable:** Users can log in; data is isolated by tenant.

---

### 1.2 Persistent Storage

**Objective:** Replace `/tmp` with durable storage

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Upload ──► S3/GCS Bucket ──► PostgreSQL (metadata)            │
│                  │                    │                          │
│                  │              ┌─────┴─────┐                   │
│                  │              │ documents │                   │
│                  │              │ - id      │                   │
│                  │              │ - s3_path │                   │
│                  │              │ - tenant  │                   │
│                  ▼              └───────────┘                   │
│         ┌───────────────┐                                       │
│         │ Bucket Layout │                                       │
│         ├───────────────┤                                       │
│         │ /{tenant_id}/ │                                       │
│         │   /rfps/      │                                       │
│         │   /library/   │                                       │
│         │   /exports/   │                                       │
│         └───────────────┘                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 1.2.1 | Create S3/GCS bucket with tenant prefixes | 1 day |
| 1.2.2 | Create `documents` table in PostgreSQL | 1 day |
| 1.2.3 | Refactor `upload_files()` to use S3 | 2 days |
| 1.2.4 | Refactor file retrieval for processing | 2 days |
| 1.2.5 | Migrate export generation to S3 | 1 day |
| 1.2.6 | Add presigned URL generation for downloads | 1 day |

**Database Schema:**

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    rfp_id UUID REFERENCES rfps(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    s3_path VARCHAR(1000) NOT NULL,
    doc_type VARCHAR(50), -- main_solicitation, sow, amendment, attachment
    file_size BIGINT,
    mime_type VARCHAR(100),
    embedding_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, complete
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE rfps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    solicitation_number VARCHAR(100),
    agency VARCHAR(255),
    due_date DATE,
    status VARCHAR(50) DEFAULT 'created',
    processing_mode VARCHAR(50), -- legacy, semantic, best_practices
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Deliverable:** Zero data loss on redeployment; files persist in S3.

---

### 1.3 PostgreSQL + pgvector Setup

**Objective:** Unified relational + vector database

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 1.3.1 | Provision PostgreSQL 15 with pgvector extension | 1 day |
| 1.3.2 | Create complete database schema (see Appendix) | 2 days |
| 1.3.3 | Set up connection pooling (PgBouncer) | 1 day |
| 1.3.4 | Implement SQLAlchemy models | 2 days |
| 1.3.5 | Create database migration system (Alembic) | 1 day |
| 1.3.6 | Replace in-memory RFPStore with PostgreSQL | 3 days |

**Vector Storage Schema:**

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Embeddings for Company Library
CREATE TABLE library_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INT,
    embedding vector(1536), -- OpenAI ada-002 or equivalent
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast similarity search
CREATE INDEX ON library_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**Deliverable:** Unified PostgreSQL database with vector search capability.

---

## Phase 2: The "Trust Gate" (Weeks 4-6)

### 2.1 Geospatial PDF Parsing

**Objective:** Capture bounding box coordinates for every extracted requirement

**Current State:**
- `pypdf` extracts text but loses position information
- Page numbers tracked but not character positions

**Target State:**
- Every requirement has `{page, x, y, width, height}` coordinates
- Enables "click-to-verify" in the UI

**Implementation Options:**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **pdfplumber** | Python native, bbox support | Slower on large docs | Good for MVP |
| **PyMuPDF (fitz)** | Fast, accurate bbox | GPL license | Best performance |
| **Google Document AI** | Enterprise-grade OCR | Cost per page | Future enterprise |

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 2.1.1 | Replace pypdf with PyMuPDF for extraction | 3 days |
| 2.1.2 | Capture bbox coordinates during parsing | 2 days |
| 2.1.3 | Store coordinates in `requirements` table | 1 day |
| 2.1.4 | Create PDF coordinate mapping API endpoint | 2 days |

**Database Schema Update:**

```sql
ALTER TABLE requirements ADD COLUMN bbox_coordinates JSONB;
-- Example: {"page": 42, "x": 72, "y": 340, "width": 468, "height": 24}

ALTER TABLE requirements ADD COLUMN source_snippet TEXT;
-- Store surrounding context for verification
```

**Deliverable:** Every requirement has verifiable source coordinates.

---

### 2.2 Interactive PDF Viewer

**Objective:** Click a requirement → see it highlighted in the source PDF

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trust Gate UI Flow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────┐      ┌─────────────────────────────┐ │
│   │  Requirements Table │      │     PDF Viewer Panel        │ │
│   │                     │      │                             │ │
│   │  [REQ-001] Shall... │─────►│  ┌─────────────────────┐   │ │
│   │  [REQ-002] Must...  │      │  │     Page 42         │   │ │
│   │  [REQ-003] Will...  │      │  │                     │   │ │
│   │                     │      │  │  ┌───────────────┐  │   │ │
│   │  Click to verify ───┼──────┼─►│  │ HIGHLIGHTED   │  │   │ │
│   │                     │      │  │  │ REQUIREMENT   │  │   │ │
│   │                     │      │  │  └───────────────┘  │   │ │
│   │                     │      │  │                     │   │ │
│   └─────────────────────┘      │  └─────────────────────┘   │ │
│                                │                             │ │
│                                └─────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Frontend Implementation:**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **react-pdf** | Simple, client-side | Limited annotation | Good for MVP |
| **PDF.js** | Full Mozilla engine | Complex setup | Most capable |
| **PSPDFKit** | Enterprise features | Expensive license | Future |

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 2.2.1 | Integrate react-pdf or PDF.js into frontend | 3 days |
| 2.2.2 | Create split-panel UI (table + PDF viewer) | 2 days |
| 2.2.3 | Implement highlight rendering on coordinates | 3 days |
| 2.2.4 | Add auto-scroll to requirement location | 1 day |
| 2.2.5 | Handle cross-document references | 2 days |

**API Endpoint:**

```python
@app.get("/api/rfp/{rfp_id}/requirements/{req_id}/source")
async def get_requirement_source(rfp_id: str, req_id: str):
    """Returns PDF viewing coordinates for a requirement."""
    return {
        "document_id": "doc_abc123",
        "pdf_url": "https://s3.../presigned-url",
        "page": 42,
        "bbox": {"x": 72, "y": 340, "width": 468, "height": 24},
        "context": {
            "before": "Previous paragraph text...",
            "after": "Following paragraph text..."
        }
    }
```

**Deliverable:** Users can click any requirement and see it highlighted in the original PDF.

---

## Phase 3: Active Company Library (Weeks 7-9)

### 3.1 Intelligent Document Parsing

**Objective:** Extract structured data from company documents

**Document Types & Extraction:**

| Document Type | Extracted Fields |
|---------------|------------------|
| **Resume** | Name, Title, Clearance, Years Experience, Skills, Certifications |
| **Past Performance** | Contract Name, Agency, Value, Period, Scope, Relevance Keywords |
| **Capability Statement** | Company Name, CAGE, DUNS, NAICS, Core Competencies |
| **Proposal Section** | Section Title, Content, Citations, Win Themes |

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                 Company Library Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Upload ──► Document Classifier ──► Specialized Parser         │
│                     │                       │                    │
│              ┌──────┴──────┐         ┌──────┴──────┐            │
│              │ resume      │         │ Resume      │            │
│              │ past_perf   │         │ Parser      │            │
│              │ capability  │         │             │            │
│              │ proposal    │         │ PastPerf    │            │
│              └─────────────┘         │ Parser      │            │
│                                      └──────┬──────┘            │
│                                             │                    │
│                                             ▼                    │
│                                    ┌────────────────┐           │
│                                    │ Chunk + Embed  │           │
│                                    │ (text-embed-3) │           │
│                                    └────────┬───────┘           │
│                                             │                    │
│                                             ▼                    │
│                                    ┌────────────────┐           │
│                                    │ pgvector Store │           │
│                                    └────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 3.1.1 | Create document type classifier (Gemini Flash) | 2 days |
| 3.1.2 | Build Resume parser with field extraction | 3 days |
| 3.1.3 | Build Past Performance parser | 3 days |
| 3.1.4 | Implement chunking strategy (512 tokens, 50 overlap) | 2 days |
| 3.1.5 | Set up embedding pipeline (OpenAI or Vertex) | 2 days |
| 3.1.6 | Store embeddings in pgvector | 1 day |

**Database Schema:**

```sql
-- Structured extraction results
CREATE TABLE library_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL, -- resume, past_performance, capability
    extracted_data JSONB NOT NULL,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Example extracted_data for resume:
-- {
--   "name": "John Smith",
--   "title": "Senior Program Manager",
--   "clearance": "TS/SCI",
--   "years_experience": 15,
--   "skills": ["Agile", "PMP", "ITIL"],
--   "certifications": ["PMP", "CISSP"]
-- }
```

**Deliverable:** Company documents are parsed and indexed with structured metadata.

---

### 3.2 RAG Search Endpoint

**Objective:** Enable natural language queries against Company Library

**API Design:**

```python
@app.post("/api/library/search")
async def search_library(query: LibrarySearchRequest):
    """
    Semantic search across Company Library with filters.

    Example queries:
    - "Find a Project Manager with TS/SCI clearance"
    - "Past performance for Army IT contracts over $5M"
    - "Our experience with cloud migration"
    """
    return {
        "results": [
            {
                "id": "entity_123",
                "type": "resume",
                "relevance_score": 0.94,
                "snippet": "John Smith, Senior PM with TS/SCI...",
                "source_document": "john_smith_resume.pdf",
                "extracted_data": {...}
            }
        ],
        "total": 15
    }
```

**Search Strategy:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Search Flow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Query: "PM with TS/SCI for Army"                              │
│                    │                                             │
│          ┌────────┴────────┐                                    │
│          ▼                 ▼                                    │
│   ┌─────────────┐   ┌─────────────┐                            │
│   │  Semantic   │   │  Keyword    │                            │
│   │  (pgvector) │   │  (GIN/tsvec)│                            │
│   └──────┬──────┘   └──────┬──────┘                            │
│          │                 │                                    │
│          └────────┬────────┘                                    │
│                   ▼                                             │
│          ┌─────────────────┐                                    │
│          │  RRF Fusion     │  (Reciprocal Rank Fusion)         │
│          │  + Reranking    │                                    │
│          └────────┬────────┘                                    │
│                   ▼                                             │
│          ┌─────────────────┐                                    │
│          │  Filter by:     │                                    │
│          │  - entity_type  │                                    │
│          │  - clearance    │                                    │
│          │  - date range   │                                    │
│          └────────┬────────┘                                    │
│                   ▼                                             │
│            Top-K Results                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 3.2.1 | Implement vector similarity search | 2 days |
| 3.2.2 | Add keyword search with PostgreSQL GIN | 2 days |
| 3.2.3 | Implement hybrid search with RRF fusion | 2 days |
| 3.2.4 | Add metadata filtering (type, clearance, etc.) | 2 days |
| 3.2.5 | Create search API endpoint | 1 day |
| 3.2.6 | Build search UI in frontend | 3 days |

**Deliverable:** Users can search their library with natural language; < 5 second latency.

---

## Phase 4: LangGraph Checkpointing (Week 10)

### 4.1 Persistent Workflow State

**Objective:** Enable resumable, long-running proposal workflows

**Current Problem:**
- Processing is "fire and forget"
- If server restarts, all progress is lost
- No human-in-the-loop capability

**Solution: LangGraph + PostgreSQL Checkpointing**

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Initialize checkpointer
checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)

# Build graph with checkpointing
graph = StateGraph(ProposalState)
graph.add_node("compliance", compliance_agent)
graph.add_node("strategy", strategy_agent)
# ... add other nodes

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Invoke with thread_id for persistence
config = {"configurable": {"thread_id": f"proposal_{proposal_id}"}}
result = app.invoke(initial_state, config)

# Resume later with same thread_id
resumed = app.invoke(None, config)  # Continues from last checkpoint
```

**Tasks:**

| Task | Description | Effort |
|------|-------------|--------|
| 4.1.1 | Set up LangGraph PostgresSaver | 1 day |
| 4.1.2 | Refactor orchestrator to use checkpointing | 3 days |
| 4.1.3 | Implement "pause/resume" API endpoints | 2 days |
| 4.1.4 | Add human-in-the-loop interrupt points | 2 days |
| 4.1.5 | Build workflow status UI | 2 days |

**Deliverable:** Proposals can be paused, resumed, and survive server restarts.

---

## Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    v5.0 Implementation Timeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Week 1-3: Phase 1 - Enterprise Infrastructure                  │
│  ═══════════════════════════════════════════                    │
│  [███████████████████████████████████████████]                  │
│  Auth + Storage + PostgreSQL                                     │
│                                                                  │
│  Week 4-6: Phase 2 - Trust Gate                                 │
│  ═══════════════════════════════                                │
│                    [██████████████████████████]                 │
│                    Geospatial Parsing + PDF Viewer              │
│                                                                  │
│  Week 7-9: Phase 3 - Active Library                             │
│  ══════════════════════════════                                 │
│                              [█████████████████████]            │
│                              Document Parsing + RAG              │
│                                                                  │
│  Week 10: Phase 4 - Checkpointing                               │
│  ════════════════════════════                                   │
│                                           [███████]             │
│                                           LangGraph              │
│                                                                  │
│  Week 11-12: Testing & Launch                                   │
│  ════════════════════════════                                   │
│                                                  [██████████]   │
│                                                  QA + Deploy    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Architecture: v5.0 Target State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PropelAI v5.0 Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   React     │     │              FastAPI Backend                     │   │
│  │  Frontend   │◄───►│                                                  │   │
│  │  + PDF.js   │     │  ┌─────────────────────────────────────────┐   │   │
│  └─────────────┘     │  │           Auth Middleware                │   │   │
│        │             │  │      (Auth0/Supabase + JWT)              │   │   │
│        │             │  └─────────────────────────────────────────┘   │   │
│        │             │                      │                          │   │
│        │             │  ┌─────────────────────────────────────────┐   │   │
│        │             │  │         API Routes (47 endpoints)        │   │   │
│        │             │  └─────────────────────────────────────────┘   │   │
│        │             └──────────────────────┬──────────────────────────┘   │
│        │                                    │                               │
│        │             ┌──────────────────────┴──────────────────────┐       │
│        │             │           LangGraph Orchestrator             │       │
│        │             │         (with PostgreSQL Checkpointing)      │       │
│        │             └──────────────────────┬──────────────────────┘       │
│        │                                    │                               │
│        │    ┌───────────────────────────────┼───────────────────────────┐  │
│        │    │                               │                           │  │
│        │    ▼                               ▼                           ▼  │
│  ┌───────────────┐                ┌─────────────────┐          ┌──────────┐│
│  │  PostgreSQL   │                │    S3 / GCS     │          │  Gemini  ││
│  │  + pgvector   │                │  Object Storage │          │  1.5 Pro ││
│  │               │                │                 │          │          ││
│  │ - tenants     │                │ /{tenant_id}/   │          │ Extract  ││
│  │ - users       │                │   /rfps/        │          │ Reason   ││
│  │ - documents   │                │   /library/     │          │ Draft    ││
│  │ - requirements│                │   /exports/     │          │          ││
│  │ - embeddings  │                │                 │          │          ││
│  │ - checkpoints │                │                 │          │          ││
│  └───────────────┘                └─────────────────┘          └──────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Data Persistence** | 0% data loss | Verify data survives 10 consecutive deployments |
| **Multi-Tenancy** | 100% isolation | Tenant A cannot access Tenant B's data |
| **Traceability** | < 2 sec | Time from click to PDF highlight |
| **Library Search** | < 5 sec | Latency for semantic search query |
| **Checkpointing** | 100% resumable | Pause/resume workflow after server restart |

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| PyMuPDF GPL license conflict | High | Medium | Use pdfplumber or negotiate license |
| pgvector performance at scale | Medium | Low | Monitor and add HNSW indexes |
| Auth0 cost at scale | Medium | Medium | Evaluate Supabase as alternative |
| S3 egress costs | Medium | Medium | Use CloudFront caching |
| Gemini API rate limits | High | Medium | Implement exponential backoff + caching |

---

## Dependencies

| Dependency | Current | Target | Notes |
|------------|---------|--------|-------|
| Python | 3.11+ | 3.11+ | No change |
| PostgreSQL | - | 15+ with pgvector | New |
| FastAPI | 0.100+ | 0.100+ | No change |
| LangGraph | - | 0.2+ | New |
| Object Storage | /tmp | S3/GCS | New |
| Auth Provider | - | Auth0/Supabase | New |
| PDF Library | pypdf | PyMuPDF | Upgrade |

---

## Appendix A: Complete Database Schema

```sql
-- ============================================
-- PropelAI v5.0 Complete Database Schema
-- ============================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================
-- Core Tables
-- ============================================

CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan_tier VARCHAR(50) DEFAULT 'free',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'member',
    auth_provider_id VARCHAR(255),
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- RFP & Document Tables
-- ============================================

CREATE TABLE rfps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    solicitation_number VARCHAR(100),
    agency VARCHAR(255),
    due_date DATE,
    status VARCHAR(50) DEFAULT 'created',
    processing_mode VARCHAR(50),
    processing_progress INT DEFAULT 0,
    processing_message TEXT,
    requirements_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    rfp_id UUID REFERENCES rfps(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    s3_path VARCHAR(1000) NOT NULL,
    doc_type VARCHAR(50),
    file_size BIGINT,
    mime_type VARCHAR(100),
    page_count INT,
    embedding_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    rfp_id UUID REFERENCES rfps(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    section VARCHAR(100),
    requirement_type VARCHAR(50),
    priority VARCHAR(20),
    confidence FLOAT,
    source_page INT,
    bbox_coordinates JSONB,
    source_snippet TEXT,
    keywords TEXT[],
    related_requirements UUID[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- Company Library Tables
-- ============================================

CREATE TABLE library_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    s3_path VARCHAR(1000) NOT NULL,
    doc_type VARCHAR(50),
    file_size BIGINT,
    embedding_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE library_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID REFERENCES library_documents(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL,
    extracted_data JSONB NOT NULL,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE library_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID REFERENCES library_documents(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES library_entities(id) ON DELETE SET NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- LangGraph Checkpointing
-- ============================================

CREATE TABLE checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    checkpoint_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- ============================================
-- Indexes
-- ============================================

CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_rfps_tenant ON rfps(tenant_id);
CREATE INDEX idx_documents_rfp ON documents(rfp_id);
CREATE INDEX idx_requirements_rfp ON requirements(rfp_id);
CREATE INDEX idx_requirements_type ON requirements(requirement_type);
CREATE INDEX idx_library_embeddings_tenant ON library_embeddings(tenant_id);
CREATE INDEX idx_checkpoints_thread ON checkpoints(thread_id, created_at DESC);

-- Vector similarity index
CREATE INDEX idx_library_embeddings_vector ON library_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================
-- Row-Level Security
-- ============================================

ALTER TABLE rfps ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_rfps ON rfps
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
CREATE POLICY tenant_isolation_requirements ON requirements
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
CREATE POLICY tenant_isolation_library_docs ON library_documents
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
CREATE POLICY tenant_isolation_library_entities ON library_entities
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
CREATE POLICY tenant_isolation_library_embeddings ON library_embeddings
    USING (tenant_id = current_setting('app.tenant_id')::UUID);
```

---

## Appendix B: API Changes Summary

| Endpoint | Change | Notes |
|----------|--------|-------|
| `POST /api/auth/login` | New | OAuth2 login |
| `POST /api/auth/logout` | New | Session termination |
| `GET /api/rfp/{id}/requirements/{req_id}/source` | New | PDF coordinates |
| `POST /api/library/search` | Enhanced | Semantic search |
| `GET /api/library/entities` | New | Structured extractions |
| `POST /api/workflow/{id}/pause` | New | LangGraph pause |
| `POST /api/workflow/{id}/resume` | New | LangGraph resume |
| All existing endpoints | Modified | Add tenant isolation |

---

*Document Version: 1.0*
*Last Updated: December 2024*
