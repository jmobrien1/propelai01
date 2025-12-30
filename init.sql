-- PropelAI v5.0 Database Initialization
-- Multi-tenant schema with pgvector support

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ===========================================
-- Core Tables: Tenants & Users
-- ===========================================

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    plan_tier VARCHAR(50) DEFAULT 'free',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'member',
    auth_provider_id VARCHAR(255),
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ===========================================
-- RFP & Document Tables
-- ===========================================

CREATE TABLE IF NOT EXISTS rfps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    solicitation_number VARCHAR(100),
    agency VARCHAR(255),
    due_date DATE,
    status VARCHAR(50) DEFAULT 'created',
    processing_mode VARCHAR(50),
    processing_progress INTEGER DEFAULT 0,
    processing_message TEXT,
    requirements_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rfps_tenant ON rfps(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rfps_status ON rfps(status);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    rfp_id UUID REFERENCES rfps(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    s3_path VARCHAR(1000) NOT NULL,
    doc_type VARCHAR(50),
    file_size BIGINT,
    mime_type VARCHAR(100),
    page_count INTEGER,
    embedding_status VARCHAR(50) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_rfp ON documents(rfp_id);
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);

CREATE TABLE IF NOT EXISTS requirements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    rfp_id UUID REFERENCES rfps(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    section VARCHAR(100),
    requirement_type VARCHAR(50),
    priority VARCHAR(20),
    confidence FLOAT,
    source_page INTEGER,
    bbox_coordinates JSONB,
    source_snippet TEXT,
    keywords TEXT[],
    related_requirements UUID[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_requirements_rfp ON requirements(rfp_id);
CREATE INDEX IF NOT EXISTS idx_requirements_type ON requirements(requirement_type);
CREATE INDEX IF NOT EXISTS idx_requirements_tenant ON requirements(tenant_id);

-- ===========================================
-- Company Library Tables
-- ===========================================

CREATE TABLE IF NOT EXISTS library_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    s3_path VARCHAR(1000),
    entity_type VARCHAR(50) NOT NULL,
    entity_name VARCHAR(500),
    doc_type VARCHAR(50),
    file_size BIGINT,
    embedding_status VARCHAR(50) DEFAULT 'pending',
    extracted_data JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_library_documents_tenant ON library_documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_library_documents_entity_type ON library_documents(entity_type);

CREATE TABLE IF NOT EXISTS library_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID REFERENCES library_documents(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL,
    extracted_data JSONB NOT NULL,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_library_entities_tenant ON library_entities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_library_entities_type ON library_entities(entity_type);

CREATE TABLE IF NOT EXISTS library_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES library_documents(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES library_entities(id) ON DELETE SET NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER,
    page_number INTEGER,
    section VARCHAR(255),
    bbox JSONB,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_library_embeddings_document ON library_embeddings(document_id);

-- Vector similarity index (HNSW for fast approximate search)
CREATE INDEX IF NOT EXISTS idx_library_embeddings_vector
    ON library_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ===========================================
-- LangGraph Checkpointing
-- ===========================================

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    checkpoint_data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_created
    ON checkpoints(thread_id, created_at DESC);

-- ===========================================
-- Agent Trace Log & Feedback
-- ===========================================

CREATE TABLE IF NOT EXISTS agent_trace_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    proposal_id VARCHAR(50),
    trace_run_id VARCHAR(100) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,
    step_type VARCHAR(20) NOT NULL,
    input_state JSONB NOT NULL,
    output_state JSONB NOT NULL,
    reasoning_content TEXT,
    tool_calls JSONB,
    tool_outputs JSONB,
    tokens_input INTEGER,
    tokens_output INTEGER,
    latency_ms INTEGER,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trace_tenant ON agent_trace_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_trace_proposal ON agent_trace_log(proposal_id);
CREATE INDEX IF NOT EXISTS idx_trace_agent ON agent_trace_log(agent_name);
CREATE INDEX IF NOT EXISTS idx_trace_created ON agent_trace_log(created_at);

CREATE TABLE IF NOT EXISTS feedback_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_log_id UUID REFERENCES agent_trace_log(id),
    tenant_id UUID NOT NULL,
    proposal_id VARCHAR(50),
    section_id VARCHAR(100),
    original_text TEXT NOT NULL,
    original_score FLOAT,
    human_edited_text TEXT NOT NULL,
    edit_type VARCHAR(50),
    prompt_context TEXT NOT NULL,
    user_role VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback_pairs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_feedback_proposal ON feedback_pairs(proposal_id);

-- ===========================================
-- Row-Level Security Policies
-- ===========================================

-- Enable RLS on tenant-scoped tables
ALTER TABLE rfps ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE library_embeddings ENABLE ROW LEVEL SECURITY;

-- Create policies (tenant isolation)
CREATE POLICY tenant_isolation_rfps ON rfps
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_requirements ON requirements
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_library_docs ON library_documents
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_library_entities ON library_entities
    USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID);

CREATE POLICY tenant_isolation_library_embeddings ON library_embeddings
    USING (document_id IN (
        SELECT id FROM library_documents
        WHERE tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::UUID
    ));

-- ===========================================
-- Utility Functions
-- ===========================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating updated_at
CREATE TRIGGER update_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rfps_updated_at
    BEFORE UPDATE ON rfps
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- Default Development Tenant
-- ===========================================

INSERT INTO tenants (id, name, plan_tier)
VALUES ('00000000-0000-0000-0000-000000000001', 'Development Tenant', 'enterprise')
ON CONFLICT (id) DO NOTHING;

INSERT INTO users (id, tenant_id, email, name, role)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    'dev@propelai.local',
    'Development User',
    'admin'
)
ON CONFLICT (id) DO NOTHING;

-- ===========================================
-- Permissions
-- ===========================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO propelai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO propelai;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO propelai;
