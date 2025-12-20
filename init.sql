-- PropelAI Database Initialization
-- Creates tables for LangGraph checkpointing, proposal storage, and Company Library

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for semantic search

-- Checkpoints table for LangGraph state persistence
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    parent_checkpoint_id VARCHAR(255),
    checkpoint_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints(created_at);

-- Proposals table for metadata
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id VARCHAR(50) PRIMARY KEY,
    client_name VARCHAR(255) NOT NULL,
    opportunity_name VARCHAR(500) NOT NULL,
    solicitation_number VARCHAR(100),
    due_date DATE,
    current_phase VARCHAR(50) NOT NULL DEFAULT 'intake',
    quality_score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent trace log for audit
CREATE TABLE IF NOT EXISTS agent_trace_log (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    proposal_id VARCHAR(50) REFERENCES proposals(proposal_id),
    agent_name VARCHAR(100) NOT NULL,
    action VARCHAR(255) NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    reasoning_trace TEXT,
    duration_ms INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for audit queries
CREATE INDEX IF NOT EXISTS idx_trace_proposal ON agent_trace_log(proposal_id);
CREATE INDEX IF NOT EXISTS idx_trace_agent ON agent_trace_log(agent_name);
CREATE INDEX IF NOT EXISTS idx_trace_created ON agent_trace_log(created_at);

-- Human feedback table (for the Data Flywheel)
CREATE TABLE IF NOT EXISTS human_feedback (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    proposal_id VARCHAR(50) REFERENCES proposals(proposal_id),
    section_id VARCHAR(100) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL,
    original_content TEXT,
    corrected_content TEXT,
    correction_reason TEXT,
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for feedback analysis
CREATE INDEX IF NOT EXISTS idx_feedback_proposal ON human_feedback(proposal_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON human_feedback(feedback_type);

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for auto-updating updated_at
CREATE TRIGGER update_proposals_updated_at
    BEFORE UPDATE ON proposals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Company Library Tables (v4.0 - pgvector semantic search)
-- =============================================================================

-- Company profile metadata
CREATE TABLE IF NOT EXISTS company_profiles (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_name VARCHAR(500) NOT NULL,
    duns_number VARCHAR(20),
    cage_code VARCHAR(10),
    naics_codes TEXT[],  -- Array of NAICS codes
    set_aside_types TEXT[],  -- Small business, WOSB, HUBZone, etc.
    clearance_level VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Capabilities with vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS capabilities (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
    name VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100),  -- Technical, Management, etc.
    keywords TEXT[],
    embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Past performance records with vector embeddings
CREATE TABLE IF NOT EXISTS past_performances (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
    project_name VARCHAR(500) NOT NULL,
    client_name VARCHAR(500),
    client_agency VARCHAR(500),
    contract_number VARCHAR(100),
    contract_value DECIMAL(15,2),
    period_of_performance VARCHAR(100),
    description TEXT NOT NULL,
    relevance_keywords TEXT[],
    metrics JSONB,  -- Key metrics and outcomes
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Key personnel with vector embeddings
CREATE TABLE IF NOT EXISTS key_personnel (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    role VARCHAR(255),  -- Proposed role in proposals
    years_experience INTEGER,
    clearance_level VARCHAR(50),
    certifications TEXT[],
    bio TEXT NOT NULL,
    expertise_areas TEXT[],
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Differentiators and discriminators with vector embeddings
CREATE TABLE IF NOT EXISTS differentiators (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_id UUID REFERENCES company_profiles(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100),  -- Technical, Cost, Schedule, Risk
    proof_points TEXT[],
    competitor_comparison TEXT,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create vector similarity search indexes (IVFFlat for performance)
CREATE INDEX IF NOT EXISTS idx_capabilities_embedding ON capabilities
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_past_performances_embedding ON past_performances
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_key_personnel_embedding ON key_personnel
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_differentiators_embedding ON differentiators
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Additional indexes for filtering
CREATE INDEX IF NOT EXISTS idx_capabilities_category ON capabilities(category);
CREATE INDEX IF NOT EXISTS idx_past_performances_agency ON past_performances(client_agency);
CREATE INDEX IF NOT EXISTS idx_key_personnel_role ON key_personnel(role);

-- Grant permissions (adjust as needed for your setup)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO propelai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO propelai;
