-- PropelAI Database Initialization
-- Creates tables for LangGraph checkpointing and proposal storage

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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

-- Grant permissions (adjust as needed for your setup)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO propelai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO propelai;
