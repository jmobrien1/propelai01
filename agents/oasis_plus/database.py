"""
OASIS+ Database Integration Layer
=================================

Persistent storage for OASIS+ proposals using PostgreSQL with pgvector.

Supports:
- Proposal and project storage
- Document chunk storage with vector embeddings
- Claim tracking and verification status
- Scorecard persistence

Designed for PostgreSQL with pgvector extension for semantic search.
Falls back to SQLite for development/testing.
"""

import logging
import os
import json
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Database availability flags
POSTGRES_AVAILABLE = False
SQLITE_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    pass

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    pass

from .models import (
    OASISDomain,
    ScoringCriteria,
    Project,
    ProjectClaim,
    DocumentChunk,
    DomainType,
    BusinessSize,
    VerificationStatus,
    ContractType,
    CriteriaType,
)


# ============== Schema Definitions ==============

POSTGRES_SCHEMA = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Proposals table
CREATE TABLE IF NOT EXISTS oasis_proposals (
    proposal_id VARCHAR(50) PRIMARY KEY,
    contractor_name VARCHAR(255) NOT NULL,
    contractor_cage VARCHAR(20),
    business_size VARCHAR(50) NOT NULL DEFAULT 'unrestricted',
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    target_domains JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Domains table (loaded from J.P-1)
CREATE TABLE IF NOT EXISTS oasis_domains (
    id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(50) REFERENCES oasis_proposals(proposal_id) ON DELETE CASCADE,
    domain_type VARCHAR(50) NOT NULL,
    name VARCHAR(255),
    description TEXT,
    unrestricted_threshold INTEGER DEFAULT 42,
    small_business_threshold INTEGER DEFAULT 36,
    criteria_json JSONB DEFAULT '[]',
    auto_relevant_naics JSONB DEFAULT '[]',
    auto_relevant_psc JSONB DEFAULT '[]',
    UNIQUE(proposal_id, domain_type)
);

-- Projects table
CREATE TABLE IF NOT EXISTS oasis_projects (
    project_id VARCHAR(50) PRIMARY KEY,
    proposal_id VARCHAR(50) REFERENCES oasis_proposals(proposal_id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    client_agency VARCHAR(255),
    contract_number VARCHAR(100),
    task_order_number VARCHAR(100),
    naics_code VARCHAR(20),
    psc_code VARCHAR(20),
    start_date DATE,
    end_date DATE,
    total_obligated_amount DECIMAL(15,2) DEFAULT 0,
    contract_type VARCHAR(50) DEFAULT 'ffp',
    is_prime BOOLEAN DEFAULT TRUE,
    prime_contractor VARCHAR(255),
    performance_location TEXT,
    is_oconus BOOLEAN DEFAULT FALSE,
    clearance_level VARCHAR(50),
    scope_description TEXT,
    average_annual_value DECIMAL(15,2),
    relevance_scores JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks with vector embeddings
CREATE TABLE IF NOT EXISTS oasis_document_chunks (
    chunk_id VARCHAR(100) PRIMARY KEY,
    document_id VARCHAR(255),
    project_id VARCHAR(50) REFERENCES oasis_projects(project_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),  -- For OpenAI text-embedding-3-small
    page_number INTEGER DEFAULT 1,
    bbox JSONB,  -- {x, y, w, h}
    chunk_index INTEGER DEFAULT 0,
    char_start INTEGER DEFAULT 0,
    char_end INTEGER DEFAULT 0,
    ocr_confidence FLOAT,
    was_ocr BOOLEAN DEFAULT FALSE
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON oasis_document_chunks
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Claims table
CREATE TABLE IF NOT EXISTS oasis_claims (
    claim_id VARCHAR(100) PRIMARY KEY,
    project_id VARCHAR(50) REFERENCES oasis_projects(project_id) ON DELETE CASCADE,
    criteria_id VARCHAR(50) NOT NULL,
    domain_type VARCHAR(50) NOT NULL,
    claimed_points INTEGER DEFAULT 0,
    verified_points INTEGER DEFAULT 0,
    evidence_snippet TEXT,
    evidence_page_number INTEGER,
    evidence_document_id VARCHAR(255),
    evidence_bbox JSONB,
    status VARCHAR(50) DEFAULT 'unverified',
    verification_notes TEXT,
    verified_by VARCHAR(100),
    verified_at TIMESTAMP,
    ai_confidence_score FLOAT DEFAULT 0,
    jp3_form_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimization results
CREATE TABLE IF NOT EXISTS oasis_optimization_results (
    id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(50) REFERENCES oasis_proposals(proposal_id) ON DELETE CASCADE,
    domain_type VARCHAR(50) NOT NULL,
    business_size VARCHAR(50) NOT NULL,
    selected_project_ids JSONB DEFAULT '[]',
    total_score INTEGER DEFAULT 0,
    verified_score INTEGER DEFAULT 0,
    threshold INTEGER DEFAULT 42,
    margin INTEGER DEFAULT 0,
    overall_risk VARCHAR(20) DEFAULT 'LOW',
    risk_factors JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(proposal_id, domain_type)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_projects_proposal ON oasis_projects(proposal_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON oasis_document_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_claims_project ON oasis_claims(project_id);
CREATE INDEX IF NOT EXISTS idx_claims_criteria ON oasis_claims(criteria_id);
"""

SQLITE_SCHEMA = """
-- Proposals table
CREATE TABLE IF NOT EXISTS oasis_proposals (
    proposal_id TEXT PRIMARY KEY,
    contractor_name TEXT NOT NULL,
    contractor_cage TEXT,
    business_size TEXT NOT NULL DEFAULT 'unrestricted',
    status TEXT NOT NULL DEFAULT 'created',
    target_domains TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Domains table
CREATE TABLE IF NOT EXISTS oasis_domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT REFERENCES oasis_proposals(proposal_id) ON DELETE CASCADE,
    domain_type TEXT NOT NULL,
    name TEXT,
    description TEXT,
    unrestricted_threshold INTEGER DEFAULT 42,
    small_business_threshold INTEGER DEFAULT 36,
    criteria_json TEXT DEFAULT '[]',
    auto_relevant_naics TEXT DEFAULT '[]',
    auto_relevant_psc TEXT DEFAULT '[]',
    UNIQUE(proposal_id, domain_type)
);

-- Projects table
CREATE TABLE IF NOT EXISTS oasis_projects (
    project_id TEXT PRIMARY KEY,
    proposal_id TEXT REFERENCES oasis_proposals(proposal_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    client_agency TEXT,
    contract_number TEXT,
    task_order_number TEXT,
    naics_code TEXT,
    psc_code TEXT,
    start_date TEXT,
    end_date TEXT,
    total_obligated_amount REAL DEFAULT 0,
    contract_type TEXT DEFAULT 'ffp',
    is_prime INTEGER DEFAULT 1,
    prime_contractor TEXT,
    performance_location TEXT,
    is_oconus INTEGER DEFAULT 0,
    clearance_level TEXT,
    scope_description TEXT,
    average_annual_value REAL,
    relevance_scores TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks (without vector support in SQLite)
CREATE TABLE IF NOT EXISTS oasis_document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,
    project_id TEXT REFERENCES oasis_projects(project_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding TEXT,  -- JSON array for SQLite
    page_number INTEGER DEFAULT 1,
    bbox TEXT,
    chunk_index INTEGER DEFAULT 0,
    char_start INTEGER DEFAULT 0,
    char_end INTEGER DEFAULT 0,
    ocr_confidence REAL,
    was_ocr INTEGER DEFAULT 0
);

-- Claims table
CREATE TABLE IF NOT EXISTS oasis_claims (
    claim_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES oasis_projects(project_id) ON DELETE CASCADE,
    criteria_id TEXT NOT NULL,
    domain_type TEXT NOT NULL,
    claimed_points INTEGER DEFAULT 0,
    verified_points INTEGER DEFAULT 0,
    evidence_snippet TEXT,
    evidence_page_number INTEGER,
    evidence_document_id TEXT,
    evidence_bbox TEXT,
    status TEXT DEFAULT 'unverified',
    verification_notes TEXT,
    verified_by TEXT,
    verified_at TEXT,
    ai_confidence_score REAL DEFAULT 0,
    jp3_form_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Optimization results
CREATE TABLE IF NOT EXISTS oasis_optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT REFERENCES oasis_proposals(proposal_id) ON DELETE CASCADE,
    domain_type TEXT NOT NULL,
    business_size TEXT NOT NULL,
    selected_project_ids TEXT DEFAULT '[]',
    total_score INTEGER DEFAULT 0,
    verified_score INTEGER DEFAULT 0,
    threshold INTEGER DEFAULT 42,
    margin INTEGER DEFAULT 0,
    overall_risk TEXT DEFAULT 'LOW',
    risk_factors TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(proposal_id, domain_type)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_projects_proposal ON oasis_projects(proposal_id);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON oasis_document_chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_claims_project ON oasis_claims(project_id);
CREATE INDEX IF NOT EXISTS idx_claims_criteria ON oasis_claims(criteria_id);
"""


class OASISDatabase:
    """
    Database interface for OASIS+ data persistence.

    Supports PostgreSQL with pgvector for production,
    SQLite for development/testing.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        sqlite_path: Optional[str] = None,
    ):
        """
        Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string
            sqlite_path: Path to SQLite database file
        """
        self.connection_string = connection_string or os.environ.get("DATABASE_URL")
        self.sqlite_path = sqlite_path
        self.conn = None
        self.is_postgres = False

        self._connect()
        self._init_schema()

    def _connect(self):
        """Establish database connection"""
        if self.connection_string and POSTGRES_AVAILABLE:
            try:
                self.conn = psycopg2.connect(self.connection_string)
                self.is_postgres = True
                logger.info("Connected to PostgreSQL database")
            except Exception as e:
                logger.warning(f"PostgreSQL connection failed: {e}")
                self._fallback_sqlite()
        else:
            self._fallback_sqlite()

    def _fallback_sqlite(self):
        """Fall back to SQLite"""
        if not SQLITE_AVAILABLE:
            raise RuntimeError("No database backend available")

        db_path = self.sqlite_path or "oasis_data.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.is_postgres = False
        logger.info(f"Using SQLite database: {db_path}")

    def _init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            # Split and execute PostgreSQL schema
            for statement in POSTGRES_SCHEMA.split(';'):
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        logger.warning(f"Schema statement failed: {e}")
        else:
            # Execute SQLite schema
            cursor.executescript(SQLITE_SCHEMA)

        self.conn.commit()
        cursor.close()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    # ==================== Proposal Operations ====================

    def save_proposal(
        self,
        proposal_id: str,
        contractor_name: str,
        contractor_cage: str = "",
        business_size: str = "unrestricted",
        status: str = "created",
        target_domains: List[str] = None,
    ) -> bool:
        """Save or update a proposal"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute("""
                INSERT INTO oasis_proposals
                (proposal_id, contractor_name, contractor_cage, business_size, status, target_domains, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (proposal_id) DO UPDATE SET
                    contractor_name = EXCLUDED.contractor_name,
                    contractor_cage = EXCLUDED.contractor_cage,
                    business_size = EXCLUDED.business_size,
                    status = EXCLUDED.status,
                    target_domains = EXCLUDED.target_domains,
                    updated_at = EXCLUDED.updated_at
            """, (
                proposal_id, contractor_name, contractor_cage, business_size,
                status, Json(target_domains or []), datetime.now()
            ))
        else:
            cursor.execute("""
                INSERT OR REPLACE INTO oasis_proposals
                (proposal_id, contractor_name, contractor_cage, business_size, status, target_domains, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                proposal_id, contractor_name, contractor_cage, business_size,
                status, json.dumps(target_domains or []), datetime.now().isoformat()
            ))

        self.conn.commit()
        cursor.close()
        return True

    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get a proposal by ID"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute(
                "SELECT * FROM oasis_proposals WHERE proposal_id = %s",
                (proposal_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM oasis_proposals WHERE proposal_id = ?",
                (proposal_id,)
            )

        row = cursor.fetchone()
        cursor.close()

        if row:
            return dict(row)
        return None

    def list_proposals(self) -> List[Dict[str, Any]]:
        """List all proposals"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM oasis_proposals ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]

    def delete_proposal(self, proposal_id: str) -> bool:
        """Delete a proposal and all related data"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute(
                "DELETE FROM oasis_proposals WHERE proposal_id = %s",
                (proposal_id,)
            )
        else:
            cursor.execute(
                "DELETE FROM oasis_proposals WHERE proposal_id = ?",
                (proposal_id,)
            )

        self.conn.commit()
        cursor.close()
        return True

    # ==================== Project Operations ====================

    def save_project(self, project: Project, proposal_id: str) -> bool:
        """Save or update a project"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute("""
                INSERT INTO oasis_projects
                (project_id, proposal_id, title, client_agency, contract_number,
                task_order_number, naics_code, psc_code, start_date, end_date,
                total_obligated_amount, contract_type, is_prime, prime_contractor,
                performance_location, is_oconus, clearance_level, scope_description,
                average_annual_value, relevance_scores)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (project_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    total_obligated_amount = EXCLUDED.total_obligated_amount,
                    average_annual_value = EXCLUDED.average_annual_value,
                    relevance_scores = EXCLUDED.relevance_scores
            """, (
                project.project_id, proposal_id, project.title, project.client_agency,
                project.contract_number, project.task_order_number, project.naics_code,
                project.psc_code, project.start_date, project.end_date,
                float(project.total_obligated_amount), project.contract_type.value,
                project.is_prime, project.prime_contractor, project.performance_location,
                project.is_oconus, project.clearance_level, project.scope_description,
                float(project.calculate_aav()),
                Json({d.value: s for d, s in project.relevance_scores.items()})
            ))
        else:
            cursor.execute("""
                INSERT OR REPLACE INTO oasis_projects
                (project_id, proposal_id, title, client_agency, contract_number,
                task_order_number, naics_code, psc_code, start_date, end_date,
                total_obligated_amount, contract_type, is_prime, prime_contractor,
                performance_location, is_oconus, clearance_level, scope_description,
                average_annual_value, relevance_scores)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project.project_id, proposal_id, project.title, project.client_agency,
                project.contract_number, project.task_order_number, project.naics_code,
                project.psc_code,
                project.start_date.isoformat() if project.start_date else None,
                project.end_date.isoformat() if project.end_date else None,
                float(project.total_obligated_amount), project.contract_type.value,
                1 if project.is_prime else 0, project.prime_contractor,
                project.performance_location, 1 if project.is_oconus else 0,
                project.clearance_level, project.scope_description,
                float(project.calculate_aav()),
                json.dumps({d.value: s for d, s in project.relevance_scores.items()})
            ))

        self.conn.commit()
        cursor.close()
        return True

    def get_projects(self, proposal_id: str) -> List[Dict[str, Any]]:
        """Get all projects for a proposal"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute(
                "SELECT * FROM oasis_projects WHERE proposal_id = %s",
                (proposal_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM oasis_projects WHERE proposal_id = ?",
                (proposal_id,)
            )

        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]

    # ==================== Chunk Operations ====================

    def save_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Save document chunks (batch insert)"""
        if not chunks:
            return 0

        cursor = self.conn.cursor()
        count = 0

        for chunk in chunks:
            try:
                if self.is_postgres:
                    embedding_str = None
                    if chunk.embedding:
                        embedding_str = f"[{','.join(str(x) for x in chunk.embedding)}]"

                    cursor.execute("""
                        INSERT INTO oasis_document_chunks
                        (chunk_id, document_id, project_id, content, embedding,
                        page_number, bbox, chunk_index, char_start, char_end,
                        ocr_confidence, was_ocr)
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """, (
                        chunk.chunk_id, chunk.document_id, chunk.project_id,
                        chunk.content, embedding_str, chunk.page_number,
                        Json(chunk.bbox) if chunk.bbox else None,
                        chunk.chunk_index, chunk.char_start, chunk.char_end,
                        chunk.ocr_confidence, chunk.was_ocr
                    ))
                else:
                    cursor.execute("""
                        INSERT OR IGNORE INTO oasis_document_chunks
                        (chunk_id, document_id, project_id, content, embedding,
                        page_number, bbox, chunk_index, char_start, char_end,
                        ocr_confidence, was_ocr)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.chunk_id, chunk.document_id, chunk.project_id,
                        chunk.content,
                        json.dumps(chunk.embedding) if chunk.embedding else None,
                        chunk.page_number,
                        json.dumps(chunk.bbox) if chunk.bbox else None,
                        chunk.chunk_index, chunk.char_start, chunk.char_end,
                        chunk.ocr_confidence, 1 if chunk.was_ocr else 0
                    ))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save chunk {chunk.chunk_id}: {e}")

        self.conn.commit()
        cursor.close()
        return count

    def get_chunks(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a project"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute(
                "SELECT * FROM oasis_document_chunks WHERE project_id = %s ORDER BY chunk_index",
                (project_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM oasis_document_chunks WHERE project_id = ? ORDER BY chunk_index",
                (project_id,)
            )

        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]

    def search_chunks_by_embedding(
        self,
        query_embedding: List[float],
        project_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search chunks by vector similarity (PostgreSQL with pgvector only).

        Returns chunks sorted by cosine similarity.
        """
        if not self.is_postgres:
            logger.warning("Vector search requires PostgreSQL with pgvector")
            return []

        cursor = self.conn.cursor()

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        if project_id:
            cursor.execute("""
                SELECT *, 1 - (embedding <=> %s::vector) as similarity
                FROM oasis_document_chunks
                WHERE project_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, project_id, embedding_str, limit))
        else:
            cursor.execute("""
                SELECT *, 1 - (embedding <=> %s::vector) as similarity
                FROM oasis_document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, limit))

        rows = cursor.fetchall()
        cursor.close()

        # Filter by threshold
        results = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get('similarity', 0) >= threshold:
                results.append(row_dict)

        return results

    # ==================== Claim Operations ====================

    def save_claim(self, claim: ProjectClaim, domain_type: str) -> bool:
        """Save or update a claim"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute("""
                INSERT INTO oasis_claims
                (claim_id, project_id, criteria_id, domain_type, claimed_points,
                verified_points, evidence_snippet, evidence_page_number,
                evidence_document_id, evidence_bbox, status, verification_notes,
                verified_by, verified_at, ai_confidence_score, jp3_form_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (claim_id) DO UPDATE SET
                    verified_points = EXCLUDED.verified_points,
                    status = EXCLUDED.status,
                    verification_notes = EXCLUDED.verification_notes,
                    verified_by = EXCLUDED.verified_by,
                    verified_at = EXCLUDED.verified_at,
                    updated_at = EXCLUDED.updated_at
            """, (
                claim.claim_id, claim.project_id, claim.criteria_id, domain_type,
                claim.claimed_points, claim.verified_points, claim.evidence_snippet,
                claim.evidence_page_number, claim.evidence_document_id,
                Json(claim.evidence_bbox) if claim.evidence_bbox else None,
                claim.status.value, claim.verification_notes, claim.verified_by,
                claim.verified_at, claim.ai_confidence_score, claim.jp3_form_id,
                datetime.now()
            ))
        else:
            cursor.execute("""
                INSERT OR REPLACE INTO oasis_claims
                (claim_id, project_id, criteria_id, domain_type, claimed_points,
                verified_points, evidence_snippet, evidence_page_number,
                evidence_document_id, evidence_bbox, status, verification_notes,
                verified_by, verified_at, ai_confidence_score, jp3_form_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                claim.claim_id, claim.project_id, claim.criteria_id, domain_type,
                claim.claimed_points, claim.verified_points, claim.evidence_snippet,
                claim.evidence_page_number, claim.evidence_document_id,
                json.dumps(claim.evidence_bbox) if claim.evidence_bbox else None,
                claim.status.value, claim.verification_notes, claim.verified_by,
                claim.verified_at.isoformat() if claim.verified_at else None,
                claim.ai_confidence_score, claim.jp3_form_id,
                datetime.now().isoformat()
            ))

        self.conn.commit()
        cursor.close()
        return True

    def get_claims(
        self,
        project_id: Optional[str] = None,
        domain_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get claims, optionally filtered by project or domain"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM oasis_claims WHERE 1=1"
        params = []

        if project_id:
            query += " AND project_id = %s" if self.is_postgres else " AND project_id = ?"
            params.append(project_id)

        if domain_type:
            query += " AND domain_type = %s" if self.is_postgres else " AND domain_type = ?"
            params.append(domain_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        return [dict(row) for row in rows]

    # ==================== Optimization Results ====================

    def save_optimization_result(
        self,
        proposal_id: str,
        domain_type: str,
        business_size: str,
        selected_project_ids: List[str],
        total_score: int,
        verified_score: int,
        threshold: int,
        margin: int,
        overall_risk: str,
        risk_factors: List[str],
    ) -> bool:
        """Save optimization result"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute("""
                INSERT INTO oasis_optimization_results
                (proposal_id, domain_type, business_size, selected_project_ids,
                total_score, verified_score, threshold, margin, overall_risk, risk_factors)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (proposal_id, domain_type) DO UPDATE SET
                    business_size = EXCLUDED.business_size,
                    selected_project_ids = EXCLUDED.selected_project_ids,
                    total_score = EXCLUDED.total_score,
                    verified_score = EXCLUDED.verified_score,
                    threshold = EXCLUDED.threshold,
                    margin = EXCLUDED.margin,
                    overall_risk = EXCLUDED.overall_risk,
                    risk_factors = EXCLUDED.risk_factors,
                    created_at = CURRENT_TIMESTAMP
            """, (
                proposal_id, domain_type, business_size,
                Json(selected_project_ids), total_score, verified_score,
                threshold, margin, overall_risk, Json(risk_factors)
            ))
        else:
            cursor.execute("""
                INSERT OR REPLACE INTO oasis_optimization_results
                (proposal_id, domain_type, business_size, selected_project_ids,
                total_score, verified_score, threshold, margin, overall_risk, risk_factors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                proposal_id, domain_type, business_size,
                json.dumps(selected_project_ids), total_score, verified_score,
                threshold, margin, overall_risk, json.dumps(risk_factors)
            ))

        self.conn.commit()
        cursor.close()
        return True

    def get_optimization_result(
        self,
        proposal_id: str,
        domain_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Get optimization result for a proposal/domain"""
        cursor = self.conn.cursor()

        if self.is_postgres:
            cursor.execute("""
                SELECT * FROM oasis_optimization_results
                WHERE proposal_id = %s AND domain_type = %s
            """, (proposal_id, domain_type))
        else:
            cursor.execute("""
                SELECT * FROM oasis_optimization_results
                WHERE proposal_id = ? AND domain_type = ?
            """, (proposal_id, domain_type))

        row = cursor.fetchone()
        cursor.close()

        if row:
            return dict(row)
        return None


# Convenience function for getting a database instance
_db_instance: Optional[OASISDatabase] = None


def get_database(
    connection_string: Optional[str] = None,
    sqlite_path: Optional[str] = None,
) -> OASISDatabase:
    """Get or create database instance"""
    global _db_instance

    if _db_instance is None:
        _db_instance = OASISDatabase(
            connection_string=connection_string,
            sqlite_path=sqlite_path,
        )

    return _db_instance
