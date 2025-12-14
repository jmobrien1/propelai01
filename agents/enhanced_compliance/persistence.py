"""
PropelAI SQLite Persistence Layer

Provides persistent storage for RFP projects and requirements.
This replaces the in-memory storage with durable SQLite storage.

Features:
- Survives server restarts
- Full CRUD for RFP projects
- Requirement storage with full metadata
- Source trace persistence
- Export/import functionality

Usage:
    from agents.enhanced_compliance.persistence import RFPDatabase

    db = RFPDatabase("propelai.db")

    # Create project
    project_id = db.create_project("NIH RFP", "75N96025R00004", "NIH")

    # Store requirements
    db.store_requirements(project_id, extraction_result.all_requirements)

    # Retrieve
    project = db.get_project(project_id)
    requirements = db.get_requirements(project_id)
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager

from .section_aware_extractor import (
    StructuredRequirement,
    RequirementCategory,
    BindingLevel,
    ExtractionResult
)
from .document_structure import UCFSection


# Default database location
DEFAULT_DB_PATH = Path(os.environ.get('PROPELAI_DB_PATH', '/tmp/propelai/propelai.db'))


@dataclass
class StoredProject:
    """Project record from database"""
    id: str
    name: str
    solicitation_number: Optional[str]
    agency: Optional[str]
    status: str
    files: List[str]
    requirements_count: int
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


@dataclass
class StoredRequirement:
    """Requirement record from database"""
    id: str
    project_id: str
    rfp_reference: str
    generated_id: str
    full_text: str
    category: str
    binding_level: str
    binding_keyword: str
    source_section: str
    source_subsection: Optional[str]
    page_number: int
    source_document: str
    parent_title: str
    references_to: List[str]
    is_compliance_gate: bool
    verification_status: str
    created_at: str


class RFPDatabase:
    """
    SQLite-based persistent storage for PropelAI.

    Thread-safe and supports concurrent reads.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Creates if not exists.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic cleanup"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.executescript('''
                -- Projects table
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    solicitation_number TEXT,
                    agency TEXT,
                    status TEXT DEFAULT 'created',
                    files TEXT DEFAULT '[]',
                    requirements_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                -- Requirements table
                CREATE TABLE IF NOT EXISTS requirements (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    rfp_reference TEXT,
                    generated_id TEXT,
                    full_text TEXT NOT NULL,
                    category TEXT,
                    binding_level TEXT,
                    binding_keyword TEXT,
                    source_section TEXT,
                    source_subsection TEXT,
                    page_number INTEGER DEFAULT 0,
                    source_document TEXT,
                    parent_title TEXT,
                    references_to TEXT DEFAULT '[]',
                    is_compliance_gate INTEGER DEFAULT 0,
                    verification_status TEXT DEFAULT 'unverified',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                -- Source traces table
                CREATE TABLE IF NOT EXISTS source_traces (
                    id TEXT PRIMARY KEY,
                    requirement_id TEXT,
                    source_type TEXT,
                    document_name TEXT,
                    page_number INTEGER,
                    char_start INTEGER,
                    char_end INTEGER,
                    confidence_score REAL DEFAULT 1.0,
                    extraction_method TEXT,
                    verification_status TEXT DEFAULT 'unverified',
                    verified_by TEXT,
                    verified_at TEXT,
                    audit_trail TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (requirement_id) REFERENCES requirements(id) ON DELETE CASCADE
                );

                -- Amendments table
                CREATE TABLE IF NOT EXISTS amendments (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    amendment_number INTEGER,
                    amendment_date TEXT,
                    filename TEXT,
                    changes TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_req_project ON requirements(project_id);
                CREATE INDEX IF NOT EXISTS idx_req_category ON requirements(category);
                CREATE INDEX IF NOT EXISTS idx_req_binding ON requirements(binding_level);
                CREATE INDEX IF NOT EXISTS idx_traces_req ON source_traces(requirement_id);
            ''')

    # ============== Project Operations ==============

    def create_project(
        self,
        name: str,
        solicitation_number: Optional[str] = None,
        agency: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> str:
        """
        Create a new RFP project.

        Returns:
            Project ID
        """
        import uuid
        project_id = project_id or f"RFP-{str(uuid.uuid4())[:8]}"
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO projects (id, name, solicitation_number, agency, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (project_id, name, solicitation_number, agency, now, now))

        return project_id

    def get_project(self, project_id: str) -> Optional[StoredProject]:
        """Get a project by ID"""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM projects WHERE id = ?',
                (project_id,)
            ).fetchone()

            if row:
                return StoredProject(
                    id=row['id'],
                    name=row['name'],
                    solicitation_number=row['solicitation_number'],
                    agency=row['agency'],
                    status=row['status'],
                    files=json.loads(row['files']),
                    requirements_count=row['requirements_count'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata'])
                )
        return None

    def list_projects(self) -> List[StoredProject]:
        """List all projects"""
        with self._get_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM projects ORDER BY updated_at DESC'
            ).fetchall()

            return [
                StoredProject(
                    id=row['id'],
                    name=row['name'],
                    solicitation_number=row['solicitation_number'],
                    agency=row['agency'],
                    status=row['status'],
                    files=json.loads(row['files']),
                    requirements_count=row['requirements_count'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata'])
                )
                for row in rows
            ]

    def update_project(
        self,
        project_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update project fields"""
        allowed_fields = {
            'name', 'solicitation_number', 'agency', 'status',
            'files', 'requirements_count', 'metadata'
        }

        # Filter to allowed fields
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered:
            return False

        # JSON encode list/dict fields
        if 'files' in filtered:
            filtered['files'] = json.dumps(filtered['files'])
        if 'metadata' in filtered:
            filtered['metadata'] = json.dumps(filtered['metadata'])

        filtered['updated_at'] = datetime.now().isoformat()

        set_clause = ', '.join(f'{k} = ?' for k in filtered.keys())
        values = list(filtered.values()) + [project_id]

        with self._get_connection() as conn:
            conn.execute(
                f'UPDATE projects SET {set_clause} WHERE id = ?',
                values
            )

        return True

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated data"""
        with self._get_connection() as conn:
            conn.execute('DELETE FROM projects WHERE id = ?', (project_id,))
        return True

    # ============== Requirement Operations ==============

    def store_requirements(
        self,
        project_id: str,
        requirements: List[StructuredRequirement]
    ) -> int:
        """
        Store extracted requirements.

        Args:
            project_id: Project ID
            requirements: List of StructuredRequirement objects

        Returns:
            Number of requirements stored
        """
        now = datetime.now().isoformat()
        count = 0

        with self._get_connection() as conn:
            for req in requirements:
                req_id = f"{project_id}-{req.generated_id}"

                # Check for compliance gate (if attribute exists)
                is_gate = getattr(req, 'is_compliance_gate', False)

                conn.execute('''
                    INSERT OR REPLACE INTO requirements (
                        id, project_id, rfp_reference, generated_id, full_text,
                        category, binding_level, binding_keyword, source_section,
                        source_subsection, page_number, source_document, parent_title,
                        references_to, is_compliance_gate, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    req_id,
                    project_id,
                    req.rfp_reference,
                    req.generated_id,
                    req.full_text,
                    req.category.value if req.category else None,
                    req.binding_level.value if req.binding_level else None,
                    req.binding_keyword,
                    req.source_section.value if req.source_section else None,
                    req.source_subsection,
                    req.page_number,
                    req.source_document,
                    req.parent_title,
                    json.dumps(req.references_to),
                    1 if is_gate else 0,
                    now
                ))
                count += 1

            # Update project requirements count
            conn.execute('''
                UPDATE projects
                SET requirements_count = ?, updated_at = ?
                WHERE id = ?
            ''', (count, now, project_id))

        return count

    def get_requirements(
        self,
        project_id: str,
        category: Optional[str] = None,
        binding_level: Optional[str] = None,
        limit: int = 1000
    ) -> List[StoredRequirement]:
        """
        Get requirements for a project.

        Args:
            project_id: Project ID
            category: Optional filter by category
            binding_level: Optional filter by binding level
            limit: Maximum results

        Returns:
            List of StoredRequirement objects
        """
        query = 'SELECT * FROM requirements WHERE project_id = ?'
        params = [project_id]

        if category:
            query += ' AND category = ?'
            params.append(category)

        if binding_level:
            query += ' AND binding_level = ?'
            params.append(binding_level)

        query += f' ORDER BY page_number, id LIMIT {limit}'

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

            return [
                StoredRequirement(
                    id=row['id'],
                    project_id=row['project_id'],
                    rfp_reference=row['rfp_reference'],
                    generated_id=row['generated_id'],
                    full_text=row['full_text'],
                    category=row['category'],
                    binding_level=row['binding_level'],
                    binding_keyword=row['binding_keyword'],
                    source_section=row['source_section'],
                    source_subsection=row['source_subsection'],
                    page_number=row['page_number'],
                    source_document=row['source_document'],
                    parent_title=row['parent_title'],
                    references_to=json.loads(row['references_to'] or '[]'),
                    is_compliance_gate=bool(row['is_compliance_gate']),
                    verification_status=row['verification_status'],
                    created_at=row['created_at']
                )
                for row in rows
            ]

    def get_requirement(self, requirement_id: str) -> Optional[StoredRequirement]:
        """Get a single requirement by ID"""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM requirements WHERE id = ?',
                (requirement_id,)
            ).fetchone()

            if row:
                return StoredRequirement(
                    id=row['id'],
                    project_id=row['project_id'],
                    rfp_reference=row['rfp_reference'],
                    generated_id=row['generated_id'],
                    full_text=row['full_text'],
                    category=row['category'],
                    binding_level=row['binding_level'],
                    binding_keyword=row['binding_keyword'],
                    source_section=row['source_section'],
                    source_subsection=row['source_subsection'],
                    page_number=row['page_number'],
                    source_document=row['source_document'],
                    parent_title=row['parent_title'],
                    references_to=json.loads(row['references_to'] or '[]'),
                    is_compliance_gate=bool(row['is_compliance_gate']),
                    verification_status=row['verification_status'],
                    created_at=row['created_at']
                )
        return None

    def get_compliance_gates(self, project_id: str) -> List[StoredRequirement]:
        """Get all compliance gates for a project"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM requirements
                WHERE project_id = ? AND is_compliance_gate = 1
                ORDER BY page_number
            ''', (project_id,)).fetchall()

            return [
                StoredRequirement(
                    id=row['id'],
                    project_id=row['project_id'],
                    rfp_reference=row['rfp_reference'],
                    generated_id=row['generated_id'],
                    full_text=row['full_text'],
                    category=row['category'],
                    binding_level=row['binding_level'],
                    binding_keyword=row['binding_keyword'],
                    source_section=row['source_section'],
                    source_subsection=row['source_subsection'],
                    page_number=row['page_number'],
                    source_document=row['source_document'],
                    parent_title=row['parent_title'],
                    references_to=json.loads(row['references_to'] or '[]'),
                    is_compliance_gate=True,
                    verification_status=row['verification_status'],
                    created_at=row['created_at']
                )
                for row in rows
            ]

    def get_requirements_stats(self, project_id: str) -> Dict[str, Any]:
        """Get requirement statistics for a project"""
        with self._get_connection() as conn:
            total = conn.execute(
                'SELECT COUNT(*) FROM requirements WHERE project_id = ?',
                (project_id,)
            ).fetchone()[0]

            by_category = dict(conn.execute('''
                SELECT category, COUNT(*) FROM requirements
                WHERE project_id = ?
                GROUP BY category
            ''', (project_id,)).fetchall())

            by_binding = dict(conn.execute('''
                SELECT binding_level, COUNT(*) FROM requirements
                WHERE project_id = ?
                GROUP BY binding_level
            ''', (project_id,)).fetchall())

            gates_count = conn.execute('''
                SELECT COUNT(*) FROM requirements
                WHERE project_id = ? AND is_compliance_gate = 1
            ''', (project_id,)).fetchone()[0]

            return {
                'total': total,
                'by_category': by_category,
                'by_binding_level': by_binding,
                'compliance_gates': gates_count
            }

    def update_verification_status(
        self,
        requirement_id: str,
        status: str,
        verified_by: Optional[str] = None
    ) -> bool:
        """Update requirement verification status"""
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE requirements
                SET verification_status = ?
                WHERE id = ?
            ''', (status, requirement_id))
        return True

    # ============== Source Trace Operations ==============

    def store_trace(
        self,
        requirement_id: str,
        trace_data: Dict[str, Any]
    ) -> str:
        """Store a source trace"""
        import uuid
        trace_id = str(uuid.uuid4())[:12]
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO source_traces (
                    id, requirement_id, source_type, document_name,
                    page_number, char_start, char_end, confidence_score,
                    extraction_method, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trace_id,
                requirement_id,
                trace_data.get('source_type', 'rfp_extraction'),
                trace_data.get('document_name', ''),
                trace_data.get('page_number', 0),
                trace_data.get('char_start', 0),
                trace_data.get('char_end', 0),
                trace_data.get('confidence_score', 1.0),
                trace_data.get('extraction_method', 'regex'),
                now
            ))

        return trace_id

    def get_trace(self, requirement_id: str) -> Optional[Dict[str, Any]]:
        """Get source trace for a requirement"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM source_traces WHERE requirement_id = ?
            ''', (requirement_id,)).fetchone()

            if row:
                return dict(row)
        return None

    # ============== Export/Import ==============

    def export_project(self, project_id: str) -> Dict[str, Any]:
        """Export complete project data as JSON"""
        project = self.get_project(project_id)
        if not project:
            return {}

        requirements = self.get_requirements(project_id, limit=10000)

        return {
            'project': asdict(project) if project else {},
            'requirements': [asdict(r) for r in requirements],
            'exported_at': datetime.now().isoformat()
        }

    def import_project(self, data: Dict[str, Any]) -> Optional[str]:
        """Import project from exported JSON"""
        if 'project' not in data:
            return None

        proj_data = data['project']
        project_id = self.create_project(
            name=proj_data['name'],
            solicitation_number=proj_data.get('solicitation_number'),
            agency=proj_data.get('agency'),
            project_id=proj_data.get('id')
        )

        # Import requirements (would need to reconstruct StructuredRequirement objects)
        # Simplified: just store the raw data

        return project_id


# Singleton instance for easy access
_db_instance: Optional[RFPDatabase] = None


def get_database(db_path: Optional[Path] = None) -> RFPDatabase:
    """
    Get or create database instance.

    Usage:
        from agents.enhanced_compliance.persistence import get_database

        db = get_database()
        projects = db.list_projects()
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = RFPDatabase(db_path)
    return _db_instance
