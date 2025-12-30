"""
PropelAI v5.0 Test Configuration
================================
QA Swarm: Trajectory-First Testing Framework

Fixtures:
- Async PostgreSQL mock engine
- GoldenRFP: Realistic C/L/M sections
- Mock EmbeddingGenerator with deterministic vectors
- Agent Trace Log fixtures
"""

import pytest
import asyncio
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_compliance.models import (
    RequirementNode,
    RequirementType,
    ConfidenceLevel,
    SourceLocation,
    DocumentType,
    BoundingBox,
    VisualRect,
    SourceCoordinate,
)
from agents.enhanced_compliance.requirements_graph import (
    RequirementsDAG,
    EdgeType,
    NodeSection,
)
from agents.enhanced_compliance.validation_engine import (
    ValidationEngine,
    ViolationType,
    Severity,
)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (may use real services)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full workflow)")
    config.addinivalue_line("markers", "slow: Slow tests (>5s)")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Golden RFP Fixture - Realistic C/L/M Content
# =============================================================================

GOLDEN_SECTION_C = """
C.1 SCOPE OF WORK
The Contractor shall provide cloud-based infrastructure management services
for the Agency's mission-critical applications.

C.2 TECHNICAL REQUIREMENTS
C.2.1 The Contractor shall ensure 99.99% system uptime for all production environments.
C.2.2 The Contractor shall implement automated failover within 30 seconds of detected failure.
C.2.3 The system shall support a minimum of 10,000 concurrent users.
C.2.4 The Contractor shall provide 24/7/365 monitoring and incident response.

C.3 SECURITY REQUIREMENTS
C.3.1 The Contractor shall comply with FedRAMP High baseline requirements.
C.3.2 All data shall be encrypted at rest using AES-256 encryption.
C.3.3 The Contractor shall implement multi-factor authentication for all user access.

C.4 DELIVERABLES
C.4.1 Monthly performance reports shall be submitted by the 5th business day.
C.4.2 Incident response documentation shall be provided within 24 hours.
"""

GOLDEN_SECTION_L = """
L.1 PROPOSAL FORMAT
L.1.1 Proposals shall be limited to 100 pages for the Technical Volume.
L.1.2 Font size shall be 11-point Times New Roman minimum.
L.1.3 Margins shall be 1 inch on all sides.

L.2 TECHNICAL VOLUME INSTRUCTIONS
L.2.1 The Offeror shall describe their approach to ensuring system uptime.
L.2.2 The Offeror shall provide evidence of FedRAMP High authorization.
L.2.3 The Offeror shall describe their incident response procedures.

L.3 PAST PERFORMANCE VOLUME
L.3.1 The Offeror shall provide three (3) relevant past performance references.
L.3.2 Each reference shall include contract value and period of performance.

L.4 MANAGEMENT VOLUME
L.4.1 The Offeror shall describe their key personnel qualifications.
L.4.2 The Offeror shall provide an organizational chart.
"""

GOLDEN_SECTION_M = """
M.1 EVALUATION FACTORS
Technical Approach is significantly more important than Past Performance.
Past Performance is more important than Cost.

M.2 TECHNICAL APPROACH (Most Important)
M.2.1 The Government will evaluate the Offeror's understanding of requirements.
M.2.2 The Government will assess the feasibility of the proposed technical solution.
M.2.3 Innovation and risk mitigation approaches will be evaluated.

M.3 PAST PERFORMANCE (Important)
M.3.1 Relevance of prior contracts to current requirement will be assessed.
M.3.2 Quality of performance on similar contracts will be evaluated.

M.4 COST
M.4.1 Cost realism analysis will be performed.
M.4.2 The Government will evaluate price reasonableness.
"""


@dataclass
class GoldenRFP:
    """A realistic RFP with C/L/M sections for testing"""
    id: str
    section_c: str
    section_l: str
    section_m: str
    requirements: List[RequirementNode]

    @property
    def full_text(self) -> str:
        return f"SECTION C - STATEMENT OF WORK\n{self.section_c}\n\n" \
               f"SECTION L - INSTRUCTIONS\n{self.section_l}\n\n" \
               f"SECTION M - EVALUATION\n{self.section_m}"


def _create_requirement(
    id: str,
    text: str,
    req_type: RequirementType,
    section_id: str,
    page: int = 1,
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
) -> RequirementNode:
    """Helper to create a RequirementNode"""
    doc_type = DocumentType.MAIN_SOLICITATION
    if section_id.startswith('C'):
        doc_type = DocumentType.STATEMENT_OF_WORK

    return RequirementNode(
        id=id,
        text=text,
        requirement_type=req_type,
        confidence=confidence,
        source=SourceLocation(
            document_name="RFP-GOLDEN.pdf",
            document_type=doc_type,
            page_number=page,
            section_id=section_id,
        )
    )


@pytest.fixture
def golden_rfp() -> GoldenRFP:
    """
    GoldenRFP fixture with realistic C/L/M requirements.

    This provides a deterministic test dataset that covers:
    - Section C: Performance requirements
    - Section L: Proposal instructions
    - Section M: Evaluation criteria
    - Cross-references between sections
    """
    requirements = [
        # Section C - Performance Requirements
        _create_requirement(
            "REQ-C-001",
            "The Contractor shall ensure 99.99% system uptime for all production environments.",
            RequirementType.PERFORMANCE,
            "C.2.1",
            page=2
        ),
        _create_requirement(
            "REQ-C-002",
            "The Contractor shall implement automated failover within 30 seconds.",
            RequirementType.PERFORMANCE,
            "C.2.2",
            page=2
        ),
        _create_requirement(
            "REQ-C-003",
            "The Contractor shall comply with FedRAMP High baseline requirements.",
            RequirementType.COMPLIANCE,
            "C.3.1",
            page=3
        ),
        _create_requirement(
            "REQ-C-004",
            "Monthly performance reports shall be submitted by the 5th business day.",
            RequirementType.DELIVERABLE,
            "C.4.1",
            page=4
        ),
        # Orphan: No L or M link
        _create_requirement(
            "REQ-C-005",
            "The system shall support a minimum of 10,000 concurrent users.",
            RequirementType.PERFORMANCE_METRIC,
            "C.2.3",
            page=2
        ),

        # Section L - Proposal Instructions
        _create_requirement(
            "REQ-L-001",
            "The Offeror shall describe their approach to ensuring system uptime.",
            RequirementType.PROPOSAL_INSTRUCTION,
            "L.2.1",
            page=5
        ),
        _create_requirement(
            "REQ-L-002",
            "The Offeror shall provide evidence of FedRAMP High authorization.",
            RequirementType.PROPOSAL_INSTRUCTION,
            "L.2.2",
            page=5
        ),
        _create_requirement(
            "REQ-L-003",
            "Proposals shall be limited to 100 pages for the Technical Volume.",
            RequirementType.FORMAT,
            "L.1.1",
            page=5
        ),
        _create_requirement(
            "REQ-L-004",
            "The Offeror shall provide three (3) relevant past performance references.",
            RequirementType.PROPOSAL_INSTRUCTION,
            "L.3.1",
            page=6
        ),

        # Section M - Evaluation Criteria
        _create_requirement(
            "REQ-M-001",
            "The Government will evaluate the Offeror's understanding of requirements.",
            RequirementType.EVALUATION_CRITERION,
            "M.2.1",
            page=7
        ),
        _create_requirement(
            "REQ-M-002",
            "Innovation and risk mitigation approaches will be evaluated.",
            RequirementType.EVALUATION_CRITERION,
            "M.2.3",
            page=7
        ),
        _create_requirement(
            "REQ-M-003",
            "Relevance of prior contracts to current requirement will be assessed.",
            RequirementType.EVALUATION_CRITERION,
            "M.3.1",
            page=8
        ),
    ]

    return GoldenRFP(
        id="RFP-GOLDEN-001",
        section_c=GOLDEN_SECTION_C,
        section_l=GOLDEN_SECTION_L,
        section_m=GOLDEN_SECTION_M,
        requirements=requirements,
    )


@pytest.fixture
def golden_requirements(golden_rfp) -> List[RequirementNode]:
    """Just the requirements from GoldenRFP"""
    return golden_rfp.requirements


@pytest.fixture
def section_c_requirements(golden_requirements) -> List[RequirementNode]:
    """Only Section C requirements"""
    return [r for r in golden_requirements if r.source.section_id.startswith('C')]


@pytest.fixture
def section_l_requirements(golden_requirements) -> List[RequirementNode]:
    """Only Section L requirements"""
    return [r for r in golden_requirements if r.source.section_id.startswith('L')]


@pytest.fixture
def section_m_requirements(golden_requirements) -> List[RequirementNode]:
    """Only Section M requirements"""
    return [r for r in golden_requirements if r.source.section_id.startswith('M')]


# =============================================================================
# Mock Embedding Generator
# =============================================================================

class MockEmbeddingGenerator:
    """
    Mock EmbeddingGenerator that returns deterministic 1536-dim vectors.

    Uses text hash to generate consistent embeddings for the same input.
    This enables testing semantic search behavior without API calls.
    """

    def __init__(self, dimension: int = 1536, fail_on: Optional[List[str]] = None):
        self.dimension = dimension
        self.fail_on = fail_on or []  # Text patterns that trigger failure
        self.call_count = 0
        self.texts_embedded = []

    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to deterministic vector using hash"""
        # Create deterministic seed from text
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Generate vector from hash
        import random
        random.seed(int(text_hash[:8], 16))

        # Normalize to unit vector
        raw = [random.gauss(0, 1) for _ in range(self.dimension)]
        magnitude = sum(x**2 for x in raw) ** 0.5
        return [x / magnitude for x in raw]

    async def generate(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text"""
        self.call_count += 1
        self.texts_embedded.append(text)

        # Check for failure conditions
        for pattern in self.fail_on:
            if pattern.lower() in text.lower():
                return None

        return self._text_to_vector(text)

    async def generate_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for batch of texts"""
        return [await self.generate(text) for text in texts]


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for semantic search tests"""
    return MockEmbeddingGenerator()


@pytest.fixture
def failing_embedding_generator():
    """Mock embedding generator that fails on specific patterns"""
    return MockEmbeddingGenerator(fail_on=["FAIL_EMBEDDING", "invalid"])


# =============================================================================
# Requirements Graph Fixtures
# =============================================================================

@pytest.fixture
def empty_dag() -> RequirementsDAG:
    """Empty requirements DAG"""
    return RequirementsDAG()


@pytest.fixture
def populated_dag(golden_requirements) -> RequirementsDAG:
    """DAG populated with golden requirements (no auto-linking)"""
    dag = RequirementsDAG()
    dag.add_requirements(golden_requirements)
    return dag


@pytest.fixture
def linked_dag(golden_requirements) -> RequirementsDAG:
    """DAG with auto-built Iron Triangle edges"""
    return RequirementsDAG.from_requirements(
        golden_requirements,
        auto_link=True,
        similarity_threshold=0.2
    )


# =============================================================================
# Validation Engine Fixtures
# =============================================================================

@pytest.fixture
def validation_engine() -> ValidationEngine:
    """Fresh validation engine"""
    return ValidationEngine()


@pytest.fixture
def past_performance_outline() -> Dict[str, Any]:
    """Proposal outline with Past Performance volume"""
    return {
        "volumes": [
            {
                "type": "technical",
                "sections": [
                    {"id": "C.2", "title": "Technical Approach"},
                ]
            },
            {
                "type": "past_performance",
                "sections": [
                    {"id": "L.3", "title": "Past Performance"},
                ]
            },
        ]
    }


# =============================================================================
# Source Coordinate Fixtures (Multi-Page Spanning)
# =============================================================================

@pytest.fixture
def single_page_coordinate() -> SourceCoordinate:
    """SourceCoordinate on a single page"""
    bbox = BoundingBox(
        x0=100, y0=200, x1=500, y1=250,
        page_width=612, page_height=792
    )
    return SourceCoordinate(
        document_id="RFP-001.pdf",
        page_number=5,
        bounding_box=bbox,
        text_snippet="The Contractor shall provide..."
    )


@pytest.fixture
def multi_page_coordinate() -> SourceCoordinate:
    """SourceCoordinate spanning multiple pages"""
    bbox1 = BoundingBox(
        x0=100, y0=700, x1=500, y1=792,  # Bottom of page 5
        page_width=612, page_height=792
    )
    bbox2 = BoundingBox(
        x0=100, y0=0, x1=500, y1=100,  # Top of page 6
        page_width=612, page_height=792
    )

    coord = SourceCoordinate(
        document_id="RFP-001.pdf",
        page_number=5,
        bounding_box=bbox1,
        text_snippet="The Contractor shall provide cloud-based services...",
        visual_rects=[
            VisualRect(page_number=5, bounding_box=bbox1),
            VisualRect(page_number=6, bounding_box=bbox2),
        ]
    )
    return coord


# =============================================================================
# Agent Trace Log Fixtures
# =============================================================================

@pytest.fixture
def trace_log_entry() -> Dict[str, Any]:
    """Sample trace log entry"""
    return {
        "rfp_id": "RFP-GOLDEN-001",
        "agent_name": "ComplianceAgent",
        "action": "extract_requirements",
        "input_data": {
            "document": "RFP-GOLDEN.pdf",
            "pages": [1, 2, 3, 4, 5]
        },
        "output_data": {
            "requirements_count": 12,
            "section_c_count": 5,
            "section_l_count": 4,
            "section_m_count": 3
        },
        "confidence_score": 0.92,
        "duration_ms": 1250,
        "model_name": "gpt-4o-mini",
        "status": "completed"
    }


@pytest.fixture
def trace_correction() -> Dict[str, Any]:
    """Sample human correction for trace log"""
    return {
        "human_correction": {
            "requirements_count": 14,  # Found 2 more
            "section_c_count": 6,
            "missed_requirements": ["REQ-C-006", "REQ-C-007"]
        },
        "correction_type": "modified",
        "correction_reason": "Missed 2 SHALL statements in C.3 Security section"
    }


# =============================================================================
# Async Database Mock (PostgreSQL)
# =============================================================================

@pytest.fixture
def mock_db_session():
    """
    Mock async database session for PostgreSQL operations.

    Simulates SQLAlchemy async session without actual database.
    """
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.scalar_one_or_none = AsyncMock(return_value=None)

    # Context manager support
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    return session


@pytest.fixture
def mock_get_session(mock_db_session):
    """Patch get_session to return mock session"""
    with patch('api.database.get_session') as mock:
        mock.return_value.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock.return_value.__aexit__ = AsyncMock(return_value=None)
        yield mock


# =============================================================================
# API Test Client Fixture
# =============================================================================

@pytest.fixture
def api_client():
    """
    FastAPI test client for integration tests.

    Note: Requires httpx for async support.
    """
    try:
        from httpx import AsyncClient
        from api.main import app

        return AsyncClient(app=app, base_url="http://test")
    except ImportError:
        pytest.skip("httpx not installed")


# =============================================================================
# Helper Functions for Tests
# =============================================================================

def assert_orphan_detected(orphans, req_id: str, reason_contains: str = None):
    """Assert that a specific orphan was detected"""
    orphan_ids = [o.orphan_id for o in orphans]
    assert req_id in orphan_ids, f"Expected {req_id} to be orphan, found: {orphan_ids}"

    if reason_contains:
        orphan = next(o for o in orphans if o.orphan_id == req_id)
        assert reason_contains.lower() in orphan.reason.lower(), \
            f"Expected reason to contain '{reason_contains}', got: {orphan.reason}"


def assert_violation_exists(
    violations,
    violation_type: ViolationType,
    severity: Severity = None,
    req_id: str = None
):
    """Assert that a specific violation exists"""
    matching = [v for v in violations if v.violation_type == violation_type]

    assert len(matching) > 0, \
        f"Expected {violation_type.value} violation, found: {[v.violation_type.value for v in violations]}"

    if severity:
        matching = [v for v in matching if v.severity == severity]
        assert len(matching) > 0, f"No {violation_type.value} with severity {severity.value}"

    if req_id:
        matching = [v for v in matching if v.requirement_id == req_id]
        assert len(matching) > 0, f"No {violation_type.value} for requirement {req_id}"


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a ** 2 for a in vec1) ** 0.5
    mag2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0
