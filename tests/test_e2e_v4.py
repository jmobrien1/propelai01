#!/usr/bin/env python3
"""
PropelAI v4.0/v4.1 End-to-End Verification Tests

Tests all five phases of the v4.x architecture:
- Phase 1: Trust Gate (PDF coordinate extraction)
- Phase 2: Iron Triangle (Strategy Agent)
- Phase 3: Drafting Agent (LangGraph workflow)
- Phase 4: Persistence (Database + Vector Store)
- Phase 5: Team Workspaces (RBAC & Collaboration) - v4.1

Run with: python -m pytest tests/test_e2e_v4.py -v
Or standalone: python tests/test_e2e_v4.py
"""

import sys
import os
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    skipped: bool = False  # For environment dependency issues


class V4EndToEndTests:
    """End-to-end verification for PropelAI v4.0"""

    def __init__(self):
        self.results: List[TestResult] = []

    def record(self, name: str, passed: bool, message: str, details: Dict = None, skipped: bool = False):
        """Record a test result"""
        self.results.append(TestResult(name, passed, message, details, skipped))
        if skipped:
            status = "○ SKIP"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed and not skipped:
            print(f"         {message}")
        elif skipped:
            print(f"         (env) {message}")

    # =========================================================================
    # Phase 1: Trust Gate Tests
    # =========================================================================

    def test_phase1_imports(self):
        """Test that Trust Gate components can be imported"""
        try:
            from agents.enhanced_compliance import (
                TRUST_GATE_AVAILABLE,
                BoundingBox,
                SourceCoordinate,
            )
            if TRUST_GATE_AVAILABLE:
                self.record(
                    "Phase 1: Trust Gate imports",
                    True,
                    "All Trust Gate components available"
                )
            else:
                # pdfplumber not installed is an environment issue, not a code bug
                self.record(
                    "Phase 1: Trust Gate imports",
                    True,  # Core imports work
                    "Core models available (pdfplumber not installed for full Trust Gate)",
                    skipped=False  # Models are available
                )
        except ImportError as e:
            self.record(
                "Phase 1: Trust Gate imports",
                False,
                f"Import failed: {e}"
            )

    def test_phase1_bounding_box(self):
        """Test BoundingBox dataclass"""
        try:
            from agents.enhanced_compliance.models import BoundingBox

            # BoundingBox uses PDF coordinate system: x0, y0, x1, y1 (points)
            box = BoundingBox(
                x0=72.0,      # Left edge (1 inch from left)
                y0=100.0,     # Bottom edge
                x1=540.0,     # Right edge
                y1=150.0,     # Top edge
                page_width=612.0,   # Letter width in points
                page_height=792.0   # Letter height in points
            )

            assert box.x0 < box.x1, "x0 should be less than x1"
            assert box.y0 < box.y1, "y0 should be less than y1"
            assert box.page_width > 0, "page_width must be positive"

            # Test CSS percent conversion
            css = box.to_css_percent()
            assert "left" in css, "CSS should have left property"
            assert "top" in css, "CSS should have top property"
            assert 0 <= css["left"] <= 100, "CSS left should be percentage"

            self.record(
                "Phase 1: BoundingBox dataclass",
                True,
                "BoundingBox creates valid PDF coordinates with CSS conversion",
                {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1, "css_left": f"{css['left']:.1f}%"}
            )
        except Exception as e:
            self.record(
                "Phase 1: BoundingBox dataclass",
                False,
                f"BoundingBox test failed: {e}"
            )

    def test_phase1_source_coordinate(self):
        """Test SourceCoordinate dataclass"""
        try:
            from agents.enhanced_compliance.models import BoundingBox, SourceCoordinate

            # SourceCoordinate uses page_number (1-indexed) and a single bounding_box
            box = BoundingBox(
                x0=72.0, y0=100.0, x1=540.0, y1=150.0,
                page_width=612.0, page_height=792.0
            )

            coord = SourceCoordinate(
                document_id="test-doc-123",
                page_number=1,
                bounding_box=box,
                text_snippet="The contractor shall provide...",
                extraction_method="pdfplumber",
                confidence=0.95
            )

            assert coord.document_id == "test-doc-123"
            assert coord.page_number == 1
            assert coord.bounding_box is not None
            assert 0.0 <= coord.confidence <= 1.0

            # Test serialization
            coord_dict = coord.to_dict()
            assert "document_id" in coord_dict
            assert "bounding_box" in coord_dict

            self.record(
                "Phase 1: SourceCoordinate dataclass",
                True,
                "SourceCoordinate links requirements to PDF locations",
                {"document_id": coord.document_id, "page": coord.page_number, "confidence": coord.confidence}
            )
        except Exception as e:
            self.record(
                "Phase 1: SourceCoordinate dataclass",
                False,
                f"SourceCoordinate test failed: {e}"
            )

    def test_phase1_pdf_extractor(self):
        """Test PDFCoordinateExtractor availability"""
        try:
            from agents.enhanced_compliance import TRUST_GATE_AVAILABLE

            if TRUST_GATE_AVAILABLE:
                from agents.enhanced_compliance.pdf_coordinate_extractor import (
                    PDFCoordinateExtractor,
                    get_coordinate_extractor,
                )
                extractor = get_coordinate_extractor()
                self.record(
                    "Phase 1: PDFCoordinateExtractor",
                    True,
                    "PDF coordinate extractor is available",
                    {"extractor_type": type(extractor).__name__}
                )
            else:
                # Skip - pdfplumber is an optional dependency
                self.record(
                    "Phase 1: PDFCoordinateExtractor",
                    True,  # Not a failure - optional dependency
                    "pdfplumber not installed (optional for production)",
                    skipped=True
                )
        except Exception as e:
            self.record(
                "Phase 1: PDFCoordinateExtractor",
                False,
                f"PDFCoordinateExtractor test failed: {e}"
            )

    # =========================================================================
    # Phase 2: Iron Triangle Tests
    # =========================================================================

    def test_phase2_imports(self):
        """Test that Iron Triangle components can be imported"""
        try:
            from agents.strategy_agent import (
                StrategyAgent,
                CompetitorAnalyzer,
                GhostingLanguageGenerator,
            )
            self.record(
                "Phase 2: Iron Triangle imports",
                True,
                "All Iron Triangle components available"
            )
        except ImportError as e:
            self.record(
                "Phase 2: Iron Triangle imports",
                False,
                f"Import failed: {e}"
            )

    def test_phase2_strategy_agent(self):
        """Test StrategyAgent instantiation"""
        try:
            from agents.strategy_agent import StrategyAgent

            agent = StrategyAgent(use_llm=False)  # Disable LLM for testing

            # Check agent has key attributes and internal methods
            assert hasattr(agent, 'llm_provider'), "Missing llm_provider attribute"
            assert hasattr(agent, 'use_llm'), "Missing use_llm attribute"
            assert hasattr(agent, '_analyze_evaluation_factors'), "Missing _analyze_evaluation_factors method"
            assert hasattr(agent, '_develop_win_themes'), "Missing _develop_win_themes method"

            self.record(
                "Phase 2: StrategyAgent",
                True,
                "StrategyAgent instantiates with Iron Triangle methods",
                {"use_llm": agent.use_llm, "provider": agent.llm_provider}
            )
        except Exception as e:
            self.record(
                "Phase 2: StrategyAgent",
                False,
                f"StrategyAgent test failed: {e}"
            )

    def test_phase2_competitor_analyzer(self):
        """Test CompetitorAnalyzer instantiation"""
        try:
            from agents.strategy_agent import CompetitorAnalyzer

            analyzer = CompetitorAnalyzer(use_llm=False)

            # Check for actual method names
            assert hasattr(analyzer, 'analyze_competitive_landscape'), "Missing analyze_competitive_landscape method"
            assert hasattr(analyzer, 'ghosting_generator'), "Missing ghosting_generator attribute"

            self.record(
                "Phase 2: CompetitorAnalyzer",
                True,
                "CompetitorAnalyzer instantiates with competitive analysis methods"
            )
        except Exception as e:
            self.record(
                "Phase 2: CompetitorAnalyzer",
                False,
                f"CompetitorAnalyzer test failed: {e}"
            )

    def test_phase2_ghosting_generator(self):
        """Test GhostingLanguageGenerator instantiation"""
        try:
            from agents.strategy_agent import GhostingLanguageGenerator

            generator = GhostingLanguageGenerator(use_llm=False)

            # Check for actual method names
            assert hasattr(generator, 'generate_ghosting_library'), "Missing generate_ghosting_library method"
            assert hasattr(generator, 'generate_for_section'), "Missing generate_for_section method"
            assert hasattr(generator, 'GHOSTING_TEMPLATES'), "Missing GHOSTING_TEMPLATES"

            self.record(
                "Phase 2: GhostingLanguageGenerator",
                True,
                "GhostingLanguageGenerator instantiates with ghosting methods",
                {"templates": list(generator.GHOSTING_TEMPLATES.keys())}
            )
        except Exception as e:
            self.record(
                "Phase 2: GhostingLanguageGenerator",
                False,
                f"GhostingLanguageGenerator test failed: {e}"
            )

    # =========================================================================
    # Phase 3: Drafting Agent Tests
    # =========================================================================

    def test_phase3_imports(self):
        """Test that Drafting Agent components can be imported"""
        try:
            from agents.drafting_workflow import (
                DraftingState,
                LANGGRAPH_AVAILABLE,
            )
            # Module exports functions, not a class
            from agents.drafting_workflow import build_drafting_graph, run_drafting_workflow

            self.record(
                "Phase 3: Drafting Agent imports",
                True,
                f"Drafting workflow components available (LangGraph: {LANGGRAPH_AVAILABLE})"
            )
        except ImportError as e:
            self.record(
                "Phase 3: Drafting Agent imports",
                False,
                f"Import failed: {e}"
            )

    def test_phase3_workflow_nodes(self):
        """Test that workflow has F-B-P framework nodes as functions"""
        try:
            # Workflow uses module-level functions, not a class
            import agents.drafting_workflow as dw

            # Check for F-B-P framework node functions
            required_nodes = [
                'research_node',
                'structure_fbp_node',
                'draft_node',
                'quality_check_node',
            ]

            missing = []
            for node in required_nodes:
                if not hasattr(dw, node) or not callable(getattr(dw, node)):
                    missing.append(node)

            if missing:
                self.record(
                    "Phase 3: F-B-P Workflow Nodes",
                    False,
                    f"Missing node functions: {missing}"
                )
            else:
                self.record(
                    "Phase 3: F-B-P Workflow Nodes",
                    True,
                    "All F-B-P framework node functions present",
                    {"nodes": required_nodes}
                )
        except Exception as e:
            self.record(
                "Phase 3: F-B-P Workflow Nodes",
                False,
                f"Workflow nodes test failed: {e}"
            )

    def test_phase3_drafting_state(self):
        """Test DraftingState schema"""
        try:
            from agents.drafting_workflow import DraftingState

            # Check state has actual required fields (TypedDict)
            required_fields = ['requirement_id', 'requirement_text', 'draft_text', 'quality_scores']

            # DraftingState is a TypedDict
            if hasattr(DraftingState, '__annotations__'):
                annotations = DraftingState.__annotations__
                missing = [f for f in required_fields if f not in annotations]
                if missing:
                    self.record(
                        "Phase 3: DraftingState schema",
                        False,
                        f"Missing fields: {missing}",
                        {"found_fields": list(annotations.keys())}
                    )
                else:
                    self.record(
                        "Phase 3: DraftingState schema",
                        True,
                        "DraftingState TypedDict has required fields",
                        {"sample_fields": list(annotations.keys())[:5]}
                    )
            else:
                self.record(
                    "Phase 3: DraftingState schema",
                    True,
                    "DraftingState exists (structure not validated)"
                )
        except Exception as e:
            self.record(
                "Phase 3: DraftingState schema",
                False,
                f"DraftingState test failed: {e}"
            )

    # =========================================================================
    # Phase 4: Persistence Tests
    # =========================================================================

    def test_phase4_database_imports(self):
        """Test database module imports"""
        try:
            from api.database import (
                init_db,
                get_db_session,
                RFPModel,
                RequirementModel,
                is_db_available,
            )
            self.record(
                "Phase 4: Database imports",
                True,
                "Database module components available"
            )
        except ImportError as e:
            if "fastapi" in str(e).lower():
                self.record(
                    "Phase 4: Database imports",
                    True,
                    "FastAPI not installed (required for full API stack)",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 4: Database imports",
                    False,
                    f"Import failed: {e}"
                )

    def test_phase4_vector_store_imports(self):
        """Test vector store module imports"""
        try:
            from api.vector_store import (
                VectorStore,
                EmbeddingGenerator,
                SQLALCHEMY_AVAILABLE,
            )
            self.record(
                "Phase 4: Vector Store imports",
                True,
                f"Vector store available (SQLAlchemy: {SQLALCHEMY_AVAILABLE})"
            )
        except ImportError as e:
            if "fastapi" in str(e).lower():
                self.record(
                    "Phase 4: Vector Store imports",
                    True,
                    "FastAPI not installed (required for full API stack)",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 4: Vector Store imports",
                    False,
                    f"Import failed: {e}"
                )

    def test_phase4_embedding_generator(self):
        """Test EmbeddingGenerator with fallback"""
        try:
            from api.vector_store import EmbeddingGenerator

            generator = EmbeddingGenerator()

            # Test simple embedding fallback
            embedding = generator._simple_embedding("test query for cloud migration")

            assert len(embedding) == 1536, f"Expected 1536 dimensions, got {len(embedding)}"
            assert all(isinstance(x, float) for x in embedding), "Embedding should be floats"

            # Check it's normalized (magnitude ~1)
            import math
            magnitude = math.sqrt(sum(x * x for x in embedding))
            assert 0.99 < magnitude < 1.01, f"Embedding should be normalized, got magnitude {magnitude}"

            self.record(
                "Phase 4: EmbeddingGenerator",
                True,
                f"Embedding generator works (provider: {generator.provider})",
                {"provider": generator.provider, "dimensions": len(embedding)}
            )
        except ImportError as e:
            if "fastapi" in str(e).lower():
                self.record(
                    "Phase 4: EmbeddingGenerator",
                    True,
                    "FastAPI not installed (required for full API stack)",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 4: EmbeddingGenerator",
                    False,
                    f"EmbeddingGenerator test failed: {e}"
                )
        except Exception as e:
            self.record(
                "Phase 4: EmbeddingGenerator",
                False,
                f"EmbeddingGenerator test failed: {e}"
            )

    def test_phase4_vector_store_init(self):
        """Test VectorStore can be instantiated"""
        try:
            from api.vector_store import VectorStore, SQLALCHEMY_AVAILABLE

            if not SQLALCHEMY_AVAILABLE:
                self.record(
                    "Phase 4: VectorStore initialization",
                    True,
                    "SQLAlchemy not available (optional dependency)",
                    skipped=True
                )
                return

            store = VectorStore()

            # Check store has required search methods
            required_methods = [
                'search_capabilities',
                'search_past_performances',
                'search_key_personnel',
                'search_differentiators',
            ]

            missing = [m for m in required_methods if not hasattr(store, m)]
            if missing:
                self.record(
                    "Phase 4: VectorStore initialization",
                    False,
                    f"Missing methods: {missing}"
                )
            else:
                self.record(
                    "Phase 4: VectorStore initialization",
                    True,
                    "VectorStore has all search methods",
                    {"methods": required_methods}
                )
        except ImportError as e:
            if "fastapi" in str(e).lower():
                self.record(
                    "Phase 4: VectorStore initialization",
                    True,
                    "FastAPI not installed (required for full API stack)",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 4: VectorStore initialization",
                    False,
                    f"VectorStore init failed: {e}"
                )
        except Exception as e:
            self.record(
                "Phase 4: VectorStore initialization",
                False,
                f"VectorStore init failed: {e}"
            )

    # =========================================================================
    # Phase 5: Team Workspaces Tests (v4.1)
    # =========================================================================

    def test_phase5_team_models_imports(self):
        """Test that Team Workspace models can be imported"""
        try:
            from api.database import (
                UserModel,
                TeamModel,
                TeamMembershipModel,
                ActivityLogModel,
                UserRole,
            )
            self.record(
                "Phase 5: Team model imports",
                True,
                "All Team Workspace models available"
            )
        except ImportError as e:
            if "fastapi" in str(e).lower() or "sqlalchemy" in str(e).lower():
                self.record(
                    "Phase 5: Team model imports",
                    True,
                    "Database dependencies not installed",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 5: Team model imports",
                    False,
                    f"Import failed: {e}"
                )

    def test_phase5_user_role_enum(self):
        """Test UserRole enum has correct values"""
        try:
            from api.database import UserRole

            # Check all required roles exist
            expected_roles = ['admin', 'contributor', 'viewer']
            actual_roles = [role.value for role in UserRole]

            missing = [r for r in expected_roles if r not in actual_roles]
            if missing:
                self.record(
                    "Phase 5: UserRole enum",
                    False,
                    f"Missing roles: {missing}"
                )
            else:
                self.record(
                    "Phase 5: UserRole enum",
                    True,
                    "All RBAC roles defined (admin, contributor, viewer)",
                    {"roles": actual_roles}
                )
        except ImportError as e:
            if "fastapi" in str(e).lower() or "sqlalchemy" in str(e).lower():
                self.record(
                    "Phase 5: UserRole enum",
                    True,
                    "Database dependencies not installed",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 5: UserRole enum",
                    False,
                    f"UserRole test failed: {e}"
                )

    def test_phase5_team_model_structure(self):
        """Test TeamModel has required fields"""
        try:
            from api.database import TeamModel

            # Check model has required columns
            required_fields = ['id', 'name', 'slug', 'description', 'settings', 'created_by']

            # Get column names from model
            if hasattr(TeamModel, '__table__'):
                columns = [c.name for c in TeamModel.__table__.columns]
                missing = [f for f in required_fields if f not in columns]

                if missing:
                    self.record(
                        "Phase 5: TeamModel structure",
                        False,
                        f"Missing columns: {missing}"
                    )
                else:
                    self.record(
                        "Phase 5: TeamModel structure",
                        True,
                        "TeamModel has all required fields",
                        {"columns": columns[:6]}
                    )
            else:
                self.record(
                    "Phase 5: TeamModel structure",
                    True,
                    "TeamModel exists (structure not validated)"
                )
        except ImportError as e:
            if "fastapi" in str(e).lower() or "sqlalchemy" in str(e).lower():
                self.record(
                    "Phase 5: TeamModel structure",
                    True,
                    "Database dependencies not installed",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 5: TeamModel structure",
                    False,
                    f"TeamModel test failed: {e}"
                )

    def test_phase5_team_membership_model(self):
        """Test TeamMembershipModel for role-based access"""
        try:
            from api.database import TeamMembershipModel

            # Check for required columns
            required_fields = ['id', 'team_id', 'user_id', 'role']

            if hasattr(TeamMembershipModel, '__table__'):
                columns = [c.name for c in TeamMembershipModel.__table__.columns]
                missing = [f for f in required_fields if f not in columns]

                if missing:
                    self.record(
                        "Phase 5: TeamMembershipModel",
                        False,
                        f"Missing columns: {missing}"
                    )
                else:
                    self.record(
                        "Phase 5: TeamMembershipModel",
                        True,
                        "TeamMembershipModel supports RBAC",
                        {"columns": columns}
                    )
            else:
                self.record(
                    "Phase 5: TeamMembershipModel",
                    True,
                    "TeamMembershipModel exists"
                )
        except ImportError as e:
            if "fastapi" in str(e).lower() or "sqlalchemy" in str(e).lower():
                self.record(
                    "Phase 5: TeamMembershipModel",
                    True,
                    "Database dependencies not installed",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 5: TeamMembershipModel",
                    False,
                    f"TeamMembershipModel test failed: {e}"
                )

    def test_phase5_activity_log_model(self):
        """Test ActivityLogModel for audit trail"""
        try:
            from api.database import ActivityLogModel

            required_fields = ['id', 'team_id', 'user_id', 'action', 'resource_type']

            if hasattr(ActivityLogModel, '__table__'):
                columns = [c.name for c in ActivityLogModel.__table__.columns]
                missing = [f for f in required_fields if f not in columns]

                if missing:
                    self.record(
                        "Phase 5: ActivityLogModel",
                        False,
                        f"Missing columns: {missing}"
                    )
                else:
                    self.record(
                        "Phase 5: ActivityLogModel",
                        True,
                        "ActivityLogModel supports audit trail",
                        {"columns": columns[:6]}
                    )
            else:
                self.record(
                    "Phase 5: ActivityLogModel",
                    True,
                    "ActivityLogModel exists"
                )
        except ImportError as e:
            if "fastapi" in str(e).lower() or "sqlalchemy" in str(e).lower():
                self.record(
                    "Phase 5: ActivityLogModel",
                    True,
                    "Database dependencies not installed",
                    skipped=True
                )
            else:
                self.record(
                    "Phase 5: ActivityLogModel",
                    False,
                    f"ActivityLogModel test failed: {e}"
                )

    # =========================================================================
    # API Endpoint Tests
    # =========================================================================

    def test_api_main_imports(self):
        """Test that API main module can be imported"""
        try:
            from api.main import app, VECTOR_STORE_AVAILABLE
            self.record(
                "API: main.py imports",
                True,
                f"FastAPI app loaded (vector_store: {VECTOR_STORE_AVAILABLE})"
            )
        except ImportError as e:
            if "fastapi" in str(e).lower():
                self.record(
                    "API: main.py imports",
                    True,
                    "FastAPI not installed (required for API server)",
                    skipped=True
                )
            else:
                self.record(
                    "API: main.py imports",
                    False,
                    f"Import failed: {e}"
                )

    def test_api_endpoints_registered(self):
        """Test that v4.0 endpoints are registered"""
        try:
            from api.main import app

            routes = [r.path for r in app.routes]

            v4_endpoints = [
                "/api/health",
                "/api/rfp/{rfp_id}/requirements",
                "/api/library/vector-search",
                # v4.1 Team Workspace endpoints
                "/api/auth/register",
                "/api/auth/login",
                "/api/teams",
            ]

            found = []
            missing = []
            for endpoint in v4_endpoints:
                # Check if endpoint pattern exists
                if any(endpoint in r or endpoint.replace("{rfp_id}", "") in r for r in routes):
                    found.append(endpoint)
                else:
                    missing.append(endpoint)

            if missing:
                self.record(
                    "API: v4.0 endpoints registered",
                    False,
                    f"Missing endpoints: {missing}"
                )
            else:
                self.record(
                    "API: v4.0 endpoints registered",
                    True,
                    "All v4.0 endpoints found",
                    {"endpoints": found}
                )
        except ImportError as e:
            if "fastapi" in str(e).lower():
                self.record(
                    "API: v4.0 endpoints registered",
                    True,
                    "FastAPI not installed (required for API server)",
                    skipped=True
                )
            else:
                self.record(
                    "API: v4.0 endpoints registered",
                    False,
                    f"Endpoint check failed: {e}"
                )
        except Exception as e:
            self.record(
                "API: v4.0 endpoints registered",
                False,
                f"Endpoint check failed: {e}"
            )

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all(self):
        """Run all end-to-end tests"""
        print("\n" + "=" * 60)
        print("PropelAI v4.0 End-to-End Verification")
        print("=" * 60)

        # Phase 1: Trust Gate
        print("\n[Phase 1: Trust Gate - Source Traceability]")
        self.test_phase1_imports()
        self.test_phase1_bounding_box()
        self.test_phase1_source_coordinate()
        self.test_phase1_pdf_extractor()

        # Phase 2: Iron Triangle
        print("\n[Phase 2: Iron Triangle - Strategy Engine]")
        self.test_phase2_imports()
        self.test_phase2_strategy_agent()
        self.test_phase2_competitor_analyzer()
        self.test_phase2_ghosting_generator()

        # Phase 3: Drafting Agent
        print("\n[Phase 3: Drafting Agent - F-B-P Framework]")
        self.test_phase3_imports()
        self.test_phase3_workflow_nodes()
        self.test_phase3_drafting_state()

        # Phase 4: Persistence
        print("\n[Phase 4: Persistence - Database + Vector Store]")
        self.test_phase4_database_imports()
        self.test_phase4_vector_store_imports()
        self.test_phase4_embedding_generator()
        self.test_phase4_vector_store_init()

        # Phase 5: Team Workspaces (v4.1)
        print("\n[Phase 5: Team Workspaces - RBAC & Collaboration]")
        self.test_phase5_team_models_imports()
        self.test_phase5_user_role_enum()
        self.test_phase5_team_model_structure()
        self.test_phase5_team_membership_model()
        self.test_phase5_activity_log_model()

        # API Verification
        print("\n[API Endpoint Verification]")
        self.test_api_main_imports()
        self.test_api_endpoints_registered()

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        print(f"\nTotal: {total} tests")
        print(f"Passed: {passed} ({100*passed//total}%)")
        print(f"Skipped: {skipped} (environment dependencies)")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        if skipped > 0:
            print("\nSkipped tests (install dependencies to run):")
            for r in self.results:
                if r.skipped:
                    print(f"  - {r.name}")

        print("\n" + "=" * 60)

        # Success if no failures (skipped tests are OK)
        return failed == 0


if __name__ == "__main__":
    tests = V4EndToEndTests()
    success = tests.run_all()
    sys.exit(0 if success else 1)
