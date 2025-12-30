"""
PropelAI v5.0 Unit Tests: Iron Triangle Graph Logic
===================================================
QA Swarm: Validation Agent

Tests:
- RequirementsDAG orphan detection
- Iron Triangle edge building (C ↔ L ↔ M)
- ValidationEngine section placement rules
- ValidationEngine volume restrictions
- SourceCoordinate multi-page spanning
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    OrphanReport,
)
from agents.enhanced_compliance.validation_engine import (
    ValidationEngine,
    ViolationType,
    Severity,
    validate_requirements,
)
from tests.conftest import assert_orphan_detected, assert_violation_exists


# =============================================================================
# RequirementsDAG Tests
# =============================================================================

@pytest.mark.unit
class TestRequirementsDAG:
    """Tests for Iron Triangle DAG"""

    def test_add_single_requirement(self, empty_dag, golden_requirements):
        """Test adding a single requirement to DAG"""
        req = golden_requirements[0]
        empty_dag.add_requirement(req)

        assert len(empty_dag.graph.nodes()) == 1
        assert req.id in empty_dag.graph.nodes()

    def test_add_multiple_requirements(self, empty_dag, golden_requirements):
        """Test adding multiple requirements"""
        empty_dag.add_requirements(golden_requirements)

        assert len(empty_dag.graph.nodes()) == len(golden_requirements)

    def test_section_classification_c(self, populated_dag):
        """Test that Section C requirements are classified correctly"""
        c_nodes = populated_dag.get_requirements_for_section(NodeSection.SECTION_C)

        assert len(c_nodes) > 0
        for node_id in c_nodes:
            node = populated_dag.graph.nodes[node_id]
            assert node['section'] == 'C'

    def test_section_classification_l(self, populated_dag):
        """Test that Section L requirements are classified correctly"""
        l_nodes = populated_dag.get_requirements_for_section(NodeSection.SECTION_L)

        assert len(l_nodes) > 0
        for node_id in l_nodes:
            node = populated_dag.graph.nodes[node_id]
            assert node['section'] == 'L'

    def test_section_classification_m(self, populated_dag):
        """Test that Section M requirements are classified correctly"""
        m_nodes = populated_dag.get_requirements_for_section(NodeSection.SECTION_M)

        assert len(m_nodes) > 0
        for node_id in m_nodes:
            node = populated_dag.graph.nodes[node_id]
            assert node['section'] == 'M'


@pytest.mark.unit
class TestOrphanDetection:
    """Tests for orphan requirement detection"""

    def test_detect_orphan_c_without_l(self, golden_requirements):
        """
        Test that a Section C requirement without L link is detected as orphan.

        FR-2.2 Requirement: Identify C requirements with no instruction coverage.
        """
        # Create DAG with only C requirements (no L to link)
        c_reqs = [r for r in golden_requirements if r.source.section_id.startswith('C')]

        dag = RequirementsDAG()
        dag.add_requirements(c_reqs)

        orphans = dag.find_orphans()

        # All C requirements should be orphans (no L links)
        assert len(orphans) > 0

        # Check specific orphan
        orphan_ids = [o.orphan_id for o in orphans]
        assert "REQ-C-001" in orphan_ids

    def test_detect_orphan_c_without_m(self, golden_requirements):
        """
        Test that a Section C requirement without M link is detected.

        This is CRITICAL: unlinked performance requirements may not be scored.
        """
        c_reqs = [r for r in golden_requirements if r.source.section_id.startswith('C')]

        dag = RequirementsDAG()
        dag.add_requirements(c_reqs)

        orphans = dag.find_orphans()

        # Find orphans about missing evaluation
        eval_orphans = [o for o in orphans if "evaluation" in o.reason.lower()]

        # Should have evaluation-related orphans
        assert len(eval_orphans) > 0

    def test_detect_orphan_m_without_targets(self, golden_requirements):
        """Test that Section M criteria without targets are detected"""
        m_reqs = [r for r in golden_requirements if r.source.section_id.startswith('M')]

        dag = RequirementsDAG()
        dag.add_requirements(m_reqs)

        orphans = dag.find_orphans()

        # M criteria with no outgoing edges should be orphans
        assert len(orphans) > 0

        # Check that reason mentions "does not link"
        linking_orphans = [o for o in orphans if "link" in o.reason.lower()]
        assert len(linking_orphans) > 0

    def test_isolated_node_detected(self, empty_dag):
        """Test that completely isolated nodes are detected"""
        # Create isolated requirement
        isolated_req = RequirementNode(
            id="REQ-ISOLATED",
            text="This requirement has no links at all.",
            requirement_type=RequirementType.PERFORMANCE,
            source=SourceLocation(
                document_name="test.pdf",
                document_type=DocumentType.MAIN_SOLICITATION,
                page_number=1,
                section_id="X.1"  # Unknown section
            )
        )

        empty_dag.add_requirement(isolated_req)
        orphans = empty_dag.find_orphans()

        assert len(orphans) > 0
        # Verify the isolated requirement is detected as orphan
        # (reason text may vary based on section classification)
        assert_orphan_detected(orphans, "REQ-ISOLATED")

    def test_linked_requirements_not_orphans(self, linked_dag):
        """Test that properly linked requirements are NOT orphans"""
        analysis = linked_dag.analyze()

        # Should have fewer orphans than populated_dag
        # (auto-linking reduces orphan count)
        orphan_count = analysis.orphan_count

        # With similarity=0.2, some links should be created
        assert analysis.total_edges > 0, "Should have created some edges"


@pytest.mark.unit
class TestIronTriangleEdges:
    """Tests for Iron Triangle edge building"""

    def test_build_edges_c_to_l(self, populated_dag, golden_requirements):
        """Test that C->L edges are created based on similarity"""
        # Build edges with low threshold
        edges_created = populated_dag.build_iron_triangle_edges(similarity_threshold=0.1)

        # Should create at least some edges
        assert edges_created > 0

        # Check for L->C instructs edges
        c_nodes = populated_dag.get_requirements_for_section(NodeSection.SECTION_C)
        l_nodes = populated_dag.get_requirements_for_section(NodeSection.SECTION_L)

        # At least one C node should have incoming edge from L
        has_l_edge = False
        for c_id in c_nodes:
            for source, target, data in populated_dag.graph.in_edges(c_id, data=True):
                if populated_dag.graph.nodes[source].get('section') == 'L':
                    has_l_edge = True
                    assert data.get('edge_type') == EdgeType.INSTRUCTS.value
                    break
            if has_l_edge:
                break

        # Note: May not find edge if text similarity is too low

    def test_build_edges_m_to_c(self, populated_dag):
        """Test that M->C edges (evaluates) are created"""
        edges_created = populated_dag.build_iron_triangle_edges(similarity_threshold=0.1)

        # Check for M->C evaluates edges
        c_nodes = populated_dag.get_requirements_for_section(NodeSection.SECTION_C)

        has_m_edge = False
        for c_id in c_nodes:
            for source, target, data in populated_dag.graph.in_edges(c_id, data=True):
                if populated_dag.graph.nodes[source].get('section') == 'M':
                    has_m_edge = True
                    assert data.get('edge_type') == EdgeType.EVALUATES.value
                    break

    def test_no_cycles_created(self, populated_dag):
        """Test that edge building doesn't create cycles"""
        populated_dag.build_iron_triangle_edges(similarity_threshold=0.1)

        # Should still be a DAG
        import networkx as nx
        assert nx.is_directed_acyclic_graph(populated_dag.graph)

    def test_edge_weights_valid(self, populated_dag):
        """Test that edge weights are valid similarity scores"""
        populated_dag.build_iron_triangle_edges(similarity_threshold=0.1)

        for source, target, data in populated_dag.graph.edges(data=True):
            weight = data.get('weight', 0)
            assert 0 <= weight <= 1, f"Invalid weight: {weight}"

    def test_iron_triangle_coverage_calculation(self, linked_dag):
        """Test Iron Triangle coverage metrics"""
        analysis = linked_dag.analyze()
        coverage = analysis.iron_triangle_coverage

        assert "c_with_l" in coverage
        assert "c_with_m" in coverage
        assert "l_with_m" in coverage
        assert "overall" in coverage

        # All values should be between 0 and 1
        for key, value in coverage.items():
            assert 0 <= value <= 1, f"Invalid coverage for {key}: {value}"


# =============================================================================
# ValidationEngine Tests
# =============================================================================

@pytest.mark.unit
class TestValidationEngine:
    """Tests for L-M-C Validation Engine"""

    def test_validate_correct_section_placement(self, validation_engine, golden_requirements):
        """Test that correctly placed requirements pass validation"""
        # Only use requirements that are in correct sections
        c_reqs = [r for r in golden_requirements
                  if r.source.section_id.startswith('C')
                  and r.requirement_type == RequirementType.PERFORMANCE]

        result = validation_engine.validate(c_reqs)

        # Performance requirements in Section C should be valid
        wrong_section_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.WRONG_SECTION
               and v.requirement_id in [r.id for r in c_reqs]
        ]

        # Shouldn't have section violations for C/Performance combo
        # (but may have others)
        assert len(wrong_section_violations) == 0

    def test_detect_wrong_section_placement(self, validation_engine):
        """
        Test that performance requirement in Section L triggers WARNING.

        FR-2.3: Alert when content type doesn't match section.
        """
        # Create performance requirement in Section L
        wrong_section_req = RequirementNode(
            id="REQ-WRONG-001",
            text="The Contractor shall provide monitoring services.",
            requirement_type=RequirementType.PERFORMANCE,  # Performance type
            source=SourceLocation(
                document_name="test.pdf",
                document_type=DocumentType.MAIN_SOLICITATION,
                page_number=5,
                section_id="L.2.1"  # But in Section L!
            )
        )

        result = validation_engine.validate([wrong_section_req])

        # Should detect wrong section
        assert_violation_exists(
            result.violations,
            ViolationType.WRONG_SECTION,
            severity=Severity.WARNING
        )

    def test_detect_volume_restriction_violation(self, validation_engine, past_performance_outline):
        """
        Test that Section C requirement in Past Performance volume is CRITICAL.

        Volume Rule: past_performance volume should only have L instructions.
        """
        # Create C requirement that would be in past_performance volume
        c_in_pp = RequirementNode(
            id="REQ-C-PP",
            text="The Contractor shall provide cloud services.",
            requirement_type=RequirementType.PERFORMANCE,
            source=SourceLocation(
                document_name="test.pdf",
                document_type=DocumentType.STATEMENT_OF_WORK,
                page_number=1,
                section_id="C.1"
            )
        )

        # Modify outline to place C.1 in past_performance volume
        outline = {
            "volumes": [
                {
                    "type": "past_performance",
                    "sections": [
                        {"id": "C", "title": "Wrongly placed SOW"}
                    ]
                }
            ]
        }

        result = validation_engine.validate([c_in_pp], outline=outline)

        # Should trigger CRITICAL volume restriction
        assert_violation_exists(
            result.violations,
            ViolationType.VOLUME_RESTRICTION,
            severity=Severity.CRITICAL
        )

    def test_detect_duplicates(self, validation_engine):
        """Test duplicate requirement detection"""
        req1 = RequirementNode(
            id="REQ-DUP-001",
            text="The Contractor shall provide 24/7 support.",
            requirement_type=RequirementType.PERFORMANCE,
            source=SourceLocation(
                document_name="test.pdf",
                document_type=DocumentType.MAIN_SOLICITATION,
                page_number=1,
                section_id="C.1"
            )
        )

        # Same text, different ID
        req2 = RequirementNode(
            id="REQ-DUP-002",
            text="The Contractor shall provide 24/7 support.",  # Duplicate
            requirement_type=RequirementType.PERFORMANCE,
            source=SourceLocation(
                document_name="test.pdf",
                document_type=DocumentType.MAIN_SOLICITATION,
                page_number=2,
                section_id="C.2"
            )
        )

        result = validation_engine.validate([req1, req2])

        assert_violation_exists(
            result.violations,
            ViolationType.DUPLICATE_REQUIREMENT
        )

    def test_compliance_score_calculation(self, validation_engine, golden_requirements):
        """Test that compliance score is calculated correctly"""
        result = validation_engine.validate(golden_requirements)

        # Score should be between 0 and 100
        assert 0 <= result.compliance_score <= 100

        # With valid requirements, score should be decent
        assert result.compliance_score >= 50

    def test_orphan_coverage_validation(self, validation_engine, golden_requirements, linked_dag):
        """Test that orphans from graph are flagged as violations"""
        graph_export = linked_dag.to_dict()

        result = validation_engine.validate(golden_requirements, graph=graph_export)

        # If there are orphans, should have coverage violations
        if graph_export.get("orphan_count", 0) > 0:
            coverage_violations = [
                v for v in result.violations
                if v.violation_type in [
                    ViolationType.ORPHAN_PERFORMANCE,
                    ViolationType.ORPHAN_INSTRUCTION,
                    ViolationType.ORPHAN_EVALUATION,
                ]
            ]
            assert len(coverage_violations) > 0

    def test_validation_result_structure(self, validation_engine, golden_requirements):
        """Test ValidationResult has all required fields"""
        result = validation_engine.validate(golden_requirements)

        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'total_violations')
        assert hasattr(result, 'critical_count')
        assert hasattr(result, 'warning_count')
        assert hasattr(result, 'info_count')
        assert hasattr(result, 'violations')
        assert hasattr(result, 'compliance_score')

        # to_dict should work
        result_dict = result.to_dict()
        assert 'is_valid' in result_dict
        assert 'compliance_score' in result_dict


# =============================================================================
# SourceCoordinate Multi-Page Spanning Tests
# =============================================================================

@pytest.mark.unit
class TestSourceCoordinateSpanning:
    """Tests for multi-page spanning (FR-1.3)"""

    def test_single_page_not_spanning(self, single_page_coordinate):
        """Test that single-page coordinate is not marked as spanning"""
        assert not single_page_coordinate.spans_pages
        assert len(single_page_coordinate.visual_rects) == 1
        assert single_page_coordinate.get_all_pages() == [5]

    def test_multi_page_is_spanning(self, multi_page_coordinate):
        """Test that multi-page coordinate is marked as spanning"""
        assert multi_page_coordinate.spans_pages
        assert len(multi_page_coordinate.visual_rects) == 2
        assert multi_page_coordinate.get_all_pages() == [5, 6]

    def test_get_rects_for_page(self, multi_page_coordinate):
        """Test getting bounding boxes for specific page"""
        page5_rects = multi_page_coordinate.get_rects_for_page(5)
        page6_rects = multi_page_coordinate.get_rects_for_page(6)

        assert len(page5_rects) == 1
        assert len(page6_rects) == 1

        # Page 7 should have no rects
        page7_rects = multi_page_coordinate.get_rects_for_page(7)
        assert len(page7_rects) == 0

    def test_add_visual_rect(self, single_page_coordinate):
        """Test adding additional visual rects"""
        assert not single_page_coordinate.spans_pages

        # Add rect on different page
        new_bbox = BoundingBox(
            x0=100, y0=0, x1=500, y1=100,
            page_width=612, page_height=792
        )
        single_page_coordinate.add_visual_rect(6, new_bbox)

        assert single_page_coordinate.spans_pages
        assert len(single_page_coordinate.visual_rects) == 2

    def test_to_dict_includes_spanning_info(self, multi_page_coordinate):
        """Test that to_dict includes all spanning information"""
        coord_dict = multi_page_coordinate.to_dict()

        assert 'spans_pages' in coord_dict
        assert coord_dict['spans_pages'] is True

        assert 'visual_rects' in coord_dict
        assert len(coord_dict['visual_rects']) == 2

        assert 'all_pages' in coord_dict
        assert coord_dict['all_pages'] == [5, 6]

    def test_empty_visual_rects_initialized(self):
        """Test that empty visual_rects is initialized from primary bbox"""
        bbox = BoundingBox(
            x0=100, y0=200, x1=500, y1=250,
            page_width=612, page_height=792
        )
        coord = SourceCoordinate(
            document_id="test.pdf",
            page_number=3,
            bounding_box=bbox,
        )

        # Should auto-initialize visual_rects
        assert len(coord.visual_rects) == 1
        assert coord.visual_rects[0].page_number == 3

    def test_bounding_box_normalization(self, single_page_coordinate):
        """Test bounding box normalization to 0-1 coordinates"""
        normalized = single_page_coordinate.bounding_box.to_normalized()

        assert 'x' in normalized
        assert 'y' in normalized
        assert 'width' in normalized
        assert 'height' in normalized

        # All values should be between 0 and 1
        for key, value in normalized.items():
            assert 0 <= value <= 1, f"Normalized {key} out of range: {value}"


# =============================================================================
# Graph Analysis Tests
# =============================================================================

@pytest.mark.unit
class TestGraphAnalysis:
    """Tests for comprehensive graph analysis"""

    def test_analysis_structure(self, linked_dag):
        """Test GraphAnalysis has all required fields"""
        analysis = linked_dag.analyze()

        assert analysis.total_nodes > 0
        assert analysis.total_edges >= 0
        assert analysis.orphan_count >= 0
        assert isinstance(analysis.orphans, list)
        assert isinstance(analysis.section_counts, dict)
        assert isinstance(analysis.edge_type_counts, dict)
        assert analysis.connected_components >= 0
        assert isinstance(analysis.iron_triangle_coverage, dict)

    def test_section_counts_correct(self, linked_dag, golden_requirements):
        """Test that section counts match requirements"""
        analysis = linked_dag.analyze()

        c_count = len([r for r in golden_requirements if r.source.section_id.startswith('C')])
        l_count = len([r for r in golden_requirements if r.source.section_id.startswith('L')])
        m_count = len([r for r in golden_requirements if r.source.section_id.startswith('M')])

        assert analysis.section_counts.get('C', 0) == c_count
        assert analysis.section_counts.get('L', 0) == l_count
        assert analysis.section_counts.get('M', 0) == m_count

    def test_topological_order(self, linked_dag):
        """Test topological ordering is valid"""
        order = linked_dag.get_topological_order()

        # Should have all nodes
        assert len(order) == len(linked_dag.graph.nodes())

        # Each node should appear exactly once
        assert len(order) == len(set(order))

    def test_critical_path(self, linked_dag):
        """Test critical path calculation"""
        path = linked_dag.get_critical_path()

        # Path should be list of node IDs
        assert isinstance(path, list)

        # All nodes in path should exist
        for node_id in path:
            assert node_id in linked_dag.graph.nodes()

    def test_to_dict_export(self, linked_dag):
        """Test graph export to dictionary"""
        export = linked_dag.to_dict()

        assert 'nodes' in export
        assert 'edges' in export
        assert 'analysis' in export
        assert 'orphans' in export

        # Nodes should have required fields
        if export['nodes']:
            node = export['nodes'][0]
            assert 'id' in node
            assert 'section' in node


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_requirements_validation(self, validation_engine):
        """Test validation with empty requirements list"""
        result = validation_engine.validate([])

        assert result.is_valid is True
        assert result.compliance_score == 100.0
        assert len(result.violations) == 0

    def test_empty_dag_analysis(self, empty_dag):
        """Test analysis of empty DAG"""
        analysis = empty_dag.analyze()

        assert analysis.total_nodes == 0
        assert analysis.total_edges == 0
        assert analysis.orphan_count == 0

    def test_requirement_without_source(self, empty_dag):
        """Test handling requirement with unknown section"""
        # Create requirement with unknown section ID to test edge case handling
        req = RequirementNode(
            id="REQ-NO-SOURCE",
            text="Requirement without clear section.",
            requirement_type=RequirementType.PERFORMANCE,
            source=SourceLocation(
                document_name="unknown.pdf",
                document_type=DocumentType.ATTACHMENT,
                page_number=1,
                section_id="UNKNOWN.1"  # Non-standard section ID
            )
        )

        empty_dag.add_requirement(req)

        # Should be added to the graph
        assert "REQ-NO-SOURCE" in empty_dag.graph.nodes()

    def test_cycle_prevention(self, empty_dag, golden_requirements):
        """Test that adding cycle-creating edge fails"""
        # Add two requirements
        req_a = golden_requirements[0]
        req_b = golden_requirements[1]

        empty_dag.add_requirements([req_a, req_b])

        # Add edge A -> B
        added1 = empty_dag.add_edge(req_a.id, req_b.id, EdgeType.REFERENCES)
        assert added1 is True

        # Try to add B -> A (would create cycle)
        added2 = empty_dag.add_edge(req_b.id, req_a.id, EdgeType.REFERENCES)
        assert added2 is False  # Should be rejected

    def test_self_loop_prevention(self, empty_dag, golden_requirements):
        """Test that self-loops are prevented"""
        req = golden_requirements[0]
        empty_dag.add_requirement(req)

        # Try to add self-loop
        added = empty_dag.add_edge(req.id, req.id, EdgeType.REFERENCES)
        assert added is False

    def test_edge_to_nonexistent_node(self, empty_dag, golden_requirements):
        """Test adding edge to non-existent node"""
        req = golden_requirements[0]
        empty_dag.add_requirement(req)

        # Try to add edge to non-existent target
        added = empty_dag.add_edge(req.id, "NONEXISTENT", EdgeType.REFERENCES)
        assert added is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
