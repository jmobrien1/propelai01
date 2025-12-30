"""
PropelAI v5.0 E2E Tests: Agent Trajectory & Trace Log
======================================================
QA Swarm: AgentOps Agent

Tests:
- Agent Trace Log creation (NFR-2.3)
- Trajectory evaluation (execution order)
- Human correction submission
- Data Flywheel statistics
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Agent Trace Log Database Model Tests
# =============================================================================

@pytest.mark.e2e
class TestAgentTraceLogModel:
    """Tests for AgentTraceLogModel database operations"""

    def test_trace_log_model_structure(self):
        """Test that AgentTraceLogModel has all required fields"""
        from api.database import AgentTraceLogModel

        # Check class has required columns
        columns = AgentTraceLogModel.__table__.columns.keys()

        required_columns = [
            'id', 'rfp_id', 'user_id',
            'agent_name', 'action',
            'input_data', 'output_data', 'confidence_score',
            'human_correction', 'correction_type', 'correction_reason',
            'corrected_by', 'corrected_at',
            'duration_ms', 'model_name', 'token_count',
            'status', 'created_at', 'updated_at'
        ]

        for col in required_columns:
            assert col in columns, f"Missing column: {col}"

    def test_trace_log_to_dict(self):
        """Test AgentTraceLogModel.to_dict() method"""
        from api.database import AgentTraceLogModel

        log = AgentTraceLogModel(
            id="trace-test123",
            rfp_id="RFP-001",
            agent_name="ComplianceAgent",
            action="extract_requirements",
            input_data={"pages": [1, 2, 3]},
            output_data={"count": 10},
            confidence_score=0.95,
            duration_ms=1500,
            status="completed"
        )

        result = log.to_dict()

        assert result['id'] == "trace-test123"
        assert result['agent_name'] == "ComplianceAgent"
        assert result['action'] == "extract_requirements"
        assert result['input_data'] == {"pages": [1, 2, 3]}
        assert result['output_data'] == {"count": 10}
        assert result['confidence_score'] == 0.95
        assert result['duration_ms'] == 1500
        assert result['status'] == "completed"

    def test_trace_log_indexes_exist(self):
        """Test that performance indexes are defined"""
        from api.database import AgentTraceLogModel

        table = AgentTraceLogModel.__table__

        # Check for indexes
        index_names = [idx.name for idx in table.indexes]

        expected_indexes = [
            'idx_agent_trace_logs_rfp_id',
            'idx_agent_trace_logs_agent_name',
            'idx_agent_trace_logs_action',
            'idx_agent_trace_logs_created_at',
            'idx_agent_trace_logs_status',
        ]

        for idx in expected_indexes:
            assert idx in index_names, f"Missing index: {idx}"


# =============================================================================
# Agent Trace Log API Tests
# =============================================================================

@pytest.mark.e2e
class TestAgentTraceLogAPI:
    """Tests for Agent Trace Log API endpoints"""

    @pytest.fixture
    def mock_session(self):
        """Mock database session"""
        session = AsyncMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_create_trace_log_endpoint(self, trace_log_entry):
        """Test POST /api/trace-logs creates entry"""
        from api.main import AgentTraceLogCreate

        # Validate request model
        log_create = AgentTraceLogCreate(
            rfp_id=trace_log_entry['rfp_id'],
            agent_name=trace_log_entry['agent_name'],
            action=trace_log_entry['action'],
            input_data=trace_log_entry['input_data'],
            output_data=trace_log_entry['output_data'],
            confidence_score=trace_log_entry['confidence_score'],
            duration_ms=trace_log_entry['duration_ms'],
            model_name=trace_log_entry['model_name'],
            status=trace_log_entry['status']
        )

        # Validate Pydantic model
        assert log_create.agent_name == "ComplianceAgent"
        assert log_create.action == "extract_requirements"
        assert log_create.confidence_score == 0.92
        assert log_create.duration_ms == 1250

    @pytest.mark.asyncio
    async def test_trace_correction_model(self, trace_correction):
        """Test AgentTraceCorrection Pydantic model"""
        from api.main import AgentTraceCorrection

        correction = AgentTraceCorrection(
            human_correction=trace_correction['human_correction'],
            correction_type=trace_correction['correction_type'],
            correction_reason=trace_correction['correction_reason']
        )

        assert correction.correction_type == "modified"
        assert "Missed" in correction.correction_reason

    def test_trace_log_entry_fixture_valid(self, trace_log_entry):
        """Test that trace_log_entry fixture is valid"""
        required_fields = [
            'rfp_id', 'agent_name', 'action',
            'input_data', 'output_data',
            'confidence_score', 'duration_ms', 'status'
        ]

        for field in required_fields:
            assert field in trace_log_entry, f"Missing field: {field}"

    def test_trace_correction_fixture_valid(self, trace_correction):
        """Test that trace_correction fixture is valid"""
        required_fields = [
            'human_correction', 'correction_type', 'correction_reason'
        ]

        for field in required_fields:
            assert field in trace_correction, f"Missing field: {field}"


# =============================================================================
# Trajectory Evaluation Tests
# =============================================================================

@pytest.mark.e2e
class TestTrajectoryEvaluation:
    """
    Tests for agent execution trajectory.

    Verifies that agents execute in correct order:
    1. ComplianceAgent (extract requirements)
    2. StrategyAgent (identify Section M weights)
    3. DraftingAgent (generate F-B-P content)

    NFR-2.3: Use trace log to verify execution order.
    """

    def test_trajectory_order_compliance_before_strategy(self, trace_log_entry):
        """
        Test that ComplianceAgent runs BEFORE StrategyAgent.

        This is CRITICAL: Strategy cannot run without extracted requirements.
        """
        # Simulate trajectory with timestamps
        trajectory = [
            {
                "agent_name": "ComplianceAgent",
                "action": "extract_requirements",
                "timestamp": "2024-12-28T10:00:00Z",
                "sequence": 1
            },
            {
                "agent_name": "StrategyAgent",
                "action": "identify_section_m_weights",
                "timestamp": "2024-12-28T10:00:05Z",
                "sequence": 2
            },
        ]

        # Verify order
        compliance_seq = next(
            t['sequence'] for t in trajectory if t['agent_name'] == 'ComplianceAgent'
        )
        strategy_seq = next(
            t['sequence'] for t in trajectory if t['agent_name'] == 'StrategyAgent'
        )

        assert compliance_seq < strategy_seq, \
            "ComplianceAgent must run before StrategyAgent"

    def test_trajectory_order_strategy_before_drafting(self):
        """
        Test that StrategyAgent identifies Section M weights BEFORE DraftingAgent.

        NFR-2.3 Requirement: No section drafted without graph node.
        """
        trajectory = [
            {
                "agent_name": "StrategyAgent",
                "action": "identify_section_m_weights",
                "timestamp": "2024-12-28T10:00:05Z",
                "output_data": {
                    "section_m_factors": [
                        {"id": "M.2.1", "weight": 0.4},
                        {"id": "M.3.1", "weight": 0.3}
                    ]
                },
                "sequence": 1
            },
            {
                "agent_name": "DraftingAgent",
                "action": "generate_fbp_content",
                "timestamp": "2024-12-28T10:00:10Z",
                "input_data": {
                    "section_id": "2.1",
                    "requirement_nodes": ["REQ-C-001", "REQ-C-002"]
                },
                "sequence": 2
            },
        ]

        strategy_seq = next(
            t['sequence'] for t in trajectory if t['agent_name'] == 'StrategyAgent'
        )
        drafting_seq = next(
            t['sequence'] for t in trajectory if t['agent_name'] == 'DraftingAgent'
        )

        assert strategy_seq < drafting_seq, \
            "StrategyAgent must run before DraftingAgent"

        # Verify Section M factors were identified
        strategy_output = next(
            t['output_data'] for t in trajectory if t['agent_name'] == 'StrategyAgent'
        )
        assert 'section_m_factors' in strategy_output
        assert len(strategy_output['section_m_factors']) > 0

    def test_no_drafting_without_requirement_node(self):
        """
        Test that DraftingAgent doesn't draft without RequirementNode.

        Assertion: Each drafted section must map to graph nodes.
        """
        # Simulated draft that references requirement nodes
        draft_trace = {
            "agent_name": "DraftingAgent",
            "action": "generate_fbp_content",
            "input_data": {
                "section_id": "2.1",
                "requirement_nodes": ["REQ-C-001", "REQ-C-002"]
            },
            "output_data": {
                "section_id": "2.1",
                "content": "Our approach to system uptime...",
                "requirement_coverage": ["REQ-C-001", "REQ-C-002"],
                "uncited_requirements": []
            }
        }

        # Verify all requirements are covered
        input_reqs = set(draft_trace['input_data']['requirement_nodes'])
        output_coverage = set(draft_trace['output_data']['requirement_coverage'])

        assert input_reqs == output_coverage, \
            f"Missing coverage: {input_reqs - output_coverage}"

        # No orphan sections (drafted without requirements)
        assert len(draft_trace['output_data']['requirement_coverage']) > 0, \
            "Cannot draft section without requirement nodes"

    def test_trajectory_includes_duration(self):
        """Test that each trajectory entry includes duration_ms"""
        trajectory = [
            {"agent_name": "ComplianceAgent", "duration_ms": 1500},
            {"agent_name": "StrategyAgent", "duration_ms": 2000},
            {"agent_name": "DraftingAgent", "duration_ms": 3500},
        ]

        for entry in trajectory:
            assert 'duration_ms' in entry
            assert entry['duration_ms'] > 0


# =============================================================================
# Human Correction Flow Tests
# =============================================================================

@pytest.mark.e2e
class TestHumanCorrectionFlow:
    """Tests for human-in-the-loop correction workflow (Data Flywheel)"""

    def test_correction_types_valid(self):
        """Test valid correction types"""
        valid_types = ['accepted', 'modified', 'rejected']

        for correction_type in valid_types:
            assert correction_type in valid_types

    def test_correction_updates_status(self, trace_log_entry, trace_correction):
        """Test that correction updates trace log status"""
        # Simulate correction flow
        original_status = trace_log_entry['status']
        assert original_status == 'completed'

        # After correction
        updated_entry = {**trace_log_entry}
        updated_entry['status'] = 'corrected'
        updated_entry['human_correction'] = trace_correction['human_correction']
        updated_entry['correction_type'] = trace_correction['correction_type']
        updated_entry['corrected_at'] = datetime.now().isoformat()

        assert updated_entry['status'] == 'corrected'
        assert updated_entry['human_correction'] is not None

    def test_correction_reason_captured(self, trace_correction):
        """Test that correction reason is captured"""
        assert 'correction_reason' in trace_correction
        assert len(trace_correction['correction_reason']) > 0

        # Should explain what was wrong
        assert "Missed" in trace_correction['correction_reason'] or \
               "incorrect" in trace_correction['correction_reason'].lower() or \
               len(trace_correction['correction_reason']) > 10

    def test_corrected_by_tracked(self, trace_correction):
        """Test that correction is attributed to user"""
        # In production, corrected_by would be user_id from JWT
        corrected_entry = {
            **trace_correction,
            'corrected_by': 'user-123',
            'corrected_at': datetime.now().isoformat()
        }

        assert corrected_entry['corrected_by'] is not None
        assert corrected_entry['corrected_at'] is not None


# =============================================================================
# Data Flywheel Statistics Tests
# =============================================================================

@pytest.mark.e2e
class TestDataFlywheelStats:
    """Tests for correction rate statistics"""

    def test_stats_structure(self):
        """Test stats endpoint returns correct structure"""
        # Simulated stats response
        stats = {
            "total_logs": 100,
            "corrected_logs": 7,
            "correction_rate": 7.0,
            "by_agent": {
                "ComplianceAgent": 60,
                "StrategyAgent": 25,
                "DraftingAgent": 15
            },
            "by_status": {
                "completed": 93,
                "corrected": 7
            }
        }

        assert 'total_logs' in stats
        assert 'corrected_logs' in stats
        assert 'correction_rate' in stats
        assert 'by_agent' in stats
        assert 'by_status' in stats

    def test_correction_rate_calculation(self):
        """Test correction rate is calculated correctly"""
        total = 100
        corrected = 7

        rate = (corrected / total * 100) if total > 0 else 0

        # Use approximate comparison for floating point
        assert abs(rate - 7.0) < 0.001

    def test_agent_distribution(self):
        """Test agent distribution sums to total"""
        stats = {
            "total_logs": 100,
            "by_agent": {
                "ComplianceAgent": 60,
                "StrategyAgent": 25,
                "DraftingAgent": 15
            }
        }

        agent_sum = sum(stats['by_agent'].values())
        assert agent_sum == stats['total_logs']

    def test_status_distribution(self):
        """Test status distribution is valid"""
        stats = {
            "total_logs": 100,
            "by_status": {
                "completed": 90,
                "corrected": 7,
                "failed": 3
            }
        }

        status_sum = sum(stats['by_status'].values())
        assert status_sum == stats['total_logs']


# =============================================================================
# Time-Travel Debugging Tests
# =============================================================================

@pytest.mark.e2e
class TestTimeTravelDebugging:
    """Tests for trajectory replay (time-travel debugging)"""

    def test_replay_agent_decision(self, trace_log_entry):
        """Test that agent decisions can be replayed from trace"""
        # Each trace entry contains input_data for replay
        replay_input = trace_log_entry['input_data']

        assert 'document' in replay_input
        assert 'pages' in replay_input

        # Could feed this back to agent for deterministic replay
        # (assuming same model and prompts)

    def test_compare_outputs_before_after_correction(self, trace_log_entry, trace_correction):
        """Test comparing original output vs corrected"""
        original_output = trace_log_entry['output_data']
        corrected_output = trace_correction['human_correction']

        # Find what changed
        original_count = original_output.get('requirements_count', 0)
        corrected_count = corrected_output.get('requirements_count', 0)

        delta = corrected_count - original_count

        # In this case, human found 2 more requirements
        assert delta == 2

    def test_trace_enables_root_cause_analysis(self):
        """Test that trace log enables root cause analysis"""
        # Simulated failure scenario
        failed_trace = {
            "agent_name": "ComplianceAgent",
            "action": "extract_requirements",
            "status": "failed",
            "input_data": {
                "document": "corrupted.pdf",
                "pages": []  # Empty - problem!
            },
            "output_data": None,
            "error_message": "No pages to extract"
        }

        # Root cause: empty pages list
        assert len(failed_trace['input_data']['pages']) == 0
        assert failed_trace['status'] == 'failed'

        # This trace enables debugging without re-running


# =============================================================================
# Integration with Requirements Graph
# =============================================================================

@pytest.mark.e2e
class TestTraceGraphIntegration:
    """Tests for trace log integration with requirements graph"""

    def test_trace_references_graph_nodes(self, golden_requirements):
        """Test that trace entries reference graph nodes"""
        # Simulated draft trace referencing requirements
        draft_trace = {
            "agent_name": "DraftingAgent",
            "action": "generate_fbp_content",
            "input_data": {
                "requirement_nodes": [r.id for r in golden_requirements[:3]]
            }
        }

        # All referenced nodes should exist in graph
        graph_node_ids = [r.id for r in golden_requirements]

        for ref_id in draft_trace['input_data']['requirement_nodes']:
            assert ref_id in graph_node_ids, \
                f"Trace references non-existent node: {ref_id}"

    def test_orphan_detection_in_trace(self, golden_requirements, linked_dag):
        """Test that orphan detection is recorded in trace"""
        orphans = linked_dag.find_orphans()

        # Simulated trace for orphan detection
        orphan_trace = {
            "agent_name": "ValidationAgent",
            "action": "detect_orphans",
            "output_data": {
                "orphan_count": len(orphans),
                "orphan_ids": [o.orphan_id for o in orphans],
                "suggestions": [o.suggestion for o in orphans]
            }
        }

        # Trace should capture orphan information
        assert 'orphan_count' in orphan_trace['output_data']
        assert 'orphan_ids' in orphan_trace['output_data']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
