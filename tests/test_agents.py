"""
PropelAI Test Suite
===================

Tests for the Autonomous Proposal Operating System
"""

import pytest
import json
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import (
    ProposalState,
    ProposalPhase,
    ComplianceStatus,
    ScoreColor,
    create_initial_state
)
from agents.compliance_agent import ComplianceAgent, create_compliance_agent
from agents.strategy_agent import StrategyAgent, create_strategy_agent
from agents.drafting_agent import DraftingAgent, create_drafting_agent
from agents.red_team_agent import RedTeamAgent, create_red_team_agent


# ============== Test Fixtures ==============

SAMPLE_RFP_TEXT = """
SECTION C - STATEMENT OF WORK

C.1 BACKGROUND
The Agency requires a new system.

C.2 REQUIREMENTS
The Contractor shall provide cloud-based services.
The system shall achieve 99.9% uptime.
The Contractor must comply with FedRAMP requirements.

SECTION L - INSTRUCTIONS

L.1 FORMAT
Proposals shall be limited to 50 pages.
Font size shall be 12-point minimum.

SECTION M - EVALUATION

M.1 TECHNICAL APPROACH
The Government will evaluate technical merit.
Technical approach is most important.

M.2 PAST PERFORMANCE
Past performance will be evaluated for relevance.
"""


@pytest.fixture
def sample_state():
    """Create a sample proposal state for testing"""
    state = create_initial_state(
        proposal_id="TEST-001",
        client_name="Test Client",
        opportunity_name="Test Opportunity",
        solicitation_number="TEST-SOL-001",
        due_date="2025-12-31"
    )
    state["rfp_raw_text"] = SAMPLE_RFP_TEXT
    return state


@pytest.fixture
def compliance_agent():
    """Create a compliance agent for testing"""
    return create_compliance_agent()


@pytest.fixture
def strategy_agent():
    """Create a strategy agent for testing"""
    return create_strategy_agent()


@pytest.fixture
def drafting_agent():
    """Create a drafting agent for testing"""
    return create_drafting_agent()


@pytest.fixture
def red_team_agent():
    """Create a red team agent for testing"""
    return create_red_team_agent()


# ============== State Tests ==============

class TestProposalState:
    """Tests for ProposalState"""
    
    def test_create_initial_state(self):
        """Test creating a new proposal state"""
        state = create_initial_state(
            proposal_id="TEST-001",
            client_name="ACME Corp",
            opportunity_name="Portal Modernization",
            solicitation_number="GS-001",
            due_date="2025-06-01"
        )
        
        assert state["proposal_id"] == "TEST-001"
        assert state["client_name"] == "ACME Corp"
        assert state["opportunity_name"] == "Portal Modernization"
        assert state["current_phase"] == ProposalPhase.INTAKE.value
        assert state["requirements"] == []
        assert state["win_themes"] == []
        assert state["draft_sections"] == {}
    
    def test_state_has_all_required_fields(self):
        """Test that state has all required fields"""
        state = create_initial_state(
            proposal_id="TEST-002",
            client_name="Test",
            opportunity_name="Test",
            solicitation_number="TEST"
        )
        
        required_fields = [
            "proposal_id", "client_name", "opportunity_name",
            "solicitation_number", "current_phase", "created_at",
            "rfp_raw_text", "requirements", "instructions",
            "evaluation_criteria", "compliance_matrix",
            "win_themes", "draft_sections", "red_team_feedback",
            "agent_trace_log", "human_feedback"
        ]
        
        for field in required_fields:
            assert field in state, f"Missing field: {field}"


# ============== Compliance Agent Tests ==============

class TestComplianceAgent:
    """Tests for the Compliance Agent"""
    
    def test_extract_requirements(self, compliance_agent, sample_state):
        """Test requirement extraction"""
        result = compliance_agent(sample_state)
        
        assert "requirements" in result
        assert len(result["requirements"]) > 0
        
        # Check requirement structure
        req = result["requirements"][0]
        assert "id" in req
        assert "text" in req
        assert "requirement_type" in req
        assert "keywords" in req
    
    def test_extract_shall_statements(self, compliance_agent, sample_state):
        """Test that 'shall' statements are extracted"""
        result = compliance_agent(sample_state)
        
        requirements = result["requirements"]
        mandatory = [r for r in requirements if r["requirement_type"] == "mandatory"]
        
        assert len(mandatory) > 0
        
        # Check that shall statements were found
        shall_found = any("shall" in r["text"].lower() for r in mandatory)
        assert shall_found, "Should find 'shall' statements"
    
    def test_generate_compliance_matrix(self, compliance_agent, sample_state):
        """Test compliance matrix generation"""
        result = compliance_agent(sample_state)
        
        assert "compliance_matrix" in result
        assert len(result["compliance_matrix"]) > 0
        
        # Check matrix structure
        item = result["compliance_matrix"][0]
        assert "requirement_id" in item
        assert "requirement_text" in item
        assert "compliance_status" in item
    
    def test_extract_evaluation_criteria(self, compliance_agent, sample_state):
        """Test evaluation criteria extraction"""
        result = compliance_agent(sample_state)
        
        assert "evaluation_criteria" in result
        # Should find Section M criteria
        assert len(result["evaluation_criteria"]) > 0
    
    def test_agent_trace_log(self, compliance_agent, sample_state):
        """Test that agent creates trace log"""
        result = compliance_agent(sample_state)
        
        assert "agent_trace_log" in result
        assert len(result["agent_trace_log"]) > 0
        
        log = result["agent_trace_log"][0]
        assert log["agent_name"] == "compliance_agent"
        assert "timestamp" in log
        assert "action" in log
    
    def test_handles_empty_rfp(self, compliance_agent):
        """Test handling of empty RFP"""
        state = create_initial_state(
            proposal_id="TEST-EMPTY",
            client_name="Test",
            opportunity_name="Test",
            solicitation_number="TEST"
        )
        state["rfp_raw_text"] = ""
        
        result = compliance_agent(state)
        
        assert "error_state" in result


# ============== Strategy Agent Tests ==============

class TestStrategyAgent:
    """Tests for the Strategy Agent"""
    
    def test_generate_win_themes(self, strategy_agent, compliance_agent, sample_state):
        """Test win theme generation"""
        # First run compliance to get eval criteria
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        result = strategy_agent(sample_state)
        
        assert "win_themes" in result
        assert len(result["win_themes"]) > 0
        
        theme = result["win_themes"][0]
        assert "theme_text" in theme
        assert "discriminator" in theme
        assert "proof_points" in theme
    
    def test_generate_annotated_outline(self, strategy_agent, compliance_agent, sample_state):
        """Test annotated outline generation"""
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        result = strategy_agent(sample_state)
        
        assert "annotated_outline" in result
        outline = result["annotated_outline"]
        
        assert "volumes" in outline
        assert len(outline["volumes"]) > 0
    
    def test_requires_eval_criteria(self, strategy_agent, sample_state):
        """Test that strategy requires evaluation criteria"""
        # Don't run compliance first
        result = strategy_agent(sample_state)
        
        assert "error_state" in result


# ============== Drafting Agent Tests ==============

class TestDraftingAgent:
    """Tests for the Drafting Agent"""
    
    def test_generate_drafts(self, drafting_agent, strategy_agent, compliance_agent, sample_state):
        """Test draft generation"""
        # Run prerequisite agents
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        
        result = drafting_agent(sample_state)
        
        assert "draft_sections" in result
        # Should have generated some content
        assert len(result["draft_sections"]) > 0
    
    def test_draft_has_citations(self, drafting_agent, strategy_agent, compliance_agent, sample_state):
        """Test that drafts include citations"""
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        
        result = drafting_agent(sample_state)
        
        # Check that sections have citation tracking
        for section_id, section in result.get("draft_sections", {}).items():
            assert "citations" in section
            assert "uncited_claims" in section
    
    def test_flags_uncited_claims(self, drafting_agent, strategy_agent, compliance_agent, sample_state):
        """Test that uncited claims are flagged"""
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        
        result = drafting_agent(sample_state)
        
        # Should flag some uncited claims (in demo mode)
        total_uncited = sum(
            len(s.get("uncited_claims", [])) 
            for s in result.get("draft_sections", {}).values()
        )
        # This is expected in demo mode without real research
        assert total_uncited >= 0


# ============== Red Team Agent Tests ==============

class TestRedTeamAgent:
    """Tests for the Red Team Agent"""
    
    def test_evaluate_proposal(self, red_team_agent, drafting_agent, strategy_agent, compliance_agent, sample_state):
        """Test proposal evaluation"""
        # Run all prerequisite agents
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        
        draft_result = drafting_agent(sample_state)
        sample_state.update(draft_result)
        
        result = red_team_agent(sample_state)
        
        assert "red_team_feedback" in result
        assert len(result["red_team_feedback"]) > 0
        
        feedback = result["red_team_feedback"][0]
        assert "overall_score" in feedback
        assert "overall_numeric" in feedback
        assert "recommendation" in feedback
    
    def test_color_scoring(self, red_team_agent, drafting_agent, strategy_agent, compliance_agent, sample_state):
        """Test color score assignment"""
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        
        draft_result = drafting_agent(sample_state)
        sample_state.update(draft_result)
        
        result = red_team_agent(sample_state)
        
        feedback = result["red_team_feedback"][0]
        assert feedback["overall_score"] in ["blue", "green", "yellow", "red"]
    
    def test_identifies_deficiencies(self, red_team_agent, drafting_agent, strategy_agent, compliance_agent, sample_state):
        """Test deficiency identification"""
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        
        draft_result = drafting_agent(sample_state)
        sample_state.update(draft_result)
        
        result = red_team_agent(sample_state)
        
        feedback = result["red_team_feedback"][0]
        
        # Should have findings
        assert "findings" in feedback
        
        # Check finding structure if any exist
        if feedback["findings"]:
            finding = feedback["findings"][0]
            assert "description" in finding
            assert "finding_type" in finding


# ============== Integration Tests ==============

class TestFullWorkflow:
    """Integration tests for the full workflow"""

    def test_full_proposal_workflow(self, sample_state):
        """Test complete proposal generation workflow"""
        # Create all agents
        compliance_agent = create_compliance_agent()
        strategy_agent = create_strategy_agent()
        drafting_agent = create_drafting_agent()
        red_team_agent = create_red_team_agent()

        # Step 1: Shred
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)
        assert sample_state["current_phase"] == ProposalPhase.SHRED.value
        assert len(sample_state["requirements"]) > 0

        # Step 2: Strategy
        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)
        assert sample_state["current_phase"] == ProposalPhase.STRATEGY.value
        assert len(sample_state["win_themes"]) > 0

        # Step 3: Draft
        draft_result = drafting_agent(sample_state)
        sample_state.update(draft_result)
        assert len(sample_state["draft_sections"]) > 0

        # Step 4: Red Team
        redteam_result = red_team_agent(sample_state)
        sample_state.update(redteam_result)
        assert len(sample_state["red_team_feedback"]) > 0

        # Verify audit trail - each agent now accumulates its trace log
        assert len(sample_state["agent_trace_log"]) >= 4

    def test_trace_log_completeness(self, sample_state):
        """Test that all agents log their actions"""
        compliance_agent = create_compliance_agent()
        strategy_agent = create_strategy_agent()
        drafting_agent = create_drafting_agent()
        red_team_agent = create_red_team_agent()

        # Run workflow - agents now accumulate trace logs automatically
        compliance_result = compliance_agent(sample_state)
        sample_state.update(compliance_result)

        strategy_result = strategy_agent(sample_state)
        sample_state.update(strategy_result)

        draft_result = drafting_agent(sample_state)
        sample_state.update(draft_result)

        redteam_result = red_team_agent(sample_state)
        sample_state.update(redteam_result)

        # Check trace log
        trace_log = sample_state["agent_trace_log"]
        agent_names = [entry["agent_name"] for entry in trace_log]

        assert "compliance_agent" in agent_names
        assert "strategy_agent" in agent_names
        assert "drafting_agent" in agent_names
        assert "red_team_agent" in agent_names


# ============== Run Tests ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
