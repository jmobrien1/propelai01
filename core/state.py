"""
PropelAI Autonomous Proposal Operating System (APOS)
Core State Schema - The Global State that persists throughout the engagement lifecycle

This state object is the single source of truth for the entire proposal workflow.
It's checkpointed to PostgreSQL via LangGraph, enabling Human-in-the-Loop pausing.
"""

from typing import TypedDict, List, Dict, Optional, Any, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import operator


class ProposalPhase(str, Enum):
    """Tracks the current phase of the proposal lifecycle"""
    INTAKE = "intake"
    SHRED = "shred"               # RFP Ingestion
    STRATEGY = "strategy"         # Win Theme Development
    OUTLINE = "outline"           # Storyboarding
    DRAFTING = "drafting"         # Content Generation
    REVIEW = "review"             # Red Team Evaluation
    FINALIZE = "finalize"         # Final Polish
    SUBMITTED = "submitted"


class ComplianceStatus(str, Enum):
    """Status of compliance for a requirement"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DRAFT = "draft"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


class ScoreColor(str, Enum):
    """Government-style color scoring"""
    BLUE = "blue"       # Exceptional - significantly exceeds requirements
    GREEN = "green"     # Acceptable - meets requirements
    YELLOW = "yellow"   # Marginal - may not meet requirements
    RED = "red"         # Unacceptable - fails to meet requirements


@dataclass
class Requirement:
    """A single requirement extracted from the RFP (Section C 'shall' statements)"""
    id: str
    text: str
    section_ref: str              # e.g., "C.4.2.1"
    requirement_type: str         # e.g., "technical", "management", "past_performance"
    keywords: List[str]
    linked_instructions: List[str]  # Section L references
    linked_criteria: List[str]      # Section M references
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_STARTED
    assigned_section: Optional[str] = None
    assigned_owner: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class EvaluationCriterion:
    """An evaluation criterion from Section M"""
    id: str
    text: str
    section_ref: str
    factor_name: str              # e.g., "Technical Approach", "Past Performance"
    weight: Optional[float] = None  # Relative importance
    subfactors: List[str] = field(default_factory=list)


@dataclass
class WinTheme:
    """A strategic win theme aligned to evaluation criteria"""
    id: str
    theme_text: str
    discriminator: str            # What makes us different
    proof_points: List[str]       # Evidence from past performance
    linked_criteria: List[str]    # Section M criteria this addresses
    ghosting_language: Optional[str] = None  # Language to de-position competitors


@dataclass
class DraftSection:
    """A section of the proposal draft"""
    section_id: str
    section_title: str
    content: str
    citations: List[Dict[str, str]]  # List of {source, text, url}
    word_count: int
    page_allocation: Optional[int] = None
    compliance_score: Optional[float] = None
    quality_score: Optional[float] = None
    uncited_claims: List[str] = field(default_factory=list)  # Flagged for review
    version: int = 1
    last_modified: datetime = field(default_factory=datetime.now)
    modified_by: str = "system"


@dataclass
class RedTeamFeedback:
    """Feedback from the Red Team evaluation agent"""
    section_id: str
    overall_score: ScoreColor
    compliance_findings: List[Dict[str, Any]]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    evaluated_at: datetime = field(default_factory=datetime.now)


@dataclass
class LogEntry:
    """Immutable audit trail entry"""
    timestamp: datetime
    agent_name: str
    action: str
    input_summary: str
    output_summary: str
    reasoning_trace: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: Optional[int] = None
    token_count: Optional[int] = None


@dataclass
class HumanFeedback:
    """Brenda's corrections - the gold for fine-tuning"""
    timestamp: datetime
    section_id: str
    feedback_type: str           # "edit", "reject", "approve", "comment"
    original_content: str
    corrected_content: Optional[str]
    correction_reason: Optional[str]
    user_id: str


# The main state object that LangGraph checkpoints
class ProposalState(TypedDict):
    """
    The Global State Schema for PropelAI
    
    This is the persistent state object maintained throughout the proposal lifecycle.
    It's serialized to PostgreSQL via LangGraph checkpointing.
    """
    # === Metadata ===
    proposal_id: str
    client_name: str
    opportunity_name: str
    solicitation_number: str
    due_date: Optional[str]
    current_phase: str
    created_at: str
    updated_at: str
    
    # === RFP Ingestion Data ===
    rfp_raw_text: str                          # Full text of the RFP
    rfp_file_paths: List[str]                  # Paths to uploaded documents
    rfp_metadata: Dict[str, Any]               # Agency, NAICS, set-aside, etc.
    
    # === The "Iron Triangle" ===
    requirements: Annotated[List[Dict], operator.add]      # Section C - What to do
    instructions: Annotated[List[Dict], operator.add]      # Section L - How to format
    evaluation_criteria: Annotated[List[Dict], operator.add]  # Section M - How scored
    
    # === Requirements Graph (dependency mapping) ===
    requirements_graph: Dict[str, List[str]]   # Adjacency list: req_id -> [linked_ids]
    
    # === Compliance Matrix ===
    compliance_matrix: List[Dict[str, Any]]    # The structured obligations
    
    # === Strategy ===
    win_themes: List[Dict[str, Any]]
    competitor_analysis: Dict[str, Any]
    price_to_win: Optional[Dict[str, Any]]
    
    # === Draft Content ===
    annotated_outline: Dict[str, Any]          # The storyboard with page allocations
    draft_sections: Dict[str, Dict]            # section_id -> DraftSection dict
    
    # === Quality & Governance ===
    red_team_feedback: Annotated[List[Dict], operator.add]
    proposal_quality_score: Optional[float]
    
    # === Audit & Learning ===
    agent_trace_log: Annotated[List[Dict], operator.add]   # Immutable audit trail
    human_feedback: Annotated[List[Dict], operator.add]    # "Brenda's" corrections
    
    # === Agent Communication ===
    messages: Annotated[List[Dict], operator.add]          # Inter-agent messages
    current_task: Optional[str]
    pending_human_review: bool
    error_state: Optional[str]


def create_initial_state(
    proposal_id: str,
    client_name: str,
    opportunity_name: str,
    solicitation_number: str,
    due_date: Optional[str] = None
) -> ProposalState:
    """Factory function to create a new ProposalState with defaults"""
    now = datetime.now().isoformat()
    
    return ProposalState(
        # Metadata
        proposal_id=proposal_id,
        client_name=client_name,
        opportunity_name=opportunity_name,
        solicitation_number=solicitation_number,
        due_date=due_date,
        current_phase=ProposalPhase.INTAKE.value,
        created_at=now,
        updated_at=now,
        
        # RFP Ingestion
        rfp_raw_text="",
        rfp_file_paths=[],
        rfp_metadata={},
        
        # Iron Triangle
        requirements=[],
        instructions=[],
        evaluation_criteria=[],
        
        # Graph
        requirements_graph={},
        
        # Compliance
        compliance_matrix=[],
        
        # Strategy
        win_themes=[],
        competitor_analysis={},
        price_to_win=None,
        
        # Drafts
        annotated_outline={},
        draft_sections={},
        
        # Quality
        red_team_feedback=[],
        proposal_quality_score=None,
        
        # Audit
        agent_trace_log=[],
        human_feedback=[],
        
        # Communication
        messages=[],
        current_task=None,
        pending_human_review=False,
        error_state=None,
    )
