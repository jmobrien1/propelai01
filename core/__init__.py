"""PropelAI Core - State management and orchestration"""

from .state import (
    ProposalState,
    ProposalPhase,
    ComplianceStatus,
    ScoreColor,
    create_initial_state
)

# Orchestrator requires LangGraph - import conditionally
try:
    from .orchestrator import (
        ProposalOrchestrator,
        create_orchestrator
    )
    __all__ = [
        "ProposalState",
        "ProposalPhase",
        "ComplianceStatus", 
        "ScoreColor",
        "create_initial_state",
        "ProposalOrchestrator",
        "create_orchestrator",
    ]
except ImportError:
    __all__ = [
        "ProposalState",
        "ProposalPhase",
        "ComplianceStatus", 
        "ScoreColor",
        "create_initial_state",
    ]
