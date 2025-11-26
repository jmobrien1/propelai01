"""
PropelAI - Autonomous Proposal Operating System (APOS)

A Stateful Agentic Swarm for government proposal generation.
Built on LangGraph with Google-Native architecture.

Components:
- Compliance Agent (The Paralegal): RFP shredding and requirement extraction
- Strategy Agent (The Capture Manager): Win theme development
- Drafting Agent (The Writer): Citation-backed content generation
- Research Agent (The Librarian): Evidence retrieval
- Red Team Agent (The Evaluator): Government-style scoring
"""

__version__ = "1.0.0"
__author__ = "PropelAI"

from core.state import (
    ProposalState,
    ProposalPhase,
    ComplianceStatus,
    ScoreColor,
    create_initial_state
)

from core.orchestrator import (
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
