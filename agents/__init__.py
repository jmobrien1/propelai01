"""PropelAI Agents - Specialized AI agents for proposal development"""

# Note: Legacy agents (ComplianceAgent, StrategyAgent, etc.) require core.state module
# which is not available in the standalone deployment. Import them conditionally.

try:
    from .compliance_agent import ComplianceAgent, create_compliance_agent
    from .strategy_agent import StrategyAgent, create_strategy_agent
    from .drafting_agent import DraftingAgent, ResearchAgent, create_drafting_agent, create_research_agent
    from .red_team_agent import RedTeamAgent, create_red_team_agent
    LEGACY_AGENTS_AVAILABLE = True
except ImportError:
    # core.state not available - legacy agents disabled
    LEGACY_AGENTS_AVAILABLE = False
    ComplianceAgent = None
    create_compliance_agent = None
    StrategyAgent = None
    create_strategy_agent = None
    DraftingAgent = None
    ResearchAgent = None
    create_drafting_agent = None
    create_research_agent = None
    RedTeamAgent = None
    create_red_team_agent = None

__all__ = [
    "LEGACY_AGENTS_AVAILABLE",
    "ComplianceAgent",
    "create_compliance_agent",
    "StrategyAgent", 
    "create_strategy_agent",
    "DraftingAgent",
    "ResearchAgent",
    "create_drafting_agent",
    "create_research_agent",
    "RedTeamAgent",
    "create_red_team_agent",
]
