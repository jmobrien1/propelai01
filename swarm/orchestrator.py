"""
PropelAI v6.0 LangGraph Orchestrator
The State Machine - defines the proposal workflow as a directed cyclic graph.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from swarm.state import ProposalState, ProposalPhase, create_initial_state
from swarm.agents.compliance import ComplianceAgent
from swarm.agents.strategy import StrategyAgent
from swarm.agents.drafting import DraftingAgent
from swarm.agents.research import ResearchAgent
from swarm.agents.red_team import RedTeamAgent
from swarm.agents.supervisor import SupervisorAgent


logger = logging.getLogger(__name__)


class ProposalOrchestrator:
    """
    LangGraph-based orchestrator for the PropelAI agent swarm.

    Architecture:
    - Uses StateGraph for cyclic workflow support
    - Agents are graph nodes that transform ProposalState
    - Edges define transitions between agents
    - Conditional edges handle routing logic (revision loops, human escalation)
    - MemorySaver enables checkpointing for pause/resume
    """

    def __init__(
        self,
        checkpointer: Optional[Any] = None,
        max_revisions: int = 3,
    ):
        self.max_revisions = max_revisions
        self.checkpointer = checkpointer or MemorySaver()

        # Initialize agents
        self.compliance_agent = ComplianceAgent()
        self.strategy_agent = StrategyAgent()
        self.drafting_agent = DraftingAgent()
        self.research_agent = ResearchAgent()
        self.red_team_agent = RedTeamAgent()
        self.supervisor_agent = SupervisorAgent()

        # Build the graph
        self.graph = self._build_graph()

        # Compile with checkpointing
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.

        Flow:
        START → supervisor → (route based on phase)

        Phase routing:
        - SHRED: supervisor → compliance → supervisor
        - STRATEGY: supervisor → strategy → supervisor
        - DRAFTING: supervisor → drafting ↔ research → supervisor
        - REVIEW: supervisor → red_team → supervisor
        - FINALIZE: supervisor → human_review
        - COMPLETE: supervisor → END

        Revision loop:
        red_team → (if revise) → drafting → red_team
        """
        # Create the graph with ProposalState schema
        graph = StateGraph(ProposalState)

        # Add nodes (agents)
        graph.add_node("supervisor", self._run_supervisor)
        graph.add_node("compliance", self._run_compliance)
        graph.add_node("strategy", self._run_strategy)
        graph.add_node("drafting", self._run_drafting)
        graph.add_node("research", self._run_research)
        graph.add_node("red_team", self._run_red_team)
        graph.add_node("human_review", self._human_review)

        # Set entry point
        graph.set_entry_point("supervisor")

        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "compliance": "compliance",
                "strategy": "strategy",
                "drafting": "drafting",
                "research": "research",
                "red_team": "red_team",
                "human": "human_review",
                "done": END,
            }
        )

        # Edges back to supervisor after each agent
        graph.add_edge("compliance", "supervisor")
        graph.add_edge("strategy", "supervisor")
        graph.add_edge("research", "drafting")  # Research feeds into drafting
        graph.add_edge("drafting", "supervisor")

        # Conditional edge from red_team (revision loop)
        graph.add_conditional_edges(
            "red_team",
            self._route_from_red_team,
            {
                "revise": "drafting",
                "accept": "supervisor",
            }
        )

        # Human review ends the workflow
        graph.add_edge("human_review", END)

        return graph

    # =========================================================================
    # Node Functions (wrap agent execution)
    # =========================================================================

    async def _run_supervisor(self, state: ProposalState) -> ProposalState:
        """Run the supervisor agent."""
        return await self.supervisor_agent.run(state)

    async def _run_compliance(self, state: ProposalState) -> ProposalState:
        """Run the compliance agent and advance phase."""
        new_state = await self.compliance_agent.run(state)

        # Advance to STRATEGY phase after extraction
        if new_state.get("requirements"):
            new_state["current_phase"] = ProposalPhase.STRATEGY.value

        return new_state

    async def _run_strategy(self, state: ProposalState) -> ProposalState:
        """Run the strategy agent and advance phase."""
        new_state = await self.strategy_agent.run(state)

        # Advance to DRAFTING phase after strategy
        if new_state.get("win_themes"):
            new_state["current_phase"] = ProposalPhase.DRAFTING.value

        return new_state

    async def _run_drafting(self, state: ProposalState) -> ProposalState:
        """Run the drafting agent and advance phase."""
        new_state = await self.drafting_agent.run(state)

        # Advance to REVIEW phase after drafting
        if new_state.get("draft_sections"):
            new_state["current_phase"] = ProposalPhase.REVIEW.value

        return new_state

    async def _run_research(self, state: ProposalState) -> ProposalState:
        """Run the research agent to gather evidence."""
        return await self.research_agent.run(state)

    async def _run_red_team(self, state: ProposalState) -> ProposalState:
        """Run the red team agent for scoring."""
        new_state = await self.red_team_agent.run(state)

        # Check if we should advance to FINALIZE
        overall_score = new_state.get("overall_score", 0.0)
        if overall_score >= 0.85:  # 85%+ passes
            new_state["current_phase"] = ProposalPhase.FINALIZE.value

        return new_state

    async def _human_review(self, state: ProposalState) -> ProposalState:
        """Prepare state for human review."""
        state["current_phase"] = ProposalPhase.FINALIZE.value
        state["needs_human_review"] = True
        state["human_review_requested_at"] = datetime.utcnow().isoformat()
        return state

    # =========================================================================
    # Routing Functions (conditional edges)
    # =========================================================================

    def _route_from_supervisor(
        self,
        state: ProposalState,
    ) -> Literal["compliance", "strategy", "drafting", "research", "red_team", "human", "done"]:
        """
        Route to the next agent based on supervisor decision.
        """
        next_step = state.get("next_step", "")

        # Map supervisor decisions to graph nodes
        routing = {
            "compliance": "compliance",
            "strategy": "strategy",
            "drafting": "drafting",
            "research": "research",
            "red_team": "red_team",
            "human": "human",
            "done": "done",
        }

        return routing.get(next_step, "human")

    def _route_from_red_team(
        self,
        state: ProposalState,
    ) -> Literal["revise", "accept"]:
        """
        Route after red team review.
        """
        revision_count = state.get("revision_count", 0)
        overall_score = state.get("overall_score", 0.0)

        # Accept if score is good enough or max revisions reached
        if overall_score >= 0.85 or revision_count >= self.max_revisions:
            return "accept"

        # Check if red team requested revisions
        if state.get("next_step") == "drafting":
            return "revise"

        return "accept"

    # =========================================================================
    # Public Interface
    # =========================================================================

    async def run(
        self,
        rfp_text: str,
        proposal_id: Optional[str] = None,
        company_context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> ProposalState:
        """
        Run the full proposal generation workflow.

        Args:
            rfp_text: The RFP document text
            proposal_id: Optional proposal ID
            company_context: Optional company information
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final ProposalState with generated proposal
        """
        # Create initial state
        state = create_initial_state(
            proposal_id=proposal_id,
            rfp_text=rfp_text,
        )

        # Add company context if provided
        if company_context:
            state["company_context"] = company_context

        # Configure thread for checkpointing
        config = {"configurable": {"thread_id": thread_id or proposal_id}}

        logger.info(f"Starting proposal workflow for {proposal_id}")

        # Run the graph
        final_state = await self.app.ainvoke(state, config)

        logger.info(f"Completed proposal workflow for {proposal_id}")

        return final_state

    async def resume(
        self,
        thread_id: str,
        human_feedback: Optional[Dict[str, Any]] = None,
    ) -> ProposalState:
        """
        Resume a paused workflow from checkpoint.

        Args:
            thread_id: The thread ID to resume
            human_feedback: Optional human feedback to incorporate

        Returns:
            Final ProposalState after resuming
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Get the current state from checkpoint
        state = await self.app.aget_state(config)

        if state is None:
            raise ValueError(f"No checkpoint found for thread {thread_id}")

        current_state = state.values

        # Incorporate human feedback
        if human_feedback:
            current_state["human_feedback"] = human_feedback
            current_state["needs_human_review"] = False
            current_state["current_phase"] = ProposalPhase.COMPLETE.value

        # Resume execution
        final_state = await self.app.ainvoke(current_state, config)

        return final_state

    def get_status(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow.

        Args:
            thread_id: The thread ID to check

        Returns:
            Status dictionary with progress and phase info
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self.app.get_state(config)
            if state is None:
                return {"status": "not_found"}

            return self.supervisor_agent.get_workflow_status(state.values)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"status": "error", "error": str(e)}


# =============================================================================
# Factory Functions
# =============================================================================

def create_orchestrator(
    checkpointer: Optional[Any] = None,
    max_revisions: int = 3,
) -> ProposalOrchestrator:
    """
    Create a new proposal orchestrator.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence
        max_revisions: Maximum revision cycles before forcing accept

    Returns:
        Configured ProposalOrchestrator
    """
    return ProposalOrchestrator(
        checkpointer=checkpointer,
        max_revisions=max_revisions,
    )


async def run_proposal_workflow(
    rfp_text: str,
    proposal_id: str,
    company_context: Optional[Dict[str, Any]] = None,
) -> ProposalState:
    """
    Convenience function to run a complete proposal workflow.

    Args:
        rfp_text: The RFP document text
        proposal_id: Unique proposal identifier
        company_context: Optional company information

    Returns:
        Final ProposalState with generated proposal
    """
    orchestrator = create_orchestrator()
    return await orchestrator.run(
        rfp_text=rfp_text,
        proposal_id=proposal_id,
        company_context=company_context,
    )
