"""
PropelAI Orchestration Engine
Implements the Orchestrator-Worker (Supervisor) pattern using LangGraph

The Supervisor Agent acts as the project manager:
- Decomposes high-level goals into tasks
- Routes tasks to specialized worker agents
- Manages state transitions and checkpointing
- Handles errors and retries
"""

from typing import Literal, List, Dict, Any, Optional, Callable
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from core.state import ProposalState, ProposalPhase, create_initial_state


# Agent routing decisions
AGENT_ROUTES = {
    "compliance": "compliance_agent",
    "strategy": "strategy_agent", 
    "drafting": "drafting_agent",
    "research": "research_agent",
    "red_team": "red_team_agent",
    "human": "human_review",
    "complete": END,
}


class ProposalOrchestrator:
    """
    The Supervisor Agent - orchestrates the proposal workflow
    
    This implements the Orchestrator-Worker pattern where:
    - Supervisor decomposes goals and routes to workers
    - Workers execute specific tasks and report back
    - State is persisted via LangGraph checkpointing
    """
    
    def __init__(
        self,
        compliance_agent: Callable,
        strategy_agent: Optional[Callable] = None,
        drafting_agent: Optional[Callable] = None,
        research_agent: Optional[Callable] = None,
        red_team_agent: Optional[Callable] = None,
        checkpointer: Optional[Any] = None
    ):
        self.compliance_agent = compliance_agent
        self.strategy_agent = strategy_agent
        self.drafting_agent = drafting_agent
        self.research_agent = research_agent
        self.red_team_agent = red_team_agent
        self.checkpointer = checkpointer or MemorySaver()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for the proposal workflow"""
        
        # Create the graph with our state schema
        builder = StateGraph(ProposalState)
        
        # Add nodes for each agent
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("compliance_agent", self._compliance_node)
        builder.add_node("strategy_agent", self._strategy_node)
        builder.add_node("drafting_agent", self._drafting_node)
        builder.add_node("research_agent", self._research_node)
        builder.add_node("red_team_agent", self._red_team_node)
        builder.add_node("human_review", self._human_review_node)
        
        # Set entry point
        builder.set_entry_point("supervisor")
        
        # Add conditional edges from supervisor
        builder.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            AGENT_ROUTES
        )
        
        # All workers report back to supervisor
        for agent in ["compliance_agent", "strategy_agent", "drafting_agent", 
                      "research_agent", "red_team_agent"]:
            builder.add_edge(agent, "supervisor")
        
        # Human review can continue or end
        builder.add_conditional_edges(
            "human_review",
            self._route_from_human,
            {"continue": "supervisor", "complete": END}
        )
        
        # Compile with checkpointer for persistence
        return builder.compile(checkpointer=self.checkpointer)
    
    def _supervisor_node(self, state: ProposalState) -> Dict[str, Any]:
        """
        The Supervisor Agent - determines next action based on current state
        
        This is the 'brain' that:
        1. Analyzes current proposal phase
        2. Determines what work is needed
        3. Routes to appropriate worker agent
        """
        now = datetime.now().isoformat()
        current_phase = state.get("current_phase", ProposalPhase.INTAKE.value)
        
        # Log the supervisor's reasoning
        log_entry = {
            "timestamp": now,
            "agent_name": "supervisor",
            "action": "analyze_state",
            "input_summary": f"Phase: {current_phase}",
            "output_summary": "",
            "reasoning_trace": ""
        }
        
        # Determine next task based on phase and state
        next_task = None
        reasoning = []
        
        if current_phase == ProposalPhase.INTAKE.value:
            # Check if we have RFP content to process
            if state.get("rfp_raw_text"):
                next_task = "compliance"
                reasoning.append("RFP content available, routing to Compliance Agent for shredding")
            else:
                next_task = "human"
                reasoning.append("Waiting for RFP upload")
                
        elif current_phase == ProposalPhase.SHRED.value:
            # Check if shredding is complete
            if state.get("compliance_matrix") and len(state.get("requirements", [])) > 0:
                next_task = "strategy"
                reasoning.append("Compliance matrix complete, routing to Strategy Agent")
            else:
                next_task = "compliance"
                reasoning.append("Continuing RFP shredding")
                
        elif current_phase == ProposalPhase.STRATEGY.value:
            # Check if win themes are defined
            if state.get("win_themes") and len(state.get("win_themes", [])) > 0:
                # Check if outline is done
                if state.get("annotated_outline"):
                    next_task = "drafting"
                    reasoning.append("Strategy complete, routing to Drafting Agent")
                else:
                    next_task = "strategy"
                    reasoning.append("Continuing storyboard development")
            else:
                next_task = "strategy"
                reasoning.append("Developing win themes")
                
        elif current_phase == ProposalPhase.DRAFTING.value:
            # Check if drafts need research
            drafts = state.get("draft_sections", {})
            uncited = any(d.get("uncited_claims", []) for d in drafts.values())
            
            if uncited:
                next_task = "research"
                reasoning.append("Drafts have uncited claims, routing to Research Agent")
            elif not all(d.get("content") for d in drafts.values()):
                next_task = "drafting"
                reasoning.append("Continuing draft development")
            else:
                next_task = "red_team"
                reasoning.append("Drafts complete, routing to Red Team for evaluation")
                
        elif current_phase == ProposalPhase.REVIEW.value:
            feedback = state.get("red_team_feedback", [])
            if feedback:
                latest = feedback[-1] if feedback else {}
                score = latest.get("overall_score", "red")
                
                if score in ["blue", "green"]:
                    next_task = "human"
                    reasoning.append("Red Team approved, routing to human for final review")
                else:
                    next_task = "drafting"
                    reasoning.append(f"Score is {score}, routing back to Drafting for remediation")
            else:
                next_task = "red_team"
                reasoning.append("Running Red Team evaluation")
                
        elif current_phase == ProposalPhase.FINALIZE.value:
            next_task = "human"
            reasoning.append("Final human review before submission")
            
        else:
            next_task = "human"
            reasoning.append("Unknown phase, requesting human guidance")
        
        log_entry["output_summary"] = f"Routing to: {next_task}"
        log_entry["reasoning_trace"] = " | ".join(reasoning)
        
        return {
            "current_task": next_task,
            "agent_trace_log": [log_entry],
            "updated_at": now
        }
    
    def _route_from_supervisor(self, state: ProposalState) -> str:
        """Routing function - returns the next node based on current_task"""
        task = state.get("current_task", "human")
        return AGENT_ROUTES.get(task, "human_review")
    
    def _route_from_human(self, state: ProposalState) -> str:
        """Route after human review"""
        if state.get("current_phase") == ProposalPhase.SUBMITTED.value:
            return "complete"
        if state.get("pending_human_review", False):
            return "continue"
        return "continue"
    
    def _compliance_node(self, state: ProposalState) -> Dict[str, Any]:
        """Execute the Compliance Agent (The Paralegal)"""
        if self.compliance_agent:
            return self.compliance_agent(state)
        return {"current_phase": ProposalPhase.SHRED.value}
    
    def _strategy_node(self, state: ProposalState) -> Dict[str, Any]:
        """Execute the Strategy Agent (The Capture Manager)"""
        if self.strategy_agent:
            return self.strategy_agent(state)
        return {"current_phase": ProposalPhase.STRATEGY.value}
    
    def _drafting_node(self, state: ProposalState) -> Dict[str, Any]:
        """Execute the Drafting Agent (The Writer)"""
        if self.drafting_agent:
            return self.drafting_agent(state)
        return {"current_phase": ProposalPhase.DRAFTING.value}
    
    def _research_node(self, state: ProposalState) -> Dict[str, Any]:
        """Execute the Research Agent (The Librarian)"""
        if self.research_agent:
            return self.research_agent(state)
        return {}
    
    def _red_team_node(self, state: ProposalState) -> Dict[str, Any]:
        """Execute the Red Team Agent (The Evaluator)"""
        if self.red_team_agent:
            return self.red_team_agent(state)
        return {"current_phase": ProposalPhase.REVIEW.value}
    
    def _human_review_node(self, state: ProposalState) -> Dict[str, Any]:
        """
        Human-in-the-Loop checkpoint
        
        This is where LangGraph pauses for human review.
        The state is checkpointed and can be resumed later.
        """
        return {
            "pending_human_review": True,
            "agent_trace_log": [{
                "timestamp": datetime.now().isoformat(),
                "agent_name": "supervisor",
                "action": "request_human_review",
                "input_summary": f"Phase: {state.get('current_phase')}",
                "output_summary": "Pausing for human review",
                "reasoning_trace": "Human approval required before continuing"
            }]
        }
    
    async def run(
        self, 
        initial_state: ProposalState,
        thread_id: str,
        interrupt_before: Optional[List[str]] = None
    ) -> ProposalState:
        """
        Execute the proposal workflow
        
        Args:
            initial_state: The starting state
            thread_id: Unique identifier for checkpointing
            interrupt_before: List of nodes to pause before (for HITL)
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }
        
        if interrupt_before:
            config["interrupt_before"] = interrupt_before
        
        result = await self.graph.ainvoke(initial_state, config)
        return result
    
    def resume(self, thread_id: str, human_feedback: Optional[Dict] = None) -> ProposalState:
        """
        Resume a paused workflow with optional human feedback
        
        This is called when Brenda comes back to review on Tuesday
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get current state from checkpoint
        state = self.graph.get_state(config)
        
        if human_feedback:
            # Add human feedback to state
            state.values["human_feedback"] = state.values.get("human_feedback", []) + [human_feedback]
            state.values["pending_human_review"] = False
        
        # Resume execution
        return self.graph.invoke(state.values, config)


def create_orchestrator(
    compliance_agent: Callable,
    strategy_agent: Optional[Callable] = None,
    drafting_agent: Optional[Callable] = None,
    research_agent: Optional[Callable] = None,
    red_team_agent: Optional[Callable] = None,
) -> ProposalOrchestrator:
    """Factory function to create a configured orchestrator"""
    return ProposalOrchestrator(
        compliance_agent=compliance_agent,
        strategy_agent=strategy_agent,
        drafting_agent=drafting_agent,
        research_agent=research_agent,
        red_team_agent=red_team_agent,
    )
