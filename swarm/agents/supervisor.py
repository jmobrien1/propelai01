"""
PropelAI v6.0 Supervisor Agent
The Orchestrator - coordinates the agent swarm and manages workflow.
"""

import json
from typing import List, Dict, Any, Optional, Literal

from swarm.agents.base import BaseAgent, AgentConfig
from swarm.state import ProposalState, ProposalPhase


SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent for PropelAI, orchestrating a team of specialized agents.

Your role is to:
1. DECIDE which agent should act next based on the current state
2. ROUTE work to the appropriate agent
3. HANDLE errors and retries
4. ESCALATE to human when needed

Your agent team:
- ComplianceAgent: Extracts requirements (Phase: SHRED)
- StrategyAgent: Develops win themes (Phase: STRATEGY)
- ResearchAgent: Gathers evidence (Called by Drafting)
- DraftingAgent: Writes proposal sections (Phase: DRAFTING)
- RedTeamAgent: Scores and reviews (Phase: REVIEW)

Decision Logic:
1. If current_phase is SHRED → route to ComplianceAgent
2. If current_phase is STRATEGY → route to StrategyAgent
3. If current_phase is DRAFTING → route to DraftingAgent (calls ResearchAgent)
4. If current_phase is REVIEW → route to RedTeamAgent
5. If RedTeam returns "revise" → route back to DraftingAgent
6. If RedTeam returns "accept" → move to FINALIZE
7. If retry_count > max_retries → ESCALATE to human

Output JSON:
{
  "next_agent": "compliance|strategy|drafting|research|red_team|human",
  "reasoning": "Why this decision",
  "instructions": "Specific instructions for the next agent"
}"""


class SupervisorAgent(BaseAgent):
    """
    The Supervisor Agent orchestrates the entire proposal swarm.

    Key capabilities:
    - Workflow routing
    - Error handling and retries
    - Human escalation
    - Progress tracking
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="SupervisorAgent",
                model="gemini-1.5-flash",  # Fast model for routing
                temperature=0.1,
                max_tokens=1024,
                system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            )
        super().__init__(config)

    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        Decide the next step in the workflow.
        """
        current_phase = state.get("current_phase", ProposalPhase.SHRED.value)
        next_step = state.get("next_step", "")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        last_error = state.get("last_error")

        # Check for too many retries
        if retry_count >= max_retries:
            self._log_action(
                state,
                "thought",
                f"Max retries ({max_retries}) exceeded. Escalating to human."
            )
            state["next_step"] = "human"
            return state

        # Handle errors
        if last_error:
            self._log_action(
                state,
                "observation",
                f"Handling error: {last_error}"
            )
            # Clear error and retry
            state["last_error"] = None
            state["retry_count"] = retry_count + 1
            # Keep the same next_step for retry
            return state

        # Route based on phase
        decision = await self._make_routing_decision(state)

        state["next_step"] = decision["next_agent"]

        # Add reasoning to scratchpad
        scratchpad = state.get("supervisor_scratchpad", [])
        scratchpad.append(f"Routing to {decision['next_agent']}: {decision['reasoning']}")
        state["supervisor_scratchpad"] = scratchpad[-10:]  # Keep last 10

        self._log_action(
            state,
            "output",
            f"Routing to {decision['next_agent']}"
        )

        return state

    async def _make_routing_decision(
        self,
        state: ProposalState,
    ) -> Dict[str, str]:
        """
        Make the routing decision based on current state.
        """
        current_phase = state.get("current_phase", ProposalPhase.SHRED.value)
        next_step = state.get("next_step", "")
        requirements_count = len(state.get("requirements", []))
        themes_count = len(state.get("win_themes", []))
        drafts_count = len(state.get("draft_sections", {}))
        overall_score = state.get("overall_score", 0.0)

        # Simple deterministic routing based on phase
        phase_routing = {
            ProposalPhase.SHRED.value: {
                "next_agent": "compliance",
                "reasoning": "Starting extraction phase",
                "instructions": "Extract all requirements with Iron Triangle mapping",
            },
            ProposalPhase.STRATEGY.value: {
                "next_agent": "strategy",
                "reasoning": f"Requirements extracted ({requirements_count}), developing strategy",
                "instructions": "Generate win themes with competitive positioning",
            },
            ProposalPhase.DRAFTING.value: {
                "next_agent": "drafting",
                "reasoning": f"Strategy complete ({themes_count} themes), drafting sections",
                "instructions": "Draft all sections using F-B-P framework",
            },
            ProposalPhase.REVIEW.value: {
                "next_agent": "red_team",
                "reasoning": f"Drafts complete ({drafts_count} sections), evaluating",
                "instructions": "Score all sections against Section M",
            },
            ProposalPhase.FINALIZE.value: {
                "next_agent": "human",
                "reasoning": f"All sections pass (score: {overall_score:.2f}), ready for review",
                "instructions": "Present final proposal for human approval",
            },
            ProposalPhase.COMPLETE.value: {
                "next_agent": "done",
                "reasoning": "Proposal generation complete",
                "instructions": "No further action needed",
            },
        }

        # Check for revision loop
        if next_step == "drafting" and current_phase == ProposalPhase.REVIEW.value:
            return {
                "next_agent": "drafting",
                "reasoning": "Red Team requested revisions",
                "instructions": "Revise sections based on feedback",
            }

        return phase_routing.get(current_phase, {
            "next_agent": "human",
            "reasoning": f"Unknown phase: {current_phase}",
            "instructions": "Escalate to human for guidance",
        })

    def get_workflow_status(self, state: ProposalState) -> Dict[str, Any]:
        """
        Get the current status of the workflow.
        """
        return {
            "proposal_id": state.get("proposal_id"),
            "current_phase": state.get("current_phase"),
            "next_step": state.get("next_step"),
            "progress": self._calculate_progress(state),
            "requirements_count": len(state.get("requirements", [])),
            "themes_count": len(state.get("win_themes", [])),
            "drafts_count": len(state.get("draft_sections", {})),
            "overall_score": state.get("overall_score", 0.0),
            "retry_count": state.get("retry_count", 0),
            "last_error": state.get("last_error"),
        }

    def _calculate_progress(self, state: ProposalState) -> float:
        """
        Calculate overall progress percentage.
        """
        phase_weights = {
            ProposalPhase.SHRED.value: 0.2,
            ProposalPhase.STRATEGY.value: 0.4,
            ProposalPhase.DRAFTING.value: 0.6,
            ProposalPhase.REVIEW.value: 0.8,
            ProposalPhase.FINALIZE.value: 0.9,
            ProposalPhase.COMPLETE.value: 1.0,
        }

        current_phase = state.get("current_phase", ProposalPhase.SHRED.value)
        return phase_weights.get(current_phase, 0.0)

    async def should_escalate(self, state: ProposalState) -> bool:
        """
        Determine if the workflow should escalate to human.
        """
        # Escalate conditions
        if state.get("retry_count", 0) >= state.get("max_retries", 3):
            return True

        if state.get("next_step") == "human":
            return True

        # Check for critical errors
        if state.get("last_error") and "critical" in state.get("last_error", "").lower():
            return True

        return False
