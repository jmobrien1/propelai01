"""
Draft-Critique-Expand Loop with LangGraph

From Long-Form Generation Strategy:
"The key to defeating brevity bias is the cyclic DCE workflow:
 1. DRAFT: Generate initial section (Writer)
 2. CRITIQUE: Blue Team review (Critic)
 3. EVALUATE: Check against criteria (Criteria-Eval)
 4. EXPAND: If score < threshold, add content guided by gaps
 5. LOOP: Repeat until criteria passed or max iterations

LangGraph provides the cyclic state machine that DAGs cannot achieve."

This module implements the DCE loop as a LangGraph workflow.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class DCEPhase(str, Enum):
    """Current phase in the DCE loop"""
    DRAFT = "draft"
    CRITIQUE = "critique"
    EVALUATE = "evaluate"
    EXPAND = "expand"
    COMPLETE = "complete"
    FAILED = "failed"


class DCEState(TypedDict):
    """
    State for the Draft-Critique-Expand workflow.

    This state is passed through all nodes and updated at each step.
    """
    # Section info
    section_id: str
    section_title: str
    page_target: float
    requirements: List[str]  # Requirements to address

    # Content
    current_draft: str
    draft_history: Annotated[List[str], operator.add]  # Append-only
    version: int

    # Critique results
    critique_score: float
    critique_gaps: List[str]
    critique_recommendations: List[str]
    critique_expand_required: bool

    # Evaluation results
    eval_score: float
    eval_gaps: List[str]
    eval_passes_threshold: bool

    # Expansion guidance
    expansion_zones: List[Dict[str, Any]]
    target_expansion_factor: float

    # Loop control
    current_phase: str
    iteration: int
    max_iterations: int
    should_continue: bool
    termination_reason: str

    # Error handling
    errors: Annotated[List[str], operator.add]


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

async def draft_node(state: DCEState, config: Dict[str, Any]) -> DCEState:
    """
    Generate initial draft or expanded draft.

    Uses the Writer model (GPT-4 fine-tuned) for prose generation.
    """
    logger.info(f"[DRAFT] Section {state['section_id']} v{state['version']}")

    router = config.get("router")
    if router is None:
        state["errors"].append("No model router configured")
        state["current_phase"] = DCEPhase.FAILED.value
        return state

    try:
        from ..integrations.llm_clients import LLMMessage, GenerationConfig, TaskType, ModelRole

        # Build prompt based on whether this is initial draft or expansion
        if state["version"] == 1:
            # Initial draft
            system_prompt = """You are an expert government proposal writer with decades
of experience winning federal contracts. Generate detailed, substantive content that
directly addresses every requirement. Be thorough - government evaluators reward
depth and specificity.

DO NOT be brief. DO NOT summarize. EXPAND on every point with:
- Specific technical details
- Quantified benefits
- Past performance examples
- Risk mitigation strategies"""

            user_prompt = f"""Write a comprehensive draft for this proposal section:

SECTION: {state['section_title']} ({state['section_id']})
TARGET LENGTH: {state['page_target']} pages (~{int(state['page_target'] * 500)} words)

REQUIREMENTS TO ADDRESS:
{chr(10).join(f'- {r}' for r in state['requirements'][:20])}

Generate detailed, substantive content. Be specific. Include examples.
Every requirement must be explicitly addressed."""

        else:
            # Expansion draft
            system_prompt = """You are expanding an existing proposal section based on
critique feedback. Your job is to ADD content, not rewrite. Focus on:
- Filling identified gaps
- Adding specific details
- Strengthening weak areas
- Incorporating missing requirements

Maintain the existing structure while expanding."""

            expansion_guidance = "\n".join(
                f"- {zone.get('location', '')}: {zone.get('suggested_addition', '')}"
                for zone in state.get("expansion_zones", [])[:10]
            )

            user_prompt = f"""Expand this proposal section based on critique feedback:

CURRENT DRAFT:
{state['current_draft']}

CRITIQUE GAPS:
{chr(10).join(f'- {g}' for g in state.get('critique_gaps', [])[:10])}

EVALUATION GAPS:
{chr(10).join(f'- {g}' for g in state.get('eval_gaps', [])[:10])}

EXPANSION GUIDANCE:
{expansion_guidance}

TARGET: Expand by approximately {int((state.get('target_expansion_factor', 1.3) - 1) * 100)}%

Add new content to address gaps. Maintain existing structure."""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        # Route to Writer (GPT-4) or fall back to Architect (Claude)
        # Using higher max_tokens to encourage longer output
        config_gen = GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
            min_output_tokens=2000  # Combat brevity bias
        )

        try:
            response = await router.route(
                TaskType.SECTION_DRAFTING,
                messages,
                config_gen
            )
        except Exception:
            # Fall back to Architect
            response = await router.route_to_role(
                ModelRole.ARCHITECT,
                messages,
                config_gen,
                TaskType.SECTION_DRAFTING
            )

        # Update state
        state["current_draft"] = response.content
        state["draft_history"].append(response.content)
        state["current_phase"] = DCEPhase.CRITIQUE.value

        logger.info(f"[DRAFT] Generated {len(response.content)} chars")

    except Exception as e:
        logger.error(f"[DRAFT] Error: {e}")
        state["errors"].append(f"Draft error: {str(e)}")
        state["current_phase"] = DCEPhase.FAILED.value

    return state


async def critique_node(state: DCEState, config: Dict[str, Any]) -> DCEState:
    """
    Critique the current draft using Blue Team review.

    Uses the Critic model (Claude) for adversarial review.
    """
    logger.info(f"[CRITIQUE] Section {state['section_id']} v{state['version']}")

    router = config.get("router")
    if router is None:
        state["errors"].append("No model router configured")
        state["current_phase"] = DCEPhase.FAILED.value
        return state

    try:
        from ..integrations.llm_clients import TaskType, ModelRole
        from ..integrations.claude_client import ClaudeClient

        # Get Critic client directly for specialized critique method
        critic = router.get_client(ModelRole.CRITIC)

        if critic is None or not isinstance(critic, ClaudeClient):
            # Fall back to generic critique via router
            from ..integrations.llm_clients import LLMMessage, GenerationConfig

            messages = [
                LLMMessage(
                    role="system",
                    content="You are a demanding government proposal reviewer. Find weaknesses and gaps."
                ),
                LLMMessage(
                    role="user",
                    content=f"Critique this draft:\n\n{state['current_draft'][:15000]}\n\nRequirements:\n{state['requirements'][:10]}"
                )
            ]

            response = await router.route(
                TaskType.DRAFT_CRITIQUE,
                messages,
                GenerationConfig(temperature=0.4)
            )

            # Basic parsing
            state["critique_score"] = 70.0  # Default
            state["critique_gaps"] = ["Unable to parse detailed critique"]
            state["critique_recommendations"] = ["Review draft manually"]
            state["critique_expand_required"] = True

        else:
            # Use specialized critique method
            from ..integrations.claude_client import CritiqueResult

            # Build evaluation criteria from requirements
            eval_criteria = [
                {"id": f"REQ-{i}", "text": req, "weight": 1.0}
                for i, req in enumerate(state["requirements"][:20])
            ]

            critique_result = await critic.critique_draft(
                draft_content=state["current_draft"],
                section_requirements=state["requirements"],
                evaluation_criteria=eval_criteria,
                page_target=state["page_target"]
            )

            # Update state from critique
            state["critique_score"] = critique_result.overall_score
            state["critique_gaps"] = critique_result.gaps
            state["critique_recommendations"] = critique_result.recommendations
            state["critique_expand_required"] = critique_result.expand_required

            logger.info(
                f"[CRITIQUE] Score: {critique_result.overall_score}, "
                f"Expand required: {critique_result.expand_required}"
            )

        state["current_phase"] = DCEPhase.EVALUATE.value

    except Exception as e:
        logger.error(f"[CRITIQUE] Error: {e}")
        state["errors"].append(f"Critique error: {str(e)}")
        state["current_phase"] = DCEPhase.FAILED.value

    return state


async def evaluate_node(state: DCEState, config: Dict[str, Any]) -> DCEState:
    """
    Evaluate draft against criteria using Criteria-Eval framework.

    This is the objective scoring that determines loop continuation.
    """
    logger.info(f"[EVALUATE] Section {state['section_id']} v{state['version']}")

    try:
        from ..enhanced_compliance.criteria_eval import (
            CriteriaEvaluator,
            CriteriaChecklist,
            EvaluationCriterion,
            CriterionPriority
        )

        evaluator = CriteriaEvaluator(
            pass_threshold=config.get("pass_threshold", 80.0),
            critical_threshold=config.get("critical_threshold", 95.0)
        )

        # Build checklist from requirements
        checklist = CriteriaChecklist(
            section_id=state["section_id"],
            section_title=state["section_title"]
        )

        for i, req in enumerate(state["requirements"]):
            checklist.criteria.append(EvaluationCriterion(
                id=f"REQ-{state['section_id']}-{i:03d}",
                text=req,
                source_section=state["section_id"],
                priority=CriterionPriority.HIGH
            ))

        # Run evaluation
        result = evaluator.evaluate_draft(
            draft_text=state["current_draft"],
            checklist=checklist,
            draft_version=state["version"]
        )

        # Update state
        state["eval_score"] = result.overall_score
        state["eval_gaps"] = result.gaps
        state["eval_passes_threshold"] = result.passes_threshold
        state["target_expansion_factor"] = result.estimated_expansion_needed

        logger.info(
            f"[EVALUATE] Score: {result.overall_score:.1f}%, "
            f"Passes: {result.passes_threshold}"
        )

        # Determine next phase
        if result.passes_threshold:
            state["current_phase"] = DCEPhase.COMPLETE.value
            state["termination_reason"] = f"Passed with score {result.overall_score:.1f}%"
            state["should_continue"] = False
        elif state["iteration"] >= state["max_iterations"]:
            state["current_phase"] = DCEPhase.COMPLETE.value
            state["termination_reason"] = f"Max iterations reached ({state['max_iterations']})"
            state["should_continue"] = False
        else:
            state["current_phase"] = DCEPhase.EXPAND.value

    except Exception as e:
        logger.error(f"[EVALUATE] Error: {e}")
        state["errors"].append(f"Evaluate error: {str(e)}")
        state["current_phase"] = DCEPhase.FAILED.value

    return state


async def expand_node(state: DCEState, config: Dict[str, Any]) -> DCEState:
    """
    Generate expansion guidance for the next draft iteration.

    Uses Critic (Claude) to produce specific expansion instructions.
    """
    logger.info(f"[EXPAND] Section {state['section_id']} preparing expansion")

    router = config.get("router")

    try:
        from ..integrations.llm_clients import ModelRole
        from ..integrations.claude_client import ClaudeClient

        # Get Critic for expansion guidance
        critic = router.get_client(ModelRole.CRITIC) if router else None

        if critic and isinstance(critic, ClaudeClient):
            # Build critique result for expansion guidance
            from ..integrations.claude_client import CritiqueResult

            critique = CritiqueResult(
                overall_score=state["critique_score"],
                strengths=[],
                weaknesses=[],
                gaps=state["critique_gaps"] + state["eval_gaps"],
                recommendations=state["critique_recommendations"],
                evaluator_perspective="",
                expand_required=True,
                raw_critique=""
            )

            guidance = await critic.generate_expansion_guidance(
                draft_content=state["current_draft"],
                critique=critique,
                target_expansion=state["target_expansion_factor"]
            )

            state["expansion_zones"] = guidance.get("expansion_zones", [])

        else:
            # Simple expansion zones from gaps
            state["expansion_zones"] = [
                {"location": "Throughout", "suggested_addition": gap}
                for gap in (state["eval_gaps"] + state["critique_gaps"])[:5]
            ]

        # Update for next iteration
        state["version"] += 1
        state["iteration"] += 1
        state["current_phase"] = DCEPhase.DRAFT.value

        logger.info(f"[EXPAND] Generated {len(state['expansion_zones'])} expansion zones")

    except Exception as e:
        logger.error(f"[EXPAND] Error: {e}")
        state["errors"].append(f"Expand error: {str(e)}")
        # Continue to draft phase anyway
        state["version"] += 1
        state["iteration"] += 1
        state["current_phase"] = DCEPhase.DRAFT.value

    return state


# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_after_evaluate(state: DCEState) -> str:
    """Determine next node after evaluation"""
    if state["current_phase"] == DCEPhase.COMPLETE.value:
        return END
    elif state["current_phase"] == DCEPhase.FAILED.value:
        return END
    elif state["current_phase"] == DCEPhase.EXPAND.value:
        return "expand"
    else:
        return END


def route_after_expand(state: DCEState) -> str:
    """Determine next node after expansion"""
    if state["should_continue"] and state["iteration"] < state["max_iterations"]:
        return "draft"
    else:
        return END


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_dce_workflow() -> StateGraph:
    """
    Build the Draft-Critique-Expand workflow graph.

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph
    workflow = StateGraph(DCEState)

    # Add nodes
    workflow.add_node("draft", draft_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("expand", expand_node)

    # Add edges
    workflow.set_entry_point("draft")
    workflow.add_edge("draft", "critique")
    workflow.add_edge("critique", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "expand": "expand",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "expand",
        route_after_expand,
        {
            "draft": "draft",
            END: END
        }
    )

    return workflow


# ============================================================================
# WORKFLOW EXECUTOR
# ============================================================================

@dataclass
class DCEResult:
    """Result of the Draft-Critique-Expand workflow"""
    section_id: str
    final_draft: str
    final_score: float
    iterations: int
    passed: bool
    termination_reason: str
    draft_history: List[str]
    gaps_remaining: List[str]
    errors: List[str]


class DCEWorkflow:
    """
    Executor for the Draft-Critique-Expand workflow.

    Usage:
        workflow = DCEWorkflow(router)
        result = await workflow.run(
            section_id="3.0",
            section_title="Technical Approach",
            requirements=["Describe methodology...", "Show experience..."],
            page_target=15.0
        )
    """

    def __init__(
        self,
        router: Any = None,
        pass_threshold: float = 80.0,
        critical_threshold: float = 95.0,
        max_iterations: int = 5
    ):
        """
        Initialize the DCE workflow.

        Args:
            router: Model router for LLM access
            pass_threshold: Minimum score to pass
            critical_threshold: Minimum critical score
            max_iterations: Maximum DCE loop iterations
        """
        self.router = router
        self.pass_threshold = pass_threshold
        self.critical_threshold = critical_threshold
        self.max_iterations = max_iterations

        # Build workflow
        self.workflow = build_dce_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    async def run(
        self,
        section_id: str,
        section_title: str,
        requirements: List[str],
        page_target: float = 10.0,
        thread_id: Optional[str] = None
    ) -> DCEResult:
        """
        Run the DCE workflow for a section.

        Args:
            section_id: Section identifier
            section_title: Section title
            requirements: List of requirements to address
            page_target: Target page count
            thread_id: Optional thread ID for checkpointing

        Returns:
            DCEResult with final draft and metrics
        """
        # Initialize state
        initial_state: DCEState = {
            "section_id": section_id,
            "section_title": section_title,
            "page_target": page_target,
            "requirements": requirements,
            "current_draft": "",
            "draft_history": [],
            "version": 1,
            "critique_score": 0.0,
            "critique_gaps": [],
            "critique_recommendations": [],
            "critique_expand_required": True,
            "eval_score": 0.0,
            "eval_gaps": [],
            "eval_passes_threshold": False,
            "expansion_zones": [],
            "target_expansion_factor": 1.3,
            "current_phase": DCEPhase.DRAFT.value,
            "iteration": 1,
            "max_iterations": self.max_iterations,
            "should_continue": True,
            "termination_reason": "",
            "errors": []
        }

        # Config for nodes
        config = {
            "router": self.router,
            "pass_threshold": self.pass_threshold,
            "critical_threshold": self.critical_threshold,
            "configurable": {
                "thread_id": thread_id or f"dce-{section_id}"
            }
        }

        logger.info(f"Starting DCE workflow for {section_id}: {section_title}")

        # Run workflow
        try:
            final_state = await self.app.ainvoke(
                initial_state,
                config=config
            )

            return DCEResult(
                section_id=section_id,
                final_draft=final_state["current_draft"],
                final_score=final_state["eval_score"],
                iterations=final_state["iteration"],
                passed=final_state["eval_passes_threshold"],
                termination_reason=final_state["termination_reason"],
                draft_history=final_state["draft_history"],
                gaps_remaining=final_state["eval_gaps"],
                errors=final_state["errors"]
            )

        except Exception as e:
            logger.error(f"DCE workflow failed: {e}")
            return DCEResult(
                section_id=section_id,
                final_draft="",
                final_score=0.0,
                iterations=0,
                passed=False,
                termination_reason=f"Workflow error: {str(e)}",
                draft_history=[],
                gaps_remaining=[],
                errors=[str(e)]
            )


async def run_dce_loop(
    section_id: str,
    section_title: str,
    requirements: List[str],
    router: Any,
    page_target: float = 10.0,
    max_iterations: int = 5
) -> DCEResult:
    """
    Convenience function to run the DCE loop.

    Args:
        section_id: Section identifier
        section_title: Section title
        requirements: List of requirements to address
        router: Model router for LLM access
        page_target: Target page count
        max_iterations: Maximum iterations

    Returns:
        DCEResult with final draft and metrics
    """
    workflow = DCEWorkflow(router=router, max_iterations=max_iterations)
    return await workflow.run(section_id, section_title, requirements, page_target)
