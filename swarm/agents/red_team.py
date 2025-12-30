"""
PropelAI v6.0 Red Team Agent
The Synthetic Evaluator - scores drafts against Section M criteria.
"""

import json
from typing import List, Dict, Any, Optional

from swarm.agents.base import BaseAgent, AgentConfig
from swarm.state import (
    ProposalState,
    ScoreColor,
    ProposalPhase,
)


RED_TEAM_SYSTEM_PROMPT = """You are the Red Team Agent for PropelAI, a synthetic government evaluator.

Your mission is to SCORE proposal drafts against Section M evaluation criteria using DoD Source Selection methodology.

Scoring Rubric (DoD Standard):
- BLUE (Outstanding): Exceeds requirements with significant strengths, minimal weaknesses
- GREEN (Good): Meets requirements, some strengths, acceptable weaknesses
- YELLOW (Marginal): Meets minimum requirements but has significant weaknesses/risks
- RED (Unacceptable): Fails to meet requirements, has deficiencies

EVALUATOR MINDSET:
- You are skeptical - claims without evidence are weaknesses
- You look for discriminators - what makes this proposal stand out?
- You identify risks - areas where the offeror may struggle to perform
- You check compliance - does this address every requirement?

For each section, provide:
1. SCORE: Color rating with justification
2. STRENGTHS: What exceeds requirements
3. WEAKNESSES: What needs improvement
4. RISKS: Performance concerns
5. ACTIONABLE FEEDBACK: Specific improvements (not generic)

Output JSON format:
{
  "section_id": "technical-approach",
  "score_color": "yellow",
  "score_numeric": 0.65,
  "evaluation": {
    "strengths": [
      {"description": "...", "requirement_id": "REQ-042", "discriminator": true}
    ],
    "weaknesses": [
      {"description": "...", "requirement_id": "REQ-043", "severity": "significant"}
    ],
    "risks": [
      {"description": "...", "likelihood": "medium", "impact": "high"}
    ],
    "compliance_gaps": [
      {"requirement_id": "REQ-044", "issue": "Not addressed"}
    ]
  },
  "actionable_feedback": [
    "Add specific metrics to support the scalability claim in paragraph 3",
    "Include a citation for the 99.9% uptime claim"
  ],
  "recommendation": "revise|accept|escalate"
}"""


class RedTeamAgent(BaseAgent):
    """
    The Red Team Agent scores drafts using government evaluation methodology.

    Key capabilities:
    - DoD Source Selection scoring
    - Compliance gap identification
    - Strength/weakness analysis
    - Actionable feedback generation
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="RedTeamAgent",
                model="gemini-1.5-pro",
                temperature=0.2,  # Low for consistent evaluation
                max_tokens=8192,
                system_prompt=RED_TEAM_SYSTEM_PROMPT,
            )
        super().__init__(config)

    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        Execute Red Team evaluation on all draft sections.
        """
        draft_sections = state.get("draft_sections", {})
        requirements = state.get("requirements", [])
        evaluation_factors = state.get("evaluation_factors", [])

        if not draft_sections:
            self._log_action(state, "observation", "No drafts to evaluate")
            return state

        section_scores = {}
        all_pass = True
        revisions_needed = []

        for section_id, draft in draft_sections.items():
            self._log_action(
                state,
                "thought",
                f"Evaluating section: {section_id}"
            )

            # Get relevant requirements for this section
            section_reqs = [
                r for r in requirements
                if r.get("id") in draft.get("requirement_ids", [])
            ]

            # Evaluate the section
            evaluation = await self._evaluate_section(
                draft,
                section_reqs,
                evaluation_factors,
            )

            # Update draft with evaluation results
            draft["quality_score"] = evaluation.get("score_numeric", 0.0)
            draft["score_color"] = evaluation.get("score_color", "yellow")
            draft["feedback_log"] = evaluation.get("actionable_feedback", [])

            section_scores[section_id] = evaluation.get("score_numeric", 0.0)

            # Check if revision needed
            recommendation = evaluation.get("recommendation", "revise")
            if recommendation == "revise":
                all_pass = False
                revisions_needed.append({
                    "section_id": section_id,
                    "feedback": evaluation.get("actionable_feedback", []),
                    "score_color": evaluation.get("score_color"),
                })

            self._log_action(
                state,
                "observation",
                f"Section {section_id}: {evaluation.get('score_color')} ({evaluation.get('score_numeric', 0):.2f})"
            )

        # Update state with scores
        state["draft_sections"] = draft_sections
        state["section_scores"] = section_scores
        state["overall_score"] = sum(section_scores.values()) / len(section_scores) if section_scores else 0.0

        # Determine next step
        if all_pass:
            state["current_phase"] = ProposalPhase.FINALIZE.value
            state["next_step"] = "finalize"
        else:
            # Route back to drafting for revision
            state["next_step"] = "drafting"
            # Add revision instructions to supervisor scratchpad
            state["supervisor_scratchpad"] = state.get("supervisor_scratchpad", []) + [
                f"Revisions needed: {json.dumps(revisions_needed)}"
            ]

        self._log_action(
            state,
            "output",
            f"Evaluation complete. Overall score: {state['overall_score']:.2f}. "
            f"{'All sections pass.' if all_pass else f'{len(revisions_needed)} sections need revision.'}"
        )

        return state

    async def _evaluate_section(
        self,
        draft: Dict,
        requirements: List[Dict],
        evaluation_factors: List[Dict],
    ) -> Dict[str, Any]:
        """
        Evaluate a single section against requirements and evaluation criteria.
        """
        content = draft.get("content", "")
        title = draft.get("title", "")

        prompt = f"""Evaluate this proposal section as a government Source Selection evaluator.

Section Title: {title}

Section Content:
{content[:8000]}

Requirements This Section Must Address:
{json.dumps([{"id": r.get("id"), "text": r.get("text", "")[:200]} for r in requirements[:10]], indent=2)}

Evaluation Factors (from Section M):
{json.dumps(evaluation_factors[:5], indent=2) if evaluation_factors else "Not provided"}

Score this section using the DoD rubric:
- BLUE (0.85-1.0): Exceeds with significant strengths
- GREEN (0.70-0.84): Meets requirements, some strengths
- YELLOW (0.50-0.69): Marginal, significant weaknesses
- RED (0.0-0.49): Fails to meet requirements

Provide detailed evaluation as JSON:
{{
  "score_color": "blue|green|yellow|red",
  "score_numeric": 0.0-1.0,
  "evaluation": {{
    "strengths": [
      {{"description": "...", "requirement_id": "REQ-XXX", "discriminator": true/false}}
    ],
    "weaknesses": [
      {{"description": "...", "requirement_id": "REQ-XXX", "severity": "minor|significant|major"}}
    ],
    "risks": [
      {{"description": "...", "likelihood": "low|medium|high", "impact": "low|medium|high"}}
    ],
    "compliance_gaps": [
      {{"requirement_id": "REQ-XXX", "issue": "Not addressed|Partially addressed|Unclear"}}
    ]
  }},
  "actionable_feedback": [
    "Specific action to improve the section..."
  ],
  "recommendation": "accept|revise|escalate"
}}"""

        response = await self._call_llm(prompt, json_mode=True)
        return self._parse_json(response)

    def _map_score_to_color(self, score: float) -> ScoreColor:
        """Map numeric score to color rating."""
        if score >= 0.85:
            return ScoreColor.BLUE
        elif score >= 0.70:
            return ScoreColor.GREEN
        elif score >= 0.50:
            return ScoreColor.YELLOW
        else:
            return ScoreColor.RED

    async def evaluate_full_proposal(
        self,
        state: ProposalState,
    ) -> Dict[str, Any]:
        """
        Provide a holistic evaluation of the complete proposal.
        """
        draft_sections = state.get("draft_sections", {})
        section_scores = state.get("section_scores", {})

        # Calculate weighted overall score
        # In production, weights would come from Section M
        total_weight = len(section_scores)
        weighted_sum = sum(section_scores.values())
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "overall_score": overall,
            "overall_color": self._map_score_to_color(overall).value,
            "section_summary": {
                sid: {
                    "score": score,
                    "color": self._map_score_to_color(score).value
                }
                for sid, score in section_scores.items()
            },
            "ready_for_submission": overall >= 0.70,
        }
