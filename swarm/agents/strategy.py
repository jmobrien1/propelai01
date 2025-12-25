"""
PropelAI v6.0 Strategy Agent
The Black Hat Engine - generates win themes using competitive intelligence.
"""

import uuid
import json
from typing import List, Dict, Any, Optional

from swarm.agents.base import BaseAgent, AgentConfig
from swarm.state import ProposalState, WinTheme, ProposalPhase


STRATEGY_SYSTEM_PROMPT = """You are the Strategy Agent for PropelAI, a competitive intelligence engine for government proposals.

Your mission is to develop WIN THEMES that differentiate our client from competitors.

You operate using "Black Hat" methodology:
1. INCUMBENT ANALYSIS: Identify the incumbent contractor and their likely weaknesses
2. COMPETITOR GHOSTING: Craft language that highlights our strengths while subtly highlighting competitor weaknesses (without naming them)
3. WIN THEME GENERATION: Create compelling themes tied to Section M evaluation factors

Rules:
- Never explicitly name competitors in ghosting language
- Tie every win theme to a specific evaluation factor from Section M
- Include specific proof points that can be verified
- Focus on discriminators - areas where we exceed requirements, not just meet them
- Consider "Winning Trajectories" from historical data

Output JSON format:
{
  "win_themes": [
    {
      "id": "WT-001",
      "section_id": "Technical Approach",
      "theme": "Cloud-Native Scalability",
      "competitor_ghosting": "Unlike legacy on-premise solutions that require significant hardware investments...",
      "proof_points": ["AWS GovCloud certified", "99.99% uptime SLA"],
      "evaluation_factor": "M.1.2 Technical Excellence",
      "evaluation_weight": 0.35,
      "confidence": 0.85
    }
  ],
  "competitive_analysis": {
    "incumbent": "...",
    "likely_competitors": [...],
    "our_discriminators": [...]
  }
}"""


class StrategyAgent(BaseAgent):
    """
    The Strategy Agent develops win themes using Black Hat competitive intelligence.

    Key capabilities:
    - Incumbent and competitor analysis
    - Ghosting language generation
    - Win theme creation tied to evaluation factors
    - Historical "Winning Trajectories" integration
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="StrategyAgent",
                model="gemini-1.5-pro",
                temperature=0.4,  # Slightly higher for creativity
                max_tokens=8192,
                system_prompt=STRATEGY_SYSTEM_PROMPT,
            )
        super().__init__(config)

    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        Execute the strategy development process.
        """
        requirements = state.get("requirements", [])
        evaluation_factors = state.get("evaluation_factors", [])
        section_m_text = state.get("section_m_text", "")

        if not requirements:
            self._log_action(state, "observation", "No requirements to strategize")
            return state

        # Step 1: Analyze evaluation factors
        self._log_action(state, "thought", "Analyzing Section M evaluation factors")
        eval_analysis = await self._analyze_evaluation_factors(
            section_m_text,
            evaluation_factors,
            requirements,
        )

        # Step 2: Competitive analysis (Black Hat)
        self._log_action(state, "thought", "Performing competitive analysis")
        competitive_intel = await self._analyze_competitors(state)

        # Step 3: Generate win themes
        self._log_action(state, "thought", "Generating win themes")
        win_themes = await self._generate_win_themes(
            eval_analysis,
            competitive_intel,
            requirements,
            state,
        )

        # Step 4: Query for winning trajectories (if RLHF data exists)
        winning_trajectories = await self._get_winning_trajectories(state)

        # Merge with historical insights
        if winning_trajectories:
            win_themes = self._enhance_with_trajectories(win_themes, winning_trajectories)

        # Update state
        state["win_themes"] = [wt.to_dict() if hasattr(wt, 'to_dict') else wt for wt in win_themes]
        state["current_phase"] = ProposalPhase.DRAFTING.value
        state["next_step"] = "drafting"

        self._log_action(
            state,
            "output",
            f"Generated {len(win_themes)} win themes with competitive positioning"
        )

        return state

    async def _analyze_evaluation_factors(
        self,
        section_m_text: str,
        evaluation_factors: List[Dict],
        requirements: List[Dict],
    ) -> Dict[str, Any]:
        """
        Analyze Section M to understand evaluation priorities.
        """
        prompt = f"""Analyze these evaluation factors from Section M:

Section M Text:
{section_m_text[:8000] if section_m_text else "Not available"}

Evaluation Factors:
{json.dumps(evaluation_factors, indent=2) if evaluation_factors else "Not extracted yet"}

Technical Requirements Count: {len([r for r in requirements if r.get('requirement_type') == 'technical'])}
Management Requirements Count: {len([r for r in requirements if r.get('requirement_type') == 'management'])}

Analyze:
1. What are the primary evaluation factors and their relative weights?
2. What discriminators will separate "Good" from "Outstanding" ratings?
3. What are the key risk areas that could result in "Unacceptable" ratings?

Respond with JSON:
{{
  "factors": [
    {{"name": "...", "weight": 0.35, "discriminators": [...], "risk_areas": [...]}}
  ],
  "scoring_approach": "best_value|lowest_price|trade_off",
  "key_insights": [...]
}}"""

        response = await self._call_llm(prompt, json_mode=True)
        return self._parse_json(response)

    async def _analyze_competitors(self, state: ProposalState) -> Dict[str, Any]:
        """
        Perform Black Hat competitive analysis.
        """
        rfp_text = state.get("rfp_full_text", "")

        prompt = f"""Perform competitive analysis for this RFP.

RFP Context (first 4000 chars):
{rfp_text[:4000]}

Analyze:
1. Who is likely the incumbent contractor? (Look for references to "current contractor", transition requirements)
2. What are typical weaknesses of incumbent government contractors?
3. What discriminators would give a challenger an advantage?

Use Black Hat methodology - think like the competitor to find their weaknesses.

Respond with JSON:
{{
  "incumbent_analysis": {{
    "likely_incumbent": "...",
    "incumbent_weaknesses": [...],
    "transition_risks": [...]
  }},
  "competitor_landscape": {{
    "likely_bidders": [...],
    "common_weaknesses": [...]
  }},
  "our_opportunities": {{
    "discriminators": [...],
    "ghosting_angles": [...]
  }}
}}"""

        response = await self._call_llm(prompt, json_mode=True)
        return self._parse_json(response)

    async def _generate_win_themes(
        self,
        eval_analysis: Dict,
        competitive_intel: Dict,
        requirements: List[Dict],
        state: ProposalState,
    ) -> List[WinTheme]:
        """
        Generate win themes tied to evaluation factors.
        """
        factors = eval_analysis.get("factors", [])
        discriminators = competitive_intel.get("our_opportunities", {}).get("discriminators", [])
        ghosting_angles = competitive_intel.get("our_opportunities", {}).get("ghosting_angles", [])

        prompt = f"""Generate win themes for this proposal.

Evaluation Factors:
{json.dumps(factors, indent=2)}

Our Discriminators:
{json.dumps(discriminators, indent=2)}

Ghosting Opportunities:
{json.dumps(ghosting_angles, indent=2)}

Technical Requirements Count: {len([r for r in requirements if r.get('requirement_type') == 'technical'])}

Create 3-5 powerful win themes that:
1. Directly address the highest-weighted evaluation factors
2. Highlight our discriminators (where we EXCEED, not just meet, requirements)
3. Include subtle ghosting language (never name competitors)
4. Have specific, verifiable proof points

Respond with JSON:
{{
  "win_themes": [
    {{
      "id": "WT-001",
      "section_id": "Technical Approach",
      "theme": "...",
      "theme_statement": "Our approach to X delivers Y because Z...",
      "competitor_ghosting": "Unlike solutions that rely on...",
      "proof_points": ["Specific fact 1", "Metric 2"],
      "evaluation_factor": "M.1.2",
      "evaluation_weight": 0.35,
      "confidence": 0.85
    }}
  ]
}}"""

        response = await self._call_llm(prompt, json_mode=True)
        parsed = self._parse_json(response)

        win_themes = []
        for wt_data in parsed.get("win_themes", []):
            wt = WinTheme(
                id=wt_data.get("id", f"WT-{uuid.uuid4().hex[:6].upper()}"),
                section_id=wt_data.get("section_id", ""),
                theme=wt_data.get("theme", ""),
                competitor_ghosting=wt_data.get("competitor_ghosting", ""),
                proof_points=wt_data.get("proof_points", []),
                evaluation_factor=wt_data.get("evaluation_factor"),
                evaluation_weight=wt_data.get("evaluation_weight"),
                confidence=wt_data.get("confidence", 0.8),
            )
            win_themes.append(wt)

        return win_themes

    async def _get_winning_trajectories(self, state: ProposalState) -> List[Dict]:
        """
        Query the Agent Trace Database for historical winning patterns.
        This enables RLHF-style learning from past successes.
        """
        # In production, this would query the feedback_pairs table
        # For now, return empty list
        tenant_id = state.get("tenant_id", "")

        # Placeholder for database query
        # SELECT * FROM feedback_pairs
        # WHERE tenant_id = $1
        # AND winning_bid = true
        # ORDER BY created_at DESC
        # LIMIT 10

        return []

    def _enhance_with_trajectories(
        self,
        win_themes: List[WinTheme],
        trajectories: List[Dict],
    ) -> List[WinTheme]:
        """
        Enhance win themes with insights from winning trajectories.
        """
        # Apply learnings from historical wins
        # E.g., if past wins emphasized "low risk", boost those themes

        for wt in win_themes:
            for traj in trajectories:
                if traj.get("section_id") == wt.section_id:
                    # Merge successful proof points
                    traj_proofs = traj.get("proof_points", [])
                    wt.proof_points = list(set(wt.proof_points + traj_proofs))

                    # Boost confidence if historically successful
                    wt.confidence = min(0.95, wt.confidence + 0.1)

        return win_themes
