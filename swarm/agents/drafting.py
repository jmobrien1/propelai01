"""
PropelAI v6.0 Drafting Agent
The F-B-P Enforcer - generates citation-backed proposal content.
"""

import uuid
import json
from typing import List, Dict, Any, Optional

from swarm.agents.base import BaseAgent, AgentConfig
from swarm.state import (
    ProposalState,
    DraftSection,
    ScoreColor,
    ProposalPhase,
)


DRAFTING_SYSTEM_PROMPT = """You are the Drafting Agent for PropelAI, a proposal narrative generator.

Your mission is to write COMPLIANT, PERSUASIVE proposal text using the Feature-Benefit-Proof (F-B-P) framework.

F-B-P Framework:
1. FEATURE: What we offer (our capability, approach, or solution)
2. BENEFIT: Why the client cares (mapped directly to Section M evaluation criteria)
3. PROOF: Evidence that we can deliver (specific citations, never hallucinated)

ZERO HALLUCINATION POLICY:
- You MUST NOT generate proof points from your internal knowledge
- Every claim requires a citation from the Research Agent
- If you cannot find a citation, mark the claim as [CITATION NEEDED] for human review
- Use format: "We have successfully delivered similar services (see Past Performance: Contract ABC-123)."

Writing Rules:
- Use active voice: "We will deploy..." not "Deployment will be..."
- Be specific: "Our team includes 3 CISSP-certified engineers" not "Our team is qualified"
- Match evaluator language: Mirror Section M terminology in your responses
- Respect page limits: Keep sections concise and impactful
- Address every requirement: Each paragraph should trace to a specific requirement ID

Output JSON format:
{
  "section_id": "technical-approach-3.1",
  "title": "Technical Approach",
  "content": "...",
  "citations": ["PP-001", "RES-003"],
  "requirement_ids": ["REQ-042", "REQ-043"],
  "win_theme_id": "WT-001",
  "uncited_claims": ["Claim that needs verification..."]
}"""


class DraftingAgent(BaseAgent):
    """
    The Drafting Agent generates proposal content using F-B-P framework.

    Key capabilities:
    - Feature-Benefit-Proof structured writing
    - Zero hallucination with citation requirements
    - Win theme integration
    - Page limit awareness
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="DraftingAgent",
                model="gemini-1.5-pro",
                temperature=0.3,
                max_tokens=8192,
                system_prompt=DRAFTING_SYSTEM_PROMPT,
            )
        super().__init__(config)

    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        Execute the drafting process for all sections.
        """
        requirements = state.get("requirements", [])
        win_themes = state.get("win_themes", [])
        evidence_pool = state.get("evidence_pool", [])

        if not requirements:
            self._log_action(state, "observation", "No requirements to draft against")
            return state

        # Group requirements by section
        sections_to_draft = self._organize_sections(requirements, win_themes)

        # Draft each section
        draft_sections = {}
        for section_config in sections_to_draft:
            self._log_action(
                state,
                "thought",
                f"Drafting section: {section_config['title']}"
            )

            draft = await self._draft_section(
                section_config,
                evidence_pool,
                state,
            )

            draft_sections[draft.section_id] = draft.to_dict()

        # Update state
        state["draft_sections"] = draft_sections
        state["current_phase"] = ProposalPhase.REVIEW.value
        state["next_step"] = "red_team"

        self._log_action(
            state,
            "output",
            f"Drafted {len(draft_sections)} sections"
        )

        return state

    def _organize_sections(
        self,
        requirements: List[Dict],
        win_themes: List[Dict],
    ) -> List[Dict]:
        """
        Organize requirements into logical proposal sections.
        """
        # Group by requirement type
        technical_reqs = [r for r in requirements if r.get("requirement_type") == "technical"]
        management_reqs = [r for r in requirements if r.get("requirement_type") == "management"]
        pp_reqs = [r for r in requirements if r.get("requirement_type") == "past_performance"]

        sections = []

        # Technical Approach section
        if technical_reqs:
            # Find matching win theme
            tech_theme = next(
                (wt for wt in win_themes if "technical" in wt.get("section_id", "").lower()),
                None
            )

            sections.append({
                "section_id": "technical-approach",
                "title": "Technical Approach",
                "requirements": technical_reqs,
                "win_theme": tech_theme,
                "page_limit": None,  # Would come from Section L
            })

        # Management Approach section
        if management_reqs:
            mgmt_theme = next(
                (wt for wt in win_themes if "management" in wt.get("section_id", "").lower()),
                None
            )

            sections.append({
                "section_id": "management-approach",
                "title": "Management Approach",
                "requirements": management_reqs,
                "win_theme": mgmt_theme,
                "page_limit": None,
            })

        # Past Performance section
        if pp_reqs:
            sections.append({
                "section_id": "past-performance",
                "title": "Past Performance",
                "requirements": pp_reqs,
                "win_theme": None,
                "page_limit": None,
            })

        return sections

    async def _draft_section(
        self,
        section_config: Dict,
        evidence_pool: List[Dict],
        state: ProposalState,
    ) -> DraftSection:
        """
        Draft a single proposal section.
        """
        section_id = section_config["section_id"]
        title = section_config["title"]
        requirements = section_config["requirements"]
        win_theme = section_config.get("win_theme")
        page_limit = section_config.get("page_limit")

        # Build prompt
        prompt = self._build_drafting_prompt(
            title=title,
            requirements=requirements,
            win_theme=win_theme,
            evidence_pool=evidence_pool,
            page_limit=page_limit,
        )

        response = await self._call_llm(prompt, json_mode=True)
        parsed = self._parse_json(response)

        # Extract uncited claims for review
        content = parsed.get("content", "")
        uncited = self._find_uncited_claims(content)

        if uncited:
            self._log_action(
                state,
                "observation",
                f"Found {len(uncited)} uncited claims in {title}"
            )

        return DraftSection(
            section_id=section_id,
            title=title,
            content=content,
            version=1,
            quality_score=0.0,  # Will be set by Red Team
            score_color=ScoreColor.YELLOW,
            citations=parsed.get("citations", []),
            evidence_ids=parsed.get("evidence_ids", []),
            feedback_log=[],
            requirement_ids=[r.get("id", "") for r in requirements],
            win_theme_id=win_theme.get("id") if win_theme else None,
            page_limit=page_limit,
        )

    def _build_drafting_prompt(
        self,
        title: str,
        requirements: List[Dict],
        win_theme: Optional[Dict],
        evidence_pool: List[Dict],
        page_limit: Optional[int],
    ) -> str:
        """
        Build the drafting prompt for a section.
        """
        req_summary = "\n".join([
            f"- [{r.get('id')}] {r.get('text', '')[:200]}..."
            for r in requirements[:10]
        ])

        theme_text = ""
        if win_theme:
            theme_text = f"""
Win Theme to Integrate:
- Theme: {win_theme.get('theme', '')}
- Ghosting: {win_theme.get('competitor_ghosting', '')}
- Proof Points: {json.dumps(win_theme.get('proof_points', []))}
"""

        evidence_text = ""
        if evidence_pool:
            evidence_text = "\n".join([
                f"- [{e.get('id')}] {e.get('source_name')}: {e.get('snippet_text', '')[:100]}..."
                for e in evidence_pool[:5]
            ])

        page_guidance = ""
        if page_limit:
            page_guidance = f"\nPage Limit: {page_limit} pages (~{page_limit * 500} words)"

        return f"""Draft the "{title}" section of the proposal.

Requirements to Address:
{req_summary}
{theme_text}

Available Evidence for Citations:
{evidence_text}
{page_guidance}

Write using the Feature-Benefit-Proof framework:
1. State our FEATURE (capability/approach)
2. Connect to CLIENT BENEFIT (tied to evaluation criteria)
3. Provide PROOF (specific citations only - no hallucination)

For any claim without available evidence, use [CITATION NEEDED].

Respond with JSON:
{{
  "content": "The full section narrative...",
  "citations": ["evidence-id-1", "evidence-id-2"],
  "evidence_ids": ["EV-001", "EV-002"],
  "requirement_ids": ["REQ-042", "REQ-043"],
  "uncited_claims": ["Any claims that need verification"]
}}"""

    def _find_uncited_claims(self, content: str) -> List[str]:
        """
        Find claims in the content that lack citations.
        """
        import re

        uncited = []

        # Find explicit markers
        markers = re.findall(r'\[CITATION NEEDED\][^.]*\.', content)
        uncited.extend(markers)

        # Find quantitative claims without citations
        quant_patterns = [
            r'\d+%[^.]*\.',
            r'\$\d+[^.]*\.',
            r'\d+\s+years?[^.]*\.',
        ]

        for pattern in quant_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Check if it has a citation nearby
                if "(" not in match and "[" not in match:
                    uncited.append(match)

        return uncited[:10]  # Limit to first 10
