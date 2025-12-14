"""
Claude Client - The Architect & The Critic

Implements Claude 3.5 Sonnet for:
- ARCHITECT role: Section planning, outline generation, orchestration
- CRITIC role: Blue Team critique, compliance verification, draft review

From Long-Form Generation Strategy:
"Claude excels at structured reasoning and critique. Use it to:
 - Plan section hierarchies and dependencies
 - Perform adversarial review (what would a government evaluator say?)
 - Identify gaps against evaluation criteria"
"""

import os
import time
import logging
import json
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

import anthropic
from anthropic import AsyncAnthropic

from .llm_clients import (
    BaseLLMClient,
    ModelRole,
    TaskType,
    TokenUsage,
    LLMMessage,
    LLMResponse,
    GenerationConfig,
    calculate_cost,
)

logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result of a Blue Team critique"""
    overall_score: float  # 0-100 compliance score
    strengths: List[str]
    weaknesses: List[str]
    gaps: List[str]  # Missing requirements
    recommendations: List[str]
    evaluator_perspective: str  # What would a govt evaluator think?
    expand_required: bool  # Does draft need expansion?
    raw_critique: str


@dataclass
class SectionPlan:
    """Plan for a proposal section"""
    section_id: str
    section_title: str
    page_allocation: float
    key_themes: List[str]
    requirements_addressed: List[str]
    dependencies: List[str]  # Other sections this depends on
    win_themes: List[str]
    discriminators: List[str]
    risks: List[str]


@dataclass
class OutlinePlanResult:
    """Result of outline generation"""
    sections: List[SectionPlan]
    total_pages: float
    coverage_analysis: Dict[str, Any]
    recommended_sequence: List[str]  # Writing order


class ClaudeClient(BaseLLMClient):
    """
    Claude 3.5 Sonnet client for Architect and Critic roles.

    Architect Mode:
    - Section hierarchy planning
    - Outline generation with requirement mapping
    - Criteria extraction and structuring
    - Orchestration decisions

    Critic Mode:
    - Blue Team adversarial review
    - Draft quality assessment
    - Compliance gap identification
    - Expansion recommendations
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    OPUS_MODEL = "claude-3-opus-20240229"

    def __init__(
        self,
        role: ModelRole = ModelRole.ARCHITECT,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        super().__init__(role)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self._model = model or self.DEFAULT_MODEL
        self._client = AsyncAnthropic(api_key=self.api_key)

        logger.info(f"Initialized Claude client as {role.value} with model {self._model}")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_context_tokens(self) -> int:
        return 200_000  # Claude's context window

    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> LLMResponse:
        """Generate a completion using Claude."""
        config = config or GenerationConfig()
        start_time = time.time()

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=config.max_output_tokens,
                temperature=config.temperature,
                system=system_message or self._get_system_prompt(task_type),
                messages=anthropic_messages,
                stop_sequences=config.stop_sequences if config.stop_sequences else None,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            # Build token usage
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                estimated_cost_usd=calculate_cost(
                    self._model,
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
            )

            self._track_usage(usage)

            return LLMResponse(
                content=content,
                model=self._model,
                role=self.role,
                task_type=task_type,
                token_usage=usage,
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency_ms,
                raw_response={"id": response.id, "model": response.model}
            )

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[GenerationConfig] = None,
        task_type: Optional[TaskType] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming completion using Claude."""
        config = config or GenerationConfig()

        # Convert messages
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=config.max_output_tokens,
            temperature=config.temperature,
            system=system_message or self._get_system_prompt(task_type),
            messages=anthropic_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using Anthropic's tokenizer."""
        # Anthropic doesn't expose a public tokenizer, so we estimate
        # Average ~4 characters per token for English text
        return len(text) // 4

    # =====================================================================
    # ARCHITECT METHODS
    # =====================================================================

    async def generate_section_plan(
        self,
        rfp_content: str,
        evaluation_criteria: List[Dict[str, Any]],
        page_limit: int,
        section_l_instructions: str
    ) -> OutlinePlanResult:
        """
        Generate a comprehensive section plan for a proposal.

        This is the Architect's primary function - creating the strategic
        blueprint for the proposal structure.
        """
        system_prompt = """You are an expert government proposal architect with 20+ years
of experience winning federal contracts. Your task is to create a strategic section
plan that maximizes evaluation scores.

You must:
1. Map every evaluation criterion to specific sections
2. Allocate pages strategically (more pages for higher-weighted criteria)
3. Identify win themes and discriminators for each section
4. Determine optimal writing sequence (dependencies)
5. Anticipate evaluator concerns

Output your plan as valid JSON matching the requested schema."""

        user_prompt = f"""Create a section plan for this proposal:

=== SECTION L INSTRUCTIONS ===
{section_l_instructions}

=== EVALUATION CRITERIA (Section M) ===
{json.dumps(evaluation_criteria, indent=2)}

=== PAGE LIMIT ===
{page_limit} pages total

=== RFP CONTENT (for context) ===
{rfp_content[:50000]}  # Truncate if needed

Provide your section plan as JSON with this structure:
{{
    "sections": [
        {{
            "section_id": "1.0",
            "section_title": "Technical Approach",
            "page_allocation": 15.0,
            "key_themes": ["theme1", "theme2"],
            "requirements_addressed": ["L.4.1.a", "L.4.1.b"],
            "dependencies": [],
            "win_themes": ["Our unique advantage..."],
            "discriminators": ["What sets us apart..."],
            "risks": ["Potential weaknesses..."]
        }}
    ],
    "total_pages": 50,
    "coverage_analysis": {{
        "fully_covered": ["criteria1", "criteria2"],
        "partially_covered": ["criteria3"],
        "gaps": []
    }},
    "recommended_sequence": ["1.0", "2.0", "3.0"]
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = await self.generate(
            messages,
            GenerationConfig(temperature=0.3, max_output_tokens=8192),
            TaskType.SECTION_PLANNING
        )

        # Parse the JSON response
        try:
            # Extract JSON from response (may be wrapped in markdown)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            plan_data = json.loads(content)

            sections = [
                SectionPlan(
                    section_id=s["section_id"],
                    section_title=s["section_title"],
                    page_allocation=s["page_allocation"],
                    key_themes=s.get("key_themes", []),
                    requirements_addressed=s.get("requirements_addressed", []),
                    dependencies=s.get("dependencies", []),
                    win_themes=s.get("win_themes", []),
                    discriminators=s.get("discriminators", []),
                    risks=s.get("risks", [])
                )
                for s in plan_data.get("sections", [])
            ]

            return OutlinePlanResult(
                sections=sections,
                total_pages=plan_data.get("total_pages", page_limit),
                coverage_analysis=plan_data.get("coverage_analysis", {}),
                recommended_sequence=plan_data.get("recommended_sequence", [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse section plan JSON: {e}")
            logger.debug(f"Raw response: {response.content}")
            raise ValueError(f"Invalid section plan format: {e}")

    async def extract_evaluation_criteria(
        self,
        section_m_content: str
    ) -> List[Dict[str, Any]]:
        """
        Extract structured evaluation criteria from Section M.

        Returns criteria with:
        - ID, title, description
        - Weight (if specified)
        - Subfactors
        - Rating scale
        """
        system_prompt = """You are an expert at parsing government RFP evaluation criteria.
Extract ALL evaluation factors and subfactors from Section M content.
Be thorough - missing criteria means lost points.

Output as JSON array."""

        user_prompt = f"""Extract all evaluation criteria from this Section M:

{section_m_content}

Output format:
[
    {{
        "factor_id": "M.1",
        "factor_name": "Technical Approach",
        "weight": "Most Important",
        "description": "...",
        "subfactors": [
            {{
                "subfactor_id": "M.1.a",
                "name": "Understanding of Requirements",
                "weight": null,
                "rating_scale": ["Outstanding", "Good", "Acceptable", "Marginal", "Unacceptable"]
            }}
        ]
    }}
]"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = await self.generate(
            messages,
            GenerationConfig(temperature=0.1, max_output_tokens=8192),
            TaskType.CRITERIA_EXTRACTION
        )

        # Parse JSON
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)

    # =====================================================================
    # CRITIC METHODS (Blue Team)
    # =====================================================================

    async def critique_draft(
        self,
        draft_content: str,
        section_requirements: List[str],
        evaluation_criteria: List[Dict[str, Any]],
        page_target: float
    ) -> CritiqueResult:
        """
        Perform Blue Team critique of a draft section.

        This is adversarial review - channeling a skeptical government evaluator
        who is looking for reasons to score lower.
        """
        system_prompt = """You are a demanding government Source Selection Evaluation Board (SSEB)
member reviewing a proposal. You are skeptical by nature and looking for weaknesses.

Your critique must:
1. Identify what's MISSING (requirements not addressed)
2. Find vague or unsupported claims
3. Check for compliance with instructions
4. Assess from an evaluator's scoring perspective
5. Determine if content needs EXPANSION (more detail, evidence, specifics)

Be harsh but constructive. Rate on a 0-100 scale where:
- 90-100: Outstanding (exceeds requirements)
- 80-89: Good (fully meets requirements)
- 70-79: Acceptable (meets minimum)
- 60-69: Marginal (partially meets)
- 0-59: Unacceptable (does not meet)

Output as JSON."""

        user_prompt = f"""Critique this draft section:

=== DRAFT CONTENT ===
{draft_content}

=== REQUIREMENTS TO ADDRESS ===
{json.dumps(section_requirements, indent=2)}

=== EVALUATION CRITERIA ===
{json.dumps(evaluation_criteria, indent=2)}

=== PAGE TARGET ===
{page_target} pages (current: ~{len(draft_content) / 3000:.1f} pages)

Provide your critique as JSON:
{{
    "overall_score": 75,
    "strengths": ["Clear technical approach", "Good use of graphics"],
    "weaknesses": ["Lacks specific metrics", "No past performance examples"],
    "gaps": ["Requirement L.4.1.c not addressed", "Missing risk mitigation"],
    "recommendations": ["Add quantified benefits", "Include case study"],
    "evaluator_perspective": "An evaluator would likely rate this...",
    "expand_required": true
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = await self.generate(
            messages,
            GenerationConfig(temperature=0.4, max_output_tokens=4096),
            TaskType.DRAFT_CRITIQUE
        )

        # Parse JSON
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        critique_data = json.loads(content)

        return CritiqueResult(
            overall_score=critique_data.get("overall_score", 0),
            strengths=critique_data.get("strengths", []),
            weaknesses=critique_data.get("weaknesses", []),
            gaps=critique_data.get("gaps", []),
            recommendations=critique_data.get("recommendations", []),
            evaluator_perspective=critique_data.get("evaluator_perspective", ""),
            expand_required=critique_data.get("expand_required", False),
            raw_critique=response.content
        )

    async def verify_compliance(
        self,
        content: str,
        p0_constraints: List[Dict[str, Any]],
        section_l_requirements: List[str]
    ) -> Dict[str, Any]:
        """
        Verify compliance with P0 (pass/fail) constraints and Section L requirements.

        P0 violations are fatal - they can disqualify a proposal.
        """
        system_prompt = """You are a compliance verification specialist checking
proposal content against mandatory requirements. P0 constraints are PASS/FAIL -
any violation can disqualify the entire proposal.

Be extremely thorough. Check:
1. Page limits
2. Font requirements
3. Margin requirements
4. Required sections present
5. Format compliance
6. Mandatory content elements

Output as JSON."""

        user_prompt = f"""Verify this content against requirements:

=== CONTENT TO CHECK ===
{content[:30000]}  # Truncate for token limits

=== P0 CONSTRAINTS (PASS/FAIL) ===
{json.dumps(p0_constraints, indent=2)}

=== SECTION L REQUIREMENTS ===
{json.dumps(section_l_requirements, indent=2)}

Output:
{{
    "compliant": true/false,
    "p0_status": {{
        "all_passed": true/false,
        "violations": [
            {{"constraint": "Page Limit", "status": "FAIL", "details": "52 pages vs 50 limit"}}
        ]
    }},
    "section_l_status": {{
        "requirements_met": ["L.4.1.a", "L.4.1.b"],
        "requirements_missing": ["L.4.1.c"],
        "requirements_partial": ["L.4.2"]
    }},
    "critical_issues": ["List of must-fix items"],
    "warnings": ["List of should-fix items"]
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = await self.generate(
            messages,
            GenerationConfig(temperature=0.1, max_output_tokens=4096),
            TaskType.COMPLIANCE_VERIFICATION
        )

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)

    async def generate_expansion_guidance(
        self,
        draft_content: str,
        critique: CritiqueResult,
        target_expansion: float  # e.g., 1.5 for 50% more content
    ) -> Dict[str, Any]:
        """
        Generate specific guidance for expanding a draft.

        This addresses the "brevity bias" problem - giving the Writer
        specific instructions on what to add and where.
        """
        system_prompt = """You are a proposal expansion strategist. Based on critique
feedback, you provide SPECIFIC, ACTIONABLE guidance for expanding content.

Your guidance must:
1. Identify exact locations needing expansion
2. Specify what type of content to add (examples, metrics, details)
3. Provide word count targets per section
4. Maintain strategic focus on evaluation criteria

Be specific - vague guidance leads to vague content."""

        user_prompt = f"""Generate expansion guidance:

=== CURRENT DRAFT ===
{draft_content}

=== CRITIQUE RESULTS ===
Score: {critique.overall_score}
Weaknesses: {json.dumps(critique.weaknesses)}
Gaps: {json.dumps(critique.gaps)}
Recommendations: {json.dumps(critique.recommendations)}

=== TARGET ===
Expand content by {int((target_expansion - 1) * 100)}%
Current length: ~{len(draft_content)} characters
Target length: ~{int(len(draft_content) * target_expansion)} characters

Output:
{{
    "expansion_zones": [
        {{
            "location": "After paragraph about technical approach",
            "content_type": "specific example",
            "suggested_addition": "Add case study from similar project showing...",
            "target_words": 200
        }}
    ],
    "new_sections_needed": [
        {{
            "section_title": "Risk Mitigation Strategy",
            "rationale": "Required by L.4.2 but missing",
            "suggested_content": "...",
            "target_words": 300
        }}
    ],
    "enhancement_priorities": [
        "Add quantified metrics to all claims",
        "Include staff qualifications for key personnel"
    ]
}}"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = await self.generate(
            messages,
            GenerationConfig(temperature=0.5, max_output_tokens=4096),
            TaskType.BLUE_TEAM_REVIEW
        )

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)

    def _get_system_prompt(self, task_type: Optional[TaskType]) -> str:
        """Get role-appropriate system prompt."""
        if self.role == ModelRole.ARCHITECT:
            return """You are the Architect - a strategic proposal planner with expertise in
government contracting. You design winning proposal structures that maximize evaluation scores."""

        elif self.role == ModelRole.CRITIC:
            return """You are the Critic - a Blue Team reviewer who channels skeptical government
evaluators. You find weaknesses, gaps, and areas needing improvement. Be demanding but constructive."""

        else:
            return """You are Claude, an AI assistant specialized in government proposal development."""
