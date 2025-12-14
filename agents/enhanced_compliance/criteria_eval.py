"""
Criteria-Eval Framework - Objective Completeness Scoring

From Long-Form Generation Strategy:
"The key innovation is Criteria-Eval: extract a JSON checklist of ALL
requirements from Section L/M, then use an LLM-as-Judge to assess
whether the draft addresses each item. This gives us an objective
completeness score (0-100%) that drives the Draft-Critique-Expand loop."

Components:
1. Criteria extraction from Section L/M
2. Checklist generation (JSON format)
3. LLM-as-Judge evaluation
4. Gap analysis and scoring
5. Loop termination conditions
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ComplianceStatus(str, Enum):
    """Status of individual criterion compliance"""
    FULLY_MET = "fully_met"         # Requirement completely addressed
    PARTIALLY_MET = "partially_met"  # Some aspects addressed
    NOT_MET = "not_met"             # Not addressed
    NOT_APPLICABLE = "not_applicable"  # Doesn't apply to this section
    UNCLEAR = "unclear"             # Couldn't determine


class CriterionPriority(str, Enum):
    """Priority level of criterion"""
    CRITICAL = "critical"   # P0 - Must have, failure = disqualification
    HIGH = "high"           # P1 - Strongly affects score
    MEDIUM = "medium"       # P2 - Affects score
    LOW = "low"             # P3 - Nice to have


@dataclass
class EvaluationCriterion:
    """
    A single evaluation criterion extracted from Section L/M.

    This represents one item the evaluator will check for.
    """
    id: str
    text: str
    source_section: str  # L.4.1, M.2.a, etc.
    priority: CriterionPriority = CriterionPriority.MEDIUM
    weight: float = 1.0  # Relative importance

    # Derived attributes
    keywords: List[str] = field(default_factory=list)
    related_criteria: List[str] = field(default_factory=list)

    # Evaluation result
    status: ComplianceStatus = ComplianceStatus.NOT_MET
    score: float = 0.0  # 0.0 to 1.0
    evidence: str = ""  # Quote from draft showing compliance
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source_section": self.source_section,
            "priority": self.priority.value,
            "weight": self.weight,
            "status": self.status.value,
            "score": self.score,
            "evidence": self.evidence,
            "gaps": self.gaps
        }


@dataclass
class CriteriaChecklist:
    """
    Complete checklist of criteria for a proposal section.

    This is the JSON structure used for evaluation.
    """
    section_id: str
    section_title: str
    criteria: List[EvaluationCriterion] = field(default_factory=list)

    # Metadata
    extracted_from: List[str] = field(default_factory=list)  # Source documents
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1

    @property
    def total_criteria(self) -> int:
        return len(self.criteria)

    @property
    def critical_criteria(self) -> List[EvaluationCriterion]:
        return [c for c in self.criteria if c.priority == CriterionPriority.CRITICAL]

    def to_json(self) -> str:
        return json.dumps({
            "section_id": self.section_id,
            "section_title": self.section_title,
            "criteria": [c.to_dict() for c in self.criteria],
            "metadata": {
                "total_criteria": self.total_criteria,
                "critical_count": len(self.critical_criteria),
                "extracted_from": self.extracted_from,
                "created_at": self.created_at,
                "version": self.version
            }
        }, indent=2)


@dataclass
class EvaluationResult:
    """
    Result of evaluating a draft against criteria.
    """
    section_id: str
    draft_version: int

    # Scores
    overall_score: float  # 0-100
    weighted_score: float  # 0-100, considering weights
    critical_score: float  # 0-100, just critical items

    # Breakdown
    criteria_met: int
    criteria_partial: int
    criteria_not_met: int
    criteria_total: int

    # Details
    criteria_results: List[EvaluationCriterion] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Loop control
    passes_threshold: bool = False
    expansion_required: bool = True
    estimated_expansion_needed: float = 0.0  # e.g., 1.5 = 50% more content

    # Timing
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "draft_version": self.draft_version,
            "scores": {
                "overall": self.overall_score,
                "weighted": self.weighted_score,
                "critical": self.critical_score
            },
            "breakdown": {
                "met": self.criteria_met,
                "partial": self.criteria_partial,
                "not_met": self.criteria_not_met,
                "total": self.criteria_total
            },
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "loop_control": {
                "passes_threshold": self.passes_threshold,
                "expansion_required": self.expansion_required,
                "estimated_expansion_needed": self.estimated_expansion_needed
            },
            "evaluated_at": self.evaluated_at
        }


class CriteriaEvaluator:
    """
    Evaluates proposal drafts against extracted criteria.

    This is the "LLM-as-Judge" component that provides objective
    completeness scoring for the Draft-Critique-Expand loop.

    Usage:
        evaluator = CriteriaEvaluator(llm_client)
        checklist = evaluator.extract_criteria(section_l, section_m)
        result = await evaluator.evaluate(draft, checklist)

        while not result.passes_threshold:
            draft = expand_draft(draft, result.gaps)
            result = await evaluator.evaluate(draft, checklist)
    """

    # Scoring thresholds
    PASS_THRESHOLD = 80.0          # Minimum overall score to pass
    CRITICAL_PASS_THRESHOLD = 95.0  # Minimum critical score to pass
    MAX_EXPANSION_FACTOR = 2.0      # Don't expand more than 2x
    MIN_EXPANSION_FACTOR = 1.1      # At least 10% expansion when needed

    # Status to score mapping
    STATUS_SCORES = {
        ComplianceStatus.FULLY_MET: 1.0,
        ComplianceStatus.PARTIALLY_MET: 0.5,
        ComplianceStatus.NOT_MET: 0.0,
        ComplianceStatus.NOT_APPLICABLE: 1.0,  # N/A doesn't hurt score
        ComplianceStatus.UNCLEAR: 0.25
    }

    def __init__(
        self,
        llm_client: Any = None,
        pass_threshold: float = PASS_THRESHOLD,
        critical_threshold: float = CRITICAL_PASS_THRESHOLD
    ):
        """
        Initialize the evaluator.

        Args:
            llm_client: LLM client for extraction and evaluation
            pass_threshold: Minimum score to pass evaluation
            critical_threshold: Minimum score for critical items
        """
        self.llm_client = llm_client
        self.pass_threshold = pass_threshold
        self.critical_threshold = critical_threshold

    def extract_criteria_from_text(
        self,
        section_l_text: str,
        section_m_text: str,
        section_id: str,
        section_title: str
    ) -> CriteriaChecklist:
        """
        Extract evaluation criteria from Section L and M text.

        Uses regex patterns and heuristics when LLM is not available.

        Args:
            section_l_text: Instructions to Offerors (Section L)
            section_m_text: Evaluation Factors (Section M)
            section_id: ID of the proposal section
            section_title: Title of the proposal section

        Returns:
            CriteriaChecklist with extracted criteria
        """
        checklist = CriteriaChecklist(
            section_id=section_id,
            section_title=section_title,
            extracted_from=["Section L", "Section M"]
        )

        criteria = []
        criterion_id = 0

        # Extract from Section L (instructions)
        l_criteria = self._extract_from_section_l(section_l_text, section_id)
        for text, source, priority in l_criteria:
            criterion_id += 1
            criteria.append(EvaluationCriterion(
                id=f"C-{section_id}-{criterion_id:03d}",
                text=text,
                source_section=source,
                priority=priority,
                keywords=self._extract_keywords(text)
            ))

        # Extract from Section M (evaluation factors)
        m_criteria = self._extract_from_section_m(section_m_text, section_id)
        for text, source, priority, weight in m_criteria:
            criterion_id += 1
            criteria.append(EvaluationCriterion(
                id=f"C-{section_id}-{criterion_id:03d}",
                text=text,
                source_section=source,
                priority=priority,
                weight=weight,
                keywords=self._extract_keywords(text)
            ))

        checklist.criteria = criteria
        return checklist

    def _extract_from_section_l(
        self,
        text: str,
        section_id: str
    ) -> List[Tuple[str, str, CriterionPriority]]:
        """Extract criteria from Section L instructions"""
        import re
        criteria = []

        # Pattern 1: "shall include/describe/address/demonstrate"
        shall_patterns = [
            r"(?:offeror|proposal|volume)\s+shall\s+((?:include|describe|address|demonstrate|provide|identify|explain)[^.]+\.)",
            r"(?:shall|must)\s+((?:include|describe|address|demonstrate|provide|identify|explain)[^.]+\.)",
        ]

        for pattern in shall_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                criterion_text = match.group(1).strip()
                if len(criterion_text) > 20:  # Filter out fragments
                    # Determine priority from keywords
                    priority = self._determine_priority(criterion_text)
                    # Try to find section reference nearby
                    source = self._find_section_ref(text, match.start())
                    criteria.append((criterion_text, source or "L", priority))

        # Pattern 2: Numbered/lettered requirements
        numbered_pattern = r"(?:^|\n)\s*(?:\([a-z]\)|\d+\.)\s*([A-Z][^.]+shall[^.]+\.)"
        matches = re.finditer(numbered_pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            criterion_text = match.group(1).strip()
            if len(criterion_text) > 20:
                priority = self._determine_priority(criterion_text)
                source = self._find_section_ref(text, match.start())
                criteria.append((criterion_text, source or "L", priority))

        return criteria

    def _extract_from_section_m(
        self,
        text: str,
        section_id: str
    ) -> List[Tuple[str, str, CriterionPriority, float]]:
        """Extract criteria from Section M evaluation factors"""
        import re
        criteria = []

        # Pattern 1: "will be evaluated"
        eval_patterns = [
            r"(?:government|agency)\s+will\s+(?:evaluate|assess|review)\s+([^.]+\.)",
            r"(?:proposals?\s+)?will\s+be\s+evaluated\s+(?:on|based\s+on|for)\s+([^.]+\.)",
        ]

        for pattern in eval_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                criterion_text = match.group(1).strip()
                if len(criterion_text) > 15:
                    priority = CriterionPriority.HIGH  # M criteria are high priority
                    weight = self._extract_weight(text, match.start())
                    source = self._find_section_ref(text, match.start())
                    criteria.append((criterion_text, source or "M", priority, weight))

        # Pattern 2: Strength/weakness language
        strength_pattern = r"(?:strength|weakness|deficiency)\s+(?:will\s+be\s+)?(?:assigned|given)\s+(?:if|when|for)\s+([^.]+\.)"
        matches = re.finditer(strength_pattern, text, re.IGNORECASE)
        for match in matches:
            criterion_text = match.group(1).strip()
            if len(criterion_text) > 15:
                criteria.append((criterion_text, "M", CriterionPriority.HIGH, 1.5))

        # Pattern 3: Rating descriptors
        rating_pattern = r"(?:outstanding|excellent|acceptable|marginal)\s*[-:]\s*([^.]+\.)"
        matches = re.finditer(rating_pattern, text, re.IGNORECASE)
        for match in matches:
            criterion_text = match.group(1).strip()
            if len(criterion_text) > 15:
                criteria.append((criterion_text, "M", CriterionPriority.MEDIUM, 1.0))

        return criteria

    def _determine_priority(self, text: str) -> CriterionPriority:
        """Determine criterion priority from text"""
        text_lower = text.lower()

        # Critical indicators
        if any(word in text_lower for word in ["shall", "must", "required", "mandatory"]):
            if any(word in text_lower for word in ["page limit", "format", "font", "margin"]):
                return CriterionPriority.CRITICAL  # P0 constraints

        # High priority
        if any(word in text_lower for word in ["shall", "must"]):
            return CriterionPriority.HIGH

        # Medium
        if any(word in text_lower for word in ["should", "recommend"]):
            return CriterionPriority.MEDIUM

        # Low
        return CriterionPriority.LOW

    def _extract_weight(self, text: str, position: int) -> float:
        """Extract evaluation weight from context"""
        import re

        # Look for weight indicators nearby
        search_area = text[max(0, position-500):position+500]

        # "most important"
        if re.search(r"most\s+important", search_area, re.IGNORECASE):
            return 2.0

        # "more important than"
        if re.search(r"more\s+important\s+than", search_area, re.IGNORECASE):
            return 1.5

        # "equally important"
        if re.search(r"equally\s+important", search_area, re.IGNORECASE):
            return 1.0

        # "less important"
        if re.search(r"less\s+important", search_area, re.IGNORECASE):
            return 0.75

        # Percentage weights
        pct_match = re.search(r"(\d+)\s*%", search_area)
        if pct_match:
            return float(pct_match.group(1)) / 100 * 2  # Normalize

        return 1.0

    def _find_section_ref(self, text: str, position: int) -> str:
        """Find section reference near position"""
        import re

        search_area = text[max(0, position-200):position+50]
        match = re.search(r"([LM])\.(\d+)(?:\.(\d+|[a-z]))?", search_area)
        if match:
            parts = [p for p in match.groups() if p]
            return ".".join(parts)
        return ""

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from criterion text"""
        import re

        keywords = []

        # Technical terms
        tech_terms = re.findall(
            r"\b(?:approach|methodology|experience|capability|risk|mitigation|"
            r"management|quality|performance|schedule|cost|staffing|transition|"
            r"solution|technical|organizational|past performance)\b",
            text.lower()
        )
        keywords.extend(tech_terms)

        return list(set(keywords))[:10]

    def evaluate_draft(
        self,
        draft_text: str,
        checklist: CriteriaChecklist,
        draft_version: int = 1
    ) -> EvaluationResult:
        """
        Evaluate a draft against the criteria checklist.

        This is the synchronous version using heuristic matching.
        For LLM-based evaluation, use evaluate_draft_async.

        Args:
            draft_text: The proposal draft text
            checklist: Criteria checklist to evaluate against
            draft_version: Version number of the draft

        Returns:
            EvaluationResult with scores and gaps
        """
        results = []
        gaps = []
        recommendations = []

        for criterion in checklist.criteria:
            # Evaluate this criterion
            status, score, evidence = self._evaluate_criterion(
                draft_text, criterion
            )

            criterion.status = status
            criterion.score = score
            criterion.evidence = evidence

            if status == ComplianceStatus.NOT_MET:
                gap = f"[{criterion.id}] Not addressed: {criterion.text[:100]}..."
                criterion.gaps = [gap]
                gaps.append(gap)
                recommendations.append(
                    f"Add content addressing: {criterion.keywords[:3]}"
                )
            elif status == ComplianceStatus.PARTIALLY_MET:
                gap = f"[{criterion.id}] Partially addressed: needs more detail"
                criterion.gaps = [gap]
                gaps.append(gap)

            results.append(criterion)

        # Calculate scores
        overall_score = self._calculate_overall_score(results)
        weighted_score = self._calculate_weighted_score(results)
        critical_score = self._calculate_critical_score(results)

        # Count statuses
        met_count = sum(1 for c in results if c.status == ComplianceStatus.FULLY_MET)
        partial_count = sum(1 for c in results if c.status == ComplianceStatus.PARTIALLY_MET)
        not_met_count = sum(1 for c in results if c.status == ComplianceStatus.NOT_MET)

        # Determine if passing
        passes = (
            overall_score >= self.pass_threshold and
            critical_score >= self.critical_threshold
        )

        # Calculate expansion factor
        if not passes:
            expansion_factor = self._calculate_expansion_factor(
                overall_score, len(gaps)
            )
        else:
            expansion_factor = 1.0

        return EvaluationResult(
            section_id=checklist.section_id,
            draft_version=draft_version,
            overall_score=overall_score,
            weighted_score=weighted_score,
            critical_score=critical_score,
            criteria_met=met_count,
            criteria_partial=partial_count,
            criteria_not_met=not_met_count,
            criteria_total=len(results),
            criteria_results=results,
            gaps=gaps,
            recommendations=recommendations[:10],  # Limit recommendations
            passes_threshold=passes,
            expansion_required=not passes,
            estimated_expansion_needed=expansion_factor
        )

    def _evaluate_criterion(
        self,
        draft_text: str,
        criterion: EvaluationCriterion
    ) -> Tuple[ComplianceStatus, float, str]:
        """Evaluate a single criterion against draft text"""
        import re

        draft_lower = draft_text.lower()
        criterion_lower = criterion.text.lower()

        # Count keyword matches
        keyword_matches = 0
        matched_keywords = []

        for keyword in criterion.keywords:
            if keyword.lower() in draft_lower:
                keyword_matches += 1
                matched_keywords.append(keyword)

        # Look for direct text overlap
        # Extract key phrases from criterion
        key_phrases = re.findall(r'\b\w+(?:\s+\w+){1,3}\b', criterion_lower)
        phrase_matches = 0

        for phrase in key_phrases[:10]:
            if phrase in draft_lower:
                phrase_matches += 1

        # Find evidence (sentence containing keywords)
        evidence = ""
        if matched_keywords:
            # Find sentence containing first matched keyword
            pattern = rf"[^.]*\b{re.escape(matched_keywords[0])}\b[^.]*\."
            match = re.search(pattern, draft_text, re.IGNORECASE)
            if match:
                evidence = match.group(0).strip()[:200]

        # Calculate match ratio
        total_keywords = len(criterion.keywords)
        if total_keywords == 0:
            # No keywords extracted, use phrase matching
            match_ratio = phrase_matches / max(len(key_phrases), 1)
        else:
            keyword_ratio = keyword_matches / total_keywords
            phrase_ratio = phrase_matches / max(len(key_phrases), 1)
            match_ratio = (keyword_ratio * 0.6 + phrase_ratio * 0.4)

        # Determine status
        if match_ratio >= 0.7:
            return ComplianceStatus.FULLY_MET, 1.0, evidence
        elif match_ratio >= 0.4:
            return ComplianceStatus.PARTIALLY_MET, 0.5, evidence
        else:
            return ComplianceStatus.NOT_MET, 0.0, ""

    def _calculate_overall_score(self, results: List[EvaluationCriterion]) -> float:
        """Calculate overall score (simple average)"""
        if not results:
            return 0.0

        total_score = sum(self.STATUS_SCORES[c.status] for c in results)
        return (total_score / len(results)) * 100

    def _calculate_weighted_score(self, results: List[EvaluationCriterion]) -> float:
        """Calculate weighted score based on criterion weights"""
        if not results:
            return 0.0

        weighted_sum = sum(
            self.STATUS_SCORES[c.status] * c.weight
            for c in results
        )
        weight_total = sum(c.weight for c in results)

        return (weighted_sum / weight_total) * 100 if weight_total > 0 else 0.0

    def _calculate_critical_score(self, results: List[EvaluationCriterion]) -> float:
        """Calculate score for critical items only"""
        critical = [c for c in results if c.priority == CriterionPriority.CRITICAL]
        if not critical:
            return 100.0  # No critical items = pass

        total_score = sum(self.STATUS_SCORES[c.status] for c in critical)
        return (total_score / len(critical)) * 100

    def _calculate_expansion_factor(
        self,
        current_score: float,
        gap_count: int
    ) -> float:
        """Calculate how much expansion is needed"""
        # Base expansion on score gap
        score_gap = (self.pass_threshold - current_score) / 100

        # Factor in number of gaps
        gap_factor = min(gap_count * 0.1, 0.5)  # Max 50% from gaps

        expansion = 1.0 + score_gap + gap_factor

        # Clamp to reasonable range
        return max(
            self.MIN_EXPANSION_FACTOR,
            min(self.MAX_EXPANSION_FACTOR, expansion)
        )

    async def evaluate_draft_async(
        self,
        draft_text: str,
        checklist: CriteriaChecklist,
        draft_version: int = 1
    ) -> EvaluationResult:
        """
        Evaluate draft using LLM-as-Judge for more accurate assessment.

        Args:
            draft_text: The proposal draft text
            checklist: Criteria checklist to evaluate against
            draft_version: Version number of the draft

        Returns:
            EvaluationResult with scores and gaps
        """
        if self.llm_client is None:
            # Fall back to heuristic evaluation
            logger.warning("No LLM client configured, using heuristic evaluation")
            return self.evaluate_draft(draft_text, checklist, draft_version)

        # Build evaluation prompt
        system_prompt = """You are an expert government proposal evaluator acting as
an LLM-as-Judge. Your task is to objectively assess whether a proposal draft
addresses specific evaluation criteria.

For each criterion, you must:
1. Determine if it is FULLY_MET, PARTIALLY_MET, or NOT_MET
2. Provide a brief evidence quote from the draft
3. Note any gaps if not fully met

Be rigorous but fair. Award FULLY_MET only when the criterion is completely addressed."""

        user_prompt = f"""Evaluate this proposal draft against the criteria checklist.

=== DRAFT TEXT ===
{draft_text[:20000]}  # Limit for token constraints

=== CRITERIA CHECKLIST ===
{checklist.to_json()}

For each criterion, respond with JSON:
{{
    "evaluations": [
        {{
            "criterion_id": "C-xxx-001",
            "status": "FULLY_MET|PARTIALLY_MET|NOT_MET",
            "confidence": 0.0-1.0,
            "evidence": "quote from draft...",
            "gaps": ["gap1", "gap2"]
        }}
    ],
    "overall_assessment": "summary...",
    "top_gaps": ["most important gap 1", "gap 2"],
    "recommendations": ["recommendation 1", "rec 2"]
}}"""

        # Call LLM
        from .llm_clients import LLMMessage, GenerationConfig

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        try:
            response = await self.llm_client.generate(
                messages,
                GenerationConfig(temperature=0.1, max_output_tokens=4096)
            )

            # Parse response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            eval_data = json.loads(content)

            # Update criteria with LLM results
            eval_map = {e["criterion_id"]: e for e in eval_data.get("evaluations", [])}

            for criterion in checklist.criteria:
                if criterion.id in eval_map:
                    result = eval_map[criterion.id]
                    criterion.status = ComplianceStatus(result["status"].lower())
                    criterion.score = self.STATUS_SCORES[criterion.status]
                    criterion.evidence = result.get("evidence", "")
                    criterion.gaps = result.get("gaps", [])

            # Use heuristic calculation on updated criteria
            return self.evaluate_draft(draft_text, checklist, draft_version)

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}, falling back to heuristic")
            return self.evaluate_draft(draft_text, checklist, draft_version)


def create_criteria_checklist(
    section_l: str,
    section_m: str,
    section_id: str,
    section_title: str
) -> CriteriaChecklist:
    """
    Convenience function to create a criteria checklist.

    Args:
        section_l: Section L text
        section_m: Section M text
        section_id: Target section ID
        section_title: Target section title

    Returns:
        CriteriaChecklist for evaluation
    """
    evaluator = CriteriaEvaluator()
    return evaluator.extract_criteria_from_text(
        section_l, section_m, section_id, section_title
    )
