"""
PropelAI Red Team Agent - "The Evaluator"
Play 4: The War Room (Governance & Red Team)

Goal: Simulate a government evaluator to score the draft BEFORE submission

This agent:
1. Ingests the draft and original Section M criteria
2. Scores the proposal using government color scoring (Blue/Green/Yellow/Red)
3. Identifies compliance gaps and weaknesses
4. Provides specific remediation feedback
5. Maintains the audit log for governance

The Audit Log is our "Trust Layer" for C-Suite executives
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from core.state import ProposalState, ProposalPhase, ScoreColor


class EvaluationCategory(str, Enum):
    """Categories for evaluation"""
    COMPLIANCE = "compliance"
    TECHNICAL_MERIT = "technical_merit"
    PAST_PERFORMANCE = "past_performance"
    MANAGEMENT = "management"
    PRICE_REASONABLENESS = "price_reasonableness"
    OVERALL = "overall"


@dataclass
class Finding:
    """An evaluation finding (positive or negative)"""
    id: str
    category: EvaluationCategory
    finding_type: str              # "strength", "weakness", "deficiency", "risk"
    section_reference: str
    description: str
    impact: str                    # How this affects the score
    remediation: Optional[str]     # How to fix (for negatives)
    requirement_ref: Optional[str] # Linked requirement if applicable


@dataclass
class SectionScore:
    """Score for a proposal section"""
    section_id: str
    section_title: str
    color_score: ScoreColor
    numeric_score: float          # 0-100 scale
    findings: List[Finding]
    compliance_rate: float        # % of requirements addressed
    strengths_count: int
    weaknesses_count: int
    deficiencies_count: int


@dataclass
class ProposalEvaluation:
    """Complete proposal evaluation"""
    evaluation_id: str
    evaluated_at: datetime
    overall_score: ScoreColor
    overall_numeric: float
    section_scores: List[SectionScore]
    total_findings: int
    critical_deficiencies: List[Finding]
    summary_narrative: str
    recommendation: str           # "submit", "revise", "major_revision"


class RedTeamAgent:
    """
    The Red Team Agent - "The Evaluator"
    
    Simulates a government evaluator using Section M criteria.
    Provides the "Trust Layer" through rigorous scoring and audit logging.
    
    Color Scoring System:
    - BLUE: Exceptional - Significantly exceeds requirements
    - GREEN: Acceptable - Meets requirements
    - YELLOW: Marginal - May not meet requirements
    - RED: Unacceptable - Fails to meet requirements
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the Red Team Agent
        
        Args:
            llm_client: LLM for evaluation reasoning (high-capability model)
        """
        self.llm_client = llm_client
        
    def __call__(self, state: ProposalState) -> Dict[str, Any]:
        """
        Main entry point - called by the Orchestrator
        
        Performs comprehensive proposal evaluation
        """
        start_time = datetime.now()
        
        # Get inputs
        draft_sections = state.get("draft_sections", {})
        evaluation_criteria = state.get("evaluation_criteria", [])
        requirements = state.get("requirements", [])
        compliance_matrix = state.get("compliance_matrix", [])
        
        if not draft_sections:
            existing_trace = state.get("agent_trace_log", [])
            return {
                "error_state": "No draft sections to evaluate",
                "agent_trace_log": existing_trace + [{
                    "timestamp": start_time.isoformat(),
                    "agent_name": "red_team_agent",
                    "action": "evaluate_proposal",
                    "input_summary": "No drafts found",
                    "output_summary": "Error: Prerequisites not met",
                    "reasoning_trace": "Red Team requires completed drafts"
                }]
            }
        
        # Phase 1: Evaluate each section
        section_scores = []
        all_findings = []
        
        for section_id, draft in draft_sections.items():
            # Get relevant criteria for this section
            relevant_criteria = self._get_relevant_criteria(
                section_id, 
                evaluation_criteria
            )
            
            # Get relevant requirements
            relevant_reqs = self._get_relevant_requirements(
                section_id,
                requirements,
                compliance_matrix
            )
            
            # Score the section
            section_score = self._evaluate_section(
                section_id,
                draft,
                relevant_criteria,
                relevant_reqs
            )
            
            section_scores.append(section_score)
            all_findings.extend(section_score.findings)
        
        # Phase 2: Calculate overall score
        overall_evaluation = self._calculate_overall_score(
            section_scores,
            all_findings
        )
        
        # Phase 3: Identify critical deficiencies
        critical = [f for f in all_findings if f.finding_type == "deficiency"]
        
        # Phase 4: Generate remediation plan
        remediation_plan = self._generate_remediation_plan(
            section_scores,
            critical
        )
        
        # Phase 5: Generate summary narrative
        narrative = self._generate_evaluation_narrative(
            overall_evaluation,
            section_scores,
            critical
        )
        
        # Calculate processing time
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Build trace log (the Audit Log)
        trace_log = {
            "timestamp": start_time.isoformat(),
            "agent_name": "red_team_agent",
            "action": "evaluate_proposal",
            "input_summary": f"{len(draft_sections)} sections, {len(evaluation_criteria)} criteria",
            "output_summary": f"Overall: {overall_evaluation['color_score']} ({overall_evaluation['numeric_score']:.1f}), "
                            f"{len(all_findings)} findings, {len(critical)} deficiencies",
            "reasoning_trace": self._format_reasoning_trace(section_scores),
            "duration_ms": duration_ms,
            "tool_calls": [
                {"tool": "section_evaluation", "result": f"{len(section_scores)} sections scored"},
                {"tool": "compliance_check", "result": f"{sum(s.compliance_rate for s in section_scores)/len(section_scores)*100:.1f}% avg compliance"},
                {"tool": "deficiency_scan", "result": f"{len(critical)} critical deficiencies"},
            ]
        }
        
        # Determine next phase based on score
        next_phase = ProposalPhase.REVIEW.value
        if overall_evaluation["color_score"] in [ScoreColor.RED.value, ScoreColor.YELLOW.value]:
            # Needs revision
            next_phase = ProposalPhase.DRAFTING.value
        elif overall_evaluation["color_score"] in [ScoreColor.GREEN.value, ScoreColor.BLUE.value]:
            # Ready for human review
            next_phase = ProposalPhase.FINALIZE.value
        
        # Build red team feedback object
        feedback = {
            "evaluation_id": f"EVAL-{start_time.strftime('%Y%m%d%H%M%S')}",
            "evaluated_at": start_time.isoformat(),
            "overall_score": overall_evaluation["color_score"],
            "overall_numeric": overall_evaluation["numeric_score"],
            "section_scores": [self._section_score_to_dict(s) for s in section_scores],
            "findings": [self._finding_to_dict(f) for f in all_findings],
            "critical_deficiencies": [self._finding_to_dict(f) for f in critical],
            "remediation_plan": remediation_plan,
            "narrative": narrative,
            "recommendation": overall_evaluation["recommendation"]
        }
        
        # Accumulate trace logs
        existing_trace = state.get("agent_trace_log", [])

        return {
            "current_phase": next_phase,
            "red_team_feedback": [feedback],
            "proposal_quality_score": overall_evaluation["numeric_score"],
            "agent_trace_log": existing_trace + [trace_log],
            "updated_at": datetime.now().isoformat()
        }
    
    def _get_relevant_criteria(
        self, 
        section_id: str, 
        criteria: List[Dict]
    ) -> List[Dict]:
        """Get evaluation criteria relevant to a section"""
        relevant = []
        
        # Map section types to criteria
        section_lower = section_id.lower()
        
        for criterion in criteria:
            factor = criterion.get("factor_name", "").lower()
            
            if "technical" in section_lower and "technical" in factor:
                relevant.append(criterion)
            elif "management" in section_lower and "management" in factor:
                relevant.append(criterion)
            elif "past_performance" in section_lower and "past performance" in factor:
                relevant.append(criterion)
            elif "pricing" in section_lower and ("price" in factor or "cost" in factor):
                relevant.append(criterion)
        
        return relevant
    
    def _get_relevant_requirements(
        self,
        section_id: str,
        requirements: List[Dict],
        compliance_matrix: List[Dict]
    ) -> List[Dict]:
        """Get requirements assigned to this section"""
        relevant = []
        
        for matrix_item in compliance_matrix:
            if matrix_item.get("assigned_section") == section_id:
                # Find full requirement
                req_id = matrix_item.get("requirement_id")
                for req in requirements:
                    if req.get("id") == req_id:
                        relevant.append(req)
                        break
        
        return relevant
    
    def _evaluate_section(
        self,
        section_id: str,
        draft: Dict,
        criteria: List[Dict],
        requirements: List[Dict]
    ) -> SectionScore:
        """Evaluate a single section"""
        findings = []
        
        content = draft.get("content", "")
        citations = draft.get("citations", [])
        uncited_claims = draft.get("uncited_claims", [])
        word_count = draft.get("word_count", 0)
        page_allocation = draft.get("page_allocation", 0)
        
        # Compliance check - are requirements addressed?
        addressed_count = 0
        for req in requirements:
            if self._requirement_addressed(req, content):
                addressed_count += 1
            else:
                # Deficiency: requirement not addressed
                findings.append(Finding(
                    id=f"FIND-{len(findings)+1:04d}",
                    category=EvaluationCategory.COMPLIANCE,
                    finding_type="deficiency",
                    section_reference=section_id,
                    description=f"Requirement {req.get('section_ref', 'N/A')} not adequately addressed",
                    impact="May result in lower technical score or non-compliance determination",
                    remediation=f"Add explicit response to: {req.get('text', '')[:100]}...",
                    requirement_ref=req.get("id")
                ))
        
        compliance_rate = addressed_count / len(requirements) if requirements else 1.0
        
        # Citation check - are claims supported?
        if uncited_claims:
            for claim in uncited_claims:
                findings.append(Finding(
                    id=f"FIND-{len(findings)+1:04d}",
                    category=EvaluationCategory.TECHNICAL_MERIT,
                    finding_type="weakness",
                    section_reference=section_id,
                    description=f"Unsupported claim: {claim[:100]}",
                    impact="Unsubstantiated claims reduce credibility",
                    remediation="Add citation from past performance or corporate capabilities",
                    requirement_ref=None
                ))
        
        # Strength identification
        if citations:
            findings.append(Finding(
                id=f"FIND-{len(findings)+1:04d}",
                category=EvaluationCategory.TECHNICAL_MERIT,
                finding_type="strength",
                section_reference=section_id,
                description=f"Well-cited section with {len(citations)} supporting references",
                impact="Increases credibility and demonstrates relevant experience",
                remediation=None,
                requirement_ref=None
            ))
        
        # Word count check
        target_words = page_allocation * 300
        if word_count < target_words * 0.7:
            findings.append(Finding(
                id=f"FIND-{len(findings)+1:04d}",
                category=EvaluationCategory.COMPLIANCE,
                finding_type="risk",
                section_reference=section_id,
                description=f"Section under target length ({word_count} vs {target_words} target)",
                impact="May appear incomplete to evaluators",
                remediation="Expand content with additional detail and examples",
                requirement_ref=None
            ))
        
        # Calculate section score
        numeric_score = self._calculate_section_score(
            compliance_rate,
            len(citations),
            len(uncited_claims),
            findings
        )
        
        color_score = self._numeric_to_color(numeric_score)
        
        return SectionScore(
            section_id=section_id,
            section_title=draft.get("section_title", section_id),
            color_score=color_score,
            numeric_score=numeric_score,
            findings=findings,
            compliance_rate=compliance_rate,
            strengths_count=len([f for f in findings if f.finding_type == "strength"]),
            weaknesses_count=len([f for f in findings if f.finding_type == "weakness"]),
            deficiencies_count=len([f for f in findings if f.finding_type == "deficiency"])
        )
    
    def _requirement_addressed(self, requirement: Dict, content: str) -> bool:
        """Check if a requirement is addressed in the content"""
        content_lower = content.lower()
        
        # Check for keywords from requirement
        keywords = requirement.get("keywords", [])
        matches = sum(1 for kw in keywords if kw.lower() in content_lower)
        
        # Need at least 30% keyword match
        if keywords and matches / len(keywords) >= 0.3:
            return True
        
        # Check for section reference
        ref = requirement.get("section_ref", "")
        if ref and ref.lower() in content_lower:
            return True
        
        return False
    
    def _calculate_section_score(
        self,
        compliance_rate: float,
        citation_count: int,
        uncited_count: int,
        findings: List[Finding]
    ) -> float:
        """Calculate numeric score for a section (0-100)"""
        # Base score from compliance
        score = compliance_rate * 60  # 60 points for compliance
        
        # Citation bonus/penalty
        total_claims = citation_count + uncited_count
        if total_claims > 0:
            citation_rate = citation_count / total_claims
            score += citation_rate * 20  # 20 points for citations
        else:
            score += 10  # Neutral if no claims
        
        # Finding adjustments
        strengths = len([f for f in findings if f.finding_type == "strength"])
        weaknesses = len([f for f in findings if f.finding_type == "weakness"])
        deficiencies = len([f for f in findings if f.finding_type == "deficiency"])
        
        score += strengths * 3        # +3 per strength
        score -= weaknesses * 2       # -2 per weakness
        score -= deficiencies * 10    # -10 per deficiency
        
        return max(0, min(100, score))
    
    def _numeric_to_color(self, score: float) -> ScoreColor:
        """Convert numeric score to color rating"""
        if score >= 90:
            return ScoreColor.BLUE
        elif score >= 70:
            return ScoreColor.GREEN
        elif score >= 50:
            return ScoreColor.YELLOW
        else:
            return ScoreColor.RED
    
    def _calculate_overall_score(
        self,
        section_scores: List[SectionScore],
        findings: List[Finding]
    ) -> Dict[str, Any]:
        """Calculate overall proposal score"""
        if not section_scores:
            return {
                "color_score": ScoreColor.RED.value,
                "numeric_score": 0,
                "recommendation": "major_revision"
            }
        
        # Average of section scores
        avg_score = sum(s.numeric_score for s in section_scores) / len(section_scores)
        
        # Penalty for critical deficiencies
        critical_count = len([f for f in findings if f.finding_type == "deficiency"])
        avg_score -= critical_count * 5  # -5 per deficiency
        
        avg_score = max(0, min(100, avg_score))
        
        # Determine color
        color = self._numeric_to_color(avg_score)
        
        # Determine recommendation
        if color == ScoreColor.RED:
            recommendation = "major_revision"
        elif color == ScoreColor.YELLOW:
            recommendation = "revise"
        else:
            recommendation = "submit"
        
        return {
            "color_score": color.value,
            "numeric_score": avg_score,
            "recommendation": recommendation
        }
    
    def _generate_remediation_plan(
        self,
        section_scores: List[SectionScore],
        critical_deficiencies: List[Finding]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized remediation plan"""
        plan = []
        
        # Priority 1: Critical deficiencies
        for deficiency in critical_deficiencies:
            plan.append({
                "priority": "CRITICAL",
                "action": deficiency.remediation,
                "section": deficiency.section_reference,
                "finding_id": deficiency.id,
                "deadline": "Immediate - before any further drafting"
            })
        
        # Priority 2: Sections with RED/YELLOW scores
        for section in section_scores:
            if section.color_score in [ScoreColor.RED, ScoreColor.YELLOW]:
                weaknesses = [f for f in section.findings if f.finding_type == "weakness"]
                for weakness in weaknesses:
                    plan.append({
                        "priority": "HIGH" if section.color_score == ScoreColor.RED else "MEDIUM",
                        "action": weakness.remediation,
                        "section": section.section_id,
                        "finding_id": weakness.id,
                        "deadline": "Before Red Team review"
                    })
        
        return plan[:20]  # Limit to top 20 items
    
    def _generate_evaluation_narrative(
        self,
        overall: Dict,
        section_scores: List[SectionScore],
        critical: List[Finding]
    ) -> str:
        """Generate executive summary narrative"""
        color = overall["color_score"]
        score = overall["numeric_score"]
        recommendation = overall["recommendation"]
        
        # Build narrative
        narrative = f"PROPOSAL EVALUATION SUMMARY\n\n"
        narrative += f"Overall Rating: {color.upper()} ({score:.1f}/100)\n"
        narrative += f"Recommendation: {recommendation.replace('_', ' ').title()}\n\n"
        
        narrative += "KEY FINDINGS:\n"
        
        # Summarize by section
        for section in section_scores:
            narrative += f"\n• {section.section_title}: {section.color_score.value.upper()}\n"
            narrative += f"  - Compliance: {section.compliance_rate*100:.0f}%\n"
            narrative += f"  - Strengths: {section.strengths_count}, Weaknesses: {section.weaknesses_count}\n"
            
            if section.deficiencies_count > 0:
                narrative += f"  - DEFICIENCIES: {section.deficiencies_count} (MUST ADDRESS)\n"
        
        # Critical findings
        if critical:
            narrative += "\n\nCRITICAL DEFICIENCIES REQUIRING IMMEDIATE ATTENTION:\n"
            for i, deficiency in enumerate(critical[:5], 1):
                narrative += f"\n{i}. {deficiency.description}\n"
                narrative += f"   Section: {deficiency.section_reference}\n"
                narrative += f"   Remediation: {deficiency.remediation}\n"
        
        return narrative
    
    def _format_reasoning_trace(self, section_scores: List[SectionScore]) -> str:
        """Format reasoning for audit log"""
        trace_parts = []
        
        for section in section_scores:
            trace_parts.append(
                f"{section.section_id}: {section.color_score.value} "
                f"(compliance={section.compliance_rate:.0%}, "
                f"strengths={section.strengths_count}, "
                f"weaknesses={section.weaknesses_count}, "
                f"deficiencies={section.deficiencies_count})"
            )
        
        return " | ".join(trace_parts)
    
    def _section_score_to_dict(self, score: SectionScore) -> Dict[str, Any]:
        """Convert SectionScore to dictionary"""
        return {
            "section_id": score.section_id,
            "section_title": score.section_title,
            "color_score": score.color_score.value,
            "numeric_score": score.numeric_score,
            "compliance_rate": score.compliance_rate,
            "strengths_count": score.strengths_count,
            "weaknesses_count": score.weaknesses_count,
            "deficiencies_count": score.deficiencies_count
        }
    
    def _finding_to_dict(self, finding: Finding) -> Dict[str, Any]:
        """Convert Finding to dictionary"""
        return {
            "id": finding.id,
            "category": finding.category.value,
            "finding_type": finding.finding_type,
            "section_reference": finding.section_reference,
            "description": finding.description,
            "impact": finding.impact,
            "remediation": finding.remediation,
            "requirement_ref": finding.requirement_ref
        }


def create_red_team_agent(llm_client: Optional[Any] = None) -> RedTeamAgent:
    """Factory function to create a Red Team Agent"""
    return RedTeamAgent(llm_client=llm_client)
