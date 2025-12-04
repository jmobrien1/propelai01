"""
Project Selection Optimizer
===========================

Solves the combinatorial optimization problem of selecting the best
5 Qualifying Projects to maximize the OASIS+ score.

This is essentially a constrained "Knapsack Problem" where:
- Objective: Maximize total score above threshold (36 or 42 points)
- Constraint: Maximum 5 Qualifying Projects per domain
- Constraint: Projects must meet AAV and recency requirements
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Dict, Optional, Set, Tuple
from decimal import Decimal

from .models import (
    Project,
    ProjectClaim,
    OASISDomain,
    ScoringCriteria,
    ScorecardResult,
    OptimizationConstraints,
    BusinessSize,
    DomainType,
    VerificationStatus,
    CriteriaType,
)

logger = logging.getLogger(__name__)


@dataclass
class ProjectScore:
    """Score breakdown for a single project"""
    project: Project
    total_points: int = 0
    verified_points: int = 0
    claims: List[ProjectClaim] = field(default_factory=list)

    # Breakdown by type
    mandatory_points: int = 0
    optional_credit_points: int = 0

    # Risk assessment
    unverified_claims: int = 0
    low_confidence_claims: int = 0

    @property
    def risk_score(self) -> float:
        """Calculate risk score (0 = safe, 1 = risky)"""
        if self.total_points == 0:
            return 1.0
        unverified_ratio = (self.total_points - self.verified_points) / self.total_points
        return unverified_ratio


@dataclass
class OptimizationResult:
    """Result of the project optimization"""
    domain: DomainType
    business_size: BusinessSize

    # Selected projects
    selected_projects: List[Project] = field(default_factory=list)
    project_scores: List[ProjectScore] = field(default_factory=list)

    # Score summary
    total_score: int = 0
    verified_score: int = 0
    threshold: int = 42
    margin: int = 0

    # All claims for selected projects
    all_claims: List[ProjectClaim] = field(default_factory=list)

    # Alternative combinations considered
    alternatives_evaluated: int = 0
    runner_up_score: int = 0

    # Risk assessment
    overall_risk: str = "LOW"  # LOW, MEDIUM, HIGH
    risk_factors: List[str] = field(default_factory=list)

    @property
    def meets_threshold(self) -> bool:
        return self.total_score >= self.threshold

    @property
    def has_safe_margin(self) -> bool:
        return self.margin >= 3

    def to_scorecard(self) -> ScorecardResult:
        """Convert to ScorecardResult for reporting"""
        scorecard = ScorecardResult(
            domain=self.domain,
            business_size=self.business_size,
            qualifying_projects=self.selected_projects,
            claims=self.all_claims,
            total_score=self.total_score,
            threshold=self.threshold,
            margin=self.margin,
            verified_points=self.verified_score,
        )
        scorecard.calculate_totals()
        return scorecard


class ProjectOptimizer:
    """
    Optimizes project selection to maximize OASIS+ scores.

    Evaluates all valid combinations of projects to find the
    "Golden Set" that exceeds the threshold with maximum margin.
    """

    def __init__(self, constraints: Optional[OptimizationConstraints] = None):
        """
        Initialize the optimizer.

        Args:
            constraints: Optimization constraints (defaults applied if None)
        """
        self.constraints = constraints or OptimizationConstraints()

    def optimize(
        self,
        projects: List[Project],
        claims: Dict[str, List[ProjectClaim]],  # project_id -> claims
        domain: OASISDomain,
        business_size: BusinessSize,
    ) -> OptimizationResult:
        """
        Find the optimal set of projects for a domain.

        Args:
            projects: All available projects
            claims: Pre-computed claims for each project
            domain: The OASIS+ domain being optimized
            business_size: Business size category

        Returns:
            OptimizationResult with selected projects and scores
        """
        logger.info(f"Optimizing {len(projects)} projects for {domain.name}")

        # Initialize result
        result = OptimizationResult(
            domain=domain.domain_type,
            business_size=business_size,
            threshold=domain.get_threshold(business_size),
        )

        # Filter to qualifying projects
        qualifying = self._filter_qualifying_projects(projects, domain, business_size)
        logger.info(f"Found {len(qualifying)} qualifying projects")

        if not qualifying:
            result.risk_factors.append("No qualifying projects found")
            result.overall_risk = "HIGH"
            return result

        # Score each project
        project_scores = {}
        for project in qualifying:
            score = self._score_project(project, claims.get(project.project_id, []))
            project_scores[project.project_id] = score

        # Find optimal combination
        max_projects = min(
            self.constraints.max_qualifying_projects,
            len(qualifying)
        )

        best_combo = None
        best_score = 0
        runner_up_score = 0
        combos_evaluated = 0

        # Try all combinations from max down to 1
        for k in range(max_projects, 0, -1):
            for combo in combinations(qualifying, k):
                combos_evaluated += 1
                combo_score = self._evaluate_combination(
                    combo, project_scores, domain, business_size
                )

                if combo_score > best_score:
                    runner_up_score = best_score
                    best_score = combo_score
                    best_combo = combo
                elif combo_score > runner_up_score:
                    runner_up_score = combo_score

        result.alternatives_evaluated = combos_evaluated
        result.runner_up_score = runner_up_score

        if best_combo:
            result.selected_projects = list(best_combo)
            result.project_scores = [
                project_scores[p.project_id] for p in best_combo
            ]
            result.total_score = best_score
            result.verified_score = sum(
                ps.verified_points for ps in result.project_scores
            )
            result.margin = best_score - result.threshold

            # Collect all claims
            for project in best_combo:
                result.all_claims.extend(claims.get(project.project_id, []))

            # Assess risk
            result.overall_risk, result.risk_factors = self._assess_risk(result)

        logger.info(
            f"Optimization complete: {result.total_score} points "
            f"({result.margin:+d} margin) from {len(result.selected_projects)} projects"
        )

        return result

    def _filter_qualifying_projects(
        self,
        projects: List[Project],
        domain: OASISDomain,
        business_size: BusinessSize,
    ) -> List[Project]:
        """Filter projects that meet basic qualification requirements"""
        qualifying = []

        for project in projects:
            # Check AAV requirement
            aav = project.calculate_aav()
            min_aav = domain.get_min_aav(business_size)
            if aav < min_aav:
                logger.debug(
                    f"Project {project.project_id} AAV ${aav:,.0f} "
                    f"below minimum ${min_aav:,.0f}"
                )
                continue

            # Check recency
            if not project.is_recent(self.constraints.recency_cutoff_years):
                logger.debug(f"Project {project.project_id} fails recency check")
                continue

            # Check domain relevance (NAICS/PSC codes)
            if not self._check_domain_relevance(project, domain):
                logger.debug(
                    f"Project {project.project_id} not relevant to {domain.name}"
                )
                continue

            qualifying.append(project)

        return qualifying

    def _check_domain_relevance(
        self,
        project: Project,
        domain: OASISDomain,
    ) -> bool:
        """Check if project is relevant to domain via NAICS/PSC codes"""
        # Check auto-relevant codes
        if project.naics_code in domain.auto_relevant_naics:
            return True
        if project.psc_code in domain.auto_relevant_psc:
            return True

        # Check pre-computed relevance scores
        if domain.domain_type in project.relevance_scores:
            return project.relevance_scores[domain.domain_type] >= 0.5

        # Default: allow if no auto-relevant codes defined
        if not domain.auto_relevant_naics and not domain.auto_relevant_psc:
            return True

        return False

    def _score_project(
        self,
        project: Project,
        claims: List[ProjectClaim],
    ) -> ProjectScore:
        """Calculate total score for a project from its claims"""
        score = ProjectScore(project=project)

        for claim in claims:
            score.claims.append(claim)
            score.total_points += claim.claimed_points

            if claim.status == VerificationStatus.VERIFIED:
                score.verified_points += claim.verified_points
            elif claim.status == VerificationStatus.FPDS_VERIFIED:
                score.verified_points += claim.claimed_points

            if claim.status == VerificationStatus.UNVERIFIED:
                score.unverified_claims += 1

            if claim.ai_confidence_score < self.constraints.min_confidence_score:
                score.low_confidence_claims += 1

        return score

    def _evaluate_combination(
        self,
        projects: Tuple[Project, ...],
        project_scores: Dict[str, ProjectScore],
        domain: OASISDomain,
        business_size: BusinessSize,
    ) -> int:
        """Evaluate total score for a combination of projects"""
        total = 0

        # Track unique criteria claimed (can't double-count)
        claimed_criteria: Set[str] = set()

        for project in projects:
            ps = project_scores.get(project.project_id)
            if not ps:
                continue

            for claim in ps.claims:
                # Don't double-count criteria across projects
                if claim.criteria_id in claimed_criteria:
                    continue
                claimed_criteria.add(claim.criteria_id)
                total += claim.claimed_points

        # Apply diversity bonus if required
        if self.constraints.require_agency_diversity:
            unique_agencies = len(set(p.client_agency for p in projects))
            if unique_agencies >= self.constraints.min_unique_agencies:
                total += 1  # Small bonus for diversity

        return total

    def _assess_risk(
        self,
        result: OptimizationResult,
    ) -> Tuple[str, List[str]]:
        """Assess overall risk of the selected combination"""
        risk_factors = []
        risk_level = "LOW"

        # Check margin
        if result.margin < 0:
            risk_factors.append(f"Below threshold by {-result.margin} points")
            risk_level = "HIGH"
        elif result.margin < 3:
            risk_factors.append(f"Thin margin of only {result.margin} points")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"

        # Check verification status
        unverified_ratio = 1 - (result.verified_score / max(result.total_score, 1))
        if unverified_ratio > 0.5:
            risk_factors.append(f"{unverified_ratio*100:.0f}% of points unverified")
            risk_level = "HIGH"
        elif unverified_ratio > 0.25:
            risk_factors.append(f"{unverified_ratio*100:.0f}% of points unverified")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"

        # Check for low confidence claims
        low_conf_claims = sum(
            1 for ps in result.project_scores
            for c in ps.claims
            if c.ai_confidence_score < self.constraints.min_confidence_score
        )
        if low_conf_claims > 3:
            risk_factors.append(f"{low_conf_claims} claims have low AI confidence")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"

        # Check project count
        if len(result.selected_projects) < 3:
            risk_factors.append("Relying on fewer than 3 projects")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"

        return risk_level, risk_factors

    def find_gaps(
        self,
        result: OptimizationResult,
        domain: OASISDomain,
    ) -> List[Dict]:
        """Identify gaps and opportunities to improve score"""
        gaps = []

        # Check if below threshold
        if result.margin < 0:
            gaps.append({
                "type": "threshold_gap",
                "severity": "critical",
                "message": f"Need {-result.margin} more points to qualify",
                "recommendation": "Find additional qualifying projects or verify more claims",
            })

        # Check for unverified high-value claims
        for ps in result.project_scores:
            for claim in ps.claims:
                if claim.status == VerificationStatus.UNVERIFIED and claim.claimed_points >= 3:
                    gaps.append({
                        "type": "unverified_claim",
                        "severity": "high" if claim.claimed_points >= 5 else "medium",
                        "message": f"Claim for {claim.criteria_id} ({claim.claimed_points} pts) unverified",
                        "recommendation": f"Find evidence in project documents or obtain J.P-3",
                    })

        # Check for unused optional credits
        claimed_criteria = {c.criteria_id for c in result.all_claims}
        for criteria in domain.criteria:
            if criteria.criteria_type == CriteriaType.OPTIONAL_CREDIT:
                if criteria.criteria_id not in claimed_criteria:
                    gaps.append({
                        "type": "unclaimed_credit",
                        "severity": "opportunity",
                        "message": f"Optional credit {criteria.criteria_id} ({criteria.max_points} pts) not claimed",
                        "recommendation": f"Search projects for: {criteria.description[:50]}...",
                    })

        return gaps

    def suggest_improvements(
        self,
        result: OptimizationResult,
        all_projects: List[Project],
        domain: OASISDomain,
    ) -> List[str]:
        """Suggest actions to improve the score"""
        suggestions = []

        # If below threshold, find potential additional projects
        if result.margin < 3:
            unused_projects = [
                p for p in all_projects
                if p.project_id not in {sp.project_id for sp in result.selected_projects}
            ]
            if unused_projects:
                suggestions.append(
                    f"Consider {len(unused_projects)} unused projects that may add points"
                )

        # Suggest verifying unverified claims
        unverified = [
            c for c in result.all_claims
            if c.status == VerificationStatus.UNVERIFIED
        ]
        if unverified:
            high_value = [c for c in unverified if c.claimed_points >= 3]
            suggestions.append(
                f"Verify {len(high_value)} high-value unverified claims "
                f"(total {sum(c.claimed_points for c in high_value)} potential points)"
            )

        # Suggest J.P-3 forms for claims that need them
        needs_jp3 = [
            c for c in result.all_claims
            if c.status == VerificationStatus.JP3_REQUIRED
        ]
        if needs_jp3:
            suggestions.append(
                f"Obtain {len(needs_jp3)} J.P-3 verification forms from Contracting Officers"
            )

        return suggestions


def optimize_for_domain(
    projects: List[Project],
    claims: Dict[str, List[ProjectClaim]],
    domain: OASISDomain,
    business_size: BusinessSize = BusinessSize.UNRESTRICTED,
    constraints: Optional[OptimizationConstraints] = None,
) -> OptimizationResult:
    """
    Convenience function to optimize projects for a domain.

    Args:
        projects: All available projects
        claims: Pre-computed claims for each project
        domain: The OASIS+ domain
        business_size: Business size category
        constraints: Optional constraints

    Returns:
        OptimizationResult with optimal project selection
    """
    optimizer = ProjectOptimizer(constraints)
    return optimizer.optimize(projects, claims, domain, business_size)
