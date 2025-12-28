"""
PropelAI Trust Gate: Accuracy Auditor v4.0

Provides accuracy auditing capabilities for extracted requirements.
Implements stratified sampling, false positive detection, and metrics reporting.

Key Features:
- Stratified random sampling across priority buckets (High/Medium/Low)
- Heuristic-based false positive detection
- Comprehensive accuracy metrics with confidence intervals
- Exportable audit reports for quality assurance
"""

import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


class AuditStatus(Enum):
    """Status of a requirement in the audit process"""
    PENDING = "pending"
    VERIFIED = "verified"
    FALSE_POSITIVE = "false_positive"
    NEEDS_REVIEW = "needs_review"
    MODIFIED = "modified"


class FalsePositiveReason(Enum):
    """Reasons why a requirement may be a false positive"""
    BOILERPLATE = "boilerplate"
    EVALUATION_CRITERIA = "evaluation_criteria"
    FORMAT_INSTRUCTION = "format_instruction"
    INFORMATIONAL = "informational"
    DUPLICATE = "duplicate"
    INCOMPLETE = "incomplete"
    OUT_OF_SCOPE = "out_of_scope"
    HEADER_FOOTER = "header_footer"


@dataclass
class AuditedRequirement:
    """A requirement that has been through the audit process"""
    requirement_id: str
    requirement_text: str
    source_document: str
    page_number: int
    priority: str
    audit_status: AuditStatus = AuditStatus.PENDING
    false_positive_reason: Optional[FalsePositiveReason] = None
    confidence_score: float = 0.0
    auditor_notes: str = ""
    audited_at: Optional[datetime] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class AuditSample:
    """A sample of requirements for manual audit"""
    sample_id: str
    total_population: int
    sample_size: int
    stratification: Dict[str, int]  # priority -> count
    requirements: List[AuditedRequirement]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuditMetrics:
    """Accuracy metrics from an audit"""
    total_sampled: int
    verified_count: int
    false_positive_count: int
    needs_review_count: int
    accuracy_rate: float
    confidence_interval_low: float
    confidence_interval_high: float
    false_positive_breakdown: Dict[str, int]
    flagged_patterns: Dict[str, int]
    audit_timestamp: datetime = field(default_factory=datetime.now)


# Common boilerplate patterns that indicate false positives
BOILERPLATE_PATTERNS = [
    r"page\s+\d+\s+of\s+\d+",
    r"^\d+$",  # Just a number
    r"^[A-Z][-.]?\s*$",  # Just a letter with optional dash/period
    r"table\s+of\s+contents",
    r"^attachment\s+[a-z0-9]+\s*[-:–]\s*$",
    r"^exhibit\s+[a-z0-9]+\s*[-:–]\s*$",
    r"revision\s+(?:date|history|log)",
    r"^\[\s*intentionally\s+(?:left\s+)?blank\s*\]$",
    r"^continued\s+(?:on\s+)?(?:next|following)\s+page",
    r"^see\s+(?:next|following|attached)",
    r"^(?:this|the)\s+page\s+(?:is\s+)?(?:intentionally\s+)?(?:left\s+)?blank",
]

# Patterns indicating evaluation criteria (not actual requirements)
EVALUATION_PATTERNS = [
    r"will\s+be\s+evaluated",
    r"evaluation\s+(?:criteria|factors?|basis)",
    r"point\s+(?:value|score)",
    r"weighted\s+(?:at|by)\s+\d+",
    r"scoring\s+(?:criteria|factors?)",
    r"adjectival\s+rating",
    r"(?:acceptable|unacceptable|outstanding|marginal)\s+rating",
    r"technical\s+acceptability",
]

# Patterns indicating format instructions (not content requirements)
FORMAT_PATTERNS = [
    r"font\s+(?:size|type)",
    r"(?:single|double)\s+(?:spaced?|spacing)",
    r"\d+\s+(?:point|pt)\s+(?:font|type)",
    r"(?:times\s+new\s+roman|arial|calibri)",
    r"(?:page|volume)\s+limit",
    r"margin\s+(?:of\s+)?\d+",
    r"maximum\s+(?:of\s+)?\d+\s+pages",
    r"shall\s+not\s+exceed\s+\d+\s+pages",
]

# Patterns indicating informational text
INFORMATIONAL_PATTERNS = [
    r"^note\s*:",
    r"^for\s+(?:information|reference)\s+(?:only|purposes)",
    r"^(?:the|this)\s+following\s+(?:is|are)\s+(?:provided\s+)?for\s+(?:information|reference)",
    r"^background\s*:",
    r"^overview\s*:",
    r"^purpose\s*:",
    r"not\s+(?:a\s+)?(?:requirement|mandatory)",
]


class AccuracyAuditor:
    """
    Audits extraction accuracy using stratified sampling and heuristic detection.

    The auditor provides:
    1. Stratified random sampling for manual review
    2. Automatic false positive detection using heuristics
    3. Comprehensive accuracy metrics
    4. Exportable audit reports
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the auditor.

        Args:
            seed: Random seed for reproducible sampling
        """
        self.rng = random.Random(seed)
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.boilerplate_re = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]
        self.evaluation_re = [re.compile(p, re.IGNORECASE) for p in EVALUATION_PATTERNS]
        self.format_re = [re.compile(p, re.IGNORECASE) for p in FORMAT_PATTERNS]
        self.informational_re = [re.compile(p, re.IGNORECASE) for p in INFORMATIONAL_PATTERNS]

    def create_audit_sample(
        self,
        requirements: List[Dict[str, Any]],
        sample_size: int = 50,
        stratify_by: str = "priority",
        sample_id: Optional[str] = None,
    ) -> AuditSample:
        """
        Create a stratified random sample for manual audit.

        Args:
            requirements: List of requirement dicts with 'id', 'text', 'priority', etc.
            sample_size: Total number of requirements to sample
            stratify_by: Field to stratify by (default: 'priority')
            sample_id: Optional ID for this sample batch

        Returns:
            AuditSample with stratified requirements
        """
        if not requirements:
            return AuditSample(
                sample_id=sample_id or f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                total_population=0,
                sample_size=0,
                stratification={},
                requirements=[],
            )

        # Group by stratification field
        strata: Dict[str, List[Dict]] = defaultdict(list)
        for req in requirements:
            key = req.get(stratify_by, "Unknown")
            strata[key].append(req)

        # Calculate proportional sample sizes
        total = len(requirements)
        sample_size = min(sample_size, total)
        stratification: Dict[str, int] = {}
        sampled: List[AuditedRequirement] = []

        for stratum, reqs in strata.items():
            # Proportional allocation with minimum of 1 if stratum is non-empty
            proportion = len(reqs) / total
            stratum_sample = max(1, round(sample_size * proportion))
            stratum_sample = min(stratum_sample, len(reqs))
            stratification[stratum] = stratum_sample

            # Random sample from this stratum
            selected = self.rng.sample(reqs, stratum_sample)
            for req in selected:
                audited = AuditedRequirement(
                    requirement_id=req.get("id", ""),
                    requirement_text=req.get("text", req.get("requirement_text", "")),
                    source_document=req.get("source_document", ""),
                    page_number=req.get("page_number", 0),
                    priority=req.get("priority", "Medium"),
                )
                # Run automatic checks
                self._auto_check_requirement(audited)
                sampled.append(audited)

        return AuditSample(
            sample_id=sample_id or f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_population=total,
            sample_size=len(sampled),
            stratification=stratification,
            requirements=sampled,
        )

    def _auto_check_requirement(self, req: AuditedRequirement):
        """
        Run automatic heuristic checks on a requirement.

        Sets flags and confidence scores based on pattern matching.
        """
        text = req.requirement_text.strip()
        flags = []
        confidence = 1.0

        # Check for boilerplate
        for pattern in self.boilerplate_re:
            if pattern.search(text):
                flags.append("boilerplate_match")
                confidence *= 0.3
                break

        # Check for evaluation criteria
        for pattern in self.evaluation_re:
            if pattern.search(text):
                flags.append("evaluation_criteria")
                confidence *= 0.6
                break

        # Check for format instructions
        for pattern in self.format_re:
            if pattern.search(text):
                flags.append("format_instruction")
                confidence *= 0.7
                break

        # Check for informational text
        for pattern in self.informational_re:
            if pattern.search(text):
                flags.append("informational")
                confidence *= 0.5
                break

        # Check for very short text (likely incomplete)
        if len(text) < 20:
            flags.append("too_short")
            confidence *= 0.4

        # Check for very long text (might need splitting)
        if len(text) > 1000:
            flags.append("too_long")
            confidence *= 0.8

        # Check for missing action verb
        if not re.search(r"\b(?:shall|must|will|should|may|can)\b", text, re.IGNORECASE):
            flags.append("no_action_verb")
            confidence *= 0.6

        # Check for header/footer patterns
        if re.match(r"^\s*(?:section|article|part)\s+[\dIVXivx]+\s*[-:]?\s*$", text, re.IGNORECASE):
            flags.append("header_only")
            confidence *= 0.2

        req.flags = flags
        req.confidence_score = confidence

        # Auto-mark clear false positives
        if confidence < 0.4:
            req.audit_status = AuditStatus.NEEDS_REVIEW
        if "boilerplate_match" in flags or "header_only" in flags:
            req.audit_status = AuditStatus.FALSE_POSITIVE
            req.false_positive_reason = FalsePositiveReason.BOILERPLATE

    def detect_false_positives(
        self,
        requirements: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Automatically detect likely false positives in a requirement list.

        Args:
            requirements: List of requirement dicts

        Returns:
            Tuple of (valid_requirements, likely_false_positives)
        """
        valid = []
        false_positives = []

        for req in requirements:
            text = req.get("text", req.get("requirement_text", "")).strip()
            is_fp, reason = self._is_false_positive(text)

            if is_fp:
                req["false_positive_reason"] = reason.value
                false_positives.append(req)
            else:
                valid.append(req)

        return valid, false_positives

    def _is_false_positive(self, text: str) -> Tuple[bool, Optional[FalsePositiveReason]]:
        """
        Check if text is likely a false positive requirement.

        Returns:
            Tuple of (is_false_positive, reason)
        """
        text = text.strip()

        # Empty or very short
        if len(text) < 10:
            return True, FalsePositiveReason.INCOMPLETE

        # Boilerplate
        for pattern in self.boilerplate_re:
            if pattern.search(text):
                return True, FalsePositiveReason.BOILERPLATE

        # Header/footer only
        if re.match(r"^\s*(?:section|article|part)\s+[\dIVXivx]+\s*[-:]?\s*$", text, re.IGNORECASE):
            return True, FalsePositiveReason.HEADER_FOOTER

        # Evaluation criteria only (not a requirement)
        eval_matches = sum(1 for p in self.evaluation_re if p.search(text))
        if eval_matches >= 2 and not re.search(r"\b(?:shall|must)\b", text, re.IGNORECASE):
            return True, FalsePositiveReason.EVALUATION_CRITERIA

        # Format instruction only (not content requirement)
        format_matches = sum(1 for p in self.format_re if p.search(text))
        if format_matches >= 2:
            return True, FalsePositiveReason.FORMAT_INSTRUCTION

        # Informational text
        for pattern in self.informational_re:
            if pattern.match(text):
                return True, FalsePositiveReason.INFORMATIONAL

        return False, None

    def report_metrics(self, sample: AuditSample) -> AuditMetrics:
        """
        Calculate accuracy metrics from an audited sample.

        Args:
            sample: AuditSample with completed audit statuses

        Returns:
            AuditMetrics with accuracy statistics
        """
        if not sample.requirements:
            return AuditMetrics(
                total_sampled=0,
                verified_count=0,
                false_positive_count=0,
                needs_review_count=0,
                accuracy_rate=0.0,
                confidence_interval_low=0.0,
                confidence_interval_high=0.0,
                false_positive_breakdown={},
                flagged_patterns={},
            )

        # Count by status
        verified = 0
        false_positive = 0
        needs_review = 0
        fp_breakdown: Dict[str, int] = defaultdict(int)
        flag_counts: Dict[str, int] = defaultdict(int)

        for req in sample.requirements:
            if req.audit_status == AuditStatus.VERIFIED:
                verified += 1
            elif req.audit_status == AuditStatus.FALSE_POSITIVE:
                false_positive += 1
                if req.false_positive_reason:
                    fp_breakdown[req.false_positive_reason.value] += 1
            elif req.audit_status == AuditStatus.NEEDS_REVIEW:
                needs_review += 1

            for flag in req.flags:
                flag_counts[flag] += 1

        total = len(sample.requirements)
        reviewed = verified + false_positive  # Exclude needs_review from accuracy calc

        # Calculate accuracy rate (verified / reviewed)
        accuracy = verified / reviewed if reviewed > 0 else 0.0

        # Calculate 95% confidence interval using Wilson score
        ci_low, ci_high = self._wilson_confidence_interval(verified, reviewed)

        return AuditMetrics(
            total_sampled=total,
            verified_count=verified,
            false_positive_count=false_positive,
            needs_review_count=needs_review,
            accuracy_rate=accuracy,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            false_positive_breakdown=dict(fp_breakdown),
            flagged_patterns=dict(flag_counts),
        )

    def _wilson_confidence_interval(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval.

        More accurate than normal approximation for small samples.
        """
        if total == 0:
            return 0.0, 0.0

        import math

        z = 1.96  # 95% confidence
        p = successes / total
        n = total

        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator

        return max(0.0, center - spread), min(1.0, center + spread)

    def export_audit_report(self, sample: AuditSample, metrics: AuditMetrics) -> Dict[str, Any]:
        """
        Export audit results as a structured report.

        Args:
            sample: The audited sample
            metrics: Calculated metrics

        Returns:
            Dict suitable for JSON serialization or display
        """
        return {
            "audit_id": sample.sample_id,
            "audit_date": sample.created_at.isoformat(),
            "population": {
                "total_requirements": sample.total_population,
                "sample_size": sample.sample_size,
                "sampling_rate": sample.sample_size / sample.total_population if sample.total_population > 0 else 0,
                "stratification": sample.stratification,
            },
            "metrics": {
                "accuracy_rate": f"{metrics.accuracy_rate:.1%}",
                "confidence_interval": f"{metrics.confidence_interval_low:.1%} - {metrics.confidence_interval_high:.1%}",
                "verified_count": metrics.verified_count,
                "false_positive_count": metrics.false_positive_count,
                "needs_review_count": metrics.needs_review_count,
            },
            "false_positive_analysis": {
                "total": metrics.false_positive_count,
                "by_reason": metrics.false_positive_breakdown,
            },
            "flag_distribution": metrics.flagged_patterns,
            "requirements": [
                {
                    "id": req.requirement_id,
                    "text": req.requirement_text[:200] + "..." if len(req.requirement_text) > 200 else req.requirement_text,
                    "status": req.audit_status.value,
                    "confidence": f"{req.confidence_score:.0%}",
                    "flags": req.flags,
                    "false_positive_reason": req.false_positive_reason.value if req.false_positive_reason else None,
                }
                for req in sample.requirements
            ],
        }

    def audit_extraction(
        self,
        requirements: List[Dict[str, Any]],
        sample_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Run a complete audit on extracted requirements.

        Convenience method that:
        1. Creates a stratified sample
        2. Runs automatic false positive detection
        3. Calculates metrics
        4. Returns a full report

        Args:
            requirements: List of extracted requirements
            sample_size: Number of requirements to sample

        Returns:
            Complete audit report
        """
        # Create stratified sample
        sample = self.create_audit_sample(requirements, sample_size)

        # Auto-detect false positives (already done in create_audit_sample)
        # but we can run detection on full set too
        valid, fps = self.detect_false_positives(requirements)

        # Calculate metrics
        metrics = self.report_metrics(sample)

        # Build report
        report = self.export_audit_report(sample, metrics)

        # Add full dataset analysis
        report["full_dataset_analysis"] = {
            "total_requirements": len(requirements),
            "estimated_valid": len(valid),
            "estimated_false_positives": len(fps),
            "estimated_accuracy": len(valid) / len(requirements) if requirements else 0,
        }

        return report


# Factory function
def get_accuracy_auditor(seed: Optional[int] = None) -> AccuracyAuditor:
    """Get an AccuracyAuditor instance."""
    return AccuracyAuditor(seed=seed)


# Quick audit function for convenience
def audit_requirements(
    requirements: List[Dict[str, Any]],
    sample_size: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quick audit function for extracted requirements.

    Args:
        requirements: List of requirement dicts
        sample_size: Number to sample for manual review
        seed: Random seed for reproducibility

    Returns:
        Audit report dict
    """
    auditor = AccuracyAuditor(seed=seed)
    return auditor.audit_extraction(requirements, sample_size)
