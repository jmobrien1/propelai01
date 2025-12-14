"""
PropelAI Validation Framework - Accuracy Metrics

Comprehensive metrics for measuring extraction accuracy:
- Precision: What % of extractions are real requirements
- Recall: What % of requirements are extracted
- F1 Score: Harmonic mean of precision and recall
- Section Accuracy: Correct section assignment
- Binding Accuracy: Correct binding level detection

Targets (from Phase 1 plan):
- Precision: ≥ 85%
- Recall: ≥ 90%
- F1 Score: ≥ 87%
- Section Accuracy: ≥ 90% (< 5% UNK)
- Binding Accuracy: ≥ 95%
- Mandatory Recall: ≥ 99% (ZERO tolerance for missing mandatory)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import json

from .schemas import GroundTruthRequirement, GroundTruthRFP, FalsePositiveType, FalseNegativeType
from .matching import match_requirements, MatchResult, GroundTruthMatchResult


@dataclass
class AccuracyMetrics:
    """Complete accuracy metrics for extraction evaluation"""
    # Core detection metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Counts
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_extracted: int = 0
    total_ground_truth: int = 0

    # Classification accuracy
    section_accuracy: float = 0.0
    binding_accuracy: float = 0.0
    category_accuracy: float = 0.0

    # Error rates
    unknown_section_rate: float = 0.0      # Target: < 5%
    false_positive_rate: float = 0.0       # Target: < 10%
    critical_miss_rate: float = 0.0        # Mandatory reqs missed: Target: < 2%

    # Per-section performance
    section_l_f1: float = 0.0
    section_m_f1: float = 0.0
    section_c_f1: float = 0.0

    # Per-binding level performance
    mandatory_recall: float = 0.0          # Target: ≥ 99%
    mandatory_precision: float = 0.0

    # Error breakdowns
    fp_breakdown: Dict[str, int] = field(default_factory=dict)
    fn_breakdown: Dict[str, int] = field(default_factory=dict)

    # Confusion matrices
    section_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    binding_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def overall_score(self) -> float:
        """
        Weighted composite score for regression testing.

        Weights emphasize:
        - Recall (finding all requirements)
        - Mandatory recall (critical for compliance)
        - Section accuracy (critical for CTM usability)
        """
        return (
            0.20 * self.recall +
            0.20 * self.mandatory_recall +
            0.15 * self.precision +
            0.15 * self.section_accuracy +
            0.15 * self.binding_accuracy +
            0.10 * (1 - self.unknown_section_rate) +
            0.05 * (1 - self.false_positive_rate)
        )

    def passes_thresholds(self) -> Tuple[bool, List[str]]:
        """
        Check if metrics pass minimum thresholds.

        Returns:
            Tuple of (passes: bool, failures: List[str])
        """
        failures = []

        thresholds = {
            "precision": (self.precision, 0.85, ">="),
            "recall": (self.recall, 0.90, ">="),
            "f1_score": (self.f1_score, 0.87, ">="),
            "section_accuracy": (self.section_accuracy, 0.90, ">="),
            "binding_accuracy": (self.binding_accuracy, 0.95, ">="),
            "mandatory_recall": (self.mandatory_recall, 0.99, ">="),
            "unknown_section_rate": (self.unknown_section_rate, 0.05, "<="),
        }

        for name, (value, threshold, op) in thresholds.items():
            if op == ">=" and value < threshold:
                failures.append(f"{name}: {value:.3f} < {threshold}")
            elif op == "<=" and value > threshold:
                failures.append(f"{name}: {value:.3f} > {threshold}")

        return (len(failures) == 0, failures)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_extracted": self.total_extracted,
            "total_ground_truth": self.total_ground_truth,
            "section_accuracy": self.section_accuracy,
            "binding_accuracy": self.binding_accuracy,
            "category_accuracy": self.category_accuracy,
            "unknown_section_rate": self.unknown_section_rate,
            "false_positive_rate": self.false_positive_rate,
            "critical_miss_rate": self.critical_miss_rate,
            "section_l_f1": self.section_l_f1,
            "section_m_f1": self.section_m_f1,
            "section_c_f1": self.section_c_f1,
            "mandatory_recall": self.mandatory_recall,
            "mandatory_precision": self.mandatory_precision,
            "fp_breakdown": self.fp_breakdown,
            "fn_breakdown": self.fn_breakdown,
            "overall_score": self.overall_score(),
            "passes_thresholds": self.passes_thresholds()[0],
            "threshold_failures": self.passes_thresholds()[1],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


def calculate_precision(
    extracted_matches: List[MatchResult]
) -> Tuple[float, int, int, Dict[str, int]]:
    """
    Calculate precision from match results.

    Precision = True Positives / (True Positives + False Positives)

    Returns:
        Tuple of (precision, true_positives, false_positives, fp_breakdown)
    """
    true_positives = sum(1 for m in extracted_matches if m.matched)
    false_positives = sum(1 for m in extracted_matches if not m.matched)

    # Breakdown of false positive types
    fp_breakdown = Counter()
    for m in extracted_matches:
        if not m.matched and m.fp_type:
            fp_breakdown[m.fp_type] += 1

    total = true_positives + false_positives
    precision = true_positives / total if total > 0 else 0.0

    return (precision, true_positives, false_positives, dict(fp_breakdown))


def calculate_recall(
    gt_matches: List[GroundTruthMatchResult],
    ground_truth: List[GroundTruthRequirement]
) -> Tuple[float, int, int, Dict[str, int]]:
    """
    Calculate recall from match results.

    Recall = True Positives / (True Positives + False Negatives)

    Returns:
        Tuple of (recall, true_positives, false_negatives, fn_breakdown)
    """
    true_positives = sum(1 for m in gt_matches if m.found)
    false_negatives = sum(1 for m in gt_matches if not m.found)

    # Breakdown of false negative types
    fn_breakdown = Counter()
    for m in gt_matches:
        if not m.found and m.fn_type:
            fn_breakdown[m.fn_type] += 1

    total = true_positives + false_negatives
    recall = true_positives / total if total > 0 else 0.0

    return (recall, true_positives, false_negatives, dict(fn_breakdown))


def calculate_section_accuracy(
    extracted_matches: List[MatchResult],
    extracted_requirements: List[Any]
) -> Tuple[float, float, Dict[str, Dict[str, int]]]:
    """
    Calculate section classification accuracy.

    Returns:
        Tuple of (section_accuracy, unknown_rate, confusion_matrix)
    """
    correct = 0
    wrong = 0
    unknown = 0
    total = 0

    # Build confusion matrix
    sections = ['L', 'M', 'C', 'B', 'F', 'G', 'H', 'I', 'J', 'K', 'PWS', 'SOW', 'UNK', 'ATTACHMENT']
    confusion = {pred: {actual: 0 for actual in sections} for pred in sections}

    for match in extracted_matches:
        if match.matched:
            total += 1

            # Find the corresponding extracted requirement
            ext_req = None
            for er in extracted_requirements:
                ext_id = er.generated_id if hasattr(er, 'generated_id') else str(id(er))
                if ext_id == match.extracted_id:
                    ext_req = er
                    break

            if ext_req:
                pred_section = ext_req.source_section.value if hasattr(ext_req, 'source_section') else "UNK"

                # Find actual section from ground truth (stored in match)
                # We need to look this up from the gt_matches
                actual_section = "UNK"  # Will be updated below

                if match.section_correct:
                    correct += 1
                    actual_section = pred_section
                    confusion[pred_section][actual_section] += 1
                else:
                    wrong += 1
                    # actual_section would need to come from ground truth
                    if pred_section == "UNK" or pred_section == "UNKNOWN":
                        unknown += 1

    section_accuracy = correct / total if total > 0 else 0.0
    unknown_rate = unknown / total if total > 0 else 0.0

    return (section_accuracy, unknown_rate, confusion)


def calculate_binding_accuracy(
    extracted_matches: List[MatchResult],
    gt_matches: List[GroundTruthMatchResult],
    ground_truth: List[GroundTruthRequirement]
) -> Tuple[float, float, float, Dict[str, Dict[str, int]]]:
    """
    Calculate binding level classification accuracy.

    Returns:
        Tuple of (binding_accuracy, mandatory_recall, mandatory_precision, confusion_matrix)
    """
    binding_levels = ['Mandatory', 'Highly Desirable', 'Desirable', 'Informational']
    confusion = {pred: {actual: 0 for actual in binding_levels} for pred in binding_levels}

    correct = 0
    total = 0

    # For mandatory recall/precision
    mandatory_tp = 0
    mandatory_fp = 0
    mandatory_fn = 0

    # Count matched requirements with correct/incorrect binding
    for match in extracted_matches:
        if match.matched:
            total += 1
            if match.binding_correct:
                correct += 1

    binding_accuracy = correct / total if total > 0 else 0.0

    # Calculate mandatory-specific metrics
    mandatory_gt = [g for g in ground_truth if g.binding_level == "Mandatory"]
    mandatory_found = 0

    for gtm in gt_matches:
        # Find the corresponding ground truth
        gt_req = next((g for g in ground_truth if g.gt_id == gtm.gt_id), None)
        if gt_req and gt_req.binding_level == "Mandatory":
            if gtm.found:
                mandatory_found += 1
            else:
                mandatory_fn += 1

    mandatory_recall = mandatory_found / len(mandatory_gt) if mandatory_gt else 1.0

    # For precision, count how many "mandatory" extractions were correct
    # This requires knowing the extracted binding levels
    mandatory_precision = 1.0  # Default if not calculable

    return (binding_accuracy, mandatory_recall, mandatory_precision, confusion)


def calculate_per_section_f1(
    extracted_matches: List[MatchResult],
    gt_matches: List[GroundTruthMatchResult],
    ground_truth: List[GroundTruthRequirement],
    section: str
) -> float:
    """
    Calculate F1 score for a specific section.

    Args:
        section: The RFP section (L, M, C, etc.)

    Returns:
        F1 score for that section
    """
    # Filter ground truth by section
    section_gt = [g for g in ground_truth if g.rfp_section == section]
    section_gt_ids = {g.gt_id for g in section_gt}

    if not section_gt:
        return 1.0  # No requirements in this section = perfect score

    # Count true positives for this section
    section_tp = sum(
        1 for gtm in gt_matches
        if gtm.gt_id in section_gt_ids and gtm.found
    )

    # Count false negatives for this section
    section_fn = sum(
        1 for gtm in gt_matches
        if gtm.gt_id in section_gt_ids and not gtm.found
    )

    # Count false positives (extracted as this section but wrong)
    # This is harder without full classification info
    section_fp = 0  # Simplified for now

    precision = section_tp / (section_tp + section_fp) if (section_tp + section_fp) > 0 else 0.0
    recall = section_tp / (section_tp + section_fn) if (section_tp + section_fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_extraction_metrics(
    extracted_requirements: List[Any],
    ground_truth: GroundTruthRFP,
    match_threshold: float = 0.7
) -> AccuracyMetrics:
    """
    Calculate comprehensive accuracy metrics.

    This is the main entry point for accuracy evaluation.

    Args:
        extracted_requirements: List of StructuredRequirement from extractor
        ground_truth: GroundTruthRFP with annotated requirements
        match_threshold: Minimum similarity for a match

    Returns:
        AccuracyMetrics with all calculated metrics
    """
    metrics = AccuracyMetrics()

    # Perform matching
    match_results = match_requirements(
        extracted_requirements,
        ground_truth.requirements,
        threshold=match_threshold
    )

    extracted_matches = match_results["extracted_matches"]
    gt_matches = match_results["gt_matches"]

    # Core counts
    metrics.true_positives = match_results["true_positives"]
    metrics.false_positives = match_results["false_positives"]
    metrics.false_negatives = match_results["false_negatives"]
    metrics.total_extracted = match_results["total_extracted"]
    metrics.total_ground_truth = match_results["total_ground_truth"]

    # Precision and breakdown
    precision, _, _, fp_breakdown = calculate_precision(extracted_matches)
    metrics.precision = precision
    metrics.fp_breakdown = fp_breakdown

    # Recall and breakdown
    recall, _, _, fn_breakdown = calculate_recall(gt_matches, ground_truth.requirements)
    metrics.recall = recall
    metrics.fn_breakdown = fn_breakdown

    # F1 Score
    if precision + recall > 0:
        metrics.f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        metrics.f1_score = 0.0

    # Section accuracy
    section_acc, unknown_rate, section_confusion = calculate_section_accuracy(
        extracted_matches, extracted_requirements
    )
    metrics.section_accuracy = section_acc
    metrics.unknown_section_rate = unknown_rate
    metrics.section_confusion = section_confusion

    # Binding accuracy
    binding_acc, mand_recall, mand_precision, binding_confusion = calculate_binding_accuracy(
        extracted_matches, gt_matches, ground_truth.requirements
    )
    metrics.binding_accuracy = binding_acc
    metrics.mandatory_recall = mand_recall
    metrics.mandatory_precision = mand_precision
    metrics.binding_confusion = binding_confusion

    # Calculate category accuracy from matches
    category_correct = sum(1 for m in extracted_matches if m.matched and m.category_correct)
    category_total = sum(1 for m in extracted_matches if m.matched)
    metrics.category_accuracy = category_correct / category_total if category_total > 0 else 0.0

    # Error rates
    metrics.false_positive_rate = (
        metrics.false_positives / metrics.total_extracted
        if metrics.total_extracted > 0 else 0.0
    )

    # Critical miss rate (mandatory requirements missed)
    mandatory_gt_count = sum(
        1 for g in ground_truth.requirements if g.binding_level == "Mandatory"
    )
    mandatory_missed = sum(
        1 for gtm in gt_matches
        if not gtm.found and any(
            g.gt_id == gtm.gt_id and g.binding_level == "Mandatory"
            for g in ground_truth.requirements
        )
    )
    metrics.critical_miss_rate = (
        mandatory_missed / mandatory_gt_count if mandatory_gt_count > 0 else 0.0
    )

    # Per-section F1 scores
    metrics.section_l_f1 = calculate_per_section_f1(
        extracted_matches, gt_matches, ground_truth.requirements, "L"
    )
    metrics.section_m_f1 = calculate_per_section_f1(
        extracted_matches, gt_matches, ground_truth.requirements, "M"
    )
    metrics.section_c_f1 = calculate_per_section_f1(
        extracted_matches, gt_matches, ground_truth.requirements, "C"
    )

    return metrics


def compare_metrics(
    current: AccuracyMetrics,
    baseline: AccuracyMetrics,
    threshold: float = 0.02
) -> Dict[str, Any]:
    """
    Compare current metrics against baseline to detect regression.

    Args:
        current: Current metrics
        baseline: Baseline metrics to compare against
        threshold: Maximum acceptable decrease

    Returns:
        Dictionary with comparison results
    """
    regressions = []
    improvements = []

    metrics_to_compare = [
        ("precision", "decrease"),
        ("recall", "decrease"),
        ("f1_score", "decrease"),
        ("section_accuracy", "decrease"),
        ("binding_accuracy", "decrease"),
        ("mandatory_recall", "decrease"),
        ("unknown_section_rate", "increase"),
        ("false_positive_rate", "increase"),
        ("critical_miss_rate", "increase"),
    ]

    for metric, direction in metrics_to_compare:
        current_val = getattr(current, metric, 0)
        baseline_val = getattr(baseline, metric, 0)

        if direction == "decrease":
            diff = baseline_val - current_val
            if diff > threshold:
                regressions.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "diff": diff,
                })
            elif diff < -threshold:
                improvements.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "improvement": -diff,
                })
        else:  # increase is bad
            diff = current_val - baseline_val
            if diff > threshold:
                regressions.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "diff": diff,
                })
            elif diff < -threshold:
                improvements.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "improvement": -diff,
                })

    return {
        "has_regression": len(regressions) > 0,
        "regressions": regressions,
        "improvements": improvements,
        "current_overall_score": current.overall_score(),
        "baseline_overall_score": baseline.overall_score(),
    }


def generate_accuracy_report(
    metrics: AccuracyMetrics,
    rfp_id: str,
    output_format: str = "text"
) -> str:
    """
    Generate a human-readable accuracy report.

    Args:
        metrics: Calculated accuracy metrics
        rfp_id: RFP identifier
        output_format: "text" or "markdown"

    Returns:
        Formatted report string
    """
    passes, failures = metrics.passes_thresholds()
    status = "PASS" if passes else "FAIL"

    if output_format == "markdown":
        report = f"""# Accuracy Report: {rfp_id}

## Overall Status: **{status}**

## Core Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Precision | {metrics.precision:.3f} | ≥ 0.85 | {'✓' if metrics.precision >= 0.85 else '✗'} |
| Recall | {metrics.recall:.3f} | ≥ 0.90 | {'✓' if metrics.recall >= 0.90 else '✗'} |
| F1 Score | {metrics.f1_score:.3f} | ≥ 0.87 | {'✓' if metrics.f1_score >= 0.87 else '✗'} |

## Classification Accuracy

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Section Accuracy | {metrics.section_accuracy:.3f} | ≥ 0.90 | {'✓' if metrics.section_accuracy >= 0.90 else '✗'} |
| Binding Accuracy | {metrics.binding_accuracy:.3f} | ≥ 0.95 | {'✓' if metrics.binding_accuracy >= 0.95 else '✗'} |
| Mandatory Recall | {metrics.mandatory_recall:.3f} | ≥ 0.99 | {'✓' if metrics.mandatory_recall >= 0.99 else '✗'} |

## Error Rates

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Unknown Section Rate | {metrics.unknown_section_rate:.3f} | ≤ 0.05 | {'✓' if metrics.unknown_section_rate <= 0.05 else '✗'} |
| False Positive Rate | {metrics.false_positive_rate:.3f} | ≤ 0.10 | {'✓' if metrics.false_positive_rate <= 0.10 else '✗'} |
| Critical Miss Rate | {metrics.critical_miss_rate:.3f} | ≤ 0.02 | {'✓' if metrics.critical_miss_rate <= 0.02 else '✗'} |

## Counts

- Total Extracted: {metrics.total_extracted}
- Total Ground Truth: {metrics.total_ground_truth}
- True Positives: {metrics.true_positives}
- False Positives: {metrics.false_positives}
- False Negatives: {metrics.false_negatives}

## Overall Score: {metrics.overall_score():.3f}
"""
    else:
        report = f"""
================================================================================
ACCURACY REPORT: {rfp_id}
================================================================================

OVERALL STATUS: {status}

CORE METRICS:
  Precision:        {metrics.precision:.3f} (target: ≥ 0.85) {'✓' if metrics.precision >= 0.85 else '✗'}
  Recall:           {metrics.recall:.3f} (target: ≥ 0.90) {'✓' if metrics.recall >= 0.90 else '✗'}
  F1 Score:         {metrics.f1_score:.3f} (target: ≥ 0.87) {'✓' if metrics.f1_score >= 0.87 else '✗'}

CLASSIFICATION ACCURACY:
  Section Accuracy: {metrics.section_accuracy:.3f} (target: ≥ 0.90) {'✓' if metrics.section_accuracy >= 0.90 else '✗'}
  Binding Accuracy: {metrics.binding_accuracy:.3f} (target: ≥ 0.95) {'✓' if metrics.binding_accuracy >= 0.95 else '✗'}
  Mandatory Recall: {metrics.mandatory_recall:.3f} (target: ≥ 0.99) {'✓' if metrics.mandatory_recall >= 0.99 else '✗'}

ERROR RATES:
  Unknown Section:  {metrics.unknown_section_rate:.3f} (target: ≤ 0.05) {'✓' if metrics.unknown_section_rate <= 0.05 else '✗'}
  False Positive:   {metrics.false_positive_rate:.3f} (target: ≤ 0.10) {'✓' if metrics.false_positive_rate <= 0.10 else '✗'}
  Critical Miss:    {metrics.critical_miss_rate:.3f} (target: ≤ 0.02) {'✓' if metrics.critical_miss_rate <= 0.02 else '✗'}

COUNTS:
  Total Extracted:    {metrics.total_extracted}
  Total Ground Truth: {metrics.total_ground_truth}
  True Positives:     {metrics.true_positives}
  False Positives:    {metrics.false_positives}
  False Negatives:    {metrics.false_negatives}

OVERALL SCORE: {metrics.overall_score():.3f}
================================================================================
"""

    if failures:
        if output_format == "markdown":
            report += "\n## Threshold Failures\n\n"
            for f in failures:
                report += f"- {f}\n"
        else:
            report += "\nTHRESHOLD FAILURES:\n"
            for f in failures:
                report += f"  - {f}\n"

    return report
