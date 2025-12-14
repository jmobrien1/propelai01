"""
PropelAI Ground Truth Validation Framework

Validates extraction accuracy by comparing extracted requirements
against human-verified ground truth datasets.

Metrics calculated:
- Precision: % of extracted requirements that are correct
- Recall: % of ground truth requirements that were extracted
- F1 Score: Harmonic mean of precision and recall
- Binding level accuracy: % of requirements with correct binding level
- False positive rate: % of extracted items that aren't real requirements
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .section_aware_extractor import (
    BindingLevel,
    ConfidenceLevel,
    ExtractionResult,
    RequirementCategory,
    StructuredRequirement,
)


class MatchType(Enum):
    """How well does an extracted requirement match ground truth?"""
    EXACT = "exact"           # Near-identical text (>95% similarity)
    PARTIAL = "partial"       # Significant overlap (70-95% similarity)
    SEMANTIC = "semantic"     # Same meaning, different wording (manual flag)
    NO_MATCH = "no_match"     # No corresponding ground truth


@dataclass
class GroundTruthRequirement:
    """A human-verified requirement from an RFP."""
    id: str                           # Unique identifier
    rfp_reference: str                # RFP's own reference (L.4.B.2, C.3.1.a)
    full_text: str                    # Verbatim requirement text
    binding_level: str                # Mandatory, Highly Desirable, Desirable
    category: str                     # L_COMPLIANCE, TECHNICAL, EVALUATION
    source_page: int                  # Page number in source document
    verified_by: str                  # Reviewer name/ID
    verification_date: str            # ISO date
    notes: str = ""                   # Reviewer notes
    is_split_from: Optional[str] = None  # If split from a larger req


@dataclass
class GroundTruthDataset:
    """A complete ground truth dataset for an RFP."""
    rfp_id: str                       # Solicitation number
    rfp_title: str                    # Document title
    source_files: List[str]           # Source document filenames
    requirements: List[GroundTruthRequirement] = field(default_factory=list)
    created_date: str = ""
    last_updated: str = ""
    verified_by: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = self.created_date


@dataclass
class RequirementMatch:
    """A match between extracted and ground truth requirement."""
    extracted_req: StructuredRequirement
    ground_truth_req: Optional[GroundTruthRequirement]
    match_type: MatchType
    similarity_score: float           # 0.0-1.0 text similarity
    binding_level_correct: bool
    category_correct: bool
    notes: str = ""


@dataclass
class ValidationResult:
    """Results of validating extraction against ground truth."""
    rfp_id: str
    validation_date: str

    # Counts
    total_extracted: int
    total_ground_truth: int
    true_positives: int               # Correctly extracted
    false_positives: int              # Extracted but not in ground truth
    false_negatives: int              # In ground truth but not extracted

    # Metrics
    precision: float                  # TP / (TP + FP)
    recall: float                     # TP / (TP + FN)
    f1_score: float                   # 2 * (P * R) / (P + R)

    # Binding level accuracy
    binding_level_accuracy: float     # % with correct binding level

    # Confidence correlation
    high_confidence_precision: float  # Precision for HIGH confidence items
    low_confidence_precision: float   # Precision for LOW confidence items

    # Match details
    matches: List[RequirementMatch] = field(default_factory=list)
    missed_requirements: List[GroundTruthRequirement] = field(default_factory=list)
    false_positive_extractions: List[StructuredRequirement] = field(default_factory=list)

    # By category
    metrics_by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


class GroundTruthValidator:
    """
    Validates extraction accuracy against human-verified ground truth.

    Usage:
        validator = GroundTruthValidator()

        # Load ground truth
        gt_dataset = validator.load_ground_truth("path/to/ground_truth.json")

        # Run extraction
        result = extract_requirements_structured(documents)

        # Validate
        validation = validator.validate(result, gt_dataset)

        # Generate report
        report = validator.generate_report(validation)
    """

    # Similarity thresholds
    EXACT_MATCH_THRESHOLD = 0.95
    PARTIAL_MATCH_THRESHOLD = 0.70

    def __init__(self, similarity_threshold: float = 0.70):
        """
        Args:
            similarity_threshold: Minimum similarity for a match (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold

    def load_ground_truth(self, path: str) -> GroundTruthDataset:
        """Load a ground truth dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        requirements = [
            GroundTruthRequirement(**req) for req in data.get('requirements', [])
        ]

        return GroundTruthDataset(
            rfp_id=data.get('rfp_id', ''),
            rfp_title=data.get('rfp_title', ''),
            source_files=data.get('source_files', []),
            requirements=requirements,
            created_date=data.get('created_date', ''),
            last_updated=data.get('last_updated', ''),
            verified_by=data.get('verified_by', [])
        )

    def save_ground_truth(self, dataset: GroundTruthDataset, path: str) -> None:
        """Save a ground truth dataset to JSON file."""
        data = {
            'rfp_id': dataset.rfp_id,
            'rfp_title': dataset.rfp_title,
            'source_files': dataset.source_files,
            'created_date': dataset.created_date,
            'last_updated': datetime.now().isoformat(),
            'verified_by': dataset.verified_by,
            'requirements': [
                {
                    'id': req.id,
                    'rfp_reference': req.rfp_reference,
                    'full_text': req.full_text,
                    'binding_level': req.binding_level,
                    'category': req.category,
                    'source_page': req.source_page,
                    'verified_by': req.verified_by,
                    'verification_date': req.verification_date,
                    'notes': req.notes,
                    'is_split_from': req.is_split_from
                }
                for req in dataset.requirements
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def validate(self, extraction_result: ExtractionResult,
                 ground_truth: GroundTruthDataset) -> ValidationResult:
        """
        Validate extraction results against ground truth.

        Args:
            extraction_result: Results from section_aware_extractor
            ground_truth: Human-verified requirements

        Returns:
            ValidationResult with metrics and match details
        """
        matches: List[RequirementMatch] = []
        matched_gt_ids: Set[str] = set()

        # Track confidence-based metrics
        high_conf_tp = 0
        high_conf_fp = 0
        low_conf_tp = 0
        low_conf_fp = 0

        # Match each extracted requirement to ground truth
        for extracted in extraction_result.all_requirements:
            best_match = self._find_best_match(extracted, ground_truth.requirements, matched_gt_ids)

            if best_match:
                gt_req, similarity = best_match
                matched_gt_ids.add(gt_req.id)

                # Determine match type
                if similarity >= self.EXACT_MATCH_THRESHOLD:
                    match_type = MatchType.EXACT
                elif similarity >= self.PARTIAL_MATCH_THRESHOLD:
                    match_type = MatchType.PARTIAL
                else:
                    match_type = MatchType.NO_MATCH

                # Check binding level
                binding_correct = self._binding_levels_match(
                    extracted.binding_level, gt_req.binding_level
                )

                # Check category
                category_correct = self._categories_match(
                    extracted.category, gt_req.category
                )

                match = RequirementMatch(
                    extracted_req=extracted,
                    ground_truth_req=gt_req,
                    match_type=match_type,
                    similarity_score=similarity,
                    binding_level_correct=binding_correct,
                    category_correct=category_correct
                )
                matches.append(match)

                # Track confidence metrics
                is_match = match_type in [MatchType.EXACT, MatchType.PARTIAL]
                if extracted.confidence_level == ConfidenceLevel.HIGH:
                    if is_match:
                        high_conf_tp += 1
                    else:
                        high_conf_fp += 1
                elif extracted.confidence_level == ConfidenceLevel.LOW:
                    if is_match:
                        low_conf_tp += 1
                    else:
                        low_conf_fp += 1
            else:
                # No match - false positive
                match = RequirementMatch(
                    extracted_req=extracted,
                    ground_truth_req=None,
                    match_type=MatchType.NO_MATCH,
                    similarity_score=0.0,
                    binding_level_correct=False,
                    category_correct=False,
                    notes="No matching ground truth requirement"
                )
                matches.append(match)

                if extracted.confidence_level == ConfidenceLevel.HIGH:
                    high_conf_fp += 1
                elif extracted.confidence_level == ConfidenceLevel.LOW:
                    low_conf_fp += 1

        # Find missed requirements (false negatives)
        missed = [
            req for req in ground_truth.requirements
            if req.id not in matched_gt_ids
        ]

        # Calculate metrics
        true_positives = sum(
            1 for m in matches
            if m.match_type in [MatchType.EXACT, MatchType.PARTIAL]
        )
        false_positives = sum(
            1 for m in matches
            if m.match_type == MatchType.NO_MATCH
        )
        false_negatives = len(missed)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Binding level accuracy (among true positives)
        binding_correct = sum(1 for m in matches if m.binding_level_correct and m.match_type != MatchType.NO_MATCH)
        binding_accuracy = binding_correct / true_positives if true_positives > 0 else 0.0

        # Confidence-based precision
        high_conf_precision = high_conf_tp / (high_conf_tp + high_conf_fp) if (high_conf_tp + high_conf_fp) > 0 else 0.0
        low_conf_precision = low_conf_tp / (low_conf_tp + low_conf_fp) if (low_conf_tp + low_conf_fp) > 0 else 0.0

        # Metrics by category
        metrics_by_category = self._calculate_category_metrics(matches, missed)

        # False positive extractions
        false_positive_extractions = [
            m.extracted_req for m in matches if m.match_type == MatchType.NO_MATCH
        ]

        return ValidationResult(
            rfp_id=ground_truth.rfp_id,
            validation_date=datetime.now().isoformat(),
            total_extracted=len(extraction_result.all_requirements),
            total_ground_truth=len(ground_truth.requirements),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            binding_level_accuracy=round(binding_accuracy, 4),
            high_confidence_precision=round(high_conf_precision, 4),
            low_confidence_precision=round(low_conf_precision, 4),
            matches=matches,
            missed_requirements=missed,
            false_positive_extractions=false_positive_extractions,
            metrics_by_category=metrics_by_category
        )

    def _find_best_match(self, extracted: StructuredRequirement,
                         ground_truth_reqs: List[GroundTruthRequirement],
                         already_matched: Set[str]) -> Optional[Tuple[GroundTruthRequirement, float]]:
        """Find the best matching ground truth requirement for an extracted one."""
        best_match = None
        best_similarity = 0.0

        for gt_req in ground_truth_reqs:
            # Skip already matched
            if gt_req.id in already_matched:
                continue

            # Calculate text similarity
            similarity = self._calculate_similarity(extracted.full_text, gt_req.full_text)

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = gt_req

        if best_match:
            return (best_match, best_similarity)
        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings."""
        # Normalize texts
        t1 = self._normalize_text(text1)
        t2 = self._normalize_text(text2)

        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, t1, t2).ratio()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation variations
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    def _binding_levels_match(self, extracted: BindingLevel, ground_truth: str) -> bool:
        """Check if binding levels match."""
        mapping = {
            BindingLevel.MANDATORY: ["mandatory", "shall", "must", "required"],
            BindingLevel.HIGHLY_DESIRABLE: ["highly desirable", "should", "highly_desirable"],
            BindingLevel.DESIRABLE: ["desirable", "may", "can"],
            BindingLevel.INFORMATIONAL: ["informational", "info", "context"]
        }

        gt_lower = ground_truth.lower().replace(" ", "_")

        for level, keywords in mapping.items():
            if extracted == level:
                return any(kw in gt_lower for kw in keywords)

        return False

    def _categories_match(self, extracted: RequirementCategory, ground_truth: str) -> bool:
        """Check if categories match."""
        mapping = {
            RequirementCategory.SECTION_L_COMPLIANCE: ["l_compliance", "section_l", "instructions"],
            RequirementCategory.TECHNICAL_REQUIREMENT: ["technical", "c_pws", "sow", "section_c"],
            RequirementCategory.EVALUATION_FACTOR: ["evaluation", "section_m", "scoring"],
            RequirementCategory.ADMINISTRATIVE: ["administrative", "admin", "section_b", "section_f"],
            RequirementCategory.ATTACHMENT_REQUIREMENT: ["attachment", "section_j", "exhibit"]
        }

        gt_lower = ground_truth.lower().replace(" ", "_")

        for category, keywords in mapping.items():
            if extracted == category:
                return any(kw in gt_lower for kw in keywords)

        return False

    def _calculate_category_metrics(self, matches: List[RequirementMatch],
                                    missed: List[GroundTruthRequirement]) -> Dict[str, Dict[str, float]]:
        """Calculate precision/recall by category."""
        metrics = {}

        categories = ["L_COMPLIANCE", "TECHNICAL", "EVALUATION", "ADMINISTRATIVE", "ATTACHMENT"]

        for cat in categories:
            # Count matches for this category
            cat_matches = [m for m in matches if self._req_in_category(m.extracted_req, cat)]
            cat_tp = sum(1 for m in cat_matches if m.match_type != MatchType.NO_MATCH)
            cat_fp = sum(1 for m in cat_matches if m.match_type == MatchType.NO_MATCH)

            # Count missed for this category
            cat_missed = [r for r in missed if cat.lower() in r.category.lower()]
            cat_fn = len(cat_missed)

            precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
            recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[cat] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'true_positives': cat_tp,
                'false_positives': cat_fp,
                'false_negatives': cat_fn
            }

        return metrics

    def _req_in_category(self, req: StructuredRequirement, category: str) -> bool:
        """Check if requirement belongs to a category."""
        mapping = {
            "L_COMPLIANCE": RequirementCategory.SECTION_L_COMPLIANCE,
            "TECHNICAL": RequirementCategory.TECHNICAL_REQUIREMENT,
            "EVALUATION": RequirementCategory.EVALUATION_FACTOR,
            "ADMINISTRATIVE": RequirementCategory.ADMINISTRATIVE,
            "ATTACHMENT": RequirementCategory.ATTACHMENT_REQUIREMENT
        }
        return req.category == mapping.get(category)

    def generate_report(self, validation: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        lines = [
            "=" * 70,
            "GROUND TRUTH VALIDATION REPORT",
            "=" * 70,
            "",
            f"RFP ID: {validation.rfp_id}",
            f"Validation Date: {validation.validation_date}",
            "",
            "-" * 70,
            "SUMMARY METRICS",
            "-" * 70,
            "",
            f"Total Extracted:     {validation.total_extracted}",
            f"Total Ground Truth:  {validation.total_ground_truth}",
            "",
            f"True Positives:      {validation.true_positives}",
            f"False Positives:     {validation.false_positives}",
            f"False Negatives:     {validation.false_negatives}",
            "",
            f"Precision:           {validation.precision:.2%}",
            f"Recall:              {validation.recall:.2%}",
            f"F1 Score:            {validation.f1_score:.2%}",
            "",
            f"Binding Level Accuracy: {validation.binding_level_accuracy:.2%}",
            "",
            "-" * 70,
            "CONFIDENCE CORRELATION",
            "-" * 70,
            "",
            f"High Confidence Precision: {validation.high_confidence_precision:.2%}",
            f"Low Confidence Precision:  {validation.low_confidence_precision:.2%}",
            "",
        ]

        # Metrics by category
        lines.extend([
            "-" * 70,
            "METRICS BY CATEGORY",
            "-" * 70,
            ""
        ])

        for cat, metrics in validation.metrics_by_category.items():
            lines.append(f"{cat}:")
            lines.append(f"  Precision: {metrics['precision']:.2%}  "
                        f"Recall: {metrics['recall']:.2%}  "
                        f"F1: {metrics['f1']:.2%}")
            lines.append(f"  TP: {metrics['true_positives']}  "
                        f"FP: {metrics['false_positives']}  "
                        f"FN: {metrics['false_negatives']}")
            lines.append("")

        # Missed requirements (false negatives)
        if validation.missed_requirements:
            lines.extend([
                "-" * 70,
                f"MISSED REQUIREMENTS ({len(validation.missed_requirements)})",
                "-" * 70,
                ""
            ])
            for i, req in enumerate(validation.missed_requirements[:10], 1):
                lines.append(f"{i}. [{req.rfp_reference}] {req.full_text[:100]}...")
                lines.append(f"   Binding: {req.binding_level}, Category: {req.category}")
                lines.append("")

            if len(validation.missed_requirements) > 10:
                lines.append(f"... and {len(validation.missed_requirements) - 10} more")
                lines.append("")

        # False positives
        if validation.false_positive_extractions:
            lines.extend([
                "-" * 70,
                f"FALSE POSITIVES ({len(validation.false_positive_extractions)})",
                "-" * 70,
                ""
            ])
            for i, req in enumerate(validation.false_positive_extractions[:10], 1):
                lines.append(f"{i}. [{req.rfp_reference}] {req.full_text[:100]}...")
                lines.append(f"   Confidence: {req.confidence_level.value}, Score: {req.confidence_score}")
                lines.append("")

            if len(validation.false_positive_extractions) > 10:
                lines.append(f"... and {len(validation.false_positive_extractions) - 10} more")
                lines.append("")

        lines.extend([
            "=" * 70,
            "END OF REPORT",
            "=" * 70
        ])

        return "\n".join(lines)

    def create_ground_truth_template(self, extraction_result: ExtractionResult,
                                     rfp_id: str, rfp_title: str,
                                     reviewer: str) -> GroundTruthDataset:
        """
        Create a ground truth template from extraction results for human review.

        This pre-populates a dataset with extracted requirements that a human
        reviewer can then verify, correct, or remove.
        """
        requirements = []

        for i, req in enumerate(extraction_result.all_requirements, 1):
            gt_req = GroundTruthRequirement(
                id=f"GT-{rfp_id[:8]}-{i:04d}",
                rfp_reference=req.rfp_reference,
                full_text=req.full_text,
                binding_level=req.binding_level.value,
                category=req.category.value,
                source_page=req.page_number,
                verified_by="",  # To be filled by reviewer
                verification_date="",  # To be filled by reviewer
                notes=f"Auto-extracted. Confidence: {req.confidence_level.value} ({req.confidence_score})"
            )
            requirements.append(gt_req)

        return GroundTruthDataset(
            rfp_id=rfp_id,
            rfp_title=rfp_title,
            source_files=[],
            requirements=requirements,
            verified_by=[reviewer]
        )


def validate_extraction(extraction_result: ExtractionResult,
                        ground_truth_path: str) -> ValidationResult:
    """
    Convenience function to validate extraction against ground truth.

    Usage:
        result = extract_requirements_structured(documents)
        validation = validate_extraction(result, "ground_truth/rfp_123.json")
        print(f"F1 Score: {validation.f1_score:.2%}")
    """
    validator = GroundTruthValidator()
    ground_truth = validator.load_ground_truth(ground_truth_path)
    return validator.validate(extraction_result, ground_truth)
