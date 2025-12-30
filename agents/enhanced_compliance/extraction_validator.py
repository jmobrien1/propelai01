"""
PropelAI v3.0: Extraction Validator

This module provides validation and comparison capabilities:
1. Ground truth validation against annotated test sets
2. Diff/comparison between extraction runs
3. Reproducibility verification
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .extraction_models import (
    RequirementCandidate,
    ExtractionResult,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics from validating extraction against ground truth"""
    total_ground_truth: int = 0
    total_extracted: int = 0

    # Core metrics
    true_positives: int = 0     # Correctly found
    false_negatives: int = 0    # Missed (in GT but not extracted)
    false_positives: int = 0    # Extra (extracted but not in GT)

    # Derived metrics
    precision: float = 0.0      # TP / (TP + FP)
    recall: float = 0.0         # TP / (TP + FN)
    f1_score: float = 0.0       # 2 * P * R / (P + R)

    # Section-specific
    section_metrics: Dict[str, Dict] = field(default_factory=dict)

    # Matching details
    matched_pairs: List[Tuple[str, str, float]] = field(default_factory=list)  # (gt_id, ext_id, similarity)
    missed_requirements: List[Dict] = field(default_factory=list)
    extra_requirements: List[Dict] = field(default_factory=list)

    def compute_derived_metrics(self):
        """Compute precision, recall, F1 from TP/FP/FN"""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            self.precision = 0.0

        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            self.recall = 0.0

        if self.precision + self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_ground_truth": self.total_ground_truth,
            "total_extracted": self.total_extracted,
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "section_metrics": self.section_metrics,
            "missed_count": len(self.missed_requirements),
            "extra_count": len(self.extra_requirements),
        }


@dataclass
class DiffResult:
    """Result of comparing two extraction runs"""
    run1_id: str
    run2_id: str

    # Counts
    only_in_run1: int = 0
    only_in_run2: int = 0
    in_both: int = 0

    # Changed items
    section_changes: List[Dict] = field(default_factory=list)
    confidence_changes: List[Dict] = field(default_factory=list)

    # Hash comparison
    document_hashes_match: bool = True

    # Details
    differences: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "run1_id": self.run1_id,
            "run2_id": self.run2_id,
            "only_in_run1": self.only_in_run1,
            "only_in_run2": self.only_in_run2,
            "in_both": self.in_both,
            "section_changes": len(self.section_changes),
            "confidence_changes": len(self.confidence_changes),
            "document_hashes_match": self.document_hashes_match,
            "is_identical": self.only_in_run1 == 0 and self.only_in_run2 == 0 and len(self.section_changes) == 0,
        }


class ExtractionValidator:
    """
    Validates extraction results against ground truth and
    compares between extraction runs.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def validate_against_ground_truth(
        self,
        extraction_result: ExtractionResult,
        ground_truth_path: str
    ) -> ValidationMetrics:
        """
        Validate extraction against a ground truth JSON file.

        Ground truth format expected:
        {
            "requirements": [
                {
                    "id": "GT-001",
                    "text": "The contractor shall...",
                    "section": "C",
                    "binding_level": "SHALL"
                },
                ...
            ]
        }
        """
        metrics = ValidationMetrics()

        # Load ground truth
        try:
            with open(ground_truth_path, 'r') as f:
                ground_truth = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return metrics

        gt_requirements = ground_truth.get('requirements', [])
        metrics.total_ground_truth = len(gt_requirements)
        metrics.total_extracted = len(extraction_result.requirements)

        # Track which GT requirements have been matched
        matched_gt_ids: Set[str] = set()
        matched_ext_ids: Set[str] = set()

        # For each ground truth requirement, find best match in extracted
        for gt_req in gt_requirements:
            gt_text = gt_req.get('text', '')
            gt_id = gt_req.get('id', '')
            gt_section = gt_req.get('section', '')

            best_match = None
            best_similarity = 0.0

            for ext_req in extraction_result.requirements:
                if ext_req.id in matched_ext_ids:
                    continue

                similarity = self._compute_similarity(gt_text, ext_req.text)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = ext_req

            if best_match and best_similarity >= self.similarity_threshold:
                # Found a match
                metrics.true_positives += 1
                matched_gt_ids.add(gt_id)
                matched_ext_ids.add(best_match.id)
                metrics.matched_pairs.append((gt_id, best_match.id, best_similarity))

                # Track section-specific metrics
                if gt_section not in metrics.section_metrics:
                    metrics.section_metrics[gt_section] = {'tp': 0, 'fn': 0}
                metrics.section_metrics[gt_section]['tp'] += 1
            else:
                # Missed requirement
                metrics.false_negatives += 1
                metrics.missed_requirements.append({
                    'id': gt_id,
                    'text': gt_text[:200],
                    'section': gt_section,
                    'best_match_similarity': best_similarity,
                })

                if gt_section not in metrics.section_metrics:
                    metrics.section_metrics[gt_section] = {'tp': 0, 'fn': 0}
                metrics.section_metrics[gt_section]['fn'] += 1

        # Count false positives (extracted but not in GT)
        for ext_req in extraction_result.requirements:
            if ext_req.id not in matched_ext_ids:
                metrics.false_positives += 1
                metrics.extra_requirements.append({
                    'id': ext_req.id,
                    'text': ext_req.text[:200],
                    'section': ext_req.assigned_section,
                    'confidence': ext_req.confidence.value,
                })

        # Compute derived metrics
        metrics.compute_derived_metrics()

        # Log results
        logger.info("=" * 60)
        logger.info("GROUND TRUTH VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Ground Truth: {metrics.total_ground_truth}")
        logger.info(f"Extracted: {metrics.total_extracted}")
        logger.info(f"True Positives: {metrics.true_positives}")
        logger.info(f"False Negatives (Missed): {metrics.false_negatives}")
        logger.info(f"False Positives (Extra): {metrics.false_positives}")
        logger.info(f"Precision: {metrics.precision:.2%}")
        logger.info(f"Recall: {metrics.recall:.2%}")
        logger.info(f"F1 Score: {metrics.f1_score:.2%}")
        logger.info("=" * 60)

        return metrics

    def compare_runs(
        self,
        run1: ExtractionResult,
        run2: ExtractionResult
    ) -> DiffResult:
        """
        Compare two extraction runs to verify reproducibility.
        """
        diff = DiffResult(run1_id=run1.rfp_id, run2_id=run2.rfp_id)

        # Check document hashes
        if set(run1.document_hashes) != set(run2.document_hashes):
            diff.document_hashes_match = False
            logger.warning("Document hashes don't match - inputs may be different")

        # Build lookup by text hash
        run1_by_hash = {r.text_hash: r for r in run1.requirements}
        run2_by_hash = {r.text_hash: r for r in run2.requirements}

        # Find items in both
        common_hashes = set(run1_by_hash.keys()) & set(run2_by_hash.keys())
        diff.in_both = len(common_hashes)

        # Check for differences in common items
        for hash_val in common_hashes:
            r1 = run1_by_hash[hash_val]
            r2 = run2_by_hash[hash_val]

            if r1.assigned_section != r2.assigned_section:
                diff.section_changes.append({
                    'text_hash': hash_val,
                    'text_preview': r1.text[:100],
                    'run1_section': r1.assigned_section,
                    'run2_section': r2.assigned_section,
                })

            if r1.confidence != r2.confidence:
                diff.confidence_changes.append({
                    'text_hash': hash_val,
                    'text_preview': r1.text[:100],
                    'run1_confidence': r1.confidence.value,
                    'run2_confidence': r2.confidence.value,
                })

        # Find items only in run1
        only_run1_hashes = set(run1_by_hash.keys()) - set(run2_by_hash.keys())
        diff.only_in_run1 = len(only_run1_hashes)
        for hash_val in only_run1_hashes:
            diff.differences.append({
                'type': 'only_in_run1',
                'text_preview': run1_by_hash[hash_val].text[:100],
            })

        # Find items only in run2
        only_run2_hashes = set(run2_by_hash.keys()) - set(run1_by_hash.keys())
        diff.only_in_run2 = len(only_run2_hashes)
        for hash_val in only_run2_hashes:
            diff.differences.append({
                'type': 'only_in_run2',
                'text_preview': run2_by_hash[hash_val].text[:100],
            })

        # Log results
        logger.info("=" * 60)
        logger.info("RUN COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info(f"Run 1: {run1.rfp_id} ({len(run1.requirements)} requirements)")
        logger.info(f"Run 2: {run2.rfp_id} ({len(run2.requirements)} requirements)")
        logger.info(f"In Both: {diff.in_both}")
        logger.info(f"Only in Run 1: {diff.only_in_run1}")
        logger.info(f"Only in Run 2: {diff.only_in_run2}")
        logger.info(f"Section Changes: {len(diff.section_changes)}")
        logger.info(f"Confidence Changes: {len(diff.confidence_changes)}")
        logger.info(f"Document Hashes Match: {diff.document_hashes_match}")
        logger.info(f"Runs Identical: {diff.to_dict()['is_identical']}")
        logger.info("=" * 60)

        return diff

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity using SequenceMatcher"""
        # Normalize texts
        t1 = ' '.join(text1.lower().split())
        t2 = ' '.join(text2.lower().split())

        return SequenceMatcher(None, t1, t2).ratio()


class ReproducibilityTester:
    """
    Tests extraction reproducibility by running multiple times
    and verifying identical outputs.
    """

    def __init__(self, extractor):
        self.extractor = extractor
        self.validator = ExtractionValidator()

    def test_reproducibility(
        self,
        documents: List[Dict],
        num_runs: int = 5,
        rfp_id: str = "TEST"
    ) -> Dict:
        """
        Run extraction multiple times and verify consistency.

        Returns dict with:
        - is_reproducible: bool
        - num_runs: int
        - variations: list of differences found
        """
        results = []

        logger.info(f"Testing reproducibility with {num_runs} runs...")

        for i in range(num_runs):
            result = self.extractor.extract_from_parsed(
                documents,
                rfp_id=f"{rfp_id}-RUN-{i+1}"
            )
            results.append(result)

        # Compare all runs to first run
        baseline = results[0]
        variations = []
        all_identical = True

        for i, result in enumerate(results[1:], 2):
            diff = self.validator.compare_runs(baseline, result)

            if not diff.to_dict()['is_identical']:
                all_identical = False
                variations.append({
                    'run': i,
                    'only_in_baseline': diff.only_in_run1,
                    'only_in_this_run': diff.only_in_run2,
                    'section_changes': len(diff.section_changes),
                })

        result = {
            'is_reproducible': all_identical,
            'num_runs': num_runs,
            'total_requirements': len(baseline.requirements),
            'variations': variations,
        }

        if all_identical:
            logger.info(f"REPRODUCIBILITY TEST PASSED: All {num_runs} runs identical")
        else:
            logger.error(f"REPRODUCIBILITY TEST FAILED: {len(variations)} runs differed from baseline")

        return result

    def test_order_independence(
        self,
        documents: List[Dict],
        rfp_id: str = "TEST"
    ) -> Dict:
        """
        Test that document order doesn't affect results.
        """
        import random

        logger.info("Testing order independence...")

        # Run with original order
        original_result = self.extractor.extract_from_parsed(
            documents,
            rfp_id=f"{rfp_id}-ORIGINAL"
        )

        # Run with shuffled order (multiple times)
        variations = []
        all_identical = True

        for i in range(3):
            shuffled = documents.copy()
            random.shuffle(shuffled)

            shuffled_result = self.extractor.extract_from_parsed(
                shuffled,
                rfp_id=f"{rfp_id}-SHUFFLED-{i+1}"
            )

            diff = self.validator.compare_runs(original_result, shuffled_result)

            if not diff.to_dict()['is_identical']:
                all_identical = False
                variations.append({
                    'shuffle': i + 1,
                    'differences': len(diff.differences),
                    'section_changes': len(diff.section_changes),
                })

        result = {
            'is_order_independent': all_identical,
            'num_shuffles': 3,
            'total_requirements': len(original_result.requirements),
            'variations': variations,
        }

        if all_identical:
            logger.info("ORDER INDEPENDENCE TEST PASSED")
        else:
            logger.error(f"ORDER INDEPENDENCE TEST FAILED: {len(variations)} shuffles differed")

        return result
