"""
PropelAI Accuracy Regression Tests

Automated tests for validating extraction accuracy against ground truth datasets.

These tests run as part of CI/CD to catch accuracy regressions before deployment.

Thresholds:
- Precision: ≥ 85%
- Recall: ≥ 90%
- F1 Score: ≥ 87%
- Section Accuracy: ≥ 90%
- Binding Accuracy: ≥ 95%
- Mandatory Recall: ≥ 99%
- Unknown Section Rate: < 5%
"""

import pytest
import json
import os
from pathlib import Path
from typing import List, Optional

# Import validation framework
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.schemas import GroundTruthRFP, GroundTruthRequirement
from validation.metrics import (
    calculate_extraction_metrics,
    AccuracyMetrics,
    compare_metrics,
    generate_accuracy_report,
)
from validation.matching import match_requirements


# ============== Configuration ==============

GROUND_TRUTH_DIR = Path(__file__).parent.parent / "validation" / "ground_truth" / "rfps"
BASELINE_FILE = Path(__file__).parent.parent / ".accuracy_baseline.json"


# ============== Threshold Constants ==============

class AccuracyThresholds:
    """Minimum acceptable accuracy thresholds"""
    MIN_PRECISION = 0.85
    MIN_RECALL = 0.90
    MIN_F1 = 0.87
    MIN_SECTION_ACCURACY = 0.90
    MIN_BINDING_ACCURACY = 0.95
    MIN_MANDATORY_RECALL = 0.99
    MAX_UNKNOWN_RATE = 0.05
    MAX_FALSE_POSITIVE_RATE = 0.10
    MAX_CRITICAL_MISS_RATE = 0.02


# ============== Fixtures ==============

@pytest.fixture
def ground_truth_dir() -> Path:
    """Return the ground truth directory path"""
    return GROUND_TRUTH_DIR


@pytest.fixture
def available_rfps(ground_truth_dir) -> List[str]:
    """List all available ground truth RFPs"""
    if not ground_truth_dir.exists():
        return []
    return [d.name for d in ground_truth_dir.iterdir() if d.is_dir()]


@pytest.fixture
def baseline_metrics() -> Optional[AccuracyMetrics]:
    """Load baseline metrics if available"""
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            data = json.load(f)
        metrics = AccuracyMetrics()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics
    return None


def load_ground_truth(rfp_path: Path) -> Optional[GroundTruthRFP]:
    """Load ground truth from an RFP directory"""
    gt_file = rfp_path / "ground_truth.json"
    if gt_file.exists():
        return GroundTruthRFP.load(str(gt_file))
    return None


def run_extraction(rfp_path: Path) -> List:
    """
    Run the extraction pipeline on an RFP.

    This loads the documents and runs the section-aware extractor.
    """
    try:
        from agents.enhanced_compliance.section_aware_extractor import (
            SectionAwareExtractor,
            extract_requirements_structured,
        )
        from agents.enhanced_compliance.document_structure import analyze_rfp_structure
        from agents.enhanced_compliance.parser import MultiFormatParser
    except ImportError:
        pytest.skip("Extraction modules not available")
        return []

    # Find documents in the ground truth directory
    doc_dir = rfp_path / "documents"
    if not doc_dir.exists():
        pytest.skip(f"No documents directory in {rfp_path}")
        return []

    # Parse all documents
    parser = MultiFormatParser()
    documents = []

    for doc_file in doc_dir.iterdir():
        if doc_file.suffix.lower() in ['.pdf', '.docx', '.xlsx']:
            try:
                parsed = parser.parse_file(str(doc_file))
                if parsed:
                    documents.append({
                        'text': parsed.full_text,
                        'filename': parsed.filename,
                        'pages': parsed.pages,
                    })
            except Exception as e:
                print(f"Warning: Could not parse {doc_file}: {e}")

    if not documents:
        pytest.skip(f"No parseable documents in {rfp_path}")
        return []

    # Run extraction
    result = extract_requirements_structured(documents)
    return result.all_requirements


# ============== Unit Tests ==============

class TestMatchingAlgorithms:
    """Test the requirement matching algorithms"""

    def test_exact_match(self):
        """Test exact text matching"""
        gt_reqs = [
            GroundTruthRequirement(
                gt_id="GT-TEST-001",
                rfp_id="TEST",
                text="The contractor shall provide monthly status reports.",
                rfp_section="C",
                binding_level="Mandatory",
                category="TECHNICAL",
            )
        ]

        # Create a mock extracted requirement
        class MockExtracted:
            full_text = "The contractor shall provide monthly status reports."
            generated_id = "EXT-001"
            source_section = type('obj', (object,), {'value': 'C'})()
            binding_level = type('obj', (object,), {'value': 'Mandatory'})()
            category = type('obj', (object,), {'value': 'TECHNICAL'})()

        result = match_requirements([MockExtracted()], gt_reqs)

        assert result["true_positives"] == 1
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0

    def test_fuzzy_match(self):
        """Test fuzzy text matching"""
        gt_reqs = [
            GroundTruthRequirement(
                gt_id="GT-TEST-001",
                rfp_id="TEST",
                text="The contractor shall provide monthly status reports to the COR.",
                rfp_section="C",
                binding_level="Mandatory",
                category="TECHNICAL",
            )
        ]

        # Slightly different text
        class MockExtracted:
            full_text = "The contractor shall provide monthly status reports to the Contracting Officer's Representative."
            generated_id = "EXT-001"
            source_section = type('obj', (object,), {'value': 'C'})()
            binding_level = type('obj', (object,), {'value': 'Mandatory'})()
            category = type('obj', (object,), {'value': 'TECHNICAL'})()

        result = match_requirements([MockExtracted()], gt_reqs, threshold=0.6)

        # Should still match with fuzzy matching
        assert result["true_positives"] == 1

    def test_no_match_different_text(self):
        """Test that different text doesn't match"""
        gt_reqs = [
            GroundTruthRequirement(
                gt_id="GT-TEST-001",
                rfp_id="TEST",
                text="The contractor shall provide monthly status reports.",
                rfp_section="C",
                binding_level="Mandatory",
                category="TECHNICAL",
            )
        ]

        class MockExtracted:
            full_text = "The contractor shall deliver annual financial statements."
            generated_id = "EXT-001"
            source_section = type('obj', (object,), {'value': 'C'})()
            binding_level = type('obj', (object,), {'value': 'Mandatory'})()
            category = type('obj', (object,), {'value': 'TECHNICAL'})()

        result = match_requirements([MockExtracted()], gt_reqs)

        assert result["true_positives"] == 0
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1


class TestMetricsCalculation:
    """Test the metrics calculation functions"""

    def test_perfect_extraction(self):
        """Test metrics with perfect extraction (all matched)"""
        gt_rfp = GroundTruthRFP(
            rfp_id="TEST",
            solicitation_number="TEST-2024-001",
            agency="TEST",
            rfp_type="Test",
            document_format="UCF_STANDARD",
            requirements=[
                GroundTruthRequirement(
                    gt_id=f"GT-TEST-{i:03d}",
                    rfp_id="TEST",
                    text=f"The contractor shall perform task {i}.",
                    rfp_section="C",
                    binding_level="Mandatory",
                    category="TECHNICAL",
                )
                for i in range(1, 11)  # 10 requirements
            ]
        )

        # Create matching extracted requirements
        class MockExtracted:
            def __init__(self, i):
                self.full_text = f"The contractor shall perform task {i}."
                self.generated_id = f"EXT-{i:03d}"
                self.source_section = type('obj', (object,), {'value': 'C'})()
                self.binding_level = type('obj', (object,), {'value': 'Mandatory'})()
                self.category = type('obj', (object,), {'value': 'TECHNICAL'})()

        extracted = [MockExtracted(i) for i in range(1, 11)]

        metrics = calculate_extraction_metrics(extracted, gt_rfp)

        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0

    def test_partial_extraction(self):
        """Test metrics with partial extraction (some missed)"""
        gt_rfp = GroundTruthRFP(
            rfp_id="TEST",
            solicitation_number="TEST-2024-001",
            agency="TEST",
            rfp_type="Test",
            document_format="UCF_STANDARD",
            requirements=[
                GroundTruthRequirement(
                    gt_id=f"GT-TEST-{i:03d}",
                    rfp_id="TEST",
                    text=f"The contractor shall perform task {i}.",
                    rfp_section="C",
                    binding_level="Mandatory",
                    category="TECHNICAL",
                )
                for i in range(1, 11)  # 10 requirements
            ]
        )

        # Only extract 8 of 10
        class MockExtracted:
            def __init__(self, i):
                self.full_text = f"The contractor shall perform task {i}."
                self.generated_id = f"EXT-{i:03d}"
                self.source_section = type('obj', (object,), {'value': 'C'})()
                self.binding_level = type('obj', (object,), {'value': 'Mandatory'})()
                self.category = type('obj', (object,), {'value': 'TECHNICAL'})()

        extracted = [MockExtracted(i) for i in range(1, 9)]  # Only 8

        metrics = calculate_extraction_metrics(extracted, gt_rfp)

        assert metrics.precision == 1.0  # All extracted are correct
        assert metrics.recall == 0.8     # 8/10 found
        assert 0.88 < metrics.f1_score < 0.90  # F1 = 2*1*0.8/(1+0.8) ≈ 0.889

    def test_metrics_thresholds(self):
        """Test threshold checking"""
        metrics = AccuracyMetrics(
            precision=0.90,
            recall=0.95,
            f1_score=0.92,
            section_accuracy=0.88,  # Below threshold
            binding_accuracy=0.96,
            mandatory_recall=1.0,
            unknown_section_rate=0.03,
        )

        passes, failures = metrics.passes_thresholds()

        assert not passes
        assert any("section_accuracy" in f for f in failures)


# ============== Integration Tests ==============

class TestExtractionAccuracy:
    """
    Integration tests for extraction accuracy.

    These tests require ground truth datasets to be available.
    """

    @pytest.mark.skipif(
        not GROUND_TRUTH_DIR.exists() or not any(GROUND_TRUTH_DIR.iterdir()),
        reason="No ground truth datasets available"
    )
    def test_extraction_precision(self, available_rfps, ground_truth_dir):
        """Test that extraction precision meets threshold"""
        if not available_rfps:
            pytest.skip("No ground truth RFPs available")

        for rfp_name in available_rfps[:1]:  # Test first RFP only for speed
            rfp_path = ground_truth_dir / rfp_name
            gt_rfp = load_ground_truth(rfp_path)

            if not gt_rfp:
                continue

            extracted = run_extraction(rfp_path)
            if not extracted:
                continue

            metrics = calculate_extraction_metrics(extracted, gt_rfp)

            assert metrics.precision >= AccuracyThresholds.MIN_PRECISION, \
                f"Precision {metrics.precision:.3f} below threshold {AccuracyThresholds.MIN_PRECISION}"

    @pytest.mark.skipif(
        not GROUND_TRUTH_DIR.exists() or not any(GROUND_TRUTH_DIR.iterdir()),
        reason="No ground truth datasets available"
    )
    def test_extraction_recall(self, available_rfps, ground_truth_dir):
        """Test that extraction recall meets threshold"""
        if not available_rfps:
            pytest.skip("No ground truth RFPs available")

        for rfp_name in available_rfps[:1]:
            rfp_path = ground_truth_dir / rfp_name
            gt_rfp = load_ground_truth(rfp_path)

            if not gt_rfp:
                continue

            extracted = run_extraction(rfp_path)
            if not extracted:
                continue

            metrics = calculate_extraction_metrics(extracted, gt_rfp)

            assert metrics.recall >= AccuracyThresholds.MIN_RECALL, \
                f"Recall {metrics.recall:.3f} below threshold {AccuracyThresholds.MIN_RECALL}"

    @pytest.mark.skipif(
        not GROUND_TRUTH_DIR.exists() or not any(GROUND_TRUTH_DIR.iterdir()),
        reason="No ground truth datasets available"
    )
    def test_mandatory_recall(self, available_rfps, ground_truth_dir):
        """Test that mandatory requirement recall is near-perfect"""
        if not available_rfps:
            pytest.skip("No ground truth RFPs available")

        for rfp_name in available_rfps[:1]:
            rfp_path = ground_truth_dir / rfp_name
            gt_rfp = load_ground_truth(rfp_path)

            if not gt_rfp:
                continue

            extracted = run_extraction(rfp_path)
            if not extracted:
                continue

            metrics = calculate_extraction_metrics(extracted, gt_rfp)

            assert metrics.mandatory_recall >= AccuracyThresholds.MIN_MANDATORY_RECALL, \
                f"Mandatory recall {metrics.mandatory_recall:.3f} below threshold {AccuracyThresholds.MIN_MANDATORY_RECALL}"

    @pytest.mark.skipif(
        not GROUND_TRUTH_DIR.exists() or not any(GROUND_TRUTH_DIR.iterdir()),
        reason="No ground truth datasets available"
    )
    def test_unknown_section_rate(self, available_rfps, ground_truth_dir):
        """Test that UNK section rate is below threshold"""
        if not available_rfps:
            pytest.skip("No ground truth RFPs available")

        for rfp_name in available_rfps[:1]:
            rfp_path = ground_truth_dir / rfp_name
            gt_rfp = load_ground_truth(rfp_path)

            if not gt_rfp:
                continue

            extracted = run_extraction(rfp_path)
            if not extracted:
                continue

            metrics = calculate_extraction_metrics(extracted, gt_rfp)

            assert metrics.unknown_section_rate <= AccuracyThresholds.MAX_UNKNOWN_RATE, \
                f"Unknown section rate {metrics.unknown_section_rate:.3f} above threshold {AccuracyThresholds.MAX_UNKNOWN_RATE}"


class TestAccuracyRegression:
    """Test for accuracy regression against baseline"""

    def test_no_regression(self, baseline_metrics, available_rfps, ground_truth_dir):
        """Test that current accuracy doesn't regress from baseline"""
        if baseline_metrics is None:
            pytest.skip("No baseline metrics available")

        if not available_rfps:
            pytest.skip("No ground truth RFPs available")

        for rfp_name in available_rfps[:1]:
            rfp_path = ground_truth_dir / rfp_name
            gt_rfp = load_ground_truth(rfp_path)

            if not gt_rfp:
                continue

            extracted = run_extraction(rfp_path)
            if not extracted:
                continue

            current_metrics = calculate_extraction_metrics(extracted, gt_rfp)

            comparison = compare_metrics(current_metrics, baseline_metrics, threshold=0.02)

            assert not comparison["has_regression"], \
                f"Accuracy regression detected: {comparison['regressions']}"


# ============== Performance Tests ==============

class TestPerformanceBenchmarks:
    """Performance benchmarks for extraction"""

    MAX_TIME_PER_PAGE = 0.5  # seconds

    @pytest.mark.skipif(
        not GROUND_TRUTH_DIR.exists() or not any(GROUND_TRUTH_DIR.iterdir()),
        reason="No ground truth datasets available"
    )
    def test_extraction_time(self, available_rfps, ground_truth_dir):
        """Test that extraction completes within time budget"""
        import time

        if not available_rfps:
            pytest.skip("No ground truth RFPs available")

        for rfp_name in available_rfps[:1]:
            rfp_path = ground_truth_dir / rfp_name
            gt_rfp = load_ground_truth(rfp_path)

            if not gt_rfp:
                continue

            start = time.time()
            extracted = run_extraction(rfp_path)
            elapsed = time.time() - start

            if gt_rfp.total_pages > 0:
                time_per_page = elapsed / gt_rfp.total_pages

                assert time_per_page <= self.MAX_TIME_PER_PAGE, \
                    f"Extraction taking {time_per_page:.3f}s/page, max is {self.MAX_TIME_PER_PAGE}s"


# ============== Report Generation ==============

def test_report_generation():
    """Test that accuracy reports can be generated"""
    metrics = AccuracyMetrics(
        precision=0.87,
        recall=0.92,
        f1_score=0.89,
        section_accuracy=0.91,
        binding_accuracy=0.96,
        mandatory_recall=1.0,
        unknown_section_rate=0.04,
        false_positive_rate=0.08,
        critical_miss_rate=0.01,
        true_positives=180,
        false_positives=20,
        false_negatives=15,
        total_extracted=200,
        total_ground_truth=195,
    )

    # Text report
    text_report = generate_accuracy_report(metrics, "TEST-RFP", output_format="text")
    assert "ACCURACY REPORT" in text_report
    assert "PASS" in text_report or "FAIL" in text_report

    # Markdown report
    md_report = generate_accuracy_report(metrics, "TEST-RFP", output_format="markdown")
    assert "# Accuracy Report" in md_report
    assert "| Metric |" in md_report


# ============== Main ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
