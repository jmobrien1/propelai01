"""
PropelAI v3.0: Resilient Extraction Test Suite

Phase 4 testing to verify:
1. Reproducibility - same inputs produce same outputs
2. Order independence - document order doesn't affect results
3. Filename robustness - typos and variations are handled
4. Ground truth validation - accuracy against annotated test sets
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestResilientExtraction:
    """Test suite for resilient extraction pipeline"""

    @pytest.fixture
    def extractor(self):
        """Create resilient extractor instance"""
        from agents.enhanced_compliance.resilient_extractor import create_resilient_extractor
        return create_resilient_extractor()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            {
                'filename': 'Attachment 1. Stament Of Work.pdf',  # Typo intentional
                'text': '''
                STATEMENT OF WORK

                1.0 SCOPE
                The contractor shall provide 24/7 network operations center support.

                2.0 REQUIREMENTS
                2.1 The contractor shall maintain 99.9% uptime.
                2.2 The contractor shall respond to incidents within 15 minutes.
                2.3 Personnel should have Security+ certification.
                2.4 The contractor may propose additional monitoring tools.

                3.0 DELIVERABLES
                3.1 The contractor shall provide monthly reports.
                ''',
                'pages': ['STATEMENT OF WORK\n1.0 SCOPE...', '2.0 REQUIREMENTS...', '3.0 DELIVERABLES...']
            },
            {
                'filename': 'Solicitation.pdf',
                'text': '''
                SECTION L - INSTRUCTIONS TO OFFERORS

                L.1 PROPOSAL SUBMISSION
                Offerors shall submit proposals electronically.

                L.2 FORMAT REQUIREMENTS
                The technical volume shall not exceed 50 pages.
                Font size must be 12-point minimum.

                SECTION M - EVALUATION FACTORS

                M.1 EVALUATION CRITERIA
                Proposals will be evaluated on technical approach.
                The Government will assess past performance.
                ''',
                'pages': ['SECTION L...', 'SECTION M...']
            }
        ]

    def test_basic_extraction(self, extractor, sample_documents):
        """Test basic extraction works"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-001")

        assert result is not None
        assert len(result.requirements) > 0
        assert result.quality_metrics.total_documents == 2

    def test_sow_detection_with_typo(self, extractor, sample_documents):
        """Test SOW detection handles filename typos"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-002")

        # SOW should be detected despite "Stament" typo
        assert result.quality_metrics.sow_detected == True
        assert 'Stament Of Work' in result.quality_metrics.sow_source

    def test_section_c_extraction(self, extractor, sample_documents):
        """Test Section C requirements are extracted"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-003")

        section_c_count = result.quality_metrics.section_counts.get('C', 0)

        # Should have multiple Section C requirements from SOW
        assert section_c_count >= 3, f"Expected >= 3 Section C requirements, got {section_c_count}"

    def test_binding_level_detection(self, extractor, sample_documents):
        """Test binding levels are correctly detected"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-004")

        shall_count = len([r for r in result.requirements if r.binding_level == "SHALL"])
        should_count = len([r for r in result.requirements if r.binding_level == "SHOULD"])
        may_count = len([r for r in result.requirements if r.binding_level == "MAY"])

        assert shall_count >= 4, f"Expected >= 4 SHALL requirements, got {shall_count}"
        assert should_count >= 1, f"Expected >= 1 SHOULD requirement, got {should_count}"
        assert may_count >= 1, f"Expected >= 1 MAY requirement, got {may_count}"

    def test_confidence_scores(self, extractor, sample_documents):
        """Test confidence scores are assigned"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-005")

        for req in result.requirements:
            assert req.confidence is not None
            assert 0.0 <= req.confidence_score <= 1.0

        # Should have some high confidence requirements
        high_conf = result.quality_metrics.high_confidence_count
        assert high_conf > 0, "Expected some high-confidence requirements"

    def test_reproducibility(self, extractor, sample_documents):
        """Test same inputs produce same outputs"""
        result1 = extractor.extract_from_parsed(sample_documents, "TEST-006-A")
        result2 = extractor.extract_from_parsed(sample_documents, "TEST-006-B")

        # Same number of requirements
        assert len(result1.requirements) == len(result2.requirements)

        # Same text hashes
        hashes1 = {r.text_hash for r in result1.requirements}
        hashes2 = {r.text_hash for r in result2.requirements}
        assert hashes1 == hashes2, "Reproducibility failed - different requirements extracted"

    def test_order_independence(self, extractor, sample_documents):
        """Test document order doesn't affect results"""
        result1 = extractor.extract_from_parsed(sample_documents, "TEST-007-A")

        # Reverse document order
        reversed_docs = list(reversed(sample_documents))
        result2 = extractor.extract_from_parsed(reversed_docs, "TEST-007-B")

        # Should have same requirements regardless of order
        hashes1 = {r.text_hash for r in result1.requirements}
        hashes2 = {r.text_hash for r in result2.requirements}

        assert hashes1 == hashes2, "Order independence failed - different requirements extracted"

    def test_quality_metrics(self, extractor, sample_documents):
        """Test quality metrics are computed"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-008")

        metrics = result.quality_metrics
        assert metrics.total_documents == 2
        assert metrics.total_pages >= 2
        assert metrics.total_requirements > 0
        assert metrics.requirements_per_page > 0

    def test_no_silent_failures(self, extractor, sample_documents):
        """Test no requirements are silently dropped"""
        result = extractor.extract_from_parsed(sample_documents, "TEST-009")

        # Every requirement should have a section assigned (even if UNASSIGNED)
        for req in result.requirements:
            assert req.assigned_section is not None

        # Check for anomalies/warnings if Section C is low
        section_c = result.quality_metrics.section_counts.get('C', 0)
        if section_c < 5:
            # Should have warning or anomaly
            assert len(result.quality_metrics.anomalies) > 0 or len(result.quality_metrics.warnings) > 0


class TestSOWDetection:
    """Test SOW detection with various filename patterns"""

    @pytest.fixture
    def classifier(self):
        from agents.enhanced_compliance.section_classifier import SectionClassifier
        return SectionClassifier()

    @pytest.mark.parametrize("filename,expected_sow", [
        ("Attachment 1. Statement Of Work.pdf", True),
        ("Attachment 1. Stament Of Work.pdf", True),   # Typo
        ("Attachment 1. Statment Of Work.pdf", True),  # Typo
        ("SOW.pdf", True),
        ("sow_document.pdf", True),
        ("PWS.pdf", True),
        ("Performance Work Statement.pdf", True),
        ("random_attachment.pdf", False),
        ("budget_template.xlsx", False),
    ])
    def test_sow_filename_detection(self, classifier, filename, expected_sow):
        """Test SOW detection handles various filename patterns"""
        documents = [{
            'filename': filename,
            'text': 'STATEMENT OF WORK\n1.0 SCOPE\nThe contractor shall...',
            'pages': []
        }]

        sow_info = classifier.detect_sow_documents(documents)
        assert sow_info.sow_detected == expected_sow, f"Failed for {filename}"

    def test_content_based_sow_detection(self, classifier):
        """Test SOW is detected by content when filename doesn't match"""
        documents = [{
            'filename': 'attachment_1.pdf',  # No SOW in filename
            'text': '''
            STATEMENT OF WORK

            1.0 SCOPE
            The contractor shall provide support services.

            2.0 REQUIREMENTS
            The contractor shall maintain systems.
            ''',
            'pages': []
        }]

        sow_info = classifier.detect_sow_documents(documents)
        assert sow_info.sow_detected == True


class TestValidation:
    """Test validation against ground truth"""

    @pytest.fixture
    def validator(self):
        from agents.enhanced_compliance.extraction_validator import ExtractionValidator
        return ExtractionValidator()

    def test_validation_metrics(self, validator):
        """Test validation metrics computation"""
        from agents.enhanced_compliance.extraction_models import (
            RequirementCandidate, ExtractionResult, ConfidenceLevel
        )

        # Create mock extraction result
        result = ExtractionResult(rfp_id="TEST")
        result.requirements = [
            RequirementCandidate(
                id="EXT-001",
                text="The contractor shall provide 24/7 support.",
                text_hash="abc123",
                source_document="test.pdf",
                source_page=1,
                source_offset=0,
                assigned_section="C",
                binding_level="SHALL",
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9,
            ),
            RequirementCandidate(
                id="EXT-002",
                text="Offerors shall submit proposals electronically.",
                text_hash="def456",
                source_document="test.pdf",
                source_page=2,
                source_offset=100,
                assigned_section="L",
                binding_level="SHALL",
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
            ),
        ]

        # Create mock ground truth file
        import tempfile
        gt_data = {
            "requirements": [
                {"id": "GT-001", "text": "The contractor shall provide 24/7 support.", "section": "C"},
                {"id": "GT-002", "text": "Offerors shall submit proposals electronically.", "section": "L"},
                {"id": "GT-003", "text": "This requirement was missed.", "section": "C"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(gt_data, f)
            gt_path = f.name

        try:
            metrics = validator.validate_against_ground_truth(result, gt_path)

            assert metrics.total_ground_truth == 3
            assert metrics.total_extracted == 2
            assert metrics.true_positives == 2  # Matched 2
            assert metrics.false_negatives == 1  # Missed 1
            assert metrics.recall == 2/3  # 66.7%
        finally:
            os.unlink(gt_path)


class TestReproducibility:
    """Full reproducibility test suite"""

    @pytest.fixture
    def extractor(self):
        from agents.enhanced_compliance.resilient_extractor import create_resilient_extractor
        return create_resilient_extractor()

    def test_multiple_runs_identical(self, extractor):
        """Run extraction 5 times and verify identical results"""
        documents = [{
            'filename': 'test_sow.pdf',
            'text': '''
            STATEMENT OF WORK
            1.0 The contractor shall provide services.
            2.0 The contractor shall meet all deadlines.
            3.0 Personnel should have relevant experience.
            ''',
            'pages': []
        }]

        results = []
        for i in range(5):
            result = extractor.extract_from_parsed(documents, f"REPRO-{i}")
            results.append(result)

        # All runs should have same number of requirements
        counts = [len(r.requirements) for r in results]
        assert len(set(counts)) == 1, f"Different counts: {counts}"

        # All runs should have same text hashes
        all_hashes = [frozenset(r.text_hash for r in result.requirements) for result in results]
        assert len(set(all_hashes)) == 1, "Different requirements extracted across runs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
