"""
Test cases for Strict Constructionist volume structure detection.

v3.3: These tests verify that the Smart Outline Generator does NOT hallucinate
phantom volumes. It should only create volumes that are explicitly evidenced
in the source documents.

Test scenarios based on real-world failures:
- SENTRA "Request for Estimate" - only wanted 2 volumes, system created 4
- DOD NOC - wanted 3 volumes, system created 4 with phantom "Staffing"
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_compliance.smart_outline_generator import (
    SmartOutlineGenerator,
    ProposalStructureParser,
    ProposalVolume,
    VolumeType,
    EvidenceSource,
    ConfidenceLevel,
)


class TestProposalStructureParser:
    """Test the ProposalStructureParser component"""

    def setup_method(self):
        self.parser = ProposalStructureParser()

    def test_parse_explicit_volume_table(self):
        """Test parsing explicit volume table from Section L"""
        text = """
        The offeror shall submit the following volumes:

        Volume 1: Technical Proposal - 8 pages
        Volume 2: Cost Proposal - No limit
        Volume 3: Contracts - 2 pages
        """

        volumes, source, confidence = self.parser.parse_section_l_structure(text)

        assert len(volumes) == 3
        assert source in [EvidenceSource.TABLE, EvidenceSource.LIST]
        assert confidence == ConfidenceLevel.HIGH
        assert any(v["name"].lower().find("technical") >= 0 for v in volumes)
        assert any(v["name"].lower().find("cost") >= 0 for v in volumes)

    def test_no_phantom_volumes_for_rfe(self):
        """
        SENTRA Bug: Request for Estimate should NOT create Past Performance volume.

        The RFP only requested 2 volumes (Technical, Cost) but system created 4.
        """
        text = """
        Request for Estimate
        FA2377-25-R-B009

        Submit your estimate with the following:
        Volume 1: Technical Approach
        Volume 2: Cost Estimate

        Evaluation will be based on technical merit and price.
        """

        volumes, source, confidence = self.parser.parse_section_l_structure(text)

        # Should find exactly 2 volumes
        assert len(volumes) == 2

        # Should NOT contain Past Performance or Management
        volume_names = [v["name"].lower() for v in volumes]
        assert not any("past performance" in name for name in volume_names)
        assert not any("management" in name for name in volume_names)
        assert not any("staffing" in name for name in volume_names)

    def test_no_phantom_staffing_volume(self):
        """
        DOD NOC Bug: System created "Volume 4: Staffing" when RFP only had 3 volumes.

        The word "staffing" appeared in the SOW but was NOT a separate volume.
        """
        text = """
        Section L - Instructions to Offerors

        Volume 1: Technical
        Volume 2: Cost/Price
        Volume 3: Contracts

        The technical volume shall address staffing and key personnel.
        """

        volumes, source, confidence = self.parser.parse_section_l_structure(text)

        # Should find exactly 3 volumes
        assert len(volumes) == 3

        # Should NOT contain Staffing as a separate volume
        volume_names = [v["name"].lower() for v in volumes]
        assert not any("staffing" in name for name in volume_names)

    def test_detect_rfp_type_request_for_estimate(self):
        """Test detection of Request for Estimate RFP type"""
        text = "This is a Request for Estimate for the SENTRA program."

        rfp_type = self.parser.detect_rfp_type(text)

        assert rfp_type == "request_for_estimate"
        assert not self.parser.should_use_ucf_defaults(rfp_type)

    def test_detect_rfp_type_task_order(self):
        """Test detection of Task Order RFP type"""
        text = "Task Order Request under IDIQ contract."

        rfp_type = self.parser.detect_rfp_type(text)

        assert rfp_type in ["task_order", "idiq"]
        assert not self.parser.should_use_ucf_defaults(rfp_type)

    def test_no_volume_structure_returns_empty(self):
        """When no explicit structure found, return empty list"""
        text = """
        This RFP does not specify volume structure.
        Please submit your proposal.
        """

        volumes, source, confidence = self.parser.parse_section_l_structure(text)

        assert len(volumes) == 0
        assert source == EvidenceSource.DEFAULT
        assert confidence == ConfidenceLevel.LOW


class TestSmartOutlineGenerator:
    """Test the SmartOutlineGenerator with Strict Constructionist approach"""

    def setup_method(self):
        self.generator = SmartOutlineGenerator()

    def test_extract_volumes_with_explicit_structure(self):
        """Test that explicit volume structure is respected"""
        section_l = [
            {"text": "Volume 1: Technical Proposal - 8 pages"},
            {"text": "Volume 2: Cost Proposal"},
        ]
        section_m = [
            {"text": "Technical will be evaluated first."},
        ]

        volumes, warnings = self.generator._extract_volumes(
            section_l, section_m, "STANDARD_UCF"
        )

        assert len(volumes) == 2
        assert volumes[0].evidence_source in [EvidenceSource.TABLE, EvidenceSource.LIST]
        assert volumes[0].confidence == ConfidenceLevel.HIGH

    def test_fallback_creates_single_volume(self):
        """When no structure found, create single generic volume NOT 4-volume UCF"""
        section_l = [{"text": "Submit your proposal."}]
        section_m = [{"text": "Evaluation criteria."}]

        volumes, warnings = self.generator._extract_volumes(
            section_l, section_m, "STANDARD_UCF"
        )

        # Should create exactly 1 fallback volume
        assert len(volumes) == 1
        assert volumes[0].name == "Proposal Response"
        assert volumes[0].evidence_source == EvidenceSource.DEFAULT
        assert volumes[0].confidence == ConfidenceLevel.LOW

        # Should have a warning
        assert len(warnings) > 0
        assert "no explicit volume structure" in warnings[0].lower()

    def test_deprecated_create_default_volumes_returns_empty(self):
        """The deprecated _create_default_volumes should return empty list"""
        result = self.generator._create_default_volumes("STANDARD_UCF", [])

        assert len(result) == 0

    def test_sentra_rfe_exact_volumes(self):
        """
        Integration test: SENTRA Request for Estimate should have exactly 2 volumes.
        """
        section_l = [
            {"text": """
                Request for Estimate
                Solicitation FA2377-25-R-B009

                The offeror shall submit:
                Volume 1: Technical Approach (10 pages maximum)
                Volume 2: Cost/Price Estimate
            """},
        ]
        section_m = [
            {"text": "Technical merit and price will be evaluated."},
        ]

        volumes, warnings = self.generator._extract_volumes(
            section_l, section_m, "STANDARD_UCF"
        )

        assert len(volumes) == 2

        # Verify no phantom volumes
        volume_names_lower = [v.name.lower() for v in volumes]
        assert not any("past performance" in name for name in volume_names_lower)
        assert not any("management" in name for name in volume_names_lower)
        assert not any("staffing" in name for name in volume_names_lower)

    def test_dod_noc_three_volumes(self):
        """
        Integration test: DOD NOC should have exactly 3 volumes.
        """
        section_l = [
            {"text": """
                Section L - Instructions

                Volume 1: Technical (8 pages)
                Volume 2: Cost/Price
                Volume 3: Contracts/Administrative

                The technical volume shall describe your staffing approach.
            """},
        ]
        section_m = [
            {"text": "Factor 1: Technical Approach, Factor 2: Cost"},
        ]

        volumes, warnings = self.generator._extract_volumes(
            section_l, section_m, "DOD_UCF"
        )

        assert len(volumes) == 3

        # Verify no phantom Staffing volume
        volume_names_lower = [v.name.lower() for v in volumes]
        assert not any("staffing" in name for name in volume_names_lower)


class TestEvidenceTracking:
    """Test that evidence tracking is properly recorded"""

    def setup_method(self):
        self.generator = SmartOutlineGenerator()

    def test_volume_has_evidence_source(self):
        """Each volume should have an evidence source"""
        section_l = [{"text": "Volume 1: Technical"}]
        section_m = []

        volumes, _ = self.generator._extract_volumes(section_l, section_m, "STANDARD_UCF")

        for vol in volumes:
            assert hasattr(vol, 'evidence_source')
            assert isinstance(vol.evidence_source, EvidenceSource)

    def test_volume_has_confidence_level(self):
        """Each volume should have a confidence level"""
        section_l = [{"text": "Volume 1: Technical"}]
        section_m = []

        volumes, _ = self.generator._extract_volumes(section_l, section_m, "STANDARD_UCF")

        for vol in volumes:
            assert hasattr(vol, 'confidence')
            assert isinstance(vol.confidence, ConfidenceLevel)

    def test_table_evidence_is_high_confidence(self):
        """Volumes from table parsing should have HIGH confidence"""
        section_l = [{"text": "Volume 1: Technical - 8 pages"}]
        section_m = []

        volumes, _ = self.generator._extract_volumes(section_l, section_m, "STANDARD_UCF")

        if volumes and volumes[0].evidence_source in [EvidenceSource.TABLE, EvidenceSource.LIST]:
            assert volumes[0].confidence == ConfidenceLevel.HIGH


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
