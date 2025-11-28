"""
PropelAI CTM v3.0 Test Suite

Tests the enhanced CTM data models and extractor against real-world RFP patterns
from the NLM analysis (IDES, NIH, Colorado, ITS88).

Run with: python -m pytest test_ctm_v3.py -v
Or standalone: python test_ctm_v3.py

Author: PropelAI Team
Version: 3.0.0
Date: November 28, 2025
"""

import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enhanced_compliance import (
    # Data Models
    ScoringType,
    ResponseFormat,
    RequirementType,
    RFPSection,
    ComplianceStatus,
    PageLimit,
    FormattingRequirement,
    EvidenceRequirement,
    KeyPersonnelRequirement,
    EnhancedRequirement,
    ComplianceMatrix,
    
    # Factory Functions
    create_pass_fail_requirement,
    create_weighted_requirement,
    create_future_diligence_requirement,
    
    # Extractor
    EnhancedCTMExtractor,
    process_requirements_batch,
    
    # Version
    __version__,
)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with formatting."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {test_name}")
    if details and not passed:
        print(f"         {details}")


def test_scoring_type_extraction():
    """Test extraction of scoring types from RFP text."""
    print("\n=== Test: Scoring Type Extraction ===")
    
    extractor = EnhancedCTMExtractor()
    all_passed = True
    
    # Test cases based on NLM analysis
    test_cases = [
        # Pass/Fail cases
        {
            "text": "The Offeror must meet this mandatory requirement. Failure to comply will result in a non-responsive determination.",
            "expected_type": ScoringType.PASS_FAIL,
            "expected_points": None,
            "name": "Non-responsive determination"
        },
        {
            "text": "This is a Go/No-Go requirement. Offerors must provide certification.",
            "expected_type": ScoringType.PASS_FAIL,
            "expected_points": None,
            "name": "Go/No-Go requirement"
        },
        {
            "text": "Pass/Fail: The vendor shall provide proof of insurance.",
            "expected_type": ScoringType.PASS_FAIL,
            "expected_points": None,
            "name": "Explicit Pass/Fail"
        },
        
        # Weighted/Points cases (IDES style)
        {
            "text": "F.3.22 Claimant Portal and Application Development. Maximum Points: 40",
            "expected_type": ScoringType.WEIGHTED,
            "expected_points": 40,
            "name": "IDES 40-point section"
        },
        {
            "text": "This criterion is worth up to 150 points based on demonstrated experience.",
            "expected_type": ScoringType.WEIGHTED,
            "expected_points": 150,
            "name": "150 points experience"
        },
        {
            "text": "Technical Approach - 100 pts maximum",
            "expected_type": ScoringType.WEIGHTED,
            "expected_points": 100,
            "name": "100 pts technical"
        },
        
        # Qualitative cases (NIH style)
        {
            "text": "Proposals will be rated as Outstanding, Good, Acceptable, Marginal, or Unacceptable.",
            "expected_type": ScoringType.QUALITATIVE,
            "expected_points": None,
            "name": "Adjectival rating scale"
        },
        {
            "text": "Evaluators will identify strengths and weaknesses in the technical approach.",
            "expected_type": ScoringType.QUALITATIVE,
            "expected_points": None,
            "name": "Strength/weakness evaluation"
        },
    ]
    
    for tc in test_cases:
        scoring_type, max_points, confidence = extractor.extract_scoring_type(tc["text"])
        
        type_match = scoring_type == tc["expected_type"]
        points_match = max_points == tc["expected_points"]
        passed = type_match and points_match
        
        if not passed:
            all_passed = False
            details = f"Expected: {tc['expected_type'].value}/{tc['expected_points']}, Got: {scoring_type.value}/{max_points}"
        else:
            details = ""
        
        print_result(tc["name"], passed, details)
    
    return all_passed


def test_response_format_extraction():
    """Test extraction of response formats from RFP text."""
    print("\n=== Test: Response Format Extraction ===")
    
    extractor = EnhancedCTMExtractor()
    all_passed = True
    
    test_cases = [
        # Checkbox only (IDES F.1 style)
        {
            "text": "Check the box to indicate agreement with this term.",
            "expected": ResponseFormat.CHECKBOX_ONLY,
            "name": "Simple checkbox"
        },
        {
            "text": "☐ Agree  ☐ Disagree - Indicate your response by marking the appropriate box.",
            "expected": ResponseFormat.CHECKBOX_ONLY,
            "name": "Agree/Disagree checkbox"
        },
        
        # Checkbox with evidence (IDES F.2 style)
        {
            "text": "Mark 'Met' and provide the Proposal Section and Page Number where compliance is demonstrated.",
            "expected": ResponseFormat.CHECKBOX_WITH_EVIDENCE,
            "name": "Met + page reference"
        },
        {
            "text": "Check if compliant and cross-reference the location in your proposal.",
            "expected": ResponseFormat.CHECKBOX_WITH_EVIDENCE,
            "name": "Cross-reference requirement"
        },
        
        # Table format
        {
            "text": "Complete the following table with your proposed labor categories and rates.",
            "expected": ResponseFormat.TABLE,
            "name": "Labor rate table"
        },
        {
            "text": "Submit your past performance information using Table 5 format.",
            "expected": ResponseFormat.TABLE,
            "name": "Past performance table"
        },
        
        # Resume format
        {
            "text": "Provide resumes for all key personnel not to exceed 2 pages each.",
            "expected": ResponseFormat.RESUME,
            "name": "Key personnel resumes"
        },
        {
            "text": "Include biographical sketches for the PI and Co-PIs.",
            "expected": ResponseFormat.RESUME,
            "name": "Biographical sketches"
        },
        
        # Appendix format
        {
            "text": "Submit the Quality Management Plan as an appendix to Factor 2.",
            "expected": ResponseFormat.APPENDIX,
            "name": "QMP appendix"
        },
    ]
    
    for tc in test_cases:
        response_format, confidence = extractor.extract_response_format(tc["text"])
        
        passed = response_format == tc["expected"]
        if not passed:
            all_passed = False
            details = f"Expected: {tc['expected'].value}, Got: {response_format.value}"
        else:
            details = ""
        
        print_result(tc["name"], passed, details)
    
    return all_passed


def test_page_limit_extraction():
    """Test extraction of page limits from RFP text."""
    print("\n=== Test: Page Limit Extraction ===")
    
    extractor = EnhancedCTMExtractor()
    all_passed = True
    
    test_cases = [
        # NIH Factor limits
        {
            "text": "Factor 2 Technical Approach is limited to 100 pages.",
            "expected_value": 100,
            "name": "Factor 2 100-page limit"
        },
        {
            "text": "The total narrative for Factor 3 shall not exceed 200 pages, excluding appendices.",
            "expected_value": 200,
            "name": "Factor 3 200-page limit"
        },
        
        # Standard limits
        {
            "text": "Technical Volume: Maximum 50 pages",
            "expected_value": 50,
            "name": "Technical volume 50 pages"
        },
        {
            "text": "Your response should be no more than 25 pages total.",
            "expected_value": 25,
            "name": "25-page response limit"
        },
        
        # ITS88 double-sided
        {
            "text": "Responses are limited to 25 double-sided pages or 50 single-sided pages per category.",
            "expected_value": 25,
            "name": "ITS88 double-sided limit"
        },
    ]
    
    for tc in test_cases:
        page_limit = extractor.extract_page_limit(tc["text"])
        
        if page_limit is None:
            passed = False
            details = "No page limit extracted"
        else:
            passed = page_limit.limit_value == tc["expected_value"]
            if not passed:
                details = f"Expected: {tc['expected_value']}, Got: {page_limit.limit_value}"
            else:
                details = ""
        
        if not passed:
            all_passed = False
        
        print_result(tc["name"], passed, details)
    
    return all_passed


def test_future_diligence_extraction():
    """Test extraction of future diligence flags for EA/RFR requirements."""
    print("\n=== Test: Future Diligence Flag Extraction ===")
    
    extractor = EnhancedCTMExtractor()
    all_passed = True
    
    test_cases = [
        # ITS88 style
        {
            "text": "That information may be defined in the Requests for Quote (RFQs) issued by individual Eligible Entities under this contract.",
            "expected": True,
            "name": "ITS88 RFQ deferral"
        },
        {
            "text": "Specific requirements will be provided in subsequent task orders.",
            "expected": True,
            "name": "Task order deferral"
        },
        {
            "text": "Details to be determined at the individual order level.",
            "expected": True,
            "name": "Order-level TBD"
        },
        
        # Not deferred
        {
            "text": "The vendor shall provide all required documentation with this proposal.",
            "expected": False,
            "name": "Immediate requirement"
        },
    ]
    
    for tc in test_cases:
        is_deferred, note = extractor.extract_future_diligence(tc["text"])
        
        passed = is_deferred == tc["expected"]
        if not passed:
            all_passed = False
            details = f"Expected: {tc['expected']}, Got: {is_deferred}"
        else:
            details = ""
        
        print_result(tc["name"], passed, details)
    
    return all_passed


def test_key_personnel_extraction():
    """Test extraction of key personnel requirements."""
    print("\n=== Test: Key Personnel Requirements Extraction ===")
    
    extractor = EnhancedCTMExtractor()
    all_passed = True
    
    test_cases = [
        # IDES style
        {
            "text": "The Project Manager must have a minimum of 2 years experience implementing a similarly proposed solution.",
            "expected_role": "Project Manager",
            "expected_years": 2,
            "name": "PM 2 years similar experience"
        },
        {
            "text": "Technical Lead shall have at least 5 years of experience in systems architecture.",
            "expected_role": "Technical Lead",
            "expected_years": 5,
            "name": "Tech Lead 5 years"
        },
        {
            "text": "Program Manager requires PMP certification and 10+ years experience.",
            "expected_role": "Program Manager",
            "expected_years": 10,
            "name": "PM with PMP"
        },
    ]
    
    for tc in test_cases:
        kp_req = extractor.extract_key_personnel(tc["text"])
        
        if kp_req is None:
            passed = False
            details = "No key personnel extracted"
        else:
            role_match = tc["expected_role"].lower() in kp_req.role.lower()
            years_match = kp_req.min_years_experience == tc["expected_years"]
            passed = role_match and years_match
            if not passed:
                details = f"Expected: {tc['expected_role']}/{tc['expected_years']}yrs, Got: {kp_req.role}/{kp_req.min_years_experience}yrs"
            else:
                details = ""
        
        if not passed:
            all_passed = False
        
        print_result(tc["name"], passed, details)
    
    return all_passed


def test_constraint_extraction():
    """Test extraction of compliance constraints."""
    print("\n=== Test: Compliance Constraint Extraction ===")
    
    extractor = EnhancedCTMExtractor()
    all_passed = True
    
    test_cases = [
        # ITS88 GenAI constraint
        {
            "text": "AI/ML capabilities shall not be used to train any public models. Data must remain under Commonwealth ownership.",
            "expected_contains": "train",
            "name": "GenAI training restriction"
        },
        {
            "text": "Contractor data cannot be exported outside of the continental United States.",
            "expected_contains": "export",
            "name": "Data export restriction"
        },
    ]
    
    for tc in test_cases:
        constraint = extractor.extract_constraint(tc["text"])
        
        if constraint is None:
            passed = False
            details = "No constraint extracted"
        else:
            passed = tc["expected_contains"].lower() in constraint.lower()
            if not passed:
                details = f"Expected to contain '{tc['expected_contains']}', Got: '{constraint}'"
            else:
                details = ""
        
        if not passed:
            all_passed = False
        
        print_result(tc["name"], passed, details)
    
    return all_passed


def test_full_extraction():
    """Test full extraction of all metadata for a complex requirement."""
    print("\n=== Test: Full Extraction ===")
    
    extractor = EnhancedCTMExtractor("NIH_FACTOR")
    
    # Complex requirement combining multiple features
    text = """
    F.3.22 Claimant Portal and Application Development. Maximum Points: 40.
    The Offeror shall demonstrate capability to develop web-based portals.
    Mark 'Met' and indicate the Proposal Section and Page Number where addressed.
    The narrative for this section is limited to 15 pages.
    Use 12-point Times New Roman font with 1-inch margins.
    """
    
    req = extractor.extract_all_metadata(
        requirement_text=text,
        section_reference="F.3.22",
        rfp_section=RFPSection.SECTION_L,
        context="Desirable Requirements Section"
    )
    
    all_passed = True
    
    # Check scoring type
    passed = req.scoring_type == ScoringType.WEIGHTED
    print_result("Scoring type = WEIGHTED", passed, f"Got: {req.scoring_type.value}")
    all_passed = all_passed and passed
    
    # Check max points
    passed = req.max_points == 40
    print_result("Max points = 40", passed, f"Got: {req.max_points}")
    all_passed = all_passed and passed
    
    # Check response format
    passed = req.response_format == ResponseFormat.CHECKBOX_WITH_EVIDENCE
    print_result("Response format = CHECKBOX_WITH_EVIDENCE", passed, f"Got: {req.response_format.value}")
    all_passed = all_passed and passed
    
    # Check evidence location required
    passed = req.evidence_location_required == True
    print_result("Evidence location required", passed, f"Got: {req.evidence_location_required}")
    all_passed = all_passed and passed
    
    # Check page limit
    passed = req.page_limit is not None and req.page_limit.limit_value == 15
    print_result("Page limit = 15", passed, f"Got: {req.page_limit}")
    all_passed = all_passed and passed
    
    # Check content depth multiplier
    expected_multiplier = 40 / 25.0  # 1.6
    passed = abs(req.content_depth_multiplier - expected_multiplier) < 0.01
    print_result(f"Content depth multiplier = {expected_multiplier}", passed, f"Got: {req.content_depth_multiplier}")
    all_passed = all_passed and passed
    
    # Check is_scored property
    passed = req.is_scored == True
    print_result("is_scored = True", passed, f"Got: {req.is_scored}")
    all_passed = all_passed and passed
    
    return all_passed


def test_compliance_matrix_stats():
    """Test compliance matrix statistics calculation."""
    print("\n=== Test: Compliance Matrix Statistics ===")
    
    # Create a matrix with mixed requirements
    matrix = ComplianceMatrix(
        rfp_id="RFP-TEST-001",
        rfp_name="Test RFP",
        rfp_format="NIH_FACTOR"
    )
    
    # Add pass/fail requirements
    matrix.add_requirement(create_pass_fail_requirement(
        text="Must provide proof of insurance",
        section_ref="L.1.1"
    ))
    matrix.add_requirement(create_pass_fail_requirement(
        text="Must have valid business license",
        section_ref="L.1.2"
    ))
    
    # Add weighted requirements
    matrix.add_requirement(create_weighted_requirement(
        text="Technical approach",
        section_ref="M.1",
        max_points=100
    ))
    matrix.add_requirement(create_weighted_requirement(
        text="Past performance",
        section_ref="M.2",
        max_points=50
    ))
    
    # Add future diligence requirement
    matrix.add_requirement(create_future_diligence_requirement(
        text="Specific deliverables TBD",
        section_ref="C.1.1"
    ))
    
    stats = matrix.stats
    all_passed = True
    
    # Check total
    passed = stats["total"] == 5
    print_result("Total requirements = 5", passed, f"Got: {stats['total']}")
    all_passed = all_passed and passed
    
    # Check pass/fail count
    passed = stats["pass_fail_count"] == 2
    print_result("Pass/fail count = 2", passed, f"Got: {stats['pass_fail_count']}")
    all_passed = all_passed and passed
    
    # Check scored count
    passed = stats["scored_count"] == 2
    print_result("Scored count = 2", passed, f"Got: {stats['scored_count']}")
    all_passed = all_passed and passed
    
    # Check total max points
    passed = stats["total_max_points"] == 150
    print_result("Total max points = 150", passed, f"Got: {stats['total_max_points']}")
    all_passed = all_passed and passed
    
    # Check future diligence count
    passed = stats["future_diligence_count"] == 1
    print_result("Future diligence count = 1", passed, f"Got: {stats['future_diligence_count']}")
    all_passed = all_passed and passed
    
    # Check high risk count (pass/fail = HIGH risk)
    passed = stats["high_risk_count"] == 2
    print_result("High risk count = 2", passed, f"Got: {stats['high_risk_count']}")
    all_passed = all_passed and passed
    
    return all_passed


def test_serialization():
    """Test JSON serialization and deserialization."""
    print("\n=== Test: JSON Serialization ===")
    
    # Create a complex requirement
    original = EnhancedRequirement(
        requirement_text="Test requirement with all fields",
        section_reference="L.5.2.1",
        rfp_section=RFPSection.SECTION_L,
        requirement_type=RequirementType.TECHNICAL,
        scoring_type=ScoringType.WEIGHTED,
        max_points=75,
        response_format=ResponseFormat.NARRATIVE,
        evidence_location_required=True,
        future_diligence_required=False,
        constraint_detail="No cloud hosting outside US",
        page_limit=PageLimit(limit_value=25, excludes=["appendices"]),
        formatting=FormattingRequirement(font_size_min=12, margin_inches=1.0),
        key_personnel=KeyPersonnelRequirement(role="Technical Lead", min_years_experience=5)
    )
    
    # Serialize to dict
    data = original.to_dict()
    
    # Deserialize back
    restored = EnhancedRequirement.from_dict(data)
    
    all_passed = True
    
    # Check key fields survived round-trip
    passed = restored.requirement_text == original.requirement_text
    print_result("requirement_text preserved", passed)
    all_passed = all_passed and passed
    
    passed = restored.scoring_type == original.scoring_type
    print_result("scoring_type preserved", passed)
    all_passed = all_passed and passed
    
    passed = restored.max_points == original.max_points
    print_result("max_points preserved", passed)
    all_passed = all_passed and passed
    
    passed = restored.page_limit.limit_value == original.page_limit.limit_value
    print_result("page_limit preserved", passed)
    all_passed = all_passed and passed
    
    passed = restored.key_personnel.min_years_experience == original.key_personnel.min_years_experience
    print_result("key_personnel preserved", passed)
    all_passed = all_passed and passed
    
    return all_passed


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print(f"PropelAI CTM v{__version__} Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Scoring Type Extraction", test_scoring_type_extraction()))
    results.append(("Response Format Extraction", test_response_format_extraction()))
    results.append(("Page Limit Extraction", test_page_limit_extraction()))
    results.append(("Future Diligence Extraction", test_future_diligence_extraction()))
    results.append(("Key Personnel Extraction", test_key_personnel_extraction()))
    results.append(("Constraint Extraction", test_constraint_extraction()))
    results.append(("Full Extraction", test_full_extraction()))
    results.append(("Compliance Matrix Stats", test_compliance_matrix_stats()))
    results.append(("JSON Serialization", test_serialization()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} test suites passed")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
