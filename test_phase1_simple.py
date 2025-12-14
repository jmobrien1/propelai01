#!/usr/bin/env python3
"""
PropelAI Phase 1 Simple Test Suite
===================================

This version tests components individually without triggering
problematic dependency chains.

Usage: python test_phase1_simple.py
"""

import sys
import traceback

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_pass(text):
    print(f"{GREEN}✓ PASS:{RESET} {text}")

def print_fail(text, error=None):
    print(f"{RED}✗ FAIL:{RESET} {text}")
    if error:
        print(f"  {RED}Error: {error}{RESET}")

def print_skip(text, reason=None):
    print(f"{YELLOW}⊘ SKIP:{RESET} {text}")
    if reason:
        print(f"  {YELLOW}Reason: {reason}{RESET}")

def print_info(text):
    print(f"  → {text}")


# ============================================================================
# TEST 1: Metadata Extractor (Pure Python, no external deps)
# ============================================================================

def test_metadata_extractor():
    """Test metadata extraction - works with just Python standard library"""
    print_header("TEST 1: Metadata Extractor")

    tests_passed = 0
    tests_failed = 0

    try:
        # Direct import avoiding __init__.py chain
        from agents.enhanced_compliance.metadata_extractor import (
            RFPMetadataExtractor, RFPMetadata, ContractType, SetAsideType,
            extract_rfp_metadata
        )

        print_pass("metadata_extractor.py imports successfully")
        tests_passed += 1

        # Sample RFP text
        sample_rfp = """
        REQUEST FOR PROPOSAL
        Solicitation Number: 75N96025R00004

        Department of Health and Human Services
        National Institutes of Health

        Title: IT Support Services for NIH Research Programs

        NAICS Code: 541512 - Computer Systems Design Services
        Size Standard: $30 Million

        Set-Aside: Total Small Business Set-Aside

        Contract Type: Cost-Plus-Fixed-Fee (CPFF)

        Period of Performance: 5-year base period

        Proposals are due by January 15, 2025 at 2:00 PM EST

        Contracting Officer: Jane Smith
        Email: jane.smith@nih.gov
        Phone: (301) 555-1234

        Estimated Value: $50 Million ceiling

        Place of Performance: Bethesda, MD
        """

        # Test extraction
        metadata = extract_rfp_metadata(sample_rfp, "sample_rfp.pdf")

        print_info("Extraction Results:")

        if metadata.solicitation_number:
            print_pass(f"Solicitation Number: {metadata.solicitation_number}")
            tests_passed += 1
        else:
            print_fail("Solicitation Number not extracted")
            tests_failed += 1

        if metadata.agency:
            print_pass(f"Agency: {metadata.agency}")
            tests_passed += 1
        else:
            print_fail("Agency not extracted")
            tests_failed += 1

        if metadata.naics_code:
            print_pass(f"NAICS Code: {metadata.naics_code}")
            tests_passed += 1
        else:
            print_fail("NAICS Code not extracted")
            tests_failed += 1

        if metadata.contract_type.value != "UNKNOWN":
            print_pass(f"Contract Type: {metadata.contract_type.value}")
            tests_passed += 1
        else:
            print_fail("Contract Type not extracted")
            tests_failed += 1

        if metadata.proposals_due:
            print_pass(f"Due Date: {metadata.proposals_due.datetime_str}")
            tests_passed += 1
        else:
            print_fail("Due Date not extracted")
            tests_failed += 1

        print_info(f"Fields Extracted: {len(metadata.fields_extracted)}")
        print_info(f"Extraction Confidence: {metadata.extraction_confidence:.0%}")

    except Exception as e:
        print_fail("Metadata extraction test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 2: Criteria-Eval Framework (Pure Python)
# ============================================================================

def test_criteria_eval():
    """Test criteria extraction and evaluation"""
    print_header("TEST 2: Criteria-Eval Framework")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.enhanced_compliance.criteria_eval import (
            CriteriaEvaluator, CriteriaChecklist, EvaluationCriterion,
            CriterionPriority, ComplianceStatus
        )

        print_pass("criteria_eval.py imports successfully")
        tests_passed += 1

        # Sample Section L
        section_l = """
        L.4 PROPOSAL INSTRUCTIONS

        The Offeror shall describe their technical approach.
        The proposal must include a detailed methodology.
        The Offeror shall demonstrate relevant past performance.
        Offerors must provide a staffing plan.
        """

        # Sample Section M
        section_m = """
        M.1 EVALUATION FACTORS

        The Government will evaluate proposals based on technical approach.
        Past performance will be evaluated for relevance.
        """

        # Test criteria extraction
        evaluator = CriteriaEvaluator()
        checklist = evaluator.extract_criteria_from_text(
            section_l_text=section_l,
            section_m_text=section_m,
            section_id="3.0",
            section_title="Technical Approach"
        )

        print_info(f"Extracted {len(checklist.criteria)} criteria")

        if len(checklist.criteria) > 0:
            print_pass(f"Criteria extraction: {len(checklist.criteria)} criteria found")
            tests_passed += 1
        else:
            print_fail("No criteria extracted")
            tests_failed += 1

        # Test evaluation
        sample_draft = """
        Our team proposes a comprehensive technical approach.
        We will employ a structured methodology using best practices.
        """

        result = evaluator.evaluate_draft(sample_draft, checklist, 1)

        print_info(f"Evaluation Score: {result.overall_score:.1f}%")
        print_info(f"Criteria Met: {result.criteria_met}/{result.criteria_total}")
        print_info(f"Passes Threshold: {result.passes_threshold}")

        print_pass("Draft evaluation completed")
        tests_passed += 1

    except Exception as e:
        print_fail("Criteria-Eval test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 3: Deduplication (Requires numpy/sklearn)
# ============================================================================

def test_deduplication():
    """Test TF-IDF based deduplication"""
    print_header("TEST 3: TF-IDF Deduplication")

    tests_passed = 0
    tests_failed = 0

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        print_pass("numpy and scikit-learn available")
        tests_passed += 1
    except ImportError as e:
        print_skip("Deduplication test", f"Missing dependency: {e}")
        print_info("Install with: pip install numpy scikit-learn")
        return 0, 0

    try:
        from agents.enhanced_compliance.deduplication import (
            TFIDFDeduplicator, deduplicate_requirements
        )
        from agents.enhanced_compliance.models import RequirementNode

        print_pass("deduplication.py imports successfully")
        tests_passed += 1

        # Create sample requirements
        requirements = [
            RequirementNode(id="R1", text="Contractor shall provide monthly status reports"),
            RequirementNode(id="R2", text="The contractor must submit monthly status reports"),
            RequirementNode(id="R3", text="Contractor shall maintain security clearances"),
        ]

        deduplicator = TFIDFDeduplicator(similarity_threshold=0.7)
        result = deduplicator.deduplicate(requirements)

        print_info(f"Input: {len(requirements)} requirements")
        print_info(f"Output: {result.deduplicated_count} requirements")
        print_info(f"Duplicates found: {result.duplicates_found}")

        print_pass("Deduplication completed")
        tests_passed += 1

    except Exception as e:
        print_fail("Deduplication test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 4: LLM Client Base Classes (Check structure)
# ============================================================================

def test_llm_base_classes():
    """Test LLM base classes without importing providers"""
    print_header("TEST 4: LLM Base Classes")

    tests_passed = 0
    tests_failed = 0

    try:
        # Import directly from the file, not through __init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_clients",
            "agents/integrations/llm_clients.py"
        )
        llm_clients = importlib.util.module_from_spec(spec)

        # Check for required classes
        spec.loader.exec_module(llm_clients)

        # Verify key classes exist
        assert hasattr(llm_clients, 'ModelRole')
        assert hasattr(llm_clients, 'TaskType')
        assert hasattr(llm_clients, 'TokenUsage')
        assert hasattr(llm_clients, 'LLMMessage')
        assert hasattr(llm_clients, 'GenerationConfig')

        print_pass("llm_clients.py has all required classes")
        tests_passed += 1

        # Check ModelRole values
        roles = [r.value for r in llm_clients.ModelRole]
        print_info(f"ModelRole values: {roles}")

        if 'librarian' in roles and 'architect' in roles:
            print_pass("ModelRole has librarian and architect")
            tests_passed += 1
        else:
            print_fail("ModelRole missing expected values")
            tests_failed += 1

        # Check TaskType values
        tasks = [t.value for t in llm_clients.TaskType]
        print_info(f"TaskType count: {len(tasks)} task types defined")

        if len(tasks) > 5:
            print_pass(f"TaskType has {len(tasks)} task types")
            tests_passed += 1
        else:
            print_fail("TaskType has too few values")
            tests_failed += 1

    except Exception as e:
        print_fail("LLM base classes test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 5: Requirements Models
# ============================================================================

def test_requirements_models():
    """Test the enhanced compliance models"""
    print_header("TEST 5: Requirements Models")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.enhanced_compliance.models import (
            RequirementNode, RequirementType, ConfidenceLevel,
            RFPBundle, DocumentType, SourceLocation
        )

        print_pass("models.py imports successfully")
        tests_passed += 1

        # Create a requirement
        req = RequirementNode(
            id="REQ-001",
            text="The contractor shall provide monthly status reports.",
            requirement_type=RequirementType.DELIVERABLE,
            confidence=ConfidenceLevel.HIGH
        )

        print_info(f"Created requirement: {req.id}")
        print_info(f"Text hash: {req.text_hash}")
        print_info(f"Type: {req.requirement_type.value}")

        # Test to_dict
        req_dict = req.to_dict()
        assert 'id' in req_dict
        assert 'text' in req_dict
        assert 'requirement_type' in req_dict

        print_pass("RequirementNode serialization works")
        tests_passed += 1

        # Create an RFP bundle
        bundle = RFPBundle(
            solicitation_number="TEST-2025-001",
            title="Test RFP",
            agency="Test Agency"
        )

        print_info(f"Created RFP bundle: {bundle.solicitation_number}")

        print_pass("RFPBundle creation works")
        tests_passed += 1

    except Exception as e:
        print_fail("Requirements models test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 6: Existing Extractor (from before Phase 1)
# ============================================================================

def test_existing_extractor():
    """Test the existing requirement extractor"""
    print_header("TEST 6: Requirement Extractor")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.enhanced_compliance.extractor import RequirementExtractor
        from agents.enhanced_compliance.models import ParsedDocument, DocumentType

        print_pass("extractor.py imports successfully")
        tests_passed += 1

        # Create a mock parsed document
        doc = ParsedDocument(
            filepath="/test/rfp.pdf",
            filename="rfp.pdf",
            document_type=DocumentType.MAIN_SOLICITATION,
            full_text="""
            SECTION C - STATEMENT OF WORK

            C.1 The contractor shall provide IT support services.
            C.2 The contractor must maintain system availability of 99.9%.
            C.3 Monthly status reports shall be submitted by the 5th of each month.
            C.4 All personnel shall maintain appropriate security clearances.
            """,
            pages=["Page 1 content"],
            page_count=1
        )

        # Extract requirements
        extractor = RequirementExtractor(strict_mode=True)
        requirements = extractor.extract_from_document(doc)

        print_info(f"Extracted {len(requirements)} requirements")

        if len(requirements) > 0:
            print_pass(f"Requirement extraction: {len(requirements)} found")
            tests_passed += 1

            for req in requirements[:3]:
                print_info(f"  [{req.id}] {req.text[:50]}...")
        else:
            print_fail("No requirements extracted")
            tests_failed += 1

    except Exception as e:
        print_fail("Requirement extractor test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{BLUE}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║       PropelAI Phase 1 Simple Test Suite                 ║{RESET}")
    print(f"{BLUE}║       Testing without complex dependency chains          ║{RESET}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════════╝{RESET}")

    total_passed = 0
    total_failed = 0

    tests = [
        ("Metadata Extractor", test_metadata_extractor),
        ("Criteria-Eval Framework", test_criteria_eval),
        ("TF-IDF Deduplication", test_deduplication),
        ("LLM Base Classes", test_llm_base_classes),
        ("Requirements Models", test_requirements_models),
        ("Requirement Extractor", test_existing_extractor),
    ]

    for test_name, test_func in tests:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print_fail(f"{test_name} crashed", str(e))
            traceback.print_exc()
            total_failed += 1

    print_header("TEST SUMMARY")
    print(f"  Total Passed: {GREEN}{total_passed}{RESET}")
    print(f"  Total Failed: {RED}{total_failed}{RESET}")

    if total_failed == 0:
        print(f"\n{GREEN}All tests passed! Phase 1 core components are working.{RESET}")
        print(f"\n{YELLOW}Note: Full LLM integration tests require API keys.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}Some tests failed. Check the output above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
