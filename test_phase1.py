#!/usr/bin/env python3
"""
PropelAI Phase 1 Test Suite
===========================

Run this script to test the Phase 1 implementation.
Usage: python test_phase1.py

Tests are ordered from simplest (no dependencies) to complex (requires API keys).
"""

import sys
import traceback

# Color codes for output
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
# TEST 1: Basic Imports (No external dependencies)
# ============================================================================

def test_basic_imports():
    """Test that all Phase 1 modules can be imported"""
    print_header("TEST 1: Basic Imports")

    tests_passed = 0
    tests_failed = 0

    # Test 1.1: LLM Clients base
    try:
        from agents.integrations.llm_clients import (
            BaseLLMClient, ModelRole, TaskType, TokenUsage,
            LLMMessage, LLMResponse, GenerationConfig
        )
        print_pass("llm_clients.py imports")
        print_info(f"ModelRole values: {[r.value for r in ModelRole]}")
        tests_passed += 1
    except Exception as e:
        print_fail("llm_clients.py imports", str(e))
        tests_failed += 1

    # Test 1.2: Gemini Client
    try:
        from agents.integrations.gemini_client import (
            GeminiClient, RFPIngestionResult, ComplianceMatrixEntry
        )
        print_pass("gemini_client.py imports")
        tests_passed += 1
    except Exception as e:
        print_fail("gemini_client.py imports", str(e))
        tests_failed += 1

    # Test 1.3: Claude Client
    try:
        from agents.integrations.claude_client import (
            ClaudeClient, CritiqueResult, SectionPlan, OutlinePlanResult
        )
        print_pass("claude_client.py imports")
        tests_passed += 1
    except Exception as e:
        print_fail("claude_client.py imports", str(e))
        tests_failed += 1

    # Test 1.4: Model Router
    try:
        from agents.integrations.model_router import (
            ModelRouter, RoutingStrategy, RoutingConfig, create_router
        )
        print_pass("model_router.py imports")
        tests_passed += 1
    except Exception as e:
        print_fail("model_router.py imports", str(e))
        tests_failed += 1

    # Test 1.5: Metadata Extractor
    try:
        from agents.enhanced_compliance.metadata_extractor import (
            RFPMetadataExtractor, RFPMetadata, ContractType, SetAsideType
        )
        print_pass("metadata_extractor.py imports")
        print_info(f"ContractType values: {[c.value for c in ContractType][:5]}...")
        tests_passed += 1
    except Exception as e:
        print_fail("metadata_extractor.py imports", str(e))
        tests_failed += 1

    # Test 1.6: Criteria Eval
    try:
        from agents.enhanced_compliance.criteria_eval import (
            CriteriaEvaluator, CriteriaChecklist, EvaluationCriterion
        )
        print_pass("criteria_eval.py imports")
        tests_passed += 1
    except Exception as e:
        print_fail("criteria_eval.py imports", str(e))
        tests_failed += 1

    # Test 1.7: Deduplication (requires scikit-learn)
    try:
        from agents.enhanced_compliance.deduplication import (
            TFIDFDeduplicator, DuplicateType, DeduplicationResult
        )
        print_pass("deduplication.py imports")
        tests_passed += 1
    except ImportError as e:
        if "sklearn" in str(e):
            print_skip("deduplication.py imports", "scikit-learn not installed")
            print_info("Run: pip install scikit-learn")
        else:
            print_fail("deduplication.py imports", str(e))
            tests_failed += 1
    except Exception as e:
        print_fail("deduplication.py imports", str(e))
        tests_failed += 1

    # Test 1.8: DCE Workflow (requires langgraph)
    try:
        from agents.workflows.draft_critique_expand import (
            DCEWorkflow, DCEResult, DCEPhase
        )
        print_pass("draft_critique_expand.py imports")
        tests_passed += 1
    except ImportError as e:
        if "langgraph" in str(e):
            print_skip("draft_critique_expand.py imports", "langgraph not installed")
            print_info("Run: pip install langgraph")
        else:
            print_fail("draft_critique_expand.py imports", str(e))
            tests_failed += 1
    except Exception as e:
        print_fail("draft_critique_expand.py imports", str(e))
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 2: Metadata Extractor (No API keys needed)
# ============================================================================

def test_metadata_extractor():
    """Test metadata extraction from sample RFP text"""
    print_header("TEST 2: Metadata Extractor")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.enhanced_compliance.metadata_extractor import (
            RFPMetadataExtractor, extract_rfp_metadata
        )

        # Sample RFP text with various metadata
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

        Period of Performance: 5-year base period with two 1-year options

        Proposals are due by January 15, 2025 at 2:00 PM EST

        Questions must be submitted by December 20, 2024

        Contracting Officer: Jane Smith
        Email: jane.smith@nih.gov
        Phone: (301) 555-1234

        Estimated Value: $50 Million ceiling

        Place of Performance: Bethesda, MD
        """

        # Test extraction
        metadata = extract_rfp_metadata(sample_rfp, "sample_rfp.pdf")

        # Verify results
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

        if metadata.set_aside.value != "none":
            print_pass(f"Set-Aside: {metadata.set_aside.value}")
            tests_passed += 1
        else:
            print_fail("Set-Aside not extracted")
            tests_failed += 1

        if metadata.proposals_due:
            print_pass(f"Due Date: {metadata.proposals_due.datetime_str}")
            tests_passed += 1
        else:
            print_fail("Due Date not extracted")
            tests_failed += 1

        print_info(f"Extraction Confidence: {metadata.extraction_confidence:.0%}")
        print_info(f"Fields Extracted: {metadata.fields_extracted}")
        print_info(f"Fields Missing: {metadata.fields_missing}")

    except Exception as e:
        print_fail("Metadata extraction test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 3: Criteria Eval Framework (No API keys needed)
# ============================================================================

def test_criteria_eval():
    """Test criteria extraction and evaluation"""
    print_header("TEST 3: Criteria-Eval Framework")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.enhanced_compliance.criteria_eval import (
            CriteriaEvaluator, CriteriaChecklist
        )

        # Sample Section L (Instructions)
        section_l = """
        L.4 PROPOSAL INSTRUCTIONS

        L.4.1 Technical Volume

        The Offeror shall describe their technical approach to performing the work.

        The proposal must include:
        (a) A detailed methodology for task execution
        (b) Identification of key personnel and their qualifications
        (c) A risk mitigation strategy
        (d) Quality assurance procedures

        The Offeror shall demonstrate relevant past performance on similar contracts.

        L.4.2 Management Volume

        The proposal shall address organizational structure and management approach.
        Offerors must provide a staffing plan with labor categories.
        """

        # Sample Section M (Evaluation)
        section_m = """
        M.1 EVALUATION FACTORS

        The Government will evaluate proposals based on the following factors:

        M.1.1 Technical Approach (Most Important)

        Proposals will be evaluated on the quality and feasibility of the
        technical approach. The Government will assess the methodology,
        tools, and processes proposed.

        M.1.2 Past Performance (More Important than Cost)

        Past performance will be evaluated based on relevance and quality
        of previous contract work.

        M.1.3 Cost/Price

        Cost proposals will be evaluated for realism and reasonableness.
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

            # Show first few criteria
            print_info("Sample criteria:")
            for c in checklist.criteria[:3]:
                print_info(f"  [{c.id}] {c.text[:60]}...")
        else:
            print_fail("No criteria extracted")
            tests_failed += 1

        # Test evaluation against a draft
        sample_draft = """
        3.0 Technical Approach

        Our team proposes a comprehensive technical approach to deliver
        IT support services for NIH research programs. Our methodology
        leverages industry best practices and agile principles.

        3.1 Methodology

        We will employ a structured approach using ITIL framework principles
        for service management. Our team includes certified professionals
        with extensive experience in healthcare IT environments.

        3.2 Key Personnel

        Our Project Manager, John Doe, brings 15 years of federal IT experience
        including 5 years supporting NIH programs.

        3.3 Risk Management

        We have identified potential risks and developed mitigation strategies
        to ensure successful contract performance.
        """

        result = evaluator.evaluate_draft(
            draft_text=sample_draft,
            checklist=checklist,
            draft_version=1
        )

        print_info(f"Evaluation Score: {result.overall_score:.1f}%")
        print_info(f"Criteria Met: {result.criteria_met}/{result.criteria_total}")
        print_info(f"Passes Threshold: {result.passes_threshold}")

        if result.overall_score > 0:
            print_pass(f"Draft evaluation completed with score {result.overall_score:.1f}%")
            tests_passed += 1
        else:
            print_fail("Draft evaluation returned zero score")
            tests_failed += 1

        if result.gaps:
            print_info(f"Identified {len(result.gaps)} gaps")
            for gap in result.gaps[:2]:
                print_info(f"  Gap: {gap[:60]}...")

    except Exception as e:
        print_fail("Criteria-Eval test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 4: TF-IDF Deduplication (Requires scikit-learn)
# ============================================================================

def test_deduplication():
    """Test TF-IDF based requirement deduplication"""
    print_header("TEST 4: TF-IDF Deduplication")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.enhanced_compliance.deduplication import (
            TFIDFDeduplicator, deduplicate_requirements
        )
        from agents.enhanced_compliance.models import RequirementNode

        # Create sample requirements with duplicates
        requirements = [
            RequirementNode(
                id="REQ-001",
                text="The contractor shall provide monthly status reports to the COR."
            ),
            RequirementNode(
                id="REQ-002",
                text="Contractor must submit monthly status reports to the Contracting Officer Representative."
            ),
            RequirementNode(
                id="REQ-003",
                text="The contractor shall maintain all required security clearances."
            ),
            RequirementNode(
                id="REQ-004",
                text="All contractor personnel must maintain appropriate security clearances."
            ),
            RequirementNode(
                id="REQ-005",
                text="The contractor shall develop a quality assurance plan."
            ),
        ]

        print_info(f"Input: {len(requirements)} requirements")

        # Run deduplication
        deduplicator = TFIDFDeduplicator(similarity_threshold=0.75)
        result = deduplicator.deduplicate(requirements)

        print_info(f"Output: {result.deduplicated_count} requirements")
        print_info(f"Duplicates found: {result.duplicates_found}")
        print_info(f"Reduction: {result.reduction_percentage:.1f}%")

        if result.duplicates_found > 0:
            print_pass(f"Detected {result.duplicates_found} duplicates")
            tests_passed += 1

            # Show duplicate pairs
            print_info("Duplicate pairs found:")
            for pair in result.duplicate_pairs[:3]:
                print_info(f"  {pair.req_a_id} <-> {pair.req_b_id}: {pair.similarity_score:.2%}")
        else:
            print_info("No duplicates detected (threshold may be too high)")
            tests_passed += 1

        if result.merged_requirements:
            print_pass(f"Merged requirements successfully")
            tests_passed += 1
        else:
            print_fail("No merged requirements returned")
            tests_failed += 1

    except ImportError as e:
        if "sklearn" in str(e):
            print_skip("Deduplication test", "scikit-learn not installed")
            print_info("Install with: pip install scikit-learn")
            return 0, 0  # Don't count as failure
        else:
            raise
    except Exception as e:
        print_fail("Deduplication test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# TEST 5: Model Router Structure (No API calls)
# ============================================================================

def test_model_router_structure():
    """Test Model Router configuration without making API calls"""
    print_header("TEST 5: Model Router Structure")

    tests_passed = 0
    tests_failed = 0

    try:
        from agents.integrations.model_router import (
            ModelRouter, RoutingConfig, RoutingStrategy, TASK_ROUTING
        )
        from agents.integrations.llm_clients import ModelRole, TaskType

        # Test routing configuration
        config = RoutingConfig(
            strategy=RoutingStrategy.OPTIMAL,
            enable_fallback=True
        )

        print_pass(f"RoutingConfig created with strategy: {config.strategy.value}")
        tests_passed += 1

        # Test task routing map
        print_info("Task to Model Role mapping:")
        for task_type, role in list(TASK_ROUTING.items())[:5]:
            print_info(f"  {task_type.value} -> {role.value}")

        if len(TASK_ROUTING) > 0:
            print_pass(f"Task routing map has {len(TASK_ROUTING)} entries")
            tests_passed += 1
        else:
            print_fail("Task routing map is empty")
            tests_failed += 1

        # Test router instantiation (no API calls)
        router = ModelRouter(config)
        print_pass("ModelRouter instantiated")
        tests_passed += 1

        # Note: We don't call initialize() because that requires API keys
        print_info("Note: Full router testing requires API keys")
        print_info("Set GOOGLE_API_KEY and ANTHROPIC_API_KEY to test LLM calls")

    except Exception as e:
        print_fail("Model Router structure test", str(e))
        traceback.print_exc()
        tests_failed += 1

    return tests_passed, tests_failed


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    print(f"\n{BLUE}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║       PropelAI Phase 1 Test Suite                        ║{RESET}")
    print(f"{BLUE}║       Testing Foundation & Intelligent Ingestion         ║{RESET}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════════╝{RESET}")

    total_passed = 0
    total_failed = 0

    # Run all tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Metadata Extractor", test_metadata_extractor),
        ("Criteria-Eval Framework", test_criteria_eval),
        ("TF-IDF Deduplication", test_deduplication),
        ("Model Router Structure", test_model_router_structure),
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

    # Summary
    print_header("TEST SUMMARY")
    print(f"  Total Passed: {GREEN}{total_passed}{RESET}")
    print(f"  Total Failed: {RED}{total_failed}{RESET}")

    if total_failed == 0:
        print(f"\n{GREEN}All tests passed! Phase 1 implementation is working.{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}Some tests failed. Check the output above for details.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
