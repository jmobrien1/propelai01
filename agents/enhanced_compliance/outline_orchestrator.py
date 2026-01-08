"""
PropelAI v3.0: OutlineOrchestrator - Coordinates StrictStructureBuilder and ContentInjector.

This is the main entry point for the new decoupled outline generation pipeline.

Pipeline Flow:
1. SectionLParser: Raw Section L text -> SectionL_Schema
2. StrictStructureBuilder (Component A): SectionL_Schema -> ProposalSkeleton
3. ContentInjector (Component B): ProposalSkeleton + Requirements -> Annotated Outline

v6.0: Added Supervisor validation gate that validates parsing results and triggers
      re-search strategies on failures. This is the "Agentic Structural Verification"
      pattern replacing deterministic regex-only parsing.

v6.0.7: Added Multi-Vehicle Procurement Support with Polymorphic Factory Pattern.
        - Auto-detects procurement type (UCF, GSA RFQ, IDIQ, etc.)
        - Uses SOO-First fallback for GSA RFQs when Section L structure absent
        - Provides structured error messages with actionable guidance
        - "Fail Loud, Not Wrong" principle - never hallucinate structure

Key Principle: Structure comes from Section L ONLY. Content comes from Section C.
For GSA RFQs: Structure comes from SOO when Section L is absent.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from .section_l_parser import SectionLParser, StructureValidationError as ParserValidationError
from .section_l_schema import SectionL_Schema
from .strict_structure_builder import (
    StrictStructureBuilder,
    ProposalSkeleton,
    StructureValidationError
)
from .content_injector import ContentInjector, InjectionResult

# v6.0.7: Multi-Vehicle Procurement Support
from .models import (
    ProcurementType,
    DetectionConfidence,
    ProcurementDetectionResult,
    ProcurementTypeDetector
)
from .rfq_skeleton_builder import (
    RFQSkeletonBuilder,
    RFQStructureAnalysis,
    OutlineGenerationError,
    create_unsupported_format_error,
    create_missing_structure_error
)

# v6.0.12: Strategy Pattern - Isolated GSA parser
from .gsa_parser import (
    GSAParser,
    GSAParseResult,
    StrategyManager,
    ParsingStrategy,
    GSAPhase
)


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A validation issue detected by the Supervisor."""
    code: str
    message: str
    severity: ValidationSeverity
    recoverable: bool = True
    suggested_action: Optional[str] = None


@dataclass
class SupervisorValidationResult:
    """
    v6.0: Result of Supervisor validation gate.

    This captures the quality assessment of the parsing/outline generation
    and provides actionable feedback for improvement.
    """
    is_valid: bool
    confidence_score: float  # 0.0-1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    recovery_attempted: bool = False
    recovery_success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)


class SupervisorValidator:
    """
    v6.0: Supervisor validation gate for outline generation.

    This implements the "Agentic Structural Verification" pattern:
    1. Validate parsing results against quality thresholds
    2. Detect issues like volume count mismatch, missing sections, etc.
    3. Trigger recovery strategies (semantic search, relaxed parsing)
    4. Return validated result with confidence score

    The Supervisor acts as a quality gate between parsing and injection,
    ensuring the skeleton meets minimum quality standards before proceeding.
    """

    # Quality thresholds
    MIN_VOLUMES_FOR_VALID = 1
    MIN_SECTIONS_PER_VOLUME = 0  # 0 = no minimum, just warn if none
    MAX_UNMAPPED_REQUIREMENT_RATIO = 0.20  # 20% unmapped is concerning
    MIN_CONFIDENCE_FOR_VALID = 0.60

    # Iron Triangle rules: Which volumes should contain which content types
    VOLUME_CONTENT_RULES = {
        # Volume keywords → allowed content types
        'technical': ['technical', 'approach', 'management', 'staffing', 'methodology'],
        'cost': ['cost', 'price', 'pricing', 'budget', 'rates', 'labor'],
        'contract': ['certifications', 'representations', 'forms', 'administrative'],
        'past performance': ['experience', 'references', 'past performance'],
    }

    # v6.0.3: Official agency solicitation patterns for validation
    OFFICIAL_SOL_PATTERNS = [
        r'^FA\d{4}',  # Air Force
        r'^W\d{3}[A-Z]{2}',  # Army
        r'^N\d{5}',  # Navy
        r'^SP\d{4}',  # DLA
        r'^\d{2}[A-Z]\d{5}',  # NIH
    ]

    def validate_schema(
        self,
        schema: SectionL_Schema,
        stated_volume_count: Optional[int] = None,
        rfp_number: Optional[str] = None,
        has_tables_detected: bool = False
    ) -> SupervisorValidationResult:
        """
        v6.0.3: BINARY IRON TRIANGLE GATE - Validate schema with mandatory field checks.

        This is a BINARY gate: if ANY mandatory field is missing or invalid,
        the outline generation MUST fail with a hard error.

        Mandatory Fields (CRITICAL if missing):
        1. Volume count must match stated count
        2. Solicitation number must match official agency pattern (not internal ID)
        3. Technical volume MUST have page limit if tables were detected

        Args:
            schema: Parsed Section L schema
            stated_volume_count: Expected volume count from RFP (if known)
            rfp_number: Solicitation number to validate
            has_tables_detected: Whether PDF tables were found (for page limit validation)

        Returns:
            SupervisorValidationResult with issues and confidence
        """
        import re
        issues: List[ValidationIssue] = []
        confidence = 1.0

        volumes = schema.get('volumes', [])
        sections = schema.get('sections', [])

        # v6.0.3: BINARY CHECK 1 - Volume count
        if len(volumes) == 0:
            issues.append(ValidationIssue(
                code="NO_VOLUMES",
                message="BINARY GATE FAIL: No volumes found in Section L",
                severity=ValidationSeverity.CRITICAL,
                recoverable=False,  # v6.0.3: Hard block
                suggested_action="Cannot proceed without volume structure"
            ))
            confidence = 0.0

        # v6.0.3: BINARY CHECK 2 - Stated vs found volume count
        stated = stated_volume_count or schema.get('stated_volume_count')
        if stated is not None and len(volumes) != stated:
            missing_count = stated - len(volumes)
            if missing_count > 0:
                issues.append(ValidationIssue(
                    code="VOLUME_COUNT_MISMATCH",
                    message=f"BINARY GATE FAIL: RFP states {stated} volumes but only found {len(volumes)}. "
                            f"Missing {missing_count} volume(s).",
                    severity=ValidationSeverity.CRITICAL,
                    recoverable=False,
                    suggested_action=f"Search for {missing_count} missing volume(s) in source PDF"
                ))
                confidence = 0.0
            else:
                issues.append(ValidationIssue(
                    code="VOLUME_COUNT_EXCESS",
                    message=f"RFP states {stated} volumes but found {len(volumes)} ({len(volumes) - stated} extra)",
                    severity=ValidationSeverity.WARNING,
                    recoverable=True,
                    suggested_action="Review if extra volumes should be consolidated"
                ))
                confidence -= 0.1

        # v6.0.3: BINARY CHECK 3 - Solicitation number validation
        if rfp_number:
            is_official = any(
                re.match(pattern, rfp_number.upper())
                for pattern in self.OFFICIAL_SOL_PATTERNS
            )

            # Check for internal ID patterns (REJECT)
            is_internal = bool(re.match(r'^RFP[-_]?[A-F0-9]+', rfp_number.upper()))

            if is_internal:
                issues.append(ValidationIssue(
                    code="INTERNAL_ID_USED",
                    message=f"BINARY GATE FAIL: Using internal ID '{rfp_number}' instead of official solicitation number",
                    severity=ValidationSeverity.CRITICAL,
                    recoverable=False,
                    suggested_action="Extract official solicitation number from SF1449 Block 5"
                ))
                confidence = 0.0
            elif not is_official:
                issues.append(ValidationIssue(
                    code="UNRECOGNIZED_SOL_FORMAT",
                    message=f"Solicitation '{rfp_number}' does not match known agency patterns",
                    severity=ValidationSeverity.WARNING,
                    recoverable=True,
                    suggested_action="Verify solicitation number format is correct"
                ))
                confidence -= 0.1

        # v6.0.3: BINARY CHECK 4 - Technical volume page limit (if tables detected)
        if has_tables_detected and len(volumes) > 0:
            technical_volumes = [
                v for v in volumes
                if any(kw in v.get('volume_title', '').lower()
                       for kw in ['technical', 'approach', 'management'])
            ]

            for tech_vol in technical_volumes:
                if tech_vol.get('page_limit') is None:
                    issues.append(ValidationIssue(
                        code="MISSING_PAGE_LIMIT",
                        message=f"BINARY GATE FAIL: Technical volume '{tech_vol.get('volume_title')}' "
                                f"has no page limit but tables were detected in PDF",
                        severity=ValidationSeverity.CRITICAL,
                        recoverable=False,
                        suggested_action="Extract page limit from PDF table using row-index intersection"
                    ))
                    confidence = 0.0

        # Check 5: Sections distribution (WARNING only)
        orphan_sections = [s for s in sections if not s.get('parent_volume_id')]
        if orphan_sections:
            issues.append(ValidationIssue(
                code="ORPHAN_SECTIONS",
                message=f"{len(orphan_sections)} sections not assigned to any volume",
                severity=ValidationSeverity.WARNING,
                recoverable=True,
                suggested_action="Assign sections based on number prefix (1.x → Vol 1)"
            ))
            confidence -= 0.1

        # Check 6: Empty volumes (WARNING only)
        for vol in volumes:
            vol_sections = [s for s in sections if s.get('parent_volume_id') == vol.get('volume_id')]
            if not vol_sections and vol.get('page_limit') is None:
                issues.append(ValidationIssue(
                    code="EMPTY_VOLUME",
                    message=f"Volume '{vol.get('volume_title')}' has no sections or page limit",
                    severity=ValidationSeverity.WARNING,
                    recoverable=False,
                    suggested_action="Manual review - volume may need structure from attachments"
                ))
                confidence -= 0.05

        # Check 7: Parsing warnings (INFO only)
        for warning in schema.get('parsing_warnings', []):
            issues.append(ValidationIssue(
                code="PARSER_WARNING",
                message=warning,
                severity=ValidationSeverity.INFO,
                recoverable=False
            ))

        # v6.0.3: BINARY VALIDITY - Any CRITICAL issue = invalid
        is_valid = not any(i.severity == ValidationSeverity.CRITICAL for i in issues)

        return SupervisorValidationResult(
            is_valid=is_valid,
            confidence_score=max(0.0, confidence),
            issues=issues,
            metrics={
                'volume_count': len(volumes),
                'section_count': len(sections),
                'stated_volume_count': stated,
                'orphan_sections': len(orphan_sections),
                'has_critical_issues': not is_valid,
            }
        )

    def validate_injection(
        self,
        result: InjectionResult,
        skeleton_dict: Dict[str, Any]
    ) -> SupervisorValidationResult:
        """
        Validate the injection result (requirements → skeleton mapping).

        Args:
            result: InjectionResult from ContentInjector
            skeleton_dict: The skeleton that was injected into

        Returns:
            SupervisorValidationResult with injection quality metrics
        """
        issues: List[ValidationIssue] = []
        confidence = 1.0

        total_reqs = result.total_requirements
        mapped = result.mapped_count
        unmapped = len(result.unmapped_requirements)
        low_conf = len(result.low_confidence_mappings)

        # Check 1: Unmapped requirements ratio
        if total_reqs > 0:
            unmapped_ratio = unmapped / total_reqs
            if unmapped_ratio > self.MAX_UNMAPPED_REQUIREMENT_RATIO:
                issues.append(ValidationIssue(
                    code="HIGH_UNMAPPED_RATIO",
                    message=f"{unmapped}/{total_reqs} ({unmapped_ratio:.0%}) requirements unmapped",
                    severity=ValidationSeverity.ERROR,
                    recoverable=True,
                    suggested_action="Expand skeleton sections or relax matching criteria"
                ))
                confidence -= 0.3

        # Check 2: Low confidence mappings
        if total_reqs > 0 and low_conf / total_reqs > 0.30:
            issues.append(ValidationIssue(
                code="HIGH_LOW_CONFIDENCE",
                message=f"{low_conf}/{total_reqs} requirements mapped with low confidence",
                severity=ValidationSeverity.WARNING,
                recoverable=False,
                suggested_action="Review low-confidence mappings manually"
            ))
            confidence -= 0.15

        # Check 3: Volume balance (avoid all reqs in one volume)
        volumes = skeleton_dict.get('volumes', [])
        if len(volumes) >= 2 and total_reqs > 0:
            vol_req_counts = {}
            for vol in volumes:
                vol_reqs = sum(
                    len(sec.get('requirement_slots', []))
                    for sec in vol.get('sections', [])
                )
                vol_req_counts[vol.get('title', 'Unknown')] = vol_reqs

            # Check if one volume has > 80% of requirements
            max_vol_reqs = max(vol_req_counts.values()) if vol_req_counts else 0
            if max_vol_reqs > 0.8 * mapped:
                issues.append(ValidationIssue(
                    code="UNBALANCED_DISTRIBUTION",
                    message=f"Requirements concentrated in one volume ({max_vol_reqs}/{mapped})",
                    severity=ValidationSeverity.WARNING,
                    recoverable=False,
                    suggested_action="Review section keyword matching rules"
                ))

        # Check 4: Warnings from injector
        for warning in result.warnings:
            issues.append(ValidationIssue(
                code="INJECTOR_WARNING",
                message=warning,
                severity=ValidationSeverity.INFO,
                recoverable=False
            ))

        is_valid = confidence >= self.MIN_CONFIDENCE_FOR_VALID

        return SupervisorValidationResult(
            is_valid=is_valid,
            confidence_score=max(0.0, confidence),
            issues=issues,
            metrics={
                'total_requirements': total_reqs,
                'mapped_count': mapped,
                'unmapped_count': unmapped,
                'low_confidence_count': low_conf,
                'unmapped_ratio': unmapped / total_reqs if total_reqs > 0 else 0,
            }
        )


class OutlineOrchestrator:
    """
    Orchestrates the two-phase outline generation pipeline.

    Phase 1: Section L -> Skeleton (StrictStructureBuilder)
    Phase 2: Skeleton + Requirements -> Annotated Outline (ContentInjector)

    Usage:
        orchestrator = OutlineOrchestrator(strict_mode=True)
        result = orchestrator.generate_outline(
            section_l_text=state['instructions_text'],
            requirements=state['requirements'],
            evaluation_criteria=state['evaluation_criteria']
        )

        # Store results in state
        state['proposal_skeleton'] = result['proposal_skeleton']
        state['annotated_outline'] = result['annotated_outline']

    Or use directly from state:
        result = orchestrator.generate_from_state(state)
    """

    def __init__(self, strict_mode: bool = True, enable_supervisor: bool = True):
        """
        Initialize the orchestrator.

        Args:
            strict_mode: If True, fail on structure validation errors.
                        If False, continue with warnings (for debugging).
            enable_supervisor: v6.0 - If True, run Supervisor validation gate
                              to validate results and trigger recovery strategies.
        """
        self.strict_mode = strict_mode
        self.enable_supervisor = enable_supervisor

        # v6.0.7: Multi-vehicle procurement support
        self.procurement_detector = ProcurementTypeDetector()
        self.rfq_builder = RFQSkeletonBuilder()

        # v6.0.12: Strategy Pattern - ISOLATED parsers
        self.strategy_manager = StrategyManager()
        self.gsa_parser = GSAParser()  # Dedicated GSA/FAR 8.4 parser

        # UCF-style parser/builder (existing - for Air Force/OASIS)
        self.parser = SectionLParser()
        self.structure_builder = StrictStructureBuilder(strict_mode=strict_mode)
        self.content_injector = ContentInjector()
        self.supervisor = SupervisorValidator() if enable_supervisor else None

        # v6.0.7: Track detected procurement type for error messages
        self._detected_procurement: Optional[ProcurementDetectionResult] = None
        # v6.0.12: Track selected strategy
        self._selected_strategy: Optional[ParsingStrategy] = None

    def generate_outline(
        self,
        section_l_text: str,
        requirements: List[Dict],
        evaluation_criteria: List[Dict],
        rfp_number: str = "",
        rfp_title: str = "",
        attachment_texts: Optional[Dict[str, str]] = None,
        pdf_path: Optional[str] = None,
        soo_text: Optional[str] = None,
        lenient_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate annotated outline using two-phase pipeline.

        This is the main entry point for outline generation.

        v6.0.7: Added multi-vehicle procurement support with auto-detection
        and SOO-first fallback for GSA RFQs.

        Args:
            section_l_text: Full text of Section L instructions
            requirements: Section C requirements from CTM
            evaluation_criteria: Section M criteria from CTM
            rfp_number: Solicitation number
            rfp_title: RFP title
            attachment_texts: Optional dict of structural attachment texts
            pdf_path: v6.0.2 - Path to source PDF for spatial extraction (SF1449, tables)
            soo_text: v6.0.7 - Statement of Objectives text (for GSA RFQ fallback)
            lenient_mode: v6.0.7 - If True, allow fallback strategies with warnings

        Returns:
            Dict containing:
            - proposal_skeleton: Skeleton from Component A (for state storage)
            - annotated_outline: Populated outline from Component B
            - injection_metadata: Statistics about the injection
            - unmapped_requirements: Requirements that couldn't be mapped
            - procurement_detection: v6.0.7 - Detection result with confidence

        Raises:
            StructureValidationError: If strict_mode and structure is invalid
                                      OR if Supervisor detects CRITICAL validation issues
        """
        logger.info(f"Starting outline generation for RFP: {rfp_number}")
        logger.info(f"Section L text length: {len(section_l_text)} chars")
        logger.info(f"Requirements count: {len(requirements)}")
        if pdf_path:
            logger.info(f"[v6.0.2] PDF path provided for spatial extraction: {pdf_path}")

        # ========================================================================
        # v6.0.12: STRATEGY SELECTION (Architectural Separation)
        # ========================================================================
        # Select parsing strategy BEFORE any parsing begins.
        # GSA and UCF parsers are COMPLETELY ISOLATED - no shared logic.
        # ========================================================================

        self._detected_procurement = self.procurement_detector.detect(
            section_l_text,
            solicitation_number=rfp_number
        )

        # v6.0.12: Use StrategyManager for clean separation
        self._selected_strategy = self.strategy_manager.select_strategy(
            section_l_text,
            self._detected_procurement
        )

        print(f"[v6.0.12] Strategy Selection: {self._selected_strategy.value}")
        print(f"[v6.0.12] Procurement Detection: type={self._detected_procurement.procurement_type.value}, "
              f"confidence={self._detected_procurement.confidence.value} "
              f"({self._detected_procurement.confidence_score:.2%})")

        if self._detected_procurement.detected_signals:
            print(f"[v6.0.12] Detection signals: {self._detected_procurement.detected_signals[:3]}")

        # ========================================================================
        # v6.0.12: GSA STRATEGY - COMPLETELY ISOLATED PATH
        # ========================================================================
        if self._selected_strategy == ParsingStrategy.GSA_RFQ:
            print(f"[v6.0.12] Using GSA Parser (FAR 8.4) - ISOLATED from UCF")
            try:
                return self._generate_gsa_outline(
                    section_l_text=section_l_text,
                    soo_text=soo_text,
                    requirements=requirements,
                    evaluation_criteria=evaluation_criteria,
                    rfp_number=rfp_number,
                    rfp_title=rfp_title,
                    attachment_texts=attachment_texts
                )
            except StructureValidationError as e:
                print(f"[v6.0.12] GSA strategy failed: {str(e)[:100]}")
                # Do NOT fall through to UCF - fail loud instead
                if self.strict_mode:
                    raise
                # In lenient mode, try legacy RFQ builder
                print("[v6.0.12] Lenient mode: Falling back to legacy RFQ builder")

        # v6.0.7 LEGACY: If detected as GSA RFQ but not selected by new strategy, try RFQ-first approach
        if (self._detected_procurement.procurement_type == ProcurementType.GSA_RFQ_8_4 and
            self._detected_procurement.confidence in [DetectionConfidence.HIGH, DetectionConfidence.MEDIUM] and
            self._selected_strategy != ParsingStrategy.GSA_RFQ):

            print(f"[v6.0.7] Legacy GSA RFQ detection - attempting RFQ-first strategy")
            try:
                return self._generate_rfq_outline(
                    section_l_text=section_l_text,
                    soo_text=soo_text,
                    requirements=requirements,
                    evaluation_criteria=evaluation_criteria,
                    rfp_number=rfp_number,
                    rfp_title=rfp_title,
                    attachment_texts=attachment_texts
                )
            except StructureValidationError as e:
                print(f"[v6.0.7] RFQ strategy failed, will try UCF fallback: {str(e)[:100]}")
                # Fall through to UCF parsing

        # Debug: Print for Render logs
        print(f"[v3.0 Parser] Section L text preview (first 500 chars): {section_l_text[:500]}")

        # ========================================================================
        # v6.0.7: UCF PARSING WITH RFQ FALLBACK
        # ========================================================================
        # Try UCF parsing first. If it fails and we detect a non-UCF format,
        # attempt RFQ fallback strategy before giving up.
        # ========================================================================

        # Phase 1: Parse Section L into schema
        # v6.0.2: Pass pdf_path for spatial extraction (SF1449, digit-targeting tables)
        logger.info("Phase 1: Parsing Section L into schema...")

        try:
            schema = self.parser.parse(
                section_l_text=section_l_text,
                rfp_number=rfp_number,
                rfp_title=rfp_title,
                attachment_texts=attachment_texts,
                strict_mode=self.strict_mode,
                pdf_path=pdf_path
            )
        except (StructureValidationError, ParserValidationError) as ucf_error:
            # v6.0.7: UCF parsing failed - try RFQ fallback if appropriate
            print(f"[v6.0.7] UCF parsing failed: {str(ucf_error)[:100]}")

            # Check if this looks like a non-UCF format that RFQ builder might handle
            if self._detected_procurement and self._detected_procurement.procurement_type in [
                ProcurementType.GSA_RFQ_8_4,
                ProcurementType.GSA_BPA_CALL,
                ProcurementType.SIMPLIFIED_SAP
            ]:
                print(f"[v6.0.7] Detected {self._detected_procurement.procurement_type.value} - "
                      f"attempting RFQ fallback strategy")
                try:
                    return self._generate_rfq_outline(
                        section_l_text=section_l_text,
                        soo_text=soo_text,
                        requirements=requirements,
                        evaluation_criteria=evaluation_criteria,
                        rfp_number=rfp_number,
                        rfp_title=rfp_title,
                        attachment_texts=attachment_texts
                    )
                except StructureValidationError as rfq_error:
                    # Both UCF and RFQ strategies failed - provide comprehensive error
                    print(f"[v6.0.7] RFQ fallback also failed: {str(rfq_error)[:100]}")
                    raise StructureValidationError(
                        self._build_comprehensive_error_message(
                            ucf_error=str(ucf_error),
                            rfq_error=str(rfq_error),
                            detection=self._detected_procurement
                        )
                    )

            # Not a format we can handle with RFQ fallback - re-raise with enhanced message
            raise StructureValidationError(
                self._build_ucf_failure_message(str(ucf_error), self._detected_procurement)
            )

        # Debug: Print parser results for Render logs
        print(f"[v3.0 Parser] Found {len(schema.get('volumes', []))} volumes")
        print(f"[v3.0 Parser] Volume titles: {[v.get('volume_title', 'N/A') for v in schema.get('volumes', [])]}")
        print(f"[v3.0 Parser] Parsing warnings: {schema.get('parsing_warnings', [])}")

        logger.info(f"  Found {len(schema.get('volumes', []))} volumes")
        logger.info(f"  Found {len(schema.get('sections', []))} sections")
        if schema.get('parsing_warnings'):
            for warn in schema['parsing_warnings']:
                logger.warning(f"  Parser warning: {warn}")

        # v6.0.3: Detect if PDF tables were found (for page limit validation)
        has_tables_detected = bool(schema.get('_tables_detected', False))

        # ========================================================================
        # v6.0.4: BINARY INTEGRITY GATE - Hard validation before proceeding
        # ========================================================================
        # The Problem: The system returns partial, non-compliant outlines with
        # hallucinated internal IDs like "RFP-866BCC3A" instead of official
        # solicitation numbers, or missing critical page limits.
        #
        # The Fix: If rfp_number starts with "RFP-" (internal hallucination) OR
        # if Volume 1 (Technical) page limit is null, raise StructureValidationError.
        # This is a HARD BLOCK - no partial outlines allowed.
        # ========================================================================

        # v6.0.4: Get the extracted rfp_number from schema (it may have been updated by parser)
        extracted_rfp_number = schema.get('rfp_number') or rfp_number

        # v6.0.4 CHECK 1: Reject internal ID hallucinations
        if extracted_rfp_number:
            import re
            internal_id_pattern = r'^RFP[-_]?[A-F0-9]{6,}'
            if re.match(internal_id_pattern, extracted_rfp_number.upper()):
                error_msg = (
                    f"[v6.0.4 BINARY INTEGRITY GATE FAIL] "
                    f"Detected internal ID hallucination: '{extracted_rfp_number}'. "
                    f"System generated internal ID instead of extracting official solicitation number. "
                    f"Cannot generate compliant outline without valid solicitation number from SF1449 Block 5."
                )
                print(f"[v6.0.4 BINARY GATE] ✗ HARD BLOCK: {error_msg}")
                raise StructureValidationError(error_msg)

        # v6.0.4 CHECK 2: Reject missing Technical volume page limit
        volumes = schema.get('volumes', [])
        if volumes:
            # Find Volume 1 / Technical volume
            technical_volume = None
            for vol in volumes:
                vol_title = vol.get('volume_title', '').lower()
                vol_number = vol.get('volume_number', 0)
                # Volume 1 is typically Technical, or look for "technical" keyword
                if vol_number == 1 or 'technical' in vol_title or 'approach' in vol_title:
                    technical_volume = vol
                    break

            if technical_volume:
                page_limit = technical_volume.get('page_limit')
                if page_limit is None and has_tables_detected:
                    vol_title = technical_volume.get('volume_title', 'Volume 1')
                    error_msg = (
                        f"[v6.0.4 BINARY INTEGRITY GATE FAIL] "
                        f"'{vol_title}' has no page limit specified but PDF tables were detected. "
                        f"The page limit MUST be extracted from the RFP table using row-column intersection. "
                        f"Cannot generate compliant outline without deterministic page constraints."
                    )
                    print(f"[v6.0.4 BINARY GATE] ✗ HARD BLOCK: {error_msg}")
                    raise StructureValidationError(error_msg)

        print(f"[v6.0.4 BINARY GATE] ✓ Passed: rfp_number='{extracted_rfp_number}', "
              f"volumes_with_limits={sum(1 for v in volumes if v.get('page_limit') is not None)}/{len(volumes)}")

        # v6.0.3: BINARY IRON TRIANGLE GATE - Supervisor validation
        schema_validation = None
        if self.supervisor:
            logger.info("Phase 1.5: v6.0.3 Binary Iron Triangle Gate validation...")
            schema_validation = self.supervisor.validate_schema(
                schema,
                rfp_number=rfp_number,
                has_tables_detected=has_tables_detected
            )
            print(f"[v6.0.3 BINARY GATE] Schema validation: valid={schema_validation.is_valid}, "
                  f"confidence={schema_validation.confidence_score:.2f}")

            for issue in schema_validation.issues:
                if issue.severity == ValidationSeverity.CRITICAL:
                    print(f"[v6.0.3 BINARY GATE] ✗ CRITICAL: {issue.message}")
                    if issue.suggested_action:
                        print(f"[v6.0.3 BINARY GATE]   Action: {issue.suggested_action}")
                elif issue.severity == ValidationSeverity.ERROR:
                    print(f"[v6.0.3 BINARY GATE] ERROR: {issue.message}")

            # v6.0.3: BINARY GATE - Any CRITICAL issue = hard block
            if schema_validation.has_critical_issues and self.strict_mode:
                critical_issues = [
                    i for i in schema_validation.issues
                    if i.severity == ValidationSeverity.CRITICAL
                ]
                error_messages = "; ".join(i.message for i in critical_issues)
                print(f"[v6.0.3 BINARY GATE] ✗ HARD BLOCK: {len(critical_issues)} critical issue(s)")
                raise StructureValidationError(
                    f"[v6.0.3 BINARY GATE FAIL] Cannot generate outline: {error_messages}"
                )

        # Phase 2: Build skeleton from schema
        logger.info("Phase 2: Building skeleton from schema...")
        try:
            skeleton = self.structure_builder.build_from_schema(schema)
        except StructureValidationError as e:
            logger.error(f"Skeleton validation failed: {e}")
            raise

        skeleton_dict = self.structure_builder.to_state_dict(skeleton)
        logger.info(f"  Skeleton valid: {skeleton.is_valid}")
        logger.info(f"  Volumes: {[v.title for v in skeleton.volumes]}")

        # Phase 3: Inject requirements into skeleton
        logger.info("Phase 3: Injecting requirements into skeleton...")
        result = self.content_injector.inject(
            skeleton_dict=skeleton_dict,
            requirements=requirements,
            evaluation_criteria=evaluation_criteria
        )

        logger.info(f"  Mapped: {result.mapped_count}/{result.total_requirements}")
        logger.info(f"  Unmapped: {len(result.unmapped_requirements)}")
        logger.info(f"  Low confidence: {len(result.low_confidence_mappings)}")

        # v6.0: Supervisor validation gate for injection
        injection_validation = None
        if self.supervisor:
            logger.info("Phase 3.5: Supervisor validation of injection...")
            injection_validation = self.supervisor.validate_injection(result, skeleton_dict)
            print(f"[v6.0 Supervisor] Injection validation: valid={injection_validation.is_valid}, "
                  f"confidence={injection_validation.confidence_score:.2f}")

            for issue in injection_validation.issues:
                if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
                    print(f"[v6.0 Supervisor] {issue.severity.value.upper()}: {issue.message}")

        # Calculate overall confidence score
        overall_confidence = 1.0
        if schema_validation:
            overall_confidence = min(overall_confidence, schema_validation.confidence_score)
        if injection_validation:
            overall_confidence = min(overall_confidence, injection_validation.confidence_score)

        return {
            'proposal_skeleton': skeleton_dict,
            'annotated_outline': result.annotated_outline,
            'injection_metadata': {
                'total_requirements': result.total_requirements,
                'mapped_count': result.mapped_count,
                'unmapped_count': len(result.unmapped_requirements),
                'low_confidence_count': len(result.low_confidence_mappings),
                'warnings': result.warnings,
                'schema_warnings': schema.get('parsing_warnings', []),
                'skeleton_warnings': skeleton.validation_warnings
            },
            # v6.0.5: Extracted metadata from SF1449 and document parsing
            # This should be used to UPDATE the RFP's solicitation_number if valid
            'extracted_metadata': {
                'solicitation_number': schema.get('rfp_number'),  # From SF1449 Block 5
                'rfp_title': schema.get('rfp_title'),
                'due_date': schema.get('submission_rules', {}).get('deadline') if schema.get('submission_rules') else None,
                'total_page_limit': schema.get('total_page_limit'),
            },
            'unmapped_requirements': result.unmapped_requirements,
            'low_confidence_mappings': [
                {
                    'requirement_id': m.requirement_id,
                    'requirement_text': m.requirement_text,
                    'target_section': m.target_section_id,
                    'confidence': m.confidence.value,
                    'rationale': m.rationale
                }
                for m in result.low_confidence_mappings
            ],
            # v6.0: Supervisor validation results
            'supervisor_validation': {
                'overall_confidence': overall_confidence,
                'schema_validation': {
                    'is_valid': schema_validation.is_valid if schema_validation else True,
                    'confidence': schema_validation.confidence_score if schema_validation else 1.0,
                    'issues': [
                        {
                            'code': i.code,
                            'message': i.message,
                            'severity': i.severity.value,
                            'recoverable': i.recoverable
                        }
                        for i in (schema_validation.issues if schema_validation else [])
                    ],
                    'metrics': schema_validation.metrics if schema_validation else {}
                } if schema_validation else None,
                'injection_validation': {
                    'is_valid': injection_validation.is_valid if injection_validation else True,
                    'confidence': injection_validation.confidence_score if injection_validation else 1.0,
                    'issues': [
                        {
                            'code': i.code,
                            'message': i.message,
                            'severity': i.severity.value,
                            'recoverable': i.recoverable
                        }
                        for i in (injection_validation.issues if injection_validation else [])
                    ],
                    'metrics': injection_validation.metrics if injection_validation else {}
                } if injection_validation else None
            },
            # v6.0.7: Procurement detection results
            'procurement_detection': {
                'type': self._detected_procurement.procurement_type.value if self._detected_procurement else 'unknown',
                'confidence': self._detected_procurement.confidence.value if self._detected_procurement else 'very_low',
                'confidence_score': self._detected_procurement.confidence_score if self._detected_procurement else 0.0,
                'detected_signals': self._detected_procurement.detected_signals[:5] if self._detected_procurement else [],
            } if self._detected_procurement else None
        }

    def _generate_rfq_outline(
        self,
        section_l_text: str,
        soo_text: Optional[str],
        requirements: List[Dict],
        evaluation_criteria: List[Dict],
        rfp_number: str,
        rfp_title: str,
        attachment_texts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        v6.0.7: Generate outline using RFQ/SOO-first strategy.

        This is used for GSA RFQs (FAR 8.4) and similar procurements
        that don't follow UCF structure.

        Args:
            section_l_text: Main RFQ document text
            soo_text: Statement of Objectives text (may be in attachments)
            requirements: Extracted requirements
            evaluation_criteria: Extracted evaluation factors
            rfp_number: Solicitation number
            rfp_title: RFP title
            attachment_texts: Dict of attachment name -> text

        Returns:
            Dict with skeleton, outline, and metadata

        Raises:
            StructureValidationError: If structure cannot be determined
        """
        print(f"[v6.0.7] RFQ Outline Generation: Starting SOO-first strategy")

        # Try to find SOO text in attachments if not provided directly
        if not soo_text and attachment_texts:
            for name, text in attachment_texts.items():
                name_lower = name.lower()
                if 'soo' in name_lower or 'statement of objectives' in name_lower:
                    soo_text = text
                    print(f"[v6.0.7] Found SOO in attachment: {name}")
                    break
                elif 'sow' in name_lower or 'statement of work' in name_lower:
                    soo_text = text
                    print(f"[v6.0.7] Found SOW in attachment: {name}")
                    break

        # Analyze RFQ structure
        analysis = self.rfq_builder.analyze_structure(
            rfq_text=section_l_text,
            soo_text=soo_text,
            attachments=attachment_texts
        )

        print(f"[v6.0.7] RFQ Analysis: can_build={analysis.can_build_skeleton}, "
              f"confidence={analysis.confidence_score:.2%}, "
              f"sections={len(analysis.soo_sections)}")

        if not analysis.can_build_skeleton:
            # Cannot build - raise with detailed message
            error_msg = self.rfq_builder.get_failure_message(analysis)
            print(f"[v6.0.7] RFQ Strategy FAILED - insufficient structure")
            raise StructureValidationError(error_msg)

        # Build skeleton from SOO structure
        skeleton = self.rfq_builder.build_skeleton(
            analysis=analysis,
            rfp_number=rfp_number,
            rfp_title=rfp_title
        )

        print(f"[v6.0.7] Built RFQ skeleton: {len(skeleton.volumes)} volumes")

        # Convert to dict for injection
        skeleton_dict = self.structure_builder.to_state_dict(skeleton)

        # Inject requirements using standard content injector
        # v6.0.8: Pass is_soo_source=True to add PWS writing guidance
        result = self.content_injector.inject(
            skeleton_dict=skeleton_dict,
            requirements=requirements,
            evaluation_criteria=evaluation_criteria,
            is_soo_source=analysis.has_soo  # Add PWS guidance if source is SOO
        )

        print(f"[v6.0.8] Content injection: {result.mapped_count}/{result.total_requirements} mapped (SOO source: {analysis.has_soo})")

        return {
            'proposal_skeleton': skeleton_dict,
            'annotated_outline': result.annotated_outline,
            'injection_metadata': {
                'total_requirements': result.total_requirements,
                'mapped_count': result.mapped_count,
                'unmapped_count': len(result.unmapped_requirements),
                'low_confidence_count': len(result.low_confidence_mappings),
                'warnings': result.warnings + skeleton.validation_warnings,
                'schema_warnings': [],
                'skeleton_warnings': skeleton.validation_warnings,
                'rfq_analysis': {
                    'has_soo': analysis.has_soo,
                    'has_sow': analysis.has_sow,
                    'sections_found': len(analysis.soo_sections),
                    'phases': analysis.detected_phases,
                }
            },
            'extracted_metadata': {
                'solicitation_number': rfp_number,
                'rfp_title': rfp_title,
                'due_date': None,
                'total_page_limit': skeleton.total_page_limit,
            },
            'unmapped_requirements': result.unmapped_requirements,
            'low_confidence_mappings': [
                {
                    'requirement_id': m.requirement_id,
                    'requirement_text': m.requirement_text,
                    'target_section': m.target_section_id,
                    'confidence': m.confidence.value,
                    'rationale': m.rationale
                }
                for m in result.low_confidence_mappings
            ],
            'supervisor_validation': None,  # Not run for RFQ strategy
            'procurement_detection': {
                'type': self._detected_procurement.procurement_type.value if self._detected_procurement else 'gsa_rfq_8_4',
                'confidence': self._detected_procurement.confidence.value if self._detected_procurement else 'medium',
                'confidence_score': self._detected_procurement.confidence_score if self._detected_procurement else 0.7,
                'detected_signals': self._detected_procurement.detected_signals[:5] if self._detected_procurement else [],
            },
            'generation_strategy': 'rfq_soo_first',
            'requires_manual_review': True,  # Always flag RFQ outlines for review
        }

    def _generate_gsa_outline(
        self,
        section_l_text: str,
        soo_text: Optional[str],
        requirements: List[Dict],
        evaluation_criteria: List[Dict],
        rfp_number: str,
        rfp_title: str,
        attachment_texts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        v6.0.12: Generate outline using ISOLATED GSA Parser.

        This is COMPLETELY SEPARATE from UCF parsing logic.
        No code paths are shared between GSA and UCF.

        Key Features:
        - Phase-First: Phase I/II are structural roots
        - Factor Isolation: Factor 1 → Phase I, Factors 2-5 → Phase II
        - Strict enforcement: Phase II content CANNOT appear in Phase I
        """
        print(f"[v6.0.12 GSA] Starting GSA-specific outline generation")

        # Find SOO in attachments if not provided
        if not soo_text and attachment_texts:
            for name, text in attachment_texts.items():
                name_lower = name.lower()
                if 'soo' in name_lower or 'statement of objectives' in name_lower:
                    soo_text = text
                    print(f"[v6.0.12 GSA] Found SOO in attachment: {name}")
                    break

        # Parse using GSA-specific parser
        gsa_result = self.gsa_parser.parse(
            rfq_text=section_l_text,
            soo_text=soo_text,
            solicitation_number=rfp_number
        )

        # v6.0.12: VALIDATE PHASE STRUCTURE
        if not gsa_result.phases:
            raise StructureValidationError(
                "[v6.0.12 GSA] PHASE STRUCTURE FAILURE: No Phase I/II markers found. "
                "GSA RFQs require explicit Phase structure. "
                "Ensure document contains 'Phase I' and 'Phase II' markers."
            )

        # Check for Factor isolation violations
        for phase in gsa_result.phases:
            if phase['phase_type'] == GSAPhase.PHASE_I.value:
                # Phase I should only have Factor 1 (Experience)
                violating_factors = [
                    f for f in phase.get('factors', [])
                    if f['factor_number'] > 1 and f['target_phase'] != GSAPhase.PHASE_I.value
                ]
                if violating_factors:
                    print(f"[v6.0.12 GSA] WARNING: Phase I contains non-experience factors: {violating_factors}")

        # Convert to standard volume format
        volumes = self.gsa_parser.to_volumes(gsa_result)

        print(f"[v6.0.12 GSA] Created {len(volumes)} volumes from {len(gsa_result.phases)} phases")

        # Build skeleton from volumes
        from .section_l_schema import SectionL_Schema
        schema: SectionL_Schema = {
            'rfp_number': gsa_result.solicitation_number or rfp_number,
            'rfp_title': rfp_title,
            'volumes': [
                {
                    'volume_id': v.volume_id,
                    'volume_title': v.volume_title,
                    'volume_number': v.volume_number,
                    'page_limit': v.page_limit,
                    'sections': v.sections,
                    'source_reference': v.source_reference,
                    'is_mandatory': v.is_mandatory,
                }
                for v in volumes
            ],
            'sections': [],
            'format_rules': {},
            'submission_rules': {},
            'total_page_limit': None,
            'stated_volume_count': len(volumes),
            'parsing_warnings': gsa_result.warnings,
        }

        # Build skeleton
        skeleton = self.structure_builder.build_from_schema(schema)
        skeleton_dict = self.structure_builder.to_state_dict(skeleton)

        # Inject requirements (using standard injector with SOO awareness)
        result = self.content_injector.inject(
            skeleton_dict=skeleton_dict,
            requirements=requirements,
            evaluation_criteria=evaluation_criteria,
            is_soo_source=bool(soo_text)
        )

        print(f"[v6.0.12 GSA] Injection complete: {result.mapped_count}/{result.total_requirements} mapped")

        return {
            'proposal_skeleton': skeleton_dict,
            'annotated_outline': result.annotated_outline,
            'injection_metadata': {
                'total_requirements': result.total_requirements,
                'mapped_count': result.mapped_count,
                'unmapped_count': len(result.unmapped_requirements),
                'low_confidence_count': len(result.low_confidence_mappings),
                'warnings': result.warnings + gsa_result.warnings,
                'schema_warnings': gsa_result.warnings,
                'skeleton_warnings': skeleton.validation_warnings,
                'gsa_analysis': {
                    'phases': len(gsa_result.phases),
                    'factors': len(gsa_result.factors),
                    'section_11_found': gsa_result.section_11_found,
                    'section_12_found': gsa_result.section_12_found,
                    'soo_sections': len(gsa_result.sections),
                }
            },
            'extracted_metadata': {
                'solicitation_number': gsa_result.solicitation_number or rfp_number,
                'rfp_title': rfp_title,
                'due_date': None,
                'total_page_limit': skeleton.total_page_limit,
            },
            'unmapped_requirements': result.unmapped_requirements,
            'low_confidence_mappings': [
                {
                    'requirement_id': m.requirement_id,
                    'requirement_text': m.requirement_text,
                    'target_section': m.target_section_id,
                    'confidence': m.confidence.value,
                    'rationale': m.rationale
                }
                for m in result.low_confidence_mappings
            ],
            'supervisor_validation': None,  # Not run for GSA strategy
            'procurement_detection': {
                'type': 'gsa_rfq_8_4',
                'confidence': self._detected_procurement.confidence.value if self._detected_procurement else 'high',
                'confidence_score': self._detected_procurement.confidence_score if self._detected_procurement else 0.9,
                'detected_signals': self._detected_procurement.detected_signals[:5] if self._detected_procurement else [],
            },
            'generation_strategy': 'gsa_phase_first',  # v6.0.12: New strategy name
            'requires_manual_review': True,
            'factor_isolation_enforced': True,  # v6.0.12: Strict isolation flag
        }

    def _build_comprehensive_error_message(
        self,
        ucf_error: str,
        rfq_error: str,
        detection: Optional[ProcurementDetectionResult]
    ) -> str:
        """
        v6.0.7: Build comprehensive error when both UCF and RFQ strategies fail.

        This is the "Fail Loud, Not Wrong" principle - provide actionable
        guidance rather than a cryptic error.
        """
        lines = [
            "═" * 60,
            "OUTLINE GENERATION FAILED",
            "═" * 60,
            "",
            "PropelAI tried multiple strategies but could not determine",
            "the proposal structure from your uploaded documents.",
            "",
        ]

        # Detection info
        if detection:
            lines.extend([
                f"Detected Format: {detection.procurement_type.value}",
                f"Detection Confidence: {detection.confidence_score:.0%}",
                "",
            ])

        # What was tried
        lines.extend([
            "STRATEGIES ATTEMPTED:",
            "─" * 40,
            "",
            "1. UCF (Section L/M/C) Strategy:",
            f"   ✗ {ucf_error[:200]}..." if len(ucf_error) > 200 else f"   ✗ {ucf_error}",
            "",
            "2. RFQ (SOO-First) Strategy:",
            f"   ✗ {rfq_error[:200]}..." if len(rfq_error) > 200 else f"   ✗ {rfq_error}",
            "",
        ])

        # Suggestions
        lines.extend([
            "WHAT YOU CAN TRY:",
            "─" * 40,
            "",
            "1. Upload Additional Documents:",
            "   • For traditional RFPs: Upload Section L (Instructions to Offerors)",
            "   • For GSA RFQs: Upload Statement of Objectives (SOO) separately",
            "",
            "2. Check Document Classification:",
            "   • Ensure the main RFQ is uploaded to 'Main Solicitation'",
            "   • Ensure SOO/SOW is uploaded to 'Technical Requirements'",
            "",
            "3. Manual Outline Creation:",
            "   • Use the Compliance Matrix to draft your response",
            "   • Contact your proposal manager for structure guidance",
            "",
            "4. Contact Support:",
            "   • If you believe this format should be supported",
            "   • Include solicitation number in your request",
            "",
            "═" * 60,
        ])

        return "\n".join(lines)

    def _build_ucf_failure_message(
        self,
        original_error: str,
        detection: Optional[ProcurementDetectionResult]
    ) -> str:
        """
        v6.0.7: Build enhanced UCF failure message with guidance.
        """
        lines = [
            "═" * 60,
            "PROPOSAL STRUCTURE NOT FOUND",
            "═" * 60,
            "",
        ]

        # Detection info
        if detection:
            lines.extend([
                f"Detected Format: {detection.procurement_type.value}",
                f"Detection Confidence: {detection.confidence_score:.0%}",
            ])

            if detection.detected_signals:
                lines.append(f"Signals: {', '.join(detection.detected_signals[:3])}")
            lines.append("")

        # Original error
        lines.extend([
            "ISSUE:",
            "─" * 40,
            original_error,
            "",
        ])

        # Format-specific guidance
        if detection:
            if detection.procurement_type == ProcurementType.UCF_STANDARD:
                lines.extend([
                    "This appears to be a standard UCF (FAR 15) procurement.",
                    "The system expected to find 'Volume I:', 'Volume II:' declarations",
                    "in Section L but could not locate them.",
                    "",
                    "SUGGESTIONS:",
                    "• Verify Section L document is fully uploaded",
                    "• Check that volume declarations are present in the RFP",
                    "• Try re-uploading the Section L attachment specifically",
                ])
            elif detection.procurement_type in [ProcurementType.GSA_RFQ_8_4, ProcurementType.GSA_BPA_CALL]:
                lines.extend([
                    "This appears to be a GSA/FAR 8.4 procurement.",
                    "These don't use traditional Section L structure.",
                    "",
                    "SUGGESTIONS:",
                    "• Upload the Statement of Objectives (SOO) document",
                    "• Ensure quote instructions are in the main RFQ",
                    "• GSA RFQ support is being enhanced - contact support",
                ])
            elif detection.procurement_type == ProcurementType.IDIQ_TASK_ORDER:
                lines.extend([
                    "This appears to be an IDIQ Task Order (FAR 16.505).",
                    "Task orders have varying structures based on the IDIQ vehicle.",
                    "",
                    "SUGGESTIONS:",
                    "• Check the Task Order Request for structure requirements",
                    "• Reference the base IDIQ contract for proposal format",
                    "• IDIQ support is being enhanced - contact support",
                ])
            else:
                lines.extend([
                    "SUGGESTIONS:",
                    "• Upload proposal instruction documents",
                    "• Verify document classification in upload wizard",
                    "• Contact support for assistance with this format",
                ])
        else:
            lines.extend([
                "SUGGESTIONS:",
                "• Upload Section L (Instructions to Offerors) document",
                "• Verify all solicitation documents are uploaded",
                "• Check that documents contain proposal structure instructions",
            ])

        lines.extend(["", "═" * 60])

        return "\n".join(lines)

    def generate_from_state(self, state: Dict[str, Any], pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate outline directly from ProposalState.

        This is a convenience method that extracts the necessary
        data from a ProposalState TypedDict.

        Args:
            state: ProposalState TypedDict
            pdf_path: v6.0.2 - Path to source PDF for spatial extraction

        Returns:
            Dict with updates to apply to state
        """
        # Extract Section L text from instructions
        section_l_text = self._extract_section_l_text(
            state.get('instructions', [])
        )

        if not section_l_text:
            # Fallback: try to get from raw RFP text
            raw_text = state.get('rfp_raw_text', '')
            section_l_text = self._extract_section_l_from_raw(raw_text)

        # v6.0.2: Try to get pdf_path from state if not provided
        if not pdf_path:
            pdf_path = state.get('pdf_path') or state.get('source_pdf_path')

        return self.generate_outline(
            section_l_text=section_l_text,
            requirements=state.get('requirements', []),
            evaluation_criteria=state.get('evaluation_criteria', []),
            rfp_number=state.get('solicitation_number', ''),
            rfp_title=state.get('opportunity_name', ''),
            attachment_texts=None,  # Could be extended to pull from state
            pdf_path=pdf_path
        )

    def _extract_section_l_text(self, instructions: List[Dict]) -> str:
        """
        Extract Section L text from instructions list.

        Instructions may be stored in various formats depending on
        the extraction method used.
        """
        texts = []
        for instr in instructions:
            # Try various field names
            text = (
                instr.get('text') or
                instr.get('full_text') or
                instr.get('content') or
                instr.get('requirement_text') or
                ''
            )
            if text:
                texts.append(text)

        return '\n\n'.join(texts)

    def _extract_section_l_from_raw(self, raw_text: str) -> str:
        """
        Extract Section L content from raw RFP text.

        Used as fallback when structured instructions aren't available.
        """
        import re

        # Try to find Section L boundaries
        section_l_patterns = [
            r"SECTION\s+L[\s:\-–]+.*?(?=SECTION\s+[M-Z]|\Z)",
            r"INSTRUCTIONS.*?TO.*?OFFERORS.*?(?=EVALUATION|SECTION\s+M|\Z)",
        ]

        for pattern in section_l_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)

        # If no Section L found, return empty and let validation handle it
        return ""

    def validate_skeleton(self, skeleton_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a skeleton dict and return validation results.

        Useful for checking if a skeleton is valid before injection.

        Args:
            skeleton_dict: Skeleton from state['proposal_skeleton']

        Returns:
            Dict with validation status and details
        """
        return {
            'is_valid': skeleton_dict.get('is_valid', False),
            'validation_errors': skeleton_dict.get('validation_errors', []),
            'validation_warnings': skeleton_dict.get('validation_warnings', []),
            'volume_count': len(skeleton_dict.get('volumes', [])),
            'section_count': sum(
                len(v.get('sections', []))
                for v in skeleton_dict.get('volumes', [])
            ),
            'has_page_limits': any(
                v.get('page_limit') is not None
                for v in skeleton_dict.get('volumes', [])
            )
        }


def generate_outline_v3(
    section_l_text: str,
    requirements: List[Dict],
    evaluation_criteria: Optional[List[Dict]] = None,
    rfp_number: str = "",
    rfp_title: str = "",
    strict_mode: bool = True,
    pdf_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for outline generation.

    This is a simple wrapper around OutlineOrchestrator for
    quick usage without instantiation.

    Args:
        section_l_text: Section L instructions text
        requirements: Section C requirements
        evaluation_criteria: Section M criteria (optional)
        rfp_number: Solicitation number
        rfp_title: RFP title
        strict_mode: If True, fail on validation errors
        pdf_path: v6.0.2 - Path to source PDF for spatial extraction

    Returns:
        Dict with skeleton, outline, and metadata

    Raises:
        StructureValidationError: If strict_mode and validation fails
                                  (CRITICAL issues detected by Supervisor)

    Example:
        from agents.enhanced_compliance.outline_orchestrator import generate_outline_v3

        result = generate_outline_v3(
            section_l_text="Volume I: Technical Proposal (25 pages max)...",
            requirements=[{"id": "REQ-001", "text": "The contractor shall..."}],
            pdf_path="/path/to/rfp.pdf"  # v6.0.2: Enable spatial extraction
        )

        skeleton = result['proposal_skeleton']
        outline = result['annotated_outline']
    """
    orchestrator = OutlineOrchestrator(strict_mode=strict_mode)
    return orchestrator.generate_outline(
        section_l_text=section_l_text,
        requirements=requirements,
        evaluation_criteria=evaluation_criteria or [],
        rfp_number=rfp_number,
        rfp_title=rfp_title,
        pdf_path=pdf_path
    )
