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

Key Principle: Structure comes from Section L ONLY. Content comes from Section C.
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

    def validate_schema(
        self,
        schema: SectionL_Schema,
        stated_volume_count: Optional[int] = None
    ) -> SupervisorValidationResult:
        """
        Validate the parsed schema before skeleton building.

        Args:
            schema: Parsed Section L schema
            stated_volume_count: Expected volume count from RFP (if known)

        Returns:
            SupervisorValidationResult with issues and confidence
        """
        issues: List[ValidationIssue] = []
        confidence = 1.0

        volumes = schema.get('volumes', [])
        sections = schema.get('sections', [])

        # Check 1: Volume count
        if len(volumes) == 0:
            issues.append(ValidationIssue(
                code="NO_VOLUMES",
                message="No volumes found in Section L",
                severity=ValidationSeverity.CRITICAL,
                recoverable=True,
                suggested_action="Run semantic search for common volume patterns"
            ))
            confidence -= 0.5

        # Check 2: Stated vs found volume count - v6.0.1: HARD VALIDATION GATE
        # If RFP explicitly states N volumes and we found fewer, this is CRITICAL
        # The system MUST NOT generate an outline with missing volumes
        stated = stated_volume_count or schema.get('stated_volume_count')
        if stated is not None and len(volumes) != stated:
            missing_count = stated - len(volumes)
            if missing_count > 0:
                # CRITICAL: Missing volumes - MUST block outline generation
                issues.append(ValidationIssue(
                    code="VOLUME_COUNT_MISMATCH",
                    message=f"CRITICAL: RFP states {stated} volumes but only found {len(volumes)}. "
                            f"Missing {missing_count} volume(s). Cannot generate compliant outline.",
                    severity=ValidationSeverity.CRITICAL,
                    recoverable=False,  # v6.0.1: No longer recoverable - hard block
                    suggested_action=f"Manual review required: Search for {missing_count} missing volume(s) "
                                    "using pdfplumber structural search or review source PDF"
                ))
                confidence = 0.0  # v6.0.1: Zero confidence = hard failure
            else:
                # Found more volumes than stated - warning only
                issues.append(ValidationIssue(
                    code="VOLUME_COUNT_EXCESS",
                    message=f"RFP states {stated} volumes but found {len(volumes)} ({len(volumes) - stated} extra)",
                    severity=ValidationSeverity.WARNING,
                    recoverable=True,
                    suggested_action="Review if extra volumes should be consolidated or removed"
                ))
                confidence -= 0.1

        # Check 3: Sections distribution
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

        # Check 4: Empty volumes (no sections, no page limit)
        for vol in volumes:
            vol_sections = [s for s in sections if s.get('parent_volume_id') == vol.get('volume_id')]
            if not vol_sections and vol.get('page_limit') is None:
                issues.append(ValidationIssue(
                    code="EMPTY_VOLUME",
                    message=f"Volume '{vol.get('volume_title')}' has no sections or page limit",
                    severity=ValidationSeverity.WARNING,
                    recoverable=False,
                    suggested_action="Manual review required - volume may need structure from attachments"
                ))
                confidence -= 0.05

        # Check 5: Parsing warnings
        for warning in schema.get('parsing_warnings', []):
            issues.append(ValidationIssue(
                code="PARSER_WARNING",
                message=warning,
                severity=ValidationSeverity.INFO,
                recoverable=False
            ))

        # Calculate final validity
        is_valid = (
            confidence >= self.MIN_CONFIDENCE_FOR_VALID and
            not any(i.severity == ValidationSeverity.CRITICAL for i in issues)
        )

        return SupervisorValidationResult(
            is_valid=is_valid,
            confidence_score=max(0.0, confidence),
            issues=issues,
            metrics={
                'volume_count': len(volumes),
                'section_count': len(sections),
                'stated_volume_count': stated,
                'orphan_sections': len(orphan_sections),
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
        self.parser = SectionLParser()
        self.structure_builder = StrictStructureBuilder(strict_mode=strict_mode)
        self.content_injector = ContentInjector()
        self.supervisor = SupervisorValidator() if enable_supervisor else None

    def generate_outline(
        self,
        section_l_text: str,
        requirements: List[Dict],
        evaluation_criteria: List[Dict],
        rfp_number: str = "",
        rfp_title: str = "",
        attachment_texts: Optional[Dict[str, str]] = None,
        pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate annotated outline using two-phase pipeline.

        This is the main entry point for outline generation.

        Args:
            section_l_text: Full text of Section L instructions
            requirements: Section C requirements from CTM
            evaluation_criteria: Section M criteria from CTM
            rfp_number: Solicitation number
            rfp_title: RFP title
            attachment_texts: Optional dict of structural attachment texts
            pdf_path: v6.0.2 - Path to source PDF for spatial extraction (SF1449, tables)

        Returns:
            Dict containing:
            - proposal_skeleton: Skeleton from Component A (for state storage)
            - annotated_outline: Populated outline from Component B
            - injection_metadata: Statistics about the injection
            - unmapped_requirements: Requirements that couldn't be mapped

        Raises:
            StructureValidationError: If strict_mode and structure is invalid
                                      OR if Supervisor detects CRITICAL validation issues
        """
        logger.info(f"Starting outline generation for RFP: {rfp_number}")
        logger.info(f"Section L text length: {len(section_l_text)} chars")
        logger.info(f"Requirements count: {len(requirements)}")
        if pdf_path:
            logger.info(f"[v6.0.2] PDF path provided for spatial extraction: {pdf_path}")

        # Debug: Print for Render logs
        print(f"[v3.0 Parser] Section L text preview (first 500 chars): {section_l_text[:500]}")

        # Phase 1: Parse Section L into schema
        # v6.0.2: Pass pdf_path for spatial extraction (SF1449, digit-targeting tables)
        logger.info("Phase 1: Parsing Section L into schema...")
        schema = self.parser.parse(
            section_l_text=section_l_text,
            rfp_number=rfp_number,
            rfp_title=rfp_title,
            attachment_texts=attachment_texts,
            strict_mode=self.strict_mode,
            pdf_path=pdf_path
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

        # v6.0: Supervisor validation gate for schema
        schema_validation = None
        if self.supervisor:
            logger.info("Phase 1.5: Supervisor validation of schema...")
            schema_validation = self.supervisor.validate_schema(schema)
            print(f"[v6.0 Supervisor] Schema validation: valid={schema_validation.is_valid}, "
                  f"confidence={schema_validation.confidence_score:.2f}")

            for issue in schema_validation.issues:
                if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
                    print(f"[v6.0 Supervisor] {issue.severity.value.upper()}: {issue.message}")
                    if issue.suggested_action:
                        print(f"[v6.0 Supervisor]   Suggested: {issue.suggested_action}")

            # v6.0.1: HARD VALIDATION GATE - Block outline generation on CRITICAL issues
            if schema_validation.has_critical_issues and self.strict_mode:
                critical_issues = [
                    i for i in schema_validation.issues
                    if i.severity == ValidationSeverity.CRITICAL
                ]
                error_messages = "; ".join(i.message for i in critical_issues)
                raise StructureValidationError(
                    f"[v6.0.1 HARD BLOCK] Cannot generate outline due to critical validation failures: "
                    f"{error_messages}"
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
            }
        }

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
