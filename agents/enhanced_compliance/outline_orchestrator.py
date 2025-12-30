"""
PropelAI v3.0: OutlineOrchestrator - Coordinates StrictStructureBuilder and ContentInjector.

This is the main entry point for the new decoupled outline generation pipeline.

Pipeline Flow:
1. SectionLParser: Raw Section L text -> SectionL_Schema
2. StrictStructureBuilder (Component A): SectionL_Schema -> ProposalSkeleton
3. ContentInjector (Component B): ProposalSkeleton + Requirements -> Annotated Outline

Key Principle: Structure comes from Section L ONLY. Content comes from Section C.
"""

from typing import Dict, List, Any, Optional
import logging

from .section_l_parser import SectionLParser
from .section_l_schema import SectionL_Schema
from .strict_structure_builder import (
    StrictStructureBuilder,
    ProposalSkeleton,
    StructureValidationError
)
from .content_injector import ContentInjector, InjectionResult


logger = logging.getLogger(__name__)


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

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the orchestrator.

        Args:
            strict_mode: If True, fail on structure validation errors.
                        If False, continue with warnings (for debugging).
        """
        self.strict_mode = strict_mode
        self.parser = SectionLParser()
        self.structure_builder = StrictStructureBuilder(strict_mode=strict_mode)
        self.content_injector = ContentInjector()

    def generate_outline(
        self,
        section_l_text: str,
        requirements: List[Dict],
        evaluation_criteria: List[Dict],
        rfp_number: str = "",
        rfp_title: str = "",
        attachment_texts: Optional[Dict[str, str]] = None
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

        Returns:
            Dict containing:
            - proposal_skeleton: Skeleton from Component A (for state storage)
            - annotated_outline: Populated outline from Component B
            - injection_metadata: Statistics about the injection
            - unmapped_requirements: Requirements that couldn't be mapped

        Raises:
            StructureValidationError: If strict_mode and structure is invalid
        """
        logger.info(f"Starting outline generation for RFP: {rfp_number}")
        logger.info(f"Section L text length: {len(section_l_text)} chars")
        logger.info(f"Requirements count: {len(requirements)}")

        # Phase 1: Parse Section L into schema
        logger.info("Phase 1: Parsing Section L into schema...")
        schema = self.parser.parse(
            section_l_text=section_l_text,
            rfp_number=rfp_number,
            rfp_title=rfp_title,
            attachment_texts=attachment_texts
        )

        logger.info(f"  Found {len(schema.get('volumes', []))} volumes")
        logger.info(f"  Found {len(schema.get('sections', []))} sections")
        if schema.get('parsing_warnings'):
            for warn in schema['parsing_warnings']:
                logger.warning(f"  Parser warning: {warn}")

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
            ]
        }

    def generate_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate outline directly from ProposalState.

        This is a convenience method that extracts the necessary
        data from a ProposalState TypedDict.

        Args:
            state: ProposalState TypedDict

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

        return self.generate_outline(
            section_l_text=section_l_text,
            requirements=state.get('requirements', []),
            evaluation_criteria=state.get('evaluation_criteria', []),
            rfp_number=state.get('solicitation_number', ''),
            rfp_title=state.get('opportunity_name', ''),
            attachment_texts=None  # Could be extended to pull from state
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
    strict_mode: bool = True
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

    Returns:
        Dict with skeleton, outline, and metadata

    Example:
        from agents.enhanced_compliance.outline_orchestrator import generate_outline_v3

        result = generate_outline_v3(
            section_l_text="Volume I: Technical Proposal (25 pages max)...",
            requirements=[{"id": "REQ-001", "text": "The contractor shall..."}],
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
        rfp_title=rfp_title
    )
