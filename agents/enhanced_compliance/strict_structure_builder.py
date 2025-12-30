"""
PropelAI v3.0: StrictStructureBuilder - Component A

Builds proposal skeleton from Section L only.

This component:
- Accepts SectionL_Schema as input
- Produces a skeleton structure with NO content
- Does NOT look at Section C requirements
- Does NOT use default templates
- Fails loudly if structure cannot be determined

Key Principle: NO DEFAULTS. NO INFERENCE FROM CONTENT. SECTION L ONLY.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .section_l_schema import SectionL_Schema


@dataclass
class SkeletonSection:
    """
    A section in the proposal skeleton (no content yet).

    This represents a structural placeholder that will be filled
    by ContentInjector (Component B) with actual requirements.
    """
    id: str
    title: str
    page_limit: Optional[int]
    order: int
    source_reference: str
    # Slots for content injection (filled by Component B)
    requirement_slots: List[str] = field(default_factory=list)
    eval_factor_slots: List[str] = field(default_factory=list)


@dataclass
class SkeletonVolume:
    """
    A volume in the proposal skeleton.

    This represents a top-level volume as specified in Section L.
    """
    id: str
    title: str
    volume_number: int
    page_limit: Optional[int]
    source_reference: str
    sections: List[SkeletonSection] = field(default_factory=list)


@dataclass
class ProposalSkeleton:
    """
    The complete proposal structure skeleton.

    This is the output of StrictStructureBuilder and the input to ContentInjector.
    It contains ONLY structural information - no content or requirements.
    """
    rfp_number: str
    rfp_title: str
    volumes: List[SkeletonVolume]
    total_page_limit: Optional[int]
    format_rules: Dict[str, Any]
    submission_rules: Dict[str, Any]
    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


class StructureValidationError(Exception):
    """
    Raised when skeleton fails validation in strict mode.

    This exception indicates that the Section L instructions could not
    be parsed into a valid proposal structure. The error message will
    contain details about what was missing or invalid.
    """
    pass


class StrictStructureBuilder:
    """
    Component A: Builds proposal skeleton strictly from Section L.

    NO DEFAULTS. NO INFERENCE FROM CONTENT. SECTION L ONLY.

    Usage:
        builder = StrictStructureBuilder(strict_mode=True)
        skeleton = builder.build_from_schema(section_l_schema)
        skeleton_dict = builder.to_state_dict(skeleton)

    The skeleton can then be passed to ContentInjector (Component B)
    to populate with requirements.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the structure builder.

        Args:
            strict_mode: If True, raise StructureValidationError on validation failure.
                        If False, return skeleton with warnings (for debugging).
        """
        self.strict_mode = strict_mode

    def build_from_schema(self, schema: SectionL_Schema) -> ProposalSkeleton:
        """
        Build proposal skeleton from SectionL_Schema.

        This method:
        1. Converts schema volumes/sections to skeleton objects
        2. Validates against stated constraints (volume count, page limits)
        3. Returns validated skeleton or raises error

        Args:
            schema: Structured Section L data from SectionLParser

        Returns:
            ProposalSkeleton with volumes and sections

        Raises:
            StructureValidationError: If strict_mode and validation fails
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Copy warnings from parsing phase
        warnings.extend(schema.get('parsing_warnings', []))

        # Build volumes from schema
        volumes = self._build_volumes(schema, errors, warnings)

        # Validation 1: Check stated volume count
        stated_count = schema.get('stated_volume_count')
        if stated_count is not None:
            if len(volumes) != stated_count:
                errors.append(
                    f"Section L states {stated_count} volumes "
                    f"but found {len(volumes)}. "
                    f"Found: {[v.title for v in volumes]}"
                )

        # Validation 2: Check total page limits
        total_limit = schema.get('total_page_limit')
        if total_limit is not None:
            total_allocated = sum(v.page_limit or 0 for v in volumes)
            if total_allocated > total_limit:
                errors.append(
                    f"Total page limit is {total_limit} but "
                    f"volumes allocate {total_allocated} pages"
                )

        # Validation 3: Check for empty skeleton
        if not volumes:
            errors.append(
                "No volumes found in Section L. Cannot build skeleton. "
                "Ensure Section L contains explicit volume instructions."
            )

        # Validation 4: Check volumes have content
        for vol in volumes:
            if not vol.sections and vol.page_limit is None:
                warnings.append(
                    f"Volume '{vol.title}' has no sections and no page limit. "
                    f"This may indicate missing structure information."
                )

        # Build the skeleton
        skeleton = ProposalSkeleton(
            rfp_number=schema.get('rfp_number', ''),
            rfp_title=schema.get('rfp_title', ''),
            volumes=volumes,
            total_page_limit=total_limit,
            format_rules=dict(schema.get('format_rules', {})),
            submission_rules=dict(schema.get('submission_rules', {})),
            is_valid=(len(errors) == 0),
            validation_errors=errors,
            validation_warnings=warnings
        )

        # In strict mode, raise exception on errors
        if self.strict_mode and errors:
            raise StructureValidationError(
                f"Skeleton validation failed with {len(errors)} error(s): "
                f"{'; '.join(errors)}"
            )

        return skeleton

    def _build_volumes(
        self,
        schema: SectionL_Schema,
        errors: List[str],
        warnings: List[str]
    ) -> List[SkeletonVolume]:
        """
        Build volume objects from schema.

        This converts VolumeInstruction and SectionInstruction TypedDicts
        into proper SkeletonVolume and SkeletonSection dataclass objects.
        """
        volumes: List[SkeletonVolume] = []

        for vol_instr in schema.get('volumes', []):
            # Get sections belonging to this volume
            vol_sections = [
                s for s in schema.get('sections', [])
                if s.get('parent_volume_id') == vol_instr['volume_id']
            ]

            # Convert section instructions to skeleton sections
            sections = [
                SkeletonSection(
                    id=s['section_id'],
                    title=s['section_title'],
                    page_limit=s.get('page_limit'),
                    order=s.get('order', 0),
                    source_reference=s.get('source_reference', '')
                )
                for s in sorted(vol_sections, key=lambda x: x.get('order', 0))
            ]

            # Create volume
            volume = SkeletonVolume(
                id=vol_instr['volume_id'],
                title=vol_instr['volume_title'],
                volume_number=vol_instr['volume_number'],
                page_limit=vol_instr.get('page_limit'),
                source_reference=vol_instr.get('source_reference', ''),
                sections=sections
            )

            volumes.append(volume)

        # Sort by volume number
        return sorted(volumes, key=lambda v: v.volume_number)

    def to_state_dict(self, skeleton: ProposalSkeleton) -> Dict[str, Any]:
        """
        Convert skeleton to dict for ProposalState storage.

        This serializes the skeleton into a format that can be stored
        in state['proposal_skeleton'] for handoff to ContentInjector.

        Args:
            skeleton: ProposalSkeleton object

        Returns:
            Dict suitable for JSON serialization and state storage
        """
        return {
            'rfp_number': skeleton.rfp_number,
            'rfp_title': skeleton.rfp_title,
            'total_page_limit': skeleton.total_page_limit,
            'format_rules': skeleton.format_rules,
            'submission_rules': skeleton.submission_rules,
            'is_valid': skeleton.is_valid,
            'validation_errors': skeleton.validation_errors,
            'validation_warnings': skeleton.validation_warnings,
            'volumes': [
                {
                    'id': vol.id,
                    'title': vol.title,
                    'volume_number': vol.volume_number,
                    'page_limit': vol.page_limit,
                    'source_reference': vol.source_reference,
                    'sections': [
                        {
                            'id': sec.id,
                            'title': sec.title,
                            'page_limit': sec.page_limit,
                            'order': sec.order,
                            'source_reference': sec.source_reference,
                            'requirement_slots': sec.requirement_slots,
                            'eval_factor_slots': sec.eval_factor_slots,
                        }
                        for sec in vol.sections
                    ]
                }
                for vol in skeleton.volumes
            ]
        }

    def from_state_dict(self, data: Dict[str, Any]) -> ProposalSkeleton:
        """
        Reconstruct skeleton from state dict.

        This deserializes a skeleton from state storage back into
        a ProposalSkeleton object.

        Args:
            data: Dict from state['proposal_skeleton']

        Returns:
            ProposalSkeleton object
        """
        volumes = []
        for vol_data in data.get('volumes', []):
            sections = [
                SkeletonSection(
                    id=sec['id'],
                    title=sec['title'],
                    page_limit=sec.get('page_limit'),
                    order=sec.get('order', 0),
                    source_reference=sec.get('source_reference', ''),
                    requirement_slots=sec.get('requirement_slots', []),
                    eval_factor_slots=sec.get('eval_factor_slots', [])
                )
                for sec in vol_data.get('sections', [])
            ]

            volume = SkeletonVolume(
                id=vol_data['id'],
                title=vol_data['title'],
                volume_number=vol_data.get('volume_number', 0),
                page_limit=vol_data.get('page_limit'),
                source_reference=vol_data.get('source_reference', ''),
                sections=sections
            )
            volumes.append(volume)

        return ProposalSkeleton(
            rfp_number=data.get('rfp_number', ''),
            rfp_title=data.get('rfp_title', ''),
            volumes=volumes,
            total_page_limit=data.get('total_page_limit'),
            format_rules=data.get('format_rules', {}),
            submission_rules=data.get('submission_rules', {}),
            is_valid=data.get('is_valid', False),
            validation_errors=data.get('validation_errors', []),
            validation_warnings=data.get('validation_warnings', [])
        )
