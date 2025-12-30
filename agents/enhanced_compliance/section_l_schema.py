"""
PropelAI v3.0: SectionL_Schema - Structured representation of Section L instructions.

This schema captures ONLY structural instructions from Section L,
not content requirements. It is the input to StrictStructureBuilder.

The schema serves as the CONTRACT between the parsing layer and the
structure building layer, ensuring clean separation of concerns.
"""

from typing import TypedDict, List, Optional


class VolumeInstruction(TypedDict):
    """
    A volume as instructed by Section L.

    This represents a top-level proposal volume (e.g., "Volume I: Technical Proposal")
    as explicitly stated in the RFP's Section L instructions.
    """
    volume_id: str                    # "Volume I", "VOL-1"
    volume_title: str                 # "Technical Proposal"
    volume_number: int                # 1, 2, 3...
    page_limit: Optional[int]         # None if not specified
    source_reference: str             # "L.4.1", "Attachment 2"
    is_mandatory: bool


class SectionInstruction(TypedDict):
    """
    A section as instructed by Section L.

    This represents a section within a volume (e.g., "1.0 Executive Summary")
    as explicitly stated in the RFP's Section L instructions.
    """
    section_id: str                   # "L.4.B.2", "1.0"
    section_title: str                # "Technical Approach"
    parent_volume_id: str             # Which volume this belongs to
    page_limit: Optional[int]
    order: int                        # Display order within volume
    source_reference: str
    required_content_types: List[str] # ["narrative", "table", "chart"]


class FormatInstruction(TypedDict):
    """
    Formatting rules from Section L.

    These are the explicit formatting requirements that the proposal
    must follow (font, margins, spacing, etc.).
    """
    font_name: Optional[str]
    font_size: Optional[int]
    margins: Optional[str]
    line_spacing: Optional[str]
    page_size: Optional[str]
    header_footer_rules: Optional[str]


class SubmissionInstruction(TypedDict):
    """
    Submission rules from Section L.

    These are the explicit submission requirements (due date, method, etc.).
    """
    due_date: Optional[str]
    due_time: Optional[str]
    submission_method: Optional[str]
    copies_required: Optional[int]
    file_format: Optional[str]


class SectionL_Schema(TypedDict):
    """
    Complete structured representation of Section L instructions.

    This is the CONTRACT between the extraction layer and StrictStructureBuilder.
    It contains ONLY structural information extracted from Section L and
    any structural attachments (e.g., "Attachment 2: Placement Procedures").

    Key principle: This schema drives the proposal skeleton structure.
    NO DEFAULTS should be applied if information is missing - instead,
    parsing_warnings should be populated to flag gaps.
    """
    # Metadata
    rfp_number: str
    rfp_title: str

    # Structure instructions (THE SKELETON)
    volumes: List[VolumeInstruction]
    sections: List[SectionInstruction]

    # Format instructions
    format_rules: FormatInstruction
    submission_rules: SubmissionInstruction

    # Validation data
    total_page_limit: Optional[int]
    stated_volume_count: Optional[int]   # "proposal shall consist of X volumes"

    # Source tracking
    source_documents: List[str]
    parsing_warnings: List[str]
