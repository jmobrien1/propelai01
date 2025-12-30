# Implementation Plan: StrictStructureBuilder + ContentInjector

## Refactoring `smart_outline_generator.py` into Two Decoupled Components

**Date:** December 30, 2025
**Target:** Split monolithic outline generator into structure-first, content-second pipeline
**Integration Point:** `ProposalState` TypedDict from `core/state.py`

---

## Confirmation: Why Decoupling is Required

The current `SmartOutlineGenerator` violates separation of concerns:

```python
# CURRENT (BROKEN) - smart_outline_generator.py:147-165
def generate_from_compliance_matrix(
    self,
    section_l_requirements: List[Dict],   # Structure source
    section_m_requirements: List[Dict],   # Evaluation criteria
    technical_requirements: List[Dict],   # Content source (Section C)
    stats: Dict
) -> ProposalOutline:
    # PROBLEM: All three are processed together
    # Structure decisions influenced by content keywords
```

**The fix:** Two sequential components with a clear handoff point.

---

## State Flow Using `ProposalState`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ProposalState (TypedDict)                         │
│                                                                             │
│  ┌─────────────────────┐      ┌─────────────────────┐                      │
│  │ instructions        │      │ requirements        │                      │
│  │ (Section L data)    │      │ (Section C data)    │                      │
│  │ List[Dict]          │      │ List[Dict]          │                      │
│  └──────────┬──────────┘      └──────────┬──────────┘                      │
│             │                            │                                  │
│             ▼                            │                                  │
│  ┌──────────────────────┐                │                                  │
│  │ StrictStructureBuilder│                │                                  │
│  │ (Component A)        │                │                                  │
│  │                      │                │                                  │
│  │ Input: instructions  │                │                                  │
│  │ Output: skeleton     │                │                                  │
│  └──────────┬───────────┘                │                                  │
│             │                            │                                  │
│             ▼                            │                                  │
│  ┌─────────────────────┐                 │                                  │
│  │ proposal_skeleton   │ ◄───────────────┤  (NEW STATE FIELD)               │
│  │ Dict[str, Any]      │                 │                                  │
│  └──────────┬──────────┘                 │                                  │
│             │                            │                                  │
│             ▼                            ▼                                  │
│  ┌───────────────────────────────────────────────┐                         │
│  │              ContentInjector                   │                         │
│  │              (Component B)                     │                         │
│  │                                                │                         │
│  │ Input: proposal_skeleton + requirements        │                         │
│  │ Output: annotated_outline                      │                         │
│  └───────────────────────┬───────────────────────┘                         │
│                          │                                                  │
│                          ▼                                                  │
│  ┌─────────────────────────┐                                               │
│  │ annotated_outline       │  (EXISTING STATE FIELD)                       │
│  │ Dict[str, Any]          │                                               │
│  └─────────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step File Edits

### STEP 1: Define `SectionL_Schema` Data Structure

**File:** `agents/enhanced_compliance/section_l_schema.py` (NEW)

```python
"""
SectionL_Schema: Structured representation of Section L instructions.

This schema captures ONLY structural instructions from Section L,
not content requirements. It is the input to StrictStructureBuilder.
"""

from typing import TypedDict, List, Optional, Dict
from enum import Enum


class VolumeInstruction(TypedDict):
    """A volume as instructed by Section L"""
    volume_id: str                    # "Volume I", "VOL-1"
    volume_title: str                 # "Technical Proposal"
    volume_number: int                # 1, 2, 3...
    page_limit: Optional[int]         # None if not specified
    source_reference: str             # "L.4.1", "Attachment 2"
    is_mandatory: bool


class SectionInstruction(TypedDict):
    """A section as instructed by Section L"""
    section_id: str                   # "L.4.B.2", "1.0"
    section_title: str                # "Technical Approach"
    parent_volume_id: str             # Which volume this belongs to
    page_limit: Optional[int]
    order: int                        # Display order within volume
    source_reference: str
    required_content_types: List[str] # ["narrative", "table", "chart"]


class FormatInstruction(TypedDict):
    """Formatting rules from Section L"""
    font_name: Optional[str]
    font_size: Optional[int]
    margins: Optional[str]
    line_spacing: Optional[str]
    page_size: Optional[str]
    header_footer_rules: Optional[str]


class SubmissionInstruction(TypedDict):
    """Submission rules from Section L"""
    due_date: Optional[str]
    due_time: Optional[str]
    submission_method: Optional[str]
    copies_required: Optional[int]
    file_format: Optional[str]


class SectionL_Schema(TypedDict):
    """
    Complete structured representation of Section L instructions.

    This is the CONTRACT between the extraction layer and StrictStructureBuilder.
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
```

---

### STEP 2: Create `StrictStructureBuilder` (Component A)

**File:** `agents/enhanced_compliance/strict_structure_builder.py` (NEW)

```python
"""
StrictStructureBuilder: Builds proposal skeleton from Section L only.

This component:
- Accepts SectionL_Schema as input
- Produces a skeleton structure with NO content
- Does NOT look at Section C requirements
- Does NOT use default templates
- Fails loudly if structure cannot be determined
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re

from .section_l_schema import (
    SectionL_Schema,
    VolumeInstruction,
    SectionInstruction
)


@dataclass
class SkeletonSection:
    """A section in the proposal skeleton (no content yet)"""
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
    """A volume in the proposal skeleton"""
    id: str
    title: str
    volume_number: int
    page_limit: Optional[int]
    source_reference: str
    sections: List[SkeletonSection] = field(default_factory=list)


@dataclass
class ProposalSkeleton:
    """The complete proposal structure skeleton"""
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
    """Raised when skeleton fails validation"""
    pass


class StrictStructureBuilder:
    """
    Component A: Builds proposal skeleton strictly from Section L.

    NO DEFAULTS. NO INFERENCE FROM CONTENT. SECTION L ONLY.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, raise errors on validation failure.
                        If False, return skeleton with warnings.
        """
        self.strict_mode = strict_mode

    def build_from_schema(self, schema: SectionL_Schema) -> ProposalSkeleton:
        """
        Build proposal skeleton from SectionL_Schema.

        Args:
            schema: Structured Section L data

        Returns:
            ProposalSkeleton with volumes and sections

        Raises:
            StructureValidationError: If strict_mode and validation fails
        """
        errors = []
        warnings = []

        # Build volumes from schema
        volumes = self._build_volumes(schema, errors, warnings)

        # Validate volume count if stated
        if schema.get('stated_volume_count'):
            if len(volumes) != schema['stated_volume_count']:
                errors.append(
                    f"Section L states {schema['stated_volume_count']} volumes "
                    f"but found {len(volumes)}"
                )

        # Validate page limits
        if schema.get('total_page_limit'):
            total_allocated = sum(v.page_limit or 0 for v in volumes)
            if total_allocated > schema['total_page_limit']:
                errors.append(
                    f"Total page limit is {schema['total_page_limit']} but "
                    f"volumes allocate {total_allocated} pages"
                )

        # Check for empty skeleton
        if not volumes:
            errors.append(
                "No volumes found in Section L. Cannot build skeleton."
            )

        skeleton = ProposalSkeleton(
            rfp_number=schema.get('rfp_number', ''),
            rfp_title=schema.get('rfp_title', ''),
            volumes=volumes,
            total_page_limit=schema.get('total_page_limit'),
            format_rules=dict(schema.get('format_rules', {})),
            submission_rules=dict(schema.get('submission_rules', {})),
            is_valid=(len(errors) == 0),
            validation_errors=errors,
            validation_warnings=warnings
        )

        if self.strict_mode and errors:
            raise StructureValidationError(
                f"Skeleton validation failed: {errors}"
            )

        return skeleton

    def _build_volumes(
        self,
        schema: SectionL_Schema,
        errors: List[str],
        warnings: List[str]
    ) -> List[SkeletonVolume]:
        """Build volume objects from schema"""
        volumes = []

        for vol_instr in schema.get('volumes', []):
            # Get sections for this volume
            vol_sections = [
                s for s in schema.get('sections', [])
                if s.get('parent_volume_id') == vol_instr['volume_id']
            ]

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

            volume = SkeletonVolume(
                id=vol_instr['volume_id'],
                title=vol_instr['volume_title'],
                volume_number=vol_instr['volume_number'],
                page_limit=vol_instr.get('page_limit'),
                source_reference=vol_instr.get('source_reference', ''),
                sections=sections
            )

            # Validation: Volume should have sections or page limit
            if not sections and not vol_instr.get('page_limit'):
                warnings.append(
                    f"Volume '{volume.title}' has no sections and no page limit"
                )

            volumes.append(volume)

        return sorted(volumes, key=lambda v: v.volume_number)

    def to_state_dict(self, skeleton: ProposalSkeleton) -> Dict[str, Any]:
        """
        Convert skeleton to dict for ProposalState storage.

        This is stored in state['proposal_skeleton'] for Component B.
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
```

---

### STEP 3: Create `ContentInjector` (Component B)

**File:** `agents/enhanced_compliance/content_injector.py` (NEW)

```python
"""
ContentInjector: Injects Section C requirements into proposal skeleton.

This component:
- Takes skeleton from StrictStructureBuilder
- Takes requirements from compliance matrix
- Maps requirements to skeleton sections
- Does NOT modify skeleton structure
- Produces annotated_outline for ProposalState
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class MappingConfidence(Enum):
    """Confidence level of requirement-to-section mapping"""
    EXPLICIT = "explicit"      # RFP explicitly states location
    HIGH = "high"              # Clear keyword match
    MEDIUM = "medium"          # Reasonable inference
    LOW = "low"                # Weak match, needs review
    UNMAPPED = "unmapped"      # Could not map


@dataclass
class RequirementMapping:
    """A mapping of a requirement to a skeleton section"""
    requirement_id: str
    requirement_text: str
    target_section_id: str
    target_volume_id: str
    confidence: MappingConfidence
    rationale: str


@dataclass
class InjectionResult:
    """Result of content injection"""
    success: bool
    annotated_outline: Dict[str, Any]
    total_requirements: int
    mapped_count: int
    unmapped_requirements: List[Dict]
    low_confidence_mappings: List[RequirementMapping]
    warnings: List[str]


class ContentInjector:
    """
    Component B: Injects requirements into skeleton without modifying structure.

    The skeleton is IMMUTABLE. We only fill the slots.
    """

    def __init__(self):
        # Keyword mappings for semantic matching
        self.section_keywords = {
            'technical': ['technical', 'approach', 'methodology', 'solution'],
            'management': ['management', 'project', 'program', 'oversight'],
            'staffing': ['staff', 'personnel', 'team', 'key personnel', 'resume'],
            'experience': ['experience', 'past performance', 'reference', 'similar'],
            'cost': ['cost', 'price', 'budget', 'pricing', 'rate'],
            'quality': ['quality', 'assurance', 'control', 'qa', 'qc'],
            'transition': ['transition', 'phase-in', 'mobilization'],
            'risk': ['risk', 'mitigation', 'contingency'],
        }

    def inject(
        self,
        skeleton_dict: Dict[str, Any],
        requirements: List[Dict],
        evaluation_criteria: List[Dict]
    ) -> InjectionResult:
        """
        Inject requirements into skeleton sections.

        Args:
            skeleton_dict: Skeleton from state['proposal_skeleton']
            requirements: Requirements from state['requirements'] (Section C)
            evaluation_criteria: Criteria from state['evaluation_criteria'] (Section M)

        Returns:
            InjectionResult with annotated_outline for state storage
        """
        mappings: List[RequirementMapping] = []
        unmapped: List[Dict] = []
        warnings: List[str] = []

        # Build section lookup
        section_lookup = self._build_section_lookup(skeleton_dict)

        # Map each requirement to a section
        for req in requirements:
            mapping = self._map_requirement(req, skeleton_dict, section_lookup)
            if mapping.confidence == MappingConfidence.UNMAPPED:
                unmapped.append(req)
            else:
                mappings.append(mapping)

        # Build annotated outline
        annotated_outline = self._build_annotated_outline(
            skeleton_dict,
            mappings,
            evaluation_criteria
        )

        # Identify low confidence mappings
        low_confidence = [m for m in mappings if m.confidence == MappingConfidence.LOW]

        return InjectionResult(
            success=True,
            annotated_outline=annotated_outline,
            total_requirements=len(requirements),
            mapped_count=len(mappings),
            unmapped_requirements=unmapped,
            low_confidence_mappings=low_confidence,
            warnings=warnings
        )

    def _build_section_lookup(self, skeleton: Dict) -> Dict[str, Dict]:
        """Build lookup table: section_id -> section info"""
        lookup = {}
        for vol in skeleton.get('volumes', []):
            for sec in vol.get('sections', []):
                lookup[sec['id']] = {
                    'section': sec,
                    'volume_id': vol['id'],
                    'volume_title': vol['title']
                }
        return lookup

    def _map_requirement(
        self,
        requirement: Dict,
        skeleton: Dict,
        section_lookup: Dict
    ) -> RequirementMapping:
        """Map a single requirement to a skeleton section"""
        req_id = requirement.get('id', 'unknown')
        req_text = requirement.get('text', '')

        # Try explicit reference first (e.g., "Volume I", "Section 2.0")
        explicit_target = self._find_explicit_target(requirement, skeleton)
        if explicit_target:
            return RequirementMapping(
                requirement_id=req_id,
                requirement_text=req_text[:200],
                target_section_id=explicit_target['section_id'],
                target_volume_id=explicit_target['volume_id'],
                confidence=MappingConfidence.EXPLICIT,
                rationale=f"Explicit reference to {explicit_target['reference']}"
            )

        # Try semantic matching
        semantic_target, confidence = self._semantic_match(
            requirement, skeleton, section_lookup
        )
        if semantic_target:
            return RequirementMapping(
                requirement_id=req_id,
                requirement_text=req_text[:200],
                target_section_id=semantic_target['section_id'],
                target_volume_id=semantic_target['volume_id'],
                confidence=confidence,
                rationale=semantic_target['rationale']
            )

        # Could not map
        return RequirementMapping(
            requirement_id=req_id,
            requirement_text=req_text[:200],
            target_section_id='',
            target_volume_id='',
            confidence=MappingConfidence.UNMAPPED,
            rationale="No matching section found"
        )

    def _find_explicit_target(
        self,
        requirement: Dict,
        skeleton: Dict
    ) -> Optional[Dict]:
        """Find explicit volume/section reference in requirement"""
        text = requirement.get('text', '').lower()

        # Look for "Volume I", "Volume 1", etc.
        vol_match = re.search(r'volume\s*([ivx\d]+)', text, re.IGNORECASE)
        if vol_match:
            vol_ref = vol_match.group(1).upper()
            for vol in skeleton.get('volumes', []):
                if vol_ref in vol['id'].upper() or vol_ref in vol['title'].upper():
                    # Return first section of matched volume
                    if vol.get('sections'):
                        return {
                            'section_id': vol['sections'][0]['id'],
                            'volume_id': vol['id'],
                            'reference': f"Volume {vol_ref}"
                        }

        # Look for "Section X.X" references
        sec_match = re.search(r'section\s*(\d+(?:\.\d+)*)', text, re.IGNORECASE)
        if sec_match:
            sec_ref = sec_match.group(1)
            for vol in skeleton.get('volumes', []):
                for sec in vol.get('sections', []):
                    if sec_ref in sec['id']:
                        return {
                            'section_id': sec['id'],
                            'volume_id': vol['id'],
                            'reference': f"Section {sec_ref}"
                        }

        return None

    def _semantic_match(
        self,
        requirement: Dict,
        skeleton: Dict,
        section_lookup: Dict
    ) -> Tuple[Optional[Dict], MappingConfidence]:
        """Use semantic matching to find best section"""
        req_text = requirement.get('text', '').lower()
        req_type = requirement.get('requirement_type', '').lower()

        best_match = None
        best_score = 0

        for vol in skeleton.get('volumes', []):
            vol_title_lower = vol['title'].lower()

            for sec in vol.get('sections', []):
                sec_title_lower = sec['title'].lower()
                score = 0

                # Score based on keyword overlap
                for category, keywords in self.section_keywords.items():
                    # Check if section title matches category
                    if any(kw in sec_title_lower for kw in keywords):
                        # Check if requirement mentions same keywords
                        if any(kw in req_text for kw in keywords):
                            score += 10
                        if category in req_type:
                            score += 15

                # Boost for requirement type matching volume type
                if 'technical' in req_type and 'technical' in vol_title_lower:
                    score += 20
                if 'management' in req_type and 'management' in vol_title_lower:
                    score += 20

                if score > best_score:
                    best_score = score
                    best_match = {
                        'section_id': sec['id'],
                        'volume_id': vol['id'],
                        'rationale': f"Keyword match (score: {score})"
                    }

        if best_match:
            if best_score >= 20:
                return best_match, MappingConfidence.HIGH
            elif best_score >= 10:
                return best_match, MappingConfidence.MEDIUM
            else:
                return best_match, MappingConfidence.LOW

        return None, MappingConfidence.UNMAPPED

    def _build_annotated_outline(
        self,
        skeleton: Dict,
        mappings: List[RequirementMapping],
        evaluation_criteria: List[Dict]
    ) -> Dict[str, Any]:
        """Build the final annotated outline for state storage"""

        # Group mappings by section
        section_reqs: Dict[str, List[RequirementMapping]] = {}
        for mapping in mappings:
            if mapping.target_section_id:
                if mapping.target_section_id not in section_reqs:
                    section_reqs[mapping.target_section_id] = []
                section_reqs[mapping.target_section_id].append(mapping)

        # Build output structure
        volumes = []
        for vol in skeleton.get('volumes', []):
            sections = []
            for sec in vol.get('sections', []):
                sec_mappings = section_reqs.get(sec['id'], [])

                section_data = {
                    'id': sec['id'],
                    'title': sec['title'],
                    'page_limit': sec.get('page_limit'),
                    'order': sec.get('order', 0),
                    # Injected content
                    'requirements': [
                        {
                            'id': m.requirement_id,
                            'text': m.requirement_text,
                            'confidence': m.confidence.value,
                            'rationale': m.rationale
                        }
                        for m in sec_mappings
                    ],
                    'requirement_count': len(sec_mappings),
                    # Placeholders for manual completion
                    'win_themes': [],
                    'proof_points': [],
                    'graphics': [],
                    'compliance_checkpoints': [
                        f"Address: {m.requirement_text[:80]}..."
                        for m in sec_mappings[:5]
                    ]
                }
                sections.append(section_data)

            volume_data = {
                'id': vol['id'],
                'title': vol['title'],
                'volume_number': vol.get('volume_number', 0),
                'page_limit': vol.get('page_limit'),
                'sections': sections,
                'total_requirements': sum(s['requirement_count'] for s in sections)
            }
            volumes.append(volume_data)

        return {
            'rfp_number': skeleton.get('rfp_number', ''),
            'rfp_title': skeleton.get('rfp_title', ''),
            'total_page_limit': skeleton.get('total_page_limit'),
            'format_rules': skeleton.get('format_rules', {}),
            'submission_rules': skeleton.get('submission_rules', {}),
            'volumes': volumes,
            'evaluation_criteria': evaluation_criteria,
            'generation_metadata': {
                'total_requirements_mapped': len(mappings),
                'structure_source': 'Section L (StrictStructureBuilder)',
                'content_source': 'Section C (ContentInjector)'
            }
        }
```

---

### STEP 4: Add New State Field to `ProposalState`

**File:** `core/state.py` (MODIFY)

**Edit 1:** Add `proposal_skeleton` field to ProposalState TypedDict (around line 179)

```python
# Location: After line 179 (after annotated_outline)

    # === Structure (NEW) ===
    proposal_skeleton: Dict[str, Any]        # Skeleton from StrictStructureBuilder
```

**Edit 2:** Update `create_initial_state` factory function (around line 240)

```python
# Location: After line 240 (after annotated_outline={})

        # Structure
        proposal_skeleton={},
```

---

### STEP 5: Create `SectionLParser` (Parses raw text to SectionL_Schema)

**File:** `agents/enhanced_compliance/section_l_parser.py` (NEW)

```python
"""
SectionLParser: Parses Section L text into SectionL_Schema.

This is the bridge between raw RFP text and StrictStructureBuilder.
"""

from typing import List, Optional, Dict, Any
import re

from .section_l_schema import (
    SectionL_Schema,
    VolumeInstruction,
    SectionInstruction,
    FormatInstruction,
    SubmissionInstruction
)


class SectionLParser:
    """
    Parses Section L text into structured SectionL_Schema.

    This parser extracts ONLY structure information, not content requirements.
    """

    def __init__(self):
        self.volume_patterns = [
            # "Volume I: Technical Proposal"
            r"Volume\s+([IVX\d]+)\s*[:\-–]\s*([^\n]+)",
            # "The proposal shall consist of two (2) volumes:"
            r"proposal\s+shall\s+consist\s+of\s+(\w+)\s*\(?\d*\)?\s*volumes?",
            # "Submit the following volumes:"
            r"submit\s+the\s+following\s+volumes?\s*:",
        ]

        self.section_patterns = [
            # "1.0 Executive Summary"
            r"(\d+\.\d*)\s+([A-Z][^\n]+)",
            # "Section A: Technical Approach"
            r"Section\s+([A-Z])\s*[:\-]\s*([^\n]+)",
            # "(a) Technical Approach"
            r"\(([a-z])\)\s+([A-Z][^\n]+)",
        ]

        self.page_limit_patterns = [
            r"(?:not\s+to\s+exceed|maximum\s+of|limited\s+to|no\s+more\s+than)\s+(\d+)\s*pages?",
            r"(\d+)\s*page\s*(?:limit|maximum)",
            r"\((\d+)\s*pages?\s*(?:max|maximum)?\)",
        ]

    def parse(
        self,
        section_l_text: str,
        rfp_number: str = "",
        rfp_title: str = "",
        attachment_texts: Optional[Dict[str, str]] = None
    ) -> SectionL_Schema:
        """
        Parse Section L text into structured schema.

        Args:
            section_l_text: Full text of Section L
            rfp_number: RFP/Solicitation number
            rfp_title: RFP title
            attachment_texts: Dict of attachment_name -> text for structural attachments

        Returns:
            SectionL_Schema ready for StrictStructureBuilder
        """
        warnings = []

        # Combine Section L with any structural attachments
        full_text = section_l_text
        if attachment_texts:
            for name, text in attachment_texts.items():
                if self._is_structural_attachment(name):
                    full_text += f"\n\n--- {name} ---\n{text}"

        # Extract volumes
        volumes = self._extract_volumes(full_text, warnings)

        # Extract stated volume count
        stated_count = self._extract_stated_volume_count(full_text)

        # Extract sections for each volume
        sections = self._extract_sections(full_text, volumes, warnings)

        # Extract format requirements
        format_rules = self._extract_format_rules(full_text)

        # Extract submission requirements
        submission_rules = self._extract_submission_rules(full_text)

        # Extract total page limit
        total_pages = self._extract_total_page_limit(full_text)

        return SectionL_Schema(
            rfp_number=rfp_number,
            rfp_title=rfp_title,
            volumes=volumes,
            sections=sections,
            format_rules=format_rules,
            submission_rules=submission_rules,
            total_page_limit=total_pages,
            stated_volume_count=stated_count,
            source_documents=['Section L'] + list(attachment_texts.keys() if attachment_texts else []),
            parsing_warnings=warnings
        )

    def _is_structural_attachment(self, name: str) -> bool:
        """Check if attachment contains structure instructions"""
        structural_keywords = [
            'placement', 'format', 'instruction', 'procedure',
            'proposal format', 'submission format'
        ]
        name_lower = name.lower()
        return any(kw in name_lower for kw in structural_keywords)

    def _extract_volumes(self, text: str, warnings: List[str]) -> List[VolumeInstruction]:
        """Extract volume instructions from text"""
        volumes = []
        seen_titles = set()

        # Pattern 1: "Volume I: Title"
        for match in re.finditer(r"Volume\s+([IVX\d]+)\s*[:\-–]\s*([^\n]+)", text, re.IGNORECASE):
            vol_num = self._roman_to_int(match.group(1))
            title = match.group(2).strip()

            if title.lower() not in seen_titles:
                seen_titles.add(title.lower())

                # Look for page limit near this volume
                page_limit = self._find_page_limit_near(text, match.end(), title)

                volumes.append(VolumeInstruction(
                    volume_id=f"VOL-{vol_num}",
                    volume_title=title,
                    volume_number=vol_num,
                    page_limit=page_limit,
                    source_reference=f"Section L (Volume {match.group(1)})",
                    is_mandatory=True
                ))

        # If no explicit volumes, look for volume mentions
        if not volumes:
            # Look for "Technical Proposal" and "Price Proposal" etc.
            vol_indicators = [
                ("Technical Proposal", 1),
                ("Technical Volume", 1),
                ("Price Proposal", 2),
                ("Cost Proposal", 2),
                ("Business Proposal", 2),
            ]

            for title, num in vol_indicators:
                if title.lower() in text.lower() and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    page_limit = self._find_page_limit_near(text, text.lower().find(title.lower()), title)

                    volumes.append(VolumeInstruction(
                        volume_id=f"VOL-{num}",
                        volume_title=title,
                        volume_number=num,
                        page_limit=page_limit,
                        source_reference="Section L (inferred)",
                        is_mandatory=True
                    ))

            if not volumes:
                warnings.append("No explicit volumes found in Section L")

        return sorted(volumes, key=lambda v: v['volume_number'])

    def _extract_stated_volume_count(self, text: str) -> Optional[int]:
        """Extract stated volume count (e.g., 'consist of two volumes')"""
        patterns = [
            r"consist\s+of\s+(\w+)\s*\(?\d*\)?\s*volumes?",
            r"(\w+)\s*\(?\d*\)?\s*volumes?\s+(?:are\s+)?required",
            r"submit\s+(\w+)\s*\(?\d*\)?\s*volumes?",
        ]

        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num_str = match.group(1).lower()
                if num_str in number_words:
                    return number_words[num_str]
                elif num_str.isdigit():
                    return int(num_str)

        return None

    def _extract_sections(
        self,
        text: str,
        volumes: List[VolumeInstruction],
        warnings: List[str]
    ) -> List[SectionInstruction]:
        """Extract section instructions"""
        sections = []

        # For each volume, look for section structure
        for vol in volumes:
            vol_title = vol['volume_title']

            # Find the volume's text block
            vol_pattern = re.escape(vol_title)
            vol_match = re.search(vol_pattern, text, re.IGNORECASE)

            if vol_match:
                # Get text following volume title (next 3000 chars or until next volume)
                start = vol_match.end()
                end = min(start + 3000, len(text))

                # Check for next volume to limit scope
                for other_vol in volumes:
                    if other_vol['volume_number'] > vol['volume_number']:
                        other_match = re.search(
                            re.escape(other_vol['volume_title']),
                            text[start:],
                            re.IGNORECASE
                        )
                        if other_match:
                            end = min(end, start + other_match.start())

                vol_text = text[start:end]

                # Extract numbered sections
                order = 0
                for match in re.finditer(r"(\d+\.\d*)\s+([A-Z][^\n]{5,60})", vol_text):
                    sec_id = match.group(1)
                    sec_title = match.group(2).strip()

                    # Find page limit for this section
                    page_limit = self._find_page_limit_near(vol_text, match.end(), sec_title)

                    sections.append(SectionInstruction(
                        section_id=sec_id,
                        section_title=sec_title,
                        parent_volume_id=vol['volume_id'],
                        page_limit=page_limit,
                        order=order,
                        source_reference=f"Section L ({vol_title})",
                        required_content_types=[]
                    ))
                    order += 1

        return sections

    def _find_page_limit_near(self, text: str, position: int, context: str) -> Optional[int]:
        """Find page limit mentioned near a position"""
        # Look in the next 200 characters
        search_text = text[position:position+200].lower()

        for pattern in self.page_limit_patterns:
            match = re.search(pattern, search_text)
            if match:
                return int(match.group(1))

        # Also check if context contains page limit
        context_lower = context.lower()
        for pattern in self.page_limit_patterns:
            match = re.search(pattern, context_lower)
            if match:
                return int(match.group(1))

        return None

    def _extract_total_page_limit(self, text: str) -> Optional[int]:
        """Extract total page limit for entire proposal"""
        patterns = [
            r"total\s+(?:of\s+)?(\d+)\s*pages?",
            r"(?:proposal|submission)\s+(?:shall\s+)?(?:not\s+exceed|be\s+limited\s+to)\s+(\d+)\s*pages?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _extract_format_rules(self, text: str) -> FormatInstruction:
        """Extract format requirements"""
        font_match = re.search(
            r"(Times\s*New\s*Roman|Arial|Calibri|Courier)",
            text,
            re.IGNORECASE
        )
        size_match = re.search(r"(\d+)\s*[-]?\s*point", text, re.IGNORECASE)
        margin_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|in|\")\s*margins?", text, re.IGNORECASE)
        spacing_match = re.search(r"(single|double|1\.5)\s*[-]?\s*spac", text, re.IGNORECASE)

        return FormatInstruction(
            font_name=font_match.group(1).strip() if font_match else None,
            font_size=int(size_match.group(1)) if size_match else None,
            margins=f"{margin_match.group(1)} inch" if margin_match else None,
            line_spacing=spacing_match.group(1).lower() if spacing_match else None,
            page_size=None,
            header_footer_rules=None
        )

    def _extract_submission_rules(self, text: str) -> SubmissionInstruction:
        """Extract submission requirements"""
        # Due date
        date_match = re.search(
            r"(?:due|submit|submission)\s*(?:date|by)?\s*[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            text,
            re.IGNORECASE
        )

        # Method
        method_match = re.search(
            r"(?:submit|submission)\s+(?:via|through|to)\s+(email|portal|electronic|mail)",
            text,
            re.IGNORECASE
        )

        return SubmissionInstruction(
            due_date=date_match.group(1) if date_match else None,
            due_time=None,
            submission_method=method_match.group(1).lower() if method_match else None,
            copies_required=None,
            file_format=None
        )

    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral or digit string to int"""
        if roman.isdigit():
            return int(roman)

        roman_values = {'I': 1, 'V': 5, 'X': 10}
        result = 0
        roman = roman.upper()

        for i, char in enumerate(roman):
            if char not in roman_values:
                return 1  # Default if not valid

            if i + 1 < len(roman) and roman_values.get(char, 0) < roman_values.get(roman[i + 1], 0):
                result -= roman_values[char]
            else:
                result += roman_values[char]

        return result if result > 0 else 1
```

---

### STEP 6: Create Orchestrator Function

**File:** `agents/enhanced_compliance/outline_orchestrator.py` (NEW)

```python
"""
OutlineOrchestrator: Coordinates StrictStructureBuilder and ContentInjector.

This is the main entry point for the new outline generation pipeline.
"""

from typing import Dict, List, Any, Optional

from .section_l_parser import SectionLParser
from .section_l_schema import SectionL_Schema
from .strict_structure_builder import StrictStructureBuilder, ProposalSkeleton
from .content_injector import ContentInjector, InjectionResult


class OutlineOrchestrator:
    """
    Orchestrates the two-phase outline generation:
    1. StrictStructureBuilder: Section L -> Skeleton
    2. ContentInjector: Skeleton + Requirements -> Annotated Outline
    """

    def __init__(self, strict_mode: bool = True):
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

        Args:
            section_l_text: Full text of Section L (from state['instructions'])
            requirements: Section C requirements (from state['requirements'])
            evaluation_criteria: Section M criteria (from state['evaluation_criteria'])
            rfp_number: Solicitation number
            rfp_title: RFP title
            attachment_texts: Structural attachment texts

        Returns:
            Dict with 'skeleton' and 'annotated_outline' for state storage
        """
        # Phase 1: Parse Section L into schema
        schema = self.parser.parse(
            section_l_text=section_l_text,
            rfp_number=rfp_number,
            rfp_title=rfp_title,
            attachment_texts=attachment_texts
        )

        # Phase 2: Build skeleton from schema
        skeleton = self.structure_builder.build_from_schema(schema)
        skeleton_dict = self.structure_builder.to_state_dict(skeleton)

        # Phase 3: Inject requirements into skeleton
        result = self.content_injector.inject(
            skeleton_dict=skeleton_dict,
            requirements=requirements,
            evaluation_criteria=evaluation_criteria
        )

        return {
            'proposal_skeleton': skeleton_dict,
            'annotated_outline': result.annotated_outline,
            'injection_metadata': {
                'total_requirements': result.total_requirements,
                'mapped_count': result.mapped_count,
                'unmapped_count': len(result.unmapped_requirements),
                'warnings': result.warnings
            },
            'unmapped_requirements': result.unmapped_requirements
        }

    def generate_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate outline directly from ProposalState.

        Args:
            state: ProposalState TypedDict

        Returns:
            Dict with updates to apply to state
        """
        # Extract Section L text from instructions
        section_l_text = self._extract_section_l_text(state.get('instructions', []))

        return self.generate_outline(
            section_l_text=section_l_text,
            requirements=state.get('requirements', []),
            evaluation_criteria=state.get('evaluation_criteria', []),
            rfp_number=state.get('solicitation_number', ''),
            rfp_title=state.get('opportunity_name', ''),
            attachment_texts=None  # Can be extended to pull from state
        )

    def _extract_section_l_text(self, instructions: List[Dict]) -> str:
        """Extract Section L text from instructions list"""
        texts = []
        for instr in instructions:
            if instr.get('text'):
                texts.append(instr['text'])
            elif instr.get('full_text'):
                texts.append(instr['full_text'])
        return '\n\n'.join(texts)
```

---

### STEP 7: Update API Endpoint

**File:** `api/main.py` (MODIFY)

**Edit:** Add new v3 outline endpoint (after existing outline endpoints, ~line 1480)

```python
# ============== Proposal Outline v3 (Decoupled) ==============

@app.post("/api/rfp/{rfp_id}/outline/v3")
async def generate_outline_v3(rfp_id: str, strict_mode: bool = True):
    """
    Generate proposal outline using decoupled architecture.

    Phase 1: StrictStructureBuilder creates skeleton from Section L only
    Phase 2: ContentInjector maps requirements into skeleton

    Args:
        rfp_id: RFP project ID
        strict_mode: If True, fail if structure cannot be determined
    """
    from agents.enhanced_compliance.outline_orchestrator import OutlineOrchestrator

    rfp = store.get(rfp_id)
    if not rfp:
        raise HTTPException(status_code=404, detail="RFP not found")

    requirements = rfp.get("requirements", [])
    if not requirements:
        raise HTTPException(
            status_code=400,
            detail="No requirements extracted. Process RFP first."
        )

    # Get Section L instructions
    instructions = [r for r in requirements if r.get("category") == "L_COMPLIANCE"
                   or r.get("section", "").upper().startswith("L")]

    # Get Section C requirements
    technical_reqs = [r for r in requirements if r.get("category") == "TECHNICAL"
                     or r.get("section", "").upper() in ["C", "PWS", "SOW"]]

    # Get Section M evaluation criteria
    eval_criteria = [r for r in requirements if r.get("category") == "EVALUATION"
                    or r.get("section", "").upper().startswith("M")]

    # Build Section L text
    section_l_text = "\n\n".join([
        r.get("text", "") or r.get("full_text", "")
        for r in instructions
    ])

    if not section_l_text:
        raise HTTPException(
            status_code=400,
            detail="No Section L instructions found. Cannot build structure."
        )

    try:
        orchestrator = OutlineOrchestrator(strict_mode=strict_mode)
        result = orchestrator.generate_outline(
            section_l_text=section_l_text,
            requirements=technical_reqs,
            evaluation_criteria=eval_criteria,
            rfp_number=rfp.get("solicitation_number", ""),
            rfp_title=rfp.get("name", "")
        )

        # Store results
        store.update(rfp_id, {
            "proposal_skeleton": result["proposal_skeleton"],
            "outline": result["annotated_outline"],
            "outline_metadata": result["injection_metadata"]
        })

        return {
            "status": "success",
            "skeleton_valid": result["proposal_skeleton"].get("is_valid", False),
            "volumes_count": len(result["proposal_skeleton"].get("volumes", [])),
            "requirements_mapped": result["injection_metadata"]["mapped_count"],
            "requirements_unmapped": result["injection_metadata"]["unmapped_count"],
            "warnings": result["injection_metadata"]["warnings"],
            "outline": result["annotated_outline"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Outline generation failed: {str(e)}"
        )
```

---

### STEP 8: Update Module Exports

**File:** `agents/enhanced_compliance/__init__.py` (MODIFY)

**Edit:** Add exports for new components

```python
# Add to imports section
from .section_l_schema import (
    SectionL_Schema,
    VolumeInstruction,
    SectionInstruction,
    FormatInstruction,
    SubmissionInstruction
)
from .section_l_parser import SectionLParser
from .strict_structure_builder import (
    StrictStructureBuilder,
    ProposalSkeleton,
    SkeletonVolume,
    SkeletonSection,
    StructureValidationError
)
from .content_injector import (
    ContentInjector,
    InjectionResult,
    MappingConfidence
)
from .outline_orchestrator import OutlineOrchestrator

# Add to __all__ list
__all__ = [
    # ... existing exports ...
    # New components
    'SectionL_Schema',
    'SectionLParser',
    'StrictStructureBuilder',
    'ProposalSkeleton',
    'ContentInjector',
    'InjectionResult',
    'OutlineOrchestrator',
]
```

---

## Summary: File Edit Checklist

| Step | File | Action | Description |
|------|------|--------|-------------|
| 1 | `agents/enhanced_compliance/section_l_schema.py` | CREATE | SectionL_Schema TypedDict definitions |
| 2 | `agents/enhanced_compliance/strict_structure_builder.py` | CREATE | Component A - builds skeleton from Section L |
| 3 | `agents/enhanced_compliance/content_injector.py` | CREATE | Component B - injects requirements into skeleton |
| 4 | `core/state.py` | MODIFY | Add `proposal_skeleton` field to ProposalState |
| 5 | `agents/enhanced_compliance/section_l_parser.py` | CREATE | Parses raw Section L text to SectionL_Schema |
| 6 | `agents/enhanced_compliance/outline_orchestrator.py` | CREATE | Coordinates both components |
| 7 | `api/main.py` | MODIFY | Add `/outline/v3` endpoint |
| 8 | `agents/enhanced_compliance/__init__.py` | MODIFY | Export new components |

---

## State Flow Diagram

```
ProposalState
├── instructions (Section L)  ─────┐
│                                  │
│                                  ▼
│                         ┌─────────────────────┐
│                         │   SectionLParser    │
│                         │   (Text → Schema)   │
│                         └──────────┬──────────┘
│                                    │
│                                    ▼
│                         ┌─────────────────────┐
│                         │ StrictStructureBuilder│
│                         │ (Schema → Skeleton)  │
│                         └──────────┬──────────┘
│                                    │
│                                    ▼
├── proposal_skeleton (NEW)  ◄───────┘
│   └── volumes[]
│       └── sections[]
│           └── requirement_slots[]
│
├── requirements (Section C) ─────┐
│                                 │
│                                 ▼
│                         ┌─────────────────────┐
│                         │   ContentInjector   │
│                         │ (Skeleton + Reqs)   │
│                         └──────────┬──────────┘
│                                    │
│                                    ▼
└── annotated_outline  ◄─────────────┘
    └── volumes[]
        └── sections[]
            └── requirements[] (injected)
```

---

## Deprecation Note

After this refactoring, `smart_outline_generator.py` should be marked deprecated:

```python
# smart_outline_generator.py - Add at top of file
import warnings

warnings.warn(
    "SmartOutlineGenerator is deprecated. Use OutlineOrchestrator with "
    "StrictStructureBuilder and ContentInjector instead.",
    DeprecationWarning,
    stacklevel=2
)
```

The old `/outline` endpoints can remain for backward compatibility but should redirect to v3 internally in a future release.

---

*End of Implementation Plan*
