# PropelAI Technical Refactoring Plan v3.0
## Separating Outline Builder from Requirement Injector

**Date:** December 30, 2025
**Author:** Lead Systems Architect
**Target Version:** 3.0.0
**Status:** PLAN - Awaiting Approval

---

## Executive Summary

This document details a critical architectural refactoring to address a logic failure in the proposal generation pipeline. The current `SmartOutlineGenerator` conflates "Outline Structure Generation" with "Requirement Extraction/Mapping", leading to non-compliant outputs that hallucinate volumes (e.g., Past Performance) and misplace sections, ignoring explicit RFP instructions.

---

## 1. Root Cause Analysis

### 1.1 Primary Finding

The `SmartOutlineGenerator` (`agents/enhanced_compliance/smart_outline_generator.py`) uses **keyword-based inference** and **hardcoded defaults** rather than **deterministically parsing Section L instructions**.

### 1.2 Specific Code Locations

| Location | Problem | Impact |
|----------|---------|--------|
| `smart_outline_generator.py:469-484` | `_create_default_volumes()` hardcodes 4 volumes including "Past Performance" | System invents volumes not in RFP |
| `smart_outline_generator.py:182-185` | Falls back to defaults when extraction fails | Ignores explicit Section L structure |
| `smart_outline_generator.py:312-399` | `_extract_nih_volumes()` uses regex + defaults for "Factor N" names | Hallucinates factor names like "Facilities and Equipment" |
| `smart_outline_generator.py:447-467` | `_extract_standard_volumes()` matches generic keywords | Creates volumes based on keyword presence, not L instructions |
| `smart_outline_generator.py:366-381` | Hardcoded `default_factors` dictionary | Injects "default" evaluation factors not found in RFP |

### 1.3 The Failure Pattern

```
CURRENT FLOW (BROKEN):
┌─────────────────────────────────────────────────────────────────────┐
│                     SmartOutlineGenerator                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐ │
│  │ Detect RFP   │ -> │ Extract      │ -> │ Apply Defaults if     │ │
│  │ Format       │    │ Volumes by   │    │ Extraction Fails      │ │
│  │              │    │ Keywords     │    │ (HARDCODED TEMPLATES) │ │
│  └──────────────┘    └──────────────┘    └───────────────────────┘ │
│                              │                                      │
│                              v                                      │
│                  ┌─────────────────────────┐                       │
│                  │ Generate Outline with   │                       │
│                  │ Hallucinated Volumes    │ <-- PROBLEM           │
│                  └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Why Section L Instructions Were Ignored

1. **No Attachment Parsing Integration**: The `RFPStructureParser` in `document_structure.py` correctly identifies Section L boundaries but doesn't extract **specific submission format instructions** (e.g., "The proposal shall consist of TWO volumes limited to 8 pages each").

2. **No Validation Gate**: There is no checkpoint that verifies the generated outline against:
   - Explicit volume counts stated in Section L
   - Page limits per volume/section
   - Required section numbering schemes

3. **Default Fallback Strategy**: When pattern matching fails, the system uses hardcoded templates:
```python
# smart_outline_generator.py:479-484
return [
    ProposalVolume(id="VOL-TECH", name="Technical Proposal", ...),
    ProposalVolume(id="VOL-MGMT", name="Management Proposal", ...),
    ProposalVolume(id="VOL-PP", name="Past Performance", ...),  # <-- HALLUCINATED
    ProposalVolume(id="VOL-COST", name="Cost/Price Proposal", ...),
]
```

### 1.5 The "Attachment 2" Problem

The RFP included `Attachment 2. Placement Procedures.pdf` which specifies the **exact** proposal structure. However:
- The `RFPStructureParser` classifies attachments but doesn't extract structural requirements from them
- The `SmartOutlineGenerator` never receives attachment content for structure parsing
- No component treats attachment instructions as **authoritative** for outline structure

---

## 2. Proposed Architecture: The Split

### 2.1 High-Level Design

```
PROPOSED FLOW (CORRECT):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    COMPONENT A: OUTLINE BUILDER                      │   │
│  │                    (The Skeleton - Section L Driven)                 │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │   │
│  │  │ Parse        │ -> │ Extract      │ -> │ Build Strict         │  │   │
│  │  │ Section L &  │    │ Volume/      │    │ Hierarchy with       │  │   │
│  │  │ Attachments  │    │ Section      │    │ Page Limits          │  │   │
│  │  │              │    │ Instructions │    │                      │  │   │
│  │  └──────────────┘    └──────────────┘    └───────────────────────┘  │   │
│  │                              │                                       │   │
│  │                              v                                       │   │
│  │                    ┌───────────────────┐                            │   │
│  │                    │ SECTION L GATE    │ <-- VALIDATION CHECKPOINT  │   │
│  │                    │ (Validate Before  │                            │   │
│  │                    │ Proceeding)       │                            │   │
│  │                    └───────────────────┘                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    v                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  COMPONENT B: REQUIREMENT INJECTOR                   │   │
│  │                  (The Flesh - CTM Data Driven)                       │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │   │
│  │  │ Load CTM     │ -> │ Semantic     │ -> │ Inject into          │  │   │
│  │  │ Requirements │    │ Matching to  │    │ Skeleton Buckets     │  │   │
│  │  │ (L, M, C)    │    │ Skeleton     │    │ (No Structure Mod)   │  │   │
│  │  └──────────────┘    └──────────────┘    └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    v                                        │
│                         ┌─────────────────────┐                            │
│                         │  ANNOTATED OUTLINE  │                            │
│                         │  (Compliant Output) │                            │
│                         └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component A: Outline Builder (The Skeleton)

**Purpose**: Generate a proposal structure skeleton that **strictly** mirrors Section L instructions.

**Key Principles**:
1. **No Defaults**: If Section L doesn't specify a volume, it doesn't exist
2. **No Inference**: Use explicit parsing, not keyword matching
3. **Attachment Authority**: Treat structural attachments (like "Placement Procedures") as primary source
4. **Fail Loud**: If structure cannot be determined, return errors not guesses

#### Data Structure: `ProposalSkeleton`

```python
# agents/enhanced_compliance/outline_builder.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

class StructureSource(Enum):
    """Where the structural requirement came from"""
    SECTION_L = "section_l"
    ATTACHMENT = "attachment"
    AMENDMENT = "amendment"

@dataclass
class SectionConstraint:
    """A constraint on a section from Section L"""
    constraint_type: str          # "page_limit", "format", "content", "order"
    value: str                    # "8 pages", "single-spaced", etc.
    source_reference: str         # "L.4.B.2", "Attachment 2, Page 3"
    source: StructureSource
    is_mandatory: bool

@dataclass
class SkeletonSection:
    """A section in the proposal skeleton - NO CONTENT, just structure"""
    id: str                                  # RFP's own ID: "L.4.B.2", "Factor 1"
    title: str                               # Exact title from RFP
    level: int                               # 1=Volume, 2=Section, 3=Subsection
    parent_id: Optional[str] = None
    order: int = 0

    # Constraints from Section L
    page_limit: Optional[int] = None
    constraints: List[SectionConstraint] = field(default_factory=list)

    # Source tracking
    source_reference: str = ""               # Where in RFP this is defined
    source: StructureSource = StructureSource.SECTION_L

    # Children
    subsections: List['SkeletonSection'] = field(default_factory=list)

    # Placeholder for injected content (Component B fills this)
    requirement_slots: List[str] = field(default_factory=list)  # IDs only

@dataclass
class SkeletonVolume:
    """A volume in the proposal skeleton"""
    id: str                                  # "Volume I", "Technical Proposal"
    title: str
    volume_number: int                       # 1, 2, 3... as per RFP

    # Constraints
    page_limit: Optional[int] = None
    constraints: List[SectionConstraint] = field(default_factory=list)

    # Structure
    sections: List[SkeletonSection] = field(default_factory=list)

    # Source
    source_reference: str = ""
    source: StructureSource = StructureSource.SECTION_L

@dataclass
class ProposalSkeleton:
    """The complete proposal structure - NO REQUIREMENTS, just skeleton"""
    rfp_number: str
    rfp_title: str

    # Structural elements (in RFP-specified order)
    volumes: List[SkeletonVolume] = field(default_factory=list)

    # Global constraints
    total_page_limit: Optional[int] = None
    format_requirements: Dict[str, str] = field(default_factory=dict)
    submission_requirements: Dict[str, str] = field(default_factory=dict)

    # Validation status
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # Source documents used
    structure_sources: List[str] = field(default_factory=list)

@dataclass
class SectionLGateResult:
    """Result of Section L validation gate"""
    passed: bool
    skeleton: Optional[ProposalSkeleton]
    errors: List[str] = field(default_factory=list)      # Blocking issues
    warnings: List[str] = field(default_factory=list)    # Non-blocking issues

    # Specific validations
    volume_count_valid: bool = False
    page_limits_valid: bool = False
    section_order_valid: bool = False
    required_sections_present: bool = False
```

#### Class: `OutlineBuilder`

```python
# agents/enhanced_compliance/outline_builder.py

class OutlineBuilder:
    """
    Component A: Builds proposal skeleton strictly from Section L.

    NO DEFAULTS. NO INFERENCE. STRICT PARSING ONLY.
    """

    def __init__(self):
        self.section_l_patterns = self._compile_patterns()

    def build_skeleton(
        self,
        section_l_content: str,
        section_l_subsections: Dict[str, SubsectionBoundary],
        attachments: Dict[str, AttachmentInfo],
        amendments: Optional[List[Dict]] = None
    ) -> SectionLGateResult:
        """
        Build proposal skeleton from Section L and related documents.

        This method:
        1. Parses Section L for explicit volume/section structure
        2. Parses structural attachments (e.g., "Placement Procedures")
        3. Applies any amendments that modify structure
        4. Validates the skeleton through the Section L Gate
        5. Returns ONLY if validation passes

        Args:
            section_l_content: Full text of Section L
            section_l_subsections: Parsed subsections from RFPStructureParser
            attachments: Parsed attachments from RFPStructureParser
            amendments: Any amendments that modify structure

        Returns:
            SectionLGateResult with validated skeleton or errors
        """
        pass

    def _parse_explicit_volumes(self, text: str) -> List[SkeletonVolume]:
        """
        Extract ONLY volumes explicitly stated in Section L.

        Patterns to match:
        - "Volume I: Technical Proposal"
        - "The proposal shall consist of two (2) volumes"
        - "Submit the following volumes:"

        DOES NOT:
        - Infer volumes from keywords
        - Add default volumes
        - Guess based on evaluation factors
        """
        pass

    def _parse_structural_attachment(
        self,
        attachment: AttachmentInfo
    ) -> List[SkeletonSection]:
        """
        Parse attachments that define proposal structure.

        Identifies attachments like:
        - "Placement Procedures"
        - "Proposal Format Instructions"
        - "Submission Requirements"

        These are treated as AUTHORITATIVE for structure.
        """
        pass

    def _extract_page_limits(self, text: str) -> Dict[str, int]:
        """
        Extract page limits with their targets.

        Returns dict mapping section/volume to page limit.
        Example: {"Volume I": 8, "Technical Approach": 4, "Total": 25}
        """
        pass

    def validate_through_gate(
        self,
        skeleton: ProposalSkeleton
    ) -> SectionLGateResult:
        """
        The Section L Gate - validates skeleton before proceeding.

        Checks:
        1. Volume count matches explicit statements
        2. All required sections are present
        3. Page limits are assigned
        4. Section order matches RFP
        5. No hallucinated volumes/sections

        Returns:
            SectionLGateResult with pass/fail and details
        """
        pass
```

### 2.3 Component B: Requirement Injector (The Flesh)

**Purpose**: Map extracted requirements into the skeleton structure without modifying it.

**Key Principles**:
1. **No Structure Modification**: Cannot add/remove volumes or sections
2. **Semantic Matching**: Use intelligent matching to assign requirements
3. **Respect Section L/M/C Separation**: L requirements go to L slots, etc.
4. **Unmapped Requirements Flag**: Surface any requirements that couldn't be placed

#### Data Structure: `InjectionResult`

```python
# agents/enhanced_compliance/requirement_injector.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

class InjectionConfidence(Enum):
    """Confidence level of requirement-to-section mapping"""
    EXPLICIT = "explicit"        # RFP explicitly states where this goes
    HIGH = "high"                # Clear semantic match
    MEDIUM = "medium"            # Reasonable match, may need review
    LOW = "low"                  # Weak match, definitely needs review

@dataclass
class RequirementMapping:
    """A mapping of a requirement to a skeleton section"""
    requirement_id: str
    requirement_text: str
    requirement_category: str     # "L_COMPLIANCE", "TECHNICAL", "EVALUATION"

    target_section_id: str        # ID of skeleton section
    confidence: InjectionConfidence
    matching_rationale: str       # Why this mapping was made

    # For evaluation requirements
    eval_factor_reference: Optional[str] = None

@dataclass
class SectionWithRequirements:
    """A skeleton section populated with requirements"""
    section_id: str
    section_title: str
    page_limit: Optional[int]

    # Injected requirements by category
    l_requirements: List[RequirementMapping] = field(default_factory=list)
    m_requirements: List[RequirementMapping] = field(default_factory=list)
    c_requirements: List[RequirementMapping] = field(default_factory=list)

    # Derived guidance
    compliance_checkpoints: List[str] = field(default_factory=list)
    win_theme_placeholders: List[str] = field(default_factory=list)
    proof_point_suggestions: List[str] = field(default_factory=list)

@dataclass
class PopulatedVolume:
    """A skeleton volume populated with requirements"""
    volume_id: str
    volume_title: str
    volume_number: int
    page_limit: Optional[int]
    sections: List[SectionWithRequirements] = field(default_factory=list)

    # Evaluation factors that apply to this volume
    evaluation_factors: List[str] = field(default_factory=list)

@dataclass
class InjectionResult:
    """Result of requirement injection into skeleton"""
    success: bool

    # The populated outline
    populated_volumes: List[PopulatedVolume] = field(default_factory=list)

    # Statistics
    total_requirements: int = 0
    mapped_requirements: int = 0
    unmapped_requirements: int = 0

    # Unmapped requirements need manual review
    unmapped: List[Dict] = field(default_factory=list)

    # Low confidence mappings need review
    low_confidence_mappings: List[RequirementMapping] = field(default_factory=list)

    # Warnings
    warnings: List[str] = field(default_factory=list)
```

#### Class: `RequirementInjector`

```python
# agents/enhanced_compliance/requirement_injector.py

class RequirementInjector:
    """
    Component B: Injects requirements into skeleton without modifying structure.

    The skeleton is IMMUTABLE. We only fill the slots.
    """

    def __init__(self, use_semantic_matching: bool = True):
        self.use_semantic = use_semantic_matching

    def inject_requirements(
        self,
        skeleton: ProposalSkeleton,
        l_requirements: List[Dict],
        m_requirements: List[Dict],
        c_requirements: List[Dict]
    ) -> InjectionResult:
        """
        Inject requirements into skeleton sections.

        This method:
        1. For each requirement, finds the best matching section
        2. Uses explicit references first (e.g., "see L.4.B.2")
        3. Falls back to semantic matching
        4. Flags unmapped requirements for review

        DOES NOT:
        - Add new sections
        - Modify page limits
        - Change section order
        - Remove sections with no requirements

        Args:
            skeleton: Validated ProposalSkeleton from Component A
            l_requirements: Section L requirements from CTM
            m_requirements: Section M/Evaluation requirements from CTM
            c_requirements: Section C/Technical requirements from CTM

        Returns:
            InjectionResult with populated outline or warnings
        """
        pass

    def _find_explicit_target(self, requirement: Dict) -> Optional[str]:
        """
        Find section target from explicit references in requirement.

        Patterns:
        - "Volume I" -> maps to Volume I
        - "Factor 2" -> maps to Factor 2 section
        - "L.4.B.2" -> maps to that subsection
        """
        pass

    def _semantic_match(
        self,
        requirement: Dict,
        sections: List[SkeletonSection]
    ) -> Tuple[Optional[str], InjectionConfidence]:
        """
        Use semantic matching to find best section for requirement.

        Considers:
        - Keyword overlap between requirement and section title
        - Evaluation factor alignment
        - Requirement type (technical -> technical sections)
        """
        pass

    def _derive_compliance_checkpoints(
        self,
        section: SectionWithRequirements
    ) -> List[str]:
        """
        Generate compliance checkpoints from injected requirements.
        """
        pass
```

### 2.4 The Section L Gate (Validation Checkpoint)

```python
# agents/enhanced_compliance/section_l_gate.py

class SectionLGate:
    """
    Validation checkpoint between outline building and requirement injection.

    This gate ensures the skeleton matches RFP constraints before proceeding.
    """

    def validate(
        self,
        skeleton: ProposalSkeleton,
        section_l_content: str,
        attachments: Dict[str, AttachmentInfo]
    ) -> SectionLGateResult:
        """
        Validate skeleton against Section L constraints.

        Checks:
        1. Volume count matches RFP statements
        2. All mandatory sections present
        3. Page limits don't exceed RFP limits
        4. No hallucinated volumes/sections
        5. Section order matches RFP
        """
        errors = []
        warnings = []

        # Check 1: Volume count
        stated_count = self._extract_volume_count(section_l_content)
        if stated_count and len(skeleton.volumes) != stated_count:
            errors.append(
                f"RFP specifies {stated_count} volumes but skeleton has "
                f"{len(skeleton.volumes)}. Volumes: {[v.title for v in skeleton.volumes]}"
            )

        # Check 2: Total page limit
        stated_limit = self._extract_total_page_limit(section_l_content)
        skeleton_total = sum(v.page_limit or 0 for v in skeleton.volumes)
        if stated_limit and skeleton_total > stated_limit:
            errors.append(
                f"Total page limit is {stated_limit} but skeleton allocates "
                f"{skeleton_total} pages"
            )

        # Check 3: No hallucinated sections
        # Compare skeleton sections against explicit RFP mentions
        for volume in skeleton.volumes:
            if not self._section_mentioned_in_rfp(volume.title, section_l_content, attachments):
                errors.append(
                    f"Volume '{volume.title}' not found in Section L or attachments. "
                    f"May be hallucinated."
                )

        # More checks...

        passed = len(errors) == 0

        return SectionLGateResult(
            passed=passed,
            skeleton=skeleton if passed else None,
            errors=errors,
            warnings=warnings,
            volume_count_valid=(stated_count is None or len(skeleton.volumes) == stated_count),
            page_limits_valid=(stated_limit is None or skeleton_total <= stated_limit),
            section_order_valid=True,  # Implement order checking
            required_sections_present=True  # Implement presence checking
        )
```

---

## 3. Implementation Roadmap

### 3.1 Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `agents/enhanced_compliance/outline_builder.py` | Component A - Skeleton builder | P0 |
| `agents/enhanced_compliance/requirement_injector.py` | Component B - Requirement mapper | P0 |
| `agents/enhanced_compliance/section_l_gate.py` | Validation checkpoint | P0 |
| `agents/enhanced_compliance/skeleton_models.py` | Data structures for skeleton | P0 |
| `tests/test_outline_builder.py` | Unit tests for Component A | P1 |
| `tests/test_requirement_injector.py` | Unit tests for Component B | P1 |
| `tests/test_section_l_gate.py` | Unit tests for validation | P1 |

### 3.2 Files to Modify

| File | Modification | Priority |
|------|--------------|----------|
| `agents/enhanced_compliance/smart_outline_generator.py` | Deprecate, wrap new components | P0 |
| `agents/enhanced_compliance/document_structure.py` | Add structural attachment parsing | P1 |
| `agents/enhanced_compliance/annotated_outline_exporter.py` | Accept new data structures | P1 |
| `api/main.py` | Update outline endpoints | P1 |
| `agents/enhanced_compliance/__init__.py` | Export new components | P1 |

### 3.3 Interface Contract

```python
# agents/enhanced_compliance/outline_orchestrator.py

class OutlineOrchestrator:
    """
    Orchestrates Component A and Component B with the Section L Gate.

    This is the new entry point for outline generation.
    """

    def __init__(self):
        self.structure_parser = RFPStructureParser()
        self.outline_builder = OutlineBuilder()
        self.requirement_injector = RequirementInjector()
        self.section_l_gate = SectionLGate()

    def generate_compliant_outline(
        self,
        documents: List[Dict],           # Parsed RFP documents
        ctm_requirements: List[Dict],     # Extracted requirements from CTM
        strict_mode: bool = True          # Fail if structure unclear
    ) -> InjectionResult:
        """
        Complete outline generation pipeline.

        Steps:
        1. Parse document structure (existing RFPStructureParser)
        2. Build skeleton from Section L (Component A)
        3. Validate through Section L Gate
        4. Inject requirements (Component B)
        5. Return populated outline

        Args:
            documents: List of parsed RFP documents
            ctm_requirements: Requirements from compliance matrix
            strict_mode: If True, fail on any gate errors. If False, warn.

        Returns:
            InjectionResult with compliant, populated outline
        """
        # Step 1: Parse structure
        structure = self.structure_parser.parse_structure(documents)

        # Step 2: Build skeleton
        section_l = structure.sections.get(UCFSection.SECTION_L)
        if not section_l:
            raise ValueError("Section L not found in RFP documents")

        gate_result = self.outline_builder.build_skeleton(
            section_l_content=section_l.content,
            section_l_subsections=section_l.subsections,
            attachments=structure.attachments
        )

        # Step 3: Validate through gate
        if not gate_result.passed:
            if strict_mode:
                raise ValueError(
                    f"Section L Gate failed: {gate_result.errors}"
                )
            # Non-strict: continue with warnings

        skeleton = gate_result.skeleton

        # Step 4: Separate requirements by category
        l_reqs = [r for r in ctm_requirements if r.get("category") == "L_COMPLIANCE"]
        m_reqs = [r for r in ctm_requirements if r.get("category") == "EVALUATION"]
        c_reqs = [r for r in ctm_requirements if r.get("category") == "TECHNICAL"]

        # Step 5: Inject requirements
        result = self.requirement_injector.inject_requirements(
            skeleton=skeleton,
            l_requirements=l_reqs,
            m_requirements=m_reqs,
            c_requirements=c_reqs
        )

        return result
```

### 3.4 API Endpoint Changes

```python
# api/main.py - Updated endpoint

@app.post("/api/rfp/{rfp_id}/outline/v3")
async def generate_compliant_outline(rfp_id: str, strict_mode: bool = True):
    """
    Generate proposal outline using the new Component A/B architecture.

    This endpoint:
    1. Uses the validated skeleton from Section L
    2. Injects requirements from the compliance matrix
    3. Validates through the Section L Gate
    4. Returns a compliant, annotated outline

    Args:
        rfp_id: RFP project ID
        strict_mode: If True, fail on gate errors. If False, return with warnings.
    """
    from agents.enhanced_compliance.outline_orchestrator import OutlineOrchestrator

    # ... implementation
```

### 3.5 Migration Strategy

1. **Phase 1: Parallel Operation**
   - New endpoints (`/outline/v3`) use new architecture
   - Old endpoints continue working with `SmartOutlineGenerator`
   - Compare outputs for validation

2. **Phase 2: Feature Flag**
   - Add `use_new_outline_builder` feature flag
   - Default to new architecture for new projects
   - Allow fallback for edge cases

3. **Phase 3: Deprecation**
   - Mark `SmartOutlineGenerator` as deprecated
   - Log warnings when old generator is used
   - Document migration path

4. **Phase 4: Removal**
   - Remove old generator after validation period
   - Update all endpoints to use new architecture

---

## 4. Validation Test Cases

### 4.1 Test: No Past Performance Hallucination

```python
def test_no_past_performance_hallucination():
    """
    Given: An RFP that does NOT require Past Performance volume
    When: Generating outline
    Then: No Past Performance volume should appear
    """
    rfp_text = """
    SECTION L - INSTRUCTIONS

    The proposal shall consist of TWO (2) volumes:
    1. Volume I: Technical Proposal (8 pages maximum)
    2. Volume II: Price Proposal (no page limit)
    """

    builder = OutlineBuilder()
    result = builder.build_skeleton(rfp_text, {}, {})

    assert result.passed
    assert len(result.skeleton.volumes) == 2
    assert not any("past performance" in v.title.lower() for v in result.skeleton.volumes)
```

### 4.2 Test: Section L Gate Blocks Invalid Structure

```python
def test_gate_blocks_excess_volumes():
    """
    Given: RFP says 2 volumes, skeleton has 4
    When: Validating through Section L Gate
    Then: Gate should fail with clear error
    """
    rfp_text = "The proposal shall consist of two (2) volumes."

    # Manually create invalid skeleton
    skeleton = ProposalSkeleton(
        rfp_number="TEST-001",
        rfp_title="Test RFP",
        volumes=[
            SkeletonVolume(id="1", title="Technical", volume_number=1),
            SkeletonVolume(id="2", title="Price", volume_number=2),
            SkeletonVolume(id="3", title="Past Performance", volume_number=3),  # Invalid
            SkeletonVolume(id="4", title="Management", volume_number=4),  # Invalid
        ]
    )

    gate = SectionLGate()
    result = gate.validate(skeleton, rfp_text, {})

    assert not result.passed
    assert "specifies 2 volumes but skeleton has 4" in str(result.errors)
```

### 4.3 Test: Attachment Structure Takes Precedence

```python
def test_attachment_structure_authority():
    """
    Given: Attachment 2 specifies detailed section structure
    When: Building skeleton
    Then: Skeleton should use attachment structure
    """
    attachment = AttachmentInfo(
        id="Attachment 2",
        title="Placement Procedures",
        content="""
        PROPOSAL FORMAT:
        1. Executive Summary (2 pages)
        2. Technical Approach (4 pages)
        3. Staffing (2 pages)
        Total: 8 pages
        """,
        document_type="Structure"
    )

    builder = OutlineBuilder()
    result = builder.build_skeleton(
        section_l_content="See Attachment 2 for format.",
        section_l_subsections={},
        attachments={"Attachment 2": attachment}
    )

    assert result.passed
    assert len(result.skeleton.volumes[0].sections) == 3
    assert result.skeleton.volumes[0].sections[0].title == "Executive Summary"
    assert result.skeleton.volumes[0].sections[0].page_limit == 2
```

---

## 5. Reference: AS_BUILT Document Sections

This refactoring addresses issues documented in the architecture:

| AS_BUILT Section | Issue | Addressed By |
|-----------------|-------|--------------|
| v2.9 Known Issues: "43% requirements mapping to UNK section" | Poor section detection | Component A strict parsing |
| v2.9 Known Issues: "Subsection detection not complete" | Missing L.4.B.2 patterns | Enhanced Section L parsing |
| Architecture Overview: "Document Structure Parser details" | Parser doesn't drive structure | Component A uses parser output |
| Best Practices: "Anchor CTM to RFP Structure" | Structure from keywords | Component A from Section L only |

---

## 6. Appendix: Key Code Locations for Reference

| Concept | Current Location | New Location |
|---------|-----------------|--------------|
| Volume extraction | `smart_outline_generator.py:276-310` | `outline_builder.py:_parse_explicit_volumes()` |
| Default volumes | `smart_outline_generator.py:469-484` | REMOVED - no defaults |
| Section L parsing | `document_structure.py:113-174` | Enhanced in `outline_builder.py` |
| Requirement categorization | `api/main.py:1345-1351` | `requirement_injector.py:inject_requirements()` |
| Outline validation | (doesn't exist) | `section_l_gate.py:validate()` |

---

## 7. Approval and Next Steps

**Approval Required From:**
- [ ] Mike (Principal/Owner) - Architecture approval
- [ ] Technical Lead - Implementation approach

**Next Steps After Approval:**
1. Create feature branch `feature/outline-refactor-v3`
2. Implement skeleton models (2 hours)
3. Implement OutlineBuilder (4 hours)
4. Implement SectionLGate (2 hours)
5. Implement RequirementInjector (4 hours)
6. Update API endpoints (2 hours)
7. Write comprehensive tests (4 hours)
8. Integration testing with test RFPs (2 hours)

**Estimated Total Effort:** 20 hours of implementation

---

*End of Refactoring Plan*
