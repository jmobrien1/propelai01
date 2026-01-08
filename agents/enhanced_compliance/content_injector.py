"""
PropelAI v3.0: ContentInjector - Component B

Injects Section C requirements into proposal skeleton.

This component:
- Takes skeleton from StrictStructureBuilder (Component A)
- Takes requirements from compliance matrix
- Maps requirements to skeleton sections using semantic matching
- Does NOT modify skeleton structure
- Produces annotated_outline for ProposalState

Key Principle: The skeleton is IMMUTABLE. We only fill the slots.

v5.0.7: Added Iron Triangle validation to enforce Section C → Technical Volume only.
v5.0.8: Imported VOLUME_SECTION_RULES for hard volume filtering (Iron Triangle Determinism).
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

# v5.0.8: Import Iron Triangle rules from validation engine
from .validation_engine import VOLUME_SECTION_RULES


class MappingConfidence(Enum):
    """
    Confidence level of requirement-to-section mapping.

    Used to flag mappings that need human review.
    """
    EXPLICIT = "explicit"      # RFP explicitly states location (e.g., "in Volume I")
    HIGH = "high"              # Clear keyword match between requirement and section
    MEDIUM = "medium"          # Reasonable inference based on content type
    LOW = "low"                # Weak match, definitely needs review
    UNMAPPED = "unmapped"      # Could not map to any section


@dataclass
class RequirementMapping:
    """
    A mapping of a requirement to a skeleton section.

    This captures the relationship between a requirement and its
    target section, along with the confidence level and rationale.
    """
    requirement_id: str
    requirement_text: str
    target_section_id: str
    target_volume_id: str
    confidence: MappingConfidence
    rationale: str


@dataclass
class InjectionResult:
    """
    Result of content injection into skeleton.

    This is the output of ContentInjector and contains the final
    annotated outline ready for storage in ProposalState.
    """
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

    The skeleton is IMMUTABLE. We only fill the requirement slots.

    Usage:
        injector = ContentInjector()
        result = injector.inject(
            skeleton_dict=state['proposal_skeleton'],
            requirements=state['requirements'],
            evaluation_criteria=state['evaluation_criteria']
        )
        state['annotated_outline'] = result.annotated_outline

    The injector uses a multi-stage matching approach:
    1. Explicit reference matching (e.g., "Volume I", "Section 2.0")
    2. Semantic keyword matching
    3. Requirement type matching
    """

    def __init__(self):
        """Initialize the content injector with keyword mappings."""
        # Keyword mappings for semantic matching
        # Maps section types to keywords that indicate requirements belong there
        self.section_keywords = {
            'technical': [
                'technical', 'approach', 'methodology', 'solution',
                'system', 'design', 'architecture', 'implementation'
            ],
            'management': [
                'management', 'project', 'program', 'oversight',
                'governance', 'control', 'monitor', 'report'
            ],
            'staffing': [
                'staff', 'personnel', 'team', 'key personnel', 'resume',
                'qualification', 'experience', 'skill', 'labor'
            ],
            'experience': [
                'experience', 'past performance', 'reference', 'similar',
                'prior', 'previous', 'contract', 'history'
            ],
            'cost': [
                'cost', 'price', 'budget', 'pricing', 'rate',
                'labor rate', 'estimate', 'financial'
            ],
            'quality': [
                'quality', 'assurance', 'control', 'qa', 'qc',
                'inspection', 'test', 'verification'
            ],
            'transition': [
                'transition', 'phase-in', 'mobilization', 'startup',
                'handover', 'knowledge transfer'
            ],
            'risk': [
                'risk', 'mitigation', 'contingency', 'fallback',
                'backup', 'alternative'
            ],
            'security': [
                'security', 'clearance', 'classified', 'cyber',
                'protection', 'safeguard', 'access'
            ],
        }

        # v6.0.2: NEGATIVE MATCH FILTER - Technical operational keywords
        # These indicate operational/infrastructure requirements that should NOT
        # leak into Management or Personnel sections
        self.technical_operational_keywords = [
            'fault', 'isolation', 'hardware', 'noc', 'inc', '24x7', '24/7',
            'monitoring', 'infrastructure', 'server', 'network', 'router',
            'switch', 'firewall', 'database', 'cpu', 'memory', 'disk',
            'incident', 'outage', 'troubleshoot', 'diagnostic', 'alert',
            'ticketing', 'sla', 'mttr', 'mtbf', 'uptime', 'availability'
        ]

        # v6.0.2: Section types that should NOT receive technical operational content
        self.non_technical_section_indicators = [
            'management', 'personnel', 'staffing', 'organization',
            'key personnel', 'resume', 'qualification', 'experience'
        ]

        # v6.0.10: Document Hierarchy Priority Map
        # Higher priority documents override lower priority for personnel definitions
        self.document_priority = {
            'main_solicitation': 100,   # RFP Section 7, Section L - DEFINITIVE
            'rfp': 100,
            'instructions_evaluation': 90,  # Section L/M
            'evaluation': 85,
            'attachment': 50,           # Attachments - SUPPORTING
            'sow': 40,                  # Statement of Work
            'soo': 30,                  # Statement of Objectives - ANTICIPATED
            'statement_of_objectives': 30,
            'unknown': 10,
        }

        # v6.0.10: Keywords indicating "anticipated" vs "required" personnel
        self.anticipated_keywords = [
            'anticipated', 'expected', 'may include', 'suggested',
            'potential', 'example', 'such as', 'e.g.', 'for example'
        ]

        self.required_keywords = [
            'shall', 'must', 'required', 'mandatory', 'minimum',
            'at a minimum', 'no less than', 'at least'
        ]

        # v6.0.10: Personnel-related keywords for deduplication
        self.personnel_keywords = [
            'key personnel', 'personnel', 'position', 'staff', 'staffing',
            'resume', 'program manager', 'project manager', 'task lead',
            'technical lead', 'subject matter expert', 'sme', 'labor category',
            'labor cat', 'fte', 'full-time'
        ]

        # v5.0.7: Iron Triangle validation rules
        # Section C (PWS/SOW) requirements MUST go to Technical volumes
        # Section M (Evaluation) criteria map to appropriate volumes
        # Administrative forms go to Contract Documentation volumes

        # Volume types for Iron Triangle validation
        self.technical_volume_indicators = [
            'technical', 'approach', 'solution', 'methodology',
            'management', 'staffing', 'transition', 'quality'
        ]
        self.cost_volume_indicators = [
            'cost', 'price', 'pricing', 'budget', 'financial'
        ]
        self.admin_volume_indicators = [
            'contract documentation', 'administrative', 'representations',
            'certifications', 'forms', 'attachments', 'sf1449', 'dd254'
        ]

        # Section C/PWS requirement indicators (should go to Technical)
        self.section_c_indicators = [
            'shall', 'must', 'contractor', 'offeror shall', 'vendor',
            'provide', 'deliver', 'perform', 'support', 'maintain',
            'infrastructure', 'staffing', 'approach', 'methodology'
        ]

        # Administrative form indicators (should go to Admin volumes)
        self.admin_form_indicators = [
            'sf1449', 'sf-1449', 'dd254', 'dd-254', 'dd form',
            'certification', 'representation', 'disclosure',
            'conflict of interest', 'organizational conflict'
        ]

    def inject(
        self,
        skeleton_dict: Dict[str, Any],
        requirements: List[Dict],
        evaluation_criteria: List[Dict],
        is_soo_source: bool = False
    ) -> InjectionResult:
        """
        Inject requirements into skeleton sections.

        This method:
        1. For each requirement, finds the best matching section
        2. Uses explicit references first (e.g., "see Volume I")
        3. Falls back to semantic matching
        4. Flags unmapped requirements for review

        DOES NOT:
        - Add new sections
        - Modify page limits
        - Change section order
        - Remove sections with no requirements

        Args:
            skeleton_dict: Skeleton from state['proposal_skeleton']
            requirements: Requirements from state['requirements'] (Section C)
            evaluation_criteria: Criteria from state['evaluation_criteria'] (Section M)
            is_soo_source: v6.0.8 - True if requirements come from SOO (adds PWS guidance)

        Returns:
            InjectionResult with annotated_outline for state storage
        """
        mappings: List[RequirementMapping] = []
        unmapped: List[Dict] = []
        warnings: List[str] = []

        # Validate skeleton
        if not skeleton_dict.get('volumes'):
            warnings.append("Skeleton has no volumes - cannot inject requirements")
            return InjectionResult(
                success=False,
                annotated_outline={},
                total_requirements=len(requirements),
                mapped_count=0,
                unmapped_requirements=requirements,
                low_confidence_mappings=[],
                warnings=warnings
            )

        # Build section lookup for efficient matching
        section_lookup = self._build_section_lookup(skeleton_dict)

        # v6.0.10: Filter personnel requirements by document hierarchy
        # RFP definitions override SOO "anticipated" definitions
        filtered_requirements, priority_warnings = self._filter_personnel_by_priority(requirements)
        warnings.extend(priority_warnings)

        # Map each requirement to a section
        for req in filtered_requirements:
            mapping = self._map_requirement(req, skeleton_dict, section_lookup)
            if mapping.confidence == MappingConfidence.UNMAPPED:
                unmapped.append(req)
            else:
                mappings.append(mapping)

        # Build annotated outline
        annotated_outline = self._build_annotated_outline(
            skeleton_dict,
            mappings,
            evaluation_criteria,
            is_soo_source=is_soo_source
        )

        # Identify low confidence mappings for review
        low_confidence = [
            m for m in mappings
            if m.confidence == MappingConfidence.LOW
        ]

        if low_confidence:
            warnings.append(
                f"{len(low_confidence)} requirements have low confidence mappings "
                f"and should be reviewed"
            )

        if unmapped:
            warnings.append(
                f"{len(unmapped)} requirements could not be mapped to any section"
            )

        return InjectionResult(
            success=True,
            annotated_outline=annotated_outline,
            total_requirements=len(filtered_requirements),  # v6.0.10: Use filtered count
            mapped_count=len(mappings),
            unmapped_requirements=unmapped,
            low_confidence_mappings=low_confidence,
            warnings=warnings
        )

    def _build_section_lookup(self, skeleton: Dict) -> Dict[str, Dict]:
        """
        Build lookup table: section_id -> section info.

        This enables O(1) lookup when mapping requirements.
        """
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
        """
        Map a single requirement to a skeleton section.

        Uses a multi-stage matching approach:
        1. Explicit reference (highest confidence)
        2. Semantic keyword matching
        3. Requirement type matching
        """
        req_id = requirement.get('id', 'unknown')
        req_text = requirement.get('text', '') or requirement.get('full_text', '')

        # Stage 1: Try explicit reference first (e.g., "Volume I", "Section 2.0")
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

        # Stage 2: Try semantic matching
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
        """
        Find explicit volume/section reference in requirement.

        Looks for patterns like:
        - "Volume I" or "Volume 1"
        - "Section 2.0" or "Section A"
        - "Technical Proposal"
        """
        text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()

        # Pattern 1: "Volume I", "Volume 1", etc.
        vol_match = re.search(r'volume\s*([ivx\d]+)', text, re.IGNORECASE)
        if vol_match:
            vol_ref = vol_match.group(1).upper()
            for vol in skeleton.get('volumes', []):
                vol_id_upper = vol['id'].upper()
                vol_title_upper = vol['title'].upper()

                if vol_ref in vol_id_upper or vol_ref in vol_title_upper:
                    # Return first section of matched volume
                    if vol.get('sections'):
                        return {
                            'section_id': vol['sections'][0]['id'],
                            'volume_id': vol['id'],
                            'reference': f"Volume {vol_ref}"
                        }
                    # If no sections, use volume itself as target
                    return {
                        'section_id': vol['id'],
                        'volume_id': vol['id'],
                        'reference': f"Volume {vol_ref}"
                    }

        # Pattern 2: "Section X.X" references
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

        # Pattern 3: Volume name reference (e.g., "Technical Proposal")
        for vol in skeleton.get('volumes', []):
            vol_title_lower = vol['title'].lower()
            # Check if volume title appears in requirement text
            if vol_title_lower in text and len(vol_title_lower) > 5:
                if vol.get('sections'):
                    return {
                        'section_id': vol['sections'][0]['id'],
                        'volume_id': vol['id'],
                        'reference': f"Volume '{vol['title']}'"
                    }

        return None

    def _classify_volume_type(self, vol_title: str) -> str:
        """
        v5.0.7: Classify volume as 'technical', 'management', 'past_performance',
                'cost', or 'administrative'.

        v5.0.8: Enhanced classification to match VOLUME_SECTION_RULES keys.

        Used for Iron Triangle validation.
        """
        title_lower = vol_title.lower()

        # Check admin first (most restrictive)
        for indicator in self.admin_volume_indicators:
            if indicator in title_lower:
                return 'administrative'

        # Check past performance
        if any(kw in title_lower for kw in ['past performance', 'experience', 'reference']):
            return 'past_performance'

        # Check cost
        for indicator in self.cost_volume_indicators:
            if indicator in title_lower:
                return 'cost'

        # Check management
        if any(kw in title_lower for kw in ['management', 'staffing']):
            return 'management'

        # Default to technical (most common)
        for indicator in self.technical_volume_indicators:
            if indicator in title_lower:
                return 'technical'

        return 'technical'  # Default

    def _get_requirement_source_section(self, requirement: Dict) -> str:
        """
        v5.0.8: Extract the source section letter (C, L, M, etc.) from requirement.

        Used for Hard Volume Filtering.
        """
        # Check explicit section field
        section = requirement.get('section', '').upper()
        if section in ['C', 'L', 'M', 'K', 'B', 'SOW', 'PWS']:
            return section

        # Check category
        category = requirement.get('category', '').lower()
        if category in ['c_requirement', 'technical', 'performance', 'sow', 'pws']:
            return 'C'
        if category in ['l_instruction', 'instruction', 'format']:
            return 'L'
        if category in ['m_evaluation', 'evaluation', 'criteria']:
            return 'M'

        # Check for Section C/PWS/SOW indicators in text
        text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()
        if any(ind in text for ind in self.section_c_indicators):
            return 'C'

        return 'OTHER'

    def _hard_volume_filter(
        self,
        source_section: str,
        volume_type: str
    ) -> Tuple[bool, str]:
        """
        v5.0.8: Hard Volume Filter using VOLUME_SECTION_RULES.

        This is a BLOCKING filter - if it returns False, the requirement
        CANNOT be placed in the volume, regardless of semantic matching score.

        Args:
            source_section: Source section letter (C, L, M, etc.)
            volume_type: Volume type from _classify_volume_type()

        Returns:
            (is_allowed, reason) tuple
        """
        allowed_sections = VOLUME_SECTION_RULES.get(volume_type, [])

        # If no rules defined for this volume type, allow anything
        if not allowed_sections:
            return True, ""

        # Map SOW/PWS to C for rule matching
        normalized_section = source_section
        if source_section in ['SOW', 'PWS']:
            normalized_section = 'C'

        # Check if source section is allowed in this volume
        if normalized_section in allowed_sections or source_section in allowed_sections:
            return True, ""

        return False, (
            f"HARD BLOCK: Section {source_section} content cannot go in {volume_type} volume. "
            f"Allowed sections for {volume_type}: {allowed_sections}"
        )

    def _classify_requirement_type(self, requirement: Dict) -> str:
        """
        v5.0.7: Classify requirement as 'section_c', 'admin_form', or 'other'.

        Used for Iron Triangle validation.
        """
        req_text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()
        req_section = requirement.get('section', '').upper()
        req_category = requirement.get('category', '').lower()

        # Check if it's from Section C/PWS/SOW
        if req_section in ['C', 'PWS', 'SOW'] or req_category in ['technical', 'c_requirement']:
            return 'section_c'

        # Check for admin form indicators
        for indicator in self.admin_form_indicators:
            if indicator in req_text:
                return 'admin_form'

        # Check for Section C content indicators
        section_c_matches = sum(1 for ind in self.section_c_indicators if ind in req_text)
        if section_c_matches >= 2:
            return 'section_c'

        return 'other'

    def _is_valid_iron_triangle_mapping(
        self,
        req_type: str,
        vol_type: str
    ) -> Tuple[bool, str]:
        """
        v5.0.7: Validate mapping against Iron Triangle rules.

        Section C (PWS/SOW) requirements → Technical Volume ONLY
        Admin forms → Admin Volume ONLY
        """
        # Section C requirements MUST go to Technical volumes
        if req_type == 'section_c' and vol_type == 'admin':
            return False, "Section C/PWS requirements cannot go in Administrative volumes"

        # Admin forms should go to Admin volumes (but not a hard block)
        if req_type == 'admin_form' and vol_type == 'technical':
            return True, "Admin form in Technical volume - may need review"

        return True, ""

    def _semantic_match(
        self,
        requirement: Dict,
        skeleton: Dict,
        section_lookup: Dict
    ) -> Tuple[Optional[Dict], MappingConfidence]:
        """
        Use semantic matching to find best section for requirement.

        v5.0.7: Now enforces Iron Triangle validation:
        - Section C/PWS requirements → Technical Volume ONLY
        - Admin forms → Admin Volume (preferred)

        v5.0.8: Added HARD VOLUME FILTER using VOLUME_SECTION_RULES.
        The hard filter runs BEFORE any semantic matching and blocks
        volumes that are not allowed for the requirement's source section.

        Scores each section based on keyword overlap between
        requirement text and section title/type.
        """
        req_text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()
        req_type = requirement.get('requirement_type', '').lower()

        # v5.0.7: Classify requirement for Iron Triangle validation
        iron_req_type = self._classify_requirement_type(requirement)

        # v5.0.8: Get source section for hard volume filter
        source_section = self._get_requirement_source_section(requirement)

        best_match = None
        best_score = 0
        best_vol_type = None

        for vol in skeleton.get('volumes', []):
            vol_title_lower = vol['title'].lower()

            # v5.0.7/v5.0.8: Classify volume type
            vol_type = self._classify_volume_type(vol['title'])

            # v5.0.8: HARD VOLUME FILTER (runs FIRST, before any scoring)
            is_allowed, block_reason = self._hard_volume_filter(source_section, vol_type)
            if not is_allowed:
                print(f"[v5.0.8] {block_reason}")
                continue  # HARD BLOCK - skip this volume entirely

            # v5.0.7: Check Iron Triangle validity (soft filter)
            is_valid, violation_reason = self._is_valid_iron_triangle_mapping(
                iron_req_type, vol_type
            )
            if not is_valid:
                # Skip this volume - Iron Triangle violation
                print(f"[v5.0.7] Iron Triangle block: {violation_reason}")
                continue

            for sec in vol.get('sections', []):
                sec_title_lower = sec['title'].lower()
                score = 0
                match_reasons = []

                # v6.0.1: SECTION-LEVEL HARD FILTER
                # Block Section C requirements from going into admin form sections
                # even if they made it past the volume-level filter
                if source_section in ['C', 'SOW', 'PWS'] or iron_req_type == 'section_c':
                    # Check if this is an admin form section (not allowed for Section C)
                    admin_section_keywords = [
                        'sf1449', 'sf 1449', 'dd254', 'dd 254', 'dd form',
                        'personnel security', 'security questionnaire',
                        'representations and certifications', 'certifications',
                        'evidence of', 'draft dd', 'online representation'
                    ]
                    if any(kw in sec_title_lower for kw in admin_section_keywords):
                        print(f"[v6.0.1] SECTION-LEVEL BLOCK: Section C requirement "
                              f"cannot go into admin section '{sec['title']}'")
                        continue  # Skip this section entirely

                # v6.0.3: SEMANTIC BLOCK FILTER - Strict category enforcement
                # If requirement category is TECHNICAL or PERFORMANCE, it CANNOT
                # go into Security Questionnaire or Representation sections
                req_category = requirement.get('category', '').lower()
                is_technical_or_performance = any(
                    cat in req_category for cat in ['technical', 'performance', 'operational']
                ) or iron_req_type == 'section_c'

                semantic_block_sections = [
                    'security questionnaire', 'questionnaire', 'security clearance',
                    'representation', 'representations', 'certification',
                    'disclosure', 'conflict of interest', 'ocp', 'oci'
                ]

                if is_technical_or_performance:
                    if any(block_kw in sec_title_lower for block_kw in semantic_block_sections):
                        print(f"[v6.0.3] SEMANTIC BLOCK: Technical/Performance requirement "
                              f"(category: '{req_category}') cannot go into '{sec['title']}'")
                        continue  # HARD BLOCK - skip this section

                # v6.0.2: NEGATIVE MATCH FILTER - Content Leakage Prevention
                # Penalize technical operational content going into Management/Personnel sections
                # This prevents requirements about NOC, Fault Isolation, Hardware, etc.
                # from being assigned to Management Approach or Staffing sections
                is_non_technical_section = any(
                    ind in sec_title_lower for ind in self.non_technical_section_indicators
                )
                has_technical_operational_content = any(
                    kw in req_text for kw in self.technical_operational_keywords
                )

                if is_non_technical_section and has_technical_operational_content:
                    # Apply heavy penalty (-50) to prevent content leakage
                    score -= 50
                    match_reasons.append("PENALTY: Technical ops → Management/Personnel")
                    print(f"[v6.0.2] NEGATIVE MATCH: Technical operational content "
                          f"('{[kw for kw in self.technical_operational_keywords if kw in req_text][:3]}') "
                          f"penalty applied for section '{sec['title']}'")

                # v5.0.7: Boost for correct Iron Triangle mapping
                if iron_req_type == 'section_c' and vol_type == 'technical':
                    score += 25
                    match_reasons.append("Iron Triangle: Section C → Technical")
                elif iron_req_type == 'admin_form' and vol_type == 'admin':
                    score += 25
                    match_reasons.append("Iron Triangle: Admin form → Admin volume")

                # Score based on keyword overlap
                for category, keywords in self.section_keywords.items():
                    # Check if section title matches category
                    section_matches_category = any(
                        kw in sec_title_lower for kw in keywords
                    )
                    # Check if requirement mentions same keywords
                    req_matches_category = any(
                        kw in req_text for kw in keywords
                    )

                    if section_matches_category and req_matches_category:
                        score += 10
                        match_reasons.append(f"{category} keywords")

                    # Bonus if requirement type matches category
                    if category in req_type and section_matches_category:
                        score += 15
                        match_reasons.append(f"type '{req_type}' matches {category}")

                # Boost for requirement type matching volume type
                if 'technical' in req_type and 'technical' in vol_title_lower:
                    score += 20
                    match_reasons.append("technical type matches volume")
                if 'management' in req_type and 'management' in vol_title_lower:
                    score += 20
                    match_reasons.append("management type matches volume")
                if 'cost' in req_type or 'price' in req_type:
                    if 'cost' in vol_title_lower or 'price' in vol_title_lower:
                        score += 20
                        match_reasons.append("cost/price type matches volume")

                # Direct section title match
                for word in sec_title_lower.split():
                    if len(word) > 4 and word in req_text:
                        score += 5
                        match_reasons.append(f"title word '{word}'")

                if score > best_score:
                    best_score = score
                    best_vol_type = vol_type
                    best_match = {
                        'section_id': sec['id'],
                        'volume_id': vol['id'],
                        'rationale': f"Keyword match: {', '.join(match_reasons[:3])} (score: {score})"
                    }

        if best_match:
            # Determine confidence based on score
            if best_score >= 25:
                return best_match, MappingConfidence.HIGH
            elif best_score >= 15:
                return best_match, MappingConfidence.MEDIUM
            elif best_score >= 5:
                return best_match, MappingConfidence.LOW

        return None, MappingConfidence.UNMAPPED

    def _build_annotated_outline(
        self,
        skeleton: Dict,
        mappings: List[RequirementMapping],
        evaluation_criteria: List[Dict],
        is_soo_source: bool = False
    ) -> Dict[str, Any]:
        """
        Build the final annotated outline for state storage.

        This combines the skeleton structure with injected requirements
        to produce the complete annotated outline.

        v6.0.8: If is_soo_source=True, adds PWS writing guidance to technical sections.
        """
        # Group mappings by section
        section_reqs: Dict[str, List[RequirementMapping]] = {}
        for mapping in mappings:
            sec_id = mapping.target_section_id
            if sec_id:
                if sec_id not in section_reqs:
                    section_reqs[sec_id] = []
                section_reqs[sec_id].append(mapping)

        # Build output structure
        volumes = []
        for vol in skeleton.get('volumes', []):
            sections = []
            for sec in vol.get('sections', []):
                sec_mappings = section_reqs.get(sec['id'], [])

                # v6.0.8: Generate writing instructions based on source type
                writing_instructions = []
                if is_soo_source and sec_mappings:
                    # Check if this is a technical section (not cost/price)
                    sec_title_lower = sec.get('title', '').lower()
                    is_technical = any(kw in sec_title_lower for kw in [
                        'technical', 'approach', 'solution', 'management',
                        'staffing', 'security', 'quality', 'transition'
                    ])
                    is_cost = any(kw in sec_title_lower for kw in ['cost', 'price', 'pricing'])

                    if is_technical and not is_cost:
                        writing_instructions.append(
                            "**DRAFTING REQUIREMENT (SOO Response):** Author a Performance Work Statement (PWS) "
                            "that fulfills the objectives listed below. Transform each SOO objective into "
                            "specific, measurable tasks with defined deliverables."
                        )

                        # Add section-specific PWS guidance
                        if 'security' in sec_title_lower:
                            writing_instructions.append(
                                "**Security PWS Guidance:** Define specific security controls, clearance requirements, "
                                "incident response procedures, and compliance verification methods."
                            )
                        elif 'management' in sec_title_lower:
                            writing_instructions.append(
                                "**Management PWS Guidance:** Define program governance structure, reporting cadence, "
                                "quality assurance processes, and risk mitigation procedures."
                            )
                        elif 'transition' in sec_title_lower:
                            writing_instructions.append(
                                "**Transition PWS Guidance:** Define phase-in timeline, knowledge transfer activities, "
                                "incumbent coordination requirements, and go-live criteria."
                            )

                section_data = {
                    'id': sec['id'],
                    'title': sec['title'],
                    'page_limit': sec.get('page_limit'),
                    'order': sec.get('order', 0),
                    'source_reference': sec.get('source_reference', ''),
                    # v6.0.8: Writing instructions for proposal authors
                    'writing_instructions': writing_instructions,
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
                    'compliance_checkpoints': self._generate_checkpoints(sec_mappings)
                }
                sections.append(section_data)

            volume_data = {
                'id': vol['id'],
                'title': vol['title'],
                'volume_number': vol.get('volume_number', 0),
                'page_limit': vol.get('page_limit'),
                'source_reference': vol.get('source_reference', ''),
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
                'content_source': 'Section C (ContentInjector)',
                'version': '3.0'
            }
        }

    def _generate_checkpoints(
        self,
        mappings: List[RequirementMapping]
    ) -> List[str]:
        """
        Generate compliance checkpoints from injected requirements.

        Creates a checklist of requirements that must be addressed
        in this section.
        """
        checkpoints = []
        for m in mappings[:10]:  # Limit to first 10 for readability
            text = m.requirement_text
            if len(text) > 80:
                text = text[:77] + "..."
            checkpoints.append(f"[ ] Address: {text}")
        return checkpoints

    # =========================================================================
    # v6.0.10: Document Hierarchy Priority Methods
    # =========================================================================

    def _get_document_priority(self, requirement: Dict) -> int:
        """
        v6.0.10: Get the priority score for a requirement's source document.

        Higher priority documents override lower priority for conflicting definitions.
        RFP Section L/M (100) > Attachments (50) > SOO (30)
        """
        source = requirement.get('source_document', '').lower()
        section = requirement.get('section', '').lower()

        # Check explicit source document
        if source:
            for key, priority in self.document_priority.items():
                if key in source:
                    return priority

        # Check section for priority hints
        if section in ['l', 'm', '7', 'section l', 'section m', 'section 7']:
            return self.document_priority['main_solicitation']
        if 'soo' in section or 'statement of objectives' in section:
            return self.document_priority['soo']
        if 'sow' in section or 'pws' in section:
            return self.document_priority['sow']

        return self.document_priority.get('unknown', 10)

    def _is_personnel_requirement(self, requirement: Dict) -> bool:
        """
        v6.0.10: Check if a requirement is related to key personnel/staffing.

        Personnel requirements need deduplication based on document hierarchy.
        """
        req_text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()
        req_category = requirement.get('category', '').lower()

        # Check category
        if any(cat in req_category for cat in ['personnel', 'staffing', 'labor']):
            return True

        # Check text for personnel keywords
        return any(kw in req_text for kw in self.personnel_keywords)

    def _is_anticipated_requirement(self, requirement: Dict) -> bool:
        """
        v6.0.10: Check if a requirement uses "anticipated" language.

        Anticipated requirements are lower priority than definitive requirements.
        """
        req_text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()

        has_anticipated = any(kw in req_text for kw in self.anticipated_keywords)
        has_required = any(kw in req_text for kw in self.required_keywords)

        # If it has both anticipated AND required language, defer to required
        if has_anticipated and has_required:
            return False

        return has_anticipated

    def _extract_personnel_count(self, requirement: Dict) -> Optional[int]:
        """
        v6.0.10: Extract the number of positions from a personnel requirement.

        Looks for patterns like "3 positions", "five personnel", etc.
        """
        req_text = (requirement.get('text', '') or requirement.get('full_text', '')).lower()

        # Numeric patterns
        patterns = [
            r'(\d+)\s*(?:key\s+)?(?:positions?|personnel|staff|fte)',
            r'(?:at least|minimum|maximum)\s*(\d+)',
            r'(\d+)\s*(?:labor|labour)\s*categor',
        ]

        for pattern in patterns:
            match = re.search(pattern, req_text)
            if match:
                return int(match.group(1))

        # Word-to-number mapping for small numbers
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        for word, num in word_numbers.items():
            if word in req_text and any(kw in req_text for kw in ['position', 'personnel']):
                return num

        return None

    def _filter_personnel_by_priority(
        self,
        requirements: List[Dict]
    ) -> Tuple[List[Dict], List[str]]:
        """
        v6.0.10: Filter personnel requirements using document hierarchy priority.

        When RFP says "3 positions required" and SOO says "anticipated 5 positions",
        the RFP definition takes precedence.

        Returns:
            (filtered_requirements, warnings)
        """
        warnings = []

        # Separate personnel and non-personnel requirements
        personnel_reqs = []
        other_reqs = []

        for req in requirements:
            if self._is_personnel_requirement(req):
                personnel_reqs.append(req)
            else:
                other_reqs.append(req)

        if not personnel_reqs:
            return requirements, []

        # Sort personnel requirements by document priority (highest first)
        personnel_reqs.sort(key=lambda r: self._get_document_priority(r), reverse=True)

        # Find conflicting personnel counts
        filtered_personnel = []
        seen_topics = {}  # Track personnel topics to deduplicate

        for req in personnel_reqs:
            priority = self._get_document_priority(req)
            is_anticipated = self._is_anticipated_requirement(req)
            count = self._extract_personnel_count(req)
            req_text = (req.get('text', '') or req.get('full_text', '')).lower()

            # Create a topic key based on content (simplified)
            topic_key = 'general_personnel'
            if 'key personnel' in req_text:
                topic_key = 'key_personnel'
            elif 'program manager' in req_text:
                topic_key = 'program_manager'
            elif 'project manager' in req_text:
                topic_key = 'project_manager'

            # Check if we've seen a higher-priority definition for this topic
            if topic_key in seen_topics:
                existing_priority, existing_anticipated, existing_count = seen_topics[topic_key]

                # Skip if lower priority and existing is not anticipated
                if priority < existing_priority and not existing_anticipated:
                    source = req.get('source_document', 'unknown')
                    warnings.append(
                        f"[v6.0.10] PERSONNEL OVERRIDE: Skipped '{topic_key}' from {source} "
                        f"(priority {priority}, {'anticipated' if is_anticipated else 'required'}) - "
                        f"higher priority definition exists"
                    )
                    if count and existing_count and count != existing_count:
                        warnings.append(
                            f"[v6.0.10] COUNT CONFLICT: {source} specified {count} positions, "
                            f"but authoritative source specified {existing_count}"
                        )
                    continue

                # Skip anticipated if definitive exists at same/higher priority
                if is_anticipated and not existing_anticipated:
                    source = req.get('source_document', 'unknown')
                    warnings.append(
                        f"[v6.0.10] ANTICIPATED OVERRIDE: Skipped anticipated '{topic_key}' "
                        f"from {source} - definitive requirement exists"
                    )
                    continue

            # Record this as the authoritative source for this topic
            seen_topics[topic_key] = (priority, is_anticipated, count)
            filtered_personnel.append(req)

        if len(filtered_personnel) < len(personnel_reqs):
            removed = len(personnel_reqs) - len(filtered_personnel)
            warnings.insert(0,
                f"[v6.0.10] Document hierarchy filter removed {removed} duplicate/anticipated "
                f"personnel requirements"
            )

        return other_reqs + filtered_personnel, warnings
