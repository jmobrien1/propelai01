"""
PropelAI v2.9: Section-Aware Requirement Extractor

Per best practices:
- "Extract Every Requirement — NEVER Summarize in the CTM"
- "Separate Instructions Compliance (Section L) from Requirements Compliance (Section C/PWS)"
- "Use hierarchical numbering matching the RFP"
- "Tag Requirements Using Unique IDs" - preserve RFP's own IDs

This extractor:
1. Works WITHIN identified document structure
2. Preserves RFP's own section references (L.4.B.2, C.3.1.a)
3. Extracts complete requirement paragraphs, not sentence fragments
4. Produces three distinct requirement sets: L (Instructions), C/PWS (Technical), M (Evaluation)
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .document_structure import (
    DocumentStructure, 
    UCFSection, 
    SectionBoundary,
    SubsectionBoundary,
    AttachmentInfo,
    analyze_rfp_structure
)


class RequirementCategory(Enum):
    """High-level requirement categories per best practices CTM structure"""
    SECTION_L_COMPLIANCE = "L_COMPLIANCE"       # Instructions to follow for submission
    TECHNICAL_REQUIREMENT = "TECHNICAL"          # C/PWS/SOW - what to DO
    EVALUATION_FACTOR = "EVALUATION"             # M - how proposal will be SCORED
    ADMINISTRATIVE = "ADMINISTRATIVE"            # B/F/G/H - contract admin
    ATTACHMENT_REQUIREMENT = "ATTACHMENT"        # Requirements from attachments


class BindingLevel(Enum):
    """How binding is this requirement?"""
    MANDATORY = "Mandatory"          # SHALL, MUST, REQUIRED, WILL
    HIGHLY_DESIRABLE = "Highly Desirable"  # SHOULD
    DESIRABLE = "Desirable"          # MAY, CAN, ENCOURAGED
    INFORMATIONAL = "Informational"  # No binding language


@dataclass
class FormattingConstraint:
    """
    P0 formatting constraints that can cause disqualification if violated.
    These come from Placement Procedures in OASIS+ task orders.
    """
    constraint_type: str  # PAGE_LIMIT, FONT, MARGIN, FILE_FORMAT, VOLUME_STRUCTURE
    description: str
    value: str  # e.g., "12-point", "1 inch", "25 pages"
    applies_to: str  # Volume 1, Volume 2, All, etc.
    consequence: str  # "excess pages will not be read", "disqualification", etc.
    priority: str = "P0"  # P0 = disqualification risk


@dataclass
class EvaluationSubFactor:
    """Evaluation sub-factor with adjectival rating scale"""
    factor_name: str  # "Technical Factor"
    subfactor_name: str  # "Management Approach"
    subfactor_number: int  # 1, 2, etc.
    instruction_text: str  # What to address (Section L equivalent)
    evaluation_text: str  # How it will be evaluated (Section M equivalent)
    rating_scale: str  # "Adjectival", "Pass/Fail", "Price"
    volume: str  # "Volume 1", "Volume 2", etc.


@dataclass
class VolumeStructure:
    """Required proposal volume structure"""
    volume_number: int  # 1, 2, 3
    volume_name: str  # "Executive Summary and Technical Volume"
    required_content: List[str] = field(default_factory=list)
    page_limit: Optional[int] = None
    subfactors: List[str] = field(default_factory=list)


@dataclass
class StructuredRequirement:
    """
    A requirement extracted with full structural context.
    
    This preserves the RFP's own organization rather than imposing our own IDs.
    """
    # RFP's own reference (MUST preserve this)
    rfp_reference: str              # e.g., "L.4.B.2", "C.3.1.a", "PWS 2.3.1"
    
    # Generated ID (only if RFP doesn't provide one)
    generated_id: str               # e.g., "TW-L-001" (use prefix per best practices)
    
    # Full requirement text - VERBATIM, never summarized
    full_text: str
    
    # Category and classification
    category: RequirementCategory
    binding_level: BindingLevel
    binding_keyword: str            # The actual "shall", "must", etc.
    
    # Source location
    source_section: UCFSection
    source_subsection: Optional[str]  # e.g., "L.4.B"
    page_number: int
    source_document: str
    
    # Additional context
    parent_title: str               # Title of the subsection containing this requirement
    evaluation_factor: Optional[str]  # If linked to Section M factor
    
    # Cross-references found in text
    references_to: List[str] = field(default_factory=list)  # Other sections/attachments referenced
    
    # For deduplication
    text_hash: str = ""
    
    def __post_init__(self):
        import hashlib
        if not self.text_hash:
            # Normalize for comparison
            normalized = ' '.join(self.full_text.lower().split())
            self.text_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]


@dataclass
class ExtractionResult:
    """Results of section-aware extraction"""
    # Requirements by category
    section_l_requirements: List[StructuredRequirement] = field(default_factory=list)
    technical_requirements: List[StructuredRequirement] = field(default_factory=list)  # C/PWS/SOW
    evaluation_requirements: List[StructuredRequirement] = field(default_factory=list)  # M
    attachment_requirements: List[StructuredRequirement] = field(default_factory=list)
    administrative_requirements: List[StructuredRequirement] = field(default_factory=list)

    # All requirements (for convenience)
    all_requirements: List[StructuredRequirement] = field(default_factory=list)

    # Extraction stats
    stats: Dict[str, Any] = field(default_factory=dict)

    # Document structure used
    structure: Optional[DocumentStructure] = None

    # Detected RFP type information
    rfp_type: str = "STANDARD"  # NIH, DOD, GSA, VA, HHS, SBIR, OASIS_TASK_ORDER, etc.
    agency: str = ""  # NIH, Navy, Army, Air Force, GSA, VA, DHS, etc.
    rfp_format: str = "UCF"  # UCF, NON_UCF, GSA_BPA, RESEARCH_OUTLINE

    # OASIS+ Task Order specific fields
    formatting_constraints: List[FormattingConstraint] = field(default_factory=list)  # P0 constraints
    volume_structure: List[VolumeStructure] = field(default_factory=list)  # Required volumes
    evaluation_subfactors: List[EvaluationSubFactor] = field(default_factory=list)  # Sub-factors with ratings
    adjectival_ratings: Dict[str, str] = field(default_factory=dict)  # Rating definitions
    placement_procedures_source: str = ""  # Which attachment contains L/M equivalent


# Agency-specific patterns for enhanced extraction
AGENCY_PATTERNS = {
    "NIH": {
        "identifiers": [
            r"NIH", r"National\s*Institutes?\s*of\s*Health",
            r"NIEHS", r"NCI", r"NIAID", r"NICHD", r"NHLBI", r"NIMH",
            r"75N\d{5}[A-Z]\d{5}",
        ],
        "extra_binding_keywords": [
            r'\bPHS\s+\d+', r'\bNIH\s+policy', r'\bFunding\s+Opportunity',
        ],
        "section_patterns": {
            "research_outline": r"(?:RO|Research\s+Outline)[-\s]*([IVX]+\.?\d*)",
            "specific_aims": r"Specific\s+Aims?",
        },
        "special_attachments": ["Attachment 2", "Attachment 11", "PHS 398"],
    },
    "DOD": {
        "identifiers": [
            r"DoD", r"Department\s*of\s*Defense",
            r"Navy|NAVSEA|NAVAIR|SPAWAR",
            r"Army|USACE|AMC",
            r"Air\s*Force|AFLCMC",
            r"N\d{5}[-]?\d{2}[-]?[RQ][-]?\d+",  # Navy format
            r"W\d{5}[-]?\d{2}[-]?[RQ][-]?\d+",  # Army format
            r"FA\d{4}[-]?\d{2}[-]?[RQ][-]?\d+", # Air Force format
        ],
        "extra_binding_keywords": [
            r'\bDFARS\b', r'\bDFAS\b', r'\bDD[-\s]*254\b', r'\bSecurity\s+Classification',
            r'\bCUI\b', r'\bCDRL\b', r'\bDID\b', r'\bSOW\s+paragraph',
        ],
        "section_patterns": {
            "cdrl": r"CDRL\s+([A-Z]\d{3})",
            "did": r"DID[-\s]*(\w+-\d+)",
            "dfars": r"DFARS\s+(\d+\.\d+[-\d]*)",
        },
        "special_attachments": ["J-series", "DD-254", "CDRL", "DID"],
    },
    "GSA": {
        "identifiers": [
            r"GSA", r"General\s*Services\s*Administration",
            r"GS[-]?\d{2}[FQ]", r"Schedule\s+\d+",
            r"BPA", r"Blanket\s+Purchase\s+Agreement",
        ],
        "extra_binding_keywords": [
            r'\bGSAR\b', r'\bFSS\b', r'\bSchedule\s+pricing',
            r'\bSymphony\b', r'\beBuy\b',
        ],
        "section_patterns": {
            "sin": r"SIN\s+([\d-]+)",
            "schedule": r"Schedule\s+(\d+)",
        },
        "special_attachments": ["Price List", "Labor Categories", "Terms and Conditions"],
        "non_ucf_likely": True,
    },
    "OASIS": {
        "identifiers": [
            r"OASIS\+?", r"OASIS\s*Plus", r"47QSMD",
            r"J\.P[-\s]*1", r"Qualifications\s+Matrix",
        ],
        "extra_binding_keywords": [
            r'\bself[-\s]?scor', r'\bJ\.P[-\s]*[123]', r'\bSymphony\s+portal',
            r'\bAAV\b', r'\bAverage\s+Annual\s+Value',
        ],
        "section_patterns": {
            "jp_section": r"J\.P[-\s]*(\d)",
            "domain": r"Domain\s+(\d+)",
        },
        "special_attachments": ["J.P-1", "J.P-2", "J.P-3"],
    },
    "OASIS_TASK_ORDER": {
        "identifiers": [
            r"OASIS\+?", r"OASIS\s*Plus", r"47QSMD",
            r"Placement\s+Procedures", r"Task\s+Order",
            r"Multiple\s+Award\s+IDIQ",
        ],
        "extra_binding_keywords": [
            r'\bVolume\s+[123I]+\b', r'\bTechnical\s+Volume\b', r'\bCost\s+Volume\b',
            r'\bpage\s+limit', r'\bexcess\s+pages', r'\bwill\s+not\s+be\s+read',
            r'\bAdjectival', r'\bExceptional', r'\bUnacceptable',
        ],
        "section_patterns": {
            "volume": r"Volume\s+([123I]+)",
            "subfactor": r"Sub[-\s]?[Ff]actor\s+(\d+)",
            "factor": r"Factor\s+(\d+)",
        },
        "special_attachments": ["Placement Procedures", "Attachment 2"],
        "placement_procedures_is_lm": True,  # Key flag for OASIS+ task orders
    },
    "VA": {
        "identifiers": [
            r"VA\b", r"Veterans\s*(?:Affairs|Administration)",
            r"36C\d+",
        ],
        "extra_binding_keywords": [
            r'\bVAAR\b', r'\bVHA\b', r'\bVISN\b',
        ],
        "section_patterns": {},
        "special_attachments": [],
    },
    "SBIR": {
        "identifiers": [
            r"SBIR", r"STTR", r"Small\s+Business\s+Innovation\s+Research",
            r"Phase\s+[I123]", r"Topic\s+\d+",
        ],
        "extra_binding_keywords": [
            r'\bPhase\s+[I123]\b', r'\bTechnical\s+Volume', r'\bCost\s+Volume',
            r'\bCommercialization\b',
        ],
        "section_patterns": {
            "topic": r"Topic\s+(\d+(?:\.\d+)?)",
            "phase": r"Phase\s+([I123])",
        },
        "special_attachments": ["Technical Volume", "Cost Volume", "Commercialization Plan"],
    },
}


class SectionAwareExtractor:
    """
    Extracts requirements while respecting RFP document structure.
    
    Key principles:
    1. NEVER rename the RFP's own requirement references
    2. Extract COMPLETE requirement text, not fragments
    3. Maintain clear separation of L/M/C requirements
    4. Track cross-references for compliance mapping
    """
    
    # Binding keywords by level
    MANDATORY_KEYWORDS = [
        r'\bshall\b', r'\bmust\b', 
        r'\brequired\s+to\b', r'\bis\s+required\b', r'\bare\s+required\b',
        r'\bshall\s+not\b', r'\bmust\s+not\b',  # Prohibitions
        r'\bwill\s+(?:provide|submit|include|demonstrate|ensure|address|describe)\b',  # More specific will
    ]
    
    SHOULD_KEYWORDS = [
        r'\bshould\b', r'\bshould\s+not\b',
        r'\bis\s+(?:expected|recommended)\b',
    ]
    
    MAY_KEYWORDS = [
        r'\bmay\b', r'\bcan\b', r'\bis\s+encouraged\b', r'\bis\s+permitted\b',
        r'\bis\s+optional\b',
    ]
    
    # Special keywords for evaluation factors (Section M)
    EVALUATION_KEYWORDS = [
        r'\bwill\s+be\s+evaluat',    # "will be evaluated"
        r'\bwill\s+be\s+assessed',
        r'\bwill\s+be\s+rated',
        r'\bwill\s+be\s+scored',
        r'\bwill\s+be\s+considered',
        r'\bevaluation\s+(?:factor|criteria)',
        r'\brating\s+(?:factor|criteria)',
        r'\bbasis\s+for\s+(?:award|evaluation)',
        r'\badjectival\s+rating',
        r'\bstrength|weakness|deficien',
    ]
    
    # Pattern to find section/attachment references in text
    CROSS_REFERENCE_PATTERNS = [
        r'(?:Section|Sec\.?)\s+([A-M])(?:\.[\d\.]+)?',
        r'(?:Attachment|Exhibit|Appendix)\s+(\d+|[A-Z])',
        r'(?:FAR|DFARS)\s+(\d+\.\d+[-\d]*)',
        r'(?:PWS|SOW)\s+(\d+(?:\.\d+)*)',
        r'([A-M]\.\d+(?:\.[A-Za-z\d]+)*)',  # L.4.B.2 style
    ]
    
    # Minimum requirement length (characters) - shorter is likely not a real requirement
    MIN_REQUIREMENT_LENGTH = 40
    
    # Maximum requirement length - if longer, it's probably multiple requirements
    MAX_REQUIREMENT_LENGTH = 2000
    
    def __init__(self, preserve_rfp_ids: bool = True, agency_type: Optional[str] = None):
        """
        Args:
            preserve_rfp_ids: If True, always use RFP's own references as primary ID
            agency_type: Pre-detected agency type (NIH, DOD, GSA, etc.)
        """
        self.preserve_rfp_ids = preserve_rfp_ids
        self.agency_type = agency_type
        self.counters = {}  # For generating IDs when needed
        self.active_patterns = None  # Will be set based on agency detection

    def _detect_agency_type(self, documents: List[Dict[str, Any]], solicitation_number: str = "") -> Tuple[str, str, str]:
        """
        Detect the agency/RFP type from document content and solicitation number.

        Returns:
            Tuple of (rfp_type, agency, rfp_format)
        """
        # Combine all text for detection
        all_text = " ".join(doc.get('text', '')[:50000] for doc in documents)
        all_text_lower = all_text.lower()

        # Check solicitation number first (most reliable)
        sol_upper = solicitation_number.upper() if solicitation_number else ""

        # NIH
        if sol_upper.startswith("75N") or re.search(r"75N\d{5}[A-Z]\d{5}", sol_upper):
            return ("NIH", "NIH", "UCF")

        # DoD - Navy
        if sol_upper.startswith("N") and re.match(r"N\d{5}", sol_upper):
            return ("DOD", "Navy", "UCF")

        # DoD - Army
        if sol_upper.startswith("W"):
            return ("DOD", "Army", "UCF")

        # DoD - Air Force
        if sol_upper.startswith("FA"):
            return ("DOD", "Air Force", "UCF")

        # GSA
        if sol_upper.startswith("GS") or sol_upper.startswith("47Q"):
            if "OASIS" in all_text.upper() or "J.P-1" in all_text:
                return ("OASIS_TASK_ORDER", "GSA OASIS+", "NON_UCF")
            return ("GSA", "GSA", "NON_UCF")

        # VA
        if sol_upper.startswith("36C"):
            return ("VA", "VA", "UCF")

        # DHS
        if sol_upper.startswith("70") or sol_upper.startswith("HSC"):
            return ("DHS", "DHS", "UCF")

        # Now check content patterns
        best_match = None
        best_score = 0

        for agency_name, patterns in AGENCY_PATTERNS.items():
            score = 0
            for identifier in patterns.get("identifiers", []):
                if re.search(identifier, all_text, re.IGNORECASE):
                    score += 1

            if score > best_score:
                best_score = score
                best_match = agency_name

        if best_match and best_score >= 2:
            rfp_format = "NON_UCF" if AGENCY_PATTERNS.get(best_match, {}).get("non_ucf_likely") else "UCF"
            return (best_match, best_match, rfp_format)

        # Check for SBIR/STTR
        if re.search(r'\bSBIR\b|\bSTTR\b', all_text, re.IGNORECASE):
            return ("SBIR", "Multiple", "SBIR")

        # Default
        return ("STANDARD", "", "UCF")

    def _get_agency_patterns(self, agency_type: str) -> Dict[str, Any]:
        """Get agency-specific patterns for enhanced extraction."""
        return AGENCY_PATTERNS.get(agency_type, {})

    def extract(self, documents: List[Dict[str, Any]], structure: Optional[DocumentStructure] = None,
                solicitation_number: str = "") -> ExtractionResult:
        """
        Extract requirements from documents using structural analysis.

        Args:
            documents: List of parsed documents with 'text', 'filename', 'pages'
            structure: Pre-computed document structure (will compute if not provided)
            solicitation_number: Solicitation number for agency detection

        Returns:
            ExtractionResult with categorized requirements
        """
        # First, analyze document structure
        if structure is None:
            structure = analyze_rfp_structure(documents)

        # Detect agency type if not provided
        if self.agency_type:
            rfp_type, agency, rfp_format = self.agency_type, self.agency_type, "UCF"
        else:
            sol_num = solicitation_number or (structure.solicitation_number if structure else "")
            rfp_type, agency, rfp_format = self._detect_agency_type(documents, sol_num)

        # Get agency-specific patterns
        self.active_patterns = self._get_agency_patterns(rfp_type)

        result = ExtractionResult(
            structure=structure,
            rfp_type=rfp_type,
            agency=agency,
            rfp_format=rfp_format
        )
        seen_hashes: Set[str] = set()  # For deduplication

        # Reset counters
        self.counters = {cat: 0 for cat in RequirementCategory}
        
        # Extract from each identified section
        for section, boundary in structure.sections.items():
            category = self._section_to_category(section)
            requirements = self._extract_from_section(boundary, category)
            
            # Deduplicate and categorize
            for req in requirements:
                if req.text_hash not in seen_hashes:
                    seen_hashes.add(req.text_hash)
                    result.all_requirements.append(req)
                    
                    # Add to appropriate list
                    if category == RequirementCategory.SECTION_L_COMPLIANCE:
                        result.section_l_requirements.append(req)
                    elif category == RequirementCategory.TECHNICAL_REQUIREMENT:
                        result.technical_requirements.append(req)
                    elif category == RequirementCategory.EVALUATION_FACTOR:
                        result.evaluation_requirements.append(req)
                    elif category == RequirementCategory.ADMINISTRATIVE:
                        result.administrative_requirements.append(req)
        
        # Extract from attachments that contain requirements
        for att_id, att_info in structure.attachments.items():
            if att_info.contains_requirements and att_info.document_type not in ['Amendment', 'Budget Template']:
                requirements = self._extract_from_attachment(att_info)

                for req in requirements:
                    if req.text_hash not in seen_hashes:
                        seen_hashes.add(req.text_hash)
                        result.all_requirements.append(req)
                        result.attachment_requirements.append(req)

                        # Also add to technical if it's SOW/PWS or Technical Attachment
                        if att_info.document_type in ['SOW', 'PWS', 'Technical Attachment']:
                            result.technical_requirements.append(req)

        # Check if this is an OASIS+ task order and process Placement Procedures
        is_oasis_task_order = (
            rfp_type == "OASIS_TASK_ORDER" or
            any('oasis' in doc.get('text', '').lower()[:5000] for doc in documents) or
            any('placement procedure' in doc.get('text', '').lower()[:5000] for doc in documents) or
            any('placement' in (doc.get('filename', '') or '').lower() and
                'procedure' in (doc.get('filename', '') or '').lower() for doc in documents)
        )

        if is_oasis_task_order:
            result.rfp_type = "OASIS_TASK_ORDER"
            self._process_oasis_task_order(documents, structure, result)

        # Build stats
        result.stats = self._build_stats(result)

        return result
    
    def _section_to_category(self, section: UCFSection) -> RequirementCategory:
        """Map UCF section to requirement category"""
        mapping = {
            UCFSection.SECTION_L: RequirementCategory.SECTION_L_COMPLIANCE,
            UCFSection.SECTION_M: RequirementCategory.EVALUATION_FACTOR,
            UCFSection.SECTION_C: RequirementCategory.TECHNICAL_REQUIREMENT,
            UCFSection.SECTION_B: RequirementCategory.ADMINISTRATIVE,
            UCFSection.SECTION_F: RequirementCategory.ADMINISTRATIVE,
            UCFSection.SECTION_G: RequirementCategory.ADMINISTRATIVE,
            UCFSection.SECTION_H: RequirementCategory.ADMINISTRATIVE,
            UCFSection.SECTION_I: RequirementCategory.ADMINISTRATIVE,
            UCFSection.SECTION_J: RequirementCategory.ATTACHMENT_REQUIREMENT,
            UCFSection.SECTION_K: RequirementCategory.ADMINISTRATIVE,
        }
        return mapping.get(section, RequirementCategory.ADMINISTRATIVE)
    
    def _extract_from_section(self, boundary: SectionBoundary, category: RequirementCategory) -> List[StructuredRequirement]:
        """Extract requirements from a section boundary"""
        requirements = []
        
        # If we have identified subsections, extract from each
        if boundary.subsections:
            for ref, subsection in boundary.subsections.items():
                reqs = self._extract_from_text(
                    text=subsection.content,
                    parent_section=boundary.section,
                    subsection_ref=ref,
                    subsection_title=subsection.title,
                    page_number=subsection.page_number,
                    source_document="",  # Will be filled
                    category=category
                )
                requirements.extend(reqs)
        else:
            # Extract from full section content
            reqs = self._extract_from_text(
                text=boundary.content,
                parent_section=boundary.section,
                subsection_ref=None,
                subsection_title=boundary.title,
                page_number=boundary.start_page,
                source_document="",
                category=category
            )
            requirements.extend(reqs)
        
        return requirements
    
    def _extract_from_attachment(self, att_info: AttachmentInfo) -> List[StructuredRequirement]:
        """Extract requirements from an attachment"""
        # Determine category based on attachment type
        if att_info.document_type in ['SOW', 'PWS']:
            category = RequirementCategory.TECHNICAL_REQUIREMENT
        else:
            category = RequirementCategory.ATTACHMENT_REQUIREMENT
        
        return self._extract_from_text(
            text=att_info.content,
            parent_section=UCFSection.SECTION_J,  # Attachments are technically Section J
            subsection_ref=att_info.id,
            subsection_title=att_info.title,
            page_number=1,
            source_document=att_info.filename or att_info.id,
            category=category
        )
    
    def _extract_from_text(self, text: str, parent_section: UCFSection, 
                           subsection_ref: Optional[str], subsection_title: str,
                           page_number: int, source_document: str,
                           category: RequirementCategory) -> List[StructuredRequirement]:
        """
        Extract requirements from a block of text.
        
        This looks for paragraphs containing binding language rather than
        splitting into sentences.
        """
        requirements = []
        
        # Determine if this is an evaluation section
        is_evaluation_section = (category == RequirementCategory.EVALUATION_FACTOR or 
                                 parent_section == UCFSection.SECTION_M)
        
        # For Section L and M, we want to be more inclusive of informational items
        include_informational = parent_section in [UCFSection.SECTION_L, UCFSection.SECTION_M]
        
        # Split into paragraphs (preserve structure)
        paragraphs = self._split_into_paragraphs(text)
        
        for para in paragraphs:
            para = para.strip()
            
            # Skip if too short or too long
            if len(para) < self.MIN_REQUIREMENT_LENGTH:
                continue
            
            # Check for binding language
            binding_level, binding_keyword = self._detect_binding_level(para, is_evaluation_section)
            
            # Skip purely informational paragraphs UNLESS we're in L or M sections
            # and the paragraph contains important structural info
            if binding_level == BindingLevel.INFORMATIONAL:
                if not include_informational:
                    continue
                # Even for L/M, skip if it's generic text without submission/evaluation language
                if not re.search(r'(submit|proposal|factor|evaluat|volume|page|format|attachment)', 
                                para.lower()):
                    continue
            
            # Skip if it looks like a header or table of contents
            if self._is_header_or_toc(para):
                continue
            
            # Look for RFP's own reference in the paragraph
            rfp_reference = self._find_rfp_reference(para, parent_section.value)
            
            # Generate ID if needed
            self.counters[category] = self.counters.get(category, 0) + 1
            generated_id = f"TW-{parent_section.value}-{self.counters[category]:04d}"
            
            # Find cross-references
            cross_refs = self._find_cross_references(para)
            
            # Create requirement
            req = StructuredRequirement(
                rfp_reference=rfp_reference or (subsection_ref if subsection_ref else parent_section.value),
                generated_id=generated_id,
                full_text=para,
                category=category,
                binding_level=binding_level,
                binding_keyword=binding_keyword,
                source_section=parent_section,
                source_subsection=subsection_ref,
                page_number=page_number,
                source_document=source_document,
                parent_title=subsection_title,
                evaluation_factor=None,  # Will be linked later
                references_to=cross_refs
            )
            
            requirements.append(req)
            
            # If paragraph is very long, it might contain multiple requirements
            # But per best practices, we should NOT split - extract verbatim
        
        return requirements
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs while preserving structure.
        
        We want to keep related content together, not split mid-requirement.
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Also split on numbered items if they look like separate requirements
        result = []
        for para in paragraphs:
            # Check if this paragraph has multiple numbered items
            # Pattern: starts with number/letter followed by requirement
            numbered_items = re.split(
                r'(?:^|\n)(?:\([a-z\d]+\)|[a-z\d]+\.|[a-z\d]+\))\s+',
                para,
                flags=re.MULTILINE
            )
            
            if len(numbered_items) > 1:
                # Multiple items - keep them separate but with context
                for item in numbered_items:
                    if item.strip():
                        result.append(item.strip())
            else:
                result.append(para)
        
        return result
    
    def _detect_binding_level(self, text: str, is_evaluation_section: bool = False) -> Tuple[BindingLevel, str]:
        """
        Detect how binding this requirement is.
        
        For Section M (evaluation), we use different patterns since evaluation
        criteria rarely use "shall" but are still critical requirements.
        """
        text_lower = text.lower()
        
        # Check mandatory
        for pattern in self.MANDATORY_KEYWORDS:
            match = re.search(pattern, text_lower)
            if match:
                return BindingLevel.MANDATORY, match.group(0)
        
        # For evaluation sections, check evaluation-specific keywords
        if is_evaluation_section:
            for pattern in self.EVALUATION_KEYWORDS:
                match = re.search(pattern, text_lower)
                if match:
                    return BindingLevel.MANDATORY, match.group(0)  # Evaluation criteria are mandatory to address
        
        # Check should
        for pattern in self.SHOULD_KEYWORDS:
            match = re.search(pattern, text_lower)
            if match:
                return BindingLevel.HIGHLY_DESIRABLE, match.group(0)
        
        # Check may
        for pattern in self.MAY_KEYWORDS:
            match = re.search(pattern, text_lower)
            if match:
                return BindingLevel.DESIRABLE, match.group(0)
        
        return BindingLevel.INFORMATIONAL, ""
    
    def _is_header_or_toc(self, text: str) -> bool:
        """Check if this looks like a header rather than a requirement"""
        # Very short
        if len(text) < 50:
            return True
        
        # All caps (likely a header)
        if text.isupper() and len(text) < 200:
            return True
        
        # Table of contents patterns
        toc_patterns = [
            r'^TABLE\s+OF\s+CONTENTS',
            r'^\s*\d+\.\d+\s+[A-Z][^.]+\s+\d+$',  # "3.1 Overview 15"
            r'^SECTION\s+[A-M]\s*[-–—]',
            r'^PART\s+[IVX]+\s*[-–—]',
        ]
        for pattern in toc_patterns:
            if re.match(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        
        return False
    
    def _find_rfp_reference(self, text: str, section_letter: str) -> Optional[str]:
        """Find the RFP's own reference number in the text"""
        # Look for patterns like L.4.B.2, C.3.1.a, M.2.b
        patterns = [
            rf'\b({section_letter}\.\d+(?:\.[A-Za-z\d]+)*)\b',  # L.4.B.2
            rf'\b([A-M]\.\d+(?:\.[A-Za-z\d]+)*)\b',  # Any section reference
            r'\b(PWS\s+\d+(?:\.\d+)*)\b',  # PWS 2.3.1
            r'\b(SOW\s+\d+(?:\.\d+)*)\b',  # SOW 2.3.1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:200])  # Look in first 200 chars
            if match:
                return match.group(1)
        
        return None
    
    def _find_cross_references(self, text: str) -> List[str]:
        """Find references to other sections/attachments in the text"""
        refs = []
        
        for pattern in self.CROSS_REFERENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref = match.group(0)
                if ref not in refs:
                    refs.append(ref)
        
        return refs
    
    def _build_stats(self, result: ExtractionResult) -> Dict[str, Any]:
        """Build extraction statistics"""
        stats = {
            'total': len(result.all_requirements),
            'section_l': len(result.section_l_requirements),
            'technical': len(result.technical_requirements),
            'evaluation': len(result.evaluation_requirements),
            'attachment': len(result.attachment_requirements),
            'administrative': len(result.administrative_requirements),
            'by_binding_level': {
                'mandatory': sum(1 for r in result.all_requirements if r.binding_level == BindingLevel.MANDATORY),
                'highly_desirable': sum(1 for r in result.all_requirements if r.binding_level == BindingLevel.HIGHLY_DESIRABLE),
                'desirable': sum(1 for r in result.all_requirements if r.binding_level == BindingLevel.DESIRABLE),
            },
            'sections_found': [s.value for s in result.structure.sections.keys()] if result.structure else [],
            'sow_location': result.structure.sow_location if result.structure else None,
        }

        # Add OASIS+ task order stats if applicable
        if result.formatting_constraints:
            stats['formatting_constraints_count'] = len(result.formatting_constraints)
            stats['p0_constraints'] = sum(1 for c in result.formatting_constraints if c.priority == "P0")
        if result.volume_structure:
            stats['volumes_required'] = len(result.volume_structure)
        if result.evaluation_subfactors:
            stats['evaluation_subfactors_count'] = len(result.evaluation_subfactors)
        if result.placement_procedures_source:
            stats['placement_procedures_source'] = result.placement_procedures_source

        return stats

    def _find_placement_procedures(self, documents: List[Dict[str, Any]],
                                   structure: DocumentStructure) -> Optional[Dict[str, Any]]:
        """
        Find the Placement Procedures document which serves as L/M for OASIS+ task orders.

        Returns:
            Dict with 'text', 'filename', 'attachment_id' if found, None otherwise
        """
        # Check attachments first
        for att_id, att_info in structure.attachments.items():
            title_lower = att_info.title.lower() if att_info.title else ""
            filename_lower = att_info.filename.lower() if att_info.filename else ""

            if any(term in title_lower or term in filename_lower for term in
                   ['placement procedure', 'placement_procedure']):
                return {
                    'text': att_info.content,
                    'filename': att_info.filename,
                    'attachment_id': att_id,
                    'title': att_info.title
                }

        # Check documents directly
        for doc in documents:
            filename = doc.get('filename', '').lower()
            if 'placement' in filename and 'procedure' in filename:
                return {
                    'text': doc.get('text', ''),
                    'filename': doc.get('filename', ''),
                    'attachment_id': 'Attachment 2',
                    'title': 'Placement Procedures'
                }

        return None

    def _extract_formatting_constraints(self, text: str) -> List[FormattingConstraint]:
        """
        Extract P0 formatting constraints from Placement Procedures.

        These include page limits, font requirements, margin requirements, and file formats.
        """
        constraints = []
        text_lower = text.lower()

        # =========================================================================
        # PAGE LIMITS - Multiple pattern styles for different RFP formats
        # =========================================================================

        # Pattern 1: Standard "X page limit" or "limit of X pages"
        page_patterns = [
            (r'(\d+)\s*(?:page|pg)s?\s*(?:limit|maximum)', 'PAGE_LIMIT'),
            (r'(?:limit|maximum)\s*(?:of\s*)?(\d+)\s*(?:page|pg)s?', 'PAGE_LIMIT'),
            (r'(?:not\s+(?:to\s+)?exceed|no\s+more\s+than)\s*(\d+)\s*(?:page|pg)s?', 'PAGE_LIMIT'),
        ]

        # Pattern 2: Table format "X Pages" or just number in page limit column
        # Look for table-style entries like "8 Pages" or "10" after section names
        table_page_patterns = [
            # "Technical 8 Pages" or "Executive Summary 1"
            (r'(executive\s+summary|technical|management|infrastructure|cost|price)[^\n]*?(\d+)\s*(?:pages?)?', 'PAGE_LIMIT'),
            # "SF 1 Management Approach 10"
            (r'(?:sf\s*\d+|sub\s*factor\s*\d+)[^\n]*?(?:approach|plan)[^\n]*?(\d+)', 'PAGE_LIMIT'),
        ]

        # Check for consequence language once for all page limits
        has_excess_consequence = 'excess pages will not be read' in text_lower or 'will not be read or considered' in text_lower

        for pattern, constraint_type in page_patterns:
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 150)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end = text.find('\n', match.end())
                if line_end == -1:
                    line_end = len(text)
                line_context = text[line_start:line_end].lower()

                volume = "All"
                if re.search(r'volume\s*1\b|technical\s+volume', line_context):
                    volume = "Volume 1 (Technical)"
                elif re.search(r'volume\s*2\b|cost\s+volume|price\s+volume', line_context):
                    volume = "Volume 2 (Cost/Price)"
                elif re.search(r'volume\s*3\b|contract\s+doc', line_context):
                    volume = "Volume 3 (Contract Documentation)"

                consequence = "Excess pages will NOT be read or considered" if has_excess_consequence else "Page limits apply"

                constraints.append(FormattingConstraint(
                    constraint_type=constraint_type,
                    description=f"Page limit: {match.group(1)} pages",
                    value=match.group(1),
                    applies_to=volume,
                    consequence=consequence,
                    priority="P0"
                ))

        # Extract page limits from table format (OASIS+ style)
        # Look for patterns like "Executive Summary 1 Y" or "Management Approach 10 Y"
        section_page_patterns = [
            (r'executive\s+summary[^\d]*(\d+)', 'Executive Summary'),
            (r'management\s+(?:approach|support)[^\d]*(\d+)', 'Management Approach (SF1)'),
            (r'infrastructure\s+approach[^\d]*(\d+)', 'Infrastructure Approach (SF2)'),
            (r'sf\s*1[^\d]*management[^\d]*(\d+)', 'Management Approach (SF1)'),
            (r'sf\s*2[^\d]*infrastructure[^\d]*(\d+)', 'Infrastructure Approach (SF2)'),
        ]

        for pattern, section_name in section_page_patterns:
            match = re.search(pattern, text_lower)
            if match:
                page_count = match.group(1)
                # Only add if reasonable page count (1-100)
                if 1 <= int(page_count) <= 100:
                    constraints.append(FormattingConstraint(
                        constraint_type="PAGE_LIMIT",
                        description=f"{section_name}: {page_count} page{'s' if int(page_count) > 1 else ''} maximum",
                        value=page_count,
                        applies_to="Volume 1 (Technical)",
                        consequence="Excess pages will NOT be read or considered" if has_excess_consequence else "Page limits apply",
                        priority="P0"
                    ))

        # =========================================================================
        # FONT REQUIREMENTS
        # =========================================================================
        # Pattern for "no smaller than X-point" or "X-point font"
        font_patterns = [
            (r'no\s+smaller\s+than\s+(\d+)[-\s]*point', 'FONT_SIZE'),
            (r'(\d+)[-\s]*point\s*(?:font|type)', 'FONT_SIZE'),
            (r'(\d+)[-\s]*point\s+times', 'FONT_SIZE'),
            (r'(times\s+new\s+roman|arial|calibri)', 'FONT_FAMILY'),
        ]

        seen_fonts = set()
        for pattern, constraint_type in font_patterns:
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 80)
                end = min(len(text), match.end() + 80)
                context = text[start:end].lower()

                applies_to = "Body text"
                if 'graphic' in context or 'table' in context or 'figure' in context:
                    applies_to = "Graphics/Tables"

                if constraint_type == 'FONT_SIZE':
                    font_size = match.group(1)
                    key = f"{font_size}-{applies_to}"
                    if key not in seen_fonts:
                        seen_fonts.add(key)
                        desc = f"Font size: no smaller than {font_size}-point"
                        constraints.append(FormattingConstraint(
                            constraint_type=constraint_type,
                            description=desc,
                            value=font_size,
                            applies_to=applies_to,
                            consequence="Non-compliance may result in disqualification",
                            priority="P0"
                        ))
                else:
                    font_name = match.group(1).title()
                    key = font_name
                    if key not in seen_fonts:
                        seen_fonts.add(key)
                        constraints.append(FormattingConstraint(
                            constraint_type=constraint_type,
                            description=f"Font family: {font_name}",
                            value=font_name,
                            applies_to="All text",
                            consequence="Non-compliance may result in disqualification",
                            priority="P0"
                        ))

        # =========================================================================
        # MARGIN REQUIREMENTS
        # =========================================================================
        margin_patterns = [
            (r'(\d+(?:\.\d+)?)\s*[-\s]?inch\s*margins?\s*(?:all\s*around)?', 'standard'),
            (r'one[-\s]?inch\s*margins?', 'one-inch'),
            (r'half[-\s]?inch\s*margins?', 'half-inch'),
        ]

        for pattern, margin_type in margin_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if margin_type == 'one-inch':
                    value = "1"
                elif margin_type == 'half-inch':
                    value = "0.5"
                else:
                    value = match.group(1)

                # Check context for what it applies to
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].lower()

                applies_to = "All pages"
                if '11x17' in context or 'worksheet' in context or 'cost table' in context:
                    applies_to = "Cost worksheets (11x17)"

                constraints.append(FormattingConstraint(
                    constraint_type="MARGIN",
                    description=f"Margins: {value}-inch margins all around",
                    value=f"{value} inch",
                    applies_to=applies_to,
                    consequence="Non-compliance may result in disqualification",
                    priority="P0"
                ))
                break  # Only add one margin constraint

        # =========================================================================
        # FILE FORMAT REQUIREMENTS
        # =========================================================================
        format_patterns = [
            (r'(?:microsoft|ms)\s+word\s*(?:20)?(\d+)?', 'FILE_FORMAT', 'Microsoft Word'),
            (r'(?:microsoft|ms)\s+excel\s*(?:20)?(\d+)?', 'FILE_FORMAT', 'Microsoft Excel'),
            (r'adobe\s+acrobat(?:\s+pro)?', 'FILE_FORMAT', 'Adobe Acrobat PDF'),
            (r'\.pdf\s+format', 'FILE_FORMAT', 'PDF'),
        ]

        seen_formats = set()
        for pattern, constraint_type, format_name in format_patterns:
            if re.search(pattern, text_lower) and format_name not in seen_formats:
                seen_formats.add(format_name)
                constraints.append(FormattingConstraint(
                    constraint_type=constraint_type,
                    description=f"File format: {format_name}",
                    value=format_name,
                    applies_to="Electronic submission",
                    consequence="Files must be accessible in specified format",
                    priority="P0"
                ))

        # =========================================================================
        # SUBMISSION METHOD
        # =========================================================================
        if 'electronic submission only' in text_lower or 'electronic only' in text_lower:
            constraints.append(FormattingConstraint(
                constraint_type="SUBMISSION_METHOD",
                description="Electronic submission only via secure file transfer (eBuy/GSA)",
                value="Electronic",
                applies_to="All volumes",
                consequence="Email and physical submissions are NOT acceptable",
                priority="P0"
            ))

        # =========================================================================
        # LATE PROPOSAL WARNING
        # =========================================================================
        if 'late proposals' in text_lower and 'ineligible' in text_lower:
            constraints.append(FormattingConstraint(
                constraint_type="DEADLINE",
                description="Proposals must be received by deadline",
                value="See RFP for due date",
                applies_to="All volumes",
                consequence="Late proposals may make Offeror ineligible for award",
                priority="P0"
            ))

        # =========================================================================
        # DISQUALIFICATION CONDITIONS
        # =========================================================================
        if 'unacceptable' in text_lower and 'ineligible for award' in text_lower:
            constraints.append(FormattingConstraint(
                constraint_type="RATING_THRESHOLD",
                description="Technical subfactors must not receive 'Unacceptable' rating",
                value="Minimum: Good",
                applies_to="All technical subfactors",
                consequence="Unacceptable rating on ANY technical subfactor = ineligible for award",
                priority="P0"
            ))

        if 'non-compliant' in text_lower and 'ineligible for award' in text_lower:
            constraints.append(FormattingConstraint(
                constraint_type="RATING_THRESHOLD",
                description="Price criteria must not receive 'Non-Compliant' rating",
                value="Minimum: Compliant",
                applies_to="All price criteria",
                consequence="Non-compliant rating on ANY price criteria = ineligible for award",
                priority="P0"
            ))

        return constraints

    def _extract_volume_structure(self, text: str) -> List[VolumeStructure]:
        """
        Extract the required volume structure from Placement Procedures.
        """
        volumes = []
        text_lower = text.lower()

        # Common volume patterns for OASIS+ task orders
        volume_defs = [
            (1, r'volume\s*(?:1|i|one)[:\s]*(.{0,100})', ['Executive Summary', 'Technical']),
            (2, r'volume\s*(?:2|ii|two)[:\s]*(.{0,100})', ['Cost', 'Price']),
            (3, r'volume\s*(?:3|iii|three)[:\s]*(.{0,100})', ['Contract Documentation', 'Administrative']),
        ]

        for vol_num, pattern, default_content in volume_defs:
            match = re.search(pattern, text_lower)
            if match:
                vol_name = match.group(1).strip() if match.group(1) else f"Volume {vol_num}"
                # Clean up the name
                vol_name = re.sub(r'[:\n\r].*', '', vol_name).strip()
                if len(vol_name) > 80:
                    vol_name = vol_name[:80]

                volumes.append(VolumeStructure(
                    volume_number=vol_num,
                    volume_name=vol_name.title() if vol_name else f"Volume {vol_num}",
                    required_content=default_content,
                    page_limit=None,  # Extracted separately
                    subfactors=[]
                ))

        # If no volumes found explicitly, create default OASIS+ structure
        if not volumes and 'oasis' in text_lower:
            volumes = [
                VolumeStructure(1, "Executive Summary and Technical Volume",
                               ["Management Approach", "Infrastructure Approach"]),
                VolumeStructure(2, "Cost & Price Volume",
                               ["Supply Pricing", "Labor Rates"]),
                VolumeStructure(3, "Contract Documentation Volume",
                               ["Required Forms", "Certifications"]),
            ]

        return volumes

    def _extract_evaluation_subfactors(self, text: str) -> Tuple[List[EvaluationSubFactor], Dict[str, str]]:
        """
        Extract evaluation factors, sub-factors, and adjectival ratings from Placement Procedures.

        Returns:
            Tuple of (subfactors list, adjectival ratings dict)
        """
        subfactors = []
        adjectival_ratings = {}

        # Extract adjectival rating definitions - use sentence boundary detection
        rating_patterns = {
            'Exceptional': r'exceptional[:\s-]+([^.]+(?:expectations?|strengths?|requirements?)[^.]*\.)',
            'Very Good': r'very\s+good[:\s-]+([^.]+(?:expectations?|strengths?|requirements?)[^.]*\.)',
            'Good': r'(?<!\bvery\s)\bgood[:\s-]+([^.]+(?:expectations?|requirements?|solicitation)[^.]*\.)',
            'Unacceptable': r'unacceptable[:\s-]+([^.]+(?:requirements?|confidence|solicitation)[^.]*\.)',
        }

        for rating, pattern in rating_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                definition = match.group(1).strip()
                # Clean up the definition
                definition = re.sub(r'\s+', ' ', definition)
                # Truncate if too long
                if len(definition) > 200:
                    definition = definition[:200] + "..."
                adjectival_ratings[rating] = definition

        # Extract sub-factors
        subfactor_patterns = [
            r'sub[-\s]?factor\s*(\d+)[:\s]*([^\n]{5,100})',
            r'factor\s*(\d+)[:\s]*([^\n]{5,100})',
            r'(\d+)\.\s*(management\s+approach)',
            r'(\d+)\.\s*(infrastructure\s+approach)',
            r'(\d+)\.\s*(technical\s+approach)',
            r'(\d+)\.\s*(past\s+performance)',
        ]

        seen_subfactors = set()
        for pattern in subfactor_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num = match.group(1)
                name = match.group(2).strip().title()

                if name.lower() not in seen_subfactors:
                    seen_subfactors.add(name.lower())

                    # Determine rating type
                    rating_scale = "Adjectival"
                    if 'cost' in name.lower() or 'price' in name.lower():
                        rating_scale = "Price Analysis"
                    elif 'past performance' in name.lower():
                        rating_scale = "Confidence Rating"

                    subfactors.append(EvaluationSubFactor(
                        factor_name="Technical Factor" if 'cost' not in name.lower() else "Cost Factor",
                        subfactor_name=name,
                        subfactor_number=int(num) if num.isdigit() else len(subfactors) + 1,
                        instruction_text="",  # Would need more context
                        evaluation_text="",
                        rating_scale=rating_scale,
                        volume="Volume 1" if 'cost' not in name.lower() else "Volume 2"
                    ))

        return subfactors, adjectival_ratings

    def _extract_pass_fail_requirements(self, text: str, result: ExtractionResult) -> None:
        """
        Extract Pass/Fail (P0) requirements like mandatory forms.

        These are added as high-priority requirements to the result.
        """
        # Common mandatory forms for OASIS+ and DoD
        mandatory_forms = [
            (r'SF[-\s]?1449', 'SF1449 Standard Form'),
            (r'DD[-\s]?254', 'DD Form 254 Security Classification'),
            (r'DD[-\s]?250', 'DD Form 250 Material Inspection'),
            (r'OCI\s+(?:mitigation|checklist|plan)', 'OCI Mitigation Plan'),
            (r'representations?\s+and\s+certifications?', 'Representations and Certifications'),
            (r'sam\.gov\s+registration', 'SAM.gov Registration'),
            (r'online\s+representation', 'Online Representations and Certifications'),
        ]

        text_lower = text.lower()

        for pattern, form_name in mandatory_forms:
            if re.search(pattern, text_lower):
                # Create a P0 requirement for this form
                req = StructuredRequirement(
                    rfp_reference="FORM-P0",
                    generated_id=f"TW-FORM-{len(result.administrative_requirements) + 1:04d}",
                    full_text=f"MANDATORY FORM: {form_name} - Must be submitted as part of proposal. Pass/Fail requirement.",
                    category=RequirementCategory.ADMINISTRATIVE,
                    binding_level=BindingLevel.MANDATORY,
                    binding_keyword="required",
                    source_section=UCFSection.SECTION_K,
                    source_subsection="Forms",
                    page_number=0,
                    source_document="Placement Procedures",
                    parent_title="Mandatory Forms",
                    evaluation_factor="Pass/Fail"
                )

                # Check if not already captured
                if form_name.lower() not in ' '.join(r.full_text.lower() for r in result.administrative_requirements):
                    result.administrative_requirements.append(req)
                    result.all_requirements.append(req)

    def _process_oasis_task_order(self, documents: List[Dict[str, Any]],
                                   structure: DocumentStructure,
                                   result: ExtractionResult) -> None:
        """
        Enhanced processing for OASIS+ task orders.

        Extracts L/M requirements from Placement Procedures and adds
        formatting constraints, volume structure, and evaluation criteria.
        """
        # Find Placement Procedures
        pp_doc = self._find_placement_procedures(documents, structure)

        if not pp_doc:
            # Check if any document content mentions placement procedures
            for doc in documents:
                if 'placement procedure' in doc.get('text', '').lower():
                    pp_doc = {
                        'text': doc.get('text', ''),
                        'filename': doc.get('filename', ''),
                        'attachment_id': 'Embedded',
                        'title': 'Placement Procedures (Embedded)'
                    }
                    break

        if not pp_doc:
            return

        pp_text = pp_doc['text']
        result.placement_procedures_source = pp_doc.get('filename') or pp_doc.get('attachment_id', '')

        # Extract formatting constraints (P0)
        result.formatting_constraints = self._extract_formatting_constraints(pp_text)

        # Extract volume structure
        result.volume_structure = self._extract_volume_structure(pp_text)

        # Extract evaluation sub-factors and adjectival ratings
        subfactors, ratings = self._extract_evaluation_subfactors(pp_text)
        result.evaluation_subfactors = subfactors
        result.adjectival_ratings = ratings

        # Extract pass/fail requirements
        self._extract_pass_fail_requirements(pp_text, result)

        # Now extract L/M equivalent requirements from Placement Procedures
        # Section L equivalent: submission instructions
        l_reqs = self._extract_from_text(
            text=pp_text,
            parent_section=UCFSection.SECTION_L,
            subsection_ref="Placement Procedures",
            subsection_title="Proposal Submission Instructions (from Placement Procedures)",
            page_number=1,
            source_document=pp_doc.get('filename', 'Placement Procedures'),
            category=RequirementCategory.SECTION_L_COMPLIANCE
        )

        # Add L requirements that aren't duplicates
        seen_hashes = {r.text_hash for r in result.section_l_requirements}
        for req in l_reqs:
            if req.text_hash not in seen_hashes:
                seen_hashes.add(req.text_hash)
                result.section_l_requirements.append(req)
                result.all_requirements.append(req)

        # Section M equivalent: evaluation criteria
        m_reqs = self._extract_from_text(
            text=pp_text,
            parent_section=UCFSection.SECTION_M,
            subsection_ref="Placement Procedures",
            subsection_title="Evaluation Criteria (from Placement Procedures)",
            page_number=1,
            source_document=pp_doc.get('filename', 'Placement Procedures'),
            category=RequirementCategory.EVALUATION_FACTOR
        )

        # Add M requirements that aren't duplicates
        seen_hashes = {r.text_hash for r in result.evaluation_requirements}
        for req in m_reqs:
            if req.text_hash not in seen_hashes:
                seen_hashes.add(req.text_hash)
                result.evaluation_requirements.append(req)
                result.all_requirements.append(req)


def extract_requirements_structured(documents: List[Dict[str, Any]]) -> ExtractionResult:
    """
    Convenience function for structured requirement extraction.
    
    Usage:
        documents = [
            {'text': '...', 'filename': 'RFP.pdf', 'pages': [...]},
            {'text': '...', 'filename': 'SOW.pdf', 'pages': [...]},
        ]
        result = extract_requirements_structured(documents)
        
        # Access by category
        for req in result.section_l_requirements:
            print(f"{req.rfp_reference}: {req.full_text[:80]}...")
    """
    extractor = SectionAwareExtractor()
    return extractor.extract(documents)
