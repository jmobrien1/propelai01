"""
PropelAI v3.1: Section-Aware Requirement Extractor

v3.1 Updates:
- Integrated v3.0 Router Logic for mode-specific extraction
- MODE D: Extract from spreadsheet files (pandas-based)
- MODE A: Cross-reference Section L/M for page limits injection
- MODE A: Tag requirements with target volumes (I, II, III)

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
import sys
from pathlib import Path
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

# v3.1: Import Router Logic
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from agents.chat.rfp_chat_agent import RFPType
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    # Fallback
    class RFPType(Enum):
        FEDERAL_STANDARD = "federal_standard"
        SLED_STATE = "sled_state"
        DOD_ATTACHMENT = "dod_attachment"
        SPREADSHEET = "spreadsheet"
        UNKNOWN = "unknown"


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
    
    # v3.1: New fields for Router Logic
    page_limit: Optional[str] = None      # MODE A: Injected page limit (e.g., "5 pages")
    target_volume: Optional[str] = None   # MODE A: Target volume (Vol I, II, III)
    row_number: Optional[int] = None      # MODE D: Spreadsheet row number
    
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
    
    def __init__(self, preserve_rfp_ids: bool = True, rfp_type: Optional[RFPType] = None):
        """
        Args:
            preserve_rfp_ids: If True, always use RFP's own references as primary ID
            rfp_type: v3.1 - RFP type for mode-specific extraction
        """
        self.preserve_rfp_ids = preserve_rfp_ids
        self.rfp_type = rfp_type or RFPType.UNKNOWN
        self.counters = {}  # For generating IDs when needed
    
    def extract(self, documents: List[Dict[str, Any]], structure: Optional[DocumentStructure] = None, file_paths: List[str] = None) -> ExtractionResult:
        """
        Extract requirements from documents using structural analysis.
        
        v3.1: Dispatches to mode-specific extractors based on rfp_type.
        
        Args:
            documents: List of parsed documents with 'text', 'filename', 'pages'
            structure: Pre-computed document structure (will compute if not provided)
            file_paths: v3.1 - List of file paths for spreadsheet extraction
            
        Returns:
            ExtractionResult with categorized requirements
        """
        # v3.1: Classify RFP type if not set
        if self.rfp_type == RFPType.UNKNOWN and file_paths:
            self.rfp_type = self._classify_rfp_type(documents, file_paths)
        
        # v3.1: MODE D - Spreadsheet extraction
        if self.rfp_type == RFPType.SPREADSHEET:
            return self._extract_from_spreadsheet(documents, file_paths or [])
        
        # Standard extraction (MODE A, B, C, UNKNOWN)
        # First, analyze document structure
        if structure is None:
            structure = analyze_rfp_structure(documents)
        
        result = ExtractionResult(structure=structure)
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
                        
                        # Also add to technical if it's SOW/PWS
                        if att_info.document_type in ['SOW', 'PWS']:
                            result.technical_requirements.append(req)
        
        # v3.1: Post-processing based on RFP type
        if self.rfp_type == RFPType.FEDERAL_STANDARD:
            # MODE A: Inject page limits and structure volumes
            result = self._inject_page_limits(result)
            result = self._structure_volumes(result)
        
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
        return {
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
