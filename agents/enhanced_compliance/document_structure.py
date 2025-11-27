"""
PropelAI v2.9: RFP Document Structure Parser

This module implements proper document structure analysis BEFORE requirement extraction.
Per best practices: "Anchor the CTM to the RFP Structure — EXACTLY as Evaluators Expect"

The parser:
1. Identifies Section A-M boundaries in the document
2. Detects SOW/PWS location (embedded in Section C or as attachment)
3. Maps attachments and their contents
4. Preserves the RFP's own numbering scheme
5. Tracks page numbers for each section

This structural understanding is REQUIRED before any extraction begins.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class UCFSection(Enum):
    """Uniform Contract Format sections per FAR 15.204-1"""
    SECTION_A = "A"   # Solicitation/Contract Form (SF 33, SF 1449)
    SECTION_B = "B"   # Supplies or Services and Prices/Costs
    SECTION_C = "C"   # Description/Specifications/Statement of Work
    SECTION_D = "D"   # Packaging and Marking
    SECTION_E = "E"   # Inspection and Acceptance
    SECTION_F = "F"   # Deliveries or Performance
    SECTION_G = "G"   # Contract Administration Data
    SECTION_H = "H"   # Special Contract Requirements
    SECTION_I = "I"   # Contract Clauses
    SECTION_J = "J"   # List of Attachments
    SECTION_K = "K"   # Representations, Certifications, Other Statements
    SECTION_L = "L"   # Instructions, Conditions, Notices to Offerors
    SECTION_M = "M"   # Evaluation Factors for Award


@dataclass
class SectionBoundary:
    """Represents a detected section in the RFP"""
    section: UCFSection
    title: str                          # e.g., "SECTION L - INSTRUCTIONS TO OFFERORS"
    start_page: int
    end_page: int
    start_char: int                     # Character offset in full text
    end_char: int
    content: str                        # Full text of this section
    subsections: Dict[str, 'SubsectionBoundary'] = field(default_factory=dict)


@dataclass
class SubsectionBoundary:
    """Represents a subsection like L.4, C.3.1, M.2.a"""
    reference: str                      # e.g., "L.4.B.2", "C.3.1.a"
    title: str                          # e.g., "Technical Proposal Requirements"
    start_char: int
    end_char: int
    content: str
    page_number: int
    parent_section: UCFSection


@dataclass
class AttachmentInfo:
    """Information about an RFP attachment"""
    id: str                             # e.g., "Attachment 1", "Exhibit A", "J-1"
    title: str
    filename: Optional[str]
    content: str
    page_count: int
    document_type: str                  # "SOW", "PWS", "Budget Template", "Experience Form", etc.
    contains_requirements: bool


@dataclass
class DocumentStructure:
    """Complete structural analysis of an RFP"""
    solicitation_number: str
    title: str
    
    # Section boundaries
    sections: Dict[UCFSection, SectionBoundary] = field(default_factory=dict)
    
    # Attachments
    attachments: Dict[str, AttachmentInfo] = field(default_factory=dict)
    
    # Special locations
    sow_location: Optional[str] = None          # "SECTION_C" or "Attachment 2"
    pws_location: Optional[str] = None
    evaluation_factors_location: Optional[str] = None  # Usually "SECTION_M"
    
    # Document inventory
    total_pages: int = 0
    documents_parsed: List[str] = field(default_factory=list)
    
    # Parsing metadata
    parsing_warnings: List[str] = field(default_factory=list)


class RFPStructureParser:
    """
    Parses RFP documents to identify structural boundaries.
    
    This MUST run before requirement extraction to ensure:
    - Requirements are attributed to correct sections
    - RFP's own numbering is preserved
    - Cross-references can be resolved
    """
    
    # Section header patterns (multiple variants for flexibility)
    SECTION_PATTERNS = {
        UCFSection.SECTION_A: [
            r"SECTION\s+A[\s:\-–—]+",
            r"PART\s+I[\s:\-–—]+.*SCHEDULE",
            r"SF\s*(?:33|1449)",
            r"SOLICITATION.*CONTRACT.*FORM",
        ],
        UCFSection.SECTION_B: [
            r"SECTION\s+B[\s:\-–—]+",
            r"SUPPLIES\s+OR\s+SERVICES\s+AND\s+PRICES",
            r"B\.\d+\s+SUPPLIES\s+OR\s+SERVICES",
        ],
        UCFSection.SECTION_C: [
            r"SECTION\s+C[\s:\-–—]+",
            r"DESCRIPTION[\s/]+SPECIFICATIONS",
            r"STATEMENT\s+OF\s+WORK",
            r"C\.\d+\s+DESCRIPTION",
        ],
        UCFSection.SECTION_D: [
            r"SECTION\s+D[\s:\-–—]+",
            r"PACKAGING\s+AND\s+MARKING",
        ],
        UCFSection.SECTION_E: [
            r"SECTION\s+E[\s:\-–—]+",
            r"INSPECTION\s+AND\s+ACCEPTANCE",
        ],
        UCFSection.SECTION_F: [
            r"SECTION\s+F[\s:\-–—]+",
            r"DELIVERIES\s+OR\s+PERFORMANCE",
        ],
        UCFSection.SECTION_G: [
            r"SECTION\s+G[\s:\-–—]+",
            r"CONTRACT\s+ADMINISTRATION\s+DATA",
        ],
        UCFSection.SECTION_H: [
            r"SECTION\s+H[\s:\-–—]+",
            r"SPECIAL\s+CONTRACT\s+REQUIREMENTS",
        ],
        UCFSection.SECTION_I: [
            r"SECTION\s+I[\s:\-–—]+",
            r"CONTRACT\s+CLAUSES",
        ],
        UCFSection.SECTION_J: [
            r"SECTION\s+J[\s:\-–—]+",
            r"LIST\s+OF\s+(?:DOCUMENTS|ATTACHMENTS|EXHIBITS)",
        ],
        UCFSection.SECTION_K: [
            r"SECTION\s+K[\s:\-–—]+",
            r"REPRESENTATIONS.*CERTIFICATIONS",
        ],
        UCFSection.SECTION_L: [
            r"SECTION\s+L[\s:\-–—]+",
            r"INSTRUCTIONS.*(?:CONDITIONS|NOTICES).*OFFERORS",
            r"INSTRUCTIONS\s+TO\s+OFFERORS",
            r"L\.\d+\s+INSTRUCTIONS",
        ],
        UCFSection.SECTION_M: [
            r"SECTION\s+M[\s:\-–—]+",
            r"EVALUATION\s+FACTORS",
            r"M\.\d+\s+EVALUATION",
        ],
    }
    
    # Subsection pattern: matches L.4, L.4.B, L.4.B.2, C.3.1.a, etc.
    SUBSECTION_PATTERN = re.compile(
        r'^(?P<section>[A-M])\.(?P<num1>\d+)(?:\.(?P<num2>[A-Za-z]|\d+))?(?:\.(?P<num3>\d+|[a-z]))?'
        r'(?:\s+|\s*[-–—]\s*)(?P<title>[A-Z][^\n]{3,80})',
        re.MULTILINE
    )
    
    # Alternative: "Article L.4" style
    ARTICLE_PATTERN = re.compile(
        r'ARTICLE\s+(?P<section>[A-M])\.(?P<num>\d+)(?:\s*[-–—]\s*|\s+)(?P<title>[^\n]+)',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Attachment patterns
    ATTACHMENT_PATTERNS = [
        r"ATTACHMENT\s+(\d+|[A-Z])",
        r"EXHIBIT\s+(\d+|[A-Z])",
        r"APPENDIX\s+(\d+|[A-Z])",
        r"J-(\d+)",
    ]
    
    def __init__(self):
        self.warnings = []
    
    def parse_structure(self, documents: List[Dict[str, Any]]) -> DocumentStructure:
        """
        Parse document structure from a list of parsed documents.
        
        Args:
            documents: List of dicts with 'text', 'filename', 'pages' keys
            
        Returns:
            DocumentStructure with identified sections and boundaries
        """
        structure = DocumentStructure(
            solicitation_number="",
            title=""
        )
        
        # Combine all documents for full-text analysis
        combined_text = ""
        page_offsets = []  # Track where each page starts in combined text
        current_offset = 0
        
        for doc in documents:
            structure.documents_parsed.append(doc.get('filename', 'unknown'))
            pages = doc.get('pages', [doc.get('text', '')])
            
            for page_num, page_text in enumerate(pages, 1):
                page_offsets.append((current_offset, page_num, doc.get('filename', '')))
                combined_text += page_text + "\n\n"
                current_offset = len(combined_text)
                structure.total_pages += 1
        
        # Extract solicitation info
        structure.solicitation_number = self._extract_solicitation_number(combined_text)
        structure.title = self._extract_title(combined_text)
        
        # Find section boundaries
        section_matches = self._find_section_boundaries(combined_text)
        
        # Build section objects with content
        for section, (start, end, title) in section_matches.items():
            content = combined_text[start:end]
            start_page = self._offset_to_page(start, page_offsets)
            end_page = self._offset_to_page(end - 1, page_offsets)
            
            boundary = SectionBoundary(
                section=section,
                title=title,
                start_page=start_page,
                end_page=end_page,
                start_char=start,
                end_char=end,
                content=content
            )
            
            # Find subsections within this section
            boundary.subsections = self._find_subsections(content, section, start, page_offsets)
            
            structure.sections[section] = boundary
        
        # Identify SOW/PWS location
        structure.sow_location = self._find_sow_location(structure, combined_text)
        structure.pws_location = self._find_pws_location(structure, combined_text)
        
        # Parse attachments from separate documents or Section J
        structure.attachments = self._parse_attachments(documents, structure)
        
        structure.parsing_warnings = self.warnings
        
        return structure
    
    def _extract_solicitation_number(self, text: str) -> str:
        """Extract solicitation/RFP number"""
        patterns = [
            r"(?:Solicitation|RFP|RFQ|IFB)\s*(?:No\.?|Number|#)?\s*[:.]?\s*([A-Z0-9]{2,}[-\s]?[A-Z0-9]+)",
            r"([A-Z]{2,}\d{2}[-]?\d+[-]?\d*)",  # Pattern like 75N96025R00004
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_title(self, text: str) -> str:
        """Extract RFP title"""
        patterns = [
            r"(?:Title|Subject|For)\s*:\s*([^\n]+)",
            r"SOLICITATION.*FOR\s+([^\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:10000], re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]
        return ""
    
    def _find_section_boundaries(self, text: str) -> Dict[UCFSection, Tuple[int, int, str]]:
        """
        Find start/end positions of each UCF section.
        
        Returns:
            Dict mapping section to (start_offset, end_offset, title)
        """
        section_starts = []
        
        for section, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    # Get the full line for the title
                    line_start = text.rfind('\n', 0, match.start()) + 1
                    line_end = text.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(text)
                    title = text[line_start:line_end].strip()
                    
                    section_starts.append((match.start(), section, title))
                    break  # Only take first match per section
        
        # Sort by position
        section_starts.sort(key=lambda x: x[0])
        
        # Build boundaries (each section ends where the next begins)
        results = {}
        for i, (start, section, title) in enumerate(section_starts):
            if i + 1 < len(section_starts):
                end = section_starts[i + 1][0]
            else:
                end = len(text)
            
            # Don't override if we already found this section
            if section not in results:
                results[section] = (start, end, title)
        
        return results
    
    def _find_subsections(self, section_content: str, parent_section: UCFSection, 
                          section_start_offset: int, page_offsets: List) -> Dict[str, SubsectionBoundary]:
        """Find subsections within a section (e.g., L.4, L.4.B.2)"""
        subsections = {}
        
        # Try standard pattern: L.4.B.2 Title
        for match in self.SUBSECTION_PATTERN.finditer(section_content):
            if match.group('section') == parent_section.value:
                ref = self._build_reference(match)
                abs_start = section_start_offset + match.start()
                
                subsections[ref] = SubsectionBoundary(
                    reference=ref,
                    title=match.group('title').strip(),
                    start_char=abs_start,
                    end_char=abs_start + len(match.group(0)),  # Will be updated
                    content="",  # Will be filled later
                    page_number=self._offset_to_page(abs_start, page_offsets),
                    parent_section=parent_section
                )
        
        # Try Article pattern
        for match in self.ARTICLE_PATTERN.finditer(section_content):
            if match.group('section').upper() == parent_section.value:
                ref = f"{match.group('section').upper()}.{match.group('num')}"
                abs_start = section_start_offset + match.start()
                
                if ref not in subsections:
                    subsections[ref] = SubsectionBoundary(
                        reference=ref,
                        title=match.group('title').strip(),
                        start_char=abs_start,
                        end_char=abs_start + len(match.group(0)),
                        content="",
                        page_number=self._offset_to_page(abs_start, page_offsets),
                        parent_section=parent_section
                    )
        
        # Sort by position and fill in content/end boundaries
        sorted_refs = sorted(subsections.keys(), key=lambda r: subsections[r].start_char)
        for i, ref in enumerate(sorted_refs):
            sub = subsections[ref]
            if i + 1 < len(sorted_refs):
                next_ref = sorted_refs[i + 1]
                sub.end_char = subsections[next_ref].start_char
            else:
                sub.end_char = section_start_offset + len(section_content)
            
            # Extract content
            content_start = sub.start_char - section_start_offset
            content_end = sub.end_char - section_start_offset
            sub.content = section_content[content_start:content_end]
        
        return subsections
    
    def _build_reference(self, match) -> str:
        """Build section reference from regex match"""
        ref = f"{match.group('section')}.{match.group('num1')}"
        if match.group('num2'):
            ref += f".{match.group('num2')}"
        if match.group('num3'):
            ref += f".{match.group('num3')}"
        return ref
    
    def _offset_to_page(self, offset: int, page_offsets: List) -> int:
        """Convert character offset to page number"""
        for i, (start, page_num, _) in enumerate(page_offsets):
            if i + 1 < len(page_offsets):
                if start <= offset < page_offsets[i + 1][0]:
                    return page_num
            else:
                return page_num
        return 1
    
    def _find_sow_location(self, structure: DocumentStructure, text: str) -> Optional[str]:
        """Determine where the Statement of Work is located"""
        # Check if Section C contains SOW
        if UCFSection.SECTION_C in structure.sections:
            c_content = structure.sections[UCFSection.SECTION_C].content
            if re.search(r'STATEMENT\s+OF\s+WORK', c_content, re.IGNORECASE):
                return "SECTION_C"
        
        # Check attachments
        for att_id, att in structure.attachments.items():
            if att.document_type == "SOW":
                return att_id
        
        # Search full text for SOW indicator
        sow_match = re.search(
            r'(?:ATTACHMENT|EXHIBIT)\s+(\d+|[A-Z]).*STATEMENT\s+OF\s+WORK',
            text[:50000], re.IGNORECASE
        )
        if sow_match:
            return f"Attachment {sow_match.group(1)}"
        
        return None
    
    def _find_pws_location(self, structure: DocumentStructure, text: str) -> Optional[str]:
        """Determine where the Performance Work Statement is located"""
        # Check if Section C contains PWS
        if UCFSection.SECTION_C in structure.sections:
            c_content = structure.sections[UCFSection.SECTION_C].content
            if re.search(r'PERFORMANCE\s+WORK\s+STATEMENT', c_content, re.IGNORECASE):
                return "SECTION_C"
        
        # Check attachments
        for att_id, att in structure.attachments.items():
            if att.document_type == "PWS":
                return att_id
        
        return None
    
    def _parse_attachments(self, documents: List[Dict], structure: DocumentStructure) -> Dict[str, AttachmentInfo]:
        """Parse attachment information from documents"""
        attachments = {}
        
        for doc in documents:
            filename = doc.get('filename', '').lower()
            text = doc.get('text', '')
            pages = doc.get('pages', [])
            
            # Detect attachment type from filename
            att_id = None
            doc_type = "General"
            
            if 'attachment' in filename:
                match = re.search(r'attachment\s*(\d+|[a-z])', filename, re.IGNORECASE)
                if match:
                    att_id = f"Attachment {match.group(1).upper()}"
            elif 'exhibit' in filename:
                match = re.search(r'exhibit\s*(\d+|[a-z])', filename, re.IGNORECASE)
                if match:
                    att_id = f"Exhibit {match.group(1).upper()}"
            
            # Classify document type
            if 'sow' in filename or 'statement of work' in filename.replace('_', ' '):
                doc_type = "SOW"
            elif 'pws' in filename or 'performance work' in filename.replace('_', ' '):
                doc_type = "PWS"
            elif 'budget' in filename or 'cost' in filename or 'pricing' in filename:
                doc_type = "Budget Template"
            elif 'experience' in filename or 'past performance' in filename.replace('_', ' '):
                doc_type = "Past Performance"
            elif 'resume' in filename or 'personnel' in filename:
                doc_type = "Personnel"
            elif 'amendment' in filename:
                doc_type = "Amendment"
                # Try to get amendment number
                match = re.search(r'amendment\s*(\d+)', filename, re.IGNORECASE)
                if match:
                    att_id = f"Amendment {match.group(1)}"
            
            # Check content for requirements
            has_requirements = bool(re.search(
                r'\b(?:shall|must|will\s+be\s+required|is\s+required)\b',
                text[:20000], re.IGNORECASE
            ))
            
            if att_id:
                attachments[att_id] = AttachmentInfo(
                    id=att_id,
                    title=doc.get('filename', ''),
                    filename=doc.get('filename'),
                    content=text,
                    page_count=len(pages) if pages else 1,
                    document_type=doc_type,
                    contains_requirements=has_requirements
                )
        
        return attachments
    
    def get_section_content(self, structure: DocumentStructure, section: UCFSection) -> Optional[str]:
        """Get the full content of a specific section"""
        if section in structure.sections:
            return structure.sections[section].content
        return None
    
    def get_subsection_content(self, structure: DocumentStructure, reference: str) -> Optional[str]:
        """Get content of a specific subsection by reference (e.g., 'L.4.B.2')"""
        # Parse the reference to find parent section
        match = re.match(r'([A-M])', reference)
        if not match:
            return None
        
        section_letter = match.group(1)
        try:
            section = UCFSection(f"SECTION_{section_letter}")
        except ValueError:
            return None
        
        if section not in structure.sections:
            return None
        
        subsections = structure.sections[section].subsections
        if reference in subsections:
            return subsections[reference].content
        
        return None


def analyze_rfp_structure(documents: List[Dict[str, Any]]) -> DocumentStructure:
    """
    Convenience function to analyze RFP structure.
    
    Usage:
        documents = [
            {'text': '...', 'filename': 'RFP.pdf', 'pages': ['page1', 'page2', ...]},
            {'text': '...', 'filename': 'SOW.pdf', 'pages': [...]},
        ]
        structure = analyze_rfp_structure(documents)
        
        # Access sections
        section_l = structure.sections.get(UCFSection.SECTION_L)
        if section_l:
            print(f"Section L: pages {section_l.start_page}-{section_l.end_page}")
            for ref, sub in section_l.subsections.items():
                print(f"  {ref}: {sub.title}")
    """
    parser = RFPStructureParser()
    return parser.parse_structure(documents)
