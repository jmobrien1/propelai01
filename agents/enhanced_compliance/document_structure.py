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
        # More specific patterns that require proper solicitation number format
        # Must contain at least one digit to avoid matching prose like "requires delivery"
        patterns = [
            # NIH format: 75N96025R00004
            r"75N\d{5}[A-Z]\d{5}",
            # DoD Navy format: N0017826R30020003
            r"\b(N\d{5}\d{2}[RQ]\d+)\b",
            # DoD Air Force format without hyphens: FA880625RB003
            r"\b(FA\d{6}[RQ][A-Z]?\d{3,})\b",
            # DoD Army format without hyphens: W912HQ25R0001
            r"\b(W\d{3}[A-Z]{2}\d{2}[RQ]\d{4,})\b",
            # DoD Air Force/Army format with hyphens: FA8732-25-R-0001, W912HQ-25-R-0001
            r"\b([A-Z]{2}\d{4}[-]\d{2}[-][A-Z][-]\d{4,})\b",
            # GSA format: GS-00F-12345
            r"\b(GS[-][A-Z0-9]{2,}[-][A-Z0-9]+)\b",
            # Explicit solicitation number with colon/separator (flexible hyphens)
            r"(?:Solicitation|RFP|RFQ|IFB)\s*(?:No\.?|Number|#)\s*[:.]?\s*([A-Z0-9]+(?:[-][A-Z0-9]+)*)",
            # Generic federal format: alphanumeric with hyphens, must have digit
            r"(?:Solicitation|RFP|RFQ|IFB)\s*[:#]\s*([A-Z0-9]+(?:[-][A-Z0-9]+)*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                result = match.group(1) if match.lastindex else match.group(0)
                result = result.strip()
                # Validate: must contain at least one digit and not be common words
                if re.search(r'\d', result) and result.lower() not in ['requires', 'delivery', 'the', 'and', 'for']:
                    return result
        return ""
    
    def _extract_title(self, text: str) -> str:
        """
        Extract RFP title with smart filtering.

        Priority order:
        1. Look for explicit "Title:" or "Subject:" fields
        2. Look for contract/project descriptions
        3. Look for agency + service descriptions
        4. Never use filenames (patterns like .pdf, .xlsx, timestamps)
        """
        # Patterns to try in order of preference
        patterns = [
            # Explicit title/subject fields
            r"(?:Title|Subject)\s*:\s*([^\n]+)",
            # Description fields
            r"(?:Description|Purpose)\s*:\s*([^\n]+)",
            # Contract name
            r"(?:Contract|Project)\s+(?:Name|Title)\s*:\s*([^\n]+)",
            # Solicitation FOR something
            r"SOLICITATION\s+(?:FOR|OF)\s+([^\n]+)",
            # Request for Proposal/Quote for something
            r"(?:REQUEST\s+FOR\s+(?:PROPOSAL|QUOTATION|QUOTE)|RFP|RFQ)\s+(?:FOR\s+)?([A-Z][^\n]{10,100})",
            # Fair Opportunity for something (GSA)
            r"FAIR\s+OPPORTUNITY\s+(?:FOR|TO\s+PROVIDE)\s+([^\n]+)",
            # Task Order for something
            r"TASK\s+ORDER\s+(?:FOR|TO\s+PROVIDE)\s+([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:15000], re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # Filter out filename-like values
                if self._is_valid_title(candidate):
                    return candidate[:200]

        return ""

    def _is_valid_title(self, title: str) -> bool:
        """
        Validate that a candidate title is not a filename, clause, or garbage.

        Enhanced filtering to catch:
        - Filenames and timestamps
        - FAR/regulatory clause language
        - Procedural phrases that aren't titles
        - Partial sentences (lowercase starts)
        """
        if not title or len(title) < 5:
            return False

        # Reject filenames (contain extensions)
        filename_patterns = [
            r'\.(pdf|xlsx?|docx?|csv|txt|zip|rar)(\s|$)',
            r'\.\d{10,}',  # Timestamps like .1763761025326
            r'^[A-Z0-9_-]+\.\d+',  # File IDs like "ABC123.1234567890"
        ]
        for pattern in filename_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return False

        # Reject if it's mostly numbers/special characters
        alpha_chars = sum(1 for c in title if c.isalpha())
        if alpha_chars < len(title) * 0.3:
            return False

        # Reject if doesn't start with capital letter (partial sentence)
        # Skip leading numbers/punctuation and check first meaningful alpha char
        # But allow patterns like "24x7" where lowercase is part of alphanumeric token
        title_stripped = title.strip()
        if title_stripped:
            # Find first word (sequence of alphanumeric chars)
            import re as regex
            first_word_match = regex.match(r'^[\d\W]*([a-zA-Z][\w]*)', title_stripped)
            if first_word_match:
                first_word = first_word_match.group(1)
                # If first word is all lowercase and not a common abbreviation/code, reject
                if first_word.islower() and len(first_word) > 3:
                    # Allow technical terms like "24x7" by checking if preceded by digits
                    preceded_by_digit = regex.match(r'^\d', title_stripped)
                    if not preceded_by_digit:
                        return False

        # Reject common non-title phrases (start with)
        reject_start_phrases = [
            'attachment', 'exhibit', 'appendix', 'see section',
            'amendment', 'modification', 'page of pages',
            'procedures', 'requirements', 'instructions',
            'pursuant to', 'in accordance', 'as described',
            'the contractor', 'the offeror', 'the government',
            'all offerors', 'each offeror',
            # Conjunctions and articles - titles don't start with these
            'and ', 'or ', 'but ', 'nor ', 'yet ', 'so ',
            'a ', 'an ', 'the ',
            # Prepositions
            'for ', 'to ', 'from ', 'with ', 'of ', 'in ', 'on ', 'at ', 'by ',
            # Partial sentence indicators
            'see attached', 'see also', 'refer to', 'as per',
        ]
        title_lower = title.lower().strip()
        for phrase in reject_start_phrases:
            if title_lower.startswith(phrase):
                return False

        # Reject FAR/regulatory clause language (contains)
        reject_contains = [
            'non-government advisors',
            'written objection',
            'organizational conflict',
            'far clause', 'dfar clause',
            'shall be provided',
            'must be submitted',
            'are required to',
            'in providing written',
        ]
        for phrase in reject_contains:
            if phrase in title_lower:
                return False

        # Reject if too long (likely a sentence, not a title)
        if len(title) > 150:
            return False

        return True
    
    def _find_section_boundaries(self, text: str) -> Dict[UCFSection, Tuple[int, int, str]]:
        """
        Find start/end positions of each UCF section.
        
        CRITICAL: Must distinguish between TOC entries and actual section headers.
        TOC entries look like: "SECTION L - INSTRUCTIONS ... 69"
        Real sections look like: "SECTION L - INSTRUCTIONS, CONDITIONS, AND NOTICES TO OFFERORS\nARTICLE L.1..."
        
        Returns:
            Dict mapping section to (start_offset, end_offset, title)
        """
        section_starts = []
        
        for section, patterns in self.SECTION_PATTERNS.items():
            best_match = None
            best_score = -1
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                    # Get context around the match
                    line_start = text.rfind('\n', 0, match.start()) + 1
                    line_end = text.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(text)
                    title = text[line_start:line_end].strip()
                    
                    # Get text following this match (next 800 chars)
                    following_text = text[match.end():match.end() + 800]
                    
                    # Get preceding text to check for context
                    preceding_text = text[max(0, match.start()-200):match.start()]
                    
                    # Score this match - higher score = more likely real section
                    score = 0
                    
                    # TOC entries have "...." or end with page numbers
                    if '....' in title or re.search(r'\.\s*\d+\s*$', title):
                        score -= 100  # Strongly penalize TOC entries
                        continue  # Skip TOC entries entirely
                    
                    # Check if this looks like a reference within text (not a header)
                    # e.g., "set forth in SECTION A - SOLICITATION"
                    if re.search(r'(?:in|from|see|per|under)\s+SECTION\s+' + section.value, 
                                preceding_text, re.IGNORECASE):
                        score -= 80  # This is likely a cross-reference, not a header
                    
                    # Real section headers are typically at the start of a line
                    # and followed by content
                    if match.start() == line_start or text[match.start()-1] in '\n\r':
                        score += 30
                    
                    # Real sections have ARTICLE after them (for L, M especially)
                    if re.search(r'ARTICLE\s+' + section.value + r'\.\d+', following_text, re.IGNORECASE):
                        score += 60
                    
                    # Real sections have substantial content with requirements
                    if len(following_text.strip()) > 200:
                        req_words = len(re.findall(r'\b(shall|must|offeror|contractor|proposal|submit)\b', 
                                                   following_text, re.IGNORECASE))
                        score += min(req_words * 10, 40)
                    
                    # Section-specific validation
                    if section == UCFSection.SECTION_L:
                        # Section L should have L.1, L.2, L.4 references
                        if re.search(r'L\.\d+|Factor\s+\d|Technical\s+Proposal|Business\s+Proposal', 
                                    following_text, re.IGNORECASE):
                            score += 50
                    
                    elif section == UCFSection.SECTION_M:
                        # Section M should have evaluation language
                        if re.search(r'evaluat|rating|factor|criteria|basis\s+for\s+award', 
                                    following_text, re.IGNORECASE):
                            score += 50
                    
                    elif section == UCFSection.SECTION_A:
                        # Section A (SF 33) is usually early in doc
                        doc_position = match.start() / len(text)
                        if doc_position < 0.05:  # First 5% of document
                            score += 40
                        elif doc_position > 0.3:  # Past 30% - probably a reference
                            score -= 50
                    
                    elif section == UCFSection.SECTION_B:
                        # Section B should have pricing/cost content
                        if re.search(r'price|cost|clin|line\s+item|estimated', 
                                    following_text, re.IGNORECASE):
                            score += 40
                    
                    elif section == UCFSection.SECTION_C:
                        # Section C should have SOW/PWS or description content
                        if re.search(r'statement\s+of\s+work|performance\s+work|scope|task', 
                                    following_text, re.IGNORECASE):
                            score += 40
                    
                    # Prefer matches further into the document (past TOC) for most sections
                    # But A, B, C are expected earlier
                    doc_position = match.start() / len(text)
                    if section.value in ['L', 'M']:
                        if doc_position > 0.4:  # L and M are typically in Part IV
                            score += 30
                    elif section.value in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']:
                        if 0.02 < doc_position < 0.6:  # Parts I-III
                            score += 20
                    
                    if score > best_score:
                        best_score = score
                        best_match = (match.start(), section, title)
            
            if best_match and best_score > -50:  # Only include if reasonably confident
                section_starts.append(best_match)
        
        # Sort by position
        section_starts.sort(key=lambda x: x[0])
        
        # Build boundaries (each section ends where the next begins)
        results = {}
        for i, (start, section, title) in enumerate(section_starts):
            if i + 1 < len(section_starts):
                end = section_starts[i + 1][0]
            else:
                end = len(text)
            
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
        attachment_counter = 1  # For auto-numbering unidentified attachments

        for doc in documents:
            filename = doc.get('filename', '').lower()
            original_filename = doc.get('filename', '')
            text = doc.get('text', '')
            pages = doc.get('pages', [])

            # Skip main RFP document (identified by having sections L, M, C)
            # BUT: If filename contains 'attachment', always process as attachment
            # (bundled attachments PDFs may contain UCF section references)
            is_attachment_file = 'attachment' in filename or 'exhibit' in filename
            if not is_attachment_file and self._is_main_solicitation(text):
                continue

            # Detect attachment type from filename
            att_id = None
            doc_type = "General"

            # Handle various attachment naming patterns
            if 'attachment' in filename:
                # Try specific attachment number first
                match = re.search(r'attachment\s*(\d+|[a-z])', filename, re.IGNORECASE)
                if match:
                    att_id = f"Attachment {match.group(1).upper()}"
                else:
                    # Handle "ATTACHMENTS" plural or bundle files
                    # e.g., "ATTACHMENTS RFP 75N96025R00004 less Attach 11 Excel File.pdf"
                    if 'attachments' in filename or not re.search(r'attachment\s*\d', filename):
                        att_id = f"Attachments Bundle {attachment_counter}"
                        attachment_counter += 1
            elif 'exhibit' in filename:
                match = re.search(r'exhibit\s*(\d+|[a-z])', filename, re.IGNORECASE)
                if match:
                    att_id = f"Exhibit {match.group(1).upper()}"

            # Classify document type based on filename
            if 'sow' in filename or 'statement of work' in filename.replace('_', ' '):
                doc_type = "SOW"
                if not att_id:
                    att_id = "SOW"
            elif 'pws' in filename or 'performance work' in filename.replace('_', ' '):
                doc_type = "PWS"
                if not att_id:
                    att_id = "PWS"
            elif 'budget' in filename or 'cost' in filename or 'pricing' in filename:
                doc_type = "Budget Template"
            elif 'experience' in filename or 'past performance' in filename.replace('_', ' '):
                doc_type = "Past Performance"
            elif 'resume' in filename or 'personnel' in filename:
                doc_type = "Personnel"
            elif 'amendment' in filename:
                doc_type = "Amendment"
                match = re.search(r'amendment\s*(\d+)', filename, re.IGNORECASE)
                if match:
                    att_id = f"Amendment {match.group(1)}"

            # Check content for requirements
            # For bundled attachment files, search deeper (SOW may be after cover pages/TOC)
            is_bundled = 'attachments' in filename
            req_search_range = 100000 if is_bundled else 20000
            has_requirements = bool(re.search(
                r'\b(?:shall|must|will\s+be\s+required|is\s+required)\b',
                text[:req_search_range], re.IGNORECASE
            ))

            # Detect SOW/PWS from content if not identified from filename
            # Search more of the document for bundled files (SOW may be after cover pages)
            search_range = 100000 if is_bundled else 10000
            if doc_type == "General" and has_requirements:
                if re.search(r'\b(?:statement\s+of\s+work|scope\s+of\s+work)\b', text[:search_range], re.IGNORECASE):
                    doc_type = "SOW"
                    if not att_id:
                        att_id = "SOW"
                elif re.search(r'\bperformance\s+work\s+statement\b', text[:search_range], re.IGNORECASE):
                    doc_type = "PWS"
                    if not att_id:
                        att_id = "PWS"
                elif re.search(r'\bcontractor\s+shall\b.*\bcontractor\s+shall\b', text[:100000], re.IGNORECASE | re.DOTALL):
                    # Multiple "contractor shall" statements suggest technical requirements
                    doc_type = "Technical Attachment"
                    if not att_id:
                        att_id = f"Technical Attachment {attachment_counter}"
                        attachment_counter += 1

            # SPECIAL CASE: Bundled attachment files should ALWAYS be processed
            # even if has_requirements didn't find keywords in first N chars
            # (the requirements may be deep in the document after cover pages)
            if is_bundled and doc_type == "General" and not att_id:
                doc_type = "Technical Attachment"
                att_id = f"Attachments Bundle {attachment_counter}"
                attachment_counter += 1
                # Mark as containing requirements since we KNOW bundled attachment files have them
                has_requirements = True

            # If document has requirements but no att_id yet, assign one
            if has_requirements and not att_id:
                att_id = f"Supplemental Doc {attachment_counter}"
                attachment_counter += 1

            if att_id:
                attachments[att_id] = AttachmentInfo(
                    id=att_id,
                    title=original_filename,
                    filename=original_filename,
                    content=text,
                    page_count=len(pages) if pages else 1,
                    document_type=doc_type,
                    contains_requirements=has_requirements
                )

        return attachments

    def _is_main_solicitation(self, text: str) -> bool:
        """Check if document is the main RFP (has UCF sections L, M, C structure)"""
        # Look for clear UCF section markers
        section_patterns = [
            r'\bSECTION\s+L\b.*\bINSTRUCTIONS',
            r'\bSECTION\s+M\b.*\bEVALUATION',
            r'\bSECTION\s+C\b.*\bDESCRIPTION',
            r'\bPART\s+I+\s*[-–—]\s*THE\s+SCHEDULE',
        ]

        matches = sum(1 for p in section_patterns if re.search(p, text[:50000], re.IGNORECASE))
        return matches >= 2  # At least 2 UCF section patterns
    
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
