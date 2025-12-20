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
class BoundingBox:
    """
    Normalized bounding box for visual highlighting.

    All coordinates are normalized to 0.0-1.0 range for screen-size independence.
    Origin is TOP-LEFT (web standard), not bottom-left (PDF standard).

    Used by react-pdf-highlighter and similar frontend libraries.
    """
    x: float           # Left coordinate (0.0 = left edge, 1.0 = right edge)
    y: float           # Top coordinate (0.0 = top edge, 1.0 = bottom edge)
    width: float       # Normalized width
    height: float      # Normalized height

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_pdf_coords(
        cls,
        x0: float,
        top: float,
        x1: float,
        bottom: float,
        page_width: float,
        page_height: float
    ) -> 'BoundingBox':
        """
        Create BoundingBox from pdfplumber coordinates.

        pdfplumber uses top-left origin with points, so we just normalize.

        Args:
            x0: Left edge in points
            top: Top edge in points (distance from top of page)
            x1: Right edge in points
            bottom: Bottom edge in points
            page_width: Page width in points
            page_height: Page height in points
        """
        return cls(
            x=x0 / page_width,
            y=top / page_height,
            width=(x1 - x0) / page_width,
            height=(bottom - top) / page_height
        )

    def merge_with(self, other: 'BoundingBox') -> 'BoundingBox':
        """Merge two bounding boxes into one that encompasses both"""
        new_x = min(self.x, other.x)
        new_y = min(self.y, other.y)
        new_right = max(self.x + self.width, other.x + other.width)
        new_bottom = max(self.y + self.height, other.y + other.height)

        return BoundingBox(
            x=new_x,
            y=new_y,
            width=new_right - new_x,
            height=new_bottom - new_y
        )


@dataclass
class SourceCoordinate:
    """
    Links extracted requirements to their visual source location in the PDF.

    This is the "Trust Gate" - provides mathematical proof of extraction accuracy
    by enabling visual overlay of the exact source text.

    Supports multi-line and multi-column requirements via visual_rects list.
    """
    source_document_id: str              # UUID or filename of the source PDF
    page_index: int                      # 1-based page number
    visual_rects: List[BoundingBox] = field(default_factory=list)  # Multiple rects for multi-line text

    # Optional metadata
    char_start: Optional[int] = None     # Character offset in page text
    char_end: Optional[int] = None       # End character offset
    extraction_confidence: float = 1.0   # How confident we are in the coordinate mapping

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (API response format)"""
        # Compute bounding rect (union of all visual_rects) for scrolling
        bounding_rect = None
        if self.visual_rects:
            bounding_rect = self.visual_rects[0]
            for rect in self.visual_rects[1:]:
                bounding_rect = bounding_rect.merge_with(rect)

        return {
            "source_document_id": self.source_document_id,
            "page_index": self.page_index,
            "boundingRect": bounding_rect.to_dict() if bounding_rect else None,
            "rects": [r.to_dict() for r in self.visual_rects],
            "char_start": self.char_start,
            "char_end": self.char_end,
            "extraction_confidence": self.extraction_confidence
        }

    @classmethod
    def from_text_match(
        cls,
        document_id: str,
        page_index: int,
        char_positions: List[Dict[str, float]],
        page_width: float,
        page_height: float
    ) -> 'SourceCoordinate':
        """
        Create SourceCoordinate from a list of character position dicts.

        Groups consecutive characters into line-level bounding boxes.

        Args:
            document_id: Source document identifier
            page_index: 1-based page number
            char_positions: List of dicts with x0, top, x1, bottom keys
            page_width: Page width in points
            page_height: Page height in points
        """
        if not char_positions:
            return cls(source_document_id=document_id, page_index=page_index)

        # Group characters into lines based on y-position
        lines: List[List[Dict]] = []
        current_line: List[Dict] = [char_positions[0]]
        line_tolerance = 3  # Points tolerance for same-line detection

        for char in char_positions[1:]:
            if abs(char['top'] - current_line[0]['top']) < line_tolerance:
                current_line.append(char)
            else:
                lines.append(current_line)
                current_line = [char]
        lines.append(current_line)

        # Create bounding box for each line
        visual_rects = []
        for line in lines:
            x0 = min(c['x0'] for c in line)
            x1 = max(c['x1'] for c in line)
            top = min(c['top'] for c in line)
            bottom = max(c['bottom'] for c in line)

            visual_rects.append(BoundingBox.from_pdf_coords(
                x0, top, x1, bottom, page_width, page_height
            ))

        return cls(
            source_document_id=document_id,
            page_index=page_index,
            visual_rects=visual_rects
        )


# =============================================================================
# Phase 2: Iron Triangle Logic Engine - Strategic Data Models
# =============================================================================

class ConflictSeverity(Enum):
    """Severity levels for detected conflicts"""
    CRITICAL = "critical"      # Will cause non-compliance or disqualification
    HIGH = "high"              # Significant risk to evaluation
    MEDIUM = "medium"          # Should be addressed but not blocking
    LOW = "low"                # Minor issue or suggestion


class ConflictType(Enum):
    """Types of conflicts that can be detected"""
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    MISSING_SECTION = "missing_section"
    UNADDRESSED_FACTOR = "unaddressed_factor"
    FORMAT_MISMATCH = "format_mismatch"
    CROSS_REFERENCE_BROKEN = "cross_reference_broken"
    EVALUATION_GAP = "evaluation_gap"
    INSTRUCTION_CONFLICT = "instruction_conflict"


@dataclass
class WinTheme:
    """
    A win theme represents a competitive discriminator.

    Per the tech spec: "What we have that they don't" + "How this helps the client"

    Win themes are the foundation of a compelling proposal - they answer
    "Why should the government choose us over competitors?"
    """
    theme_id: str                                    # e.g., "WT-001"
    discriminator: str                               # What makes us unique
    benefit_statement: str                           # How this helps the client
    proof_points: List[str] = field(default_factory=list)  # IDs from Company Library

    # Links to RFP
    addresses_factors: List[str] = field(default_factory=list)  # Section M factor IDs
    addresses_requirements: List[str] = field(default_factory=list)  # Requirement IDs

    # Metadata
    priority: int = 1                                # 1 = highest priority
    confidence: float = 0.8                          # How confident we are this resonates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "discriminator": self.discriminator,
            "benefit_statement": self.benefit_statement,
            "proof_points": self.proof_points,
            "addresses_factors": self.addresses_factors,
            "addresses_requirements": self.addresses_requirements,
            "priority": self.priority,
            "confidence": self.confidence
        }


@dataclass
class CompetitorProfile:
    """
    Profile of a known or likely competitor.

    Enables "ghosting" strategies - subtle critiques that highlight
    our strengths vs competitor weaknesses without naming them.
    """
    name: str                                        # Competitor name (may be "Unknown Incumbent")
    known_weaknesses: List[str] = field(default_factory=list)
    likely_solution_approach: str = ""               # How they'll probably respond
    past_performance_issues: List[str] = field(default_factory=list)

    # Ghosting opportunities
    ghost_points: List[str] = field(default_factory=list)  # Statements that highlight their weakness

    # Intelligence source
    source: str = ""                                 # Where we learned this (public, past experience, etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "known_weaknesses": self.known_weaknesses,
            "likely_solution_approach": self.likely_solution_approach,
            "past_performance_issues": self.past_performance_issues,
            "ghost_points": self.ghost_points,
            "source": self.source
        }


@dataclass
class EvaluationFactor:
    """
    A Section M evaluation factor with scoring weight.

    Captures what the government cares about and how much.
    """
    factor_id: str                                   # e.g., "M.2.1", "Factor-1"
    name: str                                        # e.g., "Technical Approach"
    description: str = ""
    weight: Optional[str] = None                     # e.g., "significantly more important than"
    weight_numeric: Optional[float] = None           # Normalized 0-1 if extractable

    # Sub-factors
    sub_factors: List['EvaluationFactor'] = field(default_factory=list)

    # Cross-references
    maps_to_section_l: List[str] = field(default_factory=list)  # L references
    maps_to_section_c: List[str] = field(default_factory=list)  # C/SOW references

    # Source tracking
    source_page: Optional[int] = None
    source_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "weight_numeric": self.weight_numeric,
            "sub_factors": [sf.to_dict() for sf in self.sub_factors],
            "maps_to_section_l": self.maps_to_section_l,
            "maps_to_section_c": self.maps_to_section_c,
            "source_page": self.source_page
        }


@dataclass
class StructureConflict:
    """
    A detected conflict between RFP sections (L, M, C).

    Examples:
    - Section L limits Volume 1 to 20 pages, but M sub-factors imply 30 pages
    - Section M references a factor not addressed in L instructions
    - Section C requirement has no corresponding L submission instruction
    """
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str

    # Involved sections
    section_l_ref: Optional[str] = None              # L.4.B.2
    section_m_ref: Optional[str] = None              # M.2.1
    section_c_ref: Optional[str] = None              # C.3.1.a

    # Details
    expected: str = ""                               # What was expected
    actual: str = ""                                 # What was found
    recommendation: str = ""                         # How to resolve

    # Source
    detected_at: str = ""                            # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "section_l_ref": self.section_l_ref,
            "section_m_ref": self.section_m_ref,
            "section_c_ref": self.section_c_ref,
            "expected": self.expected,
            "actual": self.actual,
            "recommendation": self.recommendation,
            "detected_at": self.detected_at
        }


@dataclass
class LMCCrossWalk:
    """
    Cross-reference mapping between Section L, M, and C.

    The core of the "Iron Triangle" - shows how instructions (L),
    evaluation factors (M), and work requirements (C) relate.
    """
    # Section L (Instructions)
    l_instruction_ref: str                           # e.g., "L.4.B.2"
    l_instruction_text: str = ""
    l_page_limit: Optional[int] = None
    l_volume: Optional[str] = None                   # e.g., "Volume I - Technical"

    # Section M (Evaluation)
    m_factor_refs: List[str] = field(default_factory=list)
    m_factors: List[EvaluationFactor] = field(default_factory=list)

    # Section C (SOW/PWS)
    c_requirement_refs: List[str] = field(default_factory=list)
    c_requirement_count: int = 0

    # Analysis
    coverage_score: float = 0.0                      # 0-1, how well C maps to M
    gaps: List[str] = field(default_factory=list)    # Missing mappings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l_instruction_ref": self.l_instruction_ref,
            "l_instruction_text": self.l_instruction_text,
            "l_page_limit": self.l_page_limit,
            "l_volume": self.l_volume,
            "m_factor_refs": self.m_factor_refs,
            "m_factors": [f.to_dict() for f in self.m_factors],
            "c_requirement_refs": self.c_requirement_refs,
            "c_requirement_count": self.c_requirement_count,
            "coverage_score": self.coverage_score,
            "gaps": self.gaps
        }


@dataclass
class StrategyAnalysis:
    """
    Complete strategic analysis of an RFP.

    Output of the StrategyAgent - contains all cross-walks,
    conflicts, win themes, and recommendations.
    """
    rfp_id: str
    solicitation_number: str

    # Iron Triangle mappings
    cross_walks: List[LMCCrossWalk] = field(default_factory=list)

    # Evaluation factors extracted from Section M
    evaluation_factors: List[EvaluationFactor] = field(default_factory=list)

    # Detected conflicts
    conflicts: List[StructureConflict] = field(default_factory=list)

    # Strategic elements
    win_themes: List[WinTheme] = field(default_factory=list)
    competitor_profiles: List[CompetitorProfile] = field(default_factory=list)

    # Summary metrics
    total_l_instructions: int = 0
    total_m_factors: int = 0
    total_c_requirements: int = 0
    coverage_score: float = 0.0                      # Overall L-M-C alignment
    conflict_count: int = 0
    critical_conflicts: int = 0

    # Metadata
    analyzed_at: str = ""
    analysis_version: str = "4.0.0-phase2"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rfp_id": self.rfp_id,
            "solicitation_number": self.solicitation_number,
            "cross_walks": [cw.to_dict() for cw in self.cross_walks],
            "evaluation_factors": [ef.to_dict() for ef in self.evaluation_factors],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "win_themes": [wt.to_dict() for wt in self.win_themes],
            "competitor_profiles": [cp.to_dict() for cp in self.competitor_profiles],
            "summary": {
                "total_l_instructions": self.total_l_instructions,
                "total_m_factors": self.total_m_factors,
                "total_c_requirements": self.total_c_requirements,
                "coverage_score": self.coverage_score,
                "conflict_count": self.conflict_count,
                "critical_conflicts": self.critical_conflicts
            },
            "analyzed_at": self.analyzed_at,
            "analysis_version": self.analysis_version
        }


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
