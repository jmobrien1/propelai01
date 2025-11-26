"""
PropelAI: Section L/M Parser and Proposal Outline Generator

Parses RFP Section L (Instructions to Offerors) and Section M (Evaluation Factors)
to generate a structured proposal outline with page limits, compliance checkpoints,
and evaluation criteria mapping.

Usage:
    from agents.enhanced_compliance.outline_generator import OutlineGenerator
    
    generator = OutlineGenerator()
    result = generator.process_rfp("/path/to/rfp.pdf")
    
    print(result.outline)  # Structured proposal outline
    print(result.page_budget)  # Page allocations
    print(result.eval_mapping)  # Section -> Evaluation factor mapping
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ============== Data Models ==============

class VolumeType(Enum):
    """Standard proposal volume types"""
    TECHNICAL = "technical"
    MANAGEMENT = "management"
    PAST_PERFORMANCE = "past_performance"
    COST_PRICE = "cost_price"
    SMALL_BUSINESS = "small_business"
    ADMINISTRATIVE = "administrative"
    ORAL_PRESENTATION = "oral_presentation"
    SAMPLE_TASK = "sample_task"
    STAFFING = "staffing"
    OTHER = "other"


class EvalRating(Enum):
    """Common adjectival rating scales"""
    OUTSTANDING = "outstanding"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    UNACCEPTABLE = "unacceptable"
    # Alternative scale
    EXCELLENT = "excellent"
    SATISFACTORY = "satisfactory"
    UNSATISFACTORY = "unsatisfactory"
    # Confidence ratings (Past Performance)
    SUBSTANTIAL_CONFIDENCE = "substantial_confidence"
    SATISFACTORY_CONFIDENCE = "satisfactory_confidence"
    LIMITED_CONFIDENCE = "limited_confidence"
    NO_CONFIDENCE = "no_confidence"
    NEUTRAL = "neutral"


@dataclass
class FormatRequirement:
    """Document format requirements"""
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    line_spacing: Optional[str] = None  # single, 1.5, double
    margins: Optional[str] = None  # e.g., "1 inch"
    paper_size: Optional[str] = None  # letter, A4
    headers_footers: Optional[str] = None
    page_numbering: Optional[str] = None
    file_format: Optional[str] = None  # PDF, Word
    file_naming: Optional[str] = None
    max_file_size: Optional[str] = None


@dataclass
class SubmissionRequirement:
    """Submission requirements"""
    due_date: Optional[str] = None
    due_time: Optional[str] = None
    timezone: Optional[str] = None
    submission_method: Optional[str] = None  # email, portal, physical
    submission_address: Optional[str] = None
    num_copies: Optional[int] = None
    electronic_copies: Optional[int] = None
    late_policy: Optional[str] = None


@dataclass
class ProposalSection:
    """A section in the proposal outline"""
    id: str
    title: str
    volume: VolumeType
    page_limit: Optional[int] = None
    parent_id: Optional[str] = None
    level: int = 1  # 1 = top level, 2 = subsection, etc.
    required: bool = True
    description: Optional[str] = None
    content_requirements: List[str] = field(default_factory=list)
    eval_factors: List[str] = field(default_factory=list)
    compliance_checkpoints: List[str] = field(default_factory=list)
    suggested_win_themes: List[str] = field(default_factory=list)
    order: int = 0


@dataclass
class EvaluationFactor:
    """An evaluation factor from Section M"""
    id: str
    name: str
    weight: Optional[float] = None  # Percentage or points
    weight_type: Optional[str] = None  # "percentage", "points", "relative"
    importance: Optional[str] = None  # "most important", "equal", etc.
    description: Optional[str] = None
    subfactors: List['EvaluationFactor'] = field(default_factory=list)
    rating_scale: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    order: int = 0


@dataclass
class ProposalVolume:
    """A proposal volume"""
    id: str
    name: str
    volume_type: VolumeType
    page_limit: Optional[int] = None
    sections: List[ProposalSection] = field(default_factory=list)
    order: int = 0
    description: Optional[str] = None


@dataclass
class OutlineResult:
    """Complete outline generation result"""
    volumes: List[ProposalVolume]
    eval_factors: List[EvaluationFactor]
    format_requirements: FormatRequirement
    submission_requirements: SubmissionRequirement
    total_page_limit: Optional[int] = None
    section_to_eval_mapping: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    raw_section_l: Optional[str] = None
    raw_section_m: Optional[str] = None


# ============== Pattern Definitions ==============

class SectionPatterns:
    """Regex patterns for parsing Section L/M"""
    
    # Volume identification patterns
    VOLUME_PATTERNS = [
        # "Volume I - Technical Proposal"
        (r"Volume\s+([IVX\d]+)\s*[-:–]\s*(.+?)(?=\n|$)", "numbered"),
        # "Technical Volume" or "Technical Proposal"
        (r"(Technical|Management|Cost|Price|Past\s*Performance|Administrative|Staffing)\s*(Volume|Proposal|Submission)", "named"),
        # "Factor 1: Technical Approach"
        (r"Factor\s+(\d+)\s*[-:]\s*(.+?)(?=\n|$)", "factor"),
    ]
    
    # State RFP section patterns (Illinois, etc.)
    STATE_SECTION_PATTERNS = [
        # "Section F.1: Technical Proposal"
        (r"(?:Section\s+)?([A-Z])\.(\d+)\.?\s*[-:.]?\s*([A-Z][A-Za-z\s&]+?)(?=\n|$)", "state_section"),
        # "F.1 TECHNICAL PROPOSAL"
        (r"([A-Z])\.(\d+)\s+([A-Z][A-Z\s&]+?)(?=\n|$)", "state_header"),
    ]
    
    # State RFP evaluation patterns
    STATE_EVAL_PATTERNS = [
        # "Technical Proposal Requirements (1000 points)"
        r"(Technical\s+(?:Proposal\s+)?Requirements?)\s*\(?\s*(\d+)\s*points?\)?",
        # "Commitment to Diversity (200 points)"
        r"(Commitment\s+to\s+Diversity)\s*\(?\s*(\d+)\s*points?\)?",
        # "Demonstrations (500 points)"
        r"(Demonstrations?)\s*\(?\s*(\d+)\s*points?\)?",
        # "Price Proposal (evaluated separately)"
        r"(Price\s+(?:Proposal|Evaluation))",
        # "Past Performance (300 points)"
        r"(Past\s+Performance)\s*\(?\s*(\d+)\s*points?\)?",
        # Generic: "Factor Name (N points)"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\((\d+)\s*points?\)",
    ]
    
    # Page limit patterns
    PAGE_LIMIT_PATTERNS = [
        # "not to exceed 50 pages"
        r"(?:not\s+to\s+exceed|shall\s+not\s+exceed|limited\s+to|maximum\s+of|no\s+more\s+than)\s+(\d+)\s*pages?",
        # "50 page limit"
        r"(\d+)\s*[-]?\s*page\s+limit",
        # "maximum 50 pages"
        r"maximum\s+(?:of\s+)?(\d+)\s*pages?",
        # "(50 pages)"
        r"\((\d+)\s*pages?\s*(?:max|maximum|limit)?\)",
        # "50 pages maximum"
        r"(\d+)\s*pages?\s+(?:max|maximum)",
    ]
    
    # Section header patterns
    SECTION_PATTERNS = [
        # "L.4.1 Technical Approach"
        r"([LM])\.(\d+(?:\.\d+)*)\s+(.+?)(?=\n|$)",
        # "Section L.4 - Instructions"
        r"Section\s+([LM])\.?(\d+(?:\.\d+)*)\s*[-:–]?\s*(.+?)(?=\n|$)",
        # "(a) Technical Approach" or "(1) Technical Approach"
        r"\(([a-z\d]+)\)\s+(.+?)(?=\n|$)",
    ]
    
    # Format requirement patterns
    FORMAT_PATTERNS = {
        "font": [
            r"(?:font|typeface)\s*(?:shall\s+be|must\s+be|:)?\s*(Times\s*New\s*Roman|Arial|Calibri|Courier)",
            r"(Times\s*New\s*Roman|Arial|Calibri|Courier)\s+(?:\d+\s*-?\s*point|font)",
        ],
        "font_size": [
            r"(\d+)\s*[-]?\s*point\s+(?:font|type|minimum)",
            r"(?:font\s+size|type\s+size)\s*(?:of|:)?\s*(\d+)",
            r"(?:minimum|no\s+(?:less|smaller)\s+than)\s+(\d+)\s*[-]?\s*point",
        ],
        "margins": [
            r"(?:margins?)\s*(?:of|:)?\s*(?:at\s+least\s+)?(\d+(?:\.\d+)?)\s*(?:inch|in|\")",
            r"(\d+(?:\.\d+)?)\s*(?:inch|in|\")\s+margins?",
            r"(?:minimum\s+)?(\d+(?:\.\d+)?)\s*[-]?\s*inch\s+margins?",
        ],
        "spacing": [
            r"(single|double|1\.5|one\s+and\s+a\s+half)\s*[-]?\s*spac(?:e|ing|ed)",
            r"(?:line\s+)?spac(?:e|ing)\s*(?:shall\s+be|of|:)?\s*(single|double|1\.5)",
        ],
    }
    
    # Evaluation factor patterns
    EVAL_PATTERNS = [
        # "Factor 1: Technical Approach (40 points)"
        r"Factor\s+(\d+)\s*[-:]\s*(.+?)(?:\((\d+)\s*(?:points?|%)\))?",
        # "Technical Approach - 40%"
        r"(.+?)\s*[-–]\s*(\d+)\s*%",
        # "(a) Technical Approach"
        r"\(([a-z])\)\s*(.+?)(?=\n|\(|$)",
        # "1. Technical Approach"
        r"(\d+)\.\s+(.+?)(?=\n|$)",
    ]
    
    # Weight/importance patterns
    WEIGHT_PATTERNS = [
        r"(\d+)\s*(?:points?|%|percent)",
        r"(?:weighted|worth)\s+(\d+)\s*(?:points?|%|percent)",
        r"(?:approximately|about)\s+(\d+)\s*(?:points?|%|percent)",
    ]
    
    IMPORTANCE_PATTERNS = [
        r"(?:is|are)\s+((?:more|most|equally|less)\s+important)",
        r"(descending\s+order\s+of\s+importance)",
        r"(equal\s+(?:weight|importance))",
        r"(?:listed\s+in\s+)?(order\s+of\s+(?:importance|priority))",
    ]
    
    # Rating scale patterns
    RATING_PATTERNS = [
        r"(Outstanding|Excellent|Good|Acceptable|Marginal|Unacceptable|Satisfactory|Unsatisfactory)",
        r"(Substantial\s+Confidence|Satisfactory\s+Confidence|Limited\s+Confidence|No\s+Confidence|Neutral)",
    ]
    
    # Submission patterns
    SUBMISSION_PATTERNS = {
        "due_date": [
            r"(?:due|submit|submission)\s*(?:date|by|on|no\s+later\s+than)\s*[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\w+\s+\d{1,2},?\s+\d{4})",
            r"(?:no\s+later\s+than|by)\s+(\d{1,2}:\d{2}\s*[AP]M\s+\w+\s+on\s+.+?\d{4})",
        ],
        "method": [
            r"(?:submit|submission)\s+(?:via|through|to)\s+(email|portal|electronic|mail|hand.?deliver)",
            r"(electronic\s+submission|email\s+submission)",
        ],
    }


# ============== Section L Parser ==============

class SectionLParser:
    """Parse Section L (Instructions to Offerors)"""
    
    def __init__(self):
        self.patterns = SectionPatterns()
    
    def parse(self, text: str) -> Tuple[List[ProposalVolume], FormatRequirement, SubmissionRequirement]:
        """Parse Section L text and extract structure"""
        
        volumes = []
        format_req = FormatRequirement()
        submission_req = SubmissionRequirement()
        
        # Extract format requirements
        format_req = self._extract_format_requirements(text)
        
        # Extract submission requirements
        submission_req = self._extract_submission_requirements(text)
        
        # Extract volume structure
        volumes = self._extract_volumes(text)
        
        # If no volumes found, try to infer from content
        if not volumes:
            volumes = self._infer_volumes(text)
        
        # Extract page limits and apply to volumes/sections
        self._apply_page_limits(text, volumes)
        
        # Extract content requirements per section
        self._extract_content_requirements(text, volumes)
        
        return volumes, format_req, submission_req
    
    def _extract_format_requirements(self, text: str) -> FormatRequirement:
        """Extract document format requirements"""
        req = FormatRequirement()
        
        # Font
        for pattern in self.patterns.FORMAT_PATTERNS["font"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                req.font_name = match.group(1).strip()
                break
        
        # Font size
        for pattern in self.patterns.FORMAT_PATTERNS["font_size"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                req.font_size = int(match.group(1))
                break
        
        # Margins
        for pattern in self.patterns.FORMAT_PATTERNS["margins"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                req.margins = f"{match.group(1)} inch"
                break
        
        # Line spacing
        for pattern in self.patterns.FORMAT_PATTERNS["spacing"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                req.line_spacing = match.group(1).lower()
                break
        
        # File format
        if re.search(r"PDF\s+format|\.pdf|submit.*PDF", text, re.IGNORECASE):
            req.file_format = "PDF"
        elif re.search(r"Word\s+format|\.docx?|Microsoft\s+Word", text, re.IGNORECASE):
            req.file_format = "Word"
        
        return req
    
    def _extract_submission_requirements(self, text: str) -> SubmissionRequirement:
        """Extract submission requirements"""
        req = SubmissionRequirement()
        
        # Due date
        for pattern in self.patterns.SUBMISSION_PATTERNS["due_date"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                req.due_date = match.group(1).strip()
                break
        
        # Method
        for pattern in self.patterns.SUBMISSION_PATTERNS["method"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                req.submission_method = match.group(1).strip().lower()
                break
        
        # Number of copies
        copies_match = re.search(r"(\d+)\s+(?:hard\s+)?cop(?:y|ies)", text, re.IGNORECASE)
        if copies_match:
            req.num_copies = int(copies_match.group(1))
        
        return req
    
    def _extract_volumes(self, text: str) -> List[ProposalVolume]:
        """Extract proposal volumes from text"""
        volumes = []
        volume_map = {}
        
        # First try state RFP format
        state_volumes = self._extract_state_volumes(text)
        if state_volumes:
            return state_volumes
        
        for pattern, pattern_type in self.patterns.VOLUME_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if pattern_type == "numbered":
                    vol_num = match.group(1)
                    vol_name = match.group(2).strip()
                elif pattern_type == "named":
                    vol_num = str(len(volumes) + 1)
                    vol_name = match.group(0).strip()
                else:
                    vol_num = match.group(1)
                    vol_name = match.group(2).strip()
                
                vol_type = self._classify_volume(vol_name)
                vol_id = f"VOL-{vol_num}"
                
                if vol_id not in volume_map:
                    volume = ProposalVolume(
                        id=vol_id,
                        name=vol_name,
                        volume_type=vol_type,
                        order=len(volumes)
                    )
                    volumes.append(volume)
                    volume_map[vol_id] = volume
        
        return volumes
    
    def _extract_state_volumes(self, text: str) -> List[ProposalVolume]:
        """Extract volumes from state RFP format (IL, etc.)"""
        volumes = []
        seen = set()
        
        # Look for section headers that define proposal parts
        # Pattern: "F.1 TECHNICAL PROPOSAL" or "Section F.1: Technical Proposal"
        section_pattern = r"(?:Section\s+)?([A-Z])\.(\d+)[.:]?\s*([A-Z][A-Za-z\s&]+?)(?:\n|$)"
        
        for match in re.finditer(section_pattern, text):
            letter = match.group(1)
            num = match.group(2)
            name = match.group(3).strip()
            
            # Focus on F sections (evaluation) and related
            if letter not in ['F', 'G', 'E']:
                continue
            
            name_lower = name.lower()
            
            # Skip duplicates and non-proposal sections
            if name_lower in seen:
                continue
            
            # Only include proposal-related sections
            proposal_keywords = ['technical', 'price', 'cost', 'proposal', 'diversity', 
                                'demonstration', 'experience', 'reference', 'pricing']
            
            if not any(kw in name_lower for kw in proposal_keywords):
                continue
            
            seen.add(name_lower)
            vol_type = self._classify_volume(name)
            
            volume = ProposalVolume(
                id=f"VOL-{letter}{num}",
                name=name.title(),
                volume_type=vol_type,
                order=len(volumes)
            )
            volumes.append(volume)
        
        # If we didn't find explicit sections, look for mentions
        if not volumes:
            volume_indicators = [
                ("Technical Proposal", VolumeType.TECHNICAL),
                ("Technical Requirements", VolumeType.TECHNICAL),
                ("Price Proposal", VolumeType.COST_PRICE),
                ("Pricing", VolumeType.COST_PRICE),
                ("Commitment to Diversity", VolumeType.SMALL_BUSINESS),
                ("Past Performance", VolumeType.PAST_PERFORMANCE),
                ("References", VolumeType.PAST_PERFORMANCE),
                ("Demonstrations", VolumeType.ORAL_PRESENTATION),
                ("Staffing", VolumeType.STAFFING),
                ("Key Personnel", VolumeType.STAFFING),
            ]
            
            text_lower = text.lower()
            for name, vol_type in volume_indicators:
                if name.lower() in text_lower and name.lower() not in seen:
                    seen.add(name.lower())
                    volume = ProposalVolume(
                        id=f"VOL-{len(volumes)+1}",
                        name=name,
                        volume_type=vol_type,
                        order=len(volumes)
                    )
                    volumes.append(volume)
        
        return volumes
    
    def _infer_volumes(self, text: str) -> List[ProposalVolume]:
        """Infer volume structure when not explicitly stated"""
        volumes = []
        
        # Look for common volume indicators
        volume_indicators = [
            ("technical", VolumeType.TECHNICAL, ["technical approach", "technical proposal", "technical volume"]),
            ("management", VolumeType.MANAGEMENT, ["management approach", "management proposal", "project management"]),
            ("past_performance", VolumeType.PAST_PERFORMANCE, ["past performance", "experience", "references"]),
            ("cost", VolumeType.COST_PRICE, ["cost proposal", "price proposal", "pricing", "cost volume"]),
            ("staffing", VolumeType.STAFFING, ["staffing", "personnel", "key personnel", "resumes"]),
        ]
        
        text_lower = text.lower()
        order = 0
        
        for vol_id, vol_type, indicators in volume_indicators:
            for indicator in indicators:
                if indicator in text_lower:
                    # Check if there's a page limit nearby
                    volume = ProposalVolume(
                        id=f"VOL-{vol_id.upper()}",
                        name=indicator.title(),
                        volume_type=vol_type,
                        order=order
                    )
                    volumes.append(volume)
                    order += 1
                    break
        
        return volumes
    
    def _classify_volume(self, name: str) -> VolumeType:
        """Classify volume type from name"""
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in ["technical", "approach", "solution"]):
            return VolumeType.TECHNICAL
        elif any(kw in name_lower for kw in ["management", "project", "program"]):
            return VolumeType.MANAGEMENT
        elif any(kw in name_lower for kw in ["past performance", "experience", "reference"]):
            return VolumeType.PAST_PERFORMANCE
        elif any(kw in name_lower for kw in ["cost", "price", "pricing"]):
            return VolumeType.COST_PRICE
        elif any(kw in name_lower for kw in ["small business", "subcontract"]):
            return VolumeType.SMALL_BUSINESS
        elif any(kw in name_lower for kw in ["admin", "certif", "represent"]):
            return VolumeType.ADMINISTRATIVE
        elif any(kw in name_lower for kw in ["oral", "presentation"]):
            return VolumeType.ORAL_PRESENTATION
        elif any(kw in name_lower for kw in ["staff", "personnel", "resume"]):
            return VolumeType.STAFFING
        elif any(kw in name_lower for kw in ["sample", "task", "scenario"]):
            return VolumeType.SAMPLE_TASK
        else:
            return VolumeType.OTHER
    
    def _apply_page_limits(self, text: str, volumes: List[ProposalVolume]):
        """Extract and apply page limits to volumes and sections"""
        
        # Look for page limits in context
        for pattern in self.patterns.PAGE_LIMIT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                page_limit = int(match.group(1))
                
                # Find surrounding context to determine what the limit applies to
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 50)
                context = text[start:end].lower()
                
                # Try to match to a volume
                for volume in volumes:
                    vol_name_lower = volume.name.lower()
                    if vol_name_lower in context or volume.volume_type.value in context:
                        volume.page_limit = page_limit
                        break
    
    def _extract_content_requirements(self, text: str, volumes: List[ProposalVolume]):
        """Extract content requirements for each volume/section"""
        
        # Common content requirement indicators
        requirement_patterns = [
            r"(?:shall|must|should)\s+(?:include|provide|describe|demonstrate|address)\s+(.+?)(?:\.|;|\n)",
            r"(?:offeror|contractor|vendor)\s+(?:shall|must|should)\s+(.+?)(?:\.|;|\n)",
            r"(?:proposal|submission)\s+(?:shall|must|should)\s+(?:include|contain)\s+(.+?)(?:\.|;|\n)",
        ]
        
        for volume in volumes:
            vol_name_lower = volume.name.lower()
            
            # Find text sections related to this volume
            # Look for the volume name and extract requirements nearby
            for pattern in requirement_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start = max(0, match.start() - 300)
                    context = text[start:match.start()].lower()
                    
                    if vol_name_lower in context or volume.volume_type.value in context:
                        req_text = match.group(1).strip()
                        if len(req_text) > 20 and len(req_text) < 500:
                            # Create a section for this requirement
                            section = ProposalSection(
                                id=f"{volume.id}-SEC-{len(volume.sections)+1:02d}",
                                title=self._extract_section_title(req_text),
                                volume=volume.volume_type,
                                level=2,
                                content_requirements=[req_text],
                                order=len(volume.sections)
                            )
                            volume.sections.append(section)
    
    def _extract_section_title(self, requirement_text: str) -> str:
        """Extract a section title from requirement text"""
        # Take first few words as title
        words = requirement_text.split()[:6]
        title = " ".join(words)
        if len(title) > 50:
            title = title[:47] + "..."
        return title.title()


# ============== Section M Parser ==============

class SectionMParser:
    """Parse Section M (Evaluation Factors)"""
    
    def __init__(self):
        self.patterns = SectionPatterns()
    
    def parse(self, text: str) -> List[EvaluationFactor]:
        """Parse Section M text and extract evaluation factors"""
        
        factors = []
        
        # Try different patterns to extract factors
        factors = self._extract_factors(text)
        
        # Extract weights/importance
        self._extract_weights(text, factors)
        
        # Extract rating scales
        self._extract_rating_scales(text, factors)
        
        # Extract subfactors
        self._extract_subfactors(text, factors)
        
        return factors
    
    def _extract_factors(self, text: str) -> List[EvaluationFactor]:
        """Extract main evaluation factors"""
        factors = []
        seen_names = set()
        
        # First, try state RFP patterns (more specific)
        state_factors = self._extract_state_factors(text)
        if state_factors:
            return state_factors
        
        for pattern in self.patterns.EVAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                
                if len(groups) >= 2:
                    factor_id = groups[0]
                    factor_name = groups[1].strip()
                    
                    # Clean up factor name
                    factor_name = re.sub(r"\s+", " ", factor_name)
                    factor_name = factor_name.rstrip(".:;,")
                    
                    # Skip if too short or already seen
                    if len(factor_name) < 5 or factor_name.lower() in seen_names:
                        continue
                    
                    seen_names.add(factor_name.lower())
                    
                    weight = None
                    if len(groups) >= 3 and groups[2]:
                        try:
                            weight = float(groups[2])
                        except:
                            pass
                    
                    factor = EvaluationFactor(
                        id=f"EVAL-{factor_id}",
                        name=factor_name,
                        weight=weight,
                        order=len(factors)
                    )
                    factors.append(factor)
        
        # If no factors found, try to infer from common terms
        if not factors:
            factors = self._infer_factors(text)
        
        return factors
    
    def _extract_state_factors(self, text: str) -> List[EvaluationFactor]:
        """Extract evaluation factors from state RFP format"""
        factors = []
        seen = set()
        
        # State-specific patterns
        state_patterns = [
            # "Technical Proposal Requirements (1000 points)" or just mentions
            (r"(Technical\s+(?:Proposal\s+)?Requirements?)\s*\(?(\d+)\s*points?\)?", "technical"),
            (r"(Technical\s+Proposal)\s+(?:is\s+)?(?:worth|valued\s+at|=)\s*(\d+)\s*points?", "technical"),
            # Commitment to Diversity
            (r"(Commitment\s+to\s+Diversity)\s*\(?(\d+)\s*points?\)?", "diversity"),
            (r"(Commitment\s+to\s+Diversity)\s+(?:is\s+)?(?:worth|valued|=)\s*(\d+)", "diversity"),
            # Demonstrations
            (r"(Demonstrations?)\s*\(?(\d+)\s*points?\)?", "demo"),
            # Price
            (r"(Price\s+(?:Proposal|Evaluation))\s*\(?(\d+)\s*points?\)?", "price"),
            # Past Performance
            (r"(Past\s+Performance)\s*\(?(\d+)\s*points?\)?", "past_perf"),
            # Generic pattern
            (r"([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,3})\s+\((\d+)\s*points?\)", "generic"),
        ]
        
        for pattern, factor_type in state_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                name_lower = name.lower()
                
                # Skip duplicates
                if name_lower in seen:
                    continue
                
                # Skip noise (too short, generic words)
                if len(name) < 8 or name_lower in ['the', 'and', 'for', 'with']:
                    continue
                
                try:
                    weight = float(match.group(2)) if len(match.groups()) > 1 else None
                except:
                    weight = None
                
                seen.add(name_lower)
                
                factor = EvaluationFactor(
                    id=f"EVAL-{len(factors)+1}",
                    name=name,
                    weight=weight,
                    weight_type="points" if weight else None,
                    order=len(factors)
                )
                factors.append(factor)
        
        # Also look for evaluation sections by header
        section_patterns = [
            r"(?:Section\s+)?F\.(\d+)[.:]?\s*([A-Z][A-Z\s]+?)(?:\n|$)",
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                sec_num = match.group(1)
                sec_name = match.group(2).strip()
                name_lower = sec_name.lower()
                
                # Only add evaluation-related sections
                eval_keywords = ['technical', 'price', 'cost', 'diversity', 'demonstration', 
                                'experience', 'past performance', 'reference']
                
                if any(kw in name_lower for kw in eval_keywords) and name_lower not in seen:
                    seen.add(name_lower)
                    factor = EvaluationFactor(
                        id=f"EVAL-F{sec_num}",
                        name=sec_name.title(),
                        order=len(factors)
                    )
                    factors.append(factor)
        
        return factors
    
    def _infer_factors(self, text: str) -> List[EvaluationFactor]:
        """Infer evaluation factors from common terms"""
        factors = []
        text_lower = text.lower()
        
        common_factors = [
            ("Technical Approach", ["technical approach", "technical capability", "technical merit"]),
            ("Management Approach", ["management approach", "management capability", "project management"]),
            ("Past Performance", ["past performance", "relevant experience", "performance history"]),
            ("Price/Cost", ["price", "cost", "cost realism", "price reasonableness"]),
            ("Small Business", ["small business", "subcontracting plan"]),
            ("Staffing", ["staffing", "key personnel", "personnel qualifications"]),
        ]
        
        order = 0
        for factor_name, indicators in common_factors:
            for indicator in indicators:
                if indicator in text_lower:
                    factor = EvaluationFactor(
                        id=f"EVAL-{order+1}",
                        name=factor_name,
                        order=order
                    )
                    factors.append(factor)
                    order += 1
                    break
        
        return factors
    
    def _extract_weights(self, text: str, factors: List[EvaluationFactor]):
        """Extract weights and importance for factors"""
        
        # Check for relative importance statements
        importance_match = None
        for pattern in self.patterns.IMPORTANCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                importance_match = match.group(1).lower()
                break
        
        if importance_match:
            if "descending" in importance_match or "order of importance" in importance_match:
                # First factor is most important
                for i, factor in enumerate(factors):
                    if i == 0:
                        factor.importance = "most important"
                    else:
                        factor.importance = f"less important than {factors[0].name}"
            elif "equal" in importance_match:
                for factor in factors:
                    factor.importance = "equal weight"
        
        # Try to extract specific weights
        for factor in factors:
            factor_pattern = re.escape(factor.name) + r".{0,50}?(\d+)\s*(?:points?|%|percent)"
            match = re.search(factor_pattern, text, re.IGNORECASE)
            if match:
                factor.weight = float(match.group(1))
                if "%" in text[match.start():match.end()+5] or "percent" in text[match.start():match.end()+10].lower():
                    factor.weight_type = "percentage"
                else:
                    factor.weight_type = "points"
    
    def _extract_rating_scales(self, text: str, factors: List[EvaluationFactor]):
        """Extract rating scales used for evaluation"""
        
        ratings = []
        for pattern in self.patterns.RATING_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                rating = match.group(1).strip()
                if rating not in ratings:
                    ratings.append(rating)
        
        # Apply to all factors (typically same scale used)
        for factor in factors:
            factor.rating_scale = ratings.copy()
    
    def _extract_subfactors(self, text: str, factors: List[EvaluationFactor]):
        """Extract subfactors for each main factor"""
        
        for factor in factors:
            # Look for subfactors after the main factor
            factor_pattern = re.escape(factor.name)
            match = re.search(factor_pattern, text, re.IGNORECASE)
            
            if match:
                # Get text after the factor (next 2000 chars)
                start = match.end()
                end = min(len(text), start + 2000)
                context = text[start:end]
                
                # Look for lettered or numbered subfactors
                subfactor_pattern = r"\(([a-z\d])\)\s+(.+?)(?=\n|\(|$)"
                
                for sub_match in re.finditer(subfactor_pattern, context):
                    sub_id = sub_match.group(1)
                    sub_name = sub_match.group(2).strip()
                    sub_name = sub_name.rstrip(".:;,")
                    
                    if len(sub_name) > 5 and len(sub_name) < 200:
                        subfactor = EvaluationFactor(
                            id=f"{factor.id}-{sub_id}",
                            name=sub_name,
                            parent_id=factor.id,
                            order=len(factor.subfactors)
                        )
                        factor.subfactors.append(subfactor)


# ============== Outline Generator ==============

class OutlineGenerator:
    """Generate proposal outline from parsed Section L/M"""
    
    def __init__(self):
        self.section_l_parser = SectionLParser()
        self.section_m_parser = SectionMParser()
    
    def process_rfp(self, file_path: str = None, text: str = None) -> OutlineResult:
        """Process RFP and generate outline"""
        
        if file_path:
            text = self._load_document(file_path)
        
        if not text:
            raise ValueError("No text provided")
        
        # Extract Section L and Section M
        section_l_text = self._extract_section(text, "L")
        section_m_text = self._extract_section(text, "M")
        
        # If sections not found, use full text
        if not section_l_text:
            section_l_text = text
        if not section_m_text:
            section_m_text = text
        
        # Parse sections
        volumes, format_req, submission_req = self.section_l_parser.parse(section_l_text)
        eval_factors = self.section_m_parser.parse(section_m_text)
        
        # Map evaluation factors to sections
        section_to_eval = self._map_factors_to_sections(volumes, eval_factors)
        
        # Generate compliance checkpoints
        self._generate_checkpoints(volumes, eval_factors)
        
        # Calculate total page limit
        total_pages = sum(v.page_limit or 0 for v in volumes)
        
        # Generate warnings
        warnings = self._generate_warnings(volumes, eval_factors, format_req, submission_req)
        
        return OutlineResult(
            volumes=volumes,
            eval_factors=eval_factors,
            format_requirements=format_req,
            submission_requirements=submission_req,
            total_page_limit=total_pages if total_pages > 0 else None,
            section_to_eval_mapping=section_to_eval,
            warnings=warnings,
            raw_section_l=section_l_text[:5000] if section_l_text else None,
            raw_section_m=section_m_text[:5000] if section_m_text else None
        )
    
    def _load_document(self, file_path: str) -> str:
        """Load document from file"""
        path = Path(file_path)
        
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except Exception as e:
                raise ValueError(f"Failed to parse PDF: {e}")
        
        elif path.suffix.lower() in [".docx", ".doc"]:
            try:
                from docx import Document
                doc = Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                return text
            except Exception as e:
                raise ValueError(f"Failed to parse DOCX: {e}")
        
        elif path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _extract_section(self, text: str, section: str) -> Optional[str]:
        """Extract Section L or M from full RFP text"""
        
        # Pattern to find section start
        start_patterns = [
            rf"SECTION\s+{section}\s*[-:–]",
            rf"Section\s+{section}\s*[-:–]",
            rf"\n{section}\.\d",
        ]
        
        # Pattern to find section end (next section)
        next_section = chr(ord(section) + 1)  # L -> M, M -> N
        end_patterns = [
            rf"SECTION\s+{next_section}\s*[-:–]",
            rf"Section\s+{next_section}\s*[-:–]",
            rf"\n{next_section}\.\d",
        ]
        
        start_pos = None
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                break
        
        if start_pos is None:
            return None
        
        end_pos = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text[start_pos:], re.IGNORECASE)
            if match:
                end_pos = start_pos + match.start()
                break
        
        return text[start_pos:end_pos]
    
    def _map_factors_to_sections(
        self, 
        volumes: List[ProposalVolume], 
        eval_factors: List[EvaluationFactor]
    ) -> Dict[str, List[str]]:
        """Map evaluation factors to proposal sections"""
        mapping = {}
        
        # Simple mapping based on volume type
        type_to_factor = {
            VolumeType.TECHNICAL: ["technical", "approach", "solution", "capability"],
            VolumeType.MANAGEMENT: ["management", "project", "program", "risk"],
            VolumeType.PAST_PERFORMANCE: ["past performance", "experience", "reference"],
            VolumeType.COST_PRICE: ["cost", "price", "pricing"],
            VolumeType.STAFFING: ["staff", "personnel", "key personnel"],
        }
        
        for volume in volumes:
            keywords = type_to_factor.get(volume.volume_type, [])
            
            matched_factors = []
            for factor in eval_factors:
                factor_lower = factor.name.lower()
                if any(kw in factor_lower for kw in keywords):
                    matched_factors.append(factor.id)
                    
                    # Also add to sections within volume
                    for section in volume.sections:
                        section.eval_factors.append(factor.id)
            
            mapping[volume.id] = matched_factors
        
        return mapping
    
    def _generate_checkpoints(
        self, 
        volumes: List[ProposalVolume], 
        eval_factors: List[EvaluationFactor]
    ):
        """Generate compliance checkpoints for sections"""
        
        for volume in volumes:
            for section in volume.sections:
                checkpoints = []
                
                # Add checkpoints based on content requirements
                for req in section.content_requirements:
                    checkpoints.append(f"☐ Address: {req[:100]}...")
                
                # Add checkpoints based on eval factors
                for factor_id in section.eval_factors:
                    factor = next((f for f in eval_factors if f.id == factor_id), None)
                    if factor:
                        checkpoints.append(f"☐ Demonstrate: {factor.name}")
                        for subfactor in factor.subfactors:
                            checkpoints.append(f"  ☐ Address subfactor: {subfactor.name}")
                
                section.compliance_checkpoints = checkpoints
    
    def _generate_warnings(
        self,
        volumes: List[ProposalVolume],
        eval_factors: List[EvaluationFactor],
        format_req: FormatRequirement,
        submission_req: SubmissionRequirement
    ) -> List[str]:
        """Generate warnings about potential issues"""
        warnings = []
        
        # Check for missing page limits
        volumes_without_limits = [v for v in volumes if v.page_limit is None]
        if volumes_without_limits:
            names = ", ".join(v.name for v in volumes_without_limits)
            warnings.append(f"⚠️ No page limit found for: {names}")
        
        # Check for unmapped eval factors
        mapped_factors = set()
        for volume in volumes:
            for section in volume.sections:
                mapped_factors.update(section.eval_factors)
        
        unmapped = [f for f in eval_factors if f.id not in mapped_factors]
        if unmapped:
            names = ", ".join(f.name for f in unmapped)
            warnings.append(f"⚠️ Evaluation factors not mapped to sections: {names}")
        
        # Check for missing format requirements
        if not format_req.font_name:
            warnings.append("⚠️ Font requirement not found - verify in RFP")
        if not format_req.margins:
            warnings.append("⚠️ Margin requirement not found - verify in RFP")
        
        # Check for missing due date
        if not submission_req.due_date:
            warnings.append("⚠️ Due date not found - verify in RFP")
        
        return warnings
    
    def generate_markdown_outline(self, result: OutlineResult) -> str:
        """Generate markdown-formatted outline"""
        
        lines = []
        lines.append("# Proposal Outline")
        lines.append("")
        
        # Submission info
        if result.submission_requirements.due_date:
            lines.append(f"**Due Date:** {result.submission_requirements.due_date}")
        if result.submission_requirements.submission_method:
            lines.append(f"**Submission:** {result.submission_requirements.submission_method}")
        if result.total_page_limit:
            lines.append(f"**Total Page Limit:** {result.total_page_limit} pages")
        lines.append("")
        
        # Format requirements
        lines.append("## Format Requirements")
        lines.append("")
        fr = result.format_requirements
        if fr.font_name:
            lines.append(f"- **Font:** {fr.font_name}")
        if fr.font_size:
            lines.append(f"- **Size:** {fr.font_size} pt")
        if fr.margins:
            lines.append(f"- **Margins:** {fr.margins}")
        if fr.line_spacing:
            lines.append(f"- **Spacing:** {fr.line_spacing}")
        if fr.file_format:
            lines.append(f"- **Format:** {fr.file_format}")
        lines.append("")
        
        # Evaluation factors
        lines.append("## Evaluation Factors")
        lines.append("")
        for factor in result.eval_factors:
            weight_str = ""
            if factor.weight:
                weight_str = f" ({factor.weight}{factor.weight_type or '%'})"
            elif factor.importance:
                weight_str = f" - {factor.importance}"
            
            lines.append(f"### {factor.name}{weight_str}")
            
            if factor.subfactors:
                for sub in factor.subfactors:
                    lines.append(f"- {sub.name}")
            
            if factor.rating_scale:
                lines.append(f"- Rating Scale: {', '.join(factor.rating_scale)}")
            
            lines.append("")
        
        # Volumes and sections
        lines.append("## Proposal Structure")
        lines.append("")
        
        for volume in sorted(result.volumes, key=lambda v: v.order):
            page_str = f" ({volume.page_limit} pages)" if volume.page_limit else ""
            lines.append(f"### {volume.name}{page_str}")
            lines.append("")
            
            # Mapped eval factors
            if volume.id in result.section_to_eval_mapping:
                factor_ids = result.section_to_eval_mapping[volume.id]
                factor_names = [f.name for f in result.eval_factors if f.id in factor_ids]
                if factor_names:
                    lines.append(f"*Evaluation Factors:* {', '.join(factor_names)}")
                    lines.append("")
            
            # Sections
            for section in sorted(volume.sections, key=lambda s: s.order):
                page_str = f" ({section.page_limit} pages)" if section.page_limit else ""
                lines.append(f"#### {section.title}{page_str}")
                
                if section.content_requirements:
                    lines.append("")
                    lines.append("**Content Requirements:**")
                    for req in section.content_requirements[:3]:
                        lines.append(f"- {req[:200]}...")
                
                if section.compliance_checkpoints:
                    lines.append("")
                    lines.append("**Compliance Checklist:**")
                    for checkpoint in section.compliance_checkpoints[:5]:
                        lines.append(checkpoint)
                
                lines.append("")
        
        # Warnings
        if result.warnings:
            lines.append("## ⚠️ Warnings")
            lines.append("")
            for warning in result.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_json_outline(self, result: OutlineResult) -> Dict:
        """Generate JSON-formatted outline"""
        
        return {
            "submission": {
                "due_date": result.submission_requirements.due_date,
                "due_time": result.submission_requirements.due_time,
                "method": result.submission_requirements.submission_method,
                "copies": result.submission_requirements.num_copies
            },
            "format": {
                "font": result.format_requirements.font_name,
                "font_size": result.format_requirements.font_size,
                "margins": result.format_requirements.margins,
                "spacing": result.format_requirements.line_spacing,
                "file_format": result.format_requirements.file_format
            },
            "total_page_limit": result.total_page_limit,
            "evaluation_factors": [
                {
                    "id": f.id,
                    "name": f.name,
                    "weight": f.weight,
                    "weight_type": f.weight_type,
                    "importance": f.importance,
                    "rating_scale": f.rating_scale,
                    "subfactors": [
                        {"id": sf.id, "name": sf.name}
                        for sf in f.subfactors
                    ]
                }
                for f in result.eval_factors
            ],
            "volumes": [
                {
                    "id": v.id,
                    "name": v.name,
                    "type": v.volume_type.value,
                    "page_limit": v.page_limit,
                    "eval_factors": result.section_to_eval_mapping.get(v.id, []),
                    "sections": [
                        {
                            "id": s.id,
                            "title": s.title,
                            "page_limit": s.page_limit,
                            "content_requirements": s.content_requirements,
                            "eval_factors": s.eval_factors,
                            "compliance_checkpoints": s.compliance_checkpoints
                        }
                        for s in v.sections
                    ]
                }
                for v in sorted(result.volumes, key=lambda x: x.order)
            ],
            "warnings": result.warnings
        }


# ============== Convenience Functions ==============

def parse_rfp_outline(file_path: str) -> OutlineResult:
    """Parse RFP and generate proposal outline"""
    generator = OutlineGenerator()
    return generator.process_rfp(file_path)


def parse_rfp_outline_from_text(text: str) -> OutlineResult:
    """Parse RFP text and generate proposal outline"""
    generator = OutlineGenerator()
    return generator.process_rfp(text=text)
