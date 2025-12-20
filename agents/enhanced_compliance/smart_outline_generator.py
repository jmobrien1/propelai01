"""
PropelAI: Smart Proposal Outline Generator v3.3

Generates proposal outlines from already-extracted compliance matrix data.
Unlike the legacy outline_generator.py, this uses the structured requirements
already parsed from Section L and M rather than re-parsing PDFs.

v3.3 Changes (2025-12-19): STRICT CONSTRUCTIONIST ARCHITECTURE
=================================================================
Major refactor to eliminate phantom volume hallucination. The system now
uses a "Hierarchy of Authority" approach - it starts with ZERO volumes
and only creates what is explicitly evidenced in the source documents.

NEW: ProposalStructureParser class
- Dedicated parser for Section L volume tables/lists
- Explicit table parsing: "Volume | Title | Page Limit"
- List parsing: "Volume 1: Technical", "Vol. 1 - Technical"
- RFP type detection to disable UCF defaults for Task Orders, RFEs, RFQs

NEW: Evidence Tracking
- EvidenceSource enum: TABLE, LIST, FACTOR, FILENAME, KEYWORD, DEFAULT
- ConfidenceLevel enum: HIGH, MEDIUM, LOW
- ProposalVolume now tracks evidence_source and confidence

HIERARCHY OF AUTHORITY (in _extract_volumes):
1. PRIMARY: Section L explicit tables/lists (HIGH confidence)
2. SECONDARY: Section M evaluation factors (MEDIUM confidence)
3. TERTIARY: Uploaded file names (MEDIUM confidence)
4. FALLBACK: Single generic volume, NOT 4-volume UCF (LOW confidence)

DEPRECATED: _create_default_volumes
- No longer creates phantom volumes
- Returns empty list to prevent hallucination
- All fallback logic now in _extract_volumes

v3.2 Changes (2025-12-19):
- FIX PHANTOM VOLUMES: _extract_standard_volumes now requires EXPLICIT volume
  definitions (e.g., "Volume 1: Technical"), not keyword matching
- FIX SF1449 BUG: Sub-factor patterns now limit to 1-2 digits
- FIX "1 PAGE" BUG: Page limit patterns require "page" keyword

v3.1 Changes (2025-12-19):
- STRUCTURE FIX: Outline now follows Section L hierarchy, NOT SOW structure
- Section L defines PROPOSAL structure (sub-factors, volumes, headings)
- SOW requirements are MAPPED to Section L-defined sections
- New _extract_l_defined_sections: Parses "Sub Factor N:" patterns from Section L
- New _extract_page_limits_from_l: Extracts page limits from tables/text
- New _extract_eval_weights_from_m: Extracts "SF1 > SF2" formulas from Section M
- New _map_requirements_to_section: Semantic mapping of SOW reqs to L sections
- Filters boilerplate (Revision History, Change Control) from extraction

v3.0 Changes (2025-12-17):
- CRITICAL FIX: Populate volume sections with actual requirements
- Root cause of generic placeholder output was that sections were empty
- New _populate_sections_from_requirements method groups requirements by section
- Now properly passes requirements and eval_criteria to JS exporter

v2.11 Changes (2025-12-15):
- Fix cross-contamination: Remove NIH-specific default factors
- Improve page limit extraction with better patterns
- Better volume structure detection from actual RFP text
- Remove hardcoded defaults that leak between RFPs

Key improvements:
- Uses compliance matrix data (already parsed and categorized)
- Detects RFP format (NIH Factor-based, GSA Volume-based, State RFP, etc.)
- Better volume/section detection using actual requirement text
- Proper evaluation factor mapping
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============== Data Models ==============

class VolumeType(Enum):
    TECHNICAL = "technical"
    MANAGEMENT = "management"
    PAST_PERFORMANCE = "past_performance"
    EXPERIENCE = "experience"
    COST_PRICE = "cost_price"
    STAFFING = "staffing"
    SMALL_BUSINESS = "small_business"
    ADMINISTRATIVE = "administrative"
    ORAL_PRESENTATION = "oral_presentation"
    OTHER = "other"


@dataclass
class ProposalSection:
    """A section within a proposal volume"""
    id: str
    name: str
    page_limit: Optional[int] = None
    requirements: List[str] = field(default_factory=list)
    eval_criteria: List[str] = field(default_factory=list)
    subsections: List['ProposalSection'] = field(default_factory=list)


class EvidenceSource(Enum):
    """v3.3: Track where volume structure evidence came from"""
    TABLE = "table"           # Explicit table in Section L (highest confidence)
    LIST = "list"             # Numbered list "Volume 1:", "Volume 2:"
    FACTOR = "factor"         # Mapped from Section M evaluation factors
    FILENAME = "filename"     # Inferred from uploaded file names
    KEYWORD = "keyword"       # Keyword match (low confidence)
    DEFAULT = "default"       # No evidence found, using fallback


class ConfidenceLevel(Enum):
    """v3.3: Confidence in the volume structure"""
    HIGH = "high"             # Explicit table or numbered list
    MEDIUM = "medium"         # Factor mapping or filename inference
    LOW = "low"               # Keyword match or default


@dataclass
class ProposalVolume:
    """A proposal volume"""
    id: str
    name: str
    volume_type: VolumeType
    page_limit: Optional[int] = None
    sections: List[ProposalSection] = field(default_factory=list)
    eval_factors: List[str] = field(default_factory=list)
    order: int = 0
    # v3.3: Evidence tracking for Strict Constructionist approach
    evidence_source: EvidenceSource = EvidenceSource.DEFAULT
    confidence: ConfidenceLevel = ConfidenceLevel.LOW


@dataclass
class FormatRequirements:
    """Document format requirements"""
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    line_spacing: Optional[str] = None
    margins: Optional[str] = None
    page_size: Optional[str] = None


@dataclass
class SubmissionInfo:
    """Submission requirements"""
    due_date: Optional[str] = None
    due_time: Optional[str] = None
    method: Optional[str] = None
    copies: Optional[int] = None
    email: Optional[str] = None


@dataclass 
class EvaluationFactor:
    """An evaluation factor from Section M"""
    id: str
    name: str
    weight: Optional[str] = None
    importance: Optional[str] = None
    criteria: List[str] = field(default_factory=list)
    rating_scale: Optional[str] = None


@dataclass
class ProposalOutline:
    """Complete proposal outline"""
    rfp_format: str  # "NIH", "GSA_BPA", "STATE_RFP", "STANDARD_UCF", etc.
    volumes: List[ProposalVolume]
    eval_factors: List[EvaluationFactor]
    format_requirements: FormatRequirements
    submission_info: SubmissionInfo
    warnings: List[str] = field(default_factory=list)
    total_pages: Optional[int] = None


# ============== Proposal Structure Parser (v3.3) ==============

class ProposalStructureParser:
    """
    v3.3: Strict Constructionist parser for proposal volume structure.

    HIERARCHY OF AUTHORITY:
    1. PRIMARY: Explicit tables in Section L with Volume/Pages columns
    2. SECONDARY: Numbered lists "Volume 1:", "Volume 2:" in Section L
    3. TERTIARY: Evaluation Factors from Section M mapped to volumes
    4. QUATERNARY: File names of uploaded documents
    5. FALLBACK: Single generic volume (NOT 4-volume UCF default)

    This parser NEVER hallucinates volumes. It only creates what is
    explicitly evidenced in the source documents.
    """

    # RFP types that should NEVER use UCF defaults
    NO_UCF_TYPES = {
        "task_order", "delivery_order", "idiq_order",
        "request_for_estimate", "rfe", "rac",
        "request_for_quote", "rfq", "bpa_call"
    }

    def __init__(self):
        # Table patterns for Section L volume definitions
        self.table_patterns = [
            # "Volume | Title | Page Limit" table format
            r'(?:volume|vol\.?)\s*(\d+)\s*[:\|\s]+([A-Za-z\s/\-]+?)\s*[:\|\s]+(\d+)\s*pages?',
            # "1. Technical Volume... 8 pages" row format
            r'^(\d+)\.\s*([A-Za-z\s/\-]+?(?:volume|proposal)?)\s*[\.:\-\s]+(\d+)\s*pages?',
        ]

        # List patterns for Section L
        self.list_patterns = [
            # "Volume 1: Technical Proposal"
            r'volume\s*(\d+|[ivx]+)[:\s]+([A-Za-z\s/\-]+?)(?:\n|$|\.|\()',
            # "Vol. 1 - Technical"
            r'vol\.?\s*(\d+|[ivx]+)\s*[\-:]\s*([A-Za-z\s/\-]+?)(?:\n|$|\.)',
            # "Book 1: Technical Volume"
            r'book\s*(\d+)[:\s]+([A-Za-z\s/\-]+?)(?:\n|$|\.)',
            # "Part 1: Technical"
            r'part\s*(\d+)[:\s]+([A-Za-z\s/\-]+?)(?:\n|$|\.)',
        ]

        # RFP type indicators
        self.rfp_type_indicators = {
            "task_order": ["task order", "to request", "delivery order"],
            "request_for_estimate": ["request for estimate", "rfe", "rac request"],
            "request_for_quote": ["request for quote", "rfq", "quotation"],
            "bpa_call": ["bpa call", "blanket purchase"],
            "idiq": ["idiq", "indefinite delivery", "indefinite quantity"],
            "full_open": ["full and open", "far 15", "competitive"],
        }

    def parse_section_l_structure(self, section_l_text: str) -> Tuple[List[Dict], EvidenceSource, ConfidenceLevel]:
        """
        Parse Section L for explicit volume structure.

        Returns:
            (volumes_list, evidence_source, confidence_level)
        """
        # Try table parsing first (highest confidence)
        volumes = self._parse_volume_tables(section_l_text)
        if volumes:
            print(f"[DEBUG] StructureParser: Found {len(volumes)} volumes from TABLE")
            return volumes, EvidenceSource.TABLE, ConfidenceLevel.HIGH

        # Try list parsing second
        volumes = self._parse_volume_lists(section_l_text)
        if volumes:
            print(f"[DEBUG] StructureParser: Found {len(volumes)} volumes from LIST")
            return volumes, EvidenceSource.LIST, ConfidenceLevel.HIGH

        # No explicit structure found
        print("[DEBUG] StructureParser: No explicit volume structure found in Section L")
        return [], EvidenceSource.DEFAULT, ConfidenceLevel.LOW

    def parse_section_m_factors(self, section_m_text: str) -> Tuple[List[Dict], EvidenceSource, ConfidenceLevel]:
        """
        Parse Section M for evaluation factors that can be mapped to volumes.

        Returns:
            (factors_as_volumes, evidence_source, confidence_level)
        """
        volumes = []

        # Look for Factor patterns
        factor_patterns = [
            r'factor\s*(\d+)[:\s]+([A-Za-z\s\-]+?)(?:\n|\.|\(|$)',
            r'evaluation\s*factor\s*(\d+)[:\s]+([A-Za-z\s\-]+?)(?:\n|\.|\(|$)',
        ]

        for pattern in factor_patterns:
            for match in re.finditer(pattern, section_m_text, re.IGNORECASE):
                num = match.group(1)
                name = match.group(2).strip()
                if len(name) > 2 and len(name) < 60:
                    vol_type = self._infer_volume_type(name)
                    volumes.append({
                        "num": int(num),
                        "name": name.title(),
                        "type": vol_type,
                        "page_limit": None
                    })

        if volumes:
            # Deduplicate by number
            seen = set()
            unique = []
            for v in sorted(volumes, key=lambda x: x["num"]):
                if v["num"] not in seen:
                    seen.add(v["num"])
                    unique.append(v)
            print(f"[DEBUG] StructureParser: Found {len(unique)} volumes from FACTORS")
            return unique, EvidenceSource.FACTOR, ConfidenceLevel.MEDIUM

        return [], EvidenceSource.DEFAULT, ConfidenceLevel.LOW

    def infer_from_filenames(self, filenames: List[str]) -> Tuple[List[Dict], EvidenceSource, ConfidenceLevel]:
        """
        Infer volume structure from uploaded file names.

        Returns:
            (inferred_volumes, evidence_source, confidence_level)
        """
        volumes = []

        # File name patterns that suggest volumes
        filename_indicators = {
            "technical": ("Technical", VolumeType.TECHNICAL),
            "cost": ("Cost/Price", VolumeType.COST_PRICE),
            "price": ("Cost/Price", VolumeType.COST_PRICE),
            "past performance": ("Past Performance", VolumeType.PAST_PERFORMANCE),
            "management": ("Management", VolumeType.MANAGEMENT),
        }

        seen_types = set()
        for filename in filenames:
            fname_lower = filename.lower()
            for indicator, (name, vol_type) in filename_indicators.items():
                if indicator in fname_lower and vol_type not in seen_types:
                    seen_types.add(vol_type)
                    volumes.append({
                        "num": len(volumes) + 1,
                        "name": name,
                        "type": vol_type,
                        "page_limit": None
                    })

        if volumes:
            print(f"[DEBUG] StructureParser: Inferred {len(volumes)} volumes from FILENAMES")
            return volumes, EvidenceSource.FILENAME, ConfidenceLevel.MEDIUM

        return [], EvidenceSource.DEFAULT, ConfidenceLevel.LOW

    def detect_rfp_type(self, all_text: str) -> str:
        """
        Detect the RFP type based on text indicators.

        Returns type string like "task_order", "request_for_estimate", etc.
        """
        text_lower = all_text.lower()

        for rfp_type, indicators in self.rfp_type_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    print(f"[DEBUG] StructureParser: Detected RFP type '{rfp_type}'")
                    return rfp_type

        return "unknown"

    def should_use_ucf_defaults(self, rfp_type: str) -> bool:
        """
        Determine if UCF defaults are appropriate for this RFP type.

        Task Orders, RFEs, RFQs should NEVER use UCF defaults.
        """
        return rfp_type not in self.NO_UCF_TYPES

    def _parse_volume_tables(self, text: str) -> List[Dict]:
        """Parse table-format volume definitions"""
        volumes = []

        for pattern in self.table_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                num = match.group(1)
                name = match.group(2).strip()
                page_limit = int(match.group(3)) if len(match.groups()) > 2 else None

                if len(name) > 2:
                    vol_type = self._infer_volume_type(name)
                    volumes.append({
                        "num": int(num) if num.isdigit() else self._roman_to_int(num),
                        "name": name.title(),
                        "type": vol_type,
                        "page_limit": page_limit
                    })

        return volumes

    def _parse_volume_lists(self, text: str) -> List[Dict]:
        """Parse list-format volume definitions"""
        volumes = []

        for pattern in self.list_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num = match.group(1)
                name = match.group(2).strip()

                # Clean up the name
                name = re.sub(r'\s+', ' ', name).strip()

                if len(name) > 2 and len(name) < 80:
                    vol_type = self._infer_volume_type(name)
                    volumes.append({
                        "num": int(num) if num.isdigit() else self._roman_to_int(num),
                        "name": name.title(),
                        "type": vol_type,
                        "page_limit": None
                    })

        # Deduplicate
        seen = set()
        unique = []
        for v in volumes:
            key = (v["num"], v["name"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(v)

        return sorted(unique, key=lambda x: x["num"])

    def _infer_volume_type(self, name: str) -> VolumeType:
        """Infer volume type from name"""
        name_lower = name.lower()

        if any(x in name_lower for x in ["technical", "approach", "solution"]):
            return VolumeType.TECHNICAL
        elif any(x in name_lower for x in ["cost", "price", "pricing"]):
            return VolumeType.COST_PRICE
        elif any(x in name_lower for x in ["past performance", "experience", "reference"]):
            return VolumeType.PAST_PERFORMANCE
        elif any(x in name_lower for x in ["management", "staffing", "personnel"]):
            return VolumeType.MANAGEMENT
        elif any(x in name_lower for x in ["admin", "contract", "compliance"]):
            return VolumeType.ADMINISTRATIVE
        else:
            return VolumeType.OTHER

    def _roman_to_int(self, s: str) -> int:
        """Convert Roman numeral to integer"""
        roman_values = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100}
        s = s.lower()
        total = 0
        prev = 0
        for char in reversed(s):
            curr = roman_values.get(char, 0)
            if curr < prev:
                total -= curr
            else:
                total += curr
            prev = curr
        return total if total > 0 else 1


# ============== Smart Outline Generator ==============

class SmartOutlineGenerator:
    """
    Generate proposal outlines from compliance matrix data.
    
    This generator uses already-extracted Section L and M data
    rather than re-parsing PDFs, making it more accurate.
    """
    
    def __init__(self):
        # v3.3: Add strict constructionist structure parser
        self.structure_parser = ProposalStructureParser()

        # Volume detection patterns (legacy - kept for backwards compatibility)
        self.volume_patterns = [
            # NIH Factor-based
            (r"factor\s*(\d+)[:\s,\-–]*([a-z\s]+)", "factor"),
            # Volume I, II, III
            (r"volume\s*([ivx\d]+)[:\s,\-–]*([a-z\s]+)?", "volume"),
            # Part 1, Part 2
            (r"part\s*(\d+)[:\s,\-–]*([a-z\s]+)?", "part"),
            # Technical Proposal, Business Proposal
            (r"(technical|business|cost|price|management|past\s*performance)\s*proposal", "named"),
        ]

        # Page limit patterns - v2.11: Improved patterns
        self.page_limit_patterns = [
            r"(?:not\s+(?:to\s+)?exceed|maximum\s+of|limit(?:ed)?\s+to|no\s+more\s+than)\s*(\d+)\s*pages?",
            r"(\d+)\s*page\s*(?:limit|maximum)",
            r"(?:shall|must|should)\s+(?:not\s+exceed|be\s+limited\s+to)\s*(\d+)\s*pages?",
            # v2.11: Additional patterns for common RFP formats
            r"(\d+)\s*pages?\s+(?:maximum|limit|total)",
            r"(?:limited\s+to|up\s+to|approximately)\s*(\d+)\s*pages?",
            r"pages?[:\s]+(\d+)",  # "Pages: 8" format
            r"(?:total|max(?:imum)?)\s*(?:of\s+)?(\d+)\s*pages?",
            r"(\d+)\s*pages?\s+(?:for|per)\s+(?:this|each|the)",
        ]
        
        # Format patterns
        self.format_patterns = {
            "font": r"(times\s*new\s*roman|arial|calibri|courier)",
            "font_size": r"(\d+)\s*[-]?\s*point",
            "margins": r"(\d+(?:\.\d+)?)\s*(?:inch|in|\")\s*margins?",
            "spacing": r"(single|double|1\.5)\s*[-]?\s*spac",
        }
        
        # Evaluation weight patterns
        self.weight_patterns = [
            r"(\d+)\s*(?:points?|%|percent)",
            r"(?:more|most|equally|less)\s+important",
            r"descending\s+order\s+of\s+importance",
        ]
    
    def generate_from_compliance_matrix(
        self,
        section_l_requirements: List[Dict],
        section_m_requirements: List[Dict],
        technical_requirements: List[Dict],
        stats: Dict
    ) -> ProposalOutline:
        """
        Generate proposal outline from compliance matrix data.
        
        Args:
            section_l_requirements: Extracted Section L requirements
            section_m_requirements: Extracted Section M/evaluation requirements
            technical_requirements: Extracted technical/SOW requirements
            stats: Extraction statistics
            
        Returns:
            ProposalOutline with volumes, eval factors, format requirements
        """
        warnings = []
        
        # Detect RFP format
        rfp_format = self._detect_rfp_format(
            section_l_requirements,
            section_m_requirements,
            stats
        )

        # v3.3: Extract volumes using STRICT CONSTRUCTIONIST approach
        # No more UCF defaults - only create what is explicitly evidenced
        volumes, vol_warnings = self._extract_volumes(
            section_l_requirements,
            section_m_requirements,
            rfp_format,
            filenames=None  # TODO: Pass actual filenames for Level 3 inference
        )
        warnings.extend(vol_warnings)

        # Log structure confidence for debugging
        if volumes:
            for vol in volumes:
                print(f"[DEBUG] Volume '{vol.name}': source={vol.evidence_source.value}, confidence={vol.confidence.value}")
        
        # Extract evaluation factors
        eval_factors = self._extract_eval_factors(section_m_requirements)
        
        # For NIH format, ensure all factors from sections are in eval_factors
        if rfp_format == "NIH_FACTOR":
            # The sections in the Technical volume ARE the evaluation factors
            for vol in volumes:
                if vol.volume_type == VolumeType.TECHNICAL and vol.sections:
                    for sec in vol.sections:
                        # Check if this factor is already in eval_factors
                        factor_id = sec.id.replace("SEC-", "EVAL-")
                        if not any(ef.id == factor_id for ef in eval_factors):
                            ef = EvaluationFactor(
                                id=factor_id,
                                name=sec.name,
                                weight=None
                            )
                            eval_factors.append(ef)
            # Sort by ID
            eval_factors.sort(key=lambda f: f.id)
        
        # Map eval factors to volumes
        self._map_eval_factors_to_volumes(volumes, eval_factors)
        
        # Extract format requirements
        format_req = self._extract_format_requirements(section_l_requirements)
        if not format_req.font_name:
            warnings.append("Font requirement not found - verify in RFP")
        if not format_req.margins:
            warnings.append("Margin requirement not found - verify in RFP")
        
        # Extract submission info
        submission = self._extract_submission_info(section_l_requirements)
        
        # Extract page limits and apply to volumes
        self._apply_page_limits(section_l_requirements, volumes)
        for vol in volumes:
            if not vol.page_limit:
                warnings.append(f"No page limit found for: {vol.name}")

        # v3.0: CRITICAL FIX - Populate volume sections with actual requirements
        # This was the root cause of generic placeholder output
        self._populate_sections_from_requirements(
            volumes,
            section_l_requirements,
            section_m_requirements,
            technical_requirements
        )

        # Calculate total pages
        total_pages = sum(v.page_limit or 0 for v in volumes) or None
        
        return ProposalOutline(
            rfp_format=rfp_format,
            volumes=volumes,
            eval_factors=eval_factors,
            format_requirements=format_req,
            submission_info=submission,
            warnings=warnings,
            total_pages=total_pages
        )
    
    def _detect_rfp_format(
        self, 
        section_l: List[Dict], 
        section_m: List[Dict],
        stats: Dict
    ) -> str:
        """Detect the RFP format type"""
        
        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "") 
            for r in (section_l + section_m)
        ]).lower()
        
        # Check for GSA/BPA indicators
        if stats.get("is_non_ucf_format") or len(section_l) == 0:
            if "gsa" in all_text or "schedule" in all_text or "bpa" in all_text:
                return "GSA_BPA"
            elif "quote" in all_text or "quotation" in all_text:
                return "GSA_RFQ"
        
        # Check for NIH Factor-based format
        factor_matches = re.findall(r"factor\s*\d+", all_text)
        if len(factor_matches) >= 3:
            return "NIH_FACTOR"
        
        # Check for state RFP format
        if re.search(r"section\s+[f-g]\.\d+", all_text):
            return "STATE_RFP"
        
        # Check for DoD format
        if "dfars" in all_text or "section l." in all_text:
            return "DOD_UCF"
        
        # Default to standard UCF
        return "STANDARD_UCF"
    
    def _extract_volumes(
        self,
        section_l: List[Dict],
        section_m: List[Dict],
        rfp_format: str,
        filenames: List[str] = None
    ) -> Tuple[List[ProposalVolume], List[str]]:
        """
        v3.3: Extract proposal volumes using STRICT CONSTRUCTIONIST approach.

        HIERARCHY OF AUTHORITY:
        1. PRIMARY: Section L tables/lists (highest confidence)
        2. SECONDARY: Section M evaluation factors
        3. TERTIARY: File names
        4. FALLBACK: Single generic volume (NOT 4-volume UCF)

        Returns:
            (volumes, warnings)
        """
        warnings = []

        # Combine text for parsing
        section_l_text = "\n".join([r.get("text", "") for r in section_l])
        section_m_text = "\n".join([r.get("text", "") for r in section_m])
        all_text = section_l_text + "\n" + section_m_text

        print(f"[DEBUG] _extract_volumes v3.3: section_l={len(section_l)}, section_m={len(section_m)}")
        print(f"[DEBUG] _extract_volumes v3.3: section_l_text={len(section_l_text)} chars, section_m_text={len(section_m_text)} chars")

        # Detect RFP type for validation
        rfp_type = self.structure_parser.detect_rfp_type(all_text)
        allow_ucf_defaults = self.structure_parser.should_use_ucf_defaults(rfp_type)
        print(f"[DEBUG] _extract_volumes v3.3: rfp_type={rfp_type}, allow_ucf_defaults={allow_ucf_defaults}")

        # =====================================================================
        # LEVEL 1: Try Section L explicit structure (PRIMARY AUTHORITY)
        # =====================================================================
        vol_dicts, evidence_source, confidence = self.structure_parser.parse_section_l_structure(section_l_text)

        if vol_dicts:
            volumes = self._convert_vol_dicts_to_volumes(vol_dicts, evidence_source, confidence)
            print(f"[DEBUG] _extract_volumes v3.3: Found {len(volumes)} volumes from Section L ({evidence_source.value})")
            return volumes, warnings

        # =====================================================================
        # LEVEL 2: Try Section M factors (SECONDARY AUTHORITY)
        # =====================================================================
        vol_dicts, evidence_source, confidence = self.structure_parser.parse_section_m_factors(section_m_text)

        if vol_dicts:
            volumes = self._convert_vol_dicts_to_volumes(vol_dicts, evidence_source, confidence)
            print(f"[DEBUG] _extract_volumes v3.3: Found {len(volumes)} volumes from Section M factors")
            return volumes, warnings

        # =====================================================================
        # LEVEL 3: Try file names (TERTIARY AUTHORITY)
        # =====================================================================
        if filenames:
            vol_dicts, evidence_source, confidence = self.structure_parser.infer_from_filenames(filenames)

            if vol_dicts:
                volumes = self._convert_vol_dicts_to_volumes(vol_dicts, evidence_source, confidence)
                warnings.append(
                    f"Volume structure inferred from file names (confidence: {confidence.value}). "
                    "Manual review recommended."
                )
                print(f"[DEBUG] _extract_volumes v3.3: Inferred {len(volumes)} volumes from filenames")
                return volumes, warnings

        # =====================================================================
        # LEVEL 4: FALLBACK - Single generic volume
        # =====================================================================
        # v3.3 STRICT CONSTRUCTIONIST: Do NOT create multi-volume UCF defaults
        warnings.append(
            "WARNING: No explicit volume structure found in Section L or M. "
            "Creating minimal single-volume response. Manual structure review required."
        )

        print("[DEBUG] _extract_volumes v3.3: No structure found, using minimal fallback")

        # Return single generic volume
        return [
            ProposalVolume(
                id="VOL-1",
                name="Proposal Response",
                volume_type=VolumeType.TECHNICAL,
                page_limit=None,
                order=0,
                evidence_source=EvidenceSource.DEFAULT,
                confidence=ConfidenceLevel.LOW
            )
        ], warnings

    def _convert_vol_dicts_to_volumes(
        self,
        vol_dicts: List[Dict],
        evidence_source: EvidenceSource,
        confidence: ConfidenceLevel
    ) -> List[ProposalVolume]:
        """Convert parsed volume dictionaries to ProposalVolume objects"""
        volumes = []
        seen = set()

        for i, v in enumerate(vol_dicts):
            name = v.get("name", f"Volume {i+1}")
            key = name.lower()

            if key not in seen:
                seen.add(key)
                volumes.append(ProposalVolume(
                    id=f"VOL-{i+1}",
                    name=name,
                    volume_type=v.get("type", VolumeType.OTHER),
                    page_limit=v.get("page_limit"),
                    order=i,
                    evidence_source=evidence_source,
                    confidence=confidence
                ))

        return volumes
    
    def _extract_nih_volumes(self, text: str, section_m: List[Dict]) -> List[ProposalVolume]:
        """Extract volumes from NIH Factor-based RFP"""
        volumes = []
        
        # Look for "Technical Proposal" and "Business Proposal" parts
        if "technical proposal" in text.lower():
            vol = ProposalVolume(
                id="VOL-TECH",
                name="Technical Proposal",
                volume_type=VolumeType.TECHNICAL
            )
            volumes.append(vol)
        
        if "business proposal" in text.lower() or "cost proposal" in text.lower():
            vol = ProposalVolume(
                id="VOL-BUS",
                name="Business Proposal",
                volume_type=VolumeType.COST_PRICE
            )
            volumes.append(vol)
        
        # Extract factors from Section M data
        # Look for clean patterns like "Factor 2, Program and Project Management"
        factor_names = {}
        
        for req in section_m:
            req_text = req.get("text", "") or req.get("full_text", "") or ""
            
            # Pattern 1: "Factor N, Full Name Here" (NIH standard)
            # This captures multi-word names until end of line or period
            match = re.search(
                r"^Factor\s*(\d+)[,:\s]+([A-Z][A-Za-z\s,\-–&]+?)(?:\.\.\.|$)",
                req_text,
                re.IGNORECASE | re.MULTILINE
            )
            if match:
                factor_num = match.group(1)
                factor_name = match.group(2).strip()
                # Clean trailing punctuation and common suffixes
                factor_name = re.sub(r'[\s,\-–]+$', '', factor_name)
                
                # Skip garbage names (too short, or contain non-name words)
                garbage_indicators = ['condition', 'shall', 'must', 'will', 'the', 'a ', 'an ']
                is_garbage = any(g in factor_name.lower() for g in garbage_indicators) or len(factor_name) < 8
                
                key = f"factor_{factor_num}"
                if factor_name and not is_garbage:
                    # Prefer longer names (more complete)
                    if key not in factor_names or len(factor_name) > len(factor_names[key]):
                        factor_names[key] = factor_name.title()
        
        # v2.11: Do NOT apply hardcoded NIH default factors
        # These were causing cross-contamination between RFPs
        # Only use factors that are explicitly found in the text
        # Removed: default_factors dictionary that leaked NIH names into other RFPs
        
        # Create sections from factors
        tech_sections = []
        for factor_key in sorted(factor_names.keys()):
            factor_num = factor_key.replace("factor_", "")
            factor_name = factor_names[factor_key]
            
            section = ProposalSection(
                id=f"SEC-F{factor_num}",
                name=f"Factor {factor_num}: {factor_name}"
            )
            tech_sections.append(section)
        
        # Add sections to Technical volume
        if volumes and tech_sections:
            volumes[0].sections = tech_sections
        
        return volumes
    
    def _extract_gsa_volumes(self, text: str, section_l: List[Dict]) -> List[ProposalVolume]:
        """Extract volumes from GSA/BPA RFP"""
        volumes = []
        text_lower = text.lower()
        
        # Look for Volume I, II, III structure
        volume_pattern = r"volume\s*([ivx\d]+)[:\s\-–]*([a-z\s]+?)(?=volume\s*[ivx\d]|$|\n)"
        
        for match in re.finditer(volume_pattern, text_lower):
            vol_num = match.group(1).upper()
            vol_name = match.group(2).strip().title() if match.group(2) else f"Volume {vol_num}"
            
            vol_type = self._classify_volume_type(vol_name)
            
            vol = ProposalVolume(
                id=f"VOL-{vol_num}",
                name=vol_name if vol_name else f"Volume {vol_num}",
                volume_type=vol_type
            )
            volumes.append(vol)
        
        # If no explicit volumes, look for common GSA structure
        if not volumes:
            indicators = [
                ("Technical Approach", VolumeType.TECHNICAL),
                ("Past Performance", VolumeType.PAST_PERFORMANCE),
                ("Price", VolumeType.COST_PRICE),
            ]
            
            for name, vol_type in indicators:
                if name.lower() in text_lower:
                    vol = ProposalVolume(
                        id=f"VOL-{len(volumes)+1}",
                        name=name,
                        volume_type=vol_type
                    )
                    volumes.append(vol)
        
        return volumes
    
    def _extract_standard_volumes(self, text: str, section_l: List[Dict]) -> List[ProposalVolume]:
        """
        v3.1: Extract volumes from standard UCF RFP.

        CRITICAL: Only create volumes that are EXPLICITLY defined in Section L.
        Previous version was too aggressive - creating volumes for ANY mention
        of keywords like "staffing" or "personnel", even when these are just
        subsections within a volume, not separate volumes.

        This caused "phantom volumes" like "Volume 4: Staffing" when the RFP
        only specified 3 volumes.
        """
        volumes = []
        text_lower = text.lower()

        # v3.1: Look for EXPLICIT volume definitions only
        # Pattern: "Volume 1", "Vol. 1:", "Volume I:", etc.
        volume_patterns = [
            r'volume\s*(\d+|[ivx]+)[:\s]+([A-Za-z\s/\-]+?)(?:\n|\.|\(|$)',
            r'vol\.?\s*(\d+|[ivx]+)[:\s]+([A-Za-z\s/\-]+?)(?:\n|\.|\(|$)',
            # Table format: "1 Technical" or "1. Technical"
            r'^(\d+)\.?\s+(technical|cost|price|management|past performance)(?:\s|$)',
        ]

        found_explicit_volumes = []
        for pattern in volume_patterns:
            for match in re.finditer(pattern, text_lower, re.MULTILINE):
                vol_num = match.group(1)
                vol_name = match.group(2).strip().title() if len(match.groups()) > 1 else ""

                # Determine volume type from name
                vol_type = VolumeType.OTHER
                name_lower = vol_name.lower()
                if "technical" in name_lower or "approach" in name_lower:
                    vol_type = VolumeType.TECHNICAL
                    vol_name = vol_name or "Technical"
                elif "cost" in name_lower or "price" in name_lower:
                    vol_type = VolumeType.COST_PRICE
                    vol_name = vol_name or "Cost/Price"
                elif "past" in name_lower or "experience" in name_lower:
                    vol_type = VolumeType.PAST_PERFORMANCE
                    vol_name = vol_name or "Past Performance"
                elif "management" in name_lower:
                    vol_type = VolumeType.MANAGEMENT
                    vol_name = vol_name or "Management"
                elif "contract" in name_lower or "admin" in name_lower:
                    vol_type = VolumeType.ADMINISTRATIVE
                    vol_name = vol_name or "Contracts/Administrative"

                found_explicit_volumes.append({
                    "num": vol_num,
                    "name": vol_name,
                    "type": vol_type
                })

        # If explicit volumes found, use them
        if found_explicit_volumes:
            # Deduplicate by name
            seen = set()
            for vol_info in found_explicit_volumes:
                key = vol_info["name"].lower()
                if key not in seen and vol_info["name"]:
                    seen.add(key)
                    vol = ProposalVolume(
                        id=f"VOL-{len(volumes)+1}",
                        name=vol_info["name"],
                        volume_type=vol_info["type"],
                        order=len(volumes)
                    )
                    volumes.append(vol)
        else:
            # v3.1: Only look for high-confidence volume indicators
            # Must include "volume" or "proposal" in the phrase
            explicit_indicators = [
                ("Technical", ["technical proposal", "technical volume"], VolumeType.TECHNICAL),
                ("Cost/Price", ["cost proposal", "price proposal", "cost volume", "price volume"], VolumeType.COST_PRICE),
                ("Past Performance", ["past performance volume"], VolumeType.PAST_PERFORMANCE),
            ]

            for name, keywords, vol_type in explicit_indicators:
                for kw in keywords:
                    if kw in text_lower:
                        vol = ProposalVolume(
                            id=f"VOL-{name.upper().replace('/', '-').replace(' ', '-')}",
                            name=name,
                            volume_type=vol_type
                        )
                        volumes.append(vol)
                        break

        return volumes
    
    def _create_default_volumes(self, rfp_format: str, section_m: List[Dict]) -> List[ProposalVolume]:
        """
        DEPRECATED in v3.3 - Strict Constructionist approach.

        This method is no longer used. The fallback logic is now built into
        _extract_volumes() which returns a single generic volume if no
        explicit structure is found.

        Keeping this method for backwards compatibility but it now returns
        an empty list to prevent phantom volume creation.
        """
        print("[WARNING] _create_default_volumes is DEPRECATED in v3.3 - use _extract_volumes() instead")

        # v3.3: Return empty list - no more UCF defaults
        # The calling code should use _extract_volumes() which has built-in fallback
        return []
    
    def _populate_sections_from_requirements(
        self,
        volumes: List[ProposalVolume],
        section_l: List[Dict],
        section_m: List[Dict],
        technical: List[Dict]
    ) -> None:
        """
        v3.0: Populate volume sections with actual requirements.

        CRITICAL: Structure must follow Section L (Instructions), NOT SOW.
        - Section L defines the PROPOSAL STRUCTURE (sub-factors, volumes, headings)
        - SOW requirements are MAPPED to Section L sections
        - This ensures evaluator compliance (they score by L/M criteria)
        """
        if not volumes:
            return

        # =====================================================================
        # STEP 1: Extract required proposal structure from Section L
        # Look for patterns like "Sub Factor 1:", "1.1 Executive Summary", etc.
        # =====================================================================
        section_l_text = "\n".join([r.get("text", "") for r in section_l])
        section_m_text = "\n".join([r.get("text", "") for r in section_m])
        all_instructions = section_l_text + "\n" + section_m_text

        # Parse Section L for required proposal sections/sub-factors
        l_defined_sections = self._extract_l_defined_sections(all_instructions, section_l)

        # =====================================================================
        # STEP 2: Extract page limits from Section L
        # =====================================================================
        page_limits = self._extract_page_limits_from_l(all_instructions)

        # =====================================================================
        # STEP 3: Extract evaluation weights from Section M
        # =====================================================================
        eval_weights = self._extract_eval_weights_from_m(section_m_text, section_m)

        # =====================================================================
        # STEP 4: Build sections from Section L structure (NOT SOW)
        # =====================================================================
        for vol in volumes:
            if vol.volume_type == VolumeType.TECHNICAL:
                if l_defined_sections:
                    # Use Section L defined structure
                    sections = []
                    for i, l_sec in enumerate(l_defined_sections):
                        sec_name = l_sec.get("name", f"Section {i+1}")
                        sec_id = l_sec.get("id", f"SEC-L-{i+1}")

                        # Find SOW requirements relevant to this section
                        relevant_reqs = self._map_requirements_to_section(
                            technical, sec_name, l_sec.get("keywords", [])
                        )

                        # Find L instructions for this section
                        l_instructions = self._find_l_instructions_for_section(
                            section_l, sec_name
                        )

                        # Find M evaluation criteria for this section
                        m_criteria = self._find_m_criteria_for_section(
                            section_m, sec_name, eval_weights
                        )

                        section = ProposalSection(
                            id=sec_id,
                            name=sec_name,
                            page_limit=page_limits.get(sec_name.lower()),
                            requirements=l_instructions + relevant_reqs,
                            eval_criteria=m_criteria,
                        )
                        sections.append(section)

                    vol.sections = sections

                    # Apply volume-level page limit
                    if "technical" in vol.name.lower() and page_limits.get("technical"):
                        vol.page_limit = page_limits.get("technical")
                else:
                    # Fallback: Create basic structure from SOW if no L structure found
                    vol.sections = self._create_fallback_sections(technical, section_l, section_m)

            elif vol.volume_type == VolumeType.COST_PRICE:
                pricing_reqs = [r.get("text", "") for r in (section_l + section_m)
                               if any(kw in r.get("text", "").lower()
                                     for kw in ["cost", "price", "pricing", "budget", "rate"])]
                if pricing_reqs and not vol.sections:
                    vol.sections = [ProposalSection(
                        id="SEC-COST",
                        name="Cost/Price Requirements",
                        page_limit=page_limits.get("cost") or page_limits.get("price"),
                        requirements=pricing_reqs[:15],
                    )]

    def _extract_l_defined_sections(self, text: str, section_l: List[Dict]) -> List[Dict]:
        """
        Extract required proposal sections from Section L instructions.

        Looks for patterns like:
        - "Sub Factor 1: Management Approach"
        - "1.1 Executive Summary"
        - "Volume I: Technical"
        """
        sections = []
        text_lower = text.lower()

        # Pattern 1: Sub-Factor patterns (common in DoD RFPs)
        # v3.2 FIX: Limit to 1-2 digit numbers to exclude Standard Forms (SF1449, SF30, etc.)
        subfactor_patterns = [
            r'sub[\-\s]*factor\s*(\d{1,2})[:\s]+([A-Za-z\s\-]+?)(?:\n|\.|\(|$)',
            r'factor\s*(\d{1,2})[:\s]+([A-Za-z\s\-]+?)(?:\n|\.|\(|$)',
            # SF1, SF2 etc. but NOT SF1449 (Standard Form)
            r'(?<![0-9])sf[\-\s]*(\d{1,2})(?![0-9])[:\s]+([A-Za-z\s\-]+?)(?:\n|\.|\(|$)',
        ]

        for pattern in subfactor_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                num = match.group(1)
                name = match.group(2).strip()
                # Clean up the name
                name = re.sub(r'\s+', ' ', name).strip()
                if len(name) > 3 and len(name) < 80:
                    sections.append({
                        "id": f"SEC-SF-{num}",
                        "name": f"Sub Factor {num}: {name.title()}",
                        "keywords": self._extract_keywords_from_name(name),
                        "order": int(num)
                    })

        # Pattern 2: Numbered section patterns (1.1 Executive Summary)
        numbered_patterns = [
            r'(\d+\.\d+)\s+([A-Z][A-Za-z\s\-]+?)(?:\n|\.|\(|:)',
        ]

        if not sections:  # Only if no sub-factors found
            for pattern in numbered_patterns:
                for match in re.finditer(pattern, text):
                    num = match.group(1)
                    name = match.group(2).strip()
                    if len(name) > 3 and len(name) < 60:
                        sections.append({
                            "id": f"SEC-{num.replace('.', '-')}",
                            "name": f"{num} {name.title()}",
                            "keywords": self._extract_keywords_from_name(name),
                            "order": float(num)
                        })

        # Pattern 3: Common section names
        common_sections = [
            ("executive summary", "Executive Summary"),
            ("management approach", "Management Approach"),
            ("technical approach", "Technical Approach"),
            ("infrastructure approach", "Infrastructure Approach"),
            ("staffing", "Staffing Plan"),
            ("past performance", "Past Performance"),
            ("key personnel", "Key Personnel"),
            ("quality", "Quality Assurance"),
        ]

        if not sections:  # Only if nothing else found
            for keyword, name in common_sections:
                if keyword in text_lower:
                    sections.append({
                        "id": f"SEC-{keyword.replace(' ', '-').upper()}",
                        "name": name,
                        "keywords": [keyword],
                        "order": len(sections)
                    })

        # Sort by order and deduplicate
        seen = set()
        unique_sections = []
        for sec in sorted(sections, key=lambda s: s.get("order", 999)):
            key = sec["name"].lower()
            if key not in seen:
                seen.add(key)
                unique_sections.append(sec)

        return unique_sections

    def _extract_keywords_from_name(self, name: str) -> List[str]:
        """Extract keywords from section name for requirement mapping"""
        name_lower = name.lower()
        keywords = []

        keyword_map = {
            "management": ["manage", "management", "project", "program", "plan", "schedule", "risk"],
            "technical": ["technical", "approach", "solution", "design", "develop", "implement"],
            "infrastructure": ["infrastructure", "system", "network", "server", "cloud", "hardware"],
            "staffing": ["staff", "personnel", "team", "resource", "labor", "employee"],
            "executive": ["executive", "summary", "overview", "introduction"],
            "past performance": ["past", "performance", "experience", "reference", "contract"],
            "quality": ["quality", "assurance", "qa", "qc", "control", "test"],
        }

        for key, kws in keyword_map.items():
            if key in name_lower:
                keywords.extend(kws)

        return list(set(keywords))

    def _extract_page_limits_from_l(self, text: str) -> Dict[str, int]:
        """
        v3.1: Extract page limits from Section L instructions.

        Now handles table formats like:
          "1 Technical 8 Pages"
          "Volume 1: Technical - 8 pages maximum"
        """
        limits = {}
        text_lower = text.lower()

        # v3.2: Table-aware patterns (e.g., "1 Technical 8 Pages")
        # CRITICAL: Must require "page" to avoid matching volume numbers
        # Bug fix: "Volume 1 Technical" was being parsed as "1 page limit"
        table_patterns = [
            # "1 Technical 8 Pages" or "1. Technical 8 Pages" - page count AFTER volume name
            r'(\d+)\.?\s*(technical|cost|price|management|past performance)[^\d]*(\d+)\s*pages?',
            # "Technical: 8 pages" or "Technical Volume: 8 pages"
            r'(technical|cost|price|management)\s*(?:volume)?[:\s]+(\d+)\s*pages?',
            # "Volume 1: Technical... 8 pages"
            r'volume\s*\d+[:\s]*(technical|cost|price|management)[^\d]*(\d+)\s*pages?',
            # "8 pages for Technical" or "8 page technical"
            r'(\d+)\s*pages?\s+(?:for\s+)?(technical|cost|price|management)',
        ]

        for pattern in table_patterns:
            for match in re.finditer(pattern, text_lower):
                groups = match.groups()
                # Find the volume type and page count from captured groups
                vol_type = None
                pages = None

                for g in groups:
                    if g and g.isdigit():
                        candidate = int(g)
                        # v3.2: Sanity check - page limits are typically 5-100, not 1-4
                        # Volume numbers are 1-4, so skip small numbers without "page" context
                        if candidate >= 5 or (candidate > 0 and pages is None):
                            pages = candidate
                    elif g and g in ["technical", "cost", "price", "management", "past performance"]:
                        vol_type = g

                # v3.2: Page limit must be > 1 (1 page limit is almost never real)
                if vol_type and pages and pages > 1 and pages < 500:
                    if vol_type == "technical":
                        limits["technical"] = pages
                    elif vol_type in ["cost", "price"]:
                        limits["cost"] = pages
                    elif vol_type == "management":
                        limits["management"] = pages

        # Original patterns for prose formats
        prose_patterns = [
            r'technical[^:]*:\s*(\d+)\s*page',
            r'cost[^:]*:\s*(\d+)\s*page',
            r'price[^:]*:\s*(\d+)\s*page',
            r'page\s*limit[^:]*:\s*(\d+)',
            r'not\s*(?:to\s*)?exceed\s*(\d+)\s*page',
            r'maximum\s*(?:of\s*)?(\d+)\s*page',
            r'(\d+)\s*page\s*(?:limit|maximum)',
            # "shall not exceed 8 pages"
            r'shall\s*not\s*exceed\s*(\d+)\s*page',
            # "limited to 8 pages"
            r'limited\s*to\s*(\d+)\s*page',
        ]

        for pattern in prose_patterns:
            for match in re.finditer(pattern, text_lower):
                pages = int(match.group(1))
                if pages > 0 and pages < 500:  # Sanity check
                    # Determine which volume this applies to
                    context = text_lower[max(0, match.start()-100):match.end()+50]
                    if "technical" in context and "technical" not in limits:
                        limits["technical"] = pages
                    elif ("cost" in context or "price" in context) and "cost" not in limits:
                        limits["cost"] = pages
                    elif "management" in context and "management" not in limits:
                        limits["management"] = pages
                    elif "total" not in limits:
                        limits["total"] = pages

        return limits

    def _extract_eval_weights_from_m(self, text: str, section_m: List[Dict]) -> Dict[str, str]:
        """Extract evaluation factor weights from Section M"""
        weights = {}
        text_lower = text.lower()

        # Look for weighting patterns
        # e.g., "SF1 > SF2", "Technical is more important than Cost"

        # Pattern: "Factor 1 is more important than Factor 2"
        importance_patterns = [
            r'(factor\s*\d+|sf\s*\d+|technical|management|cost|price|past performance)[^.]*(?:more|most|greater)\s*import',
            r'(factor\s*\d+|sf\s*\d+|technical|management|cost|price)[^.]*>\s*(factor\s*\d+|sf\s*\d+|technical|management|cost)',
        ]

        for pattern in importance_patterns:
            for match in re.finditer(pattern, text_lower):
                weights["formula"] = match.group(0)[:100]

        # Look for specific weight percentages
        weight_pattern = r'(technical|management|cost|price|past performance)[^:]*:\s*(\d+)\s*%'
        for match in re.finditer(weight_pattern, text_lower):
            factor = match.group(1)
            pct = match.group(2)
            weights[factor] = f"{pct}%"

        return weights

    def _map_requirements_to_section(
        self,
        requirements: List[Dict],
        section_name: str,
        keywords: List[str]
    ) -> List[str]:
        """Map SOW requirements to a Section L defined section using keyword matching"""
        relevant = []
        section_lower = section_name.lower()

        for req in requirements:
            text = req.get("text", "")
            text_lower = text.lower()

            # Score relevance
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    score += 1

            # Also check section name keywords
            if "management" in section_lower and any(kw in text_lower for kw in ["manage", "project", "program", "plan"]):
                score += 2
            if "technical" in section_lower and any(kw in text_lower for kw in ["technical", "approach", "solution"]):
                score += 2
            if "infrastructure" in section_lower and any(kw in text_lower for kw in ["system", "network", "infrastructure"]):
                score += 2
            if "staffing" in section_lower and any(kw in text_lower for kw in ["staff", "personnel", "team"]):
                score += 2

            if score > 0:
                relevant.append((score, text))

        # Sort by relevance and return top matches
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in relevant[:15]]

    def _find_l_instructions_for_section(self, section_l: List[Dict], section_name: str) -> List[str]:
        """Find Section L instructions relevant to a specific section"""
        relevant = []
        section_lower = section_name.lower()

        for req in section_l:
            text = req.get("text", "")
            text_lower = text.lower()

            # Check if instruction mentions this section
            if any(kw in text_lower for kw in section_lower.split()):
                relevant.append(text)

        return relevant[:5]

    def _find_m_criteria_for_section(
        self,
        section_m: List[Dict],
        section_name: str,
        eval_weights: Dict[str, str]
    ) -> List[str]:
        """Find Section M evaluation criteria for a specific section"""
        relevant = []
        section_lower = section_name.lower()

        for req in section_m:
            text = req.get("text", "")
            text_lower = text.lower()

            # Check if criteria mentions this section
            if any(kw in text_lower for kw in section_lower.split()):
                relevant.append(text)
            # Also check for evaluation language
            elif any(kw in text_lower for kw in ["evaluat", "assess", "rating", "score"]):
                relevant.append(text)

        # Add weight info if available
        for factor, weight in eval_weights.items():
            if factor in section_lower:
                relevant.insert(0, f"Weight: {weight}")

        return relevant[:5]

    def _create_fallback_sections(
        self,
        technical: List[Dict],
        section_l: List[Dict],
        section_m: List[Dict]
    ) -> List[ProposalSection]:
        """
        Fallback: Create sections from SOW structure when no Section L structure found.
        This maintains backward compatibility but is less ideal.
        """
        def get_section_ref(req: Dict) -> str:
            text = req.get("text", "") or ""
            rfp_ref = req.get("rfp_reference", "")
            if rfp_ref:
                match = re.match(r'^(\d+(?:\.\d+)*)', rfp_ref)
                if match:
                    return match.group(1)
            match = re.match(r'^(\d+(?:\.\d+)+)\s', text)
            if match:
                return match.group(1)
            return "general"

        tech_by_section: Dict[str, List[str]] = {}
        for req in technical:
            sec_ref = get_section_ref(req)
            parts = sec_ref.split('.')
            group_key = '.'.join(parts[:2]) if len(parts) >= 2 else parts[0]
            if group_key not in tech_by_section:
                tech_by_section[group_key] = []
            tech_by_section[group_key].append(req.get("text", ""))

        sections = []
        for sec_ref, reqs in sorted(tech_by_section.items()):
            if sec_ref == "general":
                sec_name = "General Requirements"
            else:
                sec_name = f"Section {sec_ref} Requirements"
            section = ProposalSection(
                id=f"SEC-{sec_ref.replace('.', '-')}",
                name=sec_name,
                requirements=reqs[:20],
            )
            sections.append(section)

        if not sections and technical:
            sections.append(ProposalSection(
                id="SEC-TECH-ALL",
                name="Technical Requirements",
                requirements=[r.get("text", "") for r in technical[:30]],
            ))

        return sections

    def _classify_volume_type(self, name: str) -> VolumeType:
        """Classify volume type from name"""
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in ["technical", "approach", "solution"]):
            return VolumeType.TECHNICAL
        elif any(kw in name_lower for kw in ["management", "project", "program"]):
            return VolumeType.MANAGEMENT
        elif any(kw in name_lower for kw in ["past performance", "experience", "reference"]):
            return VolumeType.PAST_PERFORMANCE
        elif any(kw in name_lower for kw in ["cost", "price", "pricing", "business"]):
            return VolumeType.COST_PRICE
        elif any(kw in name_lower for kw in ["staff", "personnel", "resume"]):
            return VolumeType.STAFFING
        elif any(kw in name_lower for kw in ["small business", "subcontract"]):
            return VolumeType.SMALL_BUSINESS
        else:
            return VolumeType.OTHER
    
    def _extract_eval_factors(self, section_m: List[Dict]) -> List[EvaluationFactor]:
        """Extract evaluation factors from Section M"""
        factors = []
        seen = set()
        
        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "") 
            for r in section_m
        ])
        
        # Look for clean Factor N patterns at start of lines
        for req in section_m:
            req_text = req.get("text", "") or req.get("full_text", "") or ""
            
            # Match: "Factor N, Full Name" at line start
            match = re.search(
                r"^Factor\s*(\d+)[,:\s]+([A-Z][A-Za-z\s,\-–&]+?)(?:\.\.\.|$)",
                req_text,
                re.IGNORECASE | re.MULTILINE
            )
            if match:
                factor_num = match.group(1)
                factor_name = match.group(2).strip()
                factor_name = re.sub(r'[\s,\-–]+$', '', factor_name).title()
                
                # Skip garbage
                garbage = ['condition', 'shall', 'must', 'will']
                if any(g in factor_name.lower() for g in garbage):
                    continue
                
                key = f"factor_{factor_num}"
                if key not in seen and len(factor_name) > 5:
                    seen.add(key)
                    
                    factor = EvaluationFactor(
                        id=f"EVAL-F{factor_num}",
                        name=f"Factor {factor_num}: {factor_name}",
                        weight=None
                    )
                    factors.append(factor)
        
        # v2.11: Removed hardcoded NIH default factors
        # These were causing cross-contamination between RFPs
        # If no clean factor patterns found, we skip NIH defaults
        # The factors should be extracted from the actual RFP text only
        
        # Still no factors? Look for general criteria keywords
        if not factors:
            criteria_keywords = [
                ("Technical", "EVAL-TECH"),
                ("Management", "EVAL-MGMT"),
                ("Past Performance", "EVAL-PP"),
                ("Price", "EVAL-PRICE"),
                ("Cost", "EVAL-COST"),
            ]
            
            for name, eval_id in criteria_keywords:
                if name.lower() in all_text.lower():
                    factor = EvaluationFactor(
                        id=eval_id,
                        name=name
                    )
                    factors.append(factor)
        
        # Sort factors by ID
        factors.sort(key=lambda f: f.id)
        
        return factors
    
    def _map_eval_factors_to_volumes(
        self, 
        volumes: List[ProposalVolume], 
        eval_factors: List[EvaluationFactor]
    ):
        """Map evaluation factors to corresponding volumes"""
        
        for factor in eval_factors:
            factor_name_lower = factor.name.lower()
            
            for volume in volumes:
                vol_name_lower = volume.name.lower()
                
                # Match by keywords
                if any(kw in factor_name_lower for kw in ["technical", "approach", "solution"]):
                    if volume.volume_type == VolumeType.TECHNICAL:
                        volume.eval_factors.append(factor.name)
                elif any(kw in factor_name_lower for kw in ["management", "project"]):
                    if volume.volume_type in [VolumeType.MANAGEMENT, VolumeType.TECHNICAL]:
                        volume.eval_factors.append(factor.name)
                elif any(kw in factor_name_lower for kw in ["experience", "past performance"]):
                    if volume.volume_type in [VolumeType.PAST_PERFORMANCE, VolumeType.EXPERIENCE]:
                        volume.eval_factors.append(factor.name)
                elif any(kw in factor_name_lower for kw in ["price", "cost"]):
                    if volume.volume_type == VolumeType.COST_PRICE:
                        volume.eval_factors.append(factor.name)
    
    def _extract_format_requirements(self, section_l: List[Dict]) -> FormatRequirements:
        """Extract format requirements from Section L"""
        
        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "") 
            for r in section_l
        ]).lower()
        
        format_req = FormatRequirements()
        
        # Extract font
        font_match = re.search(self.format_patterns["font"], all_text)
        if font_match:
            format_req.font_name = font_match.group(1).title()
        
        # Extract font size
        size_match = re.search(self.format_patterns["font_size"], all_text)
        if size_match:
            format_req.font_size = int(size_match.group(1))
        
        # Extract margins
        margin_match = re.search(self.format_patterns["margins"], all_text)
        if margin_match:
            format_req.margins = f"{margin_match.group(1)} inch"
        
        # Extract spacing
        spacing_match = re.search(self.format_patterns["spacing"], all_text)
        if spacing_match:
            format_req.line_spacing = spacing_match.group(1)
        
        return format_req
    
    def _extract_submission_info(self, section_l: List[Dict]) -> SubmissionInfo:
        """Extract submission info from Section L"""
        
        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "") 
            for r in section_l
        ])
        
        submission = SubmissionInfo()
        
        # Look for due date
        date_patterns = [
            r"(?:due|submit|submission)\s*(?:date|by|on)?\s*[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"no\s+later\s+than[:\s]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"(\d{1,2}:\d{2}\s*(?:AM|PM)\s+(?:ET|EST|PT|PST|CT|CST)?\s+(?:on\s+)?[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                submission.due_date = match.group(1)
                break
        
        # Look for submission method
        method_patterns = [
            r"submit\s+(?:via|through|to)\s+(email|portal|sam\.gov|electronic)",
            r"(electronic\s+submission)",
        ]
        
        for pattern in method_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                submission.method = match.group(1)
                break
        
        # Look for email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", all_text)
        if email_match:
            submission.email = email_match.group(0)
        
        return submission
    
    def _apply_page_limits(self, section_l: List[Dict], volumes: List[ProposalVolume]):
        """Extract page limits and apply to volumes"""
        
        for req in section_l:
            text = req.get("text", "") or req.get("full_text", "") or ""
            text_lower = text.lower()
            
            # Look for page limits
            for pattern in self.page_limit_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    page_limit = int(match.group(1))
                    
                    # Try to match to a volume
                    for volume in volumes:
                        vol_name_lower = volume.name.lower()
                        vol_type = volume.volume_type.value
                        
                        if vol_name_lower in text_lower or vol_type in text_lower:
                            volume.page_limit = page_limit
                            break
    
    def to_json(self, outline: ProposalOutline) -> Dict[str, Any]:
        """Convert outline to JSON format for API"""

        # v4.0 FIX: Defensive None checks to prevent iteration errors
        volumes = outline.volumes or []
        eval_factors = outline.eval_factors or []

        return {
            "rfp_format": outline.rfp_format,
            "total_pages": outline.total_pages,
            "volumes": [
                {
                    "id": vol.id,
                    "name": vol.name,
                    "type": vol.volume_type.value,
                    "page_limit": vol.page_limit,
                    "order": vol.order,
                    "eval_factors": vol.eval_factors or [],
                    # v3.3: Evidence tracking for Strict Constructionist approach
                    "evidence_source": vol.evidence_source.value if hasattr(vol, 'evidence_source') else "default",
                    "confidence": vol.confidence.value if hasattr(vol, 'confidence') else "low",
                    "sections": [
                        {
                            "id": sec.id,
                            "title": sec.name,  # UI expects 'title' not 'name'
                            "name": sec.name,   # Keep both for compatibility
                            "page_limit": sec.page_limit,
                            "content_requirements": sec.requirements or [],  # UI expects this name
                            "requirements": sec.requirements or [],
                            "eval_criteria": sec.eval_criteria or [],  # v3.0: Pass evaluation criteria
                            "compliance_checkpoints": [],  # Can be populated later
                            "subsections": [
                                {"id": sub.id, "title": sub.name, "name": sub.name}
                                for sub in (sec.subsections or [])
                            ]
                        }
                        for sec in (vol.sections or [])
                    ]
                }
                for vol in sorted(volumes, key=lambda v: v.order)
            ],
            # UI expects 'evaluation_factors' not 'eval_factors'
            "evaluation_factors": [
                {
                    "id": ef.id,
                    "name": ef.name,
                    "weight": ef.weight,
                    "importance": ef.importance,
                    "criteria": ef.criteria or [],
                    "subfactors": []  # Can be populated later
                }
                for ef in eval_factors
            ],
            "eval_factors": [  # Keep for backward compatibility
                {
                    "id": ef.id,
                    "name": ef.name,
                    "weight": ef.weight,
                    "importance": ef.importance,
                    "criteria": ef.criteria or []
                }
                for ef in eval_factors
            ],
            "format_requirements": {
                "font": outline.format_requirements.font_name,
                "font_size": outline.format_requirements.font_size,
                "margins": outline.format_requirements.margins,
                "line_spacing": outline.format_requirements.line_spacing,
                "page_size": outline.format_requirements.page_size
            },
            "submission": {
                "due_date": outline.submission_info.due_date or "TBD",
                "due_time": outline.submission_info.due_time,
                "method": outline.submission_info.method or "Not Specified",
                "email": outline.submission_info.email
            },
            "warnings": outline.warnings
        }
