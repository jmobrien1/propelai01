"""
PropelAI: Smart Proposal Outline Generator v2.12

Generates proposal outlines from already-extracted compliance matrix data.
Unlike the legacy outline_generator.py, this uses the structured requirements
already parsed from Section L and M rather than re-parsing PDFs.

Key improvements:
- Uses compliance matrix data (already parsed and categorized)
- Detects RFP format (NIH Factor-based, GSA Volume-based, State RFP, etc.)
- Better volume/section detection using actual requirement text
- Proper evaluation factor mapping
- Evaluation factor weighting (SF1 > SF2 >> Cost/Price)
- P0 formatting constraints enforcement
- Mandatory artifact tracking for Volume 3
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
    CONTRACT_DOCUMENTATION = "contract_documentation"  # Volume 3 for OASIS+
    OTHER = "other"


class AdjectivalRating(Enum):
    """Standard adjectival rating scale"""
    EXCEPTIONAL = "Exceptional"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    UNACCEPTABLE = "Unacceptable"


@dataclass
class P0Constraint:
    """P0 (Pass/Fail) constraint that can cause disqualification"""
    constraint_type: str  # PAGE_LIMIT, FONT, MARGIN, FILE_FORMAT, MANDATORY_FORM
    description: str
    value: str
    applies_to: str
    consequence: str


@dataclass
class MandatoryArtifact:
    """Mandatory artifact for Volume 3 Contract Documentation"""
    artifact_id: str
    name: str
    description: str
    far_reference: Optional[str] = None  # e.g., "FAR 52.204-7"
    form_number: Optional[str] = None  # e.g., "SF1449"
    is_pass_fail: bool = True


@dataclass
class ContentStrategy:
    """Content strategy guidance for targeting Strengths"""
    target_rating: AdjectivalRating
    strength_opportunities: List[str] = field(default_factory=list)
    discriminators: List[str] = field(default_factory=list)
    risk_areas: List[str] = field(default_factory=list)


@dataclass
class ProposalSection:
    """A section within a proposal volume"""
    id: str
    name: str
    page_limit: Optional[int] = None
    requirements: List[str] = field(default_factory=list)
    eval_criteria: List[str] = field(default_factory=list)
    subsections: List['ProposalSection'] = field(default_factory=list)
    content_strategy: Optional[ContentStrategy] = None


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
    mandatory_artifacts: List[MandatoryArtifact] = field(default_factory=list)  # For Volume 3


@dataclass
class FormatRequirements:
    """Document format requirements"""
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    line_spacing: Optional[str] = None
    margins: Optional[str] = None
    page_size: Optional[str] = None
    p0_constraints: List[P0Constraint] = field(default_factory=list)  # Disqualification risks


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
    importance: Optional[str] = None  # "Most Important", "Second Most", etc.
    importance_rank: int = 0  # Numeric rank (1 = most important)
    criteria: List[str] = field(default_factory=list)
    rating_scale: Optional[str] = None  # "Adjectival", "Pass/Fail", "Price"
    adjectival_definitions: Dict[str, str] = field(default_factory=dict)
    content_strategy: Optional[ContentStrategy] = None


@dataclass
class ProposalOutline:
    """Complete proposal outline"""
    rfp_format: str  # "NIH", "GSA_BPA", "STATE_RFP", "STANDARD_UCF", "OASIS_TASK_ORDER"
    volumes: List[ProposalVolume]
    eval_factors: List[EvaluationFactor]
    format_requirements: FormatRequirements
    submission_info: SubmissionInfo
    warnings: List[str] = field(default_factory=list)
    total_pages: Optional[int] = None
    # OASIS+ specific
    adjectival_ratings: Dict[str, str] = field(default_factory=dict)
    mandatory_artifacts: List[MandatoryArtifact] = field(default_factory=list)


# ============== Smart Outline Generator ==============

class SmartOutlineGenerator:
    """
    Generate proposal outlines from compliance matrix data.
    
    This generator uses already-extracted Section L and M data
    rather than re-parsing PDFs, making it more accurate.
    """
    
    def __init__(self):
        # Volume detection patterns
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
        
        # Page limit patterns
        self.page_limit_patterns = [
            r"(?:not\s+(?:to\s+)?exceed|maximum\s+of|limit(?:ed)?\s+to|no\s+more\s+than)\s*(\d+)\s*pages?",
            r"(\d+)\s*page\s*(?:limit|maximum)",
            r"(?:shall|must|should)\s+(?:not\s+exceed|be\s+limited\s+to)\s*(\d+)\s*pages?",
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
        
        # Extract volumes based on format
        volumes = self._extract_volumes(
            section_l_requirements,
            section_m_requirements,
            rfp_format
        )
        
        # If no volumes found, create defaults
        if not volumes:
            volumes = self._create_default_volumes(rfp_format, section_m_requirements)
            warnings.append("No explicit volumes found - using default structure")
        
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
        rfp_format: str
    ) -> List[ProposalVolume]:
        """Extract proposal volumes from requirements"""
        
        volumes = []
        seen_volumes = set()
        
        all_requirements = section_l + section_m
        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "") 
            for r in all_requirements
        ])
        
        # Strategy depends on format
        if rfp_format == "NIH_FACTOR":
            volumes = self._extract_nih_volumes(all_text, section_m)
        elif rfp_format in ["GSA_BPA", "GSA_RFQ"]:
            volumes = self._extract_gsa_volumes(all_text, section_l)
        else:
            volumes = self._extract_standard_volumes(all_text, section_l)
        
        # Deduplicate and order
        unique_volumes = []
        for vol in volumes:
            key = vol.name.lower()
            if key not in seen_volumes:
                seen_volumes.add(key)
                vol.order = len(unique_volumes)
                unique_volumes.append(vol)
        
        return unique_volumes
    
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
        
        # Also look in Section L for factor mentions
        all_text_lower = text.lower()
        
        # Common factor names to look for if not found above
        default_factors = {
            "1": "Experience",
            "2": "Program and Project Management", 
            "3": "Technical Approach",
            "4": "Key Personnel",
            "5": "Facilities and Equipment",
            "6": "Past Performance",
        }
        
        for num, default_name in default_factors.items():
            key = f"factor_{num}"
            if key not in factor_names:
                # Check if this factor is mentioned at all
                if f"factor {num}" in all_text_lower or f"factor{num}" in all_text_lower:
                    factor_names[key] = default_name
        
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
        """Extract volumes from standard UCF RFP"""
        volumes = []
        text_lower = text.lower()
        
        # Standard volume indicators
        indicators = [
            ("Technical", ["technical proposal", "technical volume", "technical approach"], VolumeType.TECHNICAL),
            ("Management", ["management proposal", "management approach", "management volume"], VolumeType.MANAGEMENT),
            ("Past Performance", ["past performance", "experience", "relevant experience"], VolumeType.PAST_PERFORMANCE),
            ("Cost/Price", ["cost proposal", "price proposal", "pricing", "business proposal"], VolumeType.COST_PRICE),
            ("Staffing", ["staffing", "key personnel", "personnel", "resumes"], VolumeType.STAFFING),
            ("Small Business", ["small business", "subcontracting plan"], VolumeType.SMALL_BUSINESS),
        ]
        
        for name, keywords, vol_type in indicators:
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
        """Create default volumes if none were extracted"""
        
        if rfp_format in ["GSA_BPA", "GSA_RFQ"]:
            return [
                ProposalVolume(id="VOL-1", name="Technical Approach", volume_type=VolumeType.TECHNICAL, order=0),
                ProposalVolume(id="VOL-2", name="Past Performance", volume_type=VolumeType.PAST_PERFORMANCE, order=1),
                ProposalVolume(id="VOL-3", name="Price", volume_type=VolumeType.COST_PRICE, order=2),
            ]
        else:
            return [
                ProposalVolume(id="VOL-TECH", name="Technical Proposal", volume_type=VolumeType.TECHNICAL, order=0),
                ProposalVolume(id="VOL-MGMT", name="Management Proposal", volume_type=VolumeType.MANAGEMENT, order=1),
                ProposalVolume(id="VOL-PP", name="Past Performance", volume_type=VolumeType.PAST_PERFORMANCE, order=2),
                ProposalVolume(id="VOL-COST", name="Cost/Price Proposal", volume_type=VolumeType.COST_PRICE, order=3),
            ]
    
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
        
        # If no factors found from clean patterns, use defaults for known factors
        if not factors:
            # Check which factor numbers are mentioned
            all_text_lower = all_text.lower()
            default_factors = {
                "1": "Experience",
                "2": "Program and Project Management",
                "3": "Technical Approach", 
                "4": "Key Personnel",
                "5": "Facilities and Equipment",
                "6": "Past Performance",
            }
            
            for num, name in default_factors.items():
                if f"factor {num}" in all_text_lower or f"factor{num}" in all_text_lower:
                    key = f"factor_{num}"
                    if key not in seen:
                        seen.add(key)
                        factor = EvaluationFactor(
                            id=f"EVAL-F{num}",
                            name=f"Factor {num}: {name}",
                            weight=None
                        )
                        factors.append(factor)
        
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
    
    def _calculate_factor_weighting(
        self,
        eval_factors: List[EvaluationFactor],
        section_m: List[Dict]
    ) -> List[EvaluationFactor]:
        """
        Calculate evaluation factor weighting/importance ranking.

        Per PropelAI mandate: SF1 > SF2 >> Cost/Price
        Uses patterns like "descending order of importance" and explicit weights.
        """
        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "")
            for r in section_m
        ]).lower()

        # Check for descending order of importance
        descending_order = "descending order of importance" in all_text

        # Pattern: "Factor X is more important than Factor Y"
        importance_patterns = [
            r"(technical|management|past\s+performance)\s+(?:is\s+)?(?:more|most)\s+important",
            r"(?:price|cost)\s+(?:is\s+)?(?:less|least)\s+important",
            r"when\s+combined[^.]*exceed\s+(?:price|cost)",
        ]

        tech_more_important = any(
            re.search(p, all_text) for p in importance_patterns[:2]
        )

        # Assign importance ranks
        for i, factor in enumerate(eval_factors):
            factor_lower = factor.name.lower()

            # Set rating scale if not already set
            if not factor.rating_scale:
                if any(kw in factor_lower for kw in ["price", "cost"]):
                    factor.rating_scale = "Price"
                elif "pass" in factor_lower or "fail" in factor_lower:
                    factor.rating_scale = "Pass/Fail"
                else:
                    factor.rating_scale = "Adjectival"

            # Determine importance
            if descending_order:
                # First factor listed is most important
                factor.importance_rank = i + 1
                if i == 0:
                    factor.importance = "Most Important"
                elif i == 1:
                    factor.importance = "Second Most Important"
                elif any(kw in factor_lower for kw in ["price", "cost"]):
                    factor.importance = "Less Important than Technical"
                    factor.importance_rank = len(eval_factors)  # Price typically last
                else:
                    factor.importance = f"#{i + 1} in Importance"

            # Create content strategy for Adjectival factors
            if factor.rating_scale == "Adjectival":
                factor.content_strategy = ContentStrategy(
                    target_rating=AdjectivalRating.EXCEPTIONAL if factor.importance_rank <= 2 else AdjectivalRating.VERY_GOOD,
                    strength_opportunities=[
                        "Demonstrate experience exceeding minimum requirements",
                        "Highlight unique discriminators",
                        "Provide specific, quantified past performance",
                    ],
                    discriminators=[
                        "Innovation beyond RFP requirements",
                        "Risk mitigation approaches",
                        "Proven methodologies with metrics",
                    ],
                    risk_areas=[
                        "Generic language that doesn't address specific requirements",
                        "Missing compliance with mandatory items",
                        "Lack of substantiation for claims",
                    ]
                )

        return eval_factors

    def _extract_p0_constraints(
        self,
        section_l: List[Dict],
        oasis_constraints: Optional[List[Any]] = None
    ) -> List[P0Constraint]:
        """
        Extract P0 (Pass/Fail) formatting constraints.

        These are constraints that can cause disqualification if not followed.
        """
        constraints = []

        # Use OASIS+ constraints if provided
        if oasis_constraints:
            for c in oasis_constraints:
                constraints.append(P0Constraint(
                    constraint_type=getattr(c, 'constraint_type', 'UNKNOWN'),
                    description=getattr(c, 'description', ''),
                    value=getattr(c, 'value', ''),
                    applies_to=getattr(c, 'applies_to', 'All'),
                    consequence=getattr(c, 'consequence', 'May cause disqualification')
                ))
            return constraints

        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "")
            for r in section_l
        ]).lower()

        # Page limit with consequence
        page_patterns = [
            (r"(\d+)\s*page\s*(?:limit|maximum)[^.]*(?:will\s+not\s+be\s+read|excess[^.]*not[^.]*read)", "PAGE_LIMIT"),
            (r"exceed\s*(\d+)\s*pages?[^.]*(?:will\s+not\s+be\s+read|disqualif)", "PAGE_LIMIT"),
        ]

        for pattern, ctype in page_patterns:
            match = re.search(pattern, all_text)
            if match:
                constraints.append(P0Constraint(
                    constraint_type=ctype,
                    description=f"Page limit: {match.group(1)} pages",
                    value=match.group(1),
                    applies_to="All",
                    consequence="Excess pages will NOT be read or considered"
                ))

        # Font requirements
        if "12" in all_text and "point" in all_text:
            constraints.append(P0Constraint(
                constraint_type="FONT_SIZE",
                description="Font size: 12-point minimum",
                value="12",
                applies_to="Body text",
                consequence="Formatting compliance required"
            ))

        # Margin requirements
        margin_match = re.search(r"(\d+)\s*(?:inch|in)\s*margins?", all_text)
        if margin_match:
            constraints.append(P0Constraint(
                constraint_type="MARGIN",
                description=f"Margins: {margin_match.group(1)}-inch all around",
                value=f"{margin_match.group(1)} inch",
                applies_to="All pages",
                consequence="Formatting compliance required"
            ))

        return constraints

    def _extract_mandatory_artifacts(self, section_l: List[Dict]) -> List[MandatoryArtifact]:
        """
        Extract mandatory artifacts for Volume 3 Contract Documentation.

        These are Pass/Fail requirements per FAR/DFARS.
        """
        artifacts = []

        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "")
            for r in section_l
        ])
        all_text_lower = all_text.lower()

        # Standard mandatory artifacts
        artifact_patterns = [
            (r"SF[-\s]?1449", "SF1449", "Standard Form 1449", "Contract Award Form", None),
            (r"DD[-\s]?254", "DD254", "DD Form 254", "Security Classification Specification", None),
            (r"FAR\s+52\.204[-\s]?7", "FAR52.204-7", "Online Representations and Certifications",
             "Evidence of registration in SAM.gov", "FAR 52.204-7"),
            (r"representations?\s+and\s+certifications?", "REPS_CERTS",
             "Representations and Certifications", "Completed certifications per solicitation", None),
            (r"OCI\s+(?:mitigation|checklist|plan)", "OCI_PLAN",
             "OCI Mitigation Plan", "Organizational Conflict of Interest disclosure", None),
            (r"small\s+business\s+subcontracting\s+plan", "SB_PLAN",
             "Small Business Subcontracting Plan", "Required for contracts over threshold", "FAR 52.219-9"),
        ]

        seen_ids = set()
        for pattern, art_id, name, desc, far_ref in artifact_patterns:
            if re.search(pattern, all_text_lower) and art_id not in seen_ids:
                seen_ids.add(art_id)
                artifacts.append(MandatoryArtifact(
                    artifact_id=art_id,
                    name=name,
                    description=desc,
                    far_reference=far_ref,
                    form_number=art_id if art_id.startswith(("SF", "DD")) else None,
                    is_pass_fail=True
                ))

        return artifacts

    def _create_volume_3_artifacts(self, outline: ProposalOutline):
        """
        Ensure Volume 3 (Contract Documentation) has mandatory artifacts tracked.
        """
        vol3 = None
        for vol in outline.volumes:
            if vol.volume_type == VolumeType.CONTRACT_DOCUMENTATION:
                vol3 = vol
                break
            elif "contract" in vol.name.lower() and "doc" in vol.name.lower():
                vol3 = vol
                vol.volume_type = VolumeType.CONTRACT_DOCUMENTATION
                break

        if not vol3:
            # Create Volume 3 if not present
            vol3 = ProposalVolume(
                id="VOL-3",
                name="Contract Documentation Volume",
                volume_type=VolumeType.CONTRACT_DOCUMENTATION,
                order=3,
                sections=[]
            )
            outline.volumes.append(vol3)

        # Add mandatory artifacts
        vol3.mandatory_artifacts = outline.mandatory_artifacts

        # Create sections for mandatory forms if not present
        if not vol3.sections:
            vol3.sections = [
                ProposalSection(
                    id="V3-FORMS",
                    name="Mandatory Forms",
                    requirements=[f"Submit {a.name}" for a in outline.mandatory_artifacts if a.form_number],
                ),
                ProposalSection(
                    id="V3-CERTS",
                    name="Certifications and Representations",
                    requirements=["FAR 52.204-7 Online Representations", "Agency-specific certifications"],
                ),
            ]

    def generate_with_oasis_data(
        self,
        section_l_requirements: List[Dict],
        section_m_requirements: List[Dict],
        technical_requirements: List[Dict],
        stats: Dict,
        oasis_task_order_data: Optional[Dict] = None
    ) -> ProposalOutline:
        """
        Generate outline with OASIS+ task order enhancements.

        This is the enhanced entry point that uses P0 constraints,
        evaluation weighting, and mandatory artifact tracking.
        """
        # First generate base outline
        outline = self.generate_from_compliance_matrix(
            section_l_requirements,
            section_m_requirements,
            technical_requirements,
            stats
        )

        # Apply evaluation factor weighting
        outline.eval_factors = self._calculate_factor_weighting(
            outline.eval_factors,
            section_m_requirements
        )

        # Extract P0 constraints
        oasis_constraints = None
        if oasis_task_order_data:
            oasis_constraints = oasis_task_order_data.get('formatting_constraints', [])
            outline.adjectival_ratings = oasis_task_order_data.get('adjectival_ratings', {})
            outline.rfp_format = "OASIS_TASK_ORDER"

        outline.format_requirements.p0_constraints = self._extract_p0_constraints(
            section_l_requirements,
            oasis_constraints
        )

        # Extract mandatory artifacts
        outline.mandatory_artifacts = self._extract_mandatory_artifacts(section_l_requirements)

        # Ensure Volume 3 tracking
        self._create_volume_3_artifacts(outline)

        return outline

    def to_json(self, outline: ProposalOutline) -> Dict[str, Any]:
        """Convert outline to JSON format for API"""

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
                    "eval_factors": vol.eval_factors,
                    "mandatory_artifacts": [
                        {
                            "id": a.artifact_id,
                            "name": a.name,
                            "description": a.description,
                            "far_reference": a.far_reference,
                            "is_pass_fail": a.is_pass_fail
                        }
                        for a in vol.mandatory_artifacts
                    ] if vol.mandatory_artifacts else [],
                    "sections": [
                        {
                            "id": sec.id,
                            "title": sec.name,  # UI expects 'title' not 'name'
                            "name": sec.name,   # Keep both for compatibility
                            "page_limit": sec.page_limit,
                            "content_requirements": sec.requirements,  # UI expects this name
                            "requirements": sec.requirements,
                            "compliance_checkpoints": [],  # Can be populated later
                            "content_strategy": {
                                "target_rating": sec.content_strategy.target_rating.value,
                                "strength_opportunities": sec.content_strategy.strength_opportunities,
                                "discriminators": sec.content_strategy.discriminators,
                                "risk_areas": sec.content_strategy.risk_areas
                            } if sec.content_strategy else None,
                            "subsections": [
                                {"id": sub.id, "title": sub.name, "name": sub.name}
                                for sub in sec.subsections
                            ]
                        }
                        for sec in vol.sections
                    ]
                }
                for vol in sorted(outline.volumes, key=lambda v: v.order)
            ],
            # UI expects 'evaluation_factors' not 'eval_factors'
            "evaluation_factors": [
                {
                    "id": ef.id,
                    "name": ef.name,
                    "weight": ef.weight,
                    "importance": ef.importance,
                    "importance_rank": ef.importance_rank,
                    "criteria": ef.criteria,
                    "rating_scale": ef.rating_scale,
                    "content_strategy": {
                        "target_rating": ef.content_strategy.target_rating.value,
                        "strength_opportunities": ef.content_strategy.strength_opportunities,
                        "discriminators": ef.content_strategy.discriminators,
                    } if ef.content_strategy else None,
                    "subfactors": []  # Can be populated later
                }
                for ef in outline.eval_factors
            ],
            "eval_factors": [  # Keep for backward compatibility
                {
                    "id": ef.id,
                    "name": ef.name,
                    "weight": ef.weight,
                    "importance": ef.importance,
                    "criteria": ef.criteria
                }
                for ef in outline.eval_factors
            ],
            "format_requirements": {
                "font": outline.format_requirements.font_name,
                "font_size": outline.format_requirements.font_size,
                "margins": outline.format_requirements.margins,
                "line_spacing": outline.format_requirements.line_spacing,
                "page_size": outline.format_requirements.page_size,
                "p0_constraints": [
                    {
                        "type": c.constraint_type,
                        "description": c.description,
                        "value": c.value,
                        "applies_to": c.applies_to,
                        "consequence": c.consequence
                    }
                    for c in outline.format_requirements.p0_constraints
                ] if outline.format_requirements.p0_constraints else []
            },
            "submission": {
                "due_date": outline.submission_info.due_date or "TBD",
                "due_time": outline.submission_info.due_time,
                "method": outline.submission_info.method or "Not Specified",
                "email": outline.submission_info.email
            },
            "warnings": outline.warnings,
            # OASIS+ specific
            "adjectival_ratings": outline.adjectival_ratings,
            "mandatory_artifacts": [
                {
                    "id": a.artifact_id,
                    "name": a.name,
                    "description": a.description,
                    "far_reference": a.far_reference,
                    "form_number": a.form_number,
                    "is_pass_fail": a.is_pass_fail
                }
                for a in outline.mandatory_artifacts
            ] if outline.mandatory_artifacts else []
        }
