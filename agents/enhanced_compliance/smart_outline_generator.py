"""
PropelAI: Smart Proposal Outline Generator v2.10

Generates proposal outlines from already-extracted compliance matrix data.
Unlike the legacy outline_generator.py, this uses the structured requirements
already parsed from Section L and M rather than re-parsing PDFs.

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


# =============================================================================
# Volume-Specific Section Templates (Master Architect Plan - P0 Fix)
# =============================================================================
# Each volume type gets its own section structure instead of cloning Volume 1

VOLUME_SECTION_TEMPLATES = {
    VolumeType.TECHNICAL: [
        ("1.0", "Executive Summary", "High-level overview demonstrating understanding of requirements and proposed solution"),
        ("2.0", "Technical Approach", "Detailed methodology, tools, technologies, and technical solution design"),
        ("3.0", "Management Approach", "Project management methodology, schedule, risk mitigation, QA/QC processes"),
        ("4.0", "Transition Plan", "Transition-in approach, Day 1 readiness, knowledge transfer"),
    ],
    VolumeType.MANAGEMENT: [
        ("1.0", "Management Philosophy", "Overall management approach and organizational structure"),
        ("2.0", "Program Management", "Program/project management methodology and tools"),
        ("3.0", "Quality Assurance", "QA/QC processes, continuous improvement, metrics"),
        ("4.0", "Risk Management", "Risk identification, mitigation strategies, contingency planning"),
        ("5.0", "Staffing Approach", "Recruitment, retention, training, and development"),
    ],
    VolumeType.PAST_PERFORMANCE: [
        ("1.0", "Past Performance Summary", "Overview of relevant contract experience and qualifications"),
        ("2.0", "Contract Reference 1", "Detailed description of most relevant prior contract"),
        ("3.0", "Contract Reference 2", "Second relevant contract with comparable scope"),
        ("4.0", "Contract Reference 3", "Third relevant contract demonstrating capability"),
        ("5.0", "Past Performance Questionnaires", "PPQ submission instructions and references"),
    ],
    VolumeType.COST_PRICE: [
        ("1.0", "Cost/Price Narrative", "Basis of estimate, assumptions, and pricing methodology"),
        ("2.0", "Labor Categories & Rates", "Proposed labor categories, qualifications, and rates"),
        ("3.0", "CLIN/Task Order Pricing", "Contract Line Item pricing breakdown"),
        ("4.0", "Subcontractor Costs", "Subcontractor pricing and rationale"),
        ("5.0", "Other Direct Costs", "ODCs, travel, materials, and other costs"),
    ],
    VolumeType.STAFFING: [
        ("1.0", "Organizational Structure", "Proposed organization chart and reporting relationships"),
        ("2.0", "Key Personnel", "Key personnel qualifications, availability, and commitment"),
        ("3.0", "Resume Section", "Detailed resumes for proposed key personnel"),
        ("4.0", "Staffing Plan", "Staffing levels, phase-in, and surge capacity"),
    ],
    VolumeType.SMALL_BUSINESS: [
        ("1.0", "Small Business Participation", "Small business participation commitment and goals"),
        ("2.0", "Subcontracting Plan", "Detailed subcontracting plan per FAR 19.704"),
        ("3.0", "Mentor-Protégé Agreements", "Existing mentor-protégé relationships if applicable"),
    ],
    VolumeType.EXPERIENCE: [
        ("1.0", "Corporate Experience", "Overview of organizational experience and capabilities"),
        ("2.0", "Relevant Projects", "Detailed descriptions of similar completed projects"),
        ("3.0", "Lessons Learned", "Knowledge gained and process improvements"),
    ],
    VolumeType.ADMINISTRATIVE: [
        ("1.0", "Administrative Compliance", "Certifications, representations, and administrative forms"),
        ("2.0", "Required Forms", "Completed required forms and attachments"),
    ],
    VolumeType.OTHER: [
        ("1.0", "Section Overview", "Overview of this proposal section"),
        ("2.0", "Requirements Response", "Detailed response to section requirements"),
    ],
}


@dataclass
class ProposalSection:
    """A section within a proposal volume"""
    id: str
    name: str
    page_limit: Optional[int] = None
    requirements: List[str] = field(default_factory=list)
    eval_criteria: List[str] = field(default_factory=list)
    subsections: List['ProposalSection'] = field(default_factory=list)


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
        stats: Dict,
        company_library_data: Optional[Dict] = None
    ) -> ProposalOutline:
        """
        Generate proposal outline from compliance matrix data.

        Args:
            section_l_requirements: Extracted Section L requirements
            section_m_requirements: Extracted Section M/evaluation requirements
            technical_requirements: Extracted technical/SOW requirements
            stats: Extraction statistics
            company_library_data: Optional company library data for win theme generation

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
        """
        Create default volumes if none were extracted.

        MASTER ARCHITECT FIX: Each volume now gets volume-type-specific sections
        instead of cloning the Technical volume structure.
        """
        if rfp_format in ["GSA_BPA", "GSA_RFQ"]:
            volume_configs = [
                ("VOL-1", "Technical Approach", VolumeType.TECHNICAL, 0),
                ("VOL-2", "Past Performance", VolumeType.PAST_PERFORMANCE, 1),
                ("VOL-3", "Price", VolumeType.COST_PRICE, 2),
            ]
        else:
            volume_configs = [
                ("VOL-TECH", "Technical Proposal", VolumeType.TECHNICAL, 0),
                ("VOL-MGMT", "Management Proposal", VolumeType.MANAGEMENT, 1),
                ("VOL-PP", "Past Performance", VolumeType.PAST_PERFORMANCE, 2),
                ("VOL-COST", "Cost/Price Proposal", VolumeType.COST_PRICE, 3),
            ]

        volumes = []
        for vol_id, vol_name, vol_type, order in volume_configs:
            # Create volume with type-specific sections
            sections = self._create_sections_for_volume_type(vol_type)

            volume = ProposalVolume(
                id=vol_id,
                name=vol_name,
                volume_type=vol_type,
                order=order,
                sections=sections
            )
            volumes.append(volume)

        return volumes

    def _create_sections_for_volume_type(self, volume_type: VolumeType) -> List[ProposalSection]:
        """
        Create volume-type-specific sections using VOLUME_SECTION_TEMPLATES.

        This ensures Past Performance volumes get PP sections,
        Cost volumes get pricing sections, etc.
        """
        template = VOLUME_SECTION_TEMPLATES.get(volume_type, VOLUME_SECTION_TEMPLATES[VolumeType.OTHER])

        sections = []
        for sec_id, sec_name, sec_desc in template:
            section = ProposalSection(
                id=sec_id,
                name=sec_name,
                requirements=[sec_desc]  # Description becomes initial requirement guidance
            )
            sections.append(section)

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
                    "sections": [
                        {
                            "id": sec.id,
                            "title": sec.name,  # UI expects 'title' not 'name'
                            "name": sec.name,   # Keep both for compatibility
                            "page_limit": sec.page_limit,
                            "content_requirements": sec.requirements,  # UI expects this name
                            "requirements": sec.requirements,
                            "compliance_checkpoints": [],  # Can be populated later
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
                    "criteria": ef.criteria,
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
