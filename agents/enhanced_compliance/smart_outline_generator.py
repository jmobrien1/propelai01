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
class P0Constraint:
    """
    A pass/fail compliance gate (P0 = Priority Zero = Must Pass).

    These are "kill switch" requirements where failure to comply
    results in a non-responsive proposal determination.
    """
    id: str
    requirement: str
    section: str  # L, M, or C
    consequence: str = "Non-responsive if not met"
    verification: str = "TBD"


@dataclass
class WinTheme:
    """
    A win theme represents a competitive discriminator mapped to RFP factors.

    Per the tech spec: "What we have that they don't" + "How this helps the client"

    Win themes are the foundation of a compelling proposal - they answer
    "Why should the government choose us over competitors?"
    """
    id: str
    discriminator: str  # What makes us unique
    benefit_statement: str  # How this helps the client
    proof_points: List[str] = field(default_factory=list)  # Evidence from Company Library
    mapped_factors: List[str] = field(default_factory=list)  # Section M factor IDs this addresses


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
    p0_constraints: List[P0Constraint] = field(default_factory=list)
    win_themes: List[WinTheme] = field(default_factory=list)


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

        # Volume-specific section templates
        self.volume_templates = {
            VolumeType.TECHNICAL: [
                ("SEC-TECH-1", "Executive Summary", "High-level summary of technical approach and key differentiators"),
                ("SEC-TECH-2", "Technical Approach", "Detailed methodology and solution architecture"),
                ("SEC-TECH-3", "Understanding of Requirements", "Demonstration of RFP comprehension"),
                ("SEC-TECH-4", "Work Plan", "Project schedule, milestones, and deliverables"),
                ("SEC-TECH-5", "Quality Assurance", "Quality control processes and metrics"),
            ],
            VolumeType.MANAGEMENT: [
                ("SEC-MGMT-1", "Management Approach", "Overall management philosophy and structure"),
                ("SEC-MGMT-2", "Organizational Structure", "Team organization and reporting relationships"),
                ("SEC-MGMT-3", "Staffing Plan", "Personnel allocation and resource management"),
                ("SEC-MGMT-4", "Risk Management", "Risk identification, mitigation, and contingency planning"),
                ("SEC-MGMT-5", "Communication Plan", "Stakeholder communication and reporting"),
            ],
            VolumeType.PAST_PERFORMANCE: [
                ("SEC-PP-1", "Recent & Relevant Projects", "Projects demonstrating similar scope, size, and complexity"),
                ("SEC-PP-2", "Performance Questionnaires", "Client reference information and CPARS ratings"),
                ("SEC-PP-3", "Lessons Learned", "How past experience informs this effort"),
                ("SEC-PP-4", "Corporate Experience Summary", "Overview of organizational capabilities"),
            ],
            VolumeType.COST_PRICE: [
                ("SEC-COST-1", "Pricing Assumptions", "Basis of estimate and pricing rationale"),
                ("SEC-COST-2", "CLIN Structure", "Contract line item breakdown and pricing"),
                ("SEC-COST-3", "Labor Categories & Rates", "Labor rate justification and basis"),
                ("SEC-COST-4", "Other Direct Costs", "ODCs, travel, materials, and subcontracts"),
                ("SEC-COST-5", "Cost Narrative", "Supporting narrative for price reasonableness"),
            ],
            VolumeType.STAFFING: [
                ("SEC-STAFF-1", "Key Personnel", "Resumes and qualifications of key staff"),
                ("SEC-STAFF-2", "Organizational Chart", "Team structure and roles"),
                ("SEC-STAFF-3", "Labor Mix", "Labor category distribution and rationale"),
                ("SEC-STAFF-4", "Succession Planning", "Contingency for key personnel changes"),
            ],
            VolumeType.SMALL_BUSINESS: [
                ("SEC-SB-1", "Small Business Subcontracting Plan", "Goals and commitments by category"),
                ("SEC-SB-2", "Subcontractor Identification", "Named small business partners"),
                ("SEC-SB-3", "Good Faith Efforts", "Outreach and mentoring activities"),
            ],
            VolumeType.EXPERIENCE: [
                ("SEC-EXP-1", "Corporate Experience", "Relevant organizational experience"),
                ("SEC-EXP-2", "Contract References", "Prior contracts and performance history"),
                ("SEC-EXP-3", "Capabilities Matrix", "Mapping of capabilities to requirements"),
            ],
            VolumeType.OTHER: [
                ("SEC-OTH-1", "Overview", "Section overview and approach"),
                ("SEC-OTH-2", "Response Content", "Detailed response to requirements"),
            ],
        }

    def get_volume_sections(self, volume_type: VolumeType) -> List[ProposalSection]:
        """
        Get appropriate sections for a volume based on its type.

        This ensures each volume gets structurally appropriate sections
        rather than generic technical sections applied to all volumes.

        Args:
            volume_type: The type of volume (Technical, Cost, Past Performance, etc.)

        Returns:
            List of ProposalSection objects appropriate for this volume type
        """
        template = self.volume_templates.get(volume_type, self.volume_templates[VolumeType.OTHER])

        sections = []
        for sec_id, sec_name, sec_desc in template:
            section = ProposalSection(
                id=sec_id,
                name=sec_name,
                requirements=[sec_desc]  # Use description as guidance
            )
            sections.append(section)

        return sections

    def _apply_volume_sections(self, volumes: List[ProposalVolume]):
        """
        Apply appropriate section templates to each volume based on its type.

        This fixes the "Templating Loop Bug" where all volumes received
        identical generic sections regardless of their purpose.
        """
        for volume in volumes:
            # Only apply template if volume has no sections already
            if not volume.sections:
                # v4.2: Special handling for SF-1449 volumes in Commercial RFQs
                # If volume is "SF-1449", use simplified pricing section
                vol_name_lower = volume.name.lower()
                if "sf-1449" in vol_name_lower or "sf 1449" in vol_name_lower:
                    volume.sections = [
                        ProposalSection(
                            id="SEC-SF1449-1",
                            name="Completed SF-1449 Schedule",
                            requirements=["Complete all required blocks of SF-1449 form with pricing"]
                        )
                    ]
                elif "quote" in vol_name_lower:
                    # v4.2: Simple Quote volume for Commercial RFQs
                    volume.sections = [
                        ProposalSection(
                            id="SEC-QUOTE-1",
                            name="Company Information",
                            requirements=["Company letterhead with POC, CAGE code, Tax ID"]
                        ),
                        ProposalSection(
                            id="SEC-QUOTE-2",
                            name="Technical Capability Statement",
                            requirements=["Brief description of relevant experience and qualifications"]
                        )
                    ]
                else:
                    volume.sections = self.get_volume_sections(volume.volume_type)
    
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
            company_library_data: Optional Company Library profile data containing
                differentiators, capabilities, and past performance for win theme generation

        Returns:
            ProposalOutline with volumes, eval factors, format requirements, and win themes
        """
        warnings = []

        # v4.2: Bridge extraction-structure gap by including ALL requirements
        # in format/volume detection, not just section L/M.
        # Volume instructions like "Volume 1: Quote" may appear in any section.
        all_requirements = section_l_requirements + section_m_requirements + technical_requirements

        # Detect RFP format (now checks all requirements for SF-1449/RFQ markers)
        rfp_format = self._detect_rfp_format(
            all_requirements,  # Pass all requirements, not just section_l
            section_m_requirements,
            stats
        )

        # Extract volumes based on format (searches all requirements for volume headers)
        volumes = self._extract_volumes(
            all_requirements,  # Pass all requirements to find volume instructions
            section_m_requirements,
            rfp_format
        )

        # If no volumes found, create defaults
        if not volumes:
            volumes = self._create_default_volumes(rfp_format, section_m_requirements)
            warnings.append("No explicit volumes found - using default structure")

        # Apply volume-specific section templates
        # This ensures Cost volumes get cost sections, PP volumes get PP sections, etc.
        self._apply_volume_sections(volumes)

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

        # Extract P0 Constraints (pass/fail compliance gates)
        p0_constraints = self._extract_p0_constraints(
            section_l_requirements,
            section_m_requirements,
            technical_requirements
        )

        # Generate Win Themes from Company Library differentiators
        win_themes = self._generate_win_themes(
            eval_factors,
            company_library_data
        )

        if not win_themes and company_library_data:
            warnings.append("No differentiators found in Company Library for win theme generation")

        return ProposalOutline(
            rfp_format=rfp_format,
            volumes=volumes,
            eval_factors=eval_factors,
            format_requirements=format_req,
            submission_info=submission,
            warnings=warnings,
            total_pages=total_pages,
            p0_constraints=p0_constraints,
            win_themes=win_themes
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

        # v4.2: Check for Commercial RFQ (SF-1449) with explicit volume headers
        # This MUST come before default template to prevent "Volume Trap"
        # Pattern: "Volume 1: Quote", "Volume 2: SF-1449", etc.
        explicit_volume_pattern = re.search(
            r"volume\s*[12ivI]+\s*[:\-–]\s*(?:quote|sf[\-\s]?1449|pricing|administrative)",
            all_text, re.IGNORECASE
        )
        is_commercial_rfq = (
            "sf 1449" in all_text or
            "sf-1449" in all_text or
            "sf1449" in all_text or
            ("rfq" in all_text and "commercial item" in all_text) or
            ("request for quot" in all_text and explicit_volume_pattern)
        )
        if is_commercial_rfq and explicit_volume_pattern:
            return "COMMERCIAL_RFQ"

        # Check for VA Commercial Services format (FAR 52.212-2, Section E.2)
        # Pattern: "Factor 1 Technical", "Factor 2 Past Performance", etc.
        va_factor_pattern = re.search(
            r"(?:e\.?2|52\.212-2).*?evaluation.*?factor\s*1.*?factor\s*2",
            all_text, re.DOTALL
        )
        if va_factor_pattern or ("comparative evaluation" in all_text and "factor 1" in all_text):
            return "VA_COMMERCIAL"

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
        if rfp_format == "COMMERCIAL_RFQ":
            volumes = self._extract_commercial_rfq_volumes(all_text, section_l)
        elif rfp_format == "VA_COMMERCIAL":
            volumes = self._extract_va_commercial_volumes(all_text, section_m)
        elif rfp_format == "NIH_FACTOR":
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

    def _extract_va_commercial_volumes(self, text: str, section_m: List[Dict]) -> List[ProposalVolume]:
        """
        Extract volumes from VA Commercial Services RFP (FAR 52.212-2).

        VA commercial solicitations typically use a Factor-based evaluation
        structure in Section E.2, with factors like:
        - Factor 1: Technical (with subfactors)
        - Factor 2: Past Performance
        - Factor 3: SDVOSB/VOSB Status
        - Factor 4: Cost/Price
        """
        volumes = []
        text_lower = text.lower()

        # VA commercial services typically have single quote/proposal structure
        main_volume = ProposalVolume(
            id="VOL-QUOTE",
            name="Quote Response",
            volume_type=VolumeType.TECHNICAL
        )

        # Extract factors from text using multiple patterns
        factor_names = {}

        # Pattern 1: "Factor N [Name]" or "Factor N: [Name]" or "Factor N. [Name]"
        factor_patterns = [
            r"factor\s*(\d+)[\.\:\s]+([a-z][a-z\s\-\/]+?)(?=factor\s*\d|$|\n|\.\.)",
            r"(\d+)\.\s*factor\s*(\d+)[:\s]+([a-z][a-z\s\-\/]+)",
        ]

        for pattern in factor_patterns:
            for match in re.finditer(pattern, text_lower):
                if len(match.groups()) == 2:
                    factor_num = match.group(1)
                    factor_name = match.group(2).strip()
                elif len(match.groups()) == 3:
                    factor_num = match.group(2)
                    factor_name = match.group(3).strip()
                else:
                    continue

                # Clean up factor name
                factor_name = re.sub(r'[\s\-]+$', '', factor_name)
                factor_name = factor_name.title()

                # Skip garbage/short names
                if len(factor_name) >= 4 and factor_name.lower() not in ['the', 'this', 'that']:
                    key = f"factor_{factor_num}"
                    if key not in factor_names or len(factor_name) > len(factor_names[key]):
                        factor_names[key] = factor_name

        # Also look for common VA factor names explicitly
        va_default_factors = {
            "1": "Technical Factors",
            "2": "Past Performance",
            "3": "SDVOSB/VOSB Status",
            "4": "Cost/Price",
        }

        # Check for specific VA patterns
        if "technical experience" in text_lower or "jci" in text_lower or "johnson controls" in text_lower:
            factor_names["factor_1"] = "Technical Factors"
        if "past performance" in text_lower and "references" in text_lower:
            factor_names["factor_2"] = "Past Performance"
        if "sdvosb" in text_lower or "vosb" in text_lower:
            factor_names["factor_3"] = "SDVOSB/VOSB Status"
        if "cost" in text_lower or "price" in text_lower:
            factor_names["factor_4"] = "Cost/Price"

        # Build sections from factors
        tech_sections = []
        for factor_key in sorted(factor_names.keys()):
            factor_num = factor_key.replace("factor_", "")
            factor_name = factor_names[factor_key]

            section = ProposalSection(
                id=f"SEC-F{factor_num}",
                name=f"Factor {factor_num}: {factor_name}"
            )

            # Add subfactors for Factor 1 (Technical) if applicable
            if factor_num == "1":
                subfactors = self._extract_va_subfactors(text_lower)
                if subfactors:
                    section.subsections = subfactors

            tech_sections.append(section)

        main_volume.sections = tech_sections
        volumes.append(main_volume)

        return volumes

    def _extract_va_subfactors(self, text_lower: str) -> List[ProposalSection]:
        """Extract subfactors for VA Technical Factor (1.1, 1.2, etc.)"""
        subfactors = []

        # Common VA subfactor patterns
        subfactor_patterns = [
            (r"1\.1\.?\s*technical\s+experience", "Technical Experience"),
            (r"1\.2\.?\s*understanding", "Understanding of the Project"),
            (r"1\.3\.?\s*experience\s+and\s+qualification", "Experience and Qualification"),
            (r"1\.4\.?\s*contingency", "Contingency Plan"),
        ]

        for pattern, name in subfactor_patterns:
            if re.search(pattern, text_lower):
                subfactors.append(ProposalSection(
                    id=f"SEC-{name.replace(' ', '-').upper()[:10]}",
                    name=name
                ))

        return subfactors

    def _extract_commercial_rfq_volumes(self, text: str, section_l: List[Dict]) -> List[ProposalVolume]:
        """
        Extract volumes from Commercial RFQ (SF-1449) with explicit volume headers.

        v4.2: Prevents "Volume Trap" by using EXACT volume names from RFP instead
        of defaulting to Shipley template. Example:
        - "Volume 1: Quote" -> Volume 1: Quote
        - "Volume 2: SF-1449" -> Volume 2: SF-1449

        This is critical for compliance - using wrong volume structure = non-responsive.
        """
        volumes = []

        # v4.2.1 FIX: Simple regex that captures ONLY the first word/phrase after colon
        # This prevents the "Frankenstein Volume" bug where entire paragraphs were captured.
        # Examples:
        #   "Volume 1: Quote" -> ("1", "Quote")
        #   "Volume 2: SF-1449" -> ("2", "SF-1449")
        #   "Volume 1: Quote\nQuotes shall be provided..." -> ("1", "Quote") NOT the whole paragraph
        volume_pattern = r"volume\s*([12ivI]+)\s*[:\-–]\s*([A-Za-z0-9][-A-Za-z0-9]*)"

        matches = list(re.finditer(volume_pattern, text, re.IGNORECASE))

        for match in matches:
            vol_num = match.group(1).strip().upper()
            vol_name = match.group(2).strip()

            # v4.2.1: With new simple regex, vol_name is just one word like "Quote" or "SF-1449"
            # Skip if volume name is empty
            if not vol_name:
                continue

            # Classify the volume type
            vol_type = VolumeType.OTHER
            name_lower = vol_name.lower()
            if "quote" in name_lower or "quotation" in name_lower:
                vol_type = VolumeType.ADMINISTRATIVE
            elif "sf" in name_lower or "1449" in name_lower or "pricing" in name_lower:
                vol_type = VolumeType.COST_PRICE
            elif "technical" in name_lower:
                vol_type = VolumeType.TECHNICAL
            elif "past" in name_lower or "performance" in name_lower:
                vol_type = VolumeType.PAST_PERFORMANCE

            # Create the volume with EXACT name from RFP
            vol = ProposalVolume(
                id=f"VOL-{vol_num}",
                name=f"Volume {vol_num}: {vol_name}",
                volume_type=vol_type
            )
            volumes.append(vol)

        # Deduplicate volumes by volume number
        seen_nums = set()
        unique_volumes = []
        for vol in volumes:
            vol_num = vol.id
            if vol_num not in seen_nums:
                seen_nums.add(vol_num)
                unique_volumes.append(vol)
        volumes = unique_volumes

        # If we found explicit volumes, don't add any defaults
        if volumes:
            return volumes

        # Fallback: Look for simpler patterns like "Volume 1" and "Volume 2"
        simple_volume_pattern = r"volume\s*([12ivI]+)\b"
        simple_matches = list(re.finditer(simple_volume_pattern, text, re.IGNORECASE))

        if len(simple_matches) >= 2:
            # Check context to determine volume purposes
            for match in simple_matches[:2]:
                vol_num = match.group(1).strip().upper()
                # Get surrounding context
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 50)
                context = text[start:end].lower()

                vol_name = f"Volume {vol_num}"
                vol_type = VolumeType.OTHER

                if "quote" in context:
                    vol_name = f"Volume {vol_num}: Quote"
                    vol_type = VolumeType.ADMINISTRATIVE
                elif "sf" in context or "1449" in context or "price" in context:
                    vol_name = f"Volume {vol_num}: SF-1449"
                    vol_type = VolumeType.COST_PRICE

                vol = ProposalVolume(
                    id=f"VOL-{vol_num}",
                    name=vol_name,
                    volume_type=vol_type
                )
                volumes.append(vol)

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

        if rfp_format == "COMMERCIAL_RFQ":
            # v4.2: Simple structure for Commercial RFQs - don't over-engineer
            return [
                ProposalVolume(id="VOL-1", name="Volume 1: Quote", volume_type=VolumeType.ADMINISTRATIVE, order=0),
                ProposalVolume(id="VOL-2", name="Volume 2: Pricing", volume_type=VolumeType.COST_PRICE, order=1),
            ]
        elif rfp_format in ["GSA_BPA", "GSA_RFQ"]:
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

    def _detect_rating_type(self, text: str) -> Optional[str]:
        """
        Detect evaluation rating methodology from Section M text.

        Returns:
            "Adjectival" if adjectival ratings are used (Outstanding/Acceptable/etc.)
            "Numerical" if point-based scoring is used
            None if unable to determine
        """
        text_lower = text.lower()

        # Adjectival rating patterns (common federal rating scales)
        adjectival_patterns = [
            r"outstanding",
            r"exceptional",
            r"very\s+good",
            r"good\s+rating",
            r"satisfactory",
            r"acceptable",
            r"marginal",
            r"unacceptable",
            r"adjectival\s+rating",
            r"color\s+rating",
            r"blue.*green.*yellow.*red",  # Color-coded ratings
            r"rated\s+as\s+(?:outstanding|exceptional|acceptable)",
        ]

        # Count adjectival indicators
        adjectival_count = sum(
            1 for p in adjectival_patterns
            if re.search(p, text_lower)
        )

        # Numerical rating patterns
        numerical_patterns = [
            r"\d+\s*(?:points?|pts)",
            r"maximum\s+(?:of\s+)?\d+\s*points?",
            r"(\d+)\s*out\s+of\s+(\d+)",
            r"weighted\s+(?:at\s+)?\d+%?",
            r"total\s+(?:possible\s+)?points?[:\s]+\d+",
        ]

        numerical_count = sum(
            1 for p in numerical_patterns
            if re.search(p, text_lower)
        )

        # Determine rating type based on patterns found
        if adjectival_count >= 2:
            return "Adjectival"
        elif numerical_count >= 2:
            return "Numerical"
        elif adjectival_count > 0 and numerical_count == 0:
            return "Adjectival"
        elif numerical_count > 0 and adjectival_count == 0:
            return "Numerical"

        return None

    def _extract_importance_order(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract relative importance ordering of evaluation factors.

        Common patterns:
        - "Technical is more important than Cost"
        - "Factors listed in descending order of importance"
        - "Technical and Management are of equal importance"

        Returns:
            List of tuples (factor_name, importance_level)
            e.g., [("Technical", "Most Important"), ("Cost", "Less Important")]
        """
        importance = []
        text_lower = text.lower()

        # Pattern: "X is more important than Y"
        more_important = re.findall(
            r"(technical|management|cost|price|past\s*performance|experience|personnel)"
            r"\s+(?:is|are)\s+(?:more|most)\s+important\s+than\s+"
            r"(technical|management|cost|price|past\s*performance|experience|personnel)",
            text_lower
        )
        for higher, lower in more_important:
            importance.append((higher.strip().title(), "More Important"))
            importance.append((lower.strip().title(), "Less Important"))

        # Pattern: "descending order of importance"
        if "descending order of importance" in text_lower:
            # The first factor mentioned is most important
            factors = re.findall(
                r"factor\s*\d+[,:\s]+([a-z\s]+?)(?:\.|,|factor)",
                text_lower
            )
            for i, factor in enumerate(factors[:5]):  # Limit to first 5
                level = "Most Important" if i == 0 else f"#{i+1} Important"
                importance.append((factor.strip().title(), level))

        # Pattern: "equally important"
        equal_match = re.search(
            r"(technical|management|cost|price|past\s*performance)\s+and\s+"
            r"(technical|management|cost|price|past\s*performance)\s+"
            r"(?:are\s+)?(?:of\s+)?equal(?:ly)?\s+import",
            text_lower
        )
        if equal_match:
            importance.append((equal_match.group(1).strip().title(), "Equal"))
            importance.append((equal_match.group(2).strip().title(), "Equal"))

        # Pattern: "significantly more important"
        sig_more = re.findall(
            r"(technical|management|past\s*performance)\s+(?:is|are)\s+"
            r"significantly\s+more\s+important",
            text_lower
        )
        for factor in sig_more:
            importance.append((factor.strip().title(), "Significantly More Important"))

        return importance

    def _get_factor_importance(
        self,
        factor_name: str,
        importance_order: List[Tuple[str, str]]
    ) -> Optional[str]:
        """Get the importance level for a specific factor."""
        factor_lower = factor_name.lower()

        for name, level in importance_order:
            if name.lower() in factor_lower or factor_lower in name.lower():
                return level

        return None

    def _extract_eval_factors(self, section_m: List[Dict]) -> List[EvaluationFactor]:
        """Extract evaluation factors from Section M"""
        factors = []
        seen = set()

        all_text = " ".join([
            r.get("text", "") or r.get("full_text", "")
            for r in section_m
        ])

        # Detect rating methodology (adjectival vs. numerical)
        rating_type = self._detect_rating_type(all_text)

        # Detect relative importance ordering
        importance_order = self._extract_importance_order(all_text)

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

                    # Determine weight based on rating type
                    weight = rating_type if rating_type else None

                    # Check if this factor has a specified importance
                    importance = self._get_factor_importance(factor_name, importance_order)

                    factor = EvaluationFactor(
                        id=f"EVAL-F{factor_num}",
                        name=f"Factor {factor_num}: {factor_name}",
                        weight=weight,
                        importance=importance
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
                        importance = self._get_factor_importance(name, importance_order)
                        factor = EvaluationFactor(
                            id=f"EVAL-F{num}",
                            name=f"Factor {num}: {name}",
                            weight=rating_type if rating_type else None,
                            importance=importance
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
                    importance = self._get_factor_importance(name, importance_order)
                    factor = EvaluationFactor(
                        id=eval_id,
                        name=name,
                        weight=rating_type if rating_type else None,
                        importance=importance
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

    def _extract_p0_constraints(
        self,
        section_l: List[Dict],
        section_m: List[Dict],
        technical: List[Dict]
    ) -> List[P0Constraint]:
        """
        Extract P0 (Priority Zero) pass/fail compliance constraints.

        These are "kill switch" requirements that result in a non-responsive
        determination if not met. Identified by pass/fail language patterns.
        """
        constraints = []
        constraint_id = 1

        # Patterns that indicate pass/fail requirements
        pass_fail_patterns = [
            r"shall\s+(?:be|include|provide|submit|demonstrate|have)",
            r"must\s+(?:be|include|provide|submit|demonstrate|have)",
            r"required\s+to\s+(?:be|include|provide|submit)",
            r"mandatory\s+(?:requirement|submission|inclusion)",
            r"failure\s+to\s+(?:comply|meet|provide|submit)\s+(?:will|shall)\s+result",
            r"non[-\s]?responsive\s+if",
            r"proposals?\s+(?:will|shall)\s+(?:be\s+)?(?:rejected|disqualified)",
            r"minimum\s+(?:requirement|qualification|standard)",
            r"go[-/]no[-]?go",
            r"pass[-/]fail",
        ]

        # Gatekeeper patterns - authorization/certification requirements
        # These are "kill switch" requirements where missing documentation = disqualification
        gatekeeper_patterns = [
            r"(?:authorized|certified|licensed)\s+(?:by|to|from)\s+(?:the\s+)?(?:manufacturer|oem|vendor)",
            r"documentation\s+from\s+[A-Z][a-zA-Z\s]+(?:Inc\.?|LLC|Corp)",  # Documentation from Company Inc.
            r"(?:jci|johnson\s+controls)\s+(?:authorized|certified|documentation)",
            r"oem\s+(?:authorization|certification|documentation)",
            r"manufacturer['']?s?\s+(?:authorization|certification|letter)",
            r"shall\s+(?:be|provide)\s+(?:a\s+)?(?:certified|authorized|licensed)",
            r"proof\s+of\s+(?:authorization|certification|licensing)",
            r"letterhead.*(?:authorization|certified)",
        ]

        # Combine all patterns into compiled regexes
        combined_pattern = re.compile(
            "|".join(pass_fail_patterns),
            re.IGNORECASE
        )
        gatekeeper_pattern = re.compile(
            "|".join(gatekeeper_patterns),
            re.IGNORECASE
        )

        # Process all requirement sources
        all_requirements = [
            (section_l, "L"),
            (section_m, "M"),
            (technical, "C"),
        ]

        for requirements, section_code in all_requirements:
            for req in requirements:
                req_text = req.get("text", "") or req.get("full_text", "") or ""
                req_text_lower = req_text.lower()

                # Check for GATEKEEPER requirements first (highest priority)
                is_gatekeeper = gatekeeper_pattern.search(req_text)

                # Check if this looks like a pass/fail requirement
                if is_gatekeeper or combined_pattern.search(req_text):
                    # Determine consequence based on language
                    consequence = "Non-responsive if not met"
                    if "reject" in req_text_lower:
                        consequence = "Proposal will be rejected"
                    elif "disqualif" in req_text_lower:
                        consequence = "Offeror will be disqualified"

                    # Gatekeeper requirements get special flagging
                    if is_gatekeeper:
                        consequence = "CRITICAL GATEKEEPER: " + consequence
                        # Add specific verification requirement
                        verification = "MUST INCLUDE: Authorization/certification documentation required"

                        # Check for specific OEM mentions
                        if "jci" in req_text_lower or "johnson controls" in req_text_lower:
                            verification = "MUST INCLUDE: JCI (Johnson Controls) authorization letter on company letterhead"
                    else:
                        verification = "Compliance review required"

                    # Create constraint
                    constraint = P0Constraint(
                        id=f"P0-{constraint_id:03d}",
                        requirement=req_text[:500],  # Truncate long requirements
                        section=section_code,
                        consequence=consequence,
                        verification=verification
                    )
                    constraints.append(constraint)
                    constraint_id += 1

        return constraints

    def _generate_win_themes(
        self,
        eval_factors: List[EvaluationFactor],
        company_library_data: Optional[Dict]
    ) -> List[WinTheme]:
        """
        Generate win themes by mapping Company Library differentiators to evaluation factors.

        Win themes answer the question: "Why should the government choose us?"
        They connect our unique strengths (differentiators) to what the government
        values (evaluation factors).

        Args:
            eval_factors: Extracted evaluation factors from Section M
            company_library_data: Company Library profile data with differentiators

        Returns:
            List of WinTheme objects mapped to evaluation factors
        """
        win_themes = []

        if not company_library_data:
            return win_themes

        # Extract differentiators from Company Library data
        differentiators = company_library_data.get("differentiators", [])

        # Also extract capabilities as potential win themes
        capabilities = company_library_data.get("capabilities", [])

        # Build a list of factor keywords for matching
        factor_keywords = {}
        for ef in eval_factors:
            factor_name_lower = ef.name.lower()
            keywords = []

            # Extract keywords from factor names
            if "technical" in factor_name_lower:
                keywords.extend(["technical", "approach", "solution", "methodology", "innovation"])
            if "management" in factor_name_lower or "program" in factor_name_lower:
                keywords.extend(["management", "program", "project", "leadership", "governance"])
            if "experience" in factor_name_lower or "past performance" in factor_name_lower:
                keywords.extend(["experience", "past performance", "track record", "proven", "success"])
            if "personnel" in factor_name_lower or "staff" in factor_name_lower:
                keywords.extend(["personnel", "staff", "team", "expertise", "qualifications"])
            if "cost" in factor_name_lower or "price" in factor_name_lower:
                keywords.extend(["cost", "price", "value", "efficiency", "savings"])

            factor_keywords[ef.id] = keywords

        theme_id = 1

        # Process differentiators
        for diff in differentiators:
            diff_title = diff.get("title", "") or diff.get("name", "")
            diff_desc = diff.get("description", "")
            diff_evidence = diff.get("evidence", [])

            if not diff_title:
                continue

            # Create benefit statement from description or generate generic one
            if diff_desc:
                benefit = self._extract_benefit_statement(diff_desc)
            else:
                benefit = f"Enables superior performance through {diff_title.lower()}"

            # Map to evaluation factors
            mapped_factors = self._map_to_factors(
                diff_title + " " + diff_desc,
                factor_keywords
            )

            # Create win theme
            theme = WinTheme(
                id=f"WT-{theme_id:03d}",
                discriminator=diff_title[:200],
                benefit_statement=benefit[:300],
                proof_points=diff_evidence[:5] if isinstance(diff_evidence, list) else [],
                mapped_factors=mapped_factors
            )
            win_themes.append(theme)
            theme_id += 1

        # If no differentiators, use top capabilities as win themes
        if not win_themes and capabilities:
            for cap in capabilities[:5]:  # Limit to top 5
                cap_name = cap.get("name", "")
                cap_desc = cap.get("description", "")

                if not cap_name:
                    continue

                benefit = self._extract_benefit_statement(cap_desc) if cap_desc else f"Delivers value through {cap_name.lower()}"

                mapped_factors = self._map_to_factors(
                    cap_name + " " + cap_desc,
                    factor_keywords
                )

                theme = WinTheme(
                    id=f"WT-{theme_id:03d}",
                    discriminator=cap_name[:200],
                    benefit_statement=benefit[:300],
                    proof_points=[],
                    mapped_factors=mapped_factors
                )
                win_themes.append(theme)
                theme_id += 1

        return win_themes

    def _extract_benefit_statement(self, description: str) -> str:
        """
        Extract or generate a benefit statement from a description.

        Looks for phrases indicating benefits (reduces, improves, enables, etc.)
        or generates a generic benefit statement.
        """
        # Patterns for benefit language
        benefit_patterns = [
            r"(?:which|that)\s+(enables?|improves?|reduces?|delivers?|provides?|ensures?)[^.]+",
            r"(?:resulting in|leading to)\s+([^.]+)",
            r"(improves?|reduces?|enables?|ensures?|delivers?)\s+[^.]+",
        ]

        for pattern in benefit_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                benefit = match.group(0).strip()
                # Capitalize first letter
                return benefit[0].upper() + benefit[1:]

        # Fallback: use first sentence or first 100 chars
        first_sentence = re.split(r'[.!?]', description)[0].strip()
        if len(first_sentence) > 20:
            return first_sentence[:200]

        return description[:200] if description else "Provides competitive advantage"

    def _map_to_factors(
        self,
        text: str,
        factor_keywords: Dict[str, List[str]]
    ) -> List[str]:
        """
        Map text content to evaluation factors based on keyword matching.

        Returns list of factor IDs that the text is relevant to.
        """
        text_lower = text.lower()
        matched_factors = []

        for factor_id, keywords in factor_keywords.items():
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count >= 1:
                matched_factors.append(factor_id)

        return matched_factors

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

        # Month name pattern for reuse
        month_names = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

        # Comprehensive date patterns (ordered by specificity)
        date_patterns = [
            # Full datetime with timezone: "2:00 PM ET on January 15, 2025"
            (r"(\d{1,2}:\d{2}\s*(?:AM|PM)?\s*(?:ET|EST|EDT|PT|PST|PDT|CT|CST|CDT|MT|MST|MDT)?\s*(?:on\s+)?)" +
             month_names + r"\s+\d{1,2},?\s+\d{4}"),
            # Written month format: "January 15, 2025" or "15 January 2025"
            (month_names + r"\s+\d{1,2},?\s+\d{4}"),
            (r"\d{1,2}\s+" + month_names + r",?\s+\d{4}"),
            # Military format: "15 JAN 2025"
            (r"\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}"),
            # ISO format: "2025-01-15"
            (r"\d{4}-\d{2}-\d{2}"),
            # US format: "01/15/2025" or "01-15-2025"
            (r"\d{1,2}[/-]\d{1,2}[/-]\d{4}"),
            # Short year: "01/15/25"
            (r"\d{1,2}[/-]\d{1,2}[/-]\d{2}"),
        ]

        # Context patterns that indicate a due date follows
        due_date_contexts = [
            r"(?:proposal|quote|quotation|response)\s+due[:\s]+",
            r"due\s+(?:date|by|on)[:\s]+",
            r"submission\s+(?:date|deadline)[:\s]+",
            r"closing\s+date[:\s]+",
            r"no\s+later\s+than[:\s]+",
            r"proposals?\s+(?:shall|must)\s+be\s+(?:received|submitted)\s+(?:by|no\s+later\s+than)[:\s]+",
            r"deadline[:\s]+",
            r"responses?\s+due[:\s]+",
        ]

        # Try context-aware extraction first
        for context in due_date_contexts:
            for date_pattern in date_patterns:
                full_pattern = context + r"\s*(" + date_pattern + r")"
                match = re.search(full_pattern, all_text, re.IGNORECASE)
                if match:
                    submission.due_date = match.group(1).strip()
                    break
            if submission.due_date:
                break

        # Fallback: try date patterns without context
        if not submission.due_date:
            for date_pattern in date_patterns:
                match = re.search(r"(" + date_pattern + r")", all_text, re.IGNORECASE)
                if match:
                    # Validate it looks like a future date (not a contract date)
                    date_str = match.group(1)
                    # Skip if it looks like a contract number
                    if not re.search(r"[A-Z]{2}\d{4}", date_str):
                        submission.due_date = date_str.strip()
                        break

        # Extract time separately if not already in due_date
        if submission.due_date and not re.search(r"\d{1,2}:\d{2}", submission.due_date):
            time_patterns = [
                r"(\d{1,2}:\d{2}\s*(?:AM|PM))\s*(?:ET|EST|EDT|PT|PST|PDT|CT|CST|CDT|MT|MST|MDT)?",
                r"(\d{1,2}:\d{2})\s*(?:hours?|hrs?)",
            ]
            for time_pattern in time_patterns:
                time_match = re.search(time_pattern, all_text, re.IGNORECASE)
                if time_match:
                    submission.due_time = time_match.group(1).strip()
                    break

        # Look for submission method
        method_patterns = [
            r"submit\s+(?:via|through|to)\s+(email|portal|sam\.gov|electronic)",
            r"(electronic\s+submission)",
            r"submit\s+(?:electronically\s+)?(?:via|to)\s+([\w\.]+)",
            r"proposals?\s+shall\s+be\s+(?:submitted|delivered)\s+(?:via|through|to)\s+(\w+)",
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
            "warnings": outline.warnings,
            "p0_constraints": [
                {
                    "id": p0.id,
                    "requirement": p0.requirement,
                    "section": p0.section,
                    "consequence": p0.consequence,
                    "verification": p0.verification
                }
                for p0 in outline.p0_constraints
            ],
            "win_themes": [
                {
                    "id": wt.id,
                    "discriminator": wt.discriminator,
                    "benefit_statement": wt.benefit_statement,
                    "proof_points": wt.proof_points,
                    "mapped_factors": wt.mapped_factors
                }
                for wt in outline.win_themes
            ]
        }
