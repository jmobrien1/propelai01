"""
PropelAI Enhanced CTM Extractor v3.0
Intelligent extraction of compliance metadata from RFP text

This module provides the extraction logic to populate the enhanced CTM data models.
It detects:
- Scoring types (Pass/Fail, Weighted, Qualitative)
- Response formats (Checkbox, Narrative, Table)
- Page limits and formatting requirements
- Key personnel requirements
- Future diligence flags for EA/RFR
- Compliance constraints

Author: PropelAI Team
Version: 3.0.0
Date: November 28, 2025
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .ctm_data_models import (
    EnhancedRequirement,
    ScoringType,
    ResponseFormat,
    RequirementType,
    RFPSection,
    PageLimit,
    FormattingRequirement,
    EvidenceRequirement,
    KeyPersonnelRequirement,
    ComplianceMatrix,
    create_pass_fail_requirement,
    create_weighted_requirement,
    create_future_diligence_requirement
)


# =============================================================================
# PATTERN LIBRARIES
# =============================================================================

class ScoringPatterns:
    """Regex patterns for detecting scoring type."""
    
    # Pass/Fail indicators
    PASS_FAIL_PATTERNS = [
        r'\b(pass[/-]fail|pass or fail)\b',
        r'\b(mandatory|required)\s+compliance\b',
        r'\b(non-?responsive|disqualif)',
        r'\bfailure to\s+.+\s+will result in\s+.*(rejection|disqualification|non-responsive)',
        r'\bminimum\s+requirement',
        r'\b(go[/-]no[/-]go|go or no-go)\b',
        r'\bacceptable[/-]unacceptable\b',
        r'\b(comply|compliant)\s+or\s+(reject|non-responsive)',
    ]
    
    # Weighted/Scored indicators - check these BEFORE pass/fail
    WEIGHTED_PATTERNS = [
        r'(?:maximum|max\.?)\s*points?[:\s]*(\d+)',           # Maximum Points: 40
        r'points?[:\s]*(\d+)',                                 # Points: 40
        r'(\d+)\s*(?:maximum\s+)?points?\b',                   # 40 points, 40 maximum points
        r'(?:worth|valued?\s+at|weighted)\s*(?:up\s+to\s+)?(\d+)\s*points?',
        r'(?:maximum|max\.?|up\s+to)\s+(\d+)\s*points?',
        r'(\d+)\s*(?:pts?)\s*(?:max|maximum)?',               # 100 pts max
        r'points?\s*(?:possible|available|maximum)[:\s]*(\d+)',
    ]
    
    # Qualitative indicators
    QUALITATIVE_PATTERNS = [
        r'\b(outstanding|excellent|good|acceptable|marginal|unacceptable)\b',
        r'\b(exceptional|very good|satisfactory|unsatisfactory)\b',
        r'\b(blue|green|yellow|red)\s+rating',
        r'\bstrengths?\s+and\s+weaknesses?\b',                # "strengths and weaknesses"
        r'\b(identify|assess)\s+.*(strength|weakness)',       # "identify strengths"
        r'\badjectival\s+rating',
    ]


class ResponseFormatPatterns:
    """Regex patterns for detecting required response format."""
    
    CHECKBOX_PATTERNS = [
        r'\b(check|mark|indicate)\s+(the\s+)?box',
        r'\bagree\s+.{0,10}\s*disagree\b',                    # Agree/Disagree with flexibility
        r'\byes\s*[/,]\s*no\b',
        r'\bcheck\s+if\s+(applicable|yes|agree)',
        r'\b(confirm|certify|acknowledge)\s+by\s+(checking|marking)',
        r'(☐|☑|□|■)',                                         # Actual checkbox characters
        r'\[\s*\]',                                            # [ ] style checkbox
    ]
    
    CHECKBOX_WITH_EVIDENCE_PATTERNS = [
        r'(met|compliant).*(page|section|reference)',
        r'(indicate|provide)\s+(the\s+)?(page|section)\s+(number|reference)',
        r'cross[- ]?reference',
        r'(cite|identify)\s+(where|the\s+location)',
        r'proposal\s+(page|section)\s+(number|reference)',
    ]
    
    TABLE_PATTERNS = [
        r'\b(complete|fill\s+(in|out)|populate)\s+(the\s+)?(following\s+)?table',
        r'\buse\s+(the\s+)?(following\s+)?table\s+format',
        r'\b(format|present|submit)\s+.+\s+(in|as)\s+a?\s*table',
        r'\btabular\s+format',
        r'\btable\s+\d+',                                     # Table 5
        r'\busing\s+table\s+\d+',                             # using Table 5
        r'(labor|rate|pricing)\s+(categories|table)',         # labor categories table
    ]
    
    APPENDIX_PATTERNS = [
        r'\b(as\s+an?\s+)?appendix',
        r'\b(attach|include)\s+as\s+(a\s+)?separate',
        r'\bseparate\s+(volume|document|attachment)',
        r'\battachment\s+[a-z0-9]+',
    ]
    
    RESUME_PATTERNS = [
        r'\bresume',
        r'\bcurriculum\s+vitae\b',
        r'\bbiographical\s+(sketch|information)',
        r'\bkey\s+personnel.*(resume|cv|bio)',
    ]


class PageLimitPatterns:
    """Regex patterns for detecting page limits."""
    
    PATTERNS = [
        r'(?:not\s+(?:to\s+)?exceed|maximum\s+of?|limited?\s+to|no\s+more\s+than|up\s+to)\s*(\d+)\s*pages?',
        r'(\d+)\s*pages?\s*(?:maximum|max\.?|limit)',
        r'page\s+limit[:\s]*(\d+)',
        r'(\d+)\s*(?:single[- ]?sided|double[- ]?sided)\s*pages?',
        r'(\d+)\s*pages?\s*(?:or\s+less|total)',
        r'(?:maximum|max\.?)[:\s]*(\d+)\s*pages?',            # Maximum: 50 pages
        r':\s*(\d+)\s*pages?\b',                              # : 50 pages
    ]
    
    DOUBLE_SIDED_PATTERNS = [
        r'double[- ]?sided',
        r'(\d+)\s*double[- ]?sided\s*(?:pages?)?(?:\s*or\s*(\d+)\s*single[- ]?sided)?',
    ]
    
    EXCLUSION_PATTERNS = [
        r'(?:exclud(?:es?|ing)|not\s+(?:counting|including)|does\s+not\s+(?:count|include))\s*[:\s]*(.*?)(?:\.|$)',
        r'(?:resumes?|cvs?|appendix|appendices|cover\s+page|table\s+of\s+contents|attachments?)\s+(?:do\s+)?not\s+count',
    ]


class FormattingPatterns:
    """Regex patterns for detecting formatting requirements."""
    
    FONT_PATTERNS = [
        r'(\d+)[- ]?point\s+(?:font|type)',
        r'font\s+size[:\s]*(\d+)',
        r'minimum\s+(?:font\s+)?(?:size\s+)?(?:of\s+)?(\d+)\s*(?:pt|point)',
        r'(times\s+new\s+roman|arial|calibri|courier)',
    ]
    
    MARGIN_PATTERNS = [
        r'(\d+(?:\.\d+)?)[- ]?inch\s+margins?',
        r'margins?\s*(?:of\s+)?(?:at\s+least\s+)?(\d+(?:\.\d+)?)\s*(?:inch|")',
        r'minimum\s+(?:margin\s+)?(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:inch|")',
    ]
    
    SPACING_PATTERNS = [
        r'(single|double|1\.5)[- ]?spac(?:ed?|ing)',
        r'line\s+spacing[:\s]*(single|double|1\.5)',
    ]


class FutureDiligencePatterns:
    """Patterns for detecting EA/RFR deferred requirements."""
    
    PATTERNS = [
        r'(?:defined|specified|provided|determined)\s+(?:in|by)\s+(?:the\s+)?(?:individual\s+)?(?:request|rfq|task\s+order)',
        r'(?:subsequent|future|individual)\s+(?:rfqs?|task\s+orders?|delivery\s+orders?)',
        r'(?:that\s+information|details?|requirements?)\s+(?:may|will)\s+be\s+(?:defined|provided|specified)',
        r'(?:eligible\s+entit(?:y|ies)|ordering\s+activit(?:y|ies))\s+(?:will|may)\s+(?:define|specify|provide)',
        r'to\s+be\s+determined\s+(?:at\s+)?(?:the\s+)?(?:task|order|rfq)\s+level',
        r'(?:tbd|to\s+be\s+determined)',
    ]


class KeyPersonnelPatterns:
    """Patterns for detecting key personnel requirements."""
    
    EXPERIENCE_PATTERNS = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|working)',
        r'(?:minimum|at\s+least)\s+(\d+)\s*years?',
        r'experience\s+(?:of\s+)?(?:at\s+least\s+)?(\d+)\s*years?',
    ]
    
    ROLE_PATTERNS = [
        r'(project\s+manager|program\s+manager|technical\s+lead|subject\s+matter\s+expert|architect|developer|analyst|engineer)',
        r'(pm|pi|co-?pi|key\s+personnel)',
    ]
    
    CERTIFICATION_PATTERNS = [
        r'(pmp|cissp|cism|itil|aws|azure|gcp)\s+(?:certification|certified)',
        r'(?:must\s+(?:have|possess|hold)|required)\s+.*(certification|license)',
    ]
    
    CLEARANCE_PATTERNS = [
        r'(secret|top\s+secret|ts/sci|public\s+trust)\s+(?:clearance|cleared)',
        r'(?:security\s+)?clearance[:\s]*(secret|top\s+secret|ts|sci)',
    ]


class ConstraintPatterns:
    """Patterns for detecting compliance constraints."""
    
    PATTERNS = [
        r'(?:shall|must)\s+not\s+(?:be\s+)?(?:used?\s+)?(?:to|for)\s+(.*?)(?:\.|$)',
        r'cannot\s+(?:be\s+)?(?:used?\s+)?(?:to|for)?\s*(.*?)(?:\.|$)',
        r'(?:prohibited|forbidden|restricted)\s+from\s+(.*?)(?:\.|$)',
        r'(?:data|information)\s+(?:shall|must|cannot)\s+(?:not\s+)?(?:remain|stay|be\s+kept|be\s+exported)\s*(.*?)(?:\.|$)',
        r'cannot\s+(?:train|share|transfer|export)\s+(.*?)(?:\.|$)',
    ]


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class EnhancedCTMExtractor:
    """
    Extracts enhanced compliance metadata from RFP text.
    
    This class provides methods to analyze requirement text and populate
    all the v3.0 CTM fields.
    """
    
    def __init__(self, rfp_format: str = "STANDARD_UCF"):
        """
        Initialize the extractor.
        
        Args:
            rfp_format: The detected RFP format (NIH_FACTOR, GSA_BPA, etc.)
        """
        self.rfp_format = rfp_format
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        # Scoring patterns
        self._pass_fail_re = [re.compile(p, re.IGNORECASE) for p in ScoringPatterns.PASS_FAIL_PATTERNS]
        self._weighted_re = [re.compile(p, re.IGNORECASE) for p in ScoringPatterns.WEIGHTED_PATTERNS]
        self._qualitative_re = [re.compile(p, re.IGNORECASE) for p in ScoringPatterns.QUALITATIVE_PATTERNS]
        
        # Response format patterns
        self._checkbox_re = [re.compile(p, re.IGNORECASE) for p in ResponseFormatPatterns.CHECKBOX_PATTERNS]
        self._checkbox_evidence_re = [re.compile(p, re.IGNORECASE) for p in ResponseFormatPatterns.CHECKBOX_WITH_EVIDENCE_PATTERNS]
        self._table_re = [re.compile(p, re.IGNORECASE) for p in ResponseFormatPatterns.TABLE_PATTERNS]
        self._appendix_re = [re.compile(p, re.IGNORECASE) for p in ResponseFormatPatterns.APPENDIX_PATTERNS]
        self._resume_re = [re.compile(p, re.IGNORECASE) for p in ResponseFormatPatterns.RESUME_PATTERNS]
        
        # Page limit patterns
        self._page_limit_re = [re.compile(p, re.IGNORECASE) for p in PageLimitPatterns.PATTERNS]
        self._double_sided_re = [re.compile(p, re.IGNORECASE) for p in PageLimitPatterns.DOUBLE_SIDED_PATTERNS]
        
        # Formatting patterns
        self._font_re = [re.compile(p, re.IGNORECASE) for p in FormattingPatterns.FONT_PATTERNS]
        self._margin_re = [re.compile(p, re.IGNORECASE) for p in FormattingPatterns.MARGIN_PATTERNS]
        self._spacing_re = [re.compile(p, re.IGNORECASE) for p in FormattingPatterns.SPACING_PATTERNS]
        
        # Future diligence patterns
        self._future_diligence_re = [re.compile(p, re.IGNORECASE) for p in FutureDiligencePatterns.PATTERNS]
        
        # Key personnel patterns
        self._experience_re = [re.compile(p, re.IGNORECASE) for p in KeyPersonnelPatterns.EXPERIENCE_PATTERNS]
        self._role_re = [re.compile(p, re.IGNORECASE) for p in KeyPersonnelPatterns.ROLE_PATTERNS]
        self._certification_re = [re.compile(p, re.IGNORECASE) for p in KeyPersonnelPatterns.CERTIFICATION_PATTERNS]
        self._clearance_re = [re.compile(p, re.IGNORECASE) for p in KeyPersonnelPatterns.CLEARANCE_PATTERNS]
        
        # Constraint patterns
        self._constraint_re = [re.compile(p, re.IGNORECASE) for p in ConstraintPatterns.PATTERNS]
    
    # =========================================================================
    # MAIN EXTRACTION METHODS
    # =========================================================================
    
    def extract_scoring_type(self, text: str, context: str = "") -> Tuple[ScoringType, Optional[int], float]:
        """
        Detect the scoring type and max points from requirement text.
        
        Args:
            text: The requirement text
            context: Additional context (e.g., section header, nearby text)
        
        Returns:
            Tuple of (ScoringType, max_points or None, confidence)
        """
        combined_text = f"{text} {context}".lower()
        
        # Check for weighted/points indicators FIRST (highest priority)
        # This prevents "shall" in "Maximum Points: 40" from triggering pass/fail
        for pattern in self._weighted_re:
            match = pattern.search(combined_text)
            if match:
                try:
                    points = int(match.group(1))
                    return ScoringType.WEIGHTED, points, 0.95
                except (ValueError, IndexError):
                    pass
        
        # Check for qualitative indicators (before pass/fail)
        for pattern in self._qualitative_re:
            if pattern.search(combined_text):
                return ScoringType.QUALITATIVE, None, 0.8
        
        # Check for pass/fail indicators
        for pattern in self._pass_fail_re:
            if pattern.search(combined_text):
                return ScoringType.PASS_FAIL, None, 0.9
        
        # Default based on mandatory language (lower confidence)
        if "mandatory" in combined_text:
            return ScoringType.PASS_FAIL, None, 0.5
        
        return ScoringType.UNKNOWN, None, 0.0
    
    def extract_response_format(self, text: str, context: str = "") -> Tuple[ResponseFormat, float]:
        """
        Detect the required response format.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            Tuple of (ResponseFormat, confidence)
        """
        combined_text = f"{text} {context}".lower()
        
        # Check for checkbox with evidence (most specific first)
        for pattern in self._checkbox_evidence_re:
            if pattern.search(combined_text):
                return ResponseFormat.CHECKBOX_WITH_EVIDENCE, 0.9
        
        # Check for simple checkbox
        for pattern in self._checkbox_re:
            if pattern.search(combined_text):
                return ResponseFormat.CHECKBOX_ONLY, 0.85
        
        # Check for table format
        for pattern in self._table_re:
            if pattern.search(combined_text):
                return ResponseFormat.TABLE, 0.85
        
        # Check for resume/CV
        for pattern in self._resume_re:
            if pattern.search(combined_text):
                return ResponseFormat.RESUME, 0.9
        
        # Check for appendix
        for pattern in self._appendix_re:
            if pattern.search(combined_text):
                return ResponseFormat.APPENDIX, 0.8
        
        # Default to narrative
        return ResponseFormat.NARRATIVE, 0.5
    
    def extract_page_limit(self, text: str, context: str = "") -> Optional[PageLimit]:
        """
        Extract page limit information.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            PageLimit object or None
        """
        combined_text = f"{text} {context}"
        
        # Check for page limit
        for pattern in self._page_limit_re:
            match = pattern.search(combined_text)
            if match:
                try:
                    limit_value = int(match.group(1))
                    
                    # Check if double-sided
                    limit_type = "single_sided"
                    for ds_pattern in self._double_sided_re:
                        ds_match = ds_pattern.search(combined_text)
                        if ds_match:
                            limit_type = "double_sided"
                            break
                    
                    # Extract exclusions
                    excludes = []
                    for excl_pattern in [re.compile(p, re.IGNORECASE) for p in PageLimitPatterns.EXCLUSION_PATTERNS]:
                        excl_match = excl_pattern.search(combined_text)
                        if excl_match:
                            excludes.append(excl_match.group(1).strip())
                    
                    return PageLimit(
                        limit_value=limit_value,
                        limit_type=limit_type,
                        excludes=excludes
                    )
                except (ValueError, IndexError):
                    pass
        
        return None
    
    def extract_formatting(self, text: str, context: str = "") -> Optional[FormattingRequirement]:
        """
        Extract formatting requirements.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            FormattingRequirement object or None
        """
        combined_text = f"{text} {context}"
        formatting = FormattingRequirement()
        found_any = False
        
        # Extract font size
        for pattern in self._font_re:
            match = pattern.search(combined_text)
            if match:
                try:
                    # Check if it's a font name
                    if match.group(1).lower() in ['times new roman', 'arial', 'calibri', 'courier']:
                        formatting.font_name = match.group(1).title()
                    else:
                        formatting.font_size_min = int(match.group(1))
                    found_any = True
                except (ValueError, IndexError):
                    pass
        
        # Extract margin
        for pattern in self._margin_re:
            match = pattern.search(combined_text)
            if match:
                try:
                    formatting.margin_inches = float(match.group(1))
                    found_any = True
                except (ValueError, IndexError):
                    pass
        
        # Extract spacing
        for pattern in self._spacing_re:
            match = pattern.search(combined_text)
            if match:
                spacing_text = match.group(1).lower()
                if spacing_text in ['single', 'double', '1.5']:
                    formatting.line_spacing = spacing_text
                    found_any = True
        
        return formatting if found_any else None
    
    def extract_future_diligence(self, text: str, context: str = "") -> Tuple[bool, Optional[str]]:
        """
        Detect if requirement is deferred to future RFQ.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            Tuple of (is_deferred, note)
        """
        combined_text = f"{text} {context}".lower()
        
        for pattern in self._future_diligence_re:
            match = pattern.search(combined_text)
            if match:
                return True, "Details to be defined in subsequent RFQ"
        
        return False, None
    
    def extract_key_personnel(self, text: str, context: str = "") -> Optional[KeyPersonnelRequirement]:
        """
        Extract key personnel requirements.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            KeyPersonnelRequirement object or None
        """
        combined_text = f"{text} {context}"
        
        # Check if this is a key personnel requirement
        is_personnel = False
        role = None
        
        for pattern in self._role_re:
            match = pattern.search(combined_text)
            if match:
                is_personnel = True
                role = match.group(1).title()
                break
        
        if not is_personnel:
            return None
        
        kp_req = KeyPersonnelRequirement(role=role or "Key Personnel")
        
        # Extract years of experience
        for pattern in self._experience_re:
            match = pattern.search(combined_text)
            if match:
                try:
                    kp_req.min_years_experience = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    pass
        
        # Extract certifications
        for pattern in self._certification_re:
            match = pattern.search(combined_text)
            if match:
                cert = match.group(1).upper()
                if cert not in kp_req.required_certifications:
                    kp_req.required_certifications.append(cert)
        
        # Extract clearances
        for pattern in self._clearance_re:
            match = pattern.search(combined_text)
            if match:
                clearance = match.group(1).title()
                if clearance not in kp_req.required_clearances:
                    kp_req.required_clearances.append(clearance)
        
        return kp_req
    
    def extract_constraint(self, text: str, context: str = "") -> Optional[str]:
        """
        Extract compliance constraints.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            Constraint detail string or None
        """
        combined_text = f"{text} {context}"
        
        for pattern in self._constraint_re:
            match = pattern.search(combined_text)
            if match:
                constraint = match.group(1).strip()
                if constraint:
                    return constraint
        
        return None
    
    def extract_evidence_requirement(self, text: str, context: str = "") -> Optional[EvidenceRequirement]:
        """
        Detect if evidence location citation is required.
        
        Args:
            text: The requirement text
            context: Additional context
        
        Returns:
            EvidenceRequirement object or None
        """
        combined_text = f"{text} {context}".lower()
        
        # Check for evidence location requirement
        location_required = False
        for pattern in self._checkbox_evidence_re:
            if pattern.search(combined_text):
                location_required = True
                break
        
        if not location_required:
            # Check for other evidence indicators
            evidence_indicators = [
                r'provide\s+(evidence|proof|documentation)',
                r'demonstrate\s+(compliance|capability)',
                r'reference\s+(contract|project|experience)',
            ]
            for indicator in evidence_indicators:
                if re.search(indicator, combined_text, re.IGNORECASE):
                    return EvidenceRequirement(
                        evidence_type="documentation",
                        location_required=False
                    )
        
        if location_required:
            return EvidenceRequirement(
                evidence_type="reference",
                location_required=True,
                location_placeholder="Section X.X, Page XX"
            )
        
        return None
    
    # =========================================================================
    # FULL EXTRACTION METHOD
    # =========================================================================
    
    def extract_all_metadata(
        self,
        requirement_text: str,
        section_reference: str = "",
        rfp_section: RFPSection = RFPSection.OTHER,
        context: str = ""
    ) -> EnhancedRequirement:
        """
        Extract all metadata for a requirement.
        
        This is the main entry point for full requirement extraction.
        
        Args:
            requirement_text: The requirement text
            section_reference: Section reference (e.g., "L.5.2.1")
            rfp_section: The RFP section (L, M, C, etc.)
            context: Additional context (section headers, nearby text)
        
        Returns:
            EnhancedRequirement with all metadata populated
        """
        # Extract scoring type and points
        scoring_type, max_points, scoring_confidence = self.extract_scoring_type(
            requirement_text, context
        )
        
        # Extract response format
        response_format, format_confidence = self.extract_response_format(
            requirement_text, context
        )
        
        # Extract page limit
        page_limit = self.extract_page_limit(requirement_text, context)
        
        # Extract formatting
        formatting = self.extract_formatting(requirement_text, context)
        
        # Extract future diligence flag
        future_diligence, diligence_note = self.extract_future_diligence(
            requirement_text, context
        )
        
        # Extract key personnel requirements
        key_personnel = self.extract_key_personnel(requirement_text, context)
        
        # Extract constraint detail
        constraint = self.extract_constraint(requirement_text, context)
        
        # Extract evidence requirement
        evidence = self.extract_evidence_requirement(requirement_text, context)
        
        # Determine requirement type
        req_type = self._infer_requirement_type(
            requirement_text, rfp_section, key_personnel is not None
        )
        
        # Calculate priority score
        priority = self._calculate_priority(
            scoring_type, max_points, rfp_section
        )
        
        # Create the requirement
        req = EnhancedRequirement(
            requirement_text=requirement_text,
            section_reference=section_reference,
            rfp_section=rfp_section,
            requirement_type=req_type,
            scoring_type=scoring_type,
            max_points=max_points,
            response_format=response_format,
            evidence_location_required=evidence.location_required if evidence else False,
            evidence_location_placeholder=evidence.location_placeholder if evidence else None,
            future_diligence_required=future_diligence,
            future_diligence_note=diligence_note,
            constraint_detail=constraint,
            priority_score=priority,
            page_limit=page_limit,
            formatting=formatting,
            evidence=evidence,
            key_personnel=key_personnel,
            extraction_confidence=max(scoring_confidence, format_confidence)
        )
        
        # Flag items needing review
        if req.extraction_confidence < 0.7:
            req.needs_review = True
        
        # Flag critical items
        if scoring_type == ScoringType.PASS_FAIL:
            req.is_critical = True
        
        return req
    
    def _infer_requirement_type(
        self,
        text: str,
        rfp_section: RFPSection,
        has_key_personnel: bool
    ) -> RequirementType:
        """Infer the requirement type from text and context."""
        text_lower = text.lower()
        
        if has_key_personnel:
            return RequirementType.KEY_PERSONNEL
        
        if any(w in text_lower for w in ['page', 'font', 'margin', 'format', 'spacing']):
            return RequirementType.FORMATTING
        
        if any(w in text_lower for w in ['security', 'clearance', 'fisma', 'fedramp']):
            return RequirementType.SECURITY
        
        if any(w in text_lower for w in ['transition', 'phase-in', 'phase-out', 'knowledge transfer']):
            return RequirementType.TRANSITION
        
        if any(w in text_lower for w in ['cost', 'price', 'budget', 'pricing', 'rate']):
            return RequirementType.COST_PRICE
        
        if any(w in text_lower for w in ['past performance', 'experience', 'reference', 'contract history']):
            return RequirementType.PAST_PERFORMANCE
        
        if any(w in text_lower for w in ['manage', 'staff', 'team', 'organization', 'approach']):
            return RequirementType.MANAGEMENT
        
        if rfp_section == RFPSection.SECTION_C:
            return RequirementType.TECHNICAL
        
        if rfp_section == RFPSection.SECTION_K:
            return RequirementType.ADMINISTRATIVE
        
        return RequirementType.TECHNICAL
    
    def _calculate_priority(
        self,
        scoring_type: ScoringType,
        max_points: Optional[int],
        rfp_section: RFPSection
    ) -> int:
        """Calculate priority score (1-5, 5 being highest)."""
        # Pass/fail = highest priority
        if scoring_type == ScoringType.PASS_FAIL:
            return 5
        
        # High points = high priority
        if max_points is not None:
            if max_points >= 100:
                return 5
            elif max_points >= 50:
                return 4
            elif max_points >= 25:
                return 3
            else:
                return 2
        
        # Section M = evaluation criteria = high priority
        if rfp_section == RFPSection.SECTION_M:
            return 4
        
        # Section L instructions = medium priority
        if rfp_section == RFPSection.SECTION_L:
            return 3
        
        # Section C technical = medium priority
        if rfp_section == RFPSection.SECTION_C:
            return 3
        
        return 2


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_requirements_batch(
    requirements: List[Dict[str, Any]],
    rfp_format: str = "STANDARD_UCF"
) -> List[EnhancedRequirement]:
    """
    Process a batch of requirements and extract enhanced metadata.
    
    Args:
        requirements: List of requirement dicts with 'text', 'section_ref', 
                     'rfp_section', 'context' keys
        rfp_format: The RFP format
    
    Returns:
        List of EnhancedRequirement objects
    """
    extractor = EnhancedCTMExtractor(rfp_format)
    results = []
    
    for req in requirements:
        enhanced = extractor.extract_all_metadata(
            requirement_text=req.get('text', ''),
            section_reference=req.get('section_ref', ''),
            rfp_section=RFPSection(req.get('rfp_section', 'OTHER')),
            context=req.get('context', '')
        )
        results.append(enhanced)
    
    return results
