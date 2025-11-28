"""
PropelAI CTM Data Models v3.0
Enhanced Compliance Traceability Matrix data structures

This module defines the core data models for the CTM, incorporating
the new fields identified in the NLM analysis:
- Scoring_Type: Pass/Fail, N/A, Qualitative, Weighted
- Mandatory_Response_Format: Checkbox, Checkbox+Evidence, Narrative, Table
- Max_Points: Numeric scoring weight
- Future_Diligence_Flag: For EA/RFR deferred requirements
- Compliance_Constraint_Detail: Specific prohibitions/restrictions

Author: PropelAI Team
Version: 3.0.0
Date: November 28, 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import re


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ScoringType(Enum):
    """
    Categorizes how a requirement affects proposal evaluation.
    
    Critical distinction: Pass/Fail requirements can disqualify even with 
    perfect scores elsewhere, while weighted requirements contribute to 
    competitive scoring.
    """
    PASS_FAIL = "pass_fail"           # Must meet or proposal is non-responsive
    NOT_APPLICABLE = "n_a"            # Tracked but doesn't affect scoring
    QUALITATIVE = "qualitative"       # Subjective evaluation (Outstanding/Good/Acceptable)
    WEIGHTED = "weighted"             # Has specific point value (Max_Points field)
    UNKNOWN = "unknown"               # Could not determine scoring type


class ResponseFormat(Enum):
    """
    Specifies the required format for responding to a requirement.
    
    Based on IDES RFP analysis:
    - F.1 requires simple checkbox (Agree)
    - F.2 requires checkbox + evidence reference (page/section citation)
    - F.3 requires full narrative with supporting detail
    """
    CHECKBOX_ONLY = "checkbox_only"              # Simple Agree/Confirm
    CHECKBOX_WITH_EVIDENCE = "checkbox_evidence" # Checkbox + page/section reference
    NARRATIVE = "narrative"                      # Full written response
    TABLE = "table"                              # Structured table format required
    FORM = "form"                                # Fill-in-the-blank form
    APPENDIX = "appendix"                        # Separate appendix document
    RESUME = "resume"                            # Key personnel resume format
    MIXED = "mixed"                              # Multiple formats required
    UNKNOWN = "unknown"


class RequirementType(Enum):
    """
    Categorizes the nature of the requirement.
    """
    TECHNICAL = "technical"           # Technical approach, methodology
    MANAGEMENT = "management"         # Management approach, staffing
    PAST_PERFORMANCE = "past_performance"  # Experience, references
    KEY_PERSONNEL = "key_personnel"   # Staff qualifications
    COST_PRICE = "cost_price"         # Pricing, cost breakdown
    FORMATTING = "formatting"         # Page limits, fonts, margins
    ADMINISTRATIVE = "administrative" # Certifications, forms
    TRANSITION = "transition"         # Transition-in/out plans
    SECURITY = "security"             # IT security, clearances
    COMPLIANCE = "compliance"         # Regulatory, contractual terms
    OTHER = "other"


class RFPSection(Enum):
    """
    Standard RFP sections per Uniform Contract Format (UCF).
    """
    SECTION_A = "A"   # Solicitation/Contract Form
    SECTION_B = "B"   # Supplies or Services and Prices/Costs
    SECTION_C = "C"   # Description/Specs/Statement of Work
    SECTION_D = "D"   # Packaging and Marking
    SECTION_E = "E"   # Inspection and Acceptance
    SECTION_F = "F"   # Deliveries or Performance
    SECTION_G = "G"   # Contract Administration Data
    SECTION_H = "H"   # Special Contract Requirements
    SECTION_I = "I"   # Contract Clauses
    SECTION_J = "J"   # List of Attachments
    SECTION_K = "K"   # Representations and Certifications
    SECTION_L = "L"   # Instructions to Offerors
    SECTION_M = "M"   # Evaluation Factors for Award
    ATTACHMENT = "ATT"  # Attachments/Exhibits
    AMENDMENT = "AMD"   # Amendments
    OTHER = "OTHER"


class ComplianceStatus(Enum):
    """
    Tracks the proposal's compliance status for each requirement.
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    DRAFT_COMPLETE = "draft_complete"
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WAIVED = "waived"
    NOT_APPLICABLE = "n_a"


# =============================================================================
# CORE DATA CLASSES
# =============================================================================

@dataclass
class PageLimit:
    """
    Captures page limit constraints for a section or factor.
    
    Handles complex limits like ITS88: "25 double-sided or 50 single-sided"
    """
    limit_value: int                          # Primary page count
    limit_type: str = "single_sided"          # single_sided, double_sided, words
    excludes: List[str] = field(default_factory=list)  # What doesn't count (appendices, etc.)
    includes: List[str] = field(default_factory=list)  # What specifically counts
    notes: Optional[str] = None               # Additional context
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit_value": self.limit_value,
            "limit_type": self.limit_type,
            "excludes": self.excludes,
            "includes": self.includes,
            "notes": self.notes
        }


@dataclass
class FormattingRequirement:
    """
    Captures document formatting requirements.
    """
    font_name: Optional[str] = None           # "Times New Roman", "Arial"
    font_size_min: Optional[int] = None       # Minimum font size (points)
    margin_inches: Optional[float] = None     # Margin size (all sides or specify)
    margin_top: Optional[float] = None
    margin_bottom: Optional[float] = None
    margin_left: Optional[float] = None
    margin_right: Optional[float] = None
    line_spacing: Optional[str] = None        # "single", "double", "1.5"
    paper_size: str = "letter"                # letter, A4
    header_footer_allowed: bool = True
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "font_name": self.font_name,
            "font_size_min": self.font_size_min,
            "margin_inches": self.margin_inches,
            "margin_top": self.margin_top,
            "margin_bottom": self.margin_bottom,
            "margin_left": self.margin_left,
            "margin_right": self.margin_right,
            "line_spacing": self.line_spacing,
            "paper_size": self.paper_size,
            "header_footer_allowed": self.header_footer_allowed,
            "notes": self.notes
        }


@dataclass
class EvidenceRequirement:
    """
    Captures what evidence/proof is required for a requirement.
    """
    evidence_type: str                        # "reference", "documentation", "certification"
    location_required: bool = False           # Must cite page/section in proposal
    location_placeholder: Optional[str] = None  # "Section X.X, Page XX"
    min_examples: int = 0                     # Minimum number of examples required
    max_examples: Optional[int] = None        # Maximum allowed
    recency_years: Optional[int] = None       # Must be within X years
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "location_required": self.location_required,
            "location_placeholder": self.location_placeholder,
            "min_examples": self.min_examples,
            "max_examples": self.max_examples,
            "recency_years": self.recency_years,
            "notes": self.notes
        }


@dataclass 
class KeyPersonnelRequirement:
    """
    Captures mandatory personnel qualification requirements.
    
    Based on IDES analysis: PM must have 2+ years implementing similar solution,
    Technical/Functional Managers need 5+ years experience.
    """
    role: str                                 # "Project Manager", "Technical Lead"
    min_years_experience: Optional[int] = None
    required_certifications: List[str] = field(default_factory=list)
    required_clearances: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    education_minimum: Optional[str] = None   # "Bachelor's", "Master's"
    similar_experience_required: bool = False # Must have done similar work
    resume_required: bool = True
    resume_page_limit: Optional[int] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "min_years_experience": self.min_years_experience,
            "required_certifications": self.required_certifications,
            "required_clearances": self.required_clearances,
            "required_skills": self.required_skills,
            "education_minimum": self.education_minimum,
            "similar_experience_required": self.similar_experience_required,
            "resume_required": self.resume_required,
            "resume_page_limit": self.resume_page_limit,
            "notes": self.notes
        }


# =============================================================================
# ENHANCED REQUIREMENT CLASS
# =============================================================================

@dataclass
class EnhancedRequirement:
    """
    Enhanced CTM Requirement with v3.0 fields.
    
    This is the core data structure for each requirement in the compliance matrix.
    It incorporates all the new fields identified in the NLM analysis.
    """
    # Core identification
    id: str = field(default_factory=lambda: f"REQ-{uuid.uuid4().hex[:8].upper()}")
    requirement_text: str = ""
    rfp_section: RFPSection = RFPSection.OTHER
    section_reference: str = ""               # "L.5.2.1", "M.3.a.ii"
    
    # Requirement classification
    requirement_type: RequirementType = RequirementType.OTHER
    
    # ==========================================================================
    # NEW v3.0 FIELDS - From NLM Analysis
    # ==========================================================================
    
    # 1. Scoring Type - Critical for compliance strategy
    scoring_type: ScoringType = ScoringType.UNKNOWN
    
    # 2. Max Points - For weighted scoring sections
    max_points: Optional[int] = None          # None for pass/fail, N/A
    
    # 3. Response Format - How to respond
    response_format: ResponseFormat = ResponseFormat.UNKNOWN
    
    # 4. Evidence Location Required - Must cite page/section
    evidence_location_required: bool = False
    evidence_location_placeholder: Optional[str] = None
    
    # 5. Future Diligence Flag - For EA/RFR deferred requirements
    future_diligence_required: bool = False
    future_diligence_note: Optional[str] = None  # "Defined in subsequent RFQ"
    
    # 6. Compliance Constraint Detail - Specific restrictions
    constraint_detail: Optional[str] = None   # "Cannot train public AI models"
    
    # ==========================================================================
    # Additional Metadata
    # ==========================================================================
    
    # Priority and weighting
    priority_score: int = 3                   # 1-5, derived from eval criteria proximity
    content_depth_multiplier: float = 1.0     # Based on max_points for drafting
    
    # Page/formatting constraints
    page_limit: Optional[PageLimit] = None
    formatting: Optional[FormattingRequirement] = None
    
    # Evidence requirements
    evidence: Optional[EvidenceRequirement] = None
    
    # Key personnel (if applicable)
    key_personnel: Optional[KeyPersonnelRequirement] = None
    
    # Evaluation factor linkage
    evaluation_factor_id: Optional[str] = None  # Link to Section M factor
    evaluation_factor_name: Optional[str] = None
    subfactor_id: Optional[str] = None
    
    # Compliance tracking
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_STARTED
    proposal_section: Optional[str] = None    # Where addressed in proposal
    proposal_page: Optional[int] = None       # Page number in proposal
    
    # Source tracking
    source_document: Optional[str] = None     # Original RFP filename
    source_page: Optional[int] = None         # Page in source document
    extraction_confidence: float = 0.0        # 0-1 confidence in extraction
    
    # Mandatory language indicators
    has_shall: bool = False
    has_must: bool = False
    has_required: bool = False
    has_will: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Notes and flags
    notes: str = ""
    is_critical: bool = False                 # Flagged for special attention
    needs_review: bool = False                # Flagged for human review
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self._calculate_content_depth()
        self._detect_mandatory_language()
    
    def _calculate_content_depth(self):
        """
        Calculate content depth multiplier based on max_points.
        
        Higher point values should receive proportionally more attention
        in the proposal narrative.
        """
        if self.max_points is not None and self.max_points > 0:
            # Normalize around 25 points as baseline (1.0x)
            self.content_depth_multiplier = self.max_points / 25.0
        elif self.scoring_type == ScoringType.PASS_FAIL:
            # Pass/fail items need complete coverage but not expanded narrative
            self.content_depth_multiplier = 1.0
        else:
            self.content_depth_multiplier = 1.0
    
    def _detect_mandatory_language(self):
        """Detect mandatory language indicators in requirement text."""
        text_lower = self.requirement_text.lower()
        self.has_shall = " shall " in text_lower or text_lower.startswith("shall ")
        self.has_must = " must " in text_lower or text_lower.startswith("must ")
        self.has_required = "required" in text_lower or "require" in text_lower
        self.has_will = " will " in text_lower and "government will" not in text_lower
    
    @property
    def is_mandatory(self) -> bool:
        """Check if this is a mandatory requirement."""
        return (
            self.scoring_type == ScoringType.PASS_FAIL or
            self.has_shall or 
            self.has_must or 
            self.has_required
        )
    
    @property
    def is_scored(self) -> bool:
        """Check if this requirement contributes to scoring."""
        return self.scoring_type in [ScoringType.WEIGHTED, ScoringType.QUALITATIVE]
    
    @property
    def disqualification_risk(self) -> str:
        """Assess risk of disqualification if requirement is missed."""
        if self.scoring_type == ScoringType.PASS_FAIL:
            return "HIGH"
        elif self.is_mandatory:
            return "MEDIUM"
        else:
            return "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "requirement_text": self.requirement_text,
            "rfp_section": self.rfp_section.value,
            "section_reference": self.section_reference,
            "requirement_type": self.requirement_type.value,
            
            # v3.0 fields
            "scoring_type": self.scoring_type.value,
            "max_points": self.max_points,
            "response_format": self.response_format.value,
            "evidence_location_required": self.evidence_location_required,
            "evidence_location_placeholder": self.evidence_location_placeholder,
            "future_diligence_required": self.future_diligence_required,
            "future_diligence_note": self.future_diligence_note,
            "constraint_detail": self.constraint_detail,
            
            # Priority and depth
            "priority_score": self.priority_score,
            "content_depth_multiplier": round(self.content_depth_multiplier, 2),
            
            # Nested objects
            "page_limit": self.page_limit.to_dict() if self.page_limit else None,
            "formatting": self.formatting.to_dict() if self.formatting else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "key_personnel": self.key_personnel.to_dict() if self.key_personnel else None,
            
            # Evaluation linkage
            "evaluation_factor_id": self.evaluation_factor_id,
            "evaluation_factor_name": self.evaluation_factor_name,
            "subfactor_id": self.subfactor_id,
            
            # Compliance tracking
            "compliance_status": self.compliance_status.value,
            "proposal_section": self.proposal_section,
            "proposal_page": self.proposal_page,
            
            # Source tracking
            "source_document": self.source_document,
            "source_page": self.source_page,
            "extraction_confidence": self.extraction_confidence,
            
            # Mandatory indicators
            "has_shall": self.has_shall,
            "has_must": self.has_must,
            "has_required": self.has_required,
            "has_will": self.has_will,
            "is_mandatory": self.is_mandatory,
            "is_scored": self.is_scored,
            "disqualification_risk": self.disqualification_risk,
            
            # Flags
            "is_critical": self.is_critical,
            "needs_review": self.needs_review,
            "notes": self.notes,
            
            # Timestamps
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedRequirement':
        """Create instance from dictionary."""
        # Handle enums
        if isinstance(data.get('rfp_section'), str):
            data['rfp_section'] = RFPSection(data['rfp_section'])
        if isinstance(data.get('requirement_type'), str):
            data['requirement_type'] = RequirementType(data['requirement_type'])
        if isinstance(data.get('scoring_type'), str):
            data['scoring_type'] = ScoringType(data['scoring_type'])
        if isinstance(data.get('response_format'), str):
            data['response_format'] = ResponseFormat(data['response_format'])
        if isinstance(data.get('compliance_status'), str):
            data['compliance_status'] = ComplianceStatus(data['compliance_status'])
        
        # Handle nested objects
        if data.get('page_limit') and isinstance(data['page_limit'], dict):
            data['page_limit'] = PageLimit(**data['page_limit'])
        if data.get('formatting') and isinstance(data['formatting'], dict):
            data['formatting'] = FormattingRequirement(**data['formatting'])
        if data.get('evidence') and isinstance(data['evidence'], dict):
            data['evidence'] = EvidenceRequirement(**data['evidence'])
        if data.get('key_personnel') and isinstance(data['key_personnel'], dict):
            data['key_personnel'] = KeyPersonnelRequirement(**data['key_personnel'])
        
        # Handle timestamps
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Remove computed properties before creating instance
        data.pop('is_mandatory', None)
        data.pop('is_scored', None)
        data.pop('disqualification_risk', None)
        
        return cls(**data)


# =============================================================================
# COMPLIANCE MATRIX CLASS
# =============================================================================

@dataclass
class ComplianceMatrix:
    """
    Complete Compliance Traceability Matrix for an RFP.
    
    Aggregates all requirements and provides analysis methods.
    """
    rfp_id: str
    rfp_name: str
    requirements: List[EnhancedRequirement] = field(default_factory=list)
    
    # RFP-level metadata
    rfp_format: str = "STANDARD_UCF"          # NIH_FACTOR, GSA_BPA, etc.
    total_max_points: Optional[int] = None
    proposal_due_date: Optional[datetime] = None
    
    # Global formatting requirements
    global_formatting: Optional[FormattingRequirement] = None
    global_page_limit: Optional[PageLimit] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_requirement(self, req: EnhancedRequirement):
        """Add a requirement to the matrix."""
        self.requirements.append(req)
        self.updated_at = datetime.now()
    
    def get_by_section(self, section: RFPSection) -> List[EnhancedRequirement]:
        """Get all requirements from a specific RFP section."""
        return [r for r in self.requirements if r.rfp_section == section]
    
    def get_pass_fail_requirements(self) -> List[EnhancedRequirement]:
        """Get all pass/fail (disqualifying) requirements."""
        return [r for r in self.requirements if r.scoring_type == ScoringType.PASS_FAIL]
    
    def get_scored_requirements(self) -> List[EnhancedRequirement]:
        """Get all scored requirements (weighted + qualitative)."""
        return [r for r in self.requirements if r.is_scored]
    
    def get_high_value_requirements(self, min_points: int = 25) -> List[EnhancedRequirement]:
        """Get requirements worth at least min_points."""
        return [
            r for r in self.requirements 
            if r.max_points is not None and r.max_points >= min_points
        ]
    
    def get_future_diligence_items(self) -> List[EnhancedRequirement]:
        """Get requirements deferred to future RFQ."""
        return [r for r in self.requirements if r.future_diligence_required]
    
    def get_evidence_required_items(self) -> List[EnhancedRequirement]:
        """Get requirements needing evidence location citation."""
        return [r for r in self.requirements if r.evidence_location_required]
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Calculate matrix statistics."""
        total = len(self.requirements)
        if total == 0:
            return {"total": 0}
        
        pass_fail = len(self.get_pass_fail_requirements())
        scored = len(self.get_scored_requirements())
        
        # Points analysis
        pointed_reqs = [r for r in self.requirements if r.max_points is not None]
        total_points = sum(r.max_points for r in pointed_reqs)
        
        # By section
        section_counts = {}
        for section in RFPSection:
            count = len(self.get_by_section(section))
            if count > 0:
                section_counts[section.value] = count
        
        # Compliance status
        status_counts = {}
        for status in ComplianceStatus:
            count = len([r for r in self.requirements if r.compliance_status == status])
            if count > 0:
                status_counts[status.value] = count
        
        return {
            "total": total,
            "pass_fail_count": pass_fail,
            "scored_count": scored,
            "total_max_points": total_points,
            "future_diligence_count": len(self.get_future_diligence_items()),
            "evidence_required_count": len(self.get_evidence_required_items()),
            "by_section": section_counts,
            "by_compliance_status": status_counts,
            "high_risk_count": len([r for r in self.requirements if r.disqualification_risk == "HIGH"]),
            "needs_review_count": len([r for r in self.requirements if r.needs_review])
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rfp_id": self.rfp_id,
            "rfp_name": self.rfp_name,
            "rfp_format": self.rfp_format,
            "total_max_points": self.total_max_points,
            "proposal_due_date": self.proposal_due_date.isoformat() if self.proposal_due_date else None,
            "global_formatting": self.global_formatting.to_dict() if self.global_formatting else None,
            "global_page_limit": self.global_page_limit.to_dict() if self.global_page_limit else None,
            "requirements": [r.to_dict() for r in self.requirements],
            "stats": self.stats,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_pass_fail_requirement(
    text: str,
    section_ref: str,
    rfp_section: RFPSection = RFPSection.SECTION_L,
    response_format: ResponseFormat = ResponseFormat.CHECKBOX_ONLY,
    **kwargs
) -> EnhancedRequirement:
    """Factory function for creating pass/fail requirements."""
    return EnhancedRequirement(
        requirement_text=text,
        section_reference=section_ref,
        rfp_section=rfp_section,
        scoring_type=ScoringType.PASS_FAIL,
        response_format=response_format,
        priority_score=5,  # Highest priority
        is_critical=True,
        **kwargs
    )


def create_weighted_requirement(
    text: str,
    section_ref: str,
    max_points: int,
    rfp_section: RFPSection = RFPSection.SECTION_M,
    response_format: ResponseFormat = ResponseFormat.NARRATIVE,
    **kwargs
) -> EnhancedRequirement:
    """Factory function for creating weighted/scored requirements."""
    return EnhancedRequirement(
        requirement_text=text,
        section_reference=section_ref,
        rfp_section=rfp_section,
        scoring_type=ScoringType.WEIGHTED,
        max_points=max_points,
        response_format=response_format,
        **kwargs
    )


def create_future_diligence_requirement(
    text: str,
    section_ref: str,
    diligence_note: str = "Details to be defined in subsequent RFQ",
    **kwargs
) -> EnhancedRequirement:
    """Factory function for EA/RFR requirements deferred to future RFQ."""
    return EnhancedRequirement(
        requirement_text=text,
        section_reference=section_ref,
        scoring_type=ScoringType.NOT_APPLICABLE,
        future_diligence_required=True,
        future_diligence_note=diligence_note,
        response_format=ResponseFormat.NARRATIVE,
        **kwargs
    )
