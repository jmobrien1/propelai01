"""
OASIS+ Data Models
==================

Database schema models for GSA OASIS+ self-scoring proposals.
Designed to mirror the J.P-1 Qualifications Matrix hierarchy.

Schema Overview:
    Solicitation → Domain → ScoringCriteria
    Project → ProjectClaim → ScoringCriteria
    Document → DocumentChunk (with embeddings)
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal


class DomainType(Enum):
    """OASIS+ Domain Types per Section C.2"""
    TECHNICAL_ENGINEERING = "technical_engineering"
    MANAGEMENT_ADVISORY = "management_advisory"
    ENTERPRISE_SOLUTIONS = "enterprise_solutions"
    INTELLIGENCE_SERVICES = "intelligence_services"
    LOGISTICS = "logistics"
    ENVIRONMENTAL = "environmental"
    FACILITIES = "facilities"
    RESEARCH_DEVELOPMENT = "research_development"


class BusinessSize(Enum):
    """Business size classification for threshold determination"""
    UNRESTRICTED = "unrestricted"  # 42 point threshold
    SMALL_BUSINESS = "small_business"  # 36 point threshold
    WOSB = "wosb"  # Women-Owned Small Business
    SDVOSB = "sdvosb"  # Service-Disabled Veteran-Owned
    HUBZONE = "hubzone"
    EIGHT_A = "8a"  # 8(a) Program


class VerificationStatus(Enum):
    """Status of evidence verification for a claim"""
    UNVERIFIED = "unverified"  # Not yet checked
    PENDING = "pending"  # Evidence found, awaiting human review
    VERIFIED = "verified"  # Human confirmed evidence
    REJECTED = "rejected"  # Evidence insufficient
    FPDS_VERIFIED = "fpds_verified"  # Auto-verified via FPDS data
    JP3_REQUIRED = "jp3_required"  # Needs J.P-3 form from CO


class ContractType(Enum):
    """Contract types for scoring credits"""
    FIRM_FIXED_PRICE = "ffp"
    COST_REIMBURSEMENT = "cost_plus"
    TIME_AND_MATERIALS = "t_and_m"
    LABOR_HOUR = "labor_hour"
    HYBRID = "hybrid"
    IDIQ = "idiq"
    BPA = "bpa"


class CriteriaType(Enum):
    """Types of scoring criteria in J.P-1"""
    MANDATORY = "mandatory"  # Must meet to qualify
    QUALIFYING_PROJECT = "qualifying_project"  # QP requirements
    FEDERAL_EXPERIENCE = "federal_experience"  # FEP requirements
    OPTIONAL_CREDIT = "optional_credit"  # Additional points
    THRESHOLD = "threshold"  # Minimum score requirement


@dataclass
class ScoringCriteria:
    """
    A single scoring criterion from the J.P-1 Matrix.

    Represents one row in the qualifications matrix that can
    earn points for a proposal.
    """
    criteria_id: str  # e.g., "L.5.2.1", "L.5.2.3.4"
    domain: DomainType
    description: str
    max_points: int
    criteria_type: CriteriaType

    # Validation rules
    validation_rule: Optional[str] = None  # e.g., "Must be in FPDS"
    threshold_value: Optional[Decimal] = None  # e.g., $500,000 AAV
    requires_jp3: bool = False

    # Evidence keywords for semantic search
    evidence_keywords: List[str] = field(default_factory=list)

    # Parent criteria for hierarchical scoring
    parent_criteria_id: Optional[str] = None

    def __post_init__(self):
        """Initialize evidence keywords if not provided"""
        if not self.evidence_keywords:
            self.evidence_keywords = self._derive_keywords()

    def _derive_keywords(self) -> List[str]:
        """Derive search keywords from description"""
        # Common OASIS+ terms to search for
        keyword_map = {
            "surge": ["surge", "rapid expansion", "ramp up", "urgent requirement"],
            "oconus": ["oconus", "overseas", "international", "foreign"],
            "clearance": ["clearance", "secret", "top secret", "ts/sci", "classified"],
            "cmmi": ["cmmi", "capability maturity", "level 3", "level 5"],
            "iso": ["iso 9001", "iso 27001", "quality management"],
            "agile": ["agile", "scrum", "devops", "continuous integration"],
        }

        desc_lower = self.description.lower()
        keywords = []
        for key, terms in keyword_map.items():
            if key in desc_lower:
                keywords.extend(terms)

        return keywords if keywords else [self.description.lower()]


@dataclass
class OASISDomain:
    """
    An OASIS+ functional domain (e.g., Technical & Engineering).

    Contains all scoring criteria for that domain and threshold requirements.
    """
    domain_type: DomainType
    name: str
    description: str

    # Scoring thresholds
    unrestricted_threshold: int = 42
    small_business_threshold: int = 36

    # Project limits
    max_qualifying_projects: int = 5
    max_federal_experience_projects: int = 5

    # Minimum AAV for qualifying projects
    min_aav_unrestricted: Decimal = Decimal("500000")
    min_aav_small_business: Decimal = Decimal("250000")

    # Scoring criteria for this domain
    criteria: List[ScoringCriteria] = field(default_factory=list)

    # NAICS codes that auto-qualify for this domain (from J.P-4)
    auto_relevant_naics: List[str] = field(default_factory=list)

    # PSC codes that auto-qualify
    auto_relevant_psc: List[str] = field(default_factory=list)

    def get_threshold(self, business_size: BusinessSize) -> int:
        """Get the point threshold based on business size"""
        if business_size == BusinessSize.UNRESTRICTED:
            return self.unrestricted_threshold
        return self.small_business_threshold

    def get_min_aav(self, business_size: BusinessSize) -> Decimal:
        """Get minimum AAV based on business size"""
        if business_size == BusinessSize.UNRESTRICTED:
            return self.min_aav_unrestricted
        return self.min_aav_small_business


@dataclass
class Project:
    """
    A past performance project in the contractor's library.

    Represents a contract that can be used as a Qualifying Project (QP)
    or Federal Experience Project (FEP) in the OASIS+ proposal.
    """
    project_id: str
    title: str
    client_agency: str

    # Contract identification
    contract_number: str
    task_order_number: Optional[str] = None

    # Classification codes
    naics_code: str = ""
    psc_code: str = ""

    # Period of performance
    start_date: date = None
    end_date: Optional[date] = None  # None = ongoing

    # Financial data
    total_obligated_amount: Decimal = Decimal("0")
    total_ceiling_amount: Optional[Decimal] = None

    # Contract characteristics
    contract_type: ContractType = ContractType.FIRM_FIXED_PRICE
    is_prime: bool = True  # vs subcontract
    prime_contractor: Optional[str] = None  # If subcontract

    # Location
    performance_location: str = ""
    is_oconus: bool = False

    # Security
    clearance_level: Optional[str] = None  # e.g., "SECRET", "TS/SCI"

    # Scope description
    scope_description: str = ""

    # Source documents
    document_ids: List[str] = field(default_factory=list)

    # Computed fields (populated during analysis)
    average_annual_value: Optional[Decimal] = None
    relevance_scores: Dict[DomainType, float] = field(default_factory=dict)

    def calculate_aav(self) -> Decimal:
        """
        Calculate Average Annual Value per OASIS+ formula.
        AAV = (Total Obligated Value / Days of Performance) × 366
        """
        if not self.start_date:
            return Decimal("0")

        end = self.end_date or date.today()
        days = (end - self.start_date).days

        if days <= 0:
            return Decimal("0")

        self.average_annual_value = (self.total_obligated_amount / days) * 366
        return self.average_annual_value

    def is_recent(self, cutoff_years: int = 5) -> bool:
        """Check if project meets recency requirement"""
        if not self.end_date:
            return True  # Ongoing projects are recent

        cutoff_date = date.today().replace(year=date.today().year - cutoff_years)
        return self.end_date >= cutoff_date

    def qualifies_for_domain(self, domain: OASISDomain, business_size: BusinessSize) -> bool:
        """Check if project meets minimum requirements for a domain"""
        aav = self.calculate_aav()
        min_aav = domain.get_min_aav(business_size)

        if aav < min_aav:
            return False

        if not self.is_recent():
            return False

        return True


@dataclass
class ProjectClaim:
    """
    A claim linking a Project to a ScoringCriteria.

    Represents an assertion that a specific project demonstrates
    a particular capability for points in the J.P-1 matrix.
    """
    claim_id: str
    project_id: str
    criteria_id: str

    # Points
    claimed_points: int
    verified_points: int = 0

    # Evidence
    evidence_snippet: str = ""
    evidence_page_number: Optional[int] = None
    evidence_document_id: Optional[str] = None
    evidence_bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h

    # Verification
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_notes: str = ""
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None

    # AI confidence
    ai_confidence_score: float = 0.0

    # J.P-3 form reference if needed
    jp3_form_id: Optional[str] = None


@dataclass
class DocumentChunk:
    """
    A chunk of text from a project document with embedding.

    Stores text with its vector embedding and location information
    for semantic search and PDF annotation.
    """
    chunk_id: str
    document_id: str
    project_id: str

    # Content
    content: str

    # Vector embedding (stored as list, converted to numpy for search)
    embedding: Optional[List[float]] = None
    embedding_model: str = "text-embedding-3-small"

    # Location in document
    page_number: int = 1
    bbox: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h

    # Metadata
    chunk_index: int = 0  # Order within document
    char_start: int = 0
    char_end: int = 0

    # OCR metadata
    ocr_confidence: Optional[float] = None
    was_ocr: bool = False


@dataclass
class ScorecardResult:
    """
    Complete scorecard for a domain proposal.

    Aggregates all projects and claims to show total score
    and threshold comparison.
    """
    domain: DomainType
    business_size: BusinessSize

    # Selected projects
    qualifying_projects: List[Project] = field(default_factory=list)
    federal_experience_projects: List[Project] = field(default_factory=list)

    # All claims
    claims: List[ProjectClaim] = field(default_factory=list)

    # Scoring summary
    total_score: int = 0
    threshold: int = 42
    margin: int = 0  # Points above/below threshold

    # Breakdown by criteria type
    mandatory_points: int = 0
    qp_points: int = 0
    fep_points: int = 0
    optional_credit_points: int = 0

    # Verification status
    verified_points: int = 0
    pending_points: int = 0
    unverified_points: int = 0

    # Risk assessment
    at_risk_claims: List[ProjectClaim] = field(default_factory=list)

    def calculate_totals(self):
        """Calculate all score totals from claims"""
        self.total_score = sum(c.claimed_points for c in self.claims)
        self.verified_points = sum(c.verified_points for c in self.claims)
        self.pending_points = sum(
            c.claimed_points for c in self.claims
            if c.status == VerificationStatus.PENDING
        )
        self.unverified_points = sum(
            c.claimed_points for c in self.claims
            if c.status == VerificationStatus.UNVERIFIED
        )
        self.margin = self.total_score - self.threshold

        # Identify at-risk claims (low confidence or unverified)
        self.at_risk_claims = [
            c for c in self.claims
            if c.ai_confidence_score < 0.7 or c.status == VerificationStatus.UNVERIFIED
        ]

    @property
    def meets_threshold(self) -> bool:
        """Check if score meets qualification threshold"""
        return self.total_score >= self.threshold

    @property
    def has_safe_margin(self) -> bool:
        """Check if score has recommended 3-5 point cushion"""
        return self.margin >= 3


@dataclass
class OptimizationConstraints:
    """Constraints for the project selection optimizer"""
    max_qualifying_projects: int = 5
    max_federal_experience_projects: int = 5
    min_threshold: int = 42
    target_margin: int = 5  # Points above threshold to target

    # Diversity constraints
    require_agency_diversity: bool = False
    min_unique_agencies: int = 2

    # Recency constraints
    recency_cutoff_years: int = 5

    # Risk tolerance
    max_unverified_points: int = 10
    min_confidence_score: float = 0.6


@dataclass
class JP3FormData:
    """Data structure for J.P-3 Project Verification Form"""
    # Project identification
    project_title: str
    contract_number: str
    task_order_number: Optional[str] = None

    # Contractor info
    contractor_name: str = ""
    contractor_cage_code: str = ""
    contractor_duns: str = ""

    # Government contact
    contracting_officer_name: str = ""
    contracting_officer_email: str = ""
    contracting_officer_phone: str = ""

    # Period of performance
    start_date: date = None
    end_date: Optional[date] = None

    # Financial
    total_obligated_value: Decimal = Decimal("0")
    average_annual_value: Decimal = Decimal("0")

    # Classification
    naics_code: str = ""
    psc_code: str = ""

    # Relevance narrative
    relevance_statement: str = ""

    # Claims being verified
    claims_description: str = ""

    # Signature fields (left blank for CO)
    signature_date: Optional[date] = None
    co_signature: str = ""  # Left blank
