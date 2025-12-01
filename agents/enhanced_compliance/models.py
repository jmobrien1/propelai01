"""
PropelAI Cycle 5: Enhanced Compliance Agent - Data Models
Requirements Graph schema with cross-document linking

Based on NIH RFP 75N96025R00004 analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import hashlib


class DocumentType(Enum):
    """Classification of documents in an RFP bundle"""
    MAIN_SOLICITATION = "main_solicitation"      # SF33, SF1449, or main RFP PDF
    STATEMENT_OF_WORK = "statement_of_work"      # SOW / PWS / Attachment with scope
    RESEARCH_OUTLINE = "research_outline"        # NIH-specific: RO I, RO II, etc.
    CDRL = "cdrl"                                # Contract Data Requirements List
    AMENDMENT = "amendment"                      # Modifications to original RFP
    ATTACHMENT = "attachment"                    # General attachments (J-series, Exhibits)
    BUDGET_TEMPLATE = "budget_template"          # Excel pricing templates
    SECURITY = "security"                        # DD254, clearance requirements
    FORM = "form"                                # Standard forms (SF, OF, etc.)
    QA_RESPONSE = "qa_response"                  # Q&A from amendments


class RequirementType(Enum):
    """Semantic classification of requirements - not all 'shall' are equal"""
    
    # What the contractor MUST DO during performance
    PERFORMANCE = "performance"                  # "Contractor shall provide..."
    
    # What the contractor must WRITE in the proposal
    PROPOSAL_INSTRUCTION = "proposal_instruction"  # "Offeror shall describe..."
    
    # How the government will EVALUATE the proposal
    EVALUATION_CRITERION = "evaluation_criterion"  # "Government will evaluate..."
    
    # Metrics for measuring performance (QASP items)
    PERFORMANCE_METRIC = "performance_metric"    # "95% on-time delivery"
    
    # Deliverable specifications
    DELIVERABLE = "deliverable"                  # "Submit monthly report..."
    
    # Labor/staffing requirements
    LABOR_REQUIREMENT = "labor_requirement"      # "5,000 labor hours..."
    
    # Qualification/eligibility requirements
    QUALIFICATION = "qualification"              # "Must be small business..."
    
    # Compliance/regulatory requirements
    COMPLIANCE = "compliance"                    # "FAR 52.xxx applies..."
    
    # Format/administrative requirements
    FORMAT = "format"                            # "12-point font, 1-inch margins..."
    
    # Prohibition
    PROHIBITION = "prohibition"                  # "Shall not..."


class RequirementStatus(Enum):
    """Lifecycle status (for amendment tracking)"""
    ACTIVE = "active"                # Current, valid requirement
    MODIFIED = "modified"            # Changed by amendment
    DELETED = "deleted"              # Removed by amendment
    SUPERSEDED = "superseded"        # Replaced by new requirement
    CLARIFIED = "clarified"          # Q&A provided clarification


class ConfidenceLevel(Enum):
    """Extraction confidence"""
    HIGH = "high"          # Clear 'shall/must' with unambiguous context
    MEDIUM = "medium"      # Implicit requirement or unclear boundary
    LOW = "low"            # Inferred from context, may need verification


@dataclass
class SourceLocation:
    """Precise location of requirement in source document"""
    document_name: str           # Filename or identifier
    document_type: DocumentType
    page_number: int             # 1-indexed page
    section_id: str              # e.g., "C.3.1", "L.4.a.2", "RO-III.2"
    paragraph_index: int = 0     # Paragraph within section
    char_start: int = 0          # Character offset for precise citation
    char_end: int = 0


@dataclass
class RequirementNode:
    """
    A node in the Requirements Graph
    
    Core entity for tracking requirements with cross-document relationships
    """
    # Identity
    id: str                                      # REQ-001, REQ-C-001, etc.
    text: str                                    # Full requirement text
    text_hash: str = ""                          # For duplicate detection
    
    # Classification
    requirement_type: RequirementType = RequirementType.PERFORMANCE
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    status: RequirementStatus = RequirementStatus.ACTIVE
    
    # Source tracking
    source: Optional[SourceLocation] = None
    context_before: str = ""                     # Surrounding text for context
    context_after: str = ""
    
    # Extracted attributes
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # Named entities
    
    # Graph edges (relationships)
    references_to: List[str] = field(default_factory=list)      # This req references these
    referenced_by: List[str] = field(default_factory=list)      # These reference this req
    parent_requirement: Optional[str] = None                     # Hierarchical parent
    child_requirements: List[str] = field(default_factory=list)  # Sub-requirements
    
    # Cross-document links
    evaluated_by: List[str] = field(default_factory=list)       # Section M factors
    instructed_by: List[str] = field(default_factory=list)      # Section L instructions
    deliverable_for: Optional[str] = None                        # CDRL/DID reference
    research_outline: Optional[str] = None                       # NIH-specific: RO link
    clin_reference: Optional[str] = None                         # Section B CLIN
    
    # Amendment tracking
    version: int = 1
    modified_by_amendment: Optional[str] = None
    previous_text: Optional[str] = None
    modification_reason: Optional[str] = None
    
    # Metadata
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_method: str = "regex"             # "regex", "llm", "hybrid"
    
    def __post_init__(self):
        if not self.text_hash and self.text:
            self.text_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "text": self.text,
            "text_hash": self.text_hash,
            "requirement_type": self.requirement_type.value,
            "confidence": self.confidence.value,
            "status": self.status.value,
            "source": {
                "document_name": self.source.document_name if self.source else None,
                "document_type": self.source.document_type.value if self.source else None,
                "page_number": self.source.page_number if self.source else None,
                "section_id": self.source.section_id if self.source else None,
            } if self.source else None,
            "keywords": self.keywords,
            "entities": self.entities,
            "references_to": self.references_to,
            "referenced_by": self.referenced_by,
            "evaluated_by": self.evaluated_by,
            "instructed_by": self.instructed_by,
            "research_outline": self.research_outline,
            "clin_reference": self.clin_reference,
            "version": self.version,
            "modified_by_amendment": self.modified_by_amendment,
            "extracted_at": self.extracted_at,
            "extraction_method": self.extraction_method,
        }


@dataclass 
class RFPBundle:
    """
    Complete RFP package with all related documents
    
    Represents the full set of documents that make up a solicitation
    """
    solicitation_number: str
    title: str = ""
    agency: str = ""
    
    # Document inventory
    main_document: Optional[str] = None          # Path to main RFP
    sow_document: Optional[str] = None           # Statement of Work (may be attachment)
    amendments: List[str] = field(default_factory=list)
    attachments: Dict[str, str] = field(default_factory=dict)  # {id: path}
    research_outlines: Dict[str, str] = field(default_factory=dict)  # NIH-specific
    budget_templates: List[str] = field(default_factory=list)
    
    # Parsed content cache
    parsed_documents: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    set_aside: Optional[str] = None              # Small business, 8(a), etc.
    naics_code: Optional[str] = None
    contract_type: Optional[str] = None          # CPFF, FFP, T&M, etc.
    estimated_value: Optional[str] = None
    
    def total_documents(self) -> int:
        """Count total documents in bundle"""
        count = 0
        if self.main_document: count += 1
        if self.sow_document: count += 1
        count += len(self.amendments)
        count += len(self.attachments)
        count += len(self.research_outlines)
        count += len(self.budget_templates)
        return count


@dataclass
class ParsedDocument:
    """Result of parsing a single document"""
    filepath: str
    filename: str
    document_type: DocumentType
    
    # Content
    full_text: str
    pages: List[str]                             # Text by page
    page_count: int
    
    # Structure detection
    sections: Dict[str, str] = field(default_factory=dict)  # {section_id: text}
    tables: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    title: Optional[str] = None
    parsed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parser_used: str = "pypdf"                   # "pypdf", "docx", "markitdown"
    
    # Quality indicators
    extraction_quality: float = 1.0              # 0-1, based on text extraction success
    has_ocr_text: bool = False                   # Was OCR needed?


@dataclass
class ComplianceMatrixRow:
    """A row in the compliance matrix (Excel-ready)"""
    requirement_id: str
    requirement_text: str
    section_reference: str
    section_type: str                            # C, L, M, or attachment ID
    requirement_type: str
    page_number: str = "UNSPEC"                  # Page number in source document
    
    # Response fields (to be filled by proposal team)
    compliance_status: str = "Not Started"       # Compliant, Partial, Non-Compliant, N/A
    response_text: str = ""
    proposal_volume: str = ""                    # Which proposal volume (I, II, III, IV)
    proposal_section: str = ""                   # Where addressed in proposal
    win_theme: str = ""                          # Strategic win theme
    response_strategy: str = ""                  # How to respond
    assigned_owner: str = ""
    evidence_required: str = ""                  # What evidence is needed
    proof_points: str = ""                       # Discriminators/proof points
    
    # Traceability
    related_requirements: List[str] = field(default_factory=list)
    evaluation_factor: Optional[str] = None
    
    # Priority/risk
    priority: str = "Medium"                     # High, Medium, Low
    risk_if_non_compliant: str = ""
    mandatory_desirable: str = "Mandatory"       # Mandatory or Desirable
    
    notes: str = ""


@dataclass
class ExtractionResult:
    """Result of the enhanced compliance extraction"""
    # Core outputs
    requirements_graph: Dict[str, RequirementNode]
    compliance_matrix: List[ComplianceMatrixRow]
    
    # Statistics
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    extraction_coverage: float = 0.0             # Estimated % of requirements found
    cross_reference_count: int = 0               # Number of graph edges
    amendment_changes_tracked: int = 0
    
    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    
    # Errors/warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
