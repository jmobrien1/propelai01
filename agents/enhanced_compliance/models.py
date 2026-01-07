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
import re


# ============================================================================
# v6.0.7: Multi-Vehicle Procurement Support
# ============================================================================

class ProcurementType(Enum):
    """
    v6.0.7: Classification of procurement vehicle types.

    Different procurement vehicles have different structures:
    - UCF (Uniform Contract Format): Traditional FAR 15 with Sections L/M/C
    - GSA RFQ: FAR 8.4 Schedule buys with simplified quote structure
    - IDIQ Task Order: FAR 16.505 with task-specific instructions
    - BPA Call: Blanket Purchase Agreement calls
    - Simplified: Under SAT threshold, minimal structure
    """
    UCF_STANDARD = "ucf_standard"        # FAR 15, DFARS - Air Force, Army, Navy
    GSA_RFQ_8_4 = "gsa_rfq_8_4"          # FAR 8.4 GSA Schedule buys
    GSA_BPA_CALL = "gsa_bpa_call"        # Blanket Purchase Agreement calls
    IDIQ_TASK_ORDER = "idiq_task_order"  # FAR 16.505 (OASIS, Alliant, SEWP)
    SIMPLIFIED_SAP = "simplified_sap"    # Under $250K threshold
    SOURCES_SOUGHT = "sources_sought"    # Market research (no proposal structure)
    UNKNOWN = "unknown"                  # Could not determine


class DetectionConfidence(Enum):
    """v6.0.7: Confidence levels for procurement type detection."""
    HIGH = "high"           # >85% - proceed silently
    MEDIUM = "medium"       # 65-85% - proceed with warning
    LOW = "low"             # 45-65% - prompt user to confirm
    VERY_LOW = "very_low"   # <45% - block and guide


@dataclass
class ProcurementDetectionResult:
    """
    v6.0.7: Result of auto-detecting procurement type.

    Contains the detected type, confidence level, and evidence
    supporting the detection decision.
    """
    procurement_type: ProcurementType
    confidence: DetectionConfidence
    confidence_score: float  # 0.0-1.0
    detected_signals: List[str] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    suggestion: Optional[str] = None

    @property
    def should_proceed(self) -> bool:
        """Whether to proceed without user confirmation."""
        return self.confidence in [DetectionConfidence.HIGH, DetectionConfidence.MEDIUM]

    @property
    def requires_confirmation(self) -> bool:
        """Whether to prompt user for confirmation."""
        return self.confidence == DetectionConfidence.LOW

    @property
    def should_block(self) -> bool:
        """Whether to block and provide guidance."""
        return self.confidence == DetectionConfidence.VERY_LOW


class ProcurementTypeDetector:
    """
    v6.0.7: Auto-detects procurement type from document text.

    Uses pattern matching to identify:
    - FAR references (8.4, 15, 16.505)
    - Solicitation number patterns (FA for Air Force, 693JJ4 for DOT, etc.)
    - Document structure markers (Section L/M, Phase I/II, etc.)
    """

    # Confidence thresholds
    THRESHOLD_HIGH = 0.85
    THRESHOLD_MEDIUM = 0.65
    THRESHOLD_LOW = 0.45

    # Pattern scores (higher = more indicative)
    PATTERNS = {
        ProcurementType.UCF_STANDARD: {
            'far_15': (r'FAR\s*15', 30),
            'dfars': (r'DFARS|DFAR\s+\d', 25),
            'section_l': (r'Section\s+L', 25),
            'section_m': (r'Section\s+M', 25),
            'volume_i': (r'Volume\s+[IVX]+\s*[:\-]', 20),
            'fa_contract': (r'\bFA\d{4}[-]?\d{2}[-]?[RQ]', 15),
            'w_contract': (r'\bW\d{3}[A-Z]{2}', 15),
            'n_contract': (r'\bN\d{5}', 15),
            'offeror_shall': (r'[Oo]fferor\s+shall', 10),
        },
        ProcurementType.GSA_RFQ_8_4: {
            'far_8_4': (r'FAR\s*8\.4', 40),
            'gsa_schedule': (r'GSA\s+Schedule|Federal\s+Supply\s+Schedule', 35),
            'rfq': (r'Request\s+for\s+Quote|RFQ', 25),
            'quote': (r'\bquote\b|\bquotation\b', 15),
            'gsa_contract': (r'\b47Q[A-Z]{2}|GS-\d{2}F', 20),
            'dot_contract': (r'\b693JJ4', 20),
            'phase_i_ii': (r'Phase\s+[I1]\s*[-/]|Phase\s+[II2]', 15),
            'bpa': (r'Blanket\s+Purchase\s+Agreement', 10),
        },
        ProcurementType.IDIQ_TASK_ORDER: {
            'far_16_505': (r'FAR\s*16\.505', 40),
            'task_order': (r'Task\s+Order|TO\s+Request', 35),
            'idiq': (r'\bIDIQ\b', 30),
            'oasis': (r'\bOASIS\b', 25),
            'alliant': (r'\bAlliant\b', 25),
            'sewp': (r'\bSEWP\b', 25),
            'cio_sp3': (r'CIO[-\s]?SP3', 25),
        },
        ProcurementType.SIMPLIFIED_SAP: {
            'simplified': (r'Simplified\s+Acquisition', 40),
            'far_13': (r'FAR\s*13', 35),
            'under_sat': (r'under\s+(?:the\s+)?(?:SAT|simplified)', 25),
            'micro_purchase': (r'micro[-\s]?purchase', 20),
        },
        ProcurementType.SOURCES_SOUGHT: {
            'sources_sought': (r'Sources\s+Sought', 50),
            'rfi': (r'Request\s+for\s+Information|RFI', 40),
            'market_research': (r'Market\s+Research', 30),
            'capability_statement': (r'Capability\s+Statement', 25),
        },
    }

    # Negative signals (reduce confidence for certain types)
    NEGATIVE_SIGNALS = {
        ProcurementType.UCF_STANDARD: [
            (r'FAR\s*8\.4', -40),  # FAR 8.4 strongly indicates NOT UCF
            (r'Request\s+for\s+Quote|RFQ', -30),
        ],
        ProcurementType.GSA_RFQ_8_4: [
            (r'Section\s+L.*Instructions', -30),  # Full Section L is NOT GSA RFQ
            (r'FAR\s*15', -40),
        ],
    }

    def detect(self, text: str, solicitation_number: Optional[str] = None) -> ProcurementDetectionResult:
        """
        Detect procurement type from document text.

        Args:
            text: Full document text to analyze
            solicitation_number: Optional known solicitation number

        Returns:
            ProcurementDetectionResult with type, confidence, and evidence
        """
        if not text:
            return ProcurementDetectionResult(
                procurement_type=ProcurementType.UNKNOWN,
                confidence=DetectionConfidence.VERY_LOW,
                confidence_score=0.0,
                detected_signals=["No document text provided"],
                suggestion="Please upload a valid solicitation document."
            )

        text_upper = text.upper()
        scores: Dict[ProcurementType, float] = {}
        signals: Dict[ProcurementType, List[str]] = {}

        # Score each procurement type
        for ptype, patterns in self.PATTERNS.items():
            score = 0
            type_signals = []

            for name, (pattern, weight) in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    score += weight
                    type_signals.append(f"Found '{name}': {len(matches)} match(es)")

            # Apply negative signals
            if ptype in self.NEGATIVE_SIGNALS:
                for pattern, penalty in self.NEGATIVE_SIGNALS[ptype]:
                    if re.search(pattern, text, re.IGNORECASE):
                        score += penalty  # penalty is negative
                        type_signals.append(f"Negative signal: {pattern}")

            scores[ptype] = max(0, score)  # No negative total scores
            signals[ptype] = type_signals

        # Find best match
        if not scores or max(scores.values()) == 0:
            return ProcurementDetectionResult(
                procurement_type=ProcurementType.UNKNOWN,
                confidence=DetectionConfidence.VERY_LOW,
                confidence_score=0.0,
                detected_signals=["No procurement type indicators found"],
                suggestion="This document doesn't contain recognizable procurement format markers. "
                          "Please verify this is a government solicitation document."
            )

        # Normalize scores to 0-1 range
        max_possible = 150  # Rough max if all patterns match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        normalized_score = min(1.0, best_score / max_possible)

        # Check for conflicting signals (second-best score close to best)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        conflicting = []
        if len(sorted_scores) > 1:
            second_type, second_score = sorted_scores[1]
            if second_score > best_score * 0.7:  # Within 70% of best
                conflicting.append(f"Also matches {second_type.value} ({second_score:.0f} points)")
                normalized_score *= 0.85  # Reduce confidence due to ambiguity

        # Determine confidence level
        if normalized_score >= self.THRESHOLD_HIGH:
            confidence = DetectionConfidence.HIGH
        elif normalized_score >= self.THRESHOLD_MEDIUM:
            confidence = DetectionConfidence.MEDIUM
        elif normalized_score >= self.THRESHOLD_LOW:
            confidence = DetectionConfidence.LOW
        else:
            confidence = DetectionConfidence.VERY_LOW

        # Build suggestion based on result
        suggestion = self._build_suggestion(best_type, confidence, conflicting)

        return ProcurementDetectionResult(
            procurement_type=best_type,
            confidence=confidence,
            confidence_score=normalized_score,
            detected_signals=signals.get(best_type, []),
            conflicting_signals=conflicting,
            suggestion=suggestion
        )

    def _build_suggestion(
        self,
        ptype: ProcurementType,
        confidence: DetectionConfidence,
        conflicts: List[str]
    ) -> str:
        """Build user-friendly suggestion based on detection result."""

        if confidence == DetectionConfidence.HIGH:
            return f"Detected as {ptype.value}. Proceeding with appropriate parser."

        if confidence == DetectionConfidence.MEDIUM:
            if conflicts:
                return (f"Detected as {ptype.value} but with some ambiguity. "
                       f"Review the generated outline carefully.")
            return f"Detected as {ptype.value}. Some manual verification recommended."

        if confidence == DetectionConfidence.LOW:
            return (f"Possibly {ptype.value}, but confidence is low. "
                   f"Please confirm this is the correct procurement type, "
                   f"or select the correct type manually.")

        # VERY_LOW
        return ("Could not confidently determine procurement type. "
               "Please select the procurement type manually or contact support.")


# ============================================================================
# Original Document Types (unchanged)
# ============================================================================


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
    
    # Response fields (to be filled by proposal team)
    compliance_status: str = "Not Started"       # Compliant, Partial, Non-Compliant, N/A
    response_text: str = ""
    proposal_section: str = ""                   # Where addressed in proposal
    assigned_owner: str = ""
    evidence_required: List[str] = field(default_factory=list)
    
    # Traceability
    related_requirements: List[str] = field(default_factory=list)
    evaluation_factor: Optional[str] = None
    
    # Priority/risk
    priority: str = "Medium"                     # High, Medium, Low
    risk_if_non_compliant: str = ""
    
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


# =============================================================================
# Trust Gate Models (v4.0)
# =============================================================================

@dataclass
class BoundingBox:
    """
    Bounding box coordinates for text in a PDF document.
    Enables one-click source verification by highlighting exact locations.
    """
    x0: float  # Left edge
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge
    page_width: float
    page_height: float

    def to_normalized(self) -> Dict[str, float]:
        """Convert to normalized 0-1 coordinates for web display"""
        return {
            "x": self.x0 / self.page_width if self.page_width else 0,
            "y": self.y0 / self.page_height if self.page_height else 0,
            "width": (self.x1 - self.x0) / self.page_width if self.page_width else 0,
            "height": (self.y1 - self.y0) / self.page_height if self.page_height else 0,
        }

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page_width": self.page_width,
            "page_height": self.page_height,
        }


@dataclass
class VisualRect:
    """
    A single visual rectangle on a specific page.
    Used for multi-page spanning requirements (FR-1.3).
    """
    page_number: int
    bounding_box: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict(),
            "normalized_box": self.bounding_box.to_normalized(),
        }


@dataclass
class SourceCoordinate:
    """
    Links a requirement to its exact location in the source document.
    Used by the Trust Gate to enable one-click verification.

    v5.0 Enhancement (FR-1.3): Supports multi-page spanning via visual_rects.
    """
    document_id: str
    page_number: int  # Primary page (first page of spanning content)
    bounding_box: BoundingBox  # Primary bounding box (for backwards compatibility)
    text_snippet: str = ""
    extraction_method: str = "pdfplumber"  # pdfplumber, pypdf, ocr
    confidence: float = 1.0
    # v5.0: Multi-page spanning support
    visual_rects: List["VisualRect"] = field(default_factory=list)
    spans_pages: bool = False

    def __post_init__(self):
        """Initialize visual_rects from primary bounding box if empty"""
        if not self.visual_rects:
            self.visual_rects = [
                VisualRect(
                    page_number=self.page_number,
                    bounding_box=self.bounding_box
                )
            ]
        self.spans_pages = len(self.visual_rects) > 1 or (
            len(self.visual_rects) == 1 and
            self.visual_rects[0].page_number != self.page_number
        )

    def add_visual_rect(self, page_number: int, bbox: BoundingBox) -> None:
        """Add an additional visual rectangle (for multi-page spanning)"""
        self.visual_rects.append(VisualRect(page_number=page_number, bounding_box=bbox))
        self.spans_pages = len(set(vr.page_number for vr in self.visual_rects)) > 1

    def get_all_pages(self) -> List[int]:
        """Get list of all pages this coordinate spans"""
        return sorted(set(vr.page_number for vr in self.visual_rects))

    def get_rects_for_page(self, page_number: int) -> List[BoundingBox]:
        """Get all bounding boxes for a specific page"""
        return [
            vr.bounding_box for vr in self.visual_rects
            if vr.page_number == page_number
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "bounding_box": self.bounding_box.to_dict(),
            "normalized_box": self.bounding_box.to_normalized(),
            "text_snippet": self.text_snippet,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            # v5.0: Multi-page spanning fields
            "spans_pages": self.spans_pages,
            "visual_rects": [vr.to_dict() for vr in self.visual_rects],
            "all_pages": self.get_all_pages(),
        }
