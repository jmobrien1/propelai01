"""
PropelAI Validation Framework - Ground Truth Schemas

Defines the data structures for annotated RFP requirements that serve as
the source of truth for accuracy measurement.

Based on best practices from:
- Government Proposal Expert recommendations
- Technical architecture specifications
- SaaS product strategy requirements
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import hashlib
import json


class AnnotationStatus(Enum):
    """Status of a ground truth annotation"""
    DRAFT = "draft"              # Initial annotation, not yet reviewed
    REVIEWED = "reviewed"        # Reviewed by second annotator
    APPROVED = "approved"        # Final, approved for use as ground truth
    DISPUTED = "disputed"        # Disagreement between annotators


class RequirementBoundary(Enum):
    """How was the requirement text boundary determined?"""
    EXACT_SENTENCE = "exact_sentence"       # Single complete sentence
    PARAGRAPH = "paragraph"                  # Full paragraph
    MULTI_SENTENCE = "multi_sentence"        # Multiple related sentences
    PARTIAL_SENTENCE = "partial_sentence"    # Part of a compound sentence
    TABLE_CELL = "table_cell"               # Extracted from table


class BindingLevel(Enum):
    """How binding is this requirement? (Matches extractor)"""
    MANDATORY = "Mandatory"                  # SHALL, MUST, REQUIRED, WILL
    HIGHLY_DESIRABLE = "Highly Desirable"    # SHOULD, EXPECTED
    DESIRABLE = "Desirable"                  # MAY, CAN, ENCOURAGED
    INFORMATIONAL = "Informational"          # No binding language


class RequirementCategory(Enum):
    """High-level requirement categories (Matches extractor)"""
    L_COMPLIANCE = "L_COMPLIANCE"            # Section L - Instructions
    TECHNICAL = "TECHNICAL"                  # Section C/PWS/SOW - Performance
    EVALUATION = "EVALUATION"                # Section M - Scoring criteria
    ADMINISTRATIVE = "ADMINISTRATIVE"        # B/F/G/H/I/K - Contract admin
    ATTACHMENT = "ATTACHMENT"                # Requirements from attachments


class FalsePositiveType(Enum):
    """Types of false positive extractions"""
    TOC_ENTRY = "toc_entry"                  # Table of contents extracted as req
    HEADER = "header"                         # Section header extracted
    BOILERPLATE = "boilerplate"              # Standard clause, not a requirement
    INFORMATIONAL = "informational"          # Pure context, no obligation
    CROSS_REFERENCE = "cross_ref"            # Reference only, not standalone
    INCOMPLETE = "incomplete"                # Fragment, not complete requirement
    DUPLICATE = "duplicate"                  # Same requirement extracted twice
    GOVERNMENT_OBLIGATION = "gov_obligation" # Government action, not contractor


class FalseNegativeType(Enum):
    """Types of missed requirements"""
    NOT_EXTRACTED = "not_extracted"          # Requirement not found at all
    PARTIAL_MATCH = "partial_match"          # Found but incomplete text
    WRONG_SECTION = "wrong_section"          # Extracted but wrong section
    WRONG_BINDING = "wrong_binding"          # Extracted but wrong binding level
    MERGED = "merged"                        # Multiple requirements merged
    FILTERED = "filtered"                    # Filtered by noise detection


@dataclass
class GroundTruthRequirement:
    """
    A human-annotated requirement from an RFP.
    This serves as the source of truth for accuracy measurement.

    Annotation Guidelines:
    1. Mark EVERY "shall/must/will" as Mandatory
    2. Mark EVERY "should/expected" as Highly Desirable
    3. Decompose compound requirements ("shall X, Y, and Z" = 3 requirements)
    4. Include implicit requirements (FAR/DFARS references)
    5. Record exact source location (page/paragraph/character offset)
    """
    # Core identification
    gt_id: str                              # GT-{rfp_id}-{sequence}, e.g., "GT-NIH-001"
    rfp_id: str                             # Reference to parent RFP

    # Requirement text (verbatim from RFP)
    text: str                               # Full requirement text (NEVER summarized)
    text_normalized: str = ""               # Lowercased, whitespace-normalized
    text_hash: str = ""                     # MD5 hash for matching

    # Location in source document
    source_document: str = ""               # Filename
    page_number: int = 0
    char_start: int = 0                     # Character offset in full text
    char_end: int = 0
    line_number: Optional[int] = None

    # RFP Structure Classification
    rfp_section: str = ""                   # L, M, C, B, F, etc.
    rfp_subsection: str = ""                # L.4.B.2, C.3.1.a
    rfp_reference_verbatim: str = ""        # Exact reference as it appears

    # Semantic Classification
    category: str = ""                      # L_COMPLIANCE, TECHNICAL, EVALUATION, etc.
    requirement_type: str = ""              # performance, proposal_instruction, etc.

    # Binding Level
    binding_level: str = ""                 # Mandatory, Highly Desirable, Desirable, Informational
    binding_keyword: str = ""               # shall, must, should, may, etc.
    binding_keyword_offset: int = 0         # Position of keyword in text

    # Boundary determination
    boundary_type: str = "paragraph"        # From RequirementBoundary enum

    # Cross-references detected in text
    references_to: List[str] = field(default_factory=list)

    # For compound requirements (e.g., "shall A, B, and C")
    is_compound: bool = False
    compound_parts: List[str] = field(default_factory=list)  # IDs of decomposed parts
    parent_compound_id: Optional[str] = None  # If this is part of a compound

    # Annotation metadata
    annotator_id: str = ""
    annotation_timestamp: str = ""
    annotation_status: str = "draft"        # From AnnotationStatus enum
    annotation_notes: str = ""
    confidence_rating: float = 1.0          # Annotator's confidence (0-1)

    # Dispute resolution (for inter-annotator disagreements)
    alternative_classifications: List[Dict] = field(default_factory=list)
    resolution_notes: str = ""
    resolver_id: str = ""

    # Version tracking
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Compute derived fields after initialization"""
        if self.text and not self.text_normalized:
            self.text_normalized = ' '.join(self.text.lower().split())

        if self.text and not self.text_hash:
            self.text_hash = hashlib.md5(self.text_normalized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "gt_id": self.gt_id,
            "rfp_id": self.rfp_id,
            "text": self.text,
            "text_normalized": self.text_normalized,
            "text_hash": self.text_hash,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "line_number": self.line_number,
            "rfp_section": self.rfp_section,
            "rfp_subsection": self.rfp_subsection,
            "rfp_reference_verbatim": self.rfp_reference_verbatim,
            "category": self.category,
            "requirement_type": self.requirement_type,
            "binding_level": self.binding_level,
            "binding_keyword": self.binding_keyword,
            "binding_keyword_offset": self.binding_keyword_offset,
            "boundary_type": self.boundary_type,
            "references_to": self.references_to,
            "is_compound": self.is_compound,
            "compound_parts": self.compound_parts,
            "parent_compound_id": self.parent_compound_id,
            "annotator_id": self.annotator_id,
            "annotation_timestamp": self.annotation_timestamp,
            "annotation_status": self.annotation_status,
            "annotation_notes": self.annotation_notes,
            "confidence_rating": self.confidence_rating,
            "alternative_classifications": self.alternative_classifications,
            "resolution_notes": self.resolution_notes,
            "resolver_id": self.resolver_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthRequirement":
        """Create from dictionary (JSON deserialization)"""
        return cls(
            gt_id=data.get("gt_id", ""),
            rfp_id=data.get("rfp_id", ""),
            text=data.get("text", ""),
            text_normalized=data.get("text_normalized", ""),
            text_hash=data.get("text_hash", ""),
            source_document=data.get("source_document", ""),
            page_number=data.get("page_number", 0),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
            line_number=data.get("line_number"),
            rfp_section=data.get("rfp_section", ""),
            rfp_subsection=data.get("rfp_subsection", ""),
            rfp_reference_verbatim=data.get("rfp_reference_verbatim", ""),
            category=data.get("category", ""),
            requirement_type=data.get("requirement_type", ""),
            binding_level=data.get("binding_level", ""),
            binding_keyword=data.get("binding_keyword", ""),
            binding_keyword_offset=data.get("binding_keyword_offset", 0),
            boundary_type=data.get("boundary_type", "paragraph"),
            references_to=data.get("references_to", []),
            is_compound=data.get("is_compound", False),
            compound_parts=data.get("compound_parts", []),
            parent_compound_id=data.get("parent_compound_id"),
            annotator_id=data.get("annotator_id", ""),
            annotation_timestamp=data.get("annotation_timestamp", ""),
            annotation_status=data.get("annotation_status", "draft"),
            annotation_notes=data.get("annotation_notes", ""),
            confidence_rating=data.get("confidence_rating", 1.0),
            alternative_classifications=data.get("alternative_classifications", []),
            resolution_notes=data.get("resolution_notes", ""),
            resolver_id=data.get("resolver_id", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class GroundTruthRFP:
    """
    Complete ground truth dataset for a single RFP.

    This is the master record containing all annotated requirements
    for an RFP, used as the source of truth for accuracy testing.
    """
    # RFP identification
    rfp_id: str                             # Unique identifier
    solicitation_number: str                # Official solicitation number
    agency: str                             # NIH, DoD, GSA, HHS, etc.
    rfp_type: str                           # Full-and-Open, IDIQ, BPA, etc.
    document_format: str                    # UCF_STANDARD, GSA_SCHEDULE, etc.

    # Descriptive fields
    title: str = ""
    issue_date: str = ""
    due_date: str = ""

    # Source documents
    documents: List[Dict[str, str]] = field(default_factory=list)
    # Each document: {filename, filepath, doc_type, page_count}
    total_pages: int = 0

    # Ground truth requirements
    requirements: List[GroundTruthRequirement] = field(default_factory=list)

    # Statistics (computed)
    stats: Dict[str, Any] = field(default_factory=dict)

    # Annotation metadata
    primary_annotator: str = ""
    secondary_annotator: str = ""           # For IAA calculation
    annotation_complete: bool = False
    annotation_start_date: str = ""
    annotation_end_date: str = ""
    annotation_guidelines_version: str = "1.0"

    # Inter-annotator agreement scores (if applicable)
    iaa_scores: Dict[str, float] = field(default_factory=dict)

    # Version tracking
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_stats(self) -> Dict[str, Any]:
        """Compute statistics from requirements"""
        self.stats = {
            "total_requirements": len(self.requirements),
            "by_section": {},
            "by_category": {},
            "by_binding_level": {},
            "by_requirement_type": {},
            "compound_requirements": sum(1 for r in self.requirements if r.is_compound),
            "with_cross_references": sum(1 for r in self.requirements if r.references_to),
        }

        # Count by section
        for req in self.requirements:
            section = req.rfp_section or "UNKNOWN"
            self.stats["by_section"][section] = self.stats["by_section"].get(section, 0) + 1

            category = req.category or "UNKNOWN"
            self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1

            binding = req.binding_level or "UNKNOWN"
            self.stats["by_binding_level"][binding] = self.stats["by_binding_level"].get(binding, 0) + 1

            req_type = req.requirement_type or "UNKNOWN"
            self.stats["by_requirement_type"][req_type] = self.stats["by_requirement_type"].get(req_type, 0) + 1

        return self.stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "rfp_id": self.rfp_id,
            "solicitation_number": self.solicitation_number,
            "agency": self.agency,
            "rfp_type": self.rfp_type,
            "document_format": self.document_format,
            "title": self.title,
            "issue_date": self.issue_date,
            "due_date": self.due_date,
            "documents": self.documents,
            "total_pages": self.total_pages,
            "requirements": [r.to_dict() for r in self.requirements],
            "stats": self.stats,
            "primary_annotator": self.primary_annotator,
            "secondary_annotator": self.secondary_annotator,
            "annotation_complete": self.annotation_complete,
            "annotation_start_date": self.annotation_start_date,
            "annotation_end_date": self.annotation_end_date,
            "annotation_guidelines_version": self.annotation_guidelines_version,
            "iaa_scores": self.iaa_scores,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundTruthRFP":
        """Create from dictionary (JSON deserialization)"""
        requirements = [
            GroundTruthRequirement.from_dict(r)
            for r in data.get("requirements", [])
        ]

        return cls(
            rfp_id=data.get("rfp_id", ""),
            solicitation_number=data.get("solicitation_number", ""),
            agency=data.get("agency", ""),
            rfp_type=data.get("rfp_type", ""),
            document_format=data.get("document_format", ""),
            title=data.get("title", ""),
            issue_date=data.get("issue_date", ""),
            due_date=data.get("due_date", ""),
            documents=data.get("documents", []),
            total_pages=data.get("total_pages", 0),
            requirements=requirements,
            stats=data.get("stats", {}),
            primary_annotator=data.get("primary_annotator", ""),
            secondary_annotator=data.get("secondary_annotator", ""),
            annotation_complete=data.get("annotation_complete", False),
            annotation_start_date=data.get("annotation_start_date", ""),
            annotation_end_date=data.get("annotation_end_date", ""),
            annotation_guidelines_version=data.get("annotation_guidelines_version", "1.0"),
            iaa_scores=data.get("iaa_scores", {}),
            schema_version=data.get("schema_version", "1.0.0"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def save(self, filepath: str) -> None:
        """Save ground truth to JSON file"""
        self.updated_at = datetime.now().isoformat()
        self.compute_stats()

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "GroundTruthRFP":
        """Load ground truth from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# JSON Schema for validation
GROUND_TRUTH_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PropelAI Ground Truth RFP",
    "type": "object",
    "required": ["rfp_id", "solicitation_number", "requirements", "schema_version"],
    "properties": {
        "rfp_id": {"type": "string"},
        "solicitation_number": {"type": "string"},
        "agency": {
            "type": "string",
            "enum": ["NIH", "DoD", "GSA", "HHS", "VA", "NASA", "DHS", "OTHER"]
        },
        "rfp_type": {"type": "string"},
        "document_format": {
            "type": "string",
            "enum": ["UCF_STANDARD", "GSA_SCHEDULE", "IDIQ_TASK", "DOD_CUSTOM", "OTHER"]
        },
        "requirements": {
            "type": "array",
            "items": {"$ref": "#/definitions/requirement"}
        },
        "schema_version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+$"
        }
    },
    "definitions": {
        "requirement": {
            "type": "object",
            "required": ["gt_id", "text", "rfp_section", "binding_level", "category"],
            "properties": {
                "gt_id": {
                    "type": "string",
                    "pattern": "^GT-.*-\\d+$"
                },
                "text": {
                    "type": "string",
                    "minLength": 20
                },
                "rfp_section": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "PWS", "SOW", "ATTACHMENT"]
                },
                "binding_level": {
                    "type": "string",
                    "enum": ["Mandatory", "Highly Desirable", "Desirable", "Informational"]
                },
                "category": {
                    "type": "string",
                    "enum": ["L_COMPLIANCE", "TECHNICAL", "EVALUATION", "ADMINISTRATIVE", "ATTACHMENT"]
                },
                "page_number": {"type": "integer", "minimum": 0},
                "char_start": {"type": "integer", "minimum": 0},
                "char_end": {"type": "integer", "minimum": 0},
                "annotation_status": {
                    "type": "string",
                    "enum": ["draft", "reviewed", "approved", "disputed"]
                }
            }
        }
    }
}
