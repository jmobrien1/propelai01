"""
PropelAI v3.0: Extract-First Architecture Models

This module defines the data structures for the resilient extraction pipeline
that extracts first, classifies later, and never silently drops requirements.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import hashlib

if TYPE_CHECKING:
    from .models import SourceCoordinate


class ConfidenceLevel(Enum):
    """Graduated confidence levels - never binary found/not-found"""
    HIGH = "HIGH"           # >80% confidence - likely correct
    MEDIUM = "MEDIUM"       # 50-80% confidence - probably correct
    LOW = "LOW"             # 20-50% confidence - needs review
    UNCERTAIN = "UNCERTAIN" # <20% confidence - flagged for manual review


class DetectionMethod(Enum):
    """How a section or requirement was detected"""
    EXPLICIT_HEADER = "explicit_header"       # Found "SECTION C" header
    SOW_ATTACHMENT = "sow_attachment"         # SOW/PWS document detected
    CONTENT_HEADER = "content_header"         # Found "STATEMENT OF WORK" in text
    KEYWORD_DENSITY = "keyword_density"       # High shall/must density
    STRUCTURAL_PATTERN = "structural_pattern" # Section numbering like "1.0 SCOPE"
    FALLBACK_SCAN = "fallback_scan"           # Full document scan fallback
    MANUAL_OVERRIDE = "manual_override"       # User manually assigned


@dataclass
class SectionCandidate:
    """A potential section detection with confidence scoring"""
    section_type: str                          # "C", "L", "M", etc.
    start_offset: int
    end_offset: int
    title: str
    confidence: ConfidenceLevel
    confidence_score: float                    # 0.0 to 1.0
    detection_method: DetectionMethod
    source_document: Optional[str] = None
    detection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "section_type": self.section_type,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "title": self.title,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "detection_method": self.detection_method.value,
            "source_document": self.source_document,
            "detection_reasons": self.detection_reasons,
        }


@dataclass
class RequirementCandidate:
    """
    A potential requirement extracted from ANY document.
    Classification happens AFTER extraction, not during.
    """
    id: str
    text: str
    text_hash: str                              # For deduplication

    # Source information (where it came from)
    source_document: str
    source_page: int
    source_offset: int

    # v4.0 Trust Gate: Visual coordinates for PDF highlighting
    source_coordinates: Optional[List["SourceCoordinate"]] = None

    # Original context (RFP's own numbering if found)
    rfp_reference: Optional[str] = None         # e.g., "C.3.1.2", "L.4.B.2"

    # Classification (assigned AFTER extraction)
    assigned_section: Optional[str] = None      # "C", "L", "M", "J", etc.
    category: Optional[str] = None              # "TECHNICAL", "PROPOSAL_INSTRUCTION", etc.

    # Binding analysis
    binding_level: str = "INFORMATIONAL"        # SHALL, SHOULD, MAY, INFORMATIONAL
    binding_keyword: Optional[str] = None       # The actual keyword found

    # Confidence tracking
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.5
    classification_reasons: List[str] = field(default_factory=list)

    # Review flags
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute hash for deduplication"""
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "source_document": self.source_document,
            "source_page": self.source_page,
            "source_coordinates": [c.to_dict() for c in self.source_coordinates] if self.source_coordinates else None,
            "rfp_reference": self.rfp_reference,
            "assigned_section": self.assigned_section,
            "category": self.category,
            "binding_level": self.binding_level,
            "binding_keyword": self.binding_keyword,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "classification_reasons": self.classification_reasons,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
        }


@dataclass
class ExtractionQualityMetrics:
    """Metrics to detect when extraction quality is poor"""
    total_documents: int = 0
    total_pages: int = 0
    total_requirements: int = 0

    # Section coverage
    section_counts: Dict[str, int] = field(default_factory=dict)

    # Confidence distribution
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    uncertain_count: int = 0

    # Anomaly flags
    anomalies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Requirements density
    requirements_per_page: float = 0.0

    # SOW detection
    sow_detected: bool = False
    sow_source: Optional[str] = None

    def add_anomaly(self, message: str):
        self.anomalies.append(message)

    def add_warning(self, message: str):
        self.warnings.append(message)

    def compute_metrics(self, requirements: List[RequirementCandidate]):
        """Compute quality metrics from extracted requirements"""
        self.total_requirements = len(requirements)

        if self.total_pages > 0:
            self.requirements_per_page = self.total_requirements / self.total_pages

        # Count by section
        for req in requirements:
            section = req.assigned_section or "UNASSIGNED"
            self.section_counts[section] = self.section_counts.get(section, 0) + 1

            # Count by confidence
            if req.confidence == ConfidenceLevel.HIGH:
                self.high_confidence_count += 1
            elif req.confidence == ConfidenceLevel.MEDIUM:
                self.medium_confidence_count += 1
            elif req.confidence == ConfidenceLevel.LOW:
                self.low_confidence_count += 1
            else:
                self.uncertain_count += 1

        # Detect anomalies
        self._detect_anomalies()

    def _detect_anomalies(self):
        """Detect quality anomalies"""
        # Section C (SOW) should typically have significant requirements
        section_c_count = self.section_counts.get("C", 0)
        if section_c_count < 10 and self.total_requirements > 50:
            self.add_anomaly(
                f"LOW_SECTION_C: Only {section_c_count} Section C requirements "
                f"out of {self.total_requirements} total. SOW may not have been detected."
            )

        # Section L should have proposal instructions
        section_l_count = self.section_counts.get("L", 0)
        if section_l_count < 5 and self.total_requirements > 30:
            self.add_warning(
                f"LOW_SECTION_L: Only {section_l_count} Section L requirements. "
                f"Proposal instructions may be missing."
            )

        # Too many low-confidence items
        if self.total_requirements > 0:
            low_conf_ratio = (self.low_confidence_count + self.uncertain_count) / self.total_requirements
            if low_conf_ratio > 0.3:
                self.add_warning(
                    f"HIGH_UNCERTAINTY: {low_conf_ratio:.0%} of requirements have "
                    f"low or uncertain confidence. Manual review recommended."
                )

        # Very few requirements overall
        if self.total_requirements < 20 and self.total_pages > 10:
            self.add_anomaly(
                f"LOW_EXTRACTION: Only {self.total_requirements} requirements from "
                f"{self.total_pages} pages. Extraction may have failed."
            )

        # SOW not detected
        if not self.sow_detected:
            self.add_warning(
                "NO_SOW_DETECTED: Statement of Work not explicitly detected. "
                "Technical requirements may be miscategorized."
            )

    def to_dict(self) -> Dict:
        return {
            "total_documents": self.total_documents,
            "total_pages": self.total_pages,
            "total_requirements": self.total_requirements,
            "section_counts": self.section_counts,
            "confidence_distribution": {
                "high": self.high_confidence_count,
                "medium": self.medium_confidence_count,
                "low": self.low_confidence_count,
                "uncertain": self.uncertain_count,
            },
            "requirements_per_page": round(self.requirements_per_page, 2),
            "sow_detected": self.sow_detected,
            "sow_source": self.sow_source,
            "anomalies": self.anomalies,
            "warnings": self.warnings,
        }


@dataclass
class ExtractionResult:
    """Complete extraction result with all requirements and quality metrics"""
    rfp_id: str

    # All extracted requirements
    requirements: List[RequirementCandidate] = field(default_factory=list)

    # Section detections
    detected_sections: List[SectionCandidate] = field(default_factory=list)

    # Quality metrics
    quality_metrics: ExtractionQualityMetrics = field(default_factory=ExtractionQualityMetrics)

    # Items needing review
    review_queue: List[RequirementCandidate] = field(default_factory=list)

    # Extraction metadata
    extraction_version: str = "3.0"
    document_hashes: List[str] = field(default_factory=list)

    def add_requirement(self, req: RequirementCandidate):
        self.requirements.append(req)
        if req.needs_review:
            self.review_queue.append(req)

    def get_by_section(self, section: str) -> List[RequirementCandidate]:
        return [r for r in self.requirements if r.assigned_section == section]

    def get_high_confidence(self) -> List[RequirementCandidate]:
        return [r for r in self.requirements if r.confidence == ConfidenceLevel.HIGH]

    def finalize(self):
        """Compute final metrics after extraction complete"""
        self.quality_metrics.compute_metrics(self.requirements)

    def to_dict(self) -> Dict:
        return {
            "rfp_id": self.rfp_id,
            "extraction_version": self.extraction_version,
            "total_requirements": len(self.requirements),
            "review_queue_size": len(self.review_queue),
            "quality_metrics": self.quality_metrics.to_dict(),
            "detected_sections": [s.to_dict() for s in self.detected_sections],
            "requirements": [r.to_dict() for r in self.requirements],
        }
