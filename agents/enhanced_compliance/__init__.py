"""
PropelAI Enhanced Compliance Module v3.1

New in v3.1:
- Accuracy Audit framework for extraction validation
- Section Inference for improved section detection
- Source Traceability with character offsets
- SQLite persistence layer
"""

__version__ = "3.1.0"
__author__ = "PropelAI Team"

# =============================================================================
# Core Data Models (v3.0)
# =============================================================================
from .ctm_data_models import (
    ScoringType, ResponseFormat, RequirementType, RFPSection, ComplianceStatus,
    PageLimit, FormattingRequirement, EvidenceRequirement, KeyPersonnelRequirement,
    EnhancedRequirement, ComplianceMatrix,
    create_pass_fail_requirement, create_weighted_requirement, create_future_diligence_requirement,
)

from .ctm_extractor import (
    EnhancedCTMExtractor, ScoringPatterns, ResponseFormatPatterns, PageLimitPatterns,
    FormattingPatterns, FutureDiligencePatterns, KeyPersonnelPatterns, ConstraintPatterns,
    process_requirements_batch,
)

from .ctm_integration import (
    LegacyRequirementAdapter, CTMEnricher, format_ctm_for_api,
    format_requirement_for_outline, get_content_allocation_guidance,
)

# =============================================================================
# Legacy imports required by api/main.py
# =============================================================================
from .agent import EnhancedComplianceAgent
from .amendment_processor import AmendmentProcessor
from .excel_export import export_to_excel

# Multi-format parser
try:
    from .parser import MultiFormatParser
except ImportError:
    MultiFormatParser = None

# Core models (DocumentType for RFP processing)
try:
    from .models import DocumentType, RequirementType as ModelRequirementType, ParsedDocument as ModelParsedDocument
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    from enum import Enum
    class DocumentType(Enum):
        MAIN_SOLICITATION = "main_solicitation"
        STATEMENT_OF_WORK = "statement_of_work"
        AMENDMENT = "amendment"
        ATTACHMENT = "attachment"
        BUDGET_TEMPLATE = "budget_template"
        QA_RESPONSE = "qa_response"

# Semantic extractor
try:
    from .semantic_extractor import SemanticRequirementExtractor
    from .semantic_ctm_export import SemanticCTMExporter, SemanticExtractionResult
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticRequirementExtractor = None
    SemanticCTMExporter = None
    SemanticExtractionResult = None

# Smart Outline Generator
try:
    from .smart_outline_generator import (
        SmartOutlineGenerator, ProposalOutline, ProposalVolume, ProposalSection,
    )
    OUTLINE_GENERATOR_AVAILABLE = True
except ImportError:
    OUTLINE_GENERATOR_AVAILABLE = False
    SmartOutlineGenerator = ProposalOutline = ProposalVolume = ProposalSection = None

# Company Library
try:
    from .company_library import (
        CompanyLibrary, CompanyLibraryParser, DocumentType as CompanyDocumentType, ParsedDocument as CompanyParsedDocument,
        CompanyProfile, Capability, PastPerformance, KeyPersonnel, Differentiator,
    )
    COMPANY_LIBRARY_AVAILABLE = True
except ImportError:
    COMPANY_LIBRARY_AVAILABLE = False
    CompanyLibrary = CompanyLibraryParser = DocumentType = ParsedDocument = None
    CompanyProfile = Capability = PastPerformance = KeyPersonnel = Differentiator = None

# Document Structure Parser (must come before section_aware_extractor)
try:
    from .document_structure import (
        RFPStructureParser, DocumentStructure, UCFSection,
        SectionBoundary, SubsectionBoundary, AttachmentInfo, analyze_rfp_structure,
    )
    DOCUMENT_STRUCTURE_AVAILABLE = True
except ImportError:
    DOCUMENT_STRUCTURE_AVAILABLE = False
    RFPStructureParser = DocumentStructure = UCFSection = None
    SectionBoundary = SubsectionBoundary = AttachmentInfo = analyze_rfp_structure = None

# Section Aware Extractor (must come before best_practices_ctm)
try:
    from .section_aware_extractor import (
        SectionAwareExtractor, StructuredRequirement, ExtractionResult,
        RequirementCategory, BindingLevel, extract_requirements_structured,
    )
    SECTION_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Section Aware Extractor not available: {e}")
    SECTION_EXTRACTOR_AVAILABLE = False
    SectionAwareExtractor = StructuredRequirement = ExtractionResult = None
    RequirementCategory = BindingLevel = extract_requirements_structured = None

# Best Practices CTM (depends on section_aware_extractor)
try:
    from .best_practices_ctm import (
        BestPracticesCTMExporter, export_ctm_best_practices,
    )
    BEST_PRACTICES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Best Practices CTM not available: {e}")
    BEST_PRACTICES_AVAILABLE = False
    BestPracticesCTMExporter = export_ctm_best_practices = None

__all__ = [
    "__version__",
    # v3.0 Data Models
    "ScoringType", "ResponseFormat", "RequirementType", "RFPSection", "ComplianceStatus",
    "PageLimit", "FormattingRequirement", "EvidenceRequirement", "KeyPersonnelRequirement",
    "EnhancedRequirement", "ComplianceMatrix",
    "create_pass_fail_requirement", "create_weighted_requirement", "create_future_diligence_requirement",
    "EnhancedCTMExtractor", "process_requirements_batch",
    "ScoringPatterns", "ResponseFormatPatterns", "PageLimitPatterns", "FormattingPatterns",
    "FutureDiligencePatterns", "KeyPersonnelPatterns", "ConstraintPatterns",
    "LegacyRequirementAdapter", "CTMEnricher", "format_ctm_for_api",
    "format_requirement_for_outline", "get_content_allocation_guidance",
    # Legacy required by main.py
    "EnhancedComplianceAgent", "AmendmentProcessor", "export_to_excel",
    "MultiFormatParser",
    "SEMANTIC_AVAILABLE", "SemanticRequirementExtractor", "SemanticCTMExporter", "SemanticExtractionResult",
    # Optional components
    "SmartOutlineGenerator", "ProposalOutline", "ProposalVolume", "ProposalSection",
    "CompanyLibrary", "CompanyLibraryParser", "DocumentType", "ParsedDocument",
    "CompanyProfile", "Capability", "PastPerformance", "KeyPersonnel", "Differentiator",
    # Document Structure
    "RFPStructureParser", "DocumentStructure", "UCFSection", "SectionBoundary",
    "SubsectionBoundary", "AttachmentInfo", "analyze_rfp_structure",
    "DOCUMENT_STRUCTURE_AVAILABLE",
    # Section Aware Extractor
    "SectionAwareExtractor", "StructuredRequirement", "ExtractionResult",
    "RequirementCategory", "BindingLevel", "extract_requirements_structured",
    "SECTION_EXTRACTOR_AVAILABLE",
    # Best Practices CTM
    "BestPracticesCTMExporter", "export_ctm_best_practices",
    "BEST_PRACTICES_AVAILABLE",
    # Availability flags
    "OUTLINE_GENERATOR_AVAILABLE", "COMPANY_LIBRARY_AVAILABLE",
]

# =============================================================================
# Annotated Outline Exporter (v1.0)
# =============================================================================
try:
    from .annotated_outline_exporter import (
        AnnotatedOutlineExporter,
        AnnotatedOutlineConfig,
        generate_annotated_outline
    )
    ANNOTATED_OUTLINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Annotated outline exporter not available: {e}")
    ANNOTATED_OUTLINE_AVAILABLE = False
    AnnotatedOutlineExporter = None
    AnnotatedOutlineConfig = None
    generate_annotated_outline = None

# =============================================================================
# v3.1 New Modules
# =============================================================================

# Accuracy Audit Framework
try:
    from .accuracy_audit import (
        AccuracyAuditor, AccuracyReport, AuditFinding, AuditSeverity,
        ConflictFinding, ComplianceGate, audit_extraction,
    )
    ACCURACY_AUDIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Accuracy audit not available: {e}")
    ACCURACY_AUDIT_AVAILABLE = False
    AccuracyAuditor = AccuracyReport = AuditFinding = AuditSeverity = None
    ConflictFinding = ComplianceGate = audit_extraction = None

# Section Inference
try:
    from .section_inference import (
        SectionInferencer, InferenceResult, infer_section,
    )
    SECTION_INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Section inference not available: {e}")
    SECTION_INFERENCE_AVAILABLE = False
    SectionInferencer = InferenceResult = infer_section = None

# Source Traceability
try:
    from .source_traceability import (
        SourceTracker, SourceTrace, SourceType, VerificationStatus,
        TraceableRequirement, SourceTrackerWithOffsets, create_trace,
    )
    SOURCE_TRACEABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Source traceability not available: {e}")
    SOURCE_TRACEABILITY_AVAILABLE = False
    SourceTracker = SourceTrace = SourceType = VerificationStatus = None
    TraceableRequirement = SourceTrackerWithOffsets = create_trace = None

# SQLite Persistence
try:
    from .persistence import (
        RFPDatabase, StoredProject, StoredRequirement, get_database,
    )
    PERSISTENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Persistence not available: {e}")
    PERSISTENCE_AVAILABLE = False
    RFPDatabase = StoredProject = StoredRequirement = get_database = None

# Add new exports to __all__
__all__.extend([
    # Accuracy Audit
    "AccuracyAuditor", "AccuracyReport", "AuditFinding", "AuditSeverity",
    "ConflictFinding", "ComplianceGate", "audit_extraction",
    "ACCURACY_AUDIT_AVAILABLE",
    # Section Inference
    "SectionInferencer", "InferenceResult", "infer_section",
    "SECTION_INFERENCE_AVAILABLE",
    # Source Traceability
    "SourceTracker", "SourceTrace", "SourceType", "VerificationStatus",
    "TraceableRequirement", "SourceTrackerWithOffsets", "create_trace",
    "SOURCE_TRACEABILITY_AVAILABLE",
    # Persistence
    "RFPDatabase", "StoredProject", "StoredRequirement", "get_database",
    "PERSISTENCE_AVAILABLE",
])
