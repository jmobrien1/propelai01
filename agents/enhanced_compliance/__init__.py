"""
PropelAI Enhanced Compliance Module v3.0
"""

__version__ = "3.0.0"
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
        CompanyLibrary, CompanyLibraryParser, DocumentType, ParsedDocument,
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
        SectionAwareExtractor, StructuredRequirement, StructuredExtractionResult,
        RequirementCategory, BindingLevel, extract_requirements_structured,
        ExtractionResult,
    )
    SECTION_EXTRACTOR_AVAILABLE = True
except ImportError:
    SECTION_EXTRACTOR_AVAILABLE = False
    SectionAwareExtractor = StructuredRequirement = StructuredExtractionResult = None
    RequirementCategory = BindingLevel = extract_requirements_structured = None
    ExtractionResult = None

# Best Practices CTM (depends on section_aware_extractor)
try:
    from .best_practices_ctm import (
        BestPracticesCTMGenerator, BestPracticesCTMExporter, export_ctm_best_practices,
    )
    BEST_PRACTICES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Best Practices CTM not available: {e}")
    BEST_PRACTICES_AVAILABLE = False
    BestPracticesCTMGenerator = BestPracticesCTMExporter = export_ctm_best_practices = None

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
    "SectionAwareExtractor", "StructuredRequirement", "StructuredExtractionResult",
    "RequirementCategory", "BindingLevel", "extract_requirements_structured",
    "ExtractionResult", "SECTION_EXTRACTOR_AVAILABLE",
    # Best Practices CTM
    "BestPracticesCTMGenerator", "BestPracticesCTMExporter", "export_ctm_best_practices",
    "BEST_PRACTICES_AVAILABLE",
    # Availability flags
    "OUTLINE_GENERATOR_AVAILABLE", "COMPANY_LIBRARY_AVAILABLE",
]
