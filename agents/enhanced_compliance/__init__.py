"""
PropelAI Enhanced Compliance Module v3.0

This module provides enhanced compliance matrix generation with support for:
- Multi-format RFP detection (NIH Factor, GSA BPA, State RFP, DOD UCF)
- Enhanced requirement metadata (scoring type, response format, points)
- Smart proposal outline generation
- Company library integration

Version: 3.0.0
Date: November 28, 2025
"""

__version__ = "3.0.0"
__author__ = "PropelAI Team"

# =============================================================================
# Core Data Models (v3.0)
# =============================================================================
from .ctm_data_models import (
    ScoringType,
    ResponseFormat,
    RequirementType,
    RFPSection,
    ComplianceStatus,
    PageLimit,
    FormattingRequirement,
    EvidenceRequirement,
    KeyPersonnelRequirement,
    EnhancedRequirement,
    ComplianceMatrix,
    create_pass_fail_requirement,
    create_weighted_requirement,
    create_future_diligence_requirement,
)

# =============================================================================
# Enhanced Extractor (v3.0)
# =============================================================================
from .ctm_extractor import (
    EnhancedCTMExtractor,
    ScoringPatterns,
    ResponseFormatPatterns,
    PageLimitPatterns,
    FormattingPatterns,
    FutureDiligencePatterns,
    KeyPersonnelPatterns,
    ConstraintPatterns,
    process_requirements_batch,
)

# =============================================================================
# Integration Module (v3.0)
# =============================================================================
from .ctm_integration import (
    LegacyRequirementAdapter,
    CTMEnricher,
    format_ctm_for_api,
    format_requirement_for_outline,
    get_content_allocation_guidance,
)

# =============================================================================
# Optional: Existing modules (with graceful fallback)
# =============================================================================

try:
    from .smart_outline_generator import (
        SmartOutlineGenerator, ProposalOutline, ProposalVolume, ProposalSection,
    )
    OUTLINE_GENERATOR_AVAILABLE = True
except ImportError:
    OUTLINE_GENERATOR_AVAILABLE = False
    SmartOutlineGenerator = ProposalOutline = ProposalVolume = ProposalSection = None

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

try:
    from .best_practices_ctm import (
        BestPracticesCTMGenerator, BestPracticesCTMExporter, export_ctm_best_practices,
    )
    BEST_PRACTICES_AVAILABLE = True
except ImportError:
    BEST_PRACTICES_AVAILABLE = False
    BestPracticesCTMGenerator = BestPracticesCTMExporter = export_ctm_best_practices = None

try:
    from .section_aware_extractor import (
        SectionAwareExtractor, StructuredRequirement, StructuredExtractionResult,
        RequirementCategory, BindingLevel, extract_requirements_structured,
    )
    SECTION_EXTRACTOR_AVAILABLE = True
except ImportError:
    SECTION_EXTRACTOR_AVAILABLE = False
    SectionAwareExtractor = StructuredRequirement = StructuredExtractionResult = None
    RequirementCategory = BindingLevel = extract_requirements_structured = None

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

__all__ = [
    "__version__", "ScoringType", "ResponseFormat", "RequirementType", "RFPSection",
    "ComplianceStatus", "PageLimit", "FormattingRequirement", "EvidenceRequirement",
    "KeyPersonnelRequirement", "EnhancedRequirement", "ComplianceMatrix",
    "create_pass_fail_requirement", "create_weighted_requirement", "create_future_diligence_requirement",
    "EnhancedCTMExtractor", "process_requirements_batch",
    "ScoringPatterns", "ResponseFormatPatterns", "PageLimitPatterns", "FormattingPatterns",
    "FutureDiligencePatterns", "KeyPersonnelPatterns", "ConstraintPatterns",
    "LegacyRequirementAdapter", "CTMEnricher", "format_ctm_for_api",
    "format_requirement_for_outline", "get_content_allocation_guidance",
    "SmartOutlineGenerator", "ProposalOutline", "ProposalVolume", "ProposalSection",
    "CompanyLibrary", "CompanyLibraryParser", "DocumentType", "ParsedDocument",
    "CompanyProfile", "Capability", "PastPerformance", "KeyPersonnel", "Differentiator",
    "BestPracticesCTMGenerator", "BestPracticesCTMExporter", "export_ctm_best_practices",
    "SectionAwareExtractor", "StructuredRequirement", "StructuredExtractionResult",
    "RequirementCategory", "BindingLevel", "extract_requirements_structured",
    "RFPStructureParser", "DocumentStructure", "UCFSection", "SectionBoundary",
    "SubsectionBoundary", "AttachmentInfo", "analyze_rfp_structure",
    "OUTLINE_GENERATOR_AVAILABLE", "COMPANY_LIBRARY_AVAILABLE", "BEST_PRACTICES_AVAILABLE",
    "SECTION_EXTRACTOR_AVAILABLE", "DOCUMENT_STRUCTURE_AVAILABLE",
]
