"""
PropelAI Cycle 5: Enhanced Compliance Agent Module

Multi-document, graph-based compliance extraction for federal RFPs.

Components:
- BundleDetector: Auto-classify RFP document bundles
- MultiFormatParser: Parse PDF/DOCX/XLSX with section detection
- RequirementExtractor: Multi-pattern extraction with semantic classification
- CrossReferenceResolver: Build requirements graph with cross-document edges
- EnhancedComplianceAgent: Main agent orchestrating all components
- ExcelExporter: Export compliance matrix to Excel
- AmendmentProcessor: Track requirement changes across amendments

Usage:
    from agents.enhanced_compliance import EnhancedComplianceAgent, export_to_excel
    
    agent = EnhancedComplianceAgent()
    result = agent.process_files(["/path/to/rfp.pdf", "/path/to/attachments.pdf"])
    
    print(f"Extracted {len(result.requirements_graph)} requirements")
    print(f"Coverage estimate: {result.extraction_coverage * 100:.1f}%")
    
    # Export to Excel
    export_to_excel(result, "compliance_matrix.xlsx", "75N96025R00004", "NIH NIEHS Contract")
    
    # Process amendments
    from agents.enhanced_compliance import AmendmentProcessor
    
    processor = AmendmentProcessor()
    processor.load_base_requirements(result.requirements_graph)
    amendment_result = processor.process_amendment("/path/to/amendment.pdf", amendment_number=2)
    print(processor.generate_change_report())
"""

from .models import (
    DocumentType,
    RequirementType,
    RequirementStatus,
    ConfidenceLevel,
    SourceLocation,
    RequirementNode,
    RFPBundle,
    ParsedDocument,
    ComplianceMatrixRow,
    ExtractionResult,
)

from .bundle_detector import BundleDetector
from .parser import MultiFormatParser
from .extractor import RequirementExtractor
from .resolver import CrossReferenceResolver
from .agent import EnhancedComplianceAgent, create_enhanced_compliance_agent
from .amendment_processor import (
    AmendmentProcessor, 
    AmendmentResult, 
    AmendmentType,
    ChangeType,
    QAPair,
    Modification,
    RequirementChange
)

from .outline_generator import (
    OutlineGenerator,
    OutlineResult,
    ProposalVolume,
    ProposalSection,
    EvaluationFactor,
    FormatRequirement,
    SubmissionRequirement,
    VolumeType,
    parse_rfp_outline,
    parse_rfp_outline_from_text
)

try:
    from .excel_export import ExcelExporter, export_to_excel
    from .excel_parser import ExcelMatrixParser, parse_excel_matrix, MatrixParseResult, ColumnType
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    ExcelExporter = None
    export_to_excel = None
    ExcelMatrixParser = None
    parse_excel_matrix = None
    MatrixParseResult = None
    ColumnType = None

__all__ = [
    # Models
    "DocumentType",
    "RequirementType", 
    "RequirementStatus",
    "ConfidenceLevel",
    "SourceLocation",
    "RequirementNode",
    "RFPBundle",
    "ParsedDocument",
    "ComplianceMatrixRow",
    "ExtractionResult",
    
    # Components
    "BundleDetector",
    "MultiFormatParser",
    "RequirementExtractor",
    "CrossReferenceResolver",
    
    # Main Agent
    "EnhancedComplianceAgent",
    "create_enhanced_compliance_agent",
    
    # Excel Export
    "ExcelExporter",
    "export_to_excel",
    "EXCEL_AVAILABLE",
    
    # Excel Parser
    "ExcelMatrixParser",
    "parse_excel_matrix",
    "MatrixParseResult",
    "ColumnType",
    
    # Amendment Processing
    "AmendmentProcessor",
    "AmendmentResult",
    "AmendmentType",
    "ChangeType",
    "QAPair",
    "Modification",
    "RequirementChange",
    
    # Outline Generator
    "OutlineGenerator",
    "OutlineResult",
    "ProposalVolume",
    "ProposalSection",
    "EvaluationFactor",
    "FormatRequirement",
    "SubmissionRequirement",
    "VolumeType",
    "parse_rfp_outline",
    "parse_rfp_outline_from_text",
]

__version__ = "2.4.0"  # Cycle 5 + quality tuning + Excel + Amendment + Outline Generator
