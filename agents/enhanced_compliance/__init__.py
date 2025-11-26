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

Usage:
    from agents.enhanced_compliance import EnhancedComplianceAgent, export_to_excel
    
    agent = EnhancedComplianceAgent()
    result = agent.process_files(["/path/to/rfp.pdf", "/path/to/attachments.pdf"])
    
    print(f"Extracted {len(result.requirements_graph)} requirements")
    print(f"Coverage estimate: {result.extraction_coverage * 100:.1f}%")
    
    # Export to Excel
    export_to_excel(result, "compliance_matrix.xlsx", "75N96025R00004", "NIH NIEHS Contract")
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

try:
    from .excel_export import ExcelExporter, export_to_excel
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    ExcelExporter = None
    export_to_excel = None

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
]

__version__ = "2.1.0"  # Cycle 5 + quality tuning + Excel export
