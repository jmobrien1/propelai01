"""
PropelAI Cycle 5: Enhanced Compliance Agent Module

Multi-document, graph-based compliance extraction for federal RFPs.

Components:
- BundleDetector: Auto-classify RFP document bundles
- MultiFormatParser: Parse PDF/DOCX/XLSX with section detection
- RequirementExtractor: Multi-pattern extraction with semantic classification
- CrossReferenceResolver: Build requirements graph with cross-document edges
- EnhancedComplianceAgent: Main agent orchestrating all components

Usage:
    from agents.enhanced_compliance import EnhancedComplianceAgent
    
    agent = EnhancedComplianceAgent()
    result = agent.process_files(["/path/to/rfp.pdf", "/path/to/attachments.pdf"])
    
    print(f"Extracted {len(result.requirements_graph)} requirements")
    print(f"Coverage estimate: {result.extraction_coverage * 100:.1f}%")
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
]

__version__ = "2.0.0"  # Cycle 5
