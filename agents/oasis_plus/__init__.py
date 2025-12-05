"""
PropelAI OASIS+ Module
======================

Specialized agents and utilities for GSA OASIS+ self-scoring proposals.

This module transforms PropelAI from an "RFP Shredder" into a "Scorecard Orchestrator"
for evidence verification and score optimization.

Components:
- models: Database schema for J.P-1 scoring hierarchy
- jp1_parser: Dynamic parsing of J.P-1 Qualifications Matrix
- evidence_hunter: RAG-based evidence verification agent
- optimizer: Combinatorial optimizer for project selection
- pdf_tagger: Symphony-ready PDF annotation
- form_generator: J.P-3 verification form automation

Version: 1.0.0
"""

__version__ = "1.0.0"

from .models import (
    OASISDomain,
    ScoringCriteria,
    Project,
    ProjectClaim,
    DocumentChunk,
    VerificationStatus,
    DomainType,
    BusinessSize,
    ContractType,
    OptimizationConstraints,
)

from .jp1_parser import JP1MatrixParser

from .evidence_hunter import EvidenceHunter, EvidenceMatch

from .optimizer import ProjectOptimizer, OptimizationResult

from .pdf_tagger import PDFTagger, TaggedPDF

from .form_generator import JP3FormGenerator

from .orchestrator import (
    OASISOrchestrator,
    OASISProposal,
    OASISProject,
    ProcessingProgress,
)

from .symphony_export import (
    SymphonyBundleGenerator,
    BundleResult,
    BundleManifest,
    generate_symphony_bundle,
)

from .database import (
    OASISDatabase,
    get_database,
)

from .document_ingestion import (
    DocumentIngestionPipeline,
    IngestionResult,
    DocumentContent,
    PageContent,
    ingest_document,
    check_ocr_availability,
)

__all__ = [
    # Models
    "OASISDomain",
    "ScoringCriteria",
    "Project",
    "ProjectClaim",
    "DocumentChunk",
    "VerificationStatus",
    "DomainType",
    "BusinessSize",
    "ContractType",
    "OptimizationConstraints",
    # Parser
    "JP1MatrixParser",
    # Evidence Hunter
    "EvidenceHunter",
    "EvidenceMatch",
    # Optimizer
    "ProjectOptimizer",
    "OptimizationResult",
    # PDF Tagger
    "PDFTagger",
    "TaggedPDF",
    # Form Generator
    "JP3FormGenerator",
    # Orchestrator
    "OASISOrchestrator",
    "OASISProposal",
    "OASISProject",
    "ProcessingProgress",
    # Symphony Export
    "SymphonyBundleGenerator",
    "BundleResult",
    "BundleManifest",
    "generate_symphony_bundle",
    # Database
    "OASISDatabase",
    "get_database",
    # Document Ingestion
    "DocumentIngestionPipeline",
    "IngestionResult",
    "DocumentContent",
    "PageContent",
    "ingest_document",
    "check_ocr_availability",
]
