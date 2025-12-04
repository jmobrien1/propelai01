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
)

from .jp1_parser import JP1MatrixParser

from .evidence_hunter import EvidenceHunter, EvidenceMatch

from .optimizer import ProjectOptimizer, OptimizationResult

from .pdf_tagger import PDFTagger, TaggedPDF

from .form_generator import JP3FormGenerator

__all__ = [
    # Models
    "OASISDomain",
    "ScoringCriteria",
    "Project",
    "ProjectClaim",
    "DocumentChunk",
    "VerificationStatus",
    "DomainType",
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
]
