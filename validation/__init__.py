"""
PropelAI Validation Framework

Phase 1: Foundation - Establish Trust Through Provable Accuracy

This module provides:
- Ground truth schema for annotated RFP requirements
- Accuracy metrics (precision, recall, F1, section accuracy, binding accuracy)
- Automated testing pipeline for CI/CD
- Annotation tooling for creating ground truth datasets

Usage:
    from validation.schemas import GroundTruthRFP, GroundTruthRequirement
    from validation.metrics import calculate_extraction_metrics
    from validation.matching import match_requirements
"""

from .schemas import (
    GroundTruthRequirement,
    GroundTruthRFP,
    AnnotationStatus,
    RequirementBoundary,
)

__version__ = "1.0.0"
