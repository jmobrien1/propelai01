"""
PropelAI v3.1: Resilient Extraction Pipeline

This module orchestrates the entire extraction process using the
"Extract First, Classify Later" architecture.

v3.1 Changes (2025-12-15):
- Binding language required for extraction
- Document filtering for non-requirement files
- 100% recall on DOD NOC and SENTRA ground truth

Key principles:
1. Never lose requirements due to classification failures
2. Fail safe, not fail silent
3. Graduated confidence, not binary found/not-found
4. Validate and warn, don't silently drop
"""

PIPELINE_VERSION = "3.1.0"

import logging
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime

from .extraction_models import (
    RequirementCandidate,
    SectionCandidate,
    ExtractionResult,
    ExtractionQualityMetrics,
    ConfidenceLevel,
    DetectionMethod,
)
from .universal_extractor import UniversalExtractor
from .section_classifier import SectionClassifier
from .parser import MultiFormatParser

logger = logging.getLogger(__name__)


class ResilientExtractor:
    """
    The main extraction orchestrator that implements the resilient pipeline.

    Pipeline flow:
    1. Parse documents
    2. Detect document types (especially SOW)
    3. Extract ALL potential requirements (universal extraction)
    4. Classify requirements into sections (post-hoc classification)
    5. Validate quality and detect anomalies
    6. Return results with confidence scores and review queue
    """

    def __init__(self):
        self.parser = MultiFormatParser()
        self.universal_extractor = UniversalExtractor()
        self.section_classifier = SectionClassifier()

    def extract(
        self,
        file_paths: List[str],
        rfp_id: str = "RFP-UNKNOWN"
    ) -> ExtractionResult:
        """
        Extract requirements from a list of file paths.

        This is the main entry point for the resilient extraction pipeline.
        """
        logger.info(f"Starting resilient extraction for {rfp_id}")
        logger.info(f"Processing {len(file_paths)} files")

        result = ExtractionResult(rfp_id=rfp_id)

        # Step 1: Parse all documents
        documents = self._parse_documents(file_paths)
        result.quality_metrics.total_documents = len(documents)
        result.quality_metrics.total_pages = sum(
            len(doc.get('pages', [doc.get('text', '')])) for doc in documents
        )

        # Compute document hashes for reproducibility tracking
        result.document_hashes = [
            self._compute_doc_hash(doc) for doc in documents
        ]

        # Step 2: Detect SOW/PWS documents
        sow_info = self.section_classifier.detect_sow_documents(documents)
        result.quality_metrics.sow_detected = sow_info.sow_detected
        if sow_info.sow_documents:
            result.quality_metrics.sow_source = ', '.join(sow_info.sow_documents)

        # Step 3: Extract ALL potential requirements
        logger.info("Extracting all potential requirements...")
        candidates = self.universal_extractor.extract_all(documents)
        logger.info(f"Extracted {len(candidates)} candidates")

        # Step 4: Classify requirements into sections
        logger.info("Classifying requirements into sections...")
        classified = self.section_classifier.classify_requirements(candidates, documents)

        # Step 5: Add to result and build review queue
        for req in classified:
            result.add_requirement(req)

        # Step 6: Validate and compute quality metrics
        result.finalize()

        # Step 7: Run additional quality checks
        self._run_quality_checks(result, documents)

        # Log summary
        self._log_extraction_summary(result)

        return result

    def extract_from_parsed(
        self,
        documents: List[Dict],
        rfp_id: str = "RFP-UNKNOWN"
    ) -> ExtractionResult:
        """
        Extract from already-parsed documents.
        Useful when documents are already in memory.
        """
        logger.info(f"Starting resilient extraction from parsed docs for {rfp_id}")

        result = ExtractionResult(rfp_id=rfp_id)
        result.quality_metrics.total_documents = len(documents)
        result.quality_metrics.total_pages = sum(
            len(doc.get('pages', [doc.get('text', '')])) for doc in documents
        )

        # Document hashes
        result.document_hashes = [
            self._compute_doc_hash(doc) for doc in documents
        ]

        # Detect SOW
        sow_info = self.section_classifier.detect_sow_documents(documents)
        result.quality_metrics.sow_detected = sow_info.sow_detected
        if sow_info.sow_documents:
            result.quality_metrics.sow_source = ', '.join(sow_info.sow_documents)

        # Extract
        candidates = self.universal_extractor.extract_all(documents)

        # Classify
        classified = self.section_classifier.classify_requirements(candidates, documents)

        # Build result
        for req in classified:
            result.add_requirement(req)

        result.finalize()
        self._run_quality_checks(result, documents)
        self._log_extraction_summary(result)

        return result

    def _parse_documents(self, file_paths: List[str]) -> List[Dict]:
        """Parse all documents from file paths"""
        documents = []

        for path in file_paths:
            try:
                parsed = self.parser.parse_file(path)
                if parsed:
                    documents.append(parsed)
                else:
                    logger.warning(f"Failed to parse: {path}")
            except Exception as e:
                logger.error(f"Error parsing {path}: {e}")

        return documents

    def _compute_doc_hash(self, doc: Dict) -> str:
        """Compute hash of document for reproducibility tracking"""
        text = doc.get('text', '')
        filename = doc.get('filename', '')
        content = f"{filename}:{len(text)}:{text[:1000]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _run_quality_checks(self, result: ExtractionResult, documents: List[Dict]):
        """Run additional quality validation checks"""
        metrics = result.quality_metrics

        # Check 1: Ensure SOW requirements are present
        section_c_count = metrics.section_counts.get('C', 0)

        if not metrics.sow_detected and section_c_count < 10:
            # SOW might be present but not detected - scan all docs
            logger.warning("SOW not detected and low Section C count - running fallback scan")

            # Look for high-density requirement areas in all documents
            fallback_candidates = self._fallback_sow_scan(documents)

            if fallback_candidates:
                logger.info(f"Fallback scan found {len(fallback_candidates)} additional candidates")

                for req in fallback_candidates:
                    # Check if it's a duplicate
                    if not self._is_duplicate(req, result.requirements):
                        req.assigned_section = 'C'
                        req.category = 'TECHNICAL_REQUIREMENT'
                        req.confidence = ConfidenceLevel.LOW
                        req.needs_review = True
                        req.review_reasons.append(
                            "FALLBACK_SCAN: Found via SOW fallback scan"
                        )
                        result.add_requirement(req)

                # Recompute metrics
                result.finalize()

        # Check 2: Warn if extraction seems too sparse
        if metrics.total_pages > 10 and metrics.total_requirements < 20:
            metrics.add_anomaly(
                f"SPARSE_EXTRACTION: Only {metrics.total_requirements} requirements "
                f"from {metrics.total_pages} pages"
            )

        # Check 3: Warn if too many items need review
        review_ratio = len(result.review_queue) / max(len(result.requirements), 1)
        if review_ratio > 0.4:
            metrics.add_warning(
                f"HIGH_REVIEW_RATE: {review_ratio:.0%} of requirements flagged for review"
            )

    def _fallback_sow_scan(self, documents: List[Dict]) -> List[RequirementCandidate]:
        """
        Fallback scan when SOW detection fails.
        Looks for high-density requirement areas in all documents.
        """
        fallback_candidates = []
        fallback_extractor = UniversalExtractor()

        for doc in documents:
            text = doc.get('text', '')
            filename = doc.get('filename', '')

            # Check for contractor SHALL density
            contractor_shall_matches = list(re.finditer(
                r'contractor\s+shall',
                text.lower()
            ))

            if len(contractor_shall_matches) >= 3:
                logger.info(f"Fallback: Found {len(contractor_shall_matches)} 'contractor shall' in {filename}")

                # Extract from regions around these matches
                for match in contractor_shall_matches:
                    # Get surrounding paragraph
                    start = max(0, text.rfind('\n\n', 0, match.start()))
                    end = text.find('\n\n', match.end())
                    if end == -1:
                        end = min(len(text), match.end() + 500)

                    paragraph = text[start:end].strip()

                    if len(paragraph) >= 40:
                        candidate = RequirementCandidate(
                            id=f"FALLBACK-{len(fallback_candidates)+1:04d}",
                            text=paragraph,
                            text_hash=RequirementCandidate.compute_hash(paragraph),
                            source_document=filename,
                            source_page=1,  # Unknown
                            source_offset=start,
                            binding_level="SHALL",
                            binding_keyword="shall",
                            confidence=ConfidenceLevel.LOW,
                            confidence_score=0.4,
                        )
                        fallback_candidates.append(candidate)

        return fallback_candidates

    def _is_duplicate(
        self,
        candidate: RequirementCandidate,
        existing: List[RequirementCandidate]
    ) -> bool:
        """Check if candidate is duplicate of existing requirement"""
        for req in existing:
            if req.text_hash == candidate.text_hash:
                return True
            # Also check text similarity
            if self._text_similarity(req.text, candidate.text) > 0.85:
                return True
        return False

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _log_extraction_summary(self, result: ExtractionResult):
        """Log comprehensive extraction summary"""
        metrics = result.quality_metrics

        logger.info("=" * 60)
        logger.info(f"EXTRACTION SUMMARY: {result.rfp_id}")
        logger.info("=" * 60)
        logger.info(f"Documents: {metrics.total_documents}")
        logger.info(f"Pages: {metrics.total_pages}")
        logger.info(f"Requirements: {metrics.total_requirements}")
        logger.info(f"Review queue: {len(result.review_queue)}")
        logger.info(f"SOW detected: {metrics.sow_detected}")

        logger.info("\nBy Section:")
        for section, count in sorted(metrics.section_counts.items()):
            logger.info(f"  Section {section}: {count}")

        logger.info("\nBy Confidence:")
        logger.info(f"  HIGH: {metrics.high_confidence_count}")
        logger.info(f"  MEDIUM: {metrics.medium_confidence_count}")
        logger.info(f"  LOW: {metrics.low_confidence_count}")
        logger.info(f"  UNCERTAIN: {metrics.uncertain_count}")

        if metrics.anomalies:
            logger.warning("\nANOMALIES:")
            for anomaly in metrics.anomalies:
                logger.warning(f"  - {anomaly}")

        if metrics.warnings:
            logger.warning("\nWARNINGS:")
            for warning in metrics.warnings:
                logger.warning(f"  - {warning}")

        logger.info("=" * 60)


# Import re for fallback scan
import re


def create_resilient_extractor() -> ResilientExtractor:
    """Factory function to create configured extractor"""
    return ResilientExtractor()
