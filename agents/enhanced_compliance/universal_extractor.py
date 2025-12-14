"""
PropelAI v3.0: Universal Requirement Extractor

This module implements the "Extract First, Classify Later" pattern.
It extracts ALL potential requirements from ALL documents without
making classification decisions that could cause silent failures.

Key principle: Never lose a requirement due to classification failure.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

from .extraction_models import (
    RequirementCandidate,
    ConfidenceLevel,
    DetectionMethod,
    ExtractionQualityMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ParagraphContext:
    """Context about an extracted paragraph"""
    text: str
    start_offset: int
    end_offset: int
    page_number: int
    source_document: str
    preceding_header: Optional[str] = None
    section_hint: Optional[str] = None


class UniversalExtractor:
    """
    Extracts ALL potential requirements from documents.

    This extractor is intentionally permissive - it's better to extract
    something that isn't a requirement (and classify it later) than to
    miss a real requirement due to overly strict filtering.
    """

    # Binding language patterns
    BINDING_PATTERNS = {
        "SHALL": [
            r'\bshall\b',
            r'\bmust\b',
            r'\bis\s+required\s+to\b',
            r'\bare\s+required\s+to\b',
            r'\bwill\s+be\s+required\b',
            r'\bmandatory\b',
        ],
        "SHOULD": [
            r'\bshould\b',
            r'\bis\s+expected\s+to\b',
            r'\bare\s+expected\s+to\b',
            r'\brecommended\b',
        ],
        "MAY": [
            r'\bmay\b',
            r'\bcan\b',
            r'\boptional\b',
            r'\bat\s+(?:its|their)\s+discretion\b',
        ],
    }

    # RFP reference patterns (to extract the RFP's own numbering)
    RFP_REFERENCE_PATTERNS = [
        r'^([A-M])\.(\d+)(?:\.(\d+|[A-Za-z]))?(?:\.(\d+|[a-z]))?',  # L.4.B.2
        r'^(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?',                    # 1.2.3.4
        r'^ARTICLE\s+([A-M])\.(\d+)',                                # ARTICLE L.4
        r'^\(([a-z])\)',                                             # (a), (b), (c)
        r'^(\d+)\)',                                                 # 1), 2), 3)
    ]

    # Section indicator patterns
    SECTION_INDICATORS = {
        "C": [
            r'statement\s+of\s+work',
            r'performance\s+work\s+statement',
            r'scope\s+of\s+work',
            r'technical\s+requirements?',
            r'task\s+\d+',
            r'contractor\s+shall',
        ],
        "L": [
            r'proposal\s+(?:preparation|format|instructions?)',
            r'offeror\s+shall\s+(?:submit|provide|include)',
            r'volume\s+[ivx\d]+',
            r'page\s+limit',
            r'font\s+size',
            r'submission\s+requirements?',
        ],
        "M": [
            r'evaluation\s+(?:factor|criteria)',
            r'(?:will|shall)\s+be\s+evaluated',
            r'basis\s+for\s+award',
            r'rating\s+(?:scale|criteria)',
            r'(?:acceptable|unacceptable|outstanding)',
            r'technical\s+acceptability',
        ],
    }

    def __init__(self, min_length: int = 40, max_length: int = 3000):
        self.min_length = min_length
        self.max_length = max_length
        self.seen_hashes: Set[str] = set()
        self.req_counter = 0

    def extract_all(self, documents: List[Dict]) -> List[RequirementCandidate]:
        """
        Extract ALL potential requirements from ALL documents.

        Args:
            documents: List of parsed documents with 'text', 'filename', 'pages'

        Returns:
            List of RequirementCandidate objects (unclassified)
        """
        self.seen_hashes = set()
        self.req_counter = 0
        all_candidates = []

        for doc in documents:
            filename = doc.get('filename', 'unknown')
            text = doc.get('text', '')
            pages = doc.get('pages', [])

            # Use pages if they have real content, otherwise fall back to full text
            # Check if pages are placeholders (very short strings) or real content
            if pages and all(len(p) > 50 for p in pages if p):
                # Pages have real content
                content_pages = pages
            elif text:
                # Fall back to full text as single page
                content_pages = [text]
            else:
                logger.warning(f"No content found in {filename}")
                continue

            logger.info(f"Extracting from {filename} ({len(content_pages)} pages)")

            # Extract from each page
            page_offset = 0
            for page_num, page_text in enumerate(content_pages, 1):
                if not page_text:
                    continue
                candidates = self._extract_from_page(
                    page_text,
                    page_num,
                    filename,
                    page_offset
                )
                all_candidates.extend(candidates)
                page_offset += len(page_text) + 2  # +2 for page separator

        logger.info(f"Extracted {len(all_candidates)} candidates from {len(documents)} documents")
        return all_candidates

    def _extract_from_page(
        self,
        text: str,
        page_number: int,
        source_document: str,
        offset: int
    ) -> List[RequirementCandidate]:
        """Extract candidates from a single page"""
        candidates = []

        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)

        for para in paragraphs:
            para_text = para['text'].strip()

            # Skip if too short or too long
            if len(para_text) < self.min_length:
                continue
            if len(para_text) > self.max_length:
                # For very long paragraphs, try to extract subsections
                sub_candidates = self._extract_from_long_paragraph(
                    para_text, page_number, source_document, offset + para['start']
                )
                candidates.extend(sub_candidates)
                continue

            # Skip if it's clearly not a requirement
            if self._is_clearly_not_requirement(para_text):
                continue

            # Create candidate
            candidate = self._create_candidate(
                text=para_text,
                page_number=page_number,
                source_document=source_document,
                offset=offset + para['start']
            )

            if candidate:
                candidates.append(candidate)

        return candidates

    def _split_into_paragraphs(self, text: str) -> List[Dict]:
        """Split text into paragraphs with position info"""
        paragraphs = []

        # Split on double newlines or significant whitespace
        pattern = r'\n\s*\n|\n(?=\s*(?:\d+\.|[a-z]\)|[A-Z]\.|\([a-z]\)|\(\d+\)))'
        parts = re.split(pattern, text)

        current_pos = 0
        for part in parts:
            if part:
                start = text.find(part, current_pos)
                if start >= 0:
                    paragraphs.append({
                        'text': part,
                        'start': start,
                        'end': start + len(part)
                    })
                    current_pos = start + len(part)

        return paragraphs

    def _extract_from_long_paragraph(
        self,
        text: str,
        page_number: int,
        source_document: str,
        offset: int
    ) -> List[RequirementCandidate]:
        """Handle very long paragraphs by extracting sub-items"""
        candidates = []

        # Try to split on numbered items
        numbered_pattern = r'(?:^|\n)\s*(?:\d+\.|[a-z]\)|[A-Z]\.|\([a-z]\)|\(\d+\))'
        parts = re.split(numbered_pattern, text)

        for part in parts:
            part = part.strip()
            if len(part) >= self.min_length and len(part) <= self.max_length:
                candidate = self._create_candidate(
                    text=part,
                    page_number=page_number,
                    source_document=source_document,
                    offset=offset
                )
                if candidate:
                    candidates.append(candidate)

        # If no sub-items found, truncate and flag for review
        if not candidates and len(text) >= self.min_length:
            truncated = text[:self.max_length] + "..."
            candidate = self._create_candidate(
                text=truncated,
                page_number=page_number,
                source_document=source_document,
                offset=offset
            )
            if candidate:
                candidate.needs_review = True
                candidate.review_reasons.append("TRUNCATED: Original text exceeded max length")
                candidates.append(candidate)

        return candidates

    def _is_clearly_not_requirement(self, text: str) -> bool:
        """Check if text is clearly not a requirement (headers, TOC, etc.)"""
        text_lower = text.lower()

        # Table of contents entries
        if '....' in text or re.search(r'\.\s*\d+\s*$', text):
            return True

        # Very short lines that are likely headers
        if len(text) < 60 and text.isupper():
            return True

        # Page numbers
        if re.match(r'^\s*(?:Page\s+)?\d+\s*(?:of\s+\d+)?\s*$', text, re.IGNORECASE):
            return True

        # Copyright/footer text
        if 'copyright' in text_lower or 'all rights reserved' in text_lower:
            return True

        # Blank or whitespace only
        if not text.strip():
            return True

        return False

    def _create_candidate(
        self,
        text: str,
        page_number: int,
        source_document: str,
        offset: int
    ) -> Optional[RequirementCandidate]:
        """Create a requirement candidate with binding analysis"""

        # Compute hash for deduplication
        text_hash = RequirementCandidate.compute_hash(text)
        if text_hash in self.seen_hashes:
            return None
        self.seen_hashes.add(text_hash)

        # Detect binding level
        binding_level, binding_keyword = self._detect_binding(text)

        # Extract RFP reference if present
        rfp_reference = self._extract_rfp_reference(text)

        # Determine initial confidence
        confidence, confidence_score = self._compute_initial_confidence(
            text, binding_level, rfp_reference
        )

        # Generate ID
        self.req_counter += 1
        req_id = f"REQ-{self.req_counter:05d}"

        candidate = RequirementCandidate(
            id=req_id,
            text=text,
            text_hash=text_hash,
            source_document=source_document,
            source_page=page_number,
            source_offset=offset,
            rfp_reference=rfp_reference,
            binding_level=binding_level,
            binding_keyword=binding_keyword,
            confidence=confidence,
            confidence_score=confidence_score,
        )

        # Flag uncertain items for review
        if confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]:
            candidate.needs_review = True
            candidate.review_reasons.append(
                f"LOW_CONFIDENCE: Score {confidence_score:.2f}"
            )

        return candidate

    def _detect_binding(self, text: str) -> Tuple[str, Optional[str]]:
        """Detect binding level and keyword"""
        text_lower = text.lower()

        # Check SHALL patterns first (most binding)
        for pattern in self.BINDING_PATTERNS["SHALL"]:
            match = re.search(pattern, text_lower)
            if match:
                return "SHALL", match.group()

        # Check SHOULD patterns
        for pattern in self.BINDING_PATTERNS["SHOULD"]:
            match = re.search(pattern, text_lower)
            if match:
                return "SHOULD", match.group()

        # Check MAY patterns
        for pattern in self.BINDING_PATTERNS["MAY"]:
            match = re.search(pattern, text_lower)
            if match:
                return "MAY", match.group()

        return "INFORMATIONAL", None

    def _extract_rfp_reference(self, text: str) -> Optional[str]:
        """Extract the RFP's own reference number from the text"""
        # Check first 50 characters for reference patterns
        prefix = text[:50]

        for pattern in self.RFP_REFERENCE_PATTERNS:
            match = re.match(pattern, prefix.strip())
            if match:
                # Build reference from captured groups
                groups = [g for g in match.groups() if g]
                return '.'.join(groups)

        return None

    def _compute_initial_confidence(
        self,
        text: str,
        binding_level: str,
        rfp_reference: Optional[str]
    ) -> Tuple[ConfidenceLevel, float]:
        """Compute initial confidence before classification"""
        score = 0.5  # Start at neutral

        # Binding language is strong signal
        if binding_level == "SHALL":
            score += 0.3
        elif binding_level == "SHOULD":
            score += 0.2
        elif binding_level == "MAY":
            score += 0.1

        # RFP reference is strong signal
        if rfp_reference:
            score += 0.15

        # Length - very short or very long is suspicious
        if len(text) < 80:
            score -= 0.1
        elif len(text) > 500:
            score -= 0.05

        # Specific requirement indicators
        text_lower = text.lower()
        if re.search(r'contractor|offeror|vendor|government', text_lower):
            score += 0.1
        if re.search(r'submit|provide|deliver|perform|ensure', text_lower):
            score += 0.1

        # Cap score at 0-1
        score = max(0.0, min(1.0, score))

        # Convert to confidence level
        if score >= 0.8:
            return ConfidenceLevel.HIGH, score
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM, score
        elif score >= 0.2:
            return ConfidenceLevel.LOW, score
        else:
            return ConfidenceLevel.UNCERTAIN, score

    def get_section_hints(self, text: str) -> Dict[str, float]:
        """
        Get hints about which section this text might belong to.
        Returns dict of section -> confidence score.
        """
        hints = {}
        text_lower = text.lower()

        for section, patterns in self.SECTION_INDICATORS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 0.2

            if score > 0:
                hints[section] = min(score, 1.0)

        return hints
