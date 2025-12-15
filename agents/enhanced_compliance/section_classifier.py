"""
PropelAI v3.0: Section Classifier

This module classifies extracted requirements into UCF sections (A-M).
Classification happens AFTER extraction to ensure no requirements are
lost due to classification failures.

Key principle: When uncertain, assign with low confidence rather than drop.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

from .extraction_models import (
    RequirementCandidate,
    SectionCandidate,
    ConfidenceLevel,
    DetectionMethod,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentSOWInfo:
    """Information about SOW/PWS detection in documents"""
    sow_detected: bool = False
    sow_documents: List[str] = field(default_factory=list)
    sow_detection_method: Optional[DetectionMethod] = None
    sow_confidence: float = 0.0


class SectionClassifier:
    """
    Classifies requirements into UCF sections.

    Uses a multi-layer approach:
    1. Document-level detection (is this document a SOW?)
    2. Context detection (what section header precedes this?)
    3. Content analysis (what does the text indicate?)
    4. Fallback assignment (assign with low confidence rather than drop)
    """

    # Section C indicators (SOW/PWS/Technical)
    SECTION_C_PATTERNS = [
        (r'contractor\s+shall', 0.8),
        (r'the\s+contractor\s+(?:will|must)', 0.8),
        (r'statement\s+of\s+work', 0.9),
        (r'performance\s+work\s+statement', 0.9),
        (r'scope\s+of\s+work', 0.85),
        (r'technical\s+requirement', 0.7),
        (r'task\s+\d+[.:]\s', 0.75),
        (r'deliverable', 0.6),
        (r'period\s+of\s+performance', 0.6),
        (r'work\s+breakdown\s+structure', 0.7),
        (r'quality\s+assurance', 0.6),
        (r'personnel\s+qualifications?', 0.65),
    ]

    # Section L indicators (Proposal Instructions)
    SECTION_L_PATTERNS = [
        (r'offeror\s+shall\s+(?:submit|provide|include)', 0.85),
        (r'proposal\s+(?:shall|must|should)', 0.8),
        (r'volume\s+[ivx\d]+', 0.75),
        (r'page\s+limit', 0.9),  # v3.2: Strong Section L indicator
        (r'font\s+(?:size|type)', 0.75),
        (r'submission\s+(?:requirements?|instructions?)', 0.85),
        (r'format\s+(?:requirements?|instructions?)', 0.75),
        (r'proposal\s+preparation', 0.8),
        (r'technical\s+(?:volume|proposal)', 0.7),
        (r'(?:past\s+performance|management)\s+(?:volume|proposal)', 0.7),
        (r'cost/price\s+(?:volume|proposal)', 0.7),
        (r'oral\s+presentation', 0.75),
        # v3.2: Additional Section L patterns
        (r'electronic\s+submission', 0.85),
        (r'instructions?:\s*(?:electronic|use|submit)', 0.85),
        (r'^\s*a\s+page\s+is\s+', 0.85),  # Page definition instructions
        (r'8\.5\s*x\s*11', 0.8),  # Paper size instructions
        (r'margin(?:s)?\s+(?:shall|should|must)', 0.8),
        (r'proposal\s+shall\s+not\s+exceed', 0.85),
        (r'pages?\s+shall\s+be\s+numbered', 0.8),
        (r'use\s+(?:the\s+)?(?:secure|safe)\s+file\s+transfer', 0.85),
        (r'all\s+electronic\s+files?\s+shall', 0.8),
        (r'offeror(?:s)?(?:\s+are)?\s+(?:cautioned|advised|encouraged)', 0.8),
        (r'without\s+conducting\s+(?:discussions|interchanges)', 0.8),  # Award intent
        (r'intends?\s+to\s+(?:make\s+)?award', 0.75),
    ]

    # Section M indicators (Evaluation)
    SECTION_M_PATTERNS = [
        (r'evaluation\s+(?:factor|criteria)', 0.9),
        (r'(?:will|shall)\s+be\s+evaluated', 0.85),
        (r'basis\s+for\s+award', 0.9),
        (r'rating\s+(?:scale|criteria)', 0.8),
        (r'(?:outstanding|good|acceptable|unacceptable|marginal)', 0.6),
        (r'technical\s+acceptability', 0.75),
        (r'best\s+value', 0.8),
        (r'trade.?off', 0.75),
        (r'lowest\s+price\s+technically\s+acceptable', 0.85),
        (r'evaluation\s+(?:approach|methodology)', 0.8),
        (r'strengths?\s+and\s+weaknesses?', 0.7),
        (r'significant\s+(?:strength|weakness)', 0.75),
    ]

    # Section K indicators (Reps & Certs)
    SECTION_K_PATTERNS = [
        (r'representation', 0.7),
        (r'certification', 0.7),
        (r'small\s+business', 0.65),
        (r'(?:woman|veteran|minority).?owned', 0.7),
        (r'debarment', 0.75),
        (r'sam\.gov', 0.8),
        (r'taxpayer\s+identification', 0.7),
    ]

    # SOW filename patterns (with typo tolerance)
    SOW_FILENAME_PATTERNS = [
        'sow',
        'statement of work',
        'stament of work',      # Common typo
        'statment of work',     # Common typo
        'statement work',
        's.o.w',
        'scope of work',
        'pws',
        'performance work statement',
        'performace work',      # Common typo
        'p.w.s',
    ]

    # SOW content header patterns
    SOW_CONTENT_PATTERNS = [
        r'STATEMENT\s+OF\s+WORK',
        r'PERFORMANCE\s+WORK\s+STATEMENT',
        r'S\.?O\.?W\.?\s*\n',
        r'P\.?W\.?S\.?\s*\n',
        r'SCOPE\s+OF\s+WORK',
        r'\bSOW\b.*\b(?:SECTION|ARTICLE|PART)\s+\d',
    ]

    def __init__(self):
        self.sow_info = DocumentSOWInfo()
        self.section_content_map: Dict[str, str] = {}  # section -> combined text

    def detect_sow_documents(self, documents: List[Dict]) -> DocumentSOWInfo:
        """
        Detect which documents contain SOW/PWS content.
        Uses multi-layer detection for robustness.
        """
        self.sow_info = DocumentSOWInfo()

        for doc in documents:
            filename = doc.get('filename', '').lower()
            text = doc.get('text', '')[:20000]  # Check first 20K chars

            is_sow, method, confidence = self._check_document_is_sow(filename, text)

            if is_sow:
                self.sow_info.sow_detected = True
                self.sow_info.sow_documents.append(doc.get('filename', 'unknown'))

                # Keep highest confidence detection
                if confidence > self.sow_info.sow_confidence:
                    self.sow_info.sow_confidence = confidence
                    self.sow_info.sow_detection_method = method

        if self.sow_info.sow_detected:
            logger.info(
                f"SOW detected in {len(self.sow_info.sow_documents)} documents "
                f"via {self.sow_info.sow_detection_method.value} "
                f"(confidence: {self.sow_info.sow_confidence:.2f})"
            )
        else:
            logger.warning("No SOW/PWS document explicitly detected")

        return self.sow_info

    def _check_document_is_sow(
        self,
        filename: str,
        text: str
    ) -> Tuple[bool, Optional[DetectionMethod], float]:
        """Check if a document is SOW/PWS using multiple methods"""
        filename_lower = filename.lower().replace('_', ' ').replace('-', ' ')
        text_upper = text.upper()

        # First, check for explicitly NON-SOW documents by filename
        non_sow_indicators = [
            'budget', 'pricing', 'cost', 'template', 'form',
            'checklist', 'amendment', 'qa ', 'q&a', 'resume',
            'personnel', 'experience', 'past performance',
            'dd254', 'dd 254', 'dd1423', 'security classification',
            'oasis', 'gsa', 'resolution matrix', 'organizational conflict',
            'surveillance', 'comments matrix', 'rac ', 'rac_',
            'labor categor', 'labor rate', 'rate card',
            'ordering instruction', 'placeholder', 'model contract',
            'representations', 'certifications', 'cdrl',
            'mapping', 'estimate', 'proposed labor',
        ]
        for indicator in non_sow_indicators:
            if indicator in filename_lower:
                return False, None, 0.0

        # Layer 1: Filename patterns (highest confidence)
        for pattern in self.SOW_FILENAME_PATTERNS:
            if pattern in filename_lower:
                return True, DetectionMethod.SOW_ATTACHMENT, 0.95

        # Layer 2: Content header patterns - require contractor language too
        contractor_shall_count = len(re.findall(r'contractor\s+shall', text.lower()))

        sow_header_found = False
        for pattern in self.SOW_CONTENT_PATTERNS:
            if re.search(pattern, text_upper):
                sow_header_found = True
                break

        # Only classify as SOW if there's a header AND meaningful contractor language
        if sow_header_found and contractor_shall_count >= 3:
            return True, DetectionMethod.CONTENT_HEADER, 0.85

        # Layer 3: High density of contractor SHALL statements (high threshold)
        if contractor_shall_count >= 10:
            return True, DetectionMethod.KEYWORD_DENSITY, 0.7

        # Layer 4: SOW-specific section numbering with contractor language
        if re.search(r'(?:^|\n)\s*\d+\.\d+\s+(?:SCOPE|BACKGROUND|REQUIREMENTS)', text_upper):
            if contractor_shall_count >= 3:
                return True, DetectionMethod.STRUCTURAL_PATTERN, 0.65

        return False, None, 0.0

    def classify_requirements(
        self,
        requirements: List[RequirementCandidate],
        documents: List[Dict]
    ) -> List[RequirementCandidate]:
        """
        Classify all requirements into UCF sections.

        This NEVER drops requirements - uncertain items get assigned
        with low confidence and flagged for review.
        """
        # First, detect SOW documents
        self.detect_sow_documents(documents)

        # Create document -> section mapping
        doc_section_map = self._build_document_section_map(documents)

        # Classify each requirement
        for req in requirements:
            section, category, confidence, reasons = self._classify_requirement(
                req, doc_section_map
            )

            req.assigned_section = section
            req.category = category
            req.classification_reasons = reasons

            # Adjust confidence based on classification confidence
            classification_confidence = confidence
            combined_score = (req.confidence_score + classification_confidence) / 2

            if combined_score >= 0.8:
                req.confidence = ConfidenceLevel.HIGH
            elif combined_score >= 0.5:
                req.confidence = ConfidenceLevel.MEDIUM
            elif combined_score >= 0.2:
                req.confidence = ConfidenceLevel.LOW
            else:
                req.confidence = ConfidenceLevel.UNCERTAIN

            req.confidence_score = combined_score

            # Flag low confidence for review
            if req.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]:
                if not req.needs_review:
                    req.needs_review = True
                    req.review_reasons.append(
                        f"CLASSIFICATION_UNCERTAIN: Section {section} assigned with "
                        f"confidence {combined_score:.2f}"
                    )

        # Log classification summary
        self._log_classification_summary(requirements)

        return requirements

    def _build_document_section_map(self, documents: List[Dict]) -> Dict[str, str]:
        """Map document filenames to their likely section"""
        doc_map = {}

        for doc in documents:
            filename = doc.get('filename', '').lower()

            # SOW documents -> Section C
            if any(p in filename for p in self.SOW_FILENAME_PATTERNS):
                doc_map[doc.get('filename', '')] = 'C'
                continue

            # Check content for SOW indicators
            text = doc.get('text', '')[:10000]
            for pattern in self.SOW_CONTENT_PATTERNS:
                if re.search(pattern, text.upper()):
                    doc_map[doc.get('filename', '')] = 'C'
                    break

        return doc_map

    def _classify_requirement(
        self,
        req: RequirementCandidate,
        doc_section_map: Dict[str, str]
    ) -> Tuple[str, str, float, List[str]]:
        """
        Classify a single requirement.

        Returns: (section, category, confidence, reasons)
        """
        reasons = []
        text = req.text.lower()

        # Layer 1: Document-level classification
        if req.source_document in doc_section_map:
            doc_section = doc_section_map[req.source_document]
            reasons.append(f"Document '{req.source_document}' classified as Section {doc_section}")
            return doc_section, self._section_to_category(doc_section), 0.85, reasons

        # Layer 2: Document is known SOW
        if req.source_document in self.sow_info.sow_documents:
            reasons.append(f"Source document is SOW")
            return 'C', 'TECHNICAL_REQUIREMENT', 0.9, reasons

        # Layer 3: RFP reference indicates section
        if req.rfp_reference:
            section_from_ref = self._section_from_reference(req.rfp_reference)
            if section_from_ref:
                reasons.append(f"RFP reference '{req.rfp_reference}' indicates Section {section_from_ref}")
                return section_from_ref, self._section_to_category(section_from_ref), 0.8, reasons

        # Layer 4: Content pattern matching
        section_scores = self._score_content_patterns(text)

        if section_scores:
            best_section = max(section_scores, key=section_scores.get)
            best_score = section_scores[best_section]

            if best_score >= 0.5:
                reasons.append(f"Content patterns match Section {best_section} (score: {best_score:.2f})")
                return best_section, self._section_to_category(best_section), best_score, reasons

        # Layer 5: Binding level hints (v3.2: Prioritize Section L detection)
        if req.binding_level == "SHALL":
            # v3.2: Check Section L FIRST - proposal instructions take priority
            section_l_indicators = [
                r'proposal|offeror|submit(?:ted)?|format|page\s*limit',
                r'electronic\s+(?:submission|file)',
                r'margin|font\s+size|8\.5\s*x\s*11',
                r'intends?\s+to\s+(?:make\s+)?award',
                r'without\s+conducting\s+(?:discussions|interchanges)',
                r'instructions?:\s*',
                r'^a\s+page\s+is\s+',
            ]
            for indicator in section_l_indicators:
                if re.search(indicator, text):
                    reasons.append(f"SHALL statement with proposal/instruction indicator '{indicator}' -> Section L")
                    return 'L', 'PROPOSAL_INSTRUCTION', 0.7, reasons

            # SHALL statements about contractor work -> Section C
            if re.search(r'contractor|perform|deliver|provide\s+(?:service|support)', text):
                reasons.append("SHALL statement about contractor work -> Section C")
                return 'C', 'TECHNICAL_REQUIREMENT', 0.6, reasons

        # Layer 6: Fallback - assign to most likely section with low confidence
        if self.sow_info.sow_detected:
            # If we have SOW and this has binding language, probably Section C
            if req.binding_level in ["SHALL", "SHOULD"]:
                reasons.append("Fallback: Binding language with SOW present -> Section C (low confidence)")
                return 'C', 'TECHNICAL_REQUIREMENT', 0.3, reasons

        # Ultimate fallback - informational/other
        reasons.append("Fallback: Could not determine section, assigned as General")
        return 'UNASSIGNED', 'GENERAL', 0.2, reasons

    def _section_from_reference(self, reference: str) -> Optional[str]:
        """Extract section letter from RFP reference"""
        if reference:
            first_char = reference[0].upper()
            if first_char in 'ABCDEFGHIJKLM':
                return first_char
        return None

    def _score_content_patterns(self, text: str) -> Dict[str, float]:
        """Score text against section-specific patterns"""
        scores = {}

        # Section C patterns
        c_score = 0.0
        for pattern, weight in self.SECTION_C_PATTERNS:
            if re.search(pattern, text):
                c_score += weight
        if c_score > 0:
            scores['C'] = min(c_score, 1.0)

        # Section L patterns
        l_score = 0.0
        for pattern, weight in self.SECTION_L_PATTERNS:
            if re.search(pattern, text):
                l_score += weight
        if l_score > 0:
            scores['L'] = min(l_score, 1.0)

        # Section M patterns
        m_score = 0.0
        for pattern, weight in self.SECTION_M_PATTERNS:
            if re.search(pattern, text):
                m_score += weight
        if m_score > 0:
            scores['M'] = min(m_score, 1.0)

        # Section K patterns
        k_score = 0.0
        for pattern, weight in self.SECTION_K_PATTERNS:
            if re.search(pattern, text):
                k_score += weight
        if k_score > 0:
            scores['K'] = min(k_score, 1.0)

        return scores

    def _section_to_category(self, section: str) -> str:
        """Map UCF section to requirement category"""
        category_map = {
            'A': 'CONTRACT_FORM',
            'B': 'PRICING',
            'C': 'TECHNICAL_REQUIREMENT',
            'D': 'PACKAGING',
            'E': 'INSPECTION',
            'F': 'DELIVERY',
            'G': 'ADMINISTRATION',
            'H': 'SPECIAL_REQUIREMENT',
            'I': 'CLAUSE',
            'J': 'ATTACHMENT',
            'K': 'REPRESENTATION',
            'L': 'PROPOSAL_INSTRUCTION',
            'M': 'EVALUATION_FACTOR',
        }
        return category_map.get(section, 'GENERAL')

    def _log_classification_summary(self, requirements: List[RequirementCandidate]):
        """Log summary of classification results"""
        section_counts = {}
        confidence_counts = {
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'UNCERTAIN': 0
        }

        for req in requirements:
            section = req.assigned_section or 'UNASSIGNED'
            section_counts[section] = section_counts.get(section, 0) + 1
            confidence_counts[req.confidence.value] += 1

        logger.info(f"Classification complete: {len(requirements)} requirements")
        logger.info(f"By section: {section_counts}")
        logger.info(f"By confidence: {confidence_counts}")

        # Warn if Section C is suspiciously low
        section_c_count = section_counts.get('C', 0)
        if section_c_count < 10 and len(requirements) > 50:
            logger.warning(
                f"LOW SECTION C: Only {section_c_count} requirements classified as Section C. "
                f"SOW may not have been properly detected."
            )
