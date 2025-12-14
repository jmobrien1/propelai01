"""
PropelAI Section Inference Module

Provides fallback section detection when explicit headers aren't found.
This addresses the "43% UNK section" problem by:

1. Content-based inference (keywords suggest section type)
2. Document-level inference (SOW.pdf → Section C content)
3. Page position inference (L/M typically in back half of document)
4. Cross-reference inference (if text mentions "Section L", assign to L)

Usage:
    from agents.enhanced_compliance.section_inference import SectionInferencer

    inferencer = SectionInferencer()
    inferred_section = inferencer.infer_section(
        text="The offeror shall submit a technical proposal...",
        filename="solicitation.pdf",
        page_number=45,
        total_pages=80
    )
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .document_structure import UCFSection


@dataclass
class InferenceResult:
    """Result of section inference"""
    section: UCFSection
    confidence: float           # 0.0 - 1.0
    method: str                 # How was this inferred
    evidence: List[str]         # What keywords/patterns supported this


class SectionInferencer:
    """
    Infers section assignment when explicit headers aren't found.

    This is a fallback system - explicit section detection always takes priority.
    """

    # Section L indicators (Instructions to Offerors)
    SECTION_L_INDICATORS = {
        'high_confidence': [
            r'\bofferor\s+shall\s+submit\b',
            r'\bproposal\s+(?:shall|must)\s+(?:include|contain|address)\b',
            r'\bpage\s+limit\b',
            r'\bfont\s+(?:size|type|requirement)\b',
            r'\bformat\s+requirement\b',
            r'\bvolume\s+(?:i+|[1-4]|one|two|three|four)\b',
            r'\btechnical\s+proposal\b',
            r'\bprice\s*(?:/|and)\s*cost\s+proposal\b',
            r'\bpast\s+performance\s+(?:volume|proposal)\b',
            r'\bsubmission\s+(?:requirement|instruction)\b',
        ],
        'medium_confidence': [
            r'\bproposal\s+format\b',
            r'\bpage\s+(?:count|number)\b',
            r'\bmargin\b',
            r'\bspacing\b',
            r'\bsection\s+(?:heading|title)\b',
            r'\btable\s+of\s+contents\b',
        ],
        'low_confidence': [
            r'\bofferor\b',
            r'\bproposal\b',
            r'\bsubmit\b',
        ]
    }

    # Section M indicators (Evaluation Factors)
    SECTION_M_INDICATORS = {
        'high_confidence': [
            r'\bwill\s+be\s+evaluated\b',
            r'\bevaluation\s+(?:factor|criteria)\b',
            r'\bbasis\s+for\s+award\b',
            r'\badjectival\s+rating\b',
            r'\boutstanding|good|acceptable|marginal|unacceptable\b',
            r'\bstrength[s]?\s+(?:and|or|,)\s+weakness\b',
            r'\bdeficienc(?:y|ies)\b',
            r'\brisk\s+(?:rating|assessment)\b',
            r'\brelative\s+importance\b',
            r'\bweighting\b',
            r'\bfactor\s+\d+\b',
        ],
        'medium_confidence': [
            r'\bwill\s+be\s+assessed\b',
            r'\bwill\s+be\s+rated\b',
            r'\bwill\s+be\s+scored\b',
            r'\bconsidered\s+(?:in|during)\s+evaluation\b',
            r'\bsub[- ]?factor\b',
        ],
        'low_confidence': [
            r'\bevaluat\b',
            r'\bfactor\b',
            r'\bcriteria\b',
        ]
    }

    # Section C indicators (SOW/PWS/Specifications)
    SECTION_C_INDICATORS = {
        'high_confidence': [
            r'\bcontractor\s+shall\s+(?:provide|perform|deliver|ensure|maintain)\b',
            r'\bstatement\s+of\s+work\b',
            r'\bperformance\s+work\s+statement\b',
            r'\bscope\s+of\s+work\b',
            r'\btask\s+(?:area|order)\s+\d+\b',
            r'\bdeliverable[s]?\b',
            r'\bperformance\s+(?:requirement|standard|objective)\b',
            r'\bservice\s+level\s+(?:agreement|requirement)\b',
        ],
        'medium_confidence': [
            r'\bcontractor\s+(?:shall|will|must)\b',
            r'\bgovernment\s+(?:will|shall)\s+provide\b',
            r'\bwork\s+requirement\b',
            r'\btechnical\s+requirement\b',
            r'\boperational\s+requirement\b',
        ],
        'low_confidence': [
            r'\bcontractor\b',
            r'\bperform\b',
            r'\bdeliver\b',
        ]
    }

    # Section B indicators (Supplies/Services and Prices)
    SECTION_B_INDICATORS = {
        'high_confidence': [
            r'\bclin\s+\d+\b',
            r'\bline\s+item\s+(?:no\.?|number)\b',
            r'\bunit\s+price\b',
            r'\btotal\s+price\b',
            r'\bestimated\s+cost\b',
            r'\bfixed\s*[- ]?price\b',
            r'\bcost\s*[- ]?plus\b',
            r'\bt&m\b',  # Time and Materials
        ],
        'medium_confidence': [
            r'\bpricing\b',
            r'\bquantity\b',
            r'\bunit\b',
            r'\bamount\b',
        ]
    }

    # Section I indicators (Contract Clauses)
    SECTION_I_INDICATORS = {
        'high_confidence': [
            r'\bfar\s+\d+\.\d+\b',
            r'\bdfars\s+\d+\.\d+\b',
            r'\b52\.\d{3}[- ]\d+\b',  # FAR clause numbers
            r'\b252\.\d{3}[- ]\d+\b',  # DFARS clause numbers
            r'\bincorporated\s+by\s+reference\b',
        ],
        'medium_confidence': [
            r'\bclause\b',
            r'\bfederal\s+acquisition\s+regulation\b',
        ]
    }

    # Document type to section mapping
    DOCUMENT_TYPE_MAPPING = {
        'sow': UCFSection.SECTION_C,
        'pws': UCFSection.SECTION_C,
        'statement of work': UCFSection.SECTION_C,
        'performance work statement': UCFSection.SECTION_C,
        'specifications': UCFSection.SECTION_C,
        'technical requirements': UCFSection.SECTION_C,
        'evaluation': UCFSection.SECTION_M,
        'pricing': UCFSection.SECTION_B,
        'cost': UCFSection.SECTION_B,
        'clauses': UCFSection.SECTION_I,
    }

    def __init__(self):
        self.all_indicators = {
            UCFSection.SECTION_L: self.SECTION_L_INDICATORS,
            UCFSection.SECTION_M: self.SECTION_M_INDICATORS,
            UCFSection.SECTION_C: self.SECTION_C_INDICATORS,
            UCFSection.SECTION_B: self.SECTION_B_INDICATORS,
            UCFSection.SECTION_I: self.SECTION_I_INDICATORS,
        }

    def infer_section(
        self,
        text: str,
        filename: Optional[str] = None,
        page_number: Optional[int] = None,
        total_pages: Optional[int] = None,
        existing_section: Optional[UCFSection] = None
    ) -> InferenceResult:
        """
        Infer the most likely section for a piece of text.

        Args:
            text: The requirement text
            filename: Source filename (optional)
            page_number: Page number in document (optional)
            total_pages: Total pages in document (optional)
            existing_section: Already assigned section (for validation)

        Returns:
            InferenceResult with section, confidence, and evidence
        """
        scores: Dict[UCFSection, Tuple[float, List[str]]] = {}

        # 1. Content-based inference (highest priority)
        for section, indicators in self.all_indicators.items():
            score, evidence = self._score_content(text, indicators)
            if score > 0:
                scores[section] = (score, evidence)

        # 2. Document-level inference
        if filename:
            doc_section, doc_evidence = self._infer_from_filename(filename)
            if doc_section:
                current = scores.get(doc_section, (0, []))
                scores[doc_section] = (
                    current[0] + 0.3,
                    current[1] + [doc_evidence]
                )

        # 3. Page position inference
        if page_number and total_pages and total_pages > 20:
            position_section, position_score = self._infer_from_position(
                page_number, total_pages
            )
            if position_section:
                current = scores.get(position_section, (0, []))
                scores[position_section] = (
                    current[0] + position_score,
                    current[1] + [f"Page position {page_number}/{total_pages}"]
                )

        # 4. Cross-reference inference
        ref_section, ref_evidence = self._infer_from_references(text)
        if ref_section:
            current = scores.get(ref_section, (0, []))
            scores[ref_section] = (
                current[0] + 0.2,
                current[1] + [ref_evidence]
            )

        # Find best match
        if scores:
            best_section = max(scores.keys(), key=lambda s: scores[s][0])
            best_score, evidence = scores[best_section]

            # Normalize confidence to 0-1
            confidence = min(best_score, 1.0)

            return InferenceResult(
                section=best_section,
                confidence=confidence,
                method="content_inference",
                evidence=evidence
            )

        # Default fallback based on content type
        if self._looks_like_instruction(text):
            return InferenceResult(
                section=UCFSection.SECTION_L,
                confidence=0.3,
                method="fallback_instruction",
                evidence=["Contains proposal/submission language"]
            )
        elif self._looks_like_technical(text):
            return InferenceResult(
                section=UCFSection.SECTION_C,
                confidence=0.3,
                method="fallback_technical",
                evidence=["Contains contractor/performance language"]
            )

        # Ultimate fallback
        return InferenceResult(
            section=UCFSection.SECTION_H,  # Special requirements as catch-all
            confidence=0.1,
            method="fallback_unknown",
            evidence=["No clear section indicators found"]
        )

    def _score_content(
        self,
        text: str,
        indicators: Dict[str, List[str]]
    ) -> Tuple[float, List[str]]:
        """Score text against section indicators"""
        text_lower = text.lower()
        score = 0.0
        evidence = []

        for pattern in indicators.get('high_confidence', []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.4
                evidence.append(f"High: {pattern[:30]}...")

        for pattern in indicators.get('medium_confidence', []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.2
                evidence.append(f"Med: {pattern[:30]}...")

        for pattern in indicators.get('low_confidence', []):
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 0.05
                # Don't add low confidence to evidence (too noisy)

        return score, evidence[:5]  # Limit evidence list

    def _infer_from_filename(
        self,
        filename: str
    ) -> Tuple[Optional[UCFSection], str]:
        """Infer section from document filename"""
        filename_lower = filename.lower()

        for keyword, section in self.DOCUMENT_TYPE_MAPPING.items():
            if keyword in filename_lower:
                return section, f"Filename contains '{keyword}'"

        # Check for section letter in filename
        section_match = re.search(r'section\s*([a-m])', filename_lower)
        if section_match:
            letter = section_match.group(1).upper()
            try:
                section = UCFSection(f"SECTION_{letter}")
                return section, f"Filename contains 'Section {letter}'"
            except ValueError:
                pass

        return None, ""

    def _infer_from_position(
        self,
        page_number: int,
        total_pages: int
    ) -> Tuple[Optional[UCFSection], float]:
        """Infer section from page position in document"""
        position_ratio = page_number / total_pages

        # Sections L and M are typically in the last 40% of UCF documents
        if position_ratio > 0.6:
            # Could be L or M - return weak signal
            return UCFSection.SECTION_L, 0.1

        # Sections A-C typically in first 30%
        if position_ratio < 0.3:
            return UCFSection.SECTION_C, 0.05

        return None, 0

    def _infer_from_references(self, text: str) -> Tuple[Optional[UCFSection], str]:
        """Infer section from cross-references in text"""
        # Look for explicit section references
        patterns = [
            (r'\bsection\s+l\b', UCFSection.SECTION_L),
            (r'\bsection\s+m\b', UCFSection.SECTION_M),
            (r'\bsection\s+c\b', UCFSection.SECTION_C),
            (r'\bl\.\d+', UCFSection.SECTION_L),
            (r'\bm\.\d+', UCFSection.SECTION_M),
            (r'\bc\.\d+', UCFSection.SECTION_C),
        ]

        for pattern, section in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return section, f"References {section.value}"

        return None, ""

    def _looks_like_instruction(self, text: str) -> bool:
        """Check if text looks like a proposal instruction"""
        text_lower = text.lower()
        instruction_words = [
            'offeror', 'proposal', 'submit', 'include', 'provide',
            'format', 'page', 'volume', 'section'
        ]
        count = sum(1 for word in instruction_words if word in text_lower)
        return count >= 3

    def _looks_like_technical(self, text: str) -> bool:
        """Check if text looks like a technical requirement"""
        text_lower = text.lower()
        technical_words = [
            'contractor', 'shall', 'perform', 'deliver', 'provide',
            'service', 'support', 'maintain', 'operate'
        ]
        count = sum(1 for word in technical_words if word in text_lower)
        return count >= 3

    def validate_section_assignment(
        self,
        requirement_text: str,
        assigned_section: UCFSection
    ) -> Tuple[bool, Optional[UCFSection], str]:
        """
        Validate if a requirement is correctly assigned to a section.

        Returns:
            (is_valid, suggested_section, reason)
        """
        inferred = self.infer_section(requirement_text)

        if inferred.section == assigned_section:
            return True, None, "Assignment matches inference"

        # Check confidence threshold
        if inferred.confidence > 0.6 and assigned_section != inferred.section:
            return False, inferred.section, f"High confidence inference suggests {inferred.section.value}"

        return True, None, "Assignment is plausible"


def infer_section(
    text: str,
    filename: Optional[str] = None,
    page_number: Optional[int] = None,
    total_pages: Optional[int] = None
) -> InferenceResult:
    """
    Convenience function to infer section.

    Usage:
        result = infer_section(
            "The offeror shall submit a technical proposal...",
            filename="solicitation.pdf",
            page_number=45,
            total_pages=80
        )
        print(f"Likely section: {result.section.value} ({result.confidence:.0%} confidence)")
    """
    inferencer = SectionInferencer()
    return inferencer.infer_section(text, filename, page_number, total_pages)
