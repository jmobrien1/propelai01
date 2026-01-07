"""
PropelAI v6.0.7: RFQSkeletonBuilder - SOO-First Fallback Strategy

This module handles non-UCF procurement formats like GSA RFQs (FAR 8.4)
that don't follow the traditional Section L/M/C structure.

Strategy:
1. Parse SOO/SOW headings as proposal sections
2. Create simplified 2-volume structure (Technical + Price)
3. Map requirements into Technical volume based on SOO structure

Key Principle: When Section L structure is absent, use SOO as structural guide.
Never hallucinate volumes - if structure can't be determined, FAIL LOUDLY.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re
import logging

from .strict_structure_builder import (
    ProposalSkeleton,
    SkeletonVolume,
    SkeletonSection,
    StructureValidationError
)

logger = logging.getLogger(__name__)


@dataclass
class SOOSection:
    """A section extracted from Statement of Objectives document."""
    id: str                    # e.g., "3.1", "C.1.4"
    title: str
    level: int                 # 1 = top level, 2 = subsection, etc.
    text_preview: str = ""     # First 200 chars for context
    parent_id: Optional[str] = None


@dataclass
class RFQStructureAnalysis:
    """
    v6.0.7: Analysis result for RFQ structure detection.

    Contains what was found and what's missing, enabling
    informative error messages rather than silent failures.
    """
    has_soo: bool = False
    has_sow: bool = False
    has_quote_instructions: bool = False
    has_price_schedule: bool = False
    has_phase_structure: bool = False

    soo_sections: List[SOOSection] = field(default_factory=list)
    detected_phases: List[str] = field(default_factory=list)
    page_limits: Dict[str, int] = field(default_factory=dict)

    # What's missing (for error messages)
    missing_elements: List[str] = field(default_factory=list)

    @property
    def can_build_skeleton(self) -> bool:
        """Whether we have enough to build a skeleton."""
        return self.has_soo or self.has_sow or self.has_quote_instructions

    @property
    def confidence_score(self) -> float:
        """Confidence in the structure analysis (0.0-1.0)."""
        score = 0.0
        if self.has_soo or self.has_sow:
            score += 0.4
        if self.has_quote_instructions:
            score += 0.3
        if self.has_price_schedule:
            score += 0.15
        if len(self.soo_sections) >= 3:
            score += 0.15
        return min(1.0, score)


class RFQSkeletonBuilder:
    """
    v6.0.7: Builds proposal skeleton for GSA RFQ and similar formats.

    Unlike StrictStructureBuilder which requires Section L volume declarations,
    this builder uses SOO/SOW structure as the basis for proposal organization.

    Usage:
        builder = RFQSkeletonBuilder()
        analysis = builder.analyze_structure(rfq_text, soo_text)
        if analysis.can_build_skeleton:
            skeleton = builder.build_skeleton(analysis, rfp_number, rfp_title)
        else:
            raise StructureValidationError(builder.get_failure_message(analysis))
    """

    # Patterns for SOO/SOW section detection
    SOO_SECTION_PATTERNS = [
        # "3.0 Title" or "3.1 Title" format
        r'^(\d+\.\d*)\s+([A-Z][A-Za-z\s&,/-]+)',
        # "C.1.4 Title" format (CLIN-based)
        r'^(C\.\d+\.?\d*)\s+([A-Z][A-Za-z\s&,/-]+)',
        # "Section 3: Title" format
        r'^Section\s+(\d+)[:\s]+([A-Z][A-Za-z\s&,/-]+)',
    ]

    # Patterns for quote instruction detection
    QUOTE_INSTRUCTION_PATTERNS = [
        r'quote\s+submission',
        r'quotation\s+instructions',
        r'instructions\s+to\s+(?:quoters|vendors|contractors)',
        r'how\s+to\s+submit',
        r'submission\s+requirements',
        r'response\s+format',
    ]

    # Phase patterns for GSA phased procurements
    PHASE_PATTERNS = [
        r'Phase\s+([I1])\s*[-:]\s*([A-Za-z\s]+)',
        r'Phase\s+([II2])\s*[-:]\s*([A-Za-z\s]+)',
        r'Factor\s+(\d+)\s*[-:]\s*([A-Za-z\s]+)',
    ]

    def analyze_structure(
        self,
        rfq_text: str,
        soo_text: Optional[str] = None,
        attachments: Optional[Dict[str, str]] = None
    ) -> RFQStructureAnalysis:
        """
        Analyze RFQ documents to determine available structure.

        Args:
            rfq_text: Main RFQ document text
            soo_text: Statement of Objectives text (if separate)
            attachments: Dict of attachment name -> text

        Returns:
            RFQStructureAnalysis with findings and confidence
        """
        analysis = RFQStructureAnalysis()

        # Combine all text for analysis
        all_text = rfq_text or ""
        if soo_text:
            all_text += "\n" + soo_text
        if attachments:
            for name, text in attachments.items():
                if 'soo' in name.lower() or 'sow' in name.lower():
                    all_text += "\n" + text

        if not all_text.strip():
            analysis.missing_elements.append("No document text available")
            return analysis

        # Check for SOO/SOW
        if soo_text or re.search(r'Statement\s+of\s+Objectives?|SOO\b', all_text, re.I):
            analysis.has_soo = True
        if re.search(r'Statement\s+of\s+Work|SOW\b|PWS\b', all_text, re.I):
            analysis.has_sow = True

        # Check for quote instructions
        for pattern in self.QUOTE_INSTRUCTION_PATTERNS:
            if re.search(pattern, all_text, re.I):
                analysis.has_quote_instructions = True
                break

        # Check for price schedule
        if re.search(r'Price\s+(?:Quote\s+)?Schedule|Pricing\s+Template|CLIN.*Price', all_text, re.I):
            analysis.has_price_schedule = True

        # Extract SOO sections
        source_text = soo_text if soo_text else all_text
        analysis.soo_sections = self._extract_soo_sections(source_text)

        # Check for phase structure
        analysis.detected_phases = self._extract_phases(all_text)
        analysis.has_phase_structure = len(analysis.detected_phases) > 0

        # Extract any page limits mentioned
        analysis.page_limits = self._extract_page_limits(all_text)

        # Determine what's missing
        if not analysis.has_soo and not analysis.has_sow:
            analysis.missing_elements.append("No SOO or SOW document found")
        if not analysis.has_quote_instructions:
            analysis.missing_elements.append("No quote submission instructions found")
        if len(analysis.soo_sections) == 0:
            analysis.missing_elements.append("Could not extract section structure from SOO/SOW")

        logger.info(f"[v6.0.7] RFQ Analysis: SOO={analysis.has_soo}, "
                   f"SOW={analysis.has_sow}, sections={len(analysis.soo_sections)}, "
                   f"phases={len(analysis.detected_phases)}, confidence={analysis.confidence_score:.2f}")

        return analysis

    def _extract_soo_sections(self, text: str) -> List[SOOSection]:
        """Extract section structure from SOO/SOW text."""
        sections: List[SOOSection] = []
        seen_ids: set = set()

        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) > 200:  # Skip empty or overly long lines
                continue

            for pattern in self.SOO_SECTION_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    sec_id = match.group(1)
                    sec_title = match.group(2).strip()

                    # Clean up title
                    sec_title = re.sub(r'[\.\,\;\:]+$', '', sec_title).strip()

                    # Skip if already seen or title too short
                    if sec_id in seen_ids or len(sec_title) < 3:
                        continue
                    seen_ids.add(sec_id)

                    # Determine level from ID
                    level = sec_id.count('.') + 1

                    # Get text preview (next ~200 chars)
                    preview_start = i + 1
                    preview_text = ""
                    for j in range(preview_start, min(preview_start + 5, len(lines))):
                        preview_text += lines[j] + " "
                        if len(preview_text) > 200:
                            break

                    sections.append(SOOSection(
                        id=sec_id,
                        title=sec_title,
                        level=level,
                        text_preview=preview_text[:200].strip()
                    ))
                    break

        # Sort by section ID
        sections.sort(key=lambda s: self._section_sort_key(s.id))

        return sections

    def _section_sort_key(self, sec_id: str) -> Tuple:
        """Generate sort key for section IDs like '3.1', 'C.1.4'."""
        parts = re.split(r'[.\-]', sec_id)
        result = []
        for part in parts:
            if part.isdigit():
                result.append((0, int(part)))
            else:
                result.append((1, part))
        return tuple(result)

    def _extract_phases(self, text: str) -> List[str]:
        """Extract phase names from GSA phased procurement."""
        phases = []
        for pattern in self.PHASE_PATTERNS:
            for match in re.finditer(pattern, text, re.I):
                phase_num = match.group(1)
                phase_name = match.group(2).strip() if len(match.groups()) > 1 else ""
                phases.append(f"Phase {phase_num}: {phase_name}".strip(': '))
        return list(set(phases))  # Deduplicate

    def _extract_page_limits(self, text: str) -> Dict[str, int]:
        """Extract any page limits mentioned in RFQ."""
        limits = {}

        # Pattern: "Technical Quote: 10 pages" or "not to exceed 15 pages"
        patterns = [
            r'(Technical|Management|Price|Quote)[^.]*?(\d+)\s*(?:pages?|pgs?)',
            r'not\s+to\s+exceed\s+(\d+)\s*(?:pages?|pgs?)',
            r'limited\s+to\s+(\d+)\s*(?:pages?|pgs?)',
            r'maximum\s+(?:of\s+)?(\d+)\s*(?:pages?|pgs?)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                if len(match.groups()) == 2:
                    category = match.group(1).lower()
                    limit = int(match.group(2))
                else:
                    category = 'general'
                    limit = int(match.group(1))

                if 1 <= limit <= 500:  # Sanity check
                    limits[category] = limit

        return limits

    def build_skeleton(
        self,
        analysis: RFQStructureAnalysis,
        rfp_number: str,
        rfp_title: str
    ) -> ProposalSkeleton:
        """
        Build proposal skeleton from RFQ structure analysis.

        Args:
            analysis: Result from analyze_structure()
            rfp_number: Solicitation number
            rfp_title: RFP/RFQ title

        Returns:
            ProposalSkeleton suitable for content injection

        Raises:
            StructureValidationError: If skeleton cannot be built
        """
        if not analysis.can_build_skeleton:
            raise StructureValidationError(self.get_failure_message(analysis))

        volumes: List[SkeletonVolume] = []

        # Volume 1: Technical Quote
        tech_sections = self._build_technical_sections(analysis)
        tech_page_limit = analysis.page_limits.get('technical') or analysis.page_limits.get('general')

        volumes.append(SkeletonVolume(
            id="VOL-1",
            title="Technical Quote",
            volume_number=1,
            page_limit=tech_page_limit,
            source_reference="Inferred from SOO structure (v6.0.7)",
            sections=tech_sections
        ))

        # Volume 2: Price Quote (always present for RFQs)
        price_page_limit = analysis.page_limits.get('price')

        volumes.append(SkeletonVolume(
            id="VOL-2",
            title="Price Quote",
            volume_number=2,
            page_limit=price_page_limit,
            source_reference="Standard RFQ structure (v6.0.7)",
            sections=[
                SkeletonSection(
                    id="VOL-2-SEC-1",
                    title="Pricing Schedule",
                    page_limit=None,
                    order=0,
                    source_reference="Standard RFQ Price Volume"
                )
            ]
        ))

        # Build skeleton
        skeleton = ProposalSkeleton(
            rfp_number=rfp_number,
            rfp_title=rfp_title,
            volumes=volumes,
            total_page_limit=sum(v.page_limit or 0 for v in volumes) or None,
            format_rules={},
            submission_rules={},
            stated_volume_count=2,
            is_valid=True,
            validation_warnings=[
                "Structure inferred from SOO - not from explicit Section L instructions",
                "Manual review recommended to verify section organization"
            ]
        )

        logger.info(f"[v6.0.7] Built RFQ skeleton: {len(volumes)} volumes, "
                   f"{sum(len(v.sections) for v in volumes)} sections")

        return skeleton

    def _build_technical_sections(self, analysis: RFQStructureAnalysis) -> List[SkeletonSection]:
        """Build technical volume sections from SOO structure."""
        sections: List[SkeletonSection] = []

        if analysis.soo_sections:
            # Use SOO sections as proposal sections
            for i, soo_sec in enumerate(analysis.soo_sections):
                # Only use top-level sections (level 1-2) as main proposal sections
                if soo_sec.level <= 2:
                    sections.append(SkeletonSection(
                        id=f"VOL-1-SEC-{i+1}",
                        title=soo_sec.title,
                        page_limit=None,
                        order=i,
                        source_reference=f"SOO Section {soo_sec.id}"
                    ))

        # If no sections found, create minimal structure
        if not sections:
            sections = [
                SkeletonSection(
                    id="VOL-1-SEC-1",
                    title="Technical Approach",
                    page_limit=None,
                    order=0,
                    source_reference="Default RFQ structure (no SOO sections found)"
                ),
                SkeletonSection(
                    id="VOL-1-SEC-2",
                    title="Management Approach",
                    page_limit=None,
                    order=1,
                    source_reference="Default RFQ structure"
                ),
                SkeletonSection(
                    id="VOL-1-SEC-3",
                    title="Staffing / Key Personnel",
                    page_limit=None,
                    order=2,
                    source_reference="Default RFQ structure"
                ),
            ]

        return sections

    def get_failure_message(self, analysis: RFQStructureAnalysis) -> str:
        """
        Generate helpful error message when skeleton cannot be built.

        This is the "Fail Loud, Not Wrong" principle - provide actionable
        guidance rather than hallucinating a structure.
        """
        lines = [
            "Cannot generate proposal outline: Insufficient structure information.",
            "",
            "What we found:",
        ]

        # What was found
        found_items = []
        if analysis.has_soo:
            found_items.append("✓ Statement of Objectives (SOO)")
        if analysis.has_sow:
            found_items.append("✓ Statement of Work (SOW)")
        if analysis.has_quote_instructions:
            found_items.append("✓ Quote submission instructions")
        if analysis.has_price_schedule:
            found_items.append("✓ Price schedule/template")
        if analysis.soo_sections:
            found_items.append(f"✓ {len(analysis.soo_sections)} SOO sections identified")

        if found_items:
            lines.extend([f"  {item}" for item in found_items])
        else:
            lines.append("  (nothing recognizable)")

        # What's missing
        lines.append("")
        lines.append("What's missing:")
        if analysis.missing_elements:
            lines.extend([f"  ✗ {item}" for item in analysis.missing_elements])
        else:
            lines.append("  ✗ No specific structure markers found")

        # Suggestions
        lines.extend([
            "",
            "Suggestions:",
            "  1. Upload the Statement of Objectives (SOO) document separately",
            "  2. Ensure the main RFQ document contains quote submission instructions",
            "  3. If this is a GSA RFQ, look for 'Instructions to Quoters' section",
            "  4. Contact support if you believe this format should be supported",
            "",
            f"Confidence: {analysis.confidence_score:.0%} (minimum 40% required)"
        ])

        return "\n".join(lines)


# ============================================================================
# Structured Error Classes for v6.0.7
# ============================================================================

@dataclass
class OutlineGenerationError:
    """
    v6.0.7: Structured error for outline generation failures.

    Provides detailed context for debugging and user guidance.
    """
    error_code: str
    message: str
    detected_signals: List[str] = field(default_factory=list)
    missing_elements: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    procurement_type: Optional[str] = None
    confidence_score: float = 0.0
    recoverable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict for API response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": {
                "detected_signals": self.detected_signals,
                "missing_elements": self.missing_elements,
                "procurement_type": self.procurement_type,
                "confidence": self.confidence_score,
            },
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
        }


def create_unsupported_format_error(
    detected_type: str,
    confidence: float,
    signals: List[str],
    missing: List[str]
) -> OutlineGenerationError:
    """Factory for unsupported format errors."""
    return OutlineGenerationError(
        error_code="UNSUPPORTED_PROCUREMENT_FORMAT",
        message=f"This solicitation uses a format PropelAI doesn't fully support yet ({detected_type}).",
        detected_signals=signals,
        missing_elements=missing,
        suggestions=[
            "Try uploading the SOO/SOW document to 'Technical Requirements' section",
            "Ensure quote instructions are included in uploaded documents",
            "Contact support for assistance with this procurement format",
        ],
        procurement_type=detected_type,
        confidence_score=confidence,
        recoverable=True,
    )


def create_missing_structure_error(
    what_found: Dict[str, Any],
    what_needed: str
) -> OutlineGenerationError:
    """Factory for missing structure errors."""
    found_list = [f"{k}: {v}" for k, v in what_found.items() if v]
    return OutlineGenerationError(
        error_code="MISSING_STRUCTURE_SOURCE",
        message="Cannot generate outline: No proposal structure information found.",
        detected_signals=found_list,
        missing_elements=[what_needed],
        suggestions=[
            "Upload a document containing proposal submission instructions",
            "For UCF RFPs: Upload Section L (Instructions to Offerors)",
            "For GSA RFQs: Upload the main RFQ with quote instructions",
        ],
        recoverable=False,
    )
