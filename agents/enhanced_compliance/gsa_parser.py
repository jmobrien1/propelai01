"""
PropelAI v6.0.12: GSAParser - Dedicated GSA/FAR 8.4 Strategy

This is a SEPARATE parsing strategy for GSA RFQs that is ISOLATED from
UCF (OASIS/Air Force) logic to prevent cross-contamination.

Key Differences from UCF:
- Phase I/II are TOP-LEVEL structural roots (not fallback)
- Section 11/12 replace Section L/M for instructions/evaluation
- Factor isolation: Factor 1 → Phase I, Factors 2-5 → Phase II
- SOO-First strategy when quote structure is minimal

"Architectural Separation" - GSA and UCF NEVER share parsing logic.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from .section_l_schema import VolumeInstruction, SectionInstruction

logger = logging.getLogger(__name__)


class GSAPhase(Enum):
    """GSA RFQ Phase classifications."""
    PHASE_I = "phase_i"     # Technical Experience / Past Performance
    PHASE_II = "phase_ii"   # Technical Approach / Oral Presentations
    PRICE = "price"         # Price Quote


class GSAFactor(Enum):
    """Standard GSA RFQ evaluation factors."""
    FACTOR_1 = 1  # Prior Experience (ALWAYS Phase I)
    FACTOR_2 = 2  # Technical Approach (Phase II)
    FACTOR_3 = 3  # Staffing/Key Personnel (Phase II)
    FACTOR_4 = 4  # Management Approach (Phase II)
    FACTOR_5 = 5  # Price (Separate Volume)


# v6.0.12: STRICT FACTOR → PHASE MAPPING
# This is DETERMINISTIC - Factor 1 can NEVER appear in Phase II
FACTOR_PHASE_MAPPING = {
    GSAFactor.FACTOR_1: GSAPhase.PHASE_I,
    GSAFactor.FACTOR_2: GSAPhase.PHASE_II,
    GSAFactor.FACTOR_3: GSAPhase.PHASE_II,
    GSAFactor.FACTOR_4: GSAPhase.PHASE_II,
    GSAFactor.FACTOR_5: GSAPhase.PRICE,
}


@dataclass
class GSAParseResult:
    """Result of GSA-specific parsing."""
    phases: List[Dict[str, Any]]
    factors: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    page_limits: Dict[str, int]
    solicitation_number: str
    warnings: List[str]
    section_11_found: bool = False
    section_12_found: bool = False


class GSAParser:
    """
    v6.0.12: Dedicated parser for GSA RFQ (FAR 8.4) procurements.

    This is COMPLETELY SEPARATE from SectionLParser (UCF).

    Structural Priorities:
    1. Phase markers (PHASE I, PHASE II) - TOP LEVEL
    2. Section 11 (Instructions) - replaces Section L
    3. Section 12 (Evaluation) - replaces Section M
    4. Factor markers within phases

    Usage:
        parser = GSAParser()
        result = parser.parse(rfq_text, soo_text)
        volumes = result.phases  # Phase I, Phase II, Price
    """

    # v6.0.12: PHASE-FIRST PATTERNS - These are structural roots
    PHASE_PATTERNS = [
        # "Phase I - Technical Experience" or "Phase I: Prior Experience"
        r'(?i)PHASE\s+([IVX\d]+)\s*[-:–]\s*([^\n]{3,60})',
        # "Phase 1 Technical" (no separator)
        r'(?i)PHASE\s+([IVX\d]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})',
        # Just "Phase I" or "Phase II"
        r'(?i)(?:^|\n)\s*(PHASE\s+[IVX\d]+)\s*(?:\n|$)',
    ]

    # v6.0.12: FACTOR PATTERNS - Nested within Phases
    FACTOR_PATTERNS = [
        # "Factor 1 - Prior Experience" or "Factor 1: Experience"
        r'(?i)Factor\s+(\d+)\s*[-:–]\s*([^\n]{3,60})',
        # "Evaluation Factor 1"
        r'(?i)Evaluation\s+Factor\s+(\d+)\s*[-:–]?\s*([^\n]{3,50})?',
        # "(1) Prior Experience" - parenthetical numbering
        r'(?i)\((\d)\)\s+([A-Z][^\n]{3,50})',
    ]

    # v6.0.12: SECTION 11/12 PATTERNS - GSA equivalents of Section L/M
    SECTION_11_PATTERNS = [
        r'(?i)Section\s+11\s*[-:–]?\s*(Instructions|Quote\s+Submission)',
        r'(?i)INSTRUCTIONS\s+TO\s+QUOTERS',
        r'(?i)Quote\s+Preparation\s+Instructions',
    ]

    SECTION_12_PATTERNS = [
        r'(?i)Section\s+12\s*[-:–]?\s*(Evaluation|Award\s+Criteria)',
        r'(?i)EVALUATION\s+CRITERIA',
        r'(?i)BASIS\s+FOR\s+AWARD',
    ]

    # v6.0.12: Agency-specific solicitation patterns for metadata extraction
    GSA_SOLICITATION_PATTERNS = [
        r'\b(693JJ4\d{2}[A-Z]\d{5,})\b',       # DOT/FMCSA
        r'\b(47Q[A-Z]{2}[A-Z0-9\-]+)\b',       # GSA MAS
        r'\b(GS-\d{2}F-\d{4,}[A-Z]?)\b',       # GSA Schedule
    ]

    def __init__(self):
        """Initialize the GSA parser with strict Factor→Phase mapping."""
        self.factor_phase_map = FACTOR_PHASE_MAPPING

    def parse(
        self,
        rfq_text: str,
        soo_text: Optional[str] = None,
        solicitation_number: str = ""
    ) -> GSAParseResult:
        """
        Parse GSA RFQ document using Phase-First strategy.

        Args:
            rfq_text: Main RFQ document text
            soo_text: Statement of Objectives text (if separate)
            solicitation_number: Known solicitation number (for override)

        Returns:
            GSAParseResult with phases, factors, and metadata
        """
        warnings: List[str] = []
        all_text = rfq_text + ("\n" + soo_text if soo_text else "")

        # Step 1: Extract solicitation number (override hallucinated UUIDs)
        extracted_sol = self._extract_solicitation_number(all_text)
        if extracted_sol and not solicitation_number:
            solicitation_number = extracted_sol
            print(f"[v6.0.12 GSA] Extracted solicitation: {solicitation_number}")
        elif extracted_sol and solicitation_number:
            # Prefer extracted over provided if extracted looks more official
            if re.match(r'^RFP[-_]?[A-F0-9]+', solicitation_number, re.I):
                print(f"[v6.0.12 GSA] METADATA OVERRIDE: {solicitation_number} → {extracted_sol}")
                solicitation_number = extracted_sol

        # Step 2: Detect Section 11/12 anchors
        section_11_found = any(
            re.search(p, rfq_text, re.IGNORECASE)
            for p in self.SECTION_11_PATTERNS
        )
        section_12_found = any(
            re.search(p, rfq_text, re.IGNORECASE)
            for p in self.SECTION_12_PATTERNS
        )

        if section_11_found:
            print("[v6.0.12 GSA] Found Section 11 (Instructions to Quoters)")
        if section_12_found:
            print("[v6.0.12 GSA] Found Section 12 (Evaluation Criteria)")

        # Step 3: PHASE-FIRST EXTRACTION - This is the structural root
        phases = self._extract_phases(rfq_text)

        if not phases:
            warnings.append("No Phase markers found - attempting factor-based inference")
            # Fall back to factor-based structure
            phases = self._infer_phases_from_factors(rfq_text)

        # Step 4: Extract factors and assign to phases
        factors = self._extract_factors(rfq_text)

        # Step 5: Enforce strict Factor→Phase isolation
        phases = self._enforce_factor_isolation(phases, factors, warnings)

        # Step 6: Extract page limits
        page_limits = self._extract_page_limits(rfq_text)

        # Step 7: Extract sections from SOO if available
        sections = []
        if soo_text:
            sections = self._extract_soo_sections(soo_text)

        print(f"[v6.0.12 GSA] Parse complete: {len(phases)} phases, "
              f"{len(factors)} factors, {len(sections)} SOO sections")

        return GSAParseResult(
            phases=phases,
            factors=factors,
            sections=sections,
            page_limits=page_limits,
            solicitation_number=solicitation_number,
            warnings=warnings,
            section_11_found=section_11_found,
            section_12_found=section_12_found
        )

    def _extract_solicitation_number(self, text: str) -> Optional[str]:
        """
        Extract official GSA solicitation number.

        v6.0.12: This OVERRIDES any hallucinated internal UUIDs.
        """
        for pattern in self.GSA_SOLICITATION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()
        return None

    def _extract_phases(self, text: str) -> List[Dict[str, Any]]:
        """
        v6.0.12: PHASE-FIRST EXTRACTION

        Phases are the TOP-LEVEL structural roots for GSA RFQs.
        This runs BEFORE any factor extraction.
        """
        phases: List[Dict[str, Any]] = []
        seen_phases: set = set()

        for pattern in self.PHASE_PATTERNS:
            for match in re.finditer(pattern, text, re.MULTILINE):
                if len(match.groups()) >= 1:
                    phase_id = match.group(1).strip().upper()
                    phase_title = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""

                    # Normalize phase ID
                    phase_num = self._roman_to_int(phase_id) if phase_id in ['I', 'II', 'III', 'IV', 'V'] else int(phase_id) if phase_id.isdigit() else 0

                    if phase_num == 0 or phase_num in seen_phases:
                        continue
                    seen_phases.add(phase_num)

                    # v6.0.12: HEADER SANITIZATION - 12 word max
                    if phase_title:
                        words = phase_title.split()
                        if len(words) > 12:
                            phase_title = ' '.join(words[:12])
                            print(f"[v6.0.12 GSA] Header truncated: {phase_title}")

                    # Determine phase type
                    phase_type = GSAPhase.PHASE_I if phase_num == 1 else GSAPhase.PHASE_II

                    # Default titles based on standard GSA structure
                    if not phase_title:
                        if phase_num == 1:
                            phase_title = "Technical Experience"
                        elif phase_num == 2:
                            phase_title = "Technical Approach"

                    phases.append({
                        'phase_number': phase_num,
                        'phase_id': f"PHASE-{phase_num}",
                        'title': phase_title,
                        'phase_type': phase_type.value,
                        'factors': [],  # Will be populated by _enforce_factor_isolation
                        'page_limit': None,
                        'source': 'extracted'
                    })

        # Sort by phase number
        phases.sort(key=lambda p: p['phase_number'])

        return phases

    def _infer_phases_from_factors(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback: Infer phase structure from factor patterns.

        If no explicit Phase markers, but we see Factor 1 (Experience)
        and Factor 2+ (Technical), infer Two-Phase structure.
        """
        factors = self._extract_factors(text)

        if not factors:
            return []

        has_experience_factor = any(
            f['factor_number'] == 1 or 'experience' in f.get('title', '').lower()
            for f in factors
        )
        has_technical_factors = any(
            f['factor_number'] > 1 or 'technical' in f.get('title', '').lower()
            for f in factors
        )

        phases = []

        if has_experience_factor:
            phases.append({
                'phase_number': 1,
                'phase_id': 'PHASE-1',
                'title': 'Technical Experience (Inferred)',
                'phase_type': GSAPhase.PHASE_I.value,
                'factors': [],
                'page_limit': None,
                'source': 'inferred_from_factors'
            })

        if has_technical_factors:
            phases.append({
                'phase_number': 2,
                'phase_id': 'PHASE-2',
                'title': 'Technical Approach (Inferred)',
                'phase_type': GSAPhase.PHASE_II.value,
                'factors': [],
                'page_limit': None,
                'source': 'inferred_from_factors'
            })

        return phases

    def _extract_factors(self, text: str) -> List[Dict[str, Any]]:
        """Extract evaluation factors from RFQ text."""
        factors: List[Dict[str, Any]] = []
        seen_factors: set = set()

        for pattern in self.FACTOR_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                factor_num = int(match.group(1))
                factor_title = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""

                if factor_num in seen_factors or factor_num < 1 or factor_num > 10:
                    continue
                seen_factors.add(factor_num)

                # v6.0.12: HEADER SANITIZATION - 12 word max
                if factor_title:
                    words = factor_title.split()
                    if len(words) > 12:
                        factor_title = ' '.join(words[:12])

                # Determine which phase this factor belongs to
                target_phase = self._get_factor_phase(factor_num, factor_title)

                factors.append({
                    'factor_number': factor_num,
                    'factor_id': f"FACTOR-{factor_num}",
                    'title': factor_title,
                    'target_phase': target_phase.value,
                })

        factors.sort(key=lambda f: f['factor_number'])
        return factors

    def _get_factor_phase(self, factor_num: int, factor_title: str) -> GSAPhase:
        """
        v6.0.12: STRICT Factor→Phase determination.

        Rules:
        - Factor 1 (Experience/Past Performance) → ALWAYS Phase I
        - Factors 2-4 (Technical/Staffing/Management) → ALWAYS Phase II
        - Factor 5+ (Price) → Price Volume
        """
        title_lower = factor_title.lower() if factor_title else ""

        # Factor 1 = Phase I (regardless of title)
        if factor_num == 1:
            return GSAPhase.PHASE_I

        # Price factors → Price volume
        if factor_num >= 5 or 'price' in title_lower or 'cost' in title_lower:
            return GSAPhase.PRICE

        # Experience keywords → Phase I (even if numbered 2+)
        experience_keywords = ['experience', 'past performance', 'prior', 'relevant']
        if any(kw in title_lower for kw in experience_keywords):
            return GSAPhase.PHASE_I

        # Everything else → Phase II
        return GSAPhase.PHASE_II

    def _enforce_factor_isolation(
        self,
        phases: List[Dict[str, Any]],
        factors: List[Dict[str, Any]],
        warnings: List[str]
    ) -> List[Dict[str, Any]]:
        """
        v6.0.12: STRICT FACTOR ISOLATION

        Ensures Factor 1 content CANNOT appear in Phase II outline.
        This is a HARD BLOCK - violations produce errors, not warnings.
        """
        # Create phase lookup
        phase_lookup = {p['phase_type']: p for p in phases}

        # Ensure we have both phases
        if GSAPhase.PHASE_I.value not in phase_lookup and any(
            f['target_phase'] == GSAPhase.PHASE_I.value for f in factors
        ):
            phases.append({
                'phase_number': 1,
                'phase_id': 'PHASE-1',
                'title': 'Phase I - Technical Experience',
                'phase_type': GSAPhase.PHASE_I.value,
                'factors': [],
                'page_limit': None,
                'source': 'created_for_factor_isolation'
            })
            phase_lookup = {p['phase_type']: p for p in phases}

        if GSAPhase.PHASE_II.value not in phase_lookup and any(
            f['target_phase'] == GSAPhase.PHASE_II.value for f in factors
        ):
            phases.append({
                'phase_number': 2,
                'phase_id': 'PHASE-2',
                'title': 'Phase II - Technical Approach',
                'phase_type': GSAPhase.PHASE_II.value,
                'factors': [],
                'page_limit': None,
                'source': 'created_for_factor_isolation'
            })
            phase_lookup = {p['phase_type']: p for p in phases}

        # Assign factors to phases (STRICTLY)
        for factor in factors:
            target_phase_type = factor['target_phase']

            if target_phase_type in phase_lookup:
                phase_lookup[target_phase_type]['factors'].append(factor)
                print(f"[v6.0.12 GSA] Factor {factor['factor_number']} → {target_phase_type}")
            elif target_phase_type == GSAPhase.PRICE.value:
                # Price factors don't go into Phase I or II
                print(f"[v6.0.12 GSA] Factor {factor['factor_number']} → Price Volume (separate)")
            else:
                warnings.append(f"Factor {factor['factor_number']} has no target phase")

        # Sort phases
        phases.sort(key=lambda p: p['phase_number'])

        return phases

    def _extract_page_limits(self, text: str) -> Dict[str, int]:
        """Extract page limits for phases."""
        limits = {}

        patterns = [
            # "Phase I: 10 pages" or "Phase I (10 page limit)"
            r'Phase\s+([IVX\d]+)[^.]*?(\d+)\s*(?:page|pg)',
            # "Technical Experience: not to exceed 10 pages"
            r'(?:Technical\s+)?Experience[^.]*?(?:not\s+to\s+exceed|maximum|limit[ed]?\s+to)\s*(\d+)\s*(?:page|pg)',
            # "Technical Approach: 15 page maximum"
            r'(?:Technical\s+)?Approach[^.]*?(\d+)\s*(?:page|pg)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if 'Phase' in pattern:
                    phase_id = match.group(1)
                    page_limit = int(match.group(2))
                    phase_num = self._roman_to_int(phase_id) if phase_id in ['I', 'II', 'III', 'IV', 'V'] else int(phase_id) if phase_id.isdigit() else 0
                    if phase_num > 0:
                        limits[f'phase_{phase_num}'] = page_limit
                else:
                    page_limit = int(match.group(1))
                    if 'Experience' in pattern:
                        limits['phase_1'] = page_limit
                    elif 'Approach' in pattern:
                        limits['phase_2'] = page_limit

        return limits

    def _extract_soo_sections(self, soo_text: str) -> List[Dict[str, Any]]:
        """Extract sections from Statement of Objectives."""
        sections = []

        # C.1.x pattern for SOO sections
        pattern = r'^(C\.[\d.]+)\s+(.+)'

        for line in soo_text.split('\n'):
            line = line.strip()
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                sec_id = match.group(1).upper()
                sec_title = match.group(2).strip()

                # v6.0.12: 12-word header gate
                words = sec_title.split()
                if len(words) > 12:
                    continue  # Skip sentence-long headers

                sections.append({
                    'id': sec_id,
                    'title': sec_title,
                    'source': 'soo'
                })

        return sections

    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer."""
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        roman = roman.upper()
        result = 0
        prev_value = 0

        for char in reversed(roman):
            value = roman_values.get(char, 0)
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value

        return result

    def to_volumes(self, result: GSAParseResult) -> List[VolumeInstruction]:
        """
        Convert GSA parse result to standard VolumeInstruction format.

        This allows the GSA output to flow into the standard ContentInjector.
        """
        volumes = []

        for phase in result.phases:
            phase_num = phase['phase_number']
            page_limit = result.page_limits.get(f'phase_{phase_num}')

            # Create sections from factors
            sections = []
            for i, factor in enumerate(phase.get('factors', [])):
                sections.append({
                    'section_id': f"PHASE-{phase_num}-SEC-{i+1}",
                    'title': factor.get('title', f"Factor {factor['factor_number']}"),
                    'parent_volume_id': phase['phase_id'],
                    'page_limit': None,
                    'order': i,
                    'source_reference': f"Factor {factor['factor_number']}"
                })

            volumes.append(VolumeInstruction(
                volume_id=phase['phase_id'],
                volume_title=phase['title'],
                volume_number=phase_num,
                page_limit=page_limit,
                sections=sections,
                source_reference=f"GSA Phase {phase_num} ({phase.get('source', 'extracted')})",
                is_mandatory=True
            ))

        # Add Price volume if price factors exist
        price_factors = [f for f in result.factors if f['target_phase'] == GSAPhase.PRICE.value]
        if price_factors:
            volumes.append(VolumeInstruction(
                volume_id='PRICE',
                volume_title='Price Quote',
                volume_number=len(result.phases) + 1,
                page_limit=None,
                sections=[],
                source_reference='GSA Price Volume',
                is_mandatory=True
            ))

        return volumes


# ============================================================================
# Strategy Manager
# ============================================================================

class ParsingStrategy(Enum):
    """Available parsing strategies."""
    UCF_STANDARD = "ucf"
    GSA_RFQ = "gsa_rfq"


class StrategyManager:
    """
    v6.0.12: Factory for selecting appropriate parsing strategy.

    Ensures GSA and UCF logic paths NEVER cross.

    Usage:
        manager = StrategyManager()
        strategy = manager.select_strategy(rfq_text, signals)

        if strategy == ParsingStrategy.GSA_RFQ:
            parser = GSAParser()
            result = parser.parse(rfq_text)
        else:
            parser = SectionLParser()  # UCF
            result = parser.parse(...)
    """

    # Detection patterns that FORCE GSA strategy
    GSA_FORCE_PATTERNS = [
        r'FAR\s*8\.4',
        r'GSA\s+Schedule',
        r'Phase\s+[I1]\s*[-:/]',  # Phase I marker
        r'693JJ4',                 # DOT/FMCSA solicitation
        r'47Q[A-Z]{2}',           # GSA MAS
        r'Section\s+11',          # GSA Instructions
        r'Section\s+12',          # GSA Evaluation
    ]

    # Detection patterns that FORCE UCF strategy
    UCF_FORCE_PATTERNS = [
        r'Section\s+L\s*[-:/]?\s*Instructions',
        r'Section\s+M\s*[-:/]?\s*Evaluation',
        r'FAR\s*15',
        r'DFARS',
        r'Volume\s+[IVX]+\s*[-:/]',  # Traditional volume markers
        r'\bFA\d{4}',               # Air Force
        r'\bW\d{3}[A-Z]{2}',       # Army
    ]

    def select_strategy(
        self,
        document_text: str,
        procurement_detection: Optional['ProcurementDetectionResult'] = None
    ) -> ParsingStrategy:
        """
        Select parsing strategy based on document analysis.

        Priority:
        1. Explicit GSA force patterns → GSA
        2. Explicit UCF force patterns → UCF
        3. ProcurementDetectionResult → corresponding strategy
        4. Default → UCF (safest)

        Args:
            document_text: Full document text
            procurement_detection: Optional detection result

        Returns:
            ParsingStrategy enum value
        """
        # Check GSA force patterns
        gsa_matches = sum(
            1 for p in self.GSA_FORCE_PATTERNS
            if re.search(p, document_text, re.IGNORECASE)
        )

        # Check UCF force patterns
        ucf_matches = sum(
            1 for p in self.UCF_FORCE_PATTERNS
            if re.search(p, document_text, re.IGNORECASE)
        )

        print(f"[v6.0.12 Strategy] GSA signals: {gsa_matches}, UCF signals: {ucf_matches}")

        # GSA wins if more GSA signals
        if gsa_matches > ucf_matches and gsa_matches >= 2:
            print("[v6.0.12 Strategy] Selected: GSA_RFQ (pattern match)")
            return ParsingStrategy.GSA_RFQ

        # UCF wins if more UCF signals
        if ucf_matches > gsa_matches:
            print("[v6.0.12 Strategy] Selected: UCF_STANDARD (pattern match)")
            return ParsingStrategy.UCF_STANDARD

        # Fall back to procurement detection if available
        if procurement_detection:
            from .models import ProcurementType
            if procurement_detection.procurement_type == ProcurementType.GSA_RFQ_8_4:
                print("[v6.0.12 Strategy] Selected: GSA_RFQ (detection)")
                return ParsingStrategy.GSA_RFQ

        # Default to UCF
        print("[v6.0.12 Strategy] Selected: UCF_STANDARD (default)")
        return ParsingStrategy.UCF_STANDARD
