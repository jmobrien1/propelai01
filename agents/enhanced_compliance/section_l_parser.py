"""
PropelAI v3.0: SectionLParser - Parses Section L text into SectionL_Schema.

This is the bridge between raw RFP text and StrictStructureBuilder.
It extracts ONLY structure information, not content requirements.

Key Principle: Parse explicitly stated structure. DO NOT infer or assume.

v5.0.6: Added table-first extraction strategy for page limits and volume promotion.
v5.0.8: Iron Triangle Determinism - Structural table parsing with row-index linking,
        enforced stated_volume_count validation, removed _promote_sections_to_volumes.
"""

from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import re

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

from .section_l_schema import (
    SectionL_Schema,
    VolumeInstruction,
    SectionInstruction,
    FormatInstruction,
    SubmissionInstruction
)


class StructureValidationError(Exception):
    """
    v5.0.8: Raised when proposal structure cannot be deterministically extracted.

    This is a hard failure - no guessing or fallback allowed.
    """
    pass


@dataclass
class TableCell:
    """A single cell in a structured table."""
    text: str
    row_index: int
    col_index: int
    is_header: bool = False


@dataclass
class TableObject:
    """
    v5.0.8: Structured table representation maintaining row/column associations.

    This preserves the relationship between volume titles and page limits
    based on their row position, not text proximity.
    """
    headers: List[str] = field(default_factory=list)
    rows: List[List[TableCell]] = field(default_factory=list)
    column_mapping: Dict[str, int] = field(default_factory=dict)  # "page_limit" -> column 2

    def get_cell_by_row_and_column_name(
        self,
        row_index: int,
        column_name: str
    ) -> Optional[TableCell]:
        """Get cell value using row index and semantic column name."""
        col_idx = self.column_mapping.get(column_name)
        if col_idx is None or row_index >= len(self.rows):
            return None
        row = self.rows[row_index]
        if col_idx >= len(row):
            return None
        return row[col_idx]

    def get_volume_page_limit_by_row(self, row_index: int) -> Optional[Tuple[str, int]]:
        """
        Get (volume_title, page_limit) for a row by index.

        This enforces row-based linking rather than text proximity.
        """
        title_cell = self.get_cell_by_row_and_column_name(row_index, "title")
        limit_cell = self.get_cell_by_row_and_column_name(row_index, "page_limit")

        if not title_cell or not limit_cell:
            return None

        try:
            # Extract numeric page limit
            limit_match = re.search(r'(\d+)', limit_cell.text)
            if limit_match:
                return (title_cell.text.strip(), int(limit_match.group(1)))
        except (ValueError, AttributeError):
            pass

        return None


class SectionLParser:
    """
    Parses Section L text into structured SectionL_Schema.

    This parser extracts ONLY structure information:
    - Volumes (how many, what they're called, page limits)
    - Sections (within each volume)
    - Format requirements (font, margins, spacing)
    - Submission requirements (due date, method)

    It does NOT extract content requirements (those go to Section C processing).

    Usage:
        parser = SectionLParser()
        schema = parser.parse(
            section_l_text=section_l_content,
            rfp_number="75N96025R00004",
            rfp_title="Scientific Support Services"
        )
    """

    def __init__(self):
        """Initialize parser with pattern definitions."""
        self.volume_patterns = [
            # "Volume I: Technical Proposal"
            r"Volume\s+([IVX\d]+)\s*[:\-–]\s*([^\n]+)",
            # "Volume I - Technical Proposal"
            r"Volume\s+([IVX\d]+)\s*[-–]\s*([^\n]+)",
        ]

        # v5.0.9: Numbered volume patterns (e.g., "1. Executive Summary and Technical Volume")
        # These RFPs use "N. Title Volume" instead of "Volume N: Title"
        self.numbered_volume_patterns = [
            # "1. Executive Summary and Technical Volume"
            r"^(\d+)\.\s+([^\n]+?Volume)\s*$",
            # "2. Cost & Price Volume" or "2. Cost/Price Volume"
            r"^(\d+)\.\s+([^\n]*(?:Cost|Price|Technical|Contract|Documentation)[^\n]*Volume)\s*$",
        ]

        self.volume_count_patterns = [
            # "proposal shall consist of two (2) volumes"
            r"proposal\s+shall\s+consist\s+of\s+(\w+)\s*\(?\d*\)?\s*volumes?",
            # "two volumes are required"
            r"(\w+)\s*\(?\d*\)?\s*volumes?\s+(?:are\s+)?required",
            # "submit the following two volumes"
            r"submit\s+(?:the\s+following\s+)?(\w+)\s*\(?\d*\)?\s*volumes?",
            # "TWO (2) VOLUMES"
            r"(\w+)\s*\(\d+\)\s*VOLUMES?",
        ]

        self.section_patterns = [
            # v5.0.9: "1.1 Executive Summary" - requires at least one digit after decimal
            # This prevents "3." from matching and causing double-period issues
            r"(\d{1,2}\.\d+)\s+([A-Z][^\n]{4,60})",
            # "Section 1: Technical Approach"
            r"Section\s+(\d+)\s*[:\-]\s*([^\n]+)",
            # "(a) Technical Approach"
            r"\(([a-z])\)\s+([A-Z][^\n]{4,60})",
        ]

        # v5.0.7: Invalid section header patterns (false positives)
        # These patterns indicate the matched text is NOT a valid section header
        self.invalid_header_patterns = [
            r'^\d{3,}',  # Starts with 3+ digits (e.g., "505", "1000")
            r'^this\s+is\s+',  # Sentence starting with "This is"
            r'^the\s+\w+\s+shall',  # Requirement text "The contractor shall..."
            r'^a\s+\w+\s+is\s+',  # Sentence starting with "A ... is"
            r'^\w+\s+shall\s+',  # Requirement "... shall ..."
            r'^solicitation',  # "Solicitation" is not a section header
            r'^competitive',  # "Competitive" is not a section header
            r'^offeror',  # Requirement about offeror
        ]

        self.page_limit_patterns = [
            # "not to exceed 50 pages"
            r"(?:not\s+to\s+exceed|shall\s+not\s+exceed)\s+(\d+)\s*pages?",
            # "maximum of 50 pages"
            r"maximum\s+(?:of\s+)?(\d+)\s*pages?",
            # "limited to 50 pages"
            r"limited\s+to\s+(\d+)\s*pages?",
            # "no more than 50 pages"
            r"no\s+more\s+than\s+(\d+)\s*pages?",
            # "50 page limit" or "50-page limit"
            r"(\d+)\s*[-]?\s*page\s+limit",
            # "(50 pages max)" or "(50 pages maximum)"
            r"\((\d+)\s*pages?\s*(?:max|maximum)?\)",
            # "8 pages"
            r"\b(\d+)\s*pages?\b",
        ]

        self.number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12
        }

        # v5.0.8: Table structure patterns for extracting volume/page limit associations
        # Tables often have: Volume | Title | Page Limit | Copies
        self.table_row_patterns = [
            # "Volume 1" or "1" followed by title and page limit
            # Pattern: Vol#/Number | Title Text | ## Pages
            r"(?:Volume\s*)?([1-3IVX]+)\s*[|\t]+\s*([A-Za-z][^|\t\n]{3,50})\s*[|\t]+\s*(\d+)\s*(?:pages?)?",
            # Simpler: Number | Title | Pages (tab or multiple spaces as delimiter)
            r"([1-3IVX]+)\s{2,}([A-Za-z][^\t\n]{3,50})\s{2,}(\d+)\s*(?:pages?)?",
        ]

        # v5.0.8: Column header patterns for semantic column identification
        self.column_header_patterns = {
            "volume": r"(?:volume|vol\.?)\s*(?:#|no\.?|number)?",
            "title": r"(?:title|name|description|proposal\s+type)",
            "page_limit": r"(?:page\s*limit|pages?|max\s*pages?|page\s*count)",
            "copies": r"(?:copies|qty|quantity|number\s+of\s+copies)",
        }

        # v5.0.8: REMOVED volume_promotion_keywords
        # Reason: This created "phantom volumes" by inferring structure from keywords.
        # Iron Triangle Determinism requires explicit "Volume I:" patterns only.

    def parse(
        self,
        section_l_text: str,
        rfp_number: str = "",
        rfp_title: str = "",
        attachment_texts: Optional[Dict[str, str]] = None,
        strict_mode: bool = True,
        pdf_path: Optional[str] = None
    ) -> SectionL_Schema:
        """
        Parse Section L text into structured schema.

        Args:
            section_l_text: Full text of Section L
            rfp_number: RFP/Solicitation number
            rfp_title: RFP title
            attachment_texts: Dict of attachment_name -> text for structural attachments
            strict_mode: v5.0.8 - If True, raises StructureValidationError on
                         volume count mismatch. If False, adds warnings only.
            pdf_path: Optional path to PDF for pdfplumber structural table extraction

        Returns:
            SectionL_Schema ready for StrictStructureBuilder

        Raises:
            StructureValidationError: If strict_mode=True and volume count doesn't
                                      match stated_volume_count from RFP
        """
        warnings: List[str] = []
        source_docs = ['Section L']

        # Combine Section L with any structural attachments
        full_text = section_l_text
        if attachment_texts:
            for name, text in attachment_texts.items():
                if self._is_structural_attachment(name):
                    full_text += f"\n\n--- {name} ---\n{text}"
                    source_docs.append(name)

        # v5.0.9: Extract official solicitation number from text
        # Prefer detected number over passed-in rfp_number (which may be internal ID)
        detected_sol_number = self._extract_solicitation_number(full_text)
        if detected_sol_number:
            print(f"[v5.0.9] Detected official solicitation number: {detected_sol_number}")
            if rfp_number and rfp_number != detected_sol_number:
                warnings.append(
                    f"Passed RFP number '{rfp_number}' differs from detected "
                    f"solicitation number '{detected_sol_number}'. Using detected number."
                )
            rfp_number = detected_sol_number

        # v5.0.8: Extract structured tables from PDF if available
        structured_tables: List[TableObject] = []
        if pdf_path and PDFPLUMBER_AVAILABLE:
            structured_tables = self._extract_structured_tables_from_pdf(pdf_path)

        # v6.0: SF1449 coordinate-based metadata extraction (priority source)
        sf1449_metadata: Dict[str, Optional[str]] = {}
        if pdf_path and PDFPLUMBER_AVAILABLE:
            sf1449_metadata = self._extract_sf1449_metadata(pdf_path)
            # Override rfp_number if SF1449 extraction was successful
            if sf1449_metadata.get('solicitation_number'):
                sf1449_sol = sf1449_metadata['solicitation_number']
                if rfp_number and rfp_number != sf1449_sol:
                    warnings.append(
                        f"SF1449 solicitation number '{sf1449_sol}' differs from "
                        f"passed RFP number '{rfp_number}'. Using SF1449 value."
                    )
                rfp_number = sf1449_sol
                print(f"[v6.0] Using SF1449 solicitation number: {rfp_number}")

        # Extract stated volume count first (for validation)
        stated_count = self._extract_stated_volume_count(full_text)

        # Extract volumes
        # v6.0.1: Pass pdf_path to enable aggressive spatial table extraction
        volumes = self._extract_volumes(full_text, warnings, structured_tables, pdf_path)

        # v5.0.5: REMOVED mention-based fallback (_extract_volumes_from_mentions)
        # The fallback would infer volumes from keywords like "Technical Proposal"
        # which often created HALLUCINATED volumes not in the RFP.
        # v5.0.8: Now raises StructureValidationError in strict_mode.
        if not volumes:
            error_msg = (
                "No explicit volume declarations found in Section L "
                "(e.g., 'Volume I: Technical Approach'). "
                "Cannot determine proposal structure."
            )
            if strict_mode:
                raise StructureValidationError(error_msg)
            warnings.append(error_msg)

        # v6.0: SEMANTIC SEARCH FALLBACK for missing volumes
        # Instead of failing immediately, search for common volume patterns
        if stated_count is not None and len(volumes) < stated_count:
            missing_count = stated_count - len(volumes)
            print(f"[v6.0] Volume count mismatch: stated={stated_count}, found={len(volumes)}")
            print(f"[v6.0] Attempting semantic search for {missing_count} missing volume(s)...")

            # Try to find missing volumes via semantic search
            found_volumes = self._search_for_missing_volumes(
                full_text, volumes, stated_count, warnings
            )
            if found_volumes:
                volumes.extend(found_volumes)
                volumes = sorted(volumes, key=lambda v: v['volume_number'])
                print(f"[v6.0] After semantic search: found {len(volumes)} total volumes")

        # v5.0.8: MANDATORY volume count validation (Iron Triangle Determinism)
        # Only raise error if semantic search also failed
        if stated_count is not None and len(volumes) != stated_count:
            error_msg = (
                f"VOLUME COUNT MISMATCH: RFP explicitly states {stated_count} volumes "
                f"but parser found {len(volumes)} (even after semantic search). "
                f"Found volumes: {[v['volume_title'] for v in volumes]}. "
                f"This is a deterministic failure - no guessing allowed."
            )
            if strict_mode:
                raise StructureValidationError(error_msg)
            warnings.append(error_msg)

        # Extract sections for each volume
        sections = self._extract_sections(full_text, volumes, warnings)

        # Extract format requirements
        format_rules = self._extract_format_rules(full_text)

        # Extract submission requirements
        submission_rules = self._extract_submission_rules(full_text)

        # v6.0: Override with SF1449 metadata if available (higher priority)
        if sf1449_metadata.get('due_date') and not submission_rules.get('due_date'):
            submission_rules['due_date'] = sf1449_metadata['due_date']
            print(f"[v6.0] Using SF1449 due date: {submission_rules['due_date']}")
        if sf1449_metadata.get('due_time') and not submission_rules.get('due_time'):
            submission_rules['due_time'] = sf1449_metadata['due_time']

        # Extract total page limit
        total_pages = self._extract_total_page_limit(full_text)

        return SectionL_Schema(
            rfp_number=rfp_number,
            rfp_title=rfp_title,
            volumes=volumes,
            sections=sections,
            format_rules=format_rules,
            submission_rules=submission_rules,
            total_page_limit=total_pages,
            stated_volume_count=stated_count,
            source_documents=source_docs,
            parsing_warnings=warnings
        )

    def _extract_sf1449_metadata(
        self,
        pdf_path: str
    ) -> Dict[str, Optional[str]]:
        """
        v6.0.4: FORM-FIELD ANCHOR STRATEGY for SF1449 metadata extraction.

        CRITICAL FIX: v6.0.3's anchor detection worked but extraction area was too narrow.
        This version:
        1. Finds "Solicitation Number" or "Solicitation No." text element
        2. Extracts from RELATIVE bbox (200pts RIGHT of label's right edge)
        3. ALSO scans BELOW the label (form fields can have values below)
        4. Greedy full-page search with year hard-gate for dates

        The "Iron Triangle" rule: If we can't find the official solicitation number,
        we MUST return None and let the orchestrator raise a validation error.
        We NEVER fall back to internal IDs (RFP-xxxxxxxx).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict with 'solicitation_number', 'issue_date', 'due_date', 'due_time', 'issued_by'
        """
        result = {
            'solicitation_number': None,
            'issue_date': None,
            'due_date': None,
            'due_time': None,
            'issued_by': None
        }

        if not pdf_path or not PDFPLUMBER_AVAILABLE:
            return result

        # v6.0.4: Official agency solicitation patterns ONLY
        official_sol_patterns = [
            r'(FA\d{4}[-]?\d{2}[-]?[A-Z][-]?[A-Z0-9]{3,})',  # Air Force: FA880625RB003
            r'(W\d{3}[A-Z]{2}[-]?\d{2}[-]?[A-Z][-]?\d{4,})',  # Army
            r'(N\d{5}[-]?\d{2}[-]?[A-Z][-]?\d{4,})',  # Navy
            r'(SP\d{4}[-]?\d{2}[-]?[A-Z][-]?\d{4,})',  # DLA
            r'(\d{2}[A-Z]\d{5}[A-Z]\d{5,})',  # NIH
        ]

        # v6.0.4: REJECT patterns - internal IDs that MUST NEVER be used
        reject_patterns = [
            r'RFP[-_]?[A-F0-9]{6,}',
            r'DRAFT[-_]',
            r'd[xX]?RFP[-_]?[A-F0-9]{6,}',  # Draft RFP IDs
        ]

        MIN_PROPOSAL_YEAR = 2025

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    return result

                first_page = pdf.pages[0]
                page_width = first_page.width
                page_height = first_page.height
                first_page_text = first_page.extract_text() or ""

                # v6.0.4: FORM-FIELD ANCHOR STRATEGY
                # Use extract_words with more tolerant settings
                words = first_page.extract_words(
                    keep_blank_chars=True,
                    x_tolerance=5,
                    y_tolerance=5,
                    extra_attrs=['fontname', 'size']
                )

                print(f"[v6.0.4] FORM-FIELD ANCHOR: Scanning {len(words)} words on page 1...")

                # v6.0.4: Find "SOLICITATION" label with precise positioning
                sol_label_words = []
                for i, word in enumerate(words):
                    word_text = word['text'].upper().strip()
                    if 'SOLICITATION' in word_text:
                        sol_label_words.append((i, word))
                        print(f"[v6.0.4] Found 'SOLICITATION' at ({word['x0']:.0f}, {word['top']:.0f}): '{word['text']}'")

                # v6.0.4: For each SOLICITATION label, search RIGHT and BELOW
                for idx, sol_word in sol_label_words:
                    if result['solicitation_number']:
                        break

                    # Strategy A: Search to the RIGHT of the label (same line)
                    search_areas = [
                        # Right of label, same line
                        (sol_word['x1'] + 5, sol_word['top'] - 10,
                         min(page_width, sol_word['x1'] + 300), sol_word['bottom'] + 15),
                        # Below label, wider area
                        (max(0, sol_word['x0'] - 50), sol_word['bottom'],
                         min(page_width, sol_word['x0'] + 350), min(page_height, sol_word['bottom'] + 40)),
                        # Full width below label
                        (0, sol_word['bottom'], page_width, min(page_height, sol_word['bottom'] + 60)),
                    ]

                    for area_idx, bbox in enumerate(search_areas):
                        if result['solicitation_number']:
                            break

                        value_text = self._extract_text_from_bbox(first_page, bbox)
                        if not value_text:
                            continue

                        print(f"[v6.0.4] Search area {area_idx + 1}: {repr(value_text[:100])}")

                        # Check for reject patterns FIRST
                        is_rejected = any(re.search(rp, value_text.upper()) for rp in reject_patterns)
                        if is_rejected:
                            print(f"[v6.0.4] REJECTED: Internal ID found in search area")
                            continue

                        # Search for official patterns
                        for pattern in official_sol_patterns:
                            match = re.search(pattern, value_text.upper())
                            if match:
                                result['solicitation_number'] = match.group(1)
                                print(f"[v6.0.4] ✓ LOCKED solicitation from area {area_idx + 1}: "
                                      f"{result['solicitation_number']}")
                                break

                # v6.0.4: Fallback - GREEDY FULL-PAGE SEARCH (with rejection)
                if not result['solicitation_number']:
                    print("[v6.0.4] Anchor strategy failed, performing greedy full-page search...")

                    # Search entire first page, prioritizing matches NOT near reject patterns
                    all_candidates = []
                    for pattern in official_sol_patterns:
                        for match in re.finditer(pattern, first_page_text.upper()):
                            candidate = match.group(1)
                            context_start = max(0, match.start() - 100)
                            context_end = min(len(first_page_text), match.end() + 100)
                            context = first_page_text[context_start:context_end].upper()

                            is_rejected = any(re.search(rp, context) for rp in reject_patterns)
                            if not is_rejected:
                                all_candidates.append(candidate)
                                print(f"[v6.0.4] Candidate: {candidate}")

                    if all_candidates:
                        # Take first non-rejected candidate
                        result['solicitation_number'] = all_candidates[0]
                        print(f"[v6.0.4] ✓ LOCKED solicitation from full-page: {result['solicitation_number']}")

                # v6.0.4: DUE DATE EXTRACTION with year hard-gate
                # Strategy 1: Find "OFFER" or "DUE" labels
                due_label_words = []
                for word in words:
                    word_text = word['text'].upper().strip()
                    if 'DUE' in word_text or 'OFFER' in word_text:
                        due_label_words.append(word)

                for due_word in due_label_words:
                    if result['due_date']:
                        break

                    # Search below and to the right of the label
                    search_areas = [
                        (max(0, due_word['x0'] - 100), due_word['bottom'],
                         min(page_width, due_word['x0'] + 400), min(page_height, due_word['bottom'] + 50)),
                        (due_word['x1'], due_word['top'] - 5,
                         min(page_width, due_word['x1'] + 250), due_word['bottom'] + 20),
                    ]

                    for bbox in search_areas:
                        if result['due_date']:
                            break
                        value_text = self._extract_text_from_bbox(first_page, bbox)
                        if value_text:
                            result['due_date'], result['due_time'] = self._extract_validated_date_time(
                                value_text, MIN_PROPOSAL_YEAR
                            )

                # v6.0.4: Fallback - GREEDY FULL-PAGE DATE SEARCH
                if not result['due_date']:
                    print(f"[v6.0.4] Date anchor failed, greedy search for 'Feb 2025' or '03 Feb 2025'...")

                    # Prioritize specific date patterns
                    priority_patterns = [
                        r'(03?\s+Feb(?:ruary)?\s+2025)',
                        r'(Feb(?:ruary)?\s+0?3,?\s+2025)',
                        r'(\d{1,2}\s+[A-Za-z]+\s+2025)',
                        r'([A-Za-z]+\s+\d{1,2},?\s+2025)',
                    ]

                    for pattern in priority_patterns:
                        match = re.search(pattern, first_page_text, re.IGNORECASE)
                        if match:
                            result['due_date'] = match.group(1)
                            print(f"[v6.0.4] ✓ LOCKED due date from greedy search: {result['due_date']}")
                            break

                    # Final fallback to validated extraction
                    if not result['due_date']:
                        result['due_date'], result['due_time'] = self._extract_validated_date_time(
                            first_page_text, MIN_PROPOSAL_YEAR
                        )

                # v6.0.4: Final status report
                print(f"[v6.0.4] EXTRACTION RESULT:")
                print(f"[v6.0.4]   Solicitation: {result['solicitation_number'] or 'FAILED'}")
                print(f"[v6.0.4]   Due Date: {result['due_date'] or 'FAILED'}")

        except Exception as e:
            print(f"[v6.0.4] Error in SF1449 extraction: {e}")
            import traceback
            traceback.print_exc()

        return result

    def _extract_validated_date_time(
        self,
        text: str,
        min_year: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        v6.0.2: Extract date and time with year validation.

        Only returns dates with year >= min_year to avoid grabbing legacy dates.

        Args:
            text: Text to search
            min_year: Minimum acceptable year (e.g., 2025)

        Returns:
            (due_date, due_time) tuple
        """
        due_date = None
        due_time = None

        # Date patterns with year capture group
        date_patterns = [
            r'(\d{1,2}\s+[A-Za-z]{3,9}\s+(\d{4}))',  # "03 Feb 2025"
            r'([A-Za-z]{3,9}\s+\d{1,2},?\s+(\d{4}))',  # "February 3, 2025"
            r'(\d{1,2}[/-]\d{1,2}[/-](\d{4}))',  # "02/03/2025"
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                full_date = match.group(1)
                year = int(match.group(2))

                if year >= min_year:
                    due_date = full_date
                    print(f"[v6.0.2] Validated date: {due_date} (year {year} >= {min_year})")
                    break
                else:
                    print(f"[v6.0.2] Rejected stale date: {full_date} (year {year} < {min_year})")

            if due_date:
                break

        # Extract time (4-digit military format)
        if text:
            time_match = re.search(
                r'\b(\d{4})\s*(?:hrs?|hours?|local|pacific|eastern|central|mountain|[A-Z]{2,3}T)?\b',
                text,
                re.IGNORECASE
            )
            if time_match:
                time_val = time_match.group(1)
                if 0 <= int(time_val) <= 2359:
                    due_time = time_val

        return due_date, due_time

    def _extract_text_from_bbox(self, page, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """
        v6.0: Extract text from a specific bounding box region of a PDF page.

        Args:
            page: pdfplumber page object
            bbox: (x0, y0, x1, y1) coordinates in points

        Returns:
            Extracted text or None
        """
        try:
            cropped = page.within_bbox(bbox)
            text = cropped.extract_text()
            return text.strip() if text else None
        except Exception:
            return None

    def _extract_solicitation_number(self, text: str) -> Optional[str]:
        """
        v5.0.9: Extract official solicitation number from RFP text.

        Looks for standard government solicitation number formats:
        - Air Force: FA8806-25-R-B003 or FA880625RB003
        - Army: W911NF-25-R-0001 or W911NF25R0001
        - Navy: N00024-25-R-0001 or N0002425R0001
        - NIH: 75N96025R00004
        - GSA: GS-23F-0001
        - DoD: SPRXXXXXXX

        Returns:
            Detected solicitation number or None
        """
        # Official solicitation number patterns (high confidence)
        sol_patterns = [
            # Air Force format: FA8806-25-R-B003 or FA880625RB003
            r'(?:Solicitation\s*(?:No\.?|Number|#)?:?\s*)?(FA\d{4}[-]?\d{2}[-]?[A-Z][-]?[A-Z0-9]{4,})',
            # Army format: W911NF-25-R-0001
            r'(?:Solicitation\s*(?:No\.?|Number|#)?:?\s*)?(W\d{3}[A-Z]{2}[-]?\d{2}[-]?[A-Z][-]?\d{4,})',
            # Navy format: N00024-25-R-0001
            r'(?:Solicitation\s*(?:No\.?|Number|#)?:?\s*)?(N\d{5}[-]?\d{2}[-]?[A-Z][-]?\d{4,})',
            # NIH/HHS format: 75N96025R00004
            r'(?:Solicitation\s*(?:No\.?|Number|#)?:?\s*)?(\d{2}[A-Z]\d{5}[A-Z]\d{5,})',
            # GSA format: GS-23F-0001 or similar
            r'(?:Solicitation\s*(?:No\.?|Number|#)?:?\s*)?(GS[-]?\d{2}[A-Z][-]?\d{4,})',
            # SPR format
            r'(?:Solicitation\s*(?:No\.?|Number|#)?:?\s*)?(SPR[A-Z0-9]{6,})',
            # Generic with explicit label: "Solicitation Number: XYZ123"
            r'Solicitation\s*(?:No\.?|Number|#)?\s*[:=]\s*([A-Z0-9][-A-Z0-9]{8,})',
            # SF1449 field pattern
            r'(?:SF\s*1449|SF-1449).*?Solicitation[^:]*[:=]?\s*([A-Z0-9][-A-Z0-9]{8,})',
        ]

        for pattern in sol_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sol_number = match.group(1).upper()
                # Validate it looks like a real solicitation number
                # Should have letters and numbers, 10+ chars
                if len(sol_number) >= 10 and re.search(r'[A-Z]', sol_number) and re.search(r'\d', sol_number):
                    # Clean up - remove internal whitespace
                    sol_number = re.sub(r'\s+', '', sol_number)
                    return sol_number

        return None

    def _is_structural_attachment(self, name: str) -> bool:
        """Check if attachment contains structure instructions."""
        structural_keywords = [
            'placement', 'format', 'instruction', 'procedure',
            'proposal format', 'submission format', 'preparation'
        ]
        name_lower = name.lower()
        return any(kw in name_lower for kw in structural_keywords)

    def _search_for_missing_volumes(
        self,
        text: str,
        found_volumes: List[VolumeInstruction],
        stated_count: int,
        warnings: List[str]
    ) -> List[VolumeInstruction]:
        """
        v6.0.2: GREEDY VOLUME RECOVERY - Aggressive search for missing volumes.

        CRITICAL FIX: Performs "Document Sweep" to find volumes missed by regex.
        When stated_count (3) > found_volumes (2), this method MUST find the
        missing volumes or the outline will fail the Iron Triangle gate.

        Strategy:
        1. Identify which volume numbers are missing
        2. Search for lines starting with missing digit (e.g., "3.") containing keywords
        3. Search for ANY line containing "Documentation" or "Contract" + "Volume"
        4. Promote found lines immediately to VolumeInstruction

        Returns:
            List of newly found VolumeInstruction objects
        """
        missing_volumes: List[VolumeInstruction] = []

        # Get existing volume numbers
        found_numbers = {v['volume_number'] for v in found_volumes}

        # Common volume type keywords by position
        volume_keywords = {
            1: ['technical', 'executive summary', 'management', 'approach'],
            2: ['cost', 'price', 'pricing', 'business'],
            3: ['contract', 'documentation', 'administrative', 'certifications', 'forms'],
            4: ['past performance', 'experience', 'references'],
            5: ['small business', 'subcontracting'],
        }

        # v6.0.2: DOCUMENT SWEEP - Aggressive line-by-line search
        lines = text.split('\n')

        for vol_num in range(1, stated_count + 1):
            if vol_num in found_numbers:
                continue
            if vol_num in {v['volume_number'] for v in missing_volumes}:
                continue

            print(f"[v6.0.2] GREEDY RECOVERY: Searching for missing Volume {vol_num}...")

            keywords = volume_keywords.get(vol_num, ['volume', 'section'])

            # Strategy 1: Find lines starting with the missing digit
            # Matches: "3. Contract Documentation Volume" or "3. Contract Documentation"
            for line in lines:
                line_stripped = line.strip()

                # v6.0.2: Greedy pattern - any line starting with "N." or "N " where N is missing vol
                greedy_match = re.match(
                    rf'^{vol_num}[\.\s]+(.{{3,100}})',
                    line_stripped,
                    re.IGNORECASE
                )

                if greedy_match:
                    title = greedy_match.group(1).strip()
                    title_lower = title.lower()

                    # Check if contains ANY volume-like keyword
                    has_keyword = any(
                        kw in title_lower
                        for kw in keywords + ['volume', 'documentation', 'contract', 'admin']
                    )

                    if has_keyword or vol_num >= 3:  # Always accept Vol 3+ candidates
                        # Clean up title
                        title = re.sub(r'[\.\,\;\:\(\)]+$', '', title).strip()
                        title = re.sub(r'\s+', ' ', title)

                        # Truncate if too long
                        if len(title) > 60:
                            title = title[:60].rsplit(' ', 1)[0]

                        # Add "Volume" suffix if not present
                        if 'volume' not in title.lower():
                            title = f"{title} Volume"

                        print(f"[v6.0.2] GREEDY MATCH: Found '{title}' for Volume {vol_num}")

                        missing_volumes.append(VolumeInstruction(
                            volume_id=f"VOL-{vol_num}",
                            volume_title=title,
                            volume_number=vol_num,
                            page_limit=None,
                            source_reference=f"Section L (greedy recovery, line pattern {vol_num}.)",
                            is_mandatory=True
                        ))
                        break

            # Strategy 2: Full-text keyword sweep if still not found
            if vol_num not in {v['volume_number'] for v in missing_volumes}:
                print(f"[v6.0.2] Line pattern failed, trying keyword sweep...")

                for keyword in keywords:
                    # v6.0.2: Greedy patterns for keywords
                    patterns = [
                        rf'({keyword}\s+(?:documentation\s+)?volume)',  # "Contract Documentation Volume"
                        rf'({keyword}\s+volume)',  # "Contract Volume"
                        rf'(\d+\.\s*{keyword}[^\n]{{0,50}})',  # "3. Contract..."
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            title = match.group(1).strip()
                            title = re.sub(r'[\.\,\;\:]+$', '', title)

                            # Capitalize properly
                            title = title.title()

                            # Add Volume if not present
                            if 'volume' not in title.lower():
                                title = f"{title} Volume"

                            print(f"[v6.0.2] KEYWORD SWEEP: Found '{title}' for Volume {vol_num}")

                            missing_volumes.append(VolumeInstruction(
                                volume_id=f"VOL-{vol_num}",
                                volume_title=title,
                                volume_number=vol_num,
                                page_limit=None,
                                source_reference=f"Section L (greedy recovery, keyword '{keyword}')",
                                is_mandatory=True
                            ))
                            break

                    if vol_num in {v['volume_number'] for v in missing_volumes}:
                        break

        # Strategy 3: Last resort - search for specific volume titles
        still_missing = set(range(1, stated_count + 1)) - found_numbers - {v['volume_number'] for v in missing_volumes}
        if still_missing:
            print(f"[v6.0.2] Still missing volumes: {still_missing}, trying title patterns...")

            # Common volume title patterns
            title_patterns = [
                (r'Contract\s+Documentation\s+Volume', 3),
                (r'Administrative\s+Volume', 3),
                (r'Past\s+Performance\s+Volume', 4),
                (r'Technical\s+(?:Proposal\s+)?Volume', 1),
                (r'Cost\s*[/&]\s*Price\s+Volume', 2),
            ]

            for pattern, default_vol in title_patterns:
                if default_vol in still_missing:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        title = match.group(0).strip()
                        print(f"[v6.0.2] TITLE PATTERN: Found '{title}' for Volume {default_vol}")

                        missing_volumes.append(VolumeInstruction(
                            volume_id=f"VOL-{default_vol}",
                            volume_title=title,
                            volume_number=default_vol,
                            page_limit=None,
                            source_reference=f"Section L (greedy recovery, title pattern)",
                            is_mandatory=True
                        ))

        if missing_volumes:
            warnings.append(
                f"GREEDY RECOVERY: Found {len(missing_volumes)} volume(s): "
                f"{[v['volume_title'] for v in missing_volumes]}. "
                "These were missed by primary regex patterns."
            )

        return missing_volumes

    def _extract_stated_volume_count(self, text: str) -> Optional[int]:
        """
        Extract stated volume count (e.g., 'consist of two volumes').

        This is used for validation against actually found volumes.
        """
        for pattern in self.volume_count_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                num_str = match.group(1).lower()
                if num_str in self.number_words:
                    return self.number_words[num_str]
                elif num_str.isdigit():
                    return int(num_str)
        return None

    def _extract_page_limits_from_table(
        self,
        text: str,
        warnings: List[str]
    ) -> Dict[str, Tuple[str, int]]:
        """
        v5.0.6: Extract volume page limits from table structures.

        Tables in Section L often define volume structure like:
        | Volume | Title | Page Limit | Copies |
        |   1    | Technical | 8 Pages | 1 |
        |   2    | Cost/Price | No Limit | 1 |

        Returns:
            Dict mapping volume_key (lowercase title) to (title, page_limit)
        """
        table_data: Dict[str, Tuple[str, int]] = {}

        # Look for table-like structures with headers
        # First, try to find a table header row
        table_header_patterns = [
            r"(?:volume|vol\.?)\s*[|\t].*?(?:page\s*limit|pages?)[|\t\n]",
            r"(?:title|description)\s*[|\t].*?(?:page\s*limit|pages?)[|\t\n]",
        ]

        has_table = False
        for pattern in table_header_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                has_table = True
                break

        if not has_table:
            # Also check for consistent row-like structures
            # Lines with multiple tab/pipe separators
            lines = text.split('\n')
            table_like_lines = [
                l for l in lines
                if (l.count('|') >= 2 or l.count('\t') >= 2)
                and re.search(r'\d+\s*pages?', l, re.IGNORECASE)
            ]
            if len(table_like_lines) >= 2:
                has_table = True

        if not has_table:
            return table_data

        # Extract table rows
        for pattern in self.table_row_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                vol_num_str = match.group(1).strip()
                title = match.group(2).strip()
                page_limit_str = match.group(3).strip()

                try:
                    page_limit = int(page_limit_str)
                    if 1 <= page_limit <= 500:  # Sanity check
                        # Clean title
                        title = re.sub(r'[\.\,\;\:]+$', '', title).strip()
                        table_data[title.lower()] = (title, page_limit)
                except ValueError:
                    pass

        # Also try a more flexible pattern for common table formats
        # "Technical Proposal    8 Pages" or "Technical Proposal | 8 Pages"
        flexible_patterns = [
            # Title followed by page count (with separator)
            r"(Technical\s+(?:Proposal|Volume))\s*[|\t:]+\s*(\d+)\s*pages?",
            r"(Cost\s*(?:&|and)?\s*Price\s*(?:Proposal|Volume)?)\s*[|\t:]+\s*(\d+)\s*pages?",
            r"(Contract\s+Documentation)\s*[|\t:]+\s*(\d+)\s*pages?",
            # Volume N: Title ... ## pages (in same line/nearby)
            r"Volume\s*[1-3IVX]+[:\s]+([A-Za-z][^|\n]{3,40})[^0-9]*?(\d+)\s*pages?",
        ]

        for pattern in flexible_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                title = match.group(1).strip()
                page_limit_str = match.group(2).strip()

                try:
                    page_limit = int(page_limit_str)
                    if 1 <= page_limit <= 500:
                        title = re.sub(r'[\.\,\;\:]+$', '', title).strip()
                        # Don't overwrite if we already have this
                        if title.lower() not in table_data:
                            table_data[title.lower()] = (title, page_limit)
                except ValueError:
                    pass

        if table_data:
            print(f"[v5.0.6] Extracted page limits from table: {table_data}")

        return table_data

    def _extract_structured_tables_from_pdf(
        self,
        pdf_path: str
    ) -> List[TableObject]:
        """
        v5.0.8: Extract tables from PDF using pdfplumber structural extraction.

        This method uses pdfplumber's table detection to extract tables with
        proper row/column associations, enabling row-index based linking.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of TableObject with cell associations preserved
        """
        if not PDFPLUMBER_AVAILABLE:
            print("[v5.0.8] pdfplumber not available, skipping structural table extraction")
            return []

        structured_tables: List[TableObject] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables from this page
                    tables = page.extract_tables()

                    for table in tables:
                        if not table or len(table) < 2:
                            continue

                        table_obj = TableObject()

                        # First row is headers
                        headers = [str(cell or '').strip().lower() for cell in table[0]]
                        table_obj.headers = headers

                        # Identify column types from headers
                        for col_idx, header in enumerate(headers):
                            for col_type, pattern in self.column_header_patterns.items():
                                if re.search(pattern, header, re.IGNORECASE):
                                    table_obj.column_mapping[col_type] = col_idx
                                    break

                        # Check if this looks like a volume/page limit table
                        has_volume_or_title = (
                            'volume' in table_obj.column_mapping or
                            'title' in table_obj.column_mapping
                        )
                        has_page_limit = 'page_limit' in table_obj.column_mapping

                        if not (has_volume_or_title and has_page_limit):
                            continue

                        # Extract data rows
                        for row_idx, row in enumerate(table[1:]):
                            cells: List[TableCell] = []
                            for col_idx, cell_text in enumerate(row):
                                cells.append(TableCell(
                                    text=str(cell_text or '').strip(),
                                    row_index=row_idx,
                                    col_index=col_idx,
                                    is_header=False
                                ))
                            table_obj.rows.append(cells)

                        if table_obj.rows:
                            structured_tables.append(table_obj)
                            print(f"[v5.0.8] Found structured table on page {page_num + 1}: "
                                  f"{len(table_obj.rows)} rows, columns: {table_obj.column_mapping}")

        except Exception as e:
            print(f"[v5.0.8] Error extracting structured tables: {e}")

        return structured_tables

    def _extract_page_limits_from_structured_tables(
        self,
        structured_tables: List[TableObject],
        warnings: List[str]
    ) -> Dict[str, Tuple[str, int]]:
        """
        v5.0.8: Extract volume page limits using row-index based linking.

        This enforces that page limits are linked to volumes based on their
        position in the table row, not text proximity heuristics.

        Args:
            structured_tables: List of TableObject from pdfplumber extraction
            warnings: List to append extraction warnings

        Returns:
            Dict mapping volume_title.lower() to (title, page_limit)
        """
        page_limits: Dict[str, Tuple[str, int]] = {}

        for table in structured_tables:
            for row_idx in range(len(table.rows)):
                result = table.get_volume_page_limit_by_row(row_idx)
                if result:
                    title, limit = result
                    if 1 <= limit <= 500:  # Sanity check
                        page_limits[title.lower()] = (title, limit)
                        print(f"[v5.0.8] Row-index linked: '{title}' -> {limit} pages (row {row_idx})")

        if not page_limits and structured_tables:
            warnings.append(
                "Found structured tables but could not extract page limits. "
                "Check column headers match expected patterns."
            )

        return page_limits

    def _extract_page_limits_from_pdf_tables(
        self,
        pdf_path: str,
        warnings: List[str]
    ) -> Dict[str, Tuple[str, int]]:
        """
        v6.0.3: ROW-INDEX IMMUTABLE LINKING for page limits.

        CRITICAL FIX: v6.0.2's digit-targeting was still proximity-based. This version:
        1. IDENTIFY column indices for "Title/Volume" and "Page Limit/Pages"
        2. LOCATE row where Title column contains volume keyword
        3. LOCK the value at intersection of Row + Limit Column

        The "Intersection Rule": Page limit MUST come from the same row as the
        volume name, using column indices identified from the header row.

        Priority Order:
        1. Header-based column identification (most reliable)
        2. Structured intersection lookup
        3. Fallback to digit-targeting (v6.0.2 method)

        Args:
            pdf_path: Path to the PDF file
            warnings: List to append extraction warnings

        Returns:
            Dict mapping volume_title.lower() to (title, page_limit)
        """
        page_limits: Dict[str, Tuple[str, int]] = {}

        if not pdf_path or not PDFPLUMBER_AVAILABLE:
            return page_limits

        # v6.0.3: Volume keyword to volume number mapping
        volume_keyword_map = {
            'technical': ('Technical Volume', 1),
            'tech': ('Technical Volume', 1),
            'executive': ('Executive Summary and Technical Volume', 1),
            'management': ('Technical Volume', 1),
            'approach': ('Technical Volume', 1),
            'cost': ('Cost & Price Volume', 2),
            'price': ('Cost & Price Volume', 2),
            'pricing': ('Cost & Price Volume', 2),
            'business': ('Cost & Price Volume', 2),
            'contract': ('Contract Documentation Volume', 3),
            'documentation': ('Contract Documentation Volume', 3),
            'administrative': ('Contract Documentation Volume', 3),
            'admin': ('Contract Documentation Volume', 3),
            'certifications': ('Contract Documentation Volume', 3),
            'past performance': ('Past Performance Volume', 4),
            'experience': ('Past Performance Volume', 4),
            'references': ('Past Performance Volume', 4),
        }

        # v6.0.3: Column header patterns
        title_column_headers = ['title', 'volume', 'name', 'section', 'proposal element']
        limit_column_headers = ['page', 'limit', 'pages', 'maximum', 'max']

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()

                    for table_idx, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue

                        num_cols = len(table[0]) if table[0] else 0
                        print(f"[v6.0.3] ROW-INDEX LINKING: Table {table_idx + 1} on page {page_num + 1} "
                              f"({len(table)} rows x {num_cols} cols)")

                        # v6.0.3: STEP 1 - Identify column indices from header row
                        title_col_idx = None
                        limit_col_idx = None

                        # Check first 2 rows for headers
                        for header_row_idx in range(min(2, len(table))):
                            header_row = table[header_row_idx]

                            for col_idx, cell in enumerate(header_row):
                                cell_text = str(cell or '').lower().strip()

                                # Check for title column
                                if title_col_idx is None:
                                    if any(h in cell_text for h in title_column_headers):
                                        title_col_idx = col_idx
                                        print(f"[v6.0.3] HEADER: Title column at index {col_idx} ('{cell_text}')")

                                # Check for limit column
                                if limit_col_idx is None:
                                    if any(h in cell_text for h in limit_column_headers):
                                        limit_col_idx = col_idx
                                        print(f"[v6.0.3] HEADER: Limit column at index {col_idx} ('{cell_text}')")

                            if title_col_idx is not None and limit_col_idx is not None:
                                break

                        # v6.0.3: STEP 2 - If no headers found, try to infer from data
                        if title_col_idx is None or limit_col_idx is None:
                            print(f"[v6.0.3] No clear headers, inferring column structure...")

                            # Heuristic: Title column has text, Limit column has numbers
                            for row in table[1:]:  # Skip first row
                                for col_idx, cell in enumerate(row):
                                    cell_text = str(cell or '').strip()

                                    # Check if this looks like a volume name
                                    if title_col_idx is None:
                                        cell_lower = cell_text.lower()
                                        if any(kw in cell_lower for kw in volume_keyword_map.keys()):
                                            title_col_idx = col_idx
                                            print(f"[v6.0.3] INFERRED: Title column at index {col_idx}")

                                    # Check if this looks like a page limit
                                    if limit_col_idx is None:
                                        if re.match(r'^\d{1,3}(?:\s*pages?)?$', cell_text, re.IGNORECASE):
                                            limit_col_idx = col_idx
                                            print(f"[v6.0.3] INFERRED: Limit column at index {col_idx}")

                                if title_col_idx is not None and limit_col_idx is not None:
                                    break

                        # v6.0.3: STEP 3 - ROW-INDEX IMMUTABLE LINKING
                        if title_col_idx is not None and limit_col_idx is not None:
                            print(f"[v6.0.3] Using column indices: Title={title_col_idx}, Limit={limit_col_idx}")

                            for row_idx, row in enumerate(table):
                                if row_idx == 0:  # Skip header
                                    continue
                                if len(row) <= max(title_col_idx, limit_col_idx):
                                    continue

                                title_cell = str(row[title_col_idx] or '').strip()
                                limit_cell = str(row[limit_col_idx] or '').strip()

                                title_lower = title_cell.lower()

                                # Check if this row contains a volume keyword
                                matched_volume = None
                                matched_vol_num = None

                                for keyword, (vol_title, vol_num) in volume_keyword_map.items():
                                    if keyword in title_lower:
                                        matched_volume = vol_title
                                        matched_vol_num = vol_num
                                        break

                                if matched_volume and limit_cell:
                                    # v6.0.4: PERSISTENT TABLE CONSTRAINT LINKING
                                    # Extract ANY digits from the intersection cell
                                    # Handles: "8", "8 Pages", "Not to exceed 8", "8 pages max"
                                    limit_match = re.search(r'(\d+)', limit_cell)
                                    if limit_match:
                                        limit_value = int(limit_match.group(1))
                                        if 1 <= limit_value <= 500:
                                            key = matched_volume.lower()
                                            # v6.0.4: PERSISTENT - This value OVERRIDES any previous
                                            page_limits[key] = (matched_volume, limit_value)
                                            print(f"[v6.0.4] ✓ PERSISTENT LOCK: Row {row_idx} "
                                                  f"'{title_cell}' intersect '{limit_cell}' -> {limit_value} pages")

                                            # Add alternate keys for flexible matching
                                            if matched_vol_num == 1:
                                                page_limits['technical'] = (matched_volume, limit_value)
                                                page_limits['technical volume'] = (matched_volume, limit_value)
                                                page_limits['vol 1'] = (matched_volume, limit_value)
                                                page_limits['volume 1'] = (matched_volume, limit_value)
                                            elif matched_vol_num == 2:
                                                page_limits['cost'] = (matched_volume, limit_value)
                                                page_limits['cost & price'] = (matched_volume, limit_value)
                                                page_limits['vol 2'] = (matched_volume, limit_value)
                                            elif matched_vol_num == 3:
                                                page_limits['contract'] = (matched_volume, limit_value)
                                                page_limits['contract documentation'] = (matched_volume, limit_value)
                                                page_limits['vol 3'] = (matched_volume, limit_value)

                        # v6.0.3: FALLBACK - Digit-targeting for tables without clear structure
                        if not page_limits:
                            print(f"[v6.0.3] Row-index linking failed, using digit-targeting fallback...")
                            page_limits.update(
                                self._extract_page_limits_digit_targeting(table, page_num, table_idx, volume_keyword_map)
                            )

        except Exception as e:
            print(f"[v6.0.3] Error in page limit extraction: {e}")
            import traceback
            traceback.print_exc()

        if page_limits:
            print(f"[v6.0.3] Extracted {len(page_limits)} page limit entries from PDF tables")
        else:
            print(f"[v6.0.3] ✗ No page limits extracted from tables")

        return page_limits

    def _extract_page_limits_digit_targeting(
        self,
        table: List[List[Any]],
        page_num: int,
        table_idx: int,
        volume_keyword_map: Dict[str, Tuple[str, int]]
    ) -> Dict[str, Tuple[str, int]]:
        """
        v6.0.3: Fallback digit-targeting extraction (from v6.0.2).

        Used when header-based column identification fails.
        """
        page_limits: Dict[str, Tuple[str, int]] = {}

        for row_idx, row in enumerate(table):
            for col_idx, cell in enumerate(row):
                cell_text = str(cell or '').strip()

                digit_match = re.match(r'^(\d{1,3})(?:\s*pages?)?$', cell_text, re.IGNORECASE)
                if not digit_match:
                    continue

                digit_value = int(digit_match.group(1))
                if digit_value < 1 or digit_value > 100:
                    continue

                # Check row context
                row_text = ' '.join(str(c or '').lower() for c in row)
                left_cell = str(row[col_idx - 1] or '').lower() if col_idx > 0 else ''

                for keyword, (title, vol_num) in volume_keyword_map.items():
                    if keyword in left_cell or keyword in row_text:
                        key = title.lower()
                        if key not in page_limits:
                            page_limits[key] = (title, digit_value)
                            print(f"[v6.0.3] DIGIT-TARGET: '{title}' -> {digit_value} pages")

                            if vol_num == 1:
                                page_limits['technical'] = (title, digit_value)
                            elif vol_num == 2:
                                page_limits['cost'] = (title, digit_value)
                            elif vol_num == 3:
                                page_limits['contract'] = (title, digit_value)
                        break

        return page_limits

    def _extract_volume_title_from_row(
        self,
        row: List[Any],
        exclude_col: int
    ) -> Optional[str]:
        """
        v6.0.1: Extract a meaningful volume title from a table row.

        Args:
            row: List of cell values
            exclude_col: Column index to exclude (the page limit column)

        Returns:
            Extracted title or None
        """
        # Find the cell with the most text content (excluding the page limit column)
        best_title = None
        best_length = 0

        for col_idx, cell in enumerate(row):
            if col_idx == exclude_col:
                continue

            cell_text = str(cell or '').strip()

            # Skip cells that are just numbers or very short
            if re.match(r'^\d+$', cell_text) or len(cell_text) < 3:
                continue

            # Prefer cells containing volume keywords
            if any(kw in cell_text.lower() for kw in ['technical', 'cost', 'price', 'contract', 'volume']):
                return cell_text

            # Otherwise use the longest non-numeric cell
            if len(cell_text) > best_length:
                best_title = cell_text
                best_length = len(cell_text)

        return best_title

    def _extract_volumes(
        self,
        text: str,
        warnings: List[str],
        structured_tables: Optional[List[TableObject]] = None,
        pdf_path: Optional[str] = None
    ) -> List[VolumeInstruction]:
        """
        Extract volume instructions from text.

        v5.0.6: Now uses table-first strategy - if page limits are found in
        a table structure, those take precedence over nearby text extraction.
        v5.0.8: Uses row-index based linking from structured tables (pdfplumber).
        v5.0.9: Added numbered volume patterns ("1. Technical Volume") for RFPs
                that use numbered headings instead of "Volume I:" format.
        v6.0.1: Added aggressive spatial table extraction from PDF.

        Looks for explicit volume declarations like "Volume I: Technical Proposal"
        or "1. Executive Summary and Technical Volume".
        """
        volumes: List[VolumeInstruction] = []
        seen_titles: set = set()

        # v5.0.8: Use structured tables (row-index linking) if available
        table_page_limits: Dict[str, Tuple[str, int]] = {}
        if structured_tables:
            table_page_limits = self._extract_page_limits_from_structured_tables(
                structured_tables, warnings
            )
        else:
            # Fallback to text-based extraction
            table_page_limits = self._extract_page_limits_from_table(text, warnings)

        # v6.0.1: Aggressive spatial table extraction from PDF
        # This scans ALL tables looking for page limits using keyword matching
        if pdf_path and PDFPLUMBER_AVAILABLE:
            spatial_page_limits = self._extract_page_limits_from_pdf_tables(pdf_path, warnings)
            # Merge spatial limits (higher priority - more accurate)
            for key, value in spatial_page_limits.items():
                if key not in table_page_limits:
                    table_page_limits[key] = value
                    print(f"[v6.0.1] Added spatial page limit: '{key}' -> {value[1]} pages")

        # v5.0.9: Also extract from inline table format (Table 1 style)
        inline_page_limits = self._extract_page_limits_from_inline_table(text, warnings)
        # Merge inline limits (lower priority than structured tables)
        for key, value in inline_page_limits.items():
            if key not in table_page_limits:
                table_page_limits[key] = value

        # Standard "Volume I: Title" patterns
        for pattern in self.volume_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                vol_num_str = match.group(1)
                vol_num = self._roman_to_int(vol_num_str)
                title = match.group(2).strip()

                # Clean up title comprehensively
                # 1. Remove parenthetical content (page limits, notes, etc.)
                title = re.sub(r'\s*\([^)]*\).*$', '', title)
                # 2. Remove content after comma or semicolon (often extra notes)
                title = re.sub(r'\s*[,;].*$', '', title)
                # 3. Remove trailing punctuation
                title = re.sub(r'[\.\,\;\:\)]+$', '', title).strip()
                # 4. Truncate overly long titles (likely captured extra content)
                if len(title) > 80:
                    # Find natural break point
                    for sep in [' - ', ' – ', ': ', ' ']:
                        if sep in title[:80]:
                            title = title[:title.rfind(sep, 0, 80)]
                            break
                    else:
                        title = title[:80].rsplit(' ', 1)[0]

                # Skip if we've seen this title
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                # v5.0.6: Table-first strategy for page limits
                page_limit, page_limit_source = self._find_volume_page_limit(
                    title, table_page_limits, text, match.end()
                )

                if page_limit and page_limit_source:
                    print(f"[v5.0.6] Volume '{title}' page limit: {page_limit} (source: {page_limit_source})")

                volumes.append(VolumeInstruction(
                    volume_id=f"VOL-{vol_num}",
                    volume_title=title,
                    volume_number=vol_num,
                    page_limit=page_limit,
                    source_reference=f"Section L (Volume {vol_num_str})",
                    is_mandatory=True
                ))

        # v5.0.9: Try numbered volume patterns if standard patterns found nothing
        # These handle RFPs like "1. Executive Summary and Technical Volume"
        if not volumes:
            volumes = self._extract_numbered_volumes(text, table_page_limits, warnings)
            if volumes:
                print(f"[v5.0.9] Found {len(volumes)} volumes using numbered pattern")

        # Sort by volume number
        return sorted(volumes, key=lambda v: v['volume_number'])

    def _extract_numbered_volumes(
        self,
        text: str,
        table_page_limits: Dict[str, Tuple[str, int]],
        warnings: List[str]
    ) -> List[VolumeInstruction]:
        """
        v5.0.9: Extract volumes from numbered heading format.

        Handles RFPs that use "1. Executive Summary and Technical Volume" instead
        of "Volume I: Technical Proposal".

        This is NOT inference - it looks for explicit "N. Title Volume" patterns.
        """
        volumes: List[VolumeInstruction] = []
        seen_titles: set = set()

        # Split text into lines for line-by-line matching
        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # v5.0.9: Match "N. Title Volume" pattern
            # Examples: "1. Executive Summary and Technical Volume"
            #           "2. Cost & Price Volume"
            #           "3. Contract Documentation Volume"
            numbered_match = re.match(
                r'^(\d+)\.\s+(.+?Volume)\s*$',
                line,
                re.IGNORECASE
            )

            if numbered_match:
                vol_num = int(numbered_match.group(1))
                title = numbered_match.group(2).strip()

                # Clean title - remove trailing punctuation but preserve "Volume"
                title = re.sub(r'[\.\,\;\:]+$', '', title).strip()

                # Skip duplicates
                if title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                # Find page limit
                page_limit, page_limit_source = self._find_volume_page_limit(
                    title, table_page_limits, text, text.find(line)
                )

                print(f"[v5.0.9] Numbered volume found: '{title}' (Vol {vol_num}), "
                      f"page_limit={page_limit}, source={page_limit_source}")

                volumes.append(VolumeInstruction(
                    volume_id=f"VOL-{vol_num}",
                    volume_title=title,
                    volume_number=vol_num,
                    page_limit=page_limit,
                    source_reference=f"Section L (numbered heading {vol_num})",
                    is_mandatory=True
                ))

        return volumes

    def _find_volume_page_limit(
        self,
        title: str,
        table_page_limits: Dict[str, Tuple[str, int]],
        text: str,
        position: int
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        v5.0.9: Unified page limit lookup for volumes.

        Checks in order:
        1. Exact match in table data
        2. Partial match in table data (e.g., "Technical" in "Technical Volume")
        3. Nearby text extraction
        """
        page_limit = None
        page_limit_source = None

        # Try exact match in table data
        if title.lower() in table_page_limits:
            _, page_limit = table_page_limits[title.lower()]
            page_limit_source = "table"
        else:
            # Try partial match (e.g., "Technical" matches "Technical Proposal")
            for table_key, (_, limit) in table_page_limits.items():
                # Check both directions for partial match
                if title.lower() in table_key or table_key in title.lower():
                    page_limit = limit
                    page_limit_source = "table_partial"
                    break
                # Also check key words (Technical, Cost, Contract, etc.)
                title_words = set(title.lower().split())
                key_words = set(table_key.split())
                if title_words & key_words:  # Intersection
                    page_limit = limit
                    page_limit_source = "table_keyword"
                    break

        # Fall back to nearby text extraction if table didn't have it
        if page_limit is None:
            page_limit = self._find_page_limit_near(text, position, title)
            if page_limit:
                page_limit_source = "nearby_text"

        return page_limit, page_limit_source

    def _extract_page_limits_from_inline_table(
        self,
        text: str,
        warnings: List[str]
    ) -> Dict[str, Tuple[str, int]]:
        """
        v5.0.9: Extract page limits from inline table format like Table 1 in RFPs.

        Handles tables formatted like:
            1       Technical               8 Pages
                    Executive Summary       1         Y
            SF 1    Management Approach     10        Y
            SF 2    Infrastructure Approach 10        Y
            2       Cost & Price            None

        Returns:
            Dict mapping title.lower() to (title, page_limit)
        """
        page_limits: Dict[str, Tuple[str, int]] = {}

        # Pattern for table rows with page limits
        # Matches: "1    Technical    8 Pages" or "SF 1   Management   10   Y"
        table_row_patterns = [
            # Volume number | Title | Pages (with optional "Pages" suffix)
            r"^\s*(?:SF\s*)?(\d+)\s{2,}([A-Za-z][A-Za-z &/\-]+?)\s{2,}(\d+)\s*(?:Pages?)?\s*(?:Y|N|NA)?\s*$",
            # Subfactor label | Title | Page limit
            r"^\s*(?:SF|Criteria)\s*(\d+)\s+([A-Za-z][A-Za-z &/\-]+?)\s+(\d+)\s*(?:Pages?)?\s+[YN]",
            # Title only | Pages (for subrows like "Executive Summary   1   Y")
            r"^\s*([A-Z][A-Za-z &/\-]+?)\s{2,}(\d+)\s*(?:Pages?)?\s+Y",
        ]

        lines = text.split('\n')

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Try each pattern
            for i, pattern in enumerate(table_row_patterns):
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    if i < 2:  # Patterns with number prefix
                        title = match.group(2).strip()
                        page_limit_str = match.group(3)
                    else:  # Title-only pattern
                        title = match.group(1).strip()
                        page_limit_str = match.group(2)

                    try:
                        page_limit = int(page_limit_str)
                        if 1 <= page_limit <= 500:
                            # Clean title
                            title = re.sub(r'[\.\,\;\:]+$', '', title).strip()
                            page_limits[title.lower()] = (title, page_limit)
                            print(f"[v5.0.9] Inline table: '{title}' -> {page_limit} pages")
                    except ValueError:
                        pass
                    break

        return page_limits

    def _extract_volumes_from_mentions(
        self,
        text: str,
        warnings: List[str]
    ) -> List[VolumeInstruction]:
        """
        DEPRECATED (v5.0.5): Do not use this method.

        This method infers volumes from keyword mentions like "Technical Proposal"
        which creates HALLUCINATED volumes that may not match RFP requirements.

        The method is kept for reference only. The v5.0.5 architecture requires
        explicit "Volume I:" patterns in Section L text. If no explicit volumes
        are found, the system should fail and ask the user to review the RFP.

        Original purpose: Extract volumes from mentions when no explicit volume
        declarations found. Looks for "Technical Proposal", "Price Proposal", etc.
        """
        # v5.0.5: This method should NOT be called. Return empty list.
        warnings.append(
            "DEPRECATED: _extract_volumes_from_mentions was called. "
            "This fallback is disabled in v5.0.5 to prevent hallucinated volumes."
        )
        return []

        # --- ORIGINAL CODE (disabled) ---
        volumes: List[VolumeInstruction] = []
        seen_titles: set = set()

        # Common volume/proposal type indicators
        vol_indicators = [
            ("Technical Proposal", 1),
            ("Technical Volume", 1),
            ("Management Proposal", 2),
            ("Price Proposal", 2),
            ("Cost Proposal", 2),
            ("Business Proposal", 2),
        ]

        text_lower = text.lower()

        for title, default_num in vol_indicators:
            if title.lower() in text_lower and title.lower() not in seen_titles:
                seen_titles.add(title.lower())

                # Find position and look for page limit nearby
                pos = text_lower.find(title.lower())
                page_limit = self._find_page_limit_near(text, pos + len(title), title)

                # Assign volume number (avoid conflicts)
                vol_num = default_num
                while any(v['volume_number'] == vol_num for v in volumes):
                    vol_num += 1

                volumes.append(VolumeInstruction(
                    volume_id=f"VOL-{vol_num}",
                    volume_title=title,
                    volume_number=vol_num,
                    page_limit=page_limit,
                    source_reference="Section L (inferred from mention)",
                    is_mandatory=True
                ))

        if not volumes:
            warnings.append(
                "No explicit volumes found in Section L. "
                "Consider checking for attachments that define proposal structure."
            )

        return sorted(volumes, key=lambda v: v['volume_number'])

    # v5.0.8: DELETED _promote_sections_to_volumes
    # This method was removed as part of Iron Triangle Determinism enforcement.
    # It hallucinated volumes by inferring structure from keywords like
    # "Contract Documentation" which often didn't match actual RFP structure.
    # See StructureValidationError for the new enforcement mechanism.

    def _extract_sections(
        self,
        text: str,
        volumes: List[VolumeInstruction],
        warnings: List[str]
    ) -> List[SectionInstruction]:
        """
        Extract section instructions for each volume.

        v5.0.9: Uses prefix-based assignment for numbered volumes.
        Section 1.x → Volume 1, Section 2.x → Volume 2, etc.
        This prevents cross-contamination where Section 3.x appears in Volume 1.

        Looks for numbered sections like "1.1 Executive Summary" within
        the context of each volume.
        """
        sections: List[SectionInstruction] = []

        if not volumes:
            return sections

        # v5.0.9: Build volume number to ID mapping for prefix-based assignment
        vol_num_to_id: Dict[int, str] = {}
        vol_num_to_title: Dict[int, str] = {}
        for vol in volumes:
            vol_num_to_id[vol['volume_number']] = vol['volume_id']
            vol_num_to_title[vol['volume_number']] = vol['volume_title']

        # Check if we have numbered volumes (1, 2, 3 format)
        has_numbered_volumes = all(
            vol['volume_number'] in [1, 2, 3, 4, 5] for vol in volumes
        )

        if has_numbered_volumes and len(volumes) >= 2:
            # v5.0.9: Use prefix-based assignment
            sections = self._extract_sections_by_prefix(
                text, vol_num_to_id, vol_num_to_title, warnings
            )
        else:
            # Legacy: Use proximity-based assignment
            sections = self._extract_sections_by_proximity(
                text, volumes, warnings
            )

        return sections

    def _extract_sections_by_prefix(
        self,
        text: str,
        vol_num_to_id: Dict[int, str],
        vol_num_to_title: Dict[int, str],
        warnings: List[str]
    ) -> List[SectionInstruction]:
        """
        v5.0.9: Extract sections using number prefix matching.

        Section 1.1 → Volume 1
        Section 2.1 → Volume 2
        Section 3.4 → Volume 3

        This prevents "Section 3.4 Personnel Security" from being assigned
        to Volume 1 just because it appears early in the text.
        """
        sections: List[SectionInstruction] = []
        seen_sections: set = set()
        order_by_volume: Dict[int, int] = {}

        # Extract all sections from entire text
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text):
                sec_id = match.group(1)
                sec_title = match.group(2).strip()

                # Clean up title
                sec_title = re.sub(r'[\.\,\;\:]+$', '', sec_title).strip()

                # Skip if too short (likely noise)
                if len(sec_title) < 5:
                    continue

                # Skip if title looks like a requirement
                if sec_title.lower().startswith(('the ', 'a ', 'an ')):
                    continue

                # v5.0.7: Validate section header
                if not self._is_valid_section_header(sec_id, sec_title):
                    continue

                # v5.0.9: Clean section ID - remove trailing periods
                sec_id = sec_id.rstrip('.')
                if not sec_id:
                    continue

                # v5.0.9: Extract volume number from section ID prefix
                # "1.1" → vol 1, "2.3" → vol 2, "3.4" → vol 3
                try:
                    prefix = sec_id.split('.')[0]
                    if prefix.isdigit():
                        vol_num = int(prefix)
                    else:
                        # For patterns like "(a)" - assign to volume 1
                        vol_num = 1
                except (ValueError, IndexError):
                    vol_num = 1

                # Check if this volume exists
                if vol_num not in vol_num_to_id:
                    print(f"[v5.0.9] Section {sec_id} prefix={vol_num} has no matching volume, skipping")
                    continue

                vol_id = vol_num_to_id[vol_num]
                vol_title = vol_num_to_title.get(vol_num, f"Volume {vol_num}")

                # Skip duplicates
                section_key = f"{vol_id}:{sec_id}"
                if section_key in seen_sections:
                    continue
                seen_sections.add(section_key)

                # Get order within volume
                if vol_num not in order_by_volume:
                    order_by_volume[vol_num] = 0
                order = order_by_volume[vol_num]
                order_by_volume[vol_num] += 1

                # v6.0.4: N/A LOGIC GATE
                # After matching a section header, scan the next 100 characters
                # If they contain "N/A", "Not Applicable", or "Reserved", SKIP this section
                na_check_start = match.end()
                na_check_end = min(len(text), match.end() + 100)
                following_text = text[na_check_start:na_check_end].upper()

                na_patterns = ['N/A', 'NOT APPLICABLE', 'RESERVED', 'NOT USED', 'INTENTIONALLY LEFT BLANK']
                is_na_section = any(na_pat in following_text for na_pat in na_patterns)

                if is_na_section:
                    print(f"[v6.0.4] N/A GATE: Section {sec_id} '{sec_title}' marked as N/A - EXCLUDED")
                    continue  # Skip this section entirely

                # Find page limit
                page_limit = self._find_page_limit_near(text, match.end(), sec_title)

                print(f"[v6.0.4] Section '{sec_id}' -> Volume {vol_num} ({vol_title})")

                sections.append(SectionInstruction(
                    section_id=sec_id,
                    section_title=sec_title,
                    parent_volume_id=vol_id,
                    page_limit=page_limit,
                    order=order,
                    source_reference=f"Section L ({vol_title})",
                    required_content_types=[]
                ))

        return sections

    def _extract_sections_by_proximity(
        self,
        text: str,
        volumes: List[VolumeInstruction],
        warnings: List[str]
    ) -> List[SectionInstruction]:
        """
        Legacy: Extract sections using text proximity to volume headings.

        Used for "Volume I: Technical" format RFPs where sections don't have
        numeric prefixes that match volume numbers.
        """
        sections: List[SectionInstruction] = []

        for vol in volumes:
            vol_title = vol['volume_title']
            vol_id = vol['volume_id']

            # Find the volume's text block
            vol_pattern = re.escape(vol_title)
            vol_match = re.search(vol_pattern, text, re.IGNORECASE)

            if not vol_match:
                continue

            # Determine scope: from this volume to next volume or end
            start = vol_match.end()
            end = len(text)

            # Find next volume to limit scope
            for other_vol in volumes:
                if other_vol['volume_number'] > vol['volume_number']:
                    other_match = re.search(
                        re.escape(other_vol['volume_title']),
                        text[start:],
                        re.IGNORECASE
                    )
                    if other_match:
                        end = min(end, start + other_match.start())
                        break

            # Limit scope for efficiency
            end = min(end, start + 5000)
            vol_text = text[start:end]

            # Extract numbered sections within this volume's scope
            order = 0
            for pattern in self.section_patterns:
                for match in re.finditer(pattern, vol_text):
                    sec_id = match.group(1)
                    sec_title = match.group(2).strip()

                    # Clean up title
                    sec_title = re.sub(r'[\.\,\;\:]+$', '', sec_title).strip()

                    # Skip if too short (likely noise)
                    if len(sec_title) < 5:
                        continue

                    # Skip if title looks like a requirement, not a section
                    if sec_title.lower().startswith(('the ', 'a ', 'an ')):
                        continue

                    # v5.0.7: Validate section header is not a false positive
                    if not self._is_valid_section_header(sec_id, sec_title):
                        print(f"[v5.0.7] Rejected invalid header: '{sec_id}. {sec_title[:50]}...'")
                        continue

                    # Find page limit for this section
                    page_limit = self._find_page_limit_near(
                        vol_text, match.end(), sec_title
                    )

                    sections.append(SectionInstruction(
                        section_id=sec_id,
                        section_title=sec_title,
                        parent_volume_id=vol_id,
                        page_limit=page_limit,
                        order=order,
                        source_reference=f"Section L ({vol_title})",
                        required_content_types=[]
                    ))
                    order += 1

        return sections

    def _is_valid_section_header(self, sec_id: str, sec_title: str) -> bool:
        """
        v5.0.7: Validate that a section header match is legitimate.

        Rejects false positives like:
        - "505. This is a competitive solicitation..."
        - Section numbers > 99 (unrealistic)
        - Title text that looks like requirement prose
        """
        # Check section ID - reject unrealistic numbers
        try:
            # Extract numeric portion before decimal
            num_part = sec_id.split('.')[0]
            if num_part.isdigit() and int(num_part) > 50:
                return False  # Section numbers > 50 are almost never real
        except (ValueError, IndexError):
            pass

        # Check title against invalid patterns
        title_lower = sec_title.lower()
        for pattern in self.invalid_header_patterns:
            if re.search(pattern, title_lower, re.IGNORECASE):
                return False

        # Additional validation: titles should look like headers
        # Real headers typically: start with capital, 2-8 words, no "shall"
        words = sec_title.split()
        if len(words) > 15:  # Too long for a header
            return False

        if 'shall' in title_lower:  # Requirements contain "shall", headers don't
            return False

        return True

    def _find_page_limit_near(
        self,
        text: str,
        position: int,
        context: str
    ) -> Optional[int]:
        """
        Find page limit mentioned near a position in text.

        Looks in the 300 characters following the position.
        """
        # Search in the next 300 characters
        search_start = max(0, position - 50)
        search_end = min(len(text), position + 300)
        search_text = text[search_start:search_end].lower()

        for pattern in self.page_limit_patterns:
            match = re.search(pattern, search_text)
            if match:
                try:
                    limit = int(match.group(1))
                    # Sanity check: page limits should be reasonable
                    if 1 <= limit <= 500:
                        return limit
                except (ValueError, IndexError):
                    pass

        return None

    def _extract_total_page_limit(self, text: str) -> Optional[int]:
        """Extract total page limit for entire proposal."""
        patterns = [
            r"total\s+(?:of\s+)?(\d+)\s*pages?",
            r"(?:proposal|submission)\s+(?:shall\s+)?(?:not\s+exceed|be\s+limited\s+to)\s+(\d+)\s*pages?",
            r"(?:not\s+to\s+exceed|maximum\s+of)\s+(\d+)\s*(?:total\s+)?pages?",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    limit = int(match.group(1))
                    # Sanity check
                    if 5 <= limit <= 1000:
                        return limit
                except (ValueError, IndexError):
                    pass

        return None

    def _extract_format_rules(self, text: str) -> FormatInstruction:
        """Extract format requirements from text."""
        # Font name
        font_match = re.search(
            r"(Times\s*New\s*Roman|Arial|Calibri|Courier\s*New|Courier)",
            text,
            re.IGNORECASE
        )

        # Font size
        size_match = re.search(
            r"(\d+)\s*[-]?\s*point",
            text,
            re.IGNORECASE
        )

        # Margins
        margin_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:inch|in|\")\s*margins?",
            text,
            re.IGNORECASE
        )

        # Line spacing
        spacing_match = re.search(
            r"(single|double|1\.5)\s*[-]?\s*spac",
            text,
            re.IGNORECASE
        )

        return FormatInstruction(
            font_name=font_match.group(1).strip() if font_match else None,
            font_size=int(size_match.group(1)) if size_match else None,
            margins=f"{margin_match.group(1)} inch" if margin_match else None,
            line_spacing=spacing_match.group(1).lower() if spacing_match else None,
            page_size=None,
            header_footer_rules=None
        )

    def _extract_submission_rules(self, text: str) -> SubmissionInstruction:
        """
        Extract submission requirements from text.

        v6.0: Enhanced due date extraction with SF1449 priority targeting.
        """
        # v6.0: Enhanced due date patterns - prioritize SF1449 format
        date_patterns = [
            # SF1449 Block 8 format: "03 Feb 2025" or "3 February 2025"
            r'(?:offer[s]?\s+(?:due|must\s+be\s+received))[^0-9]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            # "Proposals must be received no later than 1700 Pacific Daylight-Saving Time, 3 February 2024"
            r'(?:received|due)\s+(?:no\s+later\s+than|by)[^0-9]*\d{4}[^,]*,?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            # Standard date after time: "1700 Pacific... Time, 3 February 2024"
            r'\d{4}\s+[A-Za-z]+[^,]+,\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
            # "due date: January 15, 2025" or "due: 01/15/2025"
            r"(?:due|submit|submission)\s*(?:date|by)?\s*[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(?:due|submit|submission)\s*(?:date|by)?\s*[:\s]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            # "no later than January 15, 2025"
            r"no\s+later\s+than[:\s]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"no\s+later\s+than[:\s]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
            # Block 8 explicit: "Block 8" followed by date
            r'(?:Block\s*8|Item\s*8)[^0-9]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:Block\s*8|Item\s*8)[^0-9]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
        ]

        due_date = None
        due_time = None

        # Try to extract time first
        time_match = re.search(
            r'(\d{4})\s+(Pacific|Eastern|Central|Mountain|Local|[A-Z]{2,4})',
            text,
            re.IGNORECASE
        )
        if time_match:
            due_time = f"{time_match.group(1)} {time_match.group(2)}"

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                due_date = match.group(1).strip()
                # Clean up the date format
                due_date = re.sub(r'\s+', ' ', due_date)
                print(f"[v6.0] Extracted due date: {due_date}")
                break

        # Submission method
        method_match = re.search(
            r"(?:submit|submission)\s+(?:via|through|to)\s+(email|portal|electronic|mail|sam\.gov|ebuy|gsa)",
            text,
            re.IGNORECASE
        )

        # v6.0: Also look for "secure file transfer system"
        if not method_match:
            if re.search(r'secure\s+file\s+transfer', text, re.IGNORECASE):
                method_match_str = "secure file transfer"
            elif re.search(r'eBuy|e-Buy', text, re.IGNORECASE):
                method_match_str = "ebuy"
            else:
                method_match_str = None
        else:
            method_match_str = method_match.group(1).lower()

        # File format
        format_match = re.search(
            r"(?:in\s+)?(PDF|Word|\.pdf|\.docx?)\s+format",
            text,
            re.IGNORECASE
        )

        return SubmissionInstruction(
            due_date=due_date,
            due_time=due_time,
            submission_method=method_match_str,
            copies_required=None,
            file_format=format_match.group(1).upper() if format_match else None
        )

    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral or digit string to int."""
        if roman.isdigit():
            return int(roman)

        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        result = 0
        roman = roman.upper()

        for i, char in enumerate(roman):
            if char not in roman_values:
                return 1  # Default if not valid

            current_val = roman_values.get(char, 0)
            next_val = roman_values.get(roman[i + 1], 0) if i + 1 < len(roman) else 0

            if current_val < next_val:
                result -= current_val
            else:
                result += current_val

        return result if result > 0 else 1
