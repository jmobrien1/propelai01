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

        # Extract stated volume count first (for validation)
        stated_count = self._extract_stated_volume_count(full_text)

        # Extract volumes
        volumes = self._extract_volumes(full_text, warnings, structured_tables)

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

        # v5.0.8: MANDATORY volume count validation (Iron Triangle Determinism)
        if stated_count is not None and len(volumes) != stated_count:
            error_msg = (
                f"VOLUME COUNT MISMATCH: RFP explicitly states {stated_count} volumes "
                f"but parser found {len(volumes)}. "
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

    def _extract_volumes(
        self,
        text: str,
        warnings: List[str],
        structured_tables: Optional[List[TableObject]] = None
    ) -> List[VolumeInstruction]:
        """
        Extract volume instructions from text.

        v5.0.6: Now uses table-first strategy - if page limits are found in
        a table structure, those take precedence over nearby text extraction.
        v5.0.8: Uses row-index based linking from structured tables (pdfplumber).
        v5.0.9: Added numbered volume patterns ("1. Technical Volume") for RFPs
                that use numbered headings instead of "Volume I:" format.

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

                # Find page limit
                page_limit = self._find_page_limit_near(text, match.end(), sec_title)

                print(f"[v5.0.9] Section '{sec_id}' -> Volume {vol_num} ({vol_title})")

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
        """Extract submission requirements from text."""
        # Due date patterns
        date_patterns = [
            r"(?:due|submit|submission)\s*(?:date|by)?\s*[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"no\s+later\s+than[:\s]*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        ]

        due_date = None
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                due_date = match.group(1).strip()
                break

        # Submission method
        method_match = re.search(
            r"(?:submit|submission)\s+(?:via|through|to)\s+(email|portal|electronic|mail|sam\.gov)",
            text,
            re.IGNORECASE
        )

        # File format
        format_match = re.search(
            r"(?:in\s+)?(PDF|Word|\.pdf|\.docx?)\s+format",
            text,
            re.IGNORECASE
        )

        return SubmissionInstruction(
            due_date=due_date,
            due_time=None,
            submission_method=method_match.group(1).lower() if method_match else None,
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
