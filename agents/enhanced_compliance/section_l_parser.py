"""
PropelAI v3.0: SectionLParser - Parses Section L text into SectionL_Schema.

This is the bridge between raw RFP text and StrictStructureBuilder.
It extracts ONLY structure information, not content requirements.

Key Principle: Parse explicitly stated structure. DO NOT infer or assume.

v5.0.6: Added table-first extraction strategy for page limits and volume promotion.
"""

from typing import List, Optional, Dict, Tuple
import re

from .section_l_schema import (
    SectionL_Schema,
    VolumeInstruction,
    SectionInstruction,
    FormatInstruction,
    SubmissionInstruction
)


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
            # "1.0 Executive Summary"
            r"(\d+\.\d*)\s+([A-Z][^\n]{4,60})",
            # "Section 1: Technical Approach"
            r"Section\s+(\d+)\s*[:\-]\s*([^\n]+)",
            # "(a) Technical Approach"
            r"\(([a-z])\)\s+([A-Z][^\n]{4,60})",
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

        # v5.0.6: Table structure patterns for extracting volume/page limit associations
        # Tables often have: Volume | Title | Page Limit | Copies
        self.table_row_patterns = [
            # "Volume 1" or "1" followed by title and page limit
            # Pattern: Vol#/Number | Title Text | ## Pages
            r"(?:Volume\s*)?([1-3IVX]+)\s*[|\t]+\s*([A-Za-z][^|\t\n]{3,50})\s*[|\t]+\s*(\d+)\s*(?:pages?)?",
            # Simpler: Number | Title | Pages (tab or multiple spaces as delimiter)
            r"([1-3IVX]+)\s{2,}([A-Za-z][^\t\n]{3,50})\s{2,}(\d+)\s*(?:pages?)?",
        ]

        # v5.0.6: Sections that should be promoted to volumes
        self.volume_promotion_keywords = [
            'contract documentation',
            'representations and certifications',
            'administrative volume',
            'contractual documents',
            'required forms',
            'attachments volume',
        ]

    def parse(
        self,
        section_l_text: str,
        rfp_number: str = "",
        rfp_title: str = "",
        attachment_texts: Optional[Dict[str, str]] = None
    ) -> SectionL_Schema:
        """
        Parse Section L text into structured schema.

        Args:
            section_l_text: Full text of Section L
            rfp_number: RFP/Solicitation number
            rfp_title: RFP title
            attachment_texts: Dict of attachment_name -> text for structural attachments

        Returns:
            SectionL_Schema ready for StrictStructureBuilder
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

        # Extract stated volume count first (for validation)
        stated_count = self._extract_stated_volume_count(full_text)

        # Extract volumes
        volumes = self._extract_volumes(full_text, warnings)

        # v5.0.5: REMOVED mention-based fallback (_extract_volumes_from_mentions)
        # The fallback would infer volumes from keywords like "Technical Proposal"
        # which often created HALLUCINATED volumes not in the RFP.
        # If no explicit "Volume I:" patterns found, we fail and ask user to review.
        if not volumes:
            warnings.append(
                "No explicit volume declarations found in Section L "
                "(e.g., 'Volume I: Technical Approach'). "
                "Please verify the document contains proposal structure instructions."
            )

        # Validate against stated count
        if stated_count and len(volumes) != stated_count:
            warnings.append(
                f"RFP states {stated_count} volumes but found {len(volumes)}. "
                f"Please verify Section L structure."
            )

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

    def _extract_volumes(
        self,
        text: str,
        warnings: List[str]
    ) -> List[VolumeInstruction]:
        """
        Extract volume instructions from text.

        v5.0.6: Now uses table-first strategy - if page limits are found in
        a table structure, those take precedence over nearby text extraction.

        Looks for explicit volume declarations like "Volume I: Technical Proposal".
        """
        volumes: List[VolumeInstruction] = []
        seen_titles: set = set()

        # v5.0.6: Extract table data first for page limit lookup
        table_page_limits = self._extract_page_limits_from_table(text, warnings)

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
                # Check table data first (higher priority), then fall back to nearby text
                page_limit = None
                page_limit_source = None

                # Try exact match in table data
                if title.lower() in table_page_limits:
                    _, page_limit = table_page_limits[title.lower()]
                    page_limit_source = "table"
                else:
                    # Try partial match (e.g., "Technical" matches "Technical Proposal")
                    for table_key, (_, limit) in table_page_limits.items():
                        if title.lower() in table_key or table_key in title.lower():
                            page_limit = limit
                            page_limit_source = "table_partial"
                            break

                # Fall back to nearby text extraction if table didn't have it
                if page_limit is None:
                    page_limit = self._find_page_limit_near(text, match.end(), title)
                    if page_limit:
                        page_limit_source = "nearby_text"

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

        # v5.0.6: Check for sections that should be promoted to volumes
        volumes = self._promote_sections_to_volumes(text, volumes, warnings)

        # Sort by volume number
        return sorted(volumes, key=lambda v: v['volume_number'])

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

    def _promote_sections_to_volumes(
        self,
        text: str,
        existing_volumes: List[VolumeInstruction],
        warnings: List[str]
    ) -> List[VolumeInstruction]:
        """
        v5.0.6: Promote certain sections to root-level volumes.

        Some RFPs use section numbering (e.g., "Section 3: Contract Documentation")
        but these should actually be separate volumes. This method checks for
        known section types that must be elevated to volume status.

        Promotion keywords:
        - Contract Documentation
        - Representations and Certifications
        - Administrative Volume
        - Required Forms/Attachments
        """
        volumes = list(existing_volumes)
        existing_titles = {v['volume_title'].lower() for v in volumes}
        max_vol_num = max([v['volume_number'] for v in volumes], default=0)

        text_lower = text.lower()

        for keyword in self.volume_promotion_keywords:
            # Check if keyword exists but isn't already a volume
            if keyword in text_lower and keyword not in existing_titles:
                # Check if any existing volume already covers this
                already_covered = False
                for existing_title in existing_titles:
                    if keyword in existing_title or existing_title in keyword:
                        already_covered = True
                        break

                if already_covered:
                    continue

                # Find the section pattern for this keyword
                # e.g., "Section 3: Contract Documentation" or "3. Contract Documentation"
                section_patterns = [
                    rf"(?:Section\s+)?(\d+)[.:\s]+\s*({re.escape(keyword)})",
                    rf"({re.escape(keyword)})\s*(?:volume|section)?",
                ]

                for pattern in section_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        # Determine volume number
                        max_vol_num += 1

                        # Get proper title casing
                        title = keyword.title()

                        # Look for page limit
                        page_limit = self._find_page_limit_near(
                            text, match.end(), keyword
                        )

                        volumes.append(VolumeInstruction(
                            volume_id=f"VOL-{max_vol_num}",
                            volume_title=title,
                            volume_number=max_vol_num,
                            page_limit=page_limit,
                            source_reference="Section L (promoted from section)",
                            is_mandatory=True
                        ))

                        print(f"[v5.0.6] Promoted '{title}' to Volume {max_vol_num}")
                        warnings.append(
                            f"'{title}' was promoted from section to volume. "
                            f"Verify this matches RFP structure."
                        )
                        existing_titles.add(keyword)
                        break

        return volumes

    def _extract_sections(
        self,
        text: str,
        volumes: List[VolumeInstruction],
        warnings: List[str]
    ) -> List[SectionInstruction]:
        """
        Extract section instructions for each volume.

        Looks for numbered sections like "1.0 Executive Summary" within
        the context of each volume.
        """
        sections: List[SectionInstruction] = []

        if not volumes:
            return sections

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
