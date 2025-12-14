"""
PropelAI Cycle 5: Multi-Format Document Parser
Parses PDF, DOCX, XLSX with section boundary detection

Preserves page numbers and section references for traceability

Enhanced with Tensorlake integration for:
- Gemini 3 OCR for scanned/complex PDFs
- Better table extraction
- Improved handling of multi-column layouts
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .models import ParsedDocument, DocumentType, RFPBundle

logger = logging.getLogger(__name__)


class MultiFormatParser:
    """
    Parse PDF, DOCX, XLSX into unified format with section detection

    Uses:
    - Tensorlake (Gemini 3 OCR) for enhanced PDF extraction (if configured)
    - pypdf for PDFs (fallback)
    - python-docx for DOCX
    - openpyxl for XLSX
    - markitdown as fallback for complex layouts

    To enable Tensorlake:
        Set TENSORLAKE_API_KEY environment variable
        parser = MultiFormatParser(use_tensorlake=True)
    """
    
    # Enhanced section header patterns (federal RFP standard + variants)
    # Each pattern includes primary and alternative detection methods
    SECTION_PATTERNS = {
        # FAR standard sections with expanded patterns
        "section_a": r"SECTION\s*A[\s:\-–—]+|SOLICITATION.*FORM|SF[- ]?(?:33|1449)",
        "section_b": r"SECTION\s*B[\s:\-–—]+|SUPPLIES?\s*(?:OR\s+)?SERVICES?\s*AND\s*PRICES?|CONTRACT\s+LINE\s+ITEMS?|CLIN\s+(?:STRUCTURE|PRICING)",
        "section_c": r"SECTION\s*C[\s:\-–—]+|DESCRIPTION[/\s]+SPECIFICATIONS?|STATEMENT\s+OF\s+(?:WORK|OBJECTIVES?)|SCOPE\s+OF\s+(?:WORK|CONTRACT)|C\.\d+\s+[-–—]",
        "section_d": r"SECTION\s*D[\s:\-–—]+|PACKAGING\s+(?:AND\s+)?MARKING",
        "section_e": r"SECTION\s*E[\s:\-–—]+|INSPECTION\s+(?:AND\s+)?ACCEPTANCE",
        "section_f": r"SECTION\s*F[\s:\-–—]+|DELIVERIES?\s+(?:OR\s+)?PERFORMANCE|PERIOD\s+OF\s+PERFORMANCE",
        "section_g": r"SECTION\s*G[\s:\-–—]+|CONTRACT\s+ADMINISTRATION\s+DATA",
        "section_h": r"SECTION\s*H[\s:\-–—]+|SPECIAL\s+CONTRACT\s+REQUIREMENTS?",
        "section_i": r"SECTION\s*I[\s:\-–—]+|CONTRACT\s+CLAUSES",
        "section_j": r"SECTION\s*J[\s:\-–—]+|(?:LIST\s+OF\s+)?ATTACHMENTS?|EXHIBITS?",
        "section_k": r"SECTION\s*K[\s:\-–—]+|REPRESENTATIONS?\s*(?:,\s*)?(?:AND\s+)?CERTIFICATIONS?",
        "section_l": r"SECTION\s*L[\s:\-–—]+|INSTRUCTIONS?\s*(?:,\s*CONDITIONS?\s*(?:,\s*)?)?(?:AND\s+)?(?:NOTICES?\s+)?TO\s+OFFERORS?|PROPOSAL\s+(?:SUBMISSION\s+)?(?:REQUIREMENTS?|INSTRUCTIONS?)|L\.\d+\s+[-–—]",
        "section_m": r"SECTION\s*M[\s:\-–—]+|EVALUATION\s+(?:FACTORS?|CRITERIA)\s*(?:FOR\s+AWARD)?|BASIS\s+(?:FOR\s+)?(?:CONTRACT\s+)?AWARD|SOURCE\s+SELECTION\s+(?:CRITERIA|FACTORS?)|M\.\d+\s+[-–—]",
        # Additional document types
        "pws": r"PERFORMANCE\s+WORK\s+STATEMENT|PWS[\s:\-–—]+",
        "sow": r"STATEMENT\s+OF\s+WORK|SOW[\s:\-–—]+",
    }

    # Content-based heuristics for section inference when headers aren't found
    SECTION_CONTENT_HEURISTICS = {
        "section_l": [
            r'\b(?:offeror|proposer)s?\s+(?:shall|must|should)\s+(?:submit|provide|include|describe)',
            r'\b(?:technical|business|cost|price)\s+(?:proposal|volume)',
            r'\bpage\s+limit(?:ation)?s?\b',
            r'\bproposal\s+(?:format|organization|structure)',
            r'\bsubmission\s+(?:requirements?|instructions?)',
        ],
        "section_m": [
            r'\b(?:government|agency)\s+(?:will|shall)\s+(?:evaluate|assess|review)',
            r'\bevaluation\s+(?:factor|criteria)',
            r'\b(?:adjectival|color)\s+ratings?\b',
            r'\b(?:strengths?|weaknesses?|deficienc)',
            r'\bbest\s+value\b',
        ],
        "section_c": [
            r'\bcontractor\s+(?:shall|must|will)\s+(?:provide|perform|deliver|maintain)',
            r'\bthe\s+work\s+(?:shall|will)',
            r'\b(?:scope|objective)s?\s+of\s+(?:work|services?)',
            r'\bdeliverable(?:s)?\s+(?:shall|will)',
        ],
    }
    
    # Article patterns within sections
    ARTICLE_PATTERN = r"ARTICLE\s+([A-Z]\.\d+(?:\.\d+)?)"
    
    # Subsection patterns: C.3.1, L.4.a.2, etc.
    SUBSECTION_PATTERN = r"([A-Z])\.(\d+)(?:\.(\d+|[a-z]))?(?:\.(\d+|[a-z]))?"
    
    def __init__(self, use_markitdown_fallback: bool = True, use_tensorlake: bool = False):
        self.use_markitdown_fallback = use_markitdown_fallback
        self.use_tensorlake = use_tensorlake
        self._tensorlake_processor = None
        self._tensorlake_checked = False
        self.stats = {
            "pdf_parsed": 0,
            "pdf_tensorlake": 0,
            "pdf_pypdf_fallback": 0,
            "docx_parsed": 0,
            "xlsx_parsed": 0,
            "parse_failures": 0,
        }

    def _get_tensorlake_processor(self):
        """Lazy initialization of Tensorlake processor"""
        if self._tensorlake_checked:
            return self._tensorlake_processor

        self._tensorlake_checked = True

        if not self.use_tensorlake:
            return None

        try:
            from agents.integrations import TensorlakeProcessor
            processor = TensorlakeProcessor()
            if processor.is_available:
                self._tensorlake_processor = processor
                logger.info("Tensorlake processor initialized successfully")
            else:
                logger.info("Tensorlake not available (API key not configured)")
        except ImportError:
            logger.debug("Tensorlake integration not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize Tensorlake: {e}")

        return self._tensorlake_processor
    
    def parse_bundle(self, bundle: RFPBundle) -> Dict[str, ParsedDocument]:
        """
        Parse all documents in the bundle
        
        Returns:
            Dict mapping document role to ParsedDocument
        """
        results = {}
        
        # Parse main solicitation
        if bundle.main_document:
            parsed = self.parse_file(bundle.main_document, DocumentType.MAIN_SOLICITATION)
            if parsed:
                results["main"] = parsed
                bundle.parsed_documents["main"] = parsed
        
        # Parse SOW if separate
        if bundle.sow_document:
            parsed = self.parse_file(bundle.sow_document, DocumentType.STATEMENT_OF_WORK)
            if parsed:
                results["sow"] = parsed
                bundle.parsed_documents["sow"] = parsed
        
        # Parse amendments in order
        for i, amendment_path in enumerate(bundle.amendments):
            parsed = self.parse_file(amendment_path, DocumentType.AMENDMENT)
            if parsed:
                key = f"amendment_{i+1}"
                results[key] = parsed
                bundle.parsed_documents[key] = parsed
        
        # Parse research outlines (NIH-specific)
        for ro_id, ro_path in bundle.research_outlines.items():
            parsed = self.parse_file(ro_path, DocumentType.RESEARCH_OUTLINE)
            if parsed:
                results[ro_id] = parsed
                bundle.parsed_documents[ro_id] = parsed
        
        # Parse attachments
        for att_id, att_path in bundle.attachments.items():
            parsed = self.parse_file(att_path, DocumentType.ATTACHMENT)
            if parsed:
                results[att_id] = parsed
                bundle.parsed_documents[att_id] = parsed
        
        return results
    
    def parse_file(self, filepath: str, doc_type: DocumentType) -> Optional[ParsedDocument]:
        """
        Parse a single file based on extension
        
        Returns:
            ParsedDocument or None if parsing failed
        """
        if not os.path.exists(filepath):
            return None
        
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        
        try:
            if ext == ".pdf":
                return self._parse_pdf(filepath, filename, doc_type)
            elif ext in [".docx", ".doc"]:
                return self._parse_docx(filepath, filename, doc_type)
            elif ext in [".xlsx", ".xls"]:
                return self._parse_xlsx(filepath, filename, doc_type)
            elif ext == ".txt":
                return self._parse_text(filepath, filename, doc_type)
            else:
                # Try markitdown for unknown formats
                if self.use_markitdown_fallback:
                    return self._parse_with_markitdown(filepath, filename, doc_type)
                return None
        except Exception as e:
            self.stats["parse_failures"] += 1
            print(f"Warning: Failed to parse {filepath}: {e}")
            return None
    
    def _parse_pdf(self, filepath: str, filename: str, doc_type: DocumentType) -> ParsedDocument:
        """
        Parse PDF using Tensorlake (if available) or pypdf fallback.

        Tensorlake provides better extraction for:
        - Scanned PDFs requiring OCR
        - Complex tables with merged cells
        - Multi-column layouts
        - Documents with charts/diagrams containing text
        """
        # Try Tensorlake first if available
        tensorlake = self._get_tensorlake_processor()
        if tensorlake:
            try:
                result = self._parse_pdf_with_tensorlake(filepath, filename, doc_type, tensorlake)
                if result and result.extraction_quality >= 0.7:
                    self.stats["pdf_tensorlake"] += 1
                    self.stats["pdf_parsed"] += 1
                    return result
                else:
                    logger.info(f"Tensorlake extraction quality low ({result.extraction_quality if result else 0}), falling back to pypdf")
            except Exception as e:
                logger.warning(f"Tensorlake parsing failed for {filename}, falling back to pypdf: {e}")

        # Fallback to pypdf
        return self._parse_pdf_with_pypdf(filepath, filename, doc_type)

    def _parse_pdf_with_tensorlake(self, filepath: str, filename: str, doc_type: DocumentType,
                                    processor) -> Optional[ParsedDocument]:
        """Parse PDF using Tensorlake's Gemini 3 OCR"""
        logger.info(f"Parsing {filename} with Tensorlake (Gemini 3 OCR)")

        result = processor.process_document_sync(filepath, extract_structure=True)

        if not result.success:
            logger.warning(f"Tensorlake extraction failed: {result.error_message}")
            return None

        full_text = result.markdown

        # Convert Tensorlake tables to our format
        tables = []
        for tl_table in result.tables:
            tables.append({
                "headers": tl_table.headers,
                "rows": tl_table.rows,
                "page": tl_table.page_number,
                "source": "tensorlake"
            })

        # Split markdown into pages (estimate based on content)
        pages = self._split_into_pages(full_text, chars_per_page=3000)

        # Detect sections from the markdown
        sections = self._detect_sections(full_text)

        # Assess quality
        quality = self._assess_extraction_quality(full_text, result.page_count or len(pages))

        # Tensorlake typically produces higher quality for complex documents
        if result.page_count and result.page_count > 0:
            quality = min(1.0, quality + 0.1)  # Bonus for successful Tensorlake parse

        return ParsedDocument(
            filepath=filepath,
            filename=filename,
            document_type=doc_type,
            full_text=full_text,
            pages=pages,
            page_count=result.page_count or len(pages),
            sections=sections,
            tables=tables,
            title=self._extract_title(full_text),
            parser_used="tensorlake",
            extraction_quality=quality,
        )

    def _parse_pdf_with_pypdf(self, filepath: str, filename: str, doc_type: DocumentType) -> ParsedDocument:
        """Parse PDF using pypdf (fallback method)"""
        from pypdf import PdfReader

        reader = PdfReader(filepath)
        pages = []
        full_text_parts = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
            full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)

        # Detect sections
        sections = self._detect_sections(full_text)

        # Extract tables (basic detection)
        tables = self._extract_table_hints(full_text)

        # Calculate extraction quality
        quality = self._assess_extraction_quality(full_text, len(reader.pages))

        self.stats["pdf_pypdf_fallback"] += 1
        self.stats["pdf_parsed"] += 1

        return ParsedDocument(
            filepath=filepath,
            filename=filename,
            document_type=doc_type,
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
            sections=sections,
            tables=tables,
            title=self._extract_title(full_text),
            parser_used="pypdf",
            extraction_quality=quality,
        )
    
    def _parse_docx(self, filepath: str, filename: str, doc_type: DocumentType) -> ParsedDocument:
        """Parse DOCX using python-docx"""
        from docx import Document
        
        doc = Document(filepath)
        
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        full_text = "\n\n".join(paragraphs)
        
        # For DOCX, we don't have natural page breaks, estimate
        pages = self._split_into_pages(full_text, chars_per_page=3000)
        
        sections = self._detect_sections(full_text)
        tables = self._extract_docx_tables(doc)
        
        self.stats["docx_parsed"] += 1
        
        return ParsedDocument(
            filepath=filepath,
            filename=filename,
            document_type=doc_type,
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
            sections=sections,
            tables=tables,
            title=self._extract_title(full_text),
            parser_used="python-docx",
            extraction_quality=1.0,
        )
    
    def _parse_xlsx(self, filepath: str, filename: str, doc_type: DocumentType) -> ParsedDocument:
        """Parse Excel using openpyxl"""
        from openpyxl import load_workbook
        
        wb = load_workbook(filepath, data_only=True)
        
        tables = []
        full_text_parts = []
        
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            
            # Extract as table
            table_data = {
                "sheet_name": sheet_name,
                "headers": [],
                "rows": [],
            }
            
            rows_text = []
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                row_values = [str(cell) if cell is not None else "" for cell in row]
                
                # Skip completely empty rows
                if not any(row_values):
                    continue
                
                if i == 0:
                    table_data["headers"] = row_values
                else:
                    table_data["rows"].append(row_values)
                
                rows_text.append(" | ".join(row_values))
            
            tables.append(table_data)
            full_text_parts.append(f"=== Sheet: {sheet_name} ===\n" + "\n".join(rows_text))
        
        full_text = "\n\n".join(full_text_parts)
        
        self.stats["xlsx_parsed"] += 1
        
        return ParsedDocument(
            filepath=filepath,
            filename=filename,
            document_type=doc_type,
            full_text=full_text,
            pages=[full_text],  # Excel doesn't have pages
            page_count=1,
            sections={},
            tables=tables,
            title=filename,
            parser_used="openpyxl",
            extraction_quality=1.0,
        )
    
    def _parse_text(self, filepath: str, filename: str, doc_type: DocumentType) -> ParsedDocument:
        """Parse plain text file"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        
        pages = self._split_into_pages(full_text)
        sections = self._detect_sections(full_text)
        
        return ParsedDocument(
            filepath=filepath,
            filename=filename,
            document_type=doc_type,
            full_text=full_text,
            pages=pages,
            page_count=len(pages),
            sections=sections,
            tables=[],
            parser_used="text",
            extraction_quality=1.0,
        )
    
    def _parse_with_markitdown(self, filepath: str, filename: str, doc_type: DocumentType) -> Optional[ParsedDocument]:
        """Fallback parser using markitdown"""
        try:
            from markitdown import MarkItDown
            
            md = MarkItDown()
            result = md.convert(filepath)
            
            full_text = result.text_content
            pages = self._split_into_pages(full_text)
            sections = self._detect_sections(full_text)
            
            return ParsedDocument(
                filepath=filepath,
                filename=filename,
                document_type=doc_type,
                full_text=full_text,
                pages=pages,
                page_count=len(pages),
                sections=sections,
                tables=[],
                parser_used="markitdown",
                extraction_quality=0.9,
            )
        except Exception as e:
            print(f"Warning: markitdown failed for {filepath}: {e}")
            return None
    
    def _detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect section boundaries in the document

        Uses a multi-stage approach:
        1. Find explicit section headers
        2. Score matches to filter TOC entries vs real headers
        3. Apply content heuristics for undetected content

        Returns:
            Dict mapping section ID to section text
        """
        sections = {}

        # Find all section headers and their positions with scoring
        section_positions = []

        for section_id, pattern in self.SECTION_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get surrounding context
                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end = text.find('\n', match.end())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end]

                # Score this match to filter out TOC entries
                score = self._score_section_match(match, line, text)

                if score > 0:  # Only include positive scores
                    section_positions.append((match.start(), section_id, match.group(), score))

        # Sort by position, then by score (prefer higher scores for same position)
        section_positions.sort(key=lambda x: (x[0], -x[3]))

        # Remove duplicate positions (keep highest scoring)
        seen_positions = set()
        filtered_positions = []
        for pos, section_id, header, score in section_positions:
            # Consider positions within 100 chars as duplicates
            pos_bucket = pos // 100
            if pos_bucket not in seen_positions:
                seen_positions.add(pos_bucket)
                filtered_positions.append((pos, section_id, header))

        section_positions = filtered_positions

        # Extract text between sections
        for i, (start_pos, section_id, header) in enumerate(section_positions):
            # Find end position (start of next section or end of doc)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)

            section_text = text[start_pos:end_pos]
            sections[section_id] = section_text

        # Apply content heuristics to identify sections in unassigned content
        if not sections:
            # No headers found - try content-based detection
            sections = self._detect_sections_by_content(text)

        return sections

    def _score_section_match(self, match, line: str, full_text: str) -> int:
        """
        Score a potential section header match.
        Higher scores = more likely to be a real section header.
        Negative scores = likely a TOC entry or reference.
        """
        score = 10  # Base score

        # TOC indicators - strongly negative
        if '....' in line or re.search(r'\.{3,}\s*\d+\s*$', line):
            return -100  # Definitely a TOC entry

        # Page number at end of line suggests TOC
        if re.search(r'\.\s*\d+\s*$', line):
            return -50

        # Check if this is a cross-reference (not a header)
        preceding = full_text[max(0, match.start()-100):match.start()].lower()
        if re.search(r'(?:in|see|per|under|from)\s*$', preceding):
            return -30

        # Positive indicators
        following = full_text[match.end():match.end()+500]

        # Real sections have substantial content
        if len(following.strip()) > 200:
            score += 20

        # Real sections typically have requirement keywords nearby
        req_keywords = len(re.findall(r'\b(?:shall|must|will|offeror|contractor|proposal|submit)\b',
                                       following, re.IGNORECASE))
        score += min(req_keywords * 5, 30)

        # Headers at start of line
        if match.start() == 0 or full_text[match.start()-1] in '\n\r':
            score += 15

        # Document position heuristics
        doc_position = match.start() / max(len(full_text), 1)

        # Sections L, M typically appear later (past 40% of doc)
        section_id = line[:20].upper()
        if 'SECTION L' in section_id or 'SECTION M' in section_id:
            if doc_position > 0.3:
                score += 20
            elif doc_position < 0.1:
                score -= 30  # Probably TOC

        return score

    def _detect_sections_by_content(self, text: str) -> Dict[str, str]:
        """
        Detect sections using content heuristics when headers aren't found.
        This handles non-standard RFP formats.
        """
        sections = {}

        # Split text into chunks for analysis
        chunk_size = 5000
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append((i, text[i:i + chunk_size]))

        # Score each chunk for section likelihood
        for start_pos, chunk in chunks:
            section_scores = {}
            chunk_lower = chunk.lower()

            for section_id, patterns in self.SECTION_CONTENT_HEURISTICS.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, chunk_lower, re.IGNORECASE))
                    score += matches
                if score > 0:
                    section_scores[section_id] = score

            # Assign chunk to highest-scoring section
            if section_scores:
                best_section = max(section_scores.items(), key=lambda x: x[1])
                if best_section[1] >= 2:  # At least 2 pattern matches
                    if best_section[0] not in sections:
                        sections[best_section[0]] = chunk
                    else:
                        sections[best_section[0]] += chunk

        return sections
    
    def _extract_table_hints(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect potential tables in PDF text
        
        This is heuristic-based; for accurate table extraction,
        would need a table detection library
        """
        tables = []
        
        # Look for patterns that suggest tabular data
        # Multiple columns separated by whitespace
        lines = text.split('\n')
        
        table_start = None
        table_lines = []
        
        for i, line in enumerate(lines):
            # Heuristic: line with 3+ segments of 2+ spaces
            segments = re.split(r'\s{2,}', line.strip())
            if len(segments) >= 3 and all(len(s) > 0 for s in segments[:3]):
                if table_start is None:
                    table_start = i
                table_lines.append(segments)
            else:
                if table_start is not None and len(table_lines) >= 2:
                    tables.append({
                        "start_line": table_start,
                        "end_line": i,
                        "rows": table_lines,
                        "headers": table_lines[0] if table_lines else [],
                    })
                table_start = None
                table_lines = []
        
        return tables
    
    def _extract_docx_tables(self, doc) -> List[Dict[str, Any]]:
        """Extract tables from DOCX document"""
        tables = []
        
        for i, table in enumerate(doc.tables):
            table_data = {
                "table_index": i,
                "headers": [],
                "rows": [],
            }
            
            for j, row in enumerate(table.rows):
                row_data = [cell.text.strip() for cell in row.cells]
                if j == 0:
                    table_data["headers"] = row_data
                else:
                    table_data["rows"].append(row_data)
            
            tables.append(table_data)
        
        return tables
    
    def _split_into_pages(self, text: str, chars_per_page: int = 3000) -> List[str]:
        """Split text into estimated pages"""
        pages = []
        for i in range(0, len(text), chars_per_page):
            pages.append(text[i:i + chars_per_page])
        return pages if pages else [text]
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title from first few lines"""
        lines = text.split('\n')[:20]
        
        for line in lines:
            line = line.strip()
            # Look for title-like patterns
            if len(line) > 10 and len(line) < 200:
                if re.search(r"REQUEST\s*FOR\s*PROPOSAL|RFP|SOLICITATION|STATEMENT\s*OF\s*WORK", line, re.IGNORECASE):
                    return line
        
        # Return first substantial line
        for line in lines:
            if len(line.strip()) > 20:
                return line.strip()[:100]
        
        return None
    
    def _assess_extraction_quality(self, text: str, page_count: int) -> float:
        """
        Assess quality of PDF text extraction
        
        Returns:
            Float 0-1, where 1 is high quality
        """
        if not text:
            return 0.0
        
        # Check for garbled text indicators
        garbled_patterns = [
            r'[^\x00-\x7F]{5,}',  # Long non-ASCII sequences
            r'\s{10,}',          # Excessive whitespace
            r'([a-z])\1{4,}',    # Repeated characters
        ]
        
        quality = 1.0
        
        for pattern in garbled_patterns:
            matches = len(re.findall(pattern, text[:5000]))
            if matches > 10:
                quality -= 0.2
        
        # Check average chars per page
        chars_per_page = len(text) / max(page_count, 1)
        if chars_per_page < 500:  # Likely OCR issues
            quality -= 0.3
        
        return max(0.0, quality)
    
    def find_page_for_text(self, doc: ParsedDocument, search_text: str) -> int:
        """Find which page contains the given text"""
        search_lower = search_text.lower()[:100]  # First 100 chars
        
        for i, page_text in enumerate(doc.pages):
            if search_lower in page_text.lower():
                return i + 1  # 1-indexed
        
        return 0  # Not found
    
    def get_section_for_position(self, doc: ParsedDocument, position: int) -> str:
        """
        Determine which section contains the given character position.

        Uses multiple strategies:
        1. Check if position falls within detected section boundaries
        2. Extract surrounding context and use content heuristics
        3. Return best guess rather than UNKNOWN when possible
        """
        # Strategy 1: Check detected section boundaries
        for section_id, section_text in doc.sections.items():
            section_start = doc.full_text.find(section_text)
            if section_start >= 0 and section_start <= position < section_start + len(section_text):
                return section_id.replace("section_", "").upper()

        # Strategy 2: Use content heuristics on surrounding text
        context_start = max(0, position - 500)
        context_end = min(len(doc.full_text), position + 500)
        context = doc.full_text[context_start:context_end].lower()

        section_scores = {}
        for section_id, patterns in self.SECTION_CONTENT_HEURISTICS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    score += 1
            if score > 0:
                section_scores[section_id] = score

        if section_scores:
            best = max(section_scores.items(), key=lambda x: x[1])
            if best[1] >= 1:  # At least one pattern match
                return best[0].replace("section_", "").upper()

        # Strategy 3: Check for explicit section references in context
        ref_patterns = [
            (r'\bSECTION\s+([A-M])\b', lambda m: m.group(1).upper()),
            (r'\b([A-M])\.\d+', lambda m: m.group(1).upper()),
            (r'\b(PWS|SOW)\b', lambda m: m.group(1).upper()),
        ]
        for pattern, extractor in ref_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return extractor(match)

        # Strategy 4: Infer from document type
        if doc.document_type:
            type_map = {
                'STATEMENT_OF_WORK': 'SOW',
                'PERFORMANCE_WORK_STATEMENT': 'PWS',
                'MAIN_SOLICITATION': 'C',  # Default to C for main doc
            }
            doc_type_str = str(doc.document_type.value if hasattr(doc.document_type, 'value') else doc.document_type)
            for key, section in type_map.items():
                if key in doc_type_str.upper():
                    return section

        return "UNKNOWN"
