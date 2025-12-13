"""
Tensorlake Document Processor Integration

Uses Tensorlake's Gemini 3-powered OCR for enhanced document extraction,
particularly useful for:
- Scanned PDFs and image-based documents
- Complex tables with merged cells
- Multi-column layouts
- Charts and diagrams with text

Environment Variables:
    TENSORLAKE_API_KEY: Your Tensorlake API key (required)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Tensorlake output format options"""
    MARKDOWN = "markdown"
    JSON = "json"
    CHUNKS = "chunks"


@dataclass
class TensorlakeConfig:
    """Configuration for Tensorlake integration"""
    api_key: Optional[str] = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    extract_tables: bool = True
    extract_images: bool = False
    chunk_size: int = 1000  # For chunked output
    overlap: int = 100

    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("TENSORLAKE_API_KEY")

        if not self.api_key:
            logger.warning(
                "TENSORLAKE_API_KEY not set. Tensorlake features will be unavailable. "
                "Set the environment variable or pass api_key to TensorlakeConfig."
            )


@dataclass
class ExtractedTable:
    """Represents an extracted table from a document"""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    raw_markdown: str = ""
    confidence: float = 0.0


@dataclass
class ExtractedSection:
    """Represents an extracted section from a document"""
    title: str
    content: str
    page_start: int
    page_end: int
    section_type: str = "unknown"  # requirement, instruction, evaluation, etc.
    tables: List[ExtractedTable] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete extraction result from Tensorlake"""
    success: bool
    file_id: Optional[str] = None
    job_id: Optional[str] = None
    markdown: str = ""
    sections: List[ExtractedSection] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    page_count: int = 0
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class TensorlakeProcessor:
    """
    Document processor using Tensorlake's Gemini 3 OCR.

    Usage:
        processor = TensorlakeProcessor()
        result = await processor.process_document("/path/to/rfp.pdf")

        if result.success:
            print(result.markdown)
            for table in result.tables:
                print(f"Table on page {table.page_number}: {table.headers}")
    """

    def __init__(self, config: Optional[TensorlakeConfig] = None):
        self.config = config or TensorlakeConfig()
        self._client = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of Tensorlake client"""
        if self._initialized:
            return self._client is not None

        self._initialized = True

        if not self.config.api_key:
            logger.error("Cannot initialize Tensorlake: API key not configured")
            return False

        try:
            from tensorlake.documentai import DocumentAI
            self._client = DocumentAI(api_key=self.config.api_key)
            logger.info("Tensorlake client initialized successfully")
            return True
        except ImportError:
            logger.error(
                "tensorlake package not installed. "
                "Install with: pip install tensorlake"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Tensorlake client: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if Tensorlake is properly configured and available"""
        return self._ensure_initialized()

    async def process_document(
        self,
        file_path: str,
        extract_structure: bool = True
    ) -> ExtractionResult:
        """
        Process a document using Tensorlake's Gemini 3 OCR.

        Args:
            file_path: Path to the document (PDF, DOCX, images)
            extract_structure: Whether to extract document structure (sections, tables)

        Returns:
            ExtractionResult with extracted content
        """
        if not self._ensure_initialized():
            return ExtractionResult(
                success=False,
                error_message="Tensorlake not initialized. Check API key configuration."
            )

        try:
            # Upload document
            logger.info(f"Uploading document to Tensorlake: {file_path}")
            file_id = self._client.upload(file_path)

            # Parse document
            logger.info(f"Parsing document (file_id: {file_id})")
            parse_id = self._client.parse(file_id)

            # Wait for completion
            result = self._client.wait_for_completion(parse_id)

            # Process result
            extraction = self._process_result(result, file_id, parse_id)

            if extract_structure:
                extraction = self._extract_structure(extraction)

            logger.info(
                f"Extraction complete: {extraction.page_count} pages, "
                f"{len(extraction.tables)} tables"
            )
            return extraction

        except Exception as e:
            logger.error(f"Tensorlake processing failed: {e}")
            return ExtractionResult(
                success=False,
                error_message=str(e)
            )

    def process_document_sync(
        self,
        file_path: str,
        extract_structure: bool = True
    ) -> ExtractionResult:
        """
        Synchronous version of process_document.

        For use in non-async contexts.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.process_document(file_path, extract_structure)
        )

    def _process_result(
        self,
        result: Any,
        file_id: str,
        job_id: str
    ) -> ExtractionResult:
        """Process raw Tensorlake result into ExtractionResult"""
        try:
            markdown = ""
            page_count = 0

            # Handle different Tensorlake result formats
            if hasattr(result, 'markdown') and result.markdown:
                markdown = result.markdown
            elif hasattr(result, 'chunks') and result.chunks:
                # Extract text content from chunks
                chunk_texts = []
                for chunk in result.chunks:
                    if hasattr(chunk, 'content') and chunk.content:
                        chunk_texts.append(str(chunk.content))
                    elif hasattr(chunk, 'text') and chunk.text:
                        chunk_texts.append(str(chunk.text))
                    elif isinstance(chunk, str):
                        chunk_texts.append(chunk)
                markdown = "\n\n".join(chunk_texts)
            elif hasattr(result, 'pages') and result.pages:
                # Extract from pages structure
                page_texts = []
                for page in result.pages:
                    if hasattr(page, 'content'):
                        page_texts.append(str(page.content))
                    elif hasattr(page, 'text'):
                        page_texts.append(str(page.text))
                    elif hasattr(page, 'fragments'):
                        # Handle page fragments
                        frag_texts = []
                        for frag in page.fragments:
                            if hasattr(frag, 'content'):
                                frag_texts.append(str(frag.content))
                            elif hasattr(frag, 'text'):
                                frag_texts.append(str(frag.text))
                        page_texts.append(" ".join(frag_texts))
                markdown = "\n\n".join(page_texts)
            elif isinstance(result, dict):
                # Dict response
                markdown = result.get('markdown', '')
                if not markdown:
                    markdown = result.get('content', '')
                if not markdown and 'chunks' in result:
                    chunk_texts = []
                    for chunk in result['chunks']:
                        if isinstance(chunk, dict):
                            chunk_texts.append(chunk.get('content', chunk.get('text', '')))
                        else:
                            chunk_texts.append(str(chunk))
                    markdown = "\n\n".join(chunk_texts)
            elif isinstance(result, str):
                markdown = result
            else:
                # Last resort - but filter out object representations
                result_str = str(result)
                # If it looks like raw Python objects, don't use it
                if 'PageFragment(' in result_str or 'Chunk(' in result_str or 'bbox=' in result_str:
                    logger.warning("Tensorlake result contains raw object representations, extraction may be incomplete")
                    # Try to extract just text content using regex
                    import re
                    content_matches = re.findall(r"content='([^']*)'", result_str)
                    text_matches = re.findall(r"text='([^']*)'", result_str)
                    markdown = "\n".join(content_matches + text_matches)
                else:
                    markdown = result_str

            # Extract page count
            if hasattr(result, 'page_count'):
                page_count = result.page_count
            elif hasattr(result, 'pages'):
                page_count = len(result.pages)
            elif isinstance(result, dict):
                page_count = result.get('page_count', 0)

            # Clean up markdown - remove any remaining object artifacts
            if markdown:
                import re
                # Remove bbox coordinates
                markdown = re.sub(r"bbox=\{[^}]+\}", "", markdown)
                # Remove reading_order
                markdown = re.sub(r"reading_order=\d+,?\s*", "", markdown)
                # Remove PageFragment wrapper
                markdown = re.sub(r"PageFragment\([^)]+\)", "", markdown)
                # Remove fragment_type
                markdown = re.sub(r"fragment_type=<[^>]+>,?\s*", "", markdown)
                # Clean up extra whitespace
                markdown = re.sub(r'\n{3,}', '\n\n', markdown)
                markdown = markdown.strip()

            return ExtractionResult(
                success=True,
                file_id=file_id,
                job_id=job_id,
                markdown=markdown,
                page_count=page_count,
                raw_response=result if isinstance(result, dict) else None
            )
        except Exception as e:
            logger.error(f"Error processing Tensorlake result: {e}")
            return ExtractionResult(
                success=False,
                file_id=file_id,
                job_id=job_id,
                error_message=f"Result processing error: {e}"
            )

    def _extract_structure(self, extraction: ExtractionResult) -> ExtractionResult:
        """Extract document structure (sections, tables) from markdown"""
        if not extraction.success or not extraction.markdown:
            return extraction

        # Extract tables from markdown
        extraction.tables = self._extract_tables_from_markdown(extraction.markdown)

        # Extract sections based on headers
        extraction.sections = self._extract_sections_from_markdown(extraction.markdown)

        return extraction

    def _extract_tables_from_markdown(self, markdown: str) -> List[ExtractedTable]:
        """Extract tables from markdown content"""
        tables = []
        lines = markdown.split('\n')

        in_table = False
        current_table_lines = []
        table_index = 0

        for line in lines:
            # Detect table start (markdown table format: | col1 | col2 |)
            if '|' in line and line.strip().startswith('|'):
                in_table = True
                current_table_lines.append(line)
            elif in_table:
                if '|' in line:
                    current_table_lines.append(line)
                else:
                    # Table ended
                    if len(current_table_lines) >= 2:
                        table = self._parse_markdown_table(
                            current_table_lines,
                            table_index
                        )
                        if table:
                            tables.append(table)
                            table_index += 1
                    current_table_lines = []
                    in_table = False

        # Handle table at end of document
        if in_table and len(current_table_lines) >= 2:
            table = self._parse_markdown_table(current_table_lines, table_index)
            if table:
                tables.append(table)

        return tables

    def _parse_markdown_table(
        self,
        lines: List[str],
        index: int
    ) -> Optional[ExtractedTable]:
        """Parse markdown table lines into ExtractedTable"""
        try:
            # First line is headers
            header_line = lines[0]
            headers = [
                cell.strip()
                for cell in header_line.split('|')
                if cell.strip()
            ]

            # Skip separator line (|---|---|)
            data_start = 1
            if len(lines) > 1 and set(lines[1].replace('|', '').replace('-', '').replace(':', '').strip()) == set():
                data_start = 2

            # Parse data rows
            rows = []
            for line in lines[data_start:]:
                row = [
                    cell.strip()
                    for cell in line.split('|')
                    if cell.strip() or cell == ''
                ]
                # Normalize row length to match headers
                while len(row) < len(headers):
                    row.append('')
                rows.append(row[:len(headers)])

            return ExtractedTable(
                page_number=1,  # Would need page info from Tensorlake
                table_index=index,
                headers=headers,
                rows=rows,
                raw_markdown='\n'.join(lines),
                confidence=0.9
            )
        except Exception as e:
            logger.warning(f"Failed to parse table: {e}")
            return None

    def _extract_sections_from_markdown(
        self,
        markdown: str
    ) -> List[ExtractedSection]:
        """Extract sections based on markdown headers"""
        sections = []
        lines = markdown.split('\n')

        current_section = None
        current_content = []

        for line in lines:
            # Detect headers (# Header, ## Header, etc.)
            if line.strip().startswith('#'):
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                # Start new section
                header_match = line.lstrip('#').strip()
                current_section = ExtractedSection(
                    title=header_match,
                    content="",
                    page_start=1,
                    page_end=1,
                    section_type=self._classify_section(header_match)
                )
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        return sections

    def _classify_section(self, title: str) -> str:
        """Classify section type based on title"""
        title_lower = title.lower()

        if any(kw in title_lower for kw in ['requirement', 'shall', 'must']):
            return 'requirement'
        elif any(kw in title_lower for kw in ['instruction', 'submit', 'format']):
            return 'instruction'
        elif any(kw in title_lower for kw in ['evaluation', 'criteria', 'factor']):
            return 'evaluation'
        elif any(kw in title_lower for kw in ['scope', 'background', 'purpose']):
            return 'background'
        elif any(kw in title_lower for kw in ['attachment', 'exhibit', 'appendix']):
            return 'attachment'
        else:
            return 'unknown'


# Convenience function for quick usage
def extract_with_tensorlake(
    file_path: str,
    api_key: Optional[str] = None
) -> ExtractionResult:
    """
    Quick extraction using Tensorlake.

    Args:
        file_path: Path to document
        api_key: Optional API key (uses env var if not provided)

    Returns:
        ExtractionResult with extracted content
    """
    config = TensorlakeConfig(api_key=api_key) if api_key else TensorlakeConfig()
    processor = TensorlakeProcessor(config)
    return processor.process_document_sync(file_path)
