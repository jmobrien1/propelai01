"""
PropelAI PDF Parser with Geospatial Traceability
Extracts text with bounding box coordinates for click-to-verify functionality
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Try PyMuPDF first (best bbox support), fallback to pypdf
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    import pypdf


@dataclass
class BoundingBox:
    """Bounding box coordinates for a text element."""
    x0: float  # Left
    y0: float  # Top
    x1: float  # Right
    y1: float  # Bottom
    page: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x0, 2),
            "y": round(self.y0, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "page": self.page,
        }

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this bounding box."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bbox overlaps with another."""
        if self.page != other.page:
            return False
        return not (
            self.x1 < other.x0 or
            self.x0 > other.x1 or
            self.y1 < other.y0 or
            self.y0 > other.y1
        )

    def merge(self, other: "BoundingBox") -> "BoundingBox":
        """Merge two bounding boxes into one encompassing both."""
        if self.page != other.page:
            raise ValueError("Cannot merge bboxes from different pages")
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
            page=self.page,
        )


@dataclass
class TextBlock:
    """A block of text with its location in the PDF."""
    text: str
    bbox: BoundingBox
    block_type: str = "text"  # text, heading, table, list
    confidence: float = 1.0

    # Context for verification
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "block_type": self.block_type,
            "confidence": self.confidence,
        }


@dataclass
class PageContent:
    """Content of a single PDF page."""
    page_number: int
    width: float
    height: float
    blocks: List[TextBlock] = field(default_factory=list)
    full_text: str = ""

    def get_text_at_position(self, x: float, y: float) -> Optional[TextBlock]:
        """Find text block at a specific position."""
        for block in self.blocks:
            if block.bbox.contains_point(x, y):
                return block
        return None

    def search_text(self, query: str) -> List[TextBlock]:
        """Find all blocks containing the query text."""
        query_lower = query.lower()
        return [
            block for block in self.blocks
            if query_lower in block.text.lower()
        ]


@dataclass
class PDFDocument:
    """Parsed PDF document with full traceability."""
    filename: str
    pages: List[PageContent]
    total_pages: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Get full text of document."""
        return "\n\n".join(page.full_text for page in self.pages)

    def get_page(self, page_number: int) -> Optional[PageContent]:
        """Get a specific page (1-indexed)."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
        return None

    def search_text(self, query: str) -> List[Tuple[int, TextBlock]]:
        """Search for text across all pages. Returns (page_num, block) tuples."""
        results = []
        for page in self.pages:
            for block in page.search_text(query):
                results.append((page.page_number, block))
        return results

    def find_requirement_source(self, requirement_text: str) -> Optional[Dict[str, Any]]:
        """
        Find the source location of a requirement in the PDF.
        Returns bbox coordinates and context for verification.
        """
        # Clean the requirement text for matching
        clean_req = self._normalize_text(requirement_text)

        # Try exact match first
        for page in self.pages:
            for block in page.blocks:
                clean_block = self._normalize_text(block.text)
                if clean_req in clean_block or clean_block in clean_req:
                    return {
                        "found": True,
                        "page": page.page_number,
                        "bbox": block.bbox.to_dict(),
                        "matched_text": block.text,
                        "context_before": block.context_before,
                        "context_after": block.context_after,
                        "confidence": self._calculate_match_confidence(clean_req, clean_block),
                    }

        # Try fuzzy match
        best_match = None
        best_score = 0.0

        for page in self.pages:
            for block in page.blocks:
                score = self._similarity_score(clean_req, self._normalize_text(block.text))
                if score > best_score and score > 0.7:
                    best_score = score
                    best_match = {
                        "found": True,
                        "page": page.page_number,
                        "bbox": block.bbox.to_dict(),
                        "matched_text": block.text,
                        "context_before": block.context_before,
                        "context_after": block.context_after,
                        "confidence": score,
                    }

        if best_match:
            return best_match

        return {"found": False, "confidence": 0.0}

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common punctuation variations
        text = text.strip().lower()
        return text

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0

        # Simple word overlap score
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _calculate_match_confidence(self, query: str, found: str) -> float:
        """Calculate confidence of a text match."""
        if query == found:
            return 1.0
        if query in found:
            return 0.95
        if found in query:
            return 0.9
        return self._similarity_score(query, found)


class PDFParser:
    """
    PDF Parser with geospatial text extraction.
    Uses PyMuPDF for accurate bounding box extraction.
    """

    def __init__(self):
        self.use_pymupdf = PYMUPDF_AVAILABLE

    def parse(self, file_path: str) -> PDFDocument:
        """
        Parse a PDF file and extract text with bounding boxes.

        Args:
            file_path: Path to the PDF file

        Returns:
            PDFDocument with pages, blocks, and bounding boxes
        """
        if self.use_pymupdf:
            return self._parse_with_pymupdf(file_path)
        else:
            return self._parse_with_pypdf(file_path)

    def parse_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> PDFDocument:
        """Parse PDF from bytes."""
        if self.use_pymupdf:
            return self._parse_bytes_pymupdf(pdf_bytes, filename)
        else:
            return self._parse_bytes_pypdf(pdf_bytes, filename)

    def _parse_with_pymupdf(self, file_path: str) -> PDFDocument:
        """Parse using PyMuPDF (fitz) for accurate bbox extraction."""
        doc = fitz.open(file_path)
        pages = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_content = self._extract_page_pymupdf(page, page_num + 1)
                pages.append(page_content)

            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
            }

            return PDFDocument(
                filename=Path(file_path).name,
                pages=pages,
                total_pages=len(doc),
                metadata=metadata,
            )
        finally:
            doc.close()

    def _parse_bytes_pymupdf(self, pdf_bytes: bytes, filename: str) -> PDFDocument:
        """Parse PDF bytes using PyMuPDF."""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_content = self._extract_page_pymupdf(page, page_num + 1)
                pages.append(page_content)

            return PDFDocument(
                filename=filename,
                pages=pages,
                total_pages=len(doc),
                metadata={},
            )
        finally:
            doc.close()

    def _extract_page_pymupdf(self, page, page_number: int) -> PageContent:
        """Extract content from a PyMuPDF page."""
        blocks = []
        rect = page.rect

        # Get text blocks with positions
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        all_blocks_text = []

        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                block_bbox = None

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    block_text += line_text + "\n"

                block_text = block_text.strip()

                if block_text:
                    bbox = BoundingBox(
                        x0=block["bbox"][0],
                        y0=block["bbox"][1],
                        x1=block["bbox"][2],
                        y1=block["bbox"][3],
                        page=page_number,
                    )

                    text_block = TextBlock(
                        text=block_text,
                        bbox=bbox,
                        block_type=self._detect_block_type(block_text),
                    )

                    blocks.append(text_block)
                    all_blocks_text.append(block_text)

        # Add context to blocks
        for i, block in enumerate(blocks):
            if i > 0:
                block.context_before = blocks[i-1].text[-200:] if len(blocks[i-1].text) > 200 else blocks[i-1].text
            if i < len(blocks) - 1:
                block.context_after = blocks[i+1].text[:200] if len(blocks[i+1].text) > 200 else blocks[i+1].text

        return PageContent(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            blocks=blocks,
            full_text="\n\n".join(all_blocks_text),
        )

    def _parse_with_pypdf(self, file_path: str) -> PDFDocument:
        """Fallback parser using pypdf (limited bbox support)."""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            pages = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                # Without PyMuPDF, we can only provide page-level granularity
                bbox = BoundingBox(
                    x0=0,
                    y0=0,
                    x1=float(page.mediabox.width) if page.mediabox else 612,
                    y1=float(page.mediabox.height) if page.mediabox else 792,
                    page=page_num + 1,
                )

                blocks = [TextBlock(text=text, bbox=bbox)] if text else []

                pages.append(PageContent(
                    page_number=page_num + 1,
                    width=bbox.x1,
                    height=bbox.y1,
                    blocks=blocks,
                    full_text=text,
                ))

            return PDFDocument(
                filename=Path(file_path).name,
                pages=pages,
                total_pages=len(reader.pages),
                metadata={},
            )

    def _parse_bytes_pypdf(self, pdf_bytes: bytes, filename: str) -> PDFDocument:
        """Fallback parser for bytes using pypdf."""
        import io
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            bbox = BoundingBox(
                x0=0, y0=0,
                x1=float(page.mediabox.width) if page.mediabox else 612,
                y1=float(page.mediabox.height) if page.mediabox else 792,
                page=page_num + 1,
            )

            blocks = [TextBlock(text=text, bbox=bbox)] if text else []

            pages.append(PageContent(
                page_number=page_num + 1,
                width=bbox.x1,
                height=bbox.y1,
                blocks=blocks,
                full_text=text,
            ))

        return PDFDocument(
            filename=filename,
            pages=pages,
            total_pages=len(reader.pages),
            metadata={},
        )

    def _detect_block_type(self, text: str) -> str:
        """Detect the type of text block."""
        text_stripped = text.strip()

        # Heading detection
        if len(text_stripped) < 100:
            # Check for section patterns
            if re.match(r'^(SECTION\s+[A-Z]|PART\s+\d+|CHAPTER\s+\d+)', text_stripped, re.IGNORECASE):
                return "heading"
            if re.match(r'^[A-Z\s]{5,50}$', text_stripped):
                return "heading"
            if re.match(r'^\d+(\.\d+)*\s+[A-Z]', text_stripped):
                return "heading"

        # List detection
        if re.match(r'^[\u2022\u2023\u25E6\u2043\u2219•●○]\s', text_stripped):
            return "list"
        if re.match(r'^[a-z]\)\s|^\(\d+\)\s|^\d+\.\s', text_stripped):
            return "list"

        # Table detection (simple heuristic)
        if text_stripped.count('\t') > 2 or text_stripped.count('|') > 2:
            return "table"

        return "text"


# Convenience function
def parse_pdf(file_path: str) -> PDFDocument:
    """Parse a PDF file with full traceability."""
    parser = PDFParser()
    return parser.parse(file_path)
