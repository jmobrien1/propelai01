"""
PropelAI Text Chunker
Splits documents into optimal chunks for embedding and retrieval
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ChunkStrategy(str, Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source traceability
    source_page: Optional[int] = None
    source_section: Optional[str] = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "source_page": self.source_page,
            "source_section": self.source_section,
            "metadata": self.metadata,
        }


class TextChunker:
    """
    Intelligent text chunker with multiple strategies.
    Optimized for RAG retrieval quality.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Sentence splitting regex
        self._sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence end
            r'(?<=\.)\n+|'              # Period followed by newline
            r'\n\n+'                     # Paragraph break
        )

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks using the configured strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        if self.strategy == ChunkStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, metadata)
        elif self.strategy == ChunkStrategy.SENTENCE:
            return self._chunk_by_sentence(text, metadata)
        elif self.strategy == ChunkStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, metadata)
        elif self.strategy == ChunkStrategy.SEMANTIC:
            return self._chunk_semantic(text, metadata)
        else:
            return self._chunk_fixed_size(text, metadata)

    def _chunk_fixed_size(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Simple fixed-size chunking with overlap."""
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at word boundary
            if end < len(text):
                # Look for space within last 50 chars
                space_pos = text.rfind(' ', end - 50, end + 50)
                if space_pos > start:
                    end = space_pos

            chunk_text = text[start:end].strip()

            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start,
                    end_char=end,
                    metadata=metadata.copy(),
                ))
                index += 1

            start = end - self.chunk_overlap

        return chunks

    def _chunk_by_sentence(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk by sentences, combining until chunk_size is reached."""
        sentences = self._sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        start_char = 0
        index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata=metadata.copy(),
                    ))
                    index += 1

                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                start_char = text.find(current_chunk[0], start_char) if current_chunk else start_char
            else:
                current_chunk.append(sentence)
                current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=len(text),
                    metadata=metadata.copy(),
                ))

        return chunks

    def _chunk_by_paragraph(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """Chunk by paragraphs, combining small ones."""
        # Split on double newlines or multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        start_char = 0
        index = 0

        for para in paragraphs:
            para_len = len(para)

            # If single paragraph exceeds max, split it
            if para_len > self.max_chunk_size:
                # Save current accumulated chunk first
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            text=chunk_text,
                            index=index,
                            start_char=start_char,
                            end_char=start_char + len(chunk_text),
                            metadata=metadata.copy(),
                        ))
                        index += 1
                    current_chunk = []
                    current_length = 0

                # Split large paragraph by sentences
                sub_chunks = self._chunk_by_sentence(para, metadata)
                for sub in sub_chunks:
                    sub.index = index
                    chunks.append(sub)
                    index += 1
                continue

            if current_length + para_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata=metadata.copy(),
                    ))
                    index += 1

                current_chunk = [para]
                current_length = para_len
                start_char = text.find(para, start_char)
            else:
                if not current_chunk:
                    start_char = text.find(para)
                current_chunk.append(para)
                current_length += para_len

        # Last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=len(text),
                    metadata=metadata.copy(),
                ))

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Semantic chunking - attempts to keep related content together.
        Uses section headers and topic shifts as boundaries.
        """
        # Detect section headers
        header_pattern = re.compile(
            r'^(?:'
            r'(?:SECTION\s+[A-Z])|'       # SECTION A, SECTION B
            r'(?:\d+\.[\d.]*\s+[A-Z])|'   # 1.1 Title
            r'(?:[A-Z][A-Z\s]{3,50}:)|'   # ALL CAPS HEADER:
            r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}:)'  # Title Case Header:
            r')',
            re.MULTILINE
        )

        # Split by headers
        parts = header_pattern.split(text)
        headers = header_pattern.findall(text)

        chunks = []
        index = 0
        current_pos = 0

        for i, part in enumerate(parts):
            if not part.strip():
                continue

            # Reconstruct with header
            if i > 0 and i - 1 < len(headers):
                section_text = headers[i - 1] + part
                section_header = headers[i - 1].strip()
            else:
                section_text = part
                section_header = None

            # If section is too large, sub-chunk it
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._chunk_by_paragraph(section_text, metadata)
                for sub in sub_chunks:
                    sub.index = index
                    sub.source_section = section_header
                    sub.start_char += current_pos
                    sub.end_char += current_pos
                    chunks.append(sub)
                    index += 1
            elif len(section_text.strip()) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=section_text.strip(),
                    index=index,
                    start_char=current_pos,
                    end_char=current_pos + len(section_text),
                    source_section=section_header,
                    metadata=metadata.copy(),
                ))
                index += 1

            current_pos += len(section_text)

        return chunks

    def chunk_with_pages(
        self,
        pages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk content that has page information.

        Args:
            pages: List of {"page": int, "text": str} dicts
            metadata: Optional base metadata

        Returns:
            List of chunks with page numbers
        """
        metadata = metadata or {}
        all_chunks = []
        global_index = 0

        for page_data in pages:
            page_num = page_data.get("page", 1)
            page_text = page_data.get("text", "")

            page_metadata = {**metadata, "source_page": page_num}
            chunks = self.chunk(page_text, page_metadata)

            for chunk in chunks:
                chunk.index = global_index
                chunk.source_page = page_num
                all_chunks.append(chunk)
                global_index += 1

        return all_chunks


# Default chunker instance
def get_chunker(
    chunk_size: int = 512,
    strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
) -> TextChunker:
    """Get a configured text chunker."""
    return TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=50,
        strategy=strategy,
    )
