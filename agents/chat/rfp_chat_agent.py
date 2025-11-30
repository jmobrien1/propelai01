"""
PropelAI: RFP Chat Agent

Enables users to ask questions about uploaded RFP documents and receive answers
based solely on the document content (no external information).

Similar to NotebookLM - RAG-based Q&A with source citations.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from an RFP document"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata contains: source_file, section, page, chunk_index
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatMessage:
    """Represents a chat message (user or assistant)"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RFPChatAgent:
    """
    Chat agent for RFP documents using Retrieval Augmented Generation (RAG).
    
    Features:
    - Document chunking with overlap
    - Keyword-based retrieval (simple, fast)
    - Claude API for answer generation
    - Source citation extraction
    - Chat history management
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize the chat agent.
        
        Args:
            anthropic_api_key: Anthropic API key. If None, reads from env.
        """
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        # Import anthropic here to avoid import errors if not installed
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            raise
        
        # Configuration
        self.chunk_size = 800  # characters per chunk
        self.chunk_overlap = 200  # overlap between chunks
        self.max_chunks_to_retrieve = 8  # top K chunks for context
        self.max_context_length = 6000  # max characters in context
    
    # ============================================================================
    # FILE TEXT EXTRACTION
    # ============================================================================
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from PDF or DOCX files.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"[CHAT] File not found: {file_path}")
            return ""
        
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif ext in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            else:
                logger.warning(f"[CHAT] Unsupported file type: {ext}")
                return ""
        except Exception as e:
            logger.error(f"[CHAT] Error extracting text from {file_path}: {e}")
            return ""
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using pypdf"""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num}]\n{text}")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"[CHAT] Extracted {len(full_text)} characters from PDF: {file_path.name}")
            return full_text
            
        except Exception as e:
            logger.error(f"[CHAT] PDF extraction error: {e}")
            return ""
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX using python-docx"""
        try:
            from docx import Document
            
            doc = Document(str(file_path))
            text_parts = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_parts.append(cell.text)
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"[CHAT] Extracted {len(full_text)} characters from DOCX: {file_path.name}")
            return full_text
            
        except Exception as e:
            logger.error(f"[CHAT] DOCX extraction error: {e}")
            return ""
    
    # ============================================================================
    # DOCUMENT CHUNKING
    # ============================================================================
    
    def chunk_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split document text into overlapping chunks.
        
        Args:
            text: Full document text
            metadata: Metadata to attach to all chunks (source_file, section, etc.)
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split by paragraphs first (double newline)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunk_id = f"chunk-{metadata.get('source_file', 'unknown')}-{chunk_index}"
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    text=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_length": len(current_chunk)
                    }
                ))
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-50:]) if len(words) > 50 else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                chunk_index += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Save final chunk
        if current_chunk.strip():
            chunk_id = f"chunk-{metadata.get('source_file', 'unknown')}-{chunk_index}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "char_length": len(current_chunk)
                }
            ))
        
        logger.info(f"[CHAT] Created {len(chunks)} chunks from document (metadata: {metadata})")
        return chunks
    
    def chunk_rfp_documents(self, rfp_data: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk all documents associated with an RFP.
        
        Args:
            rfp_data: RFP data dict containing file paths and requirements
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        # 1. Extract text from uploaded file paths
        file_paths = rfp_data.get("file_paths", [])
        file_names = rfp_data.get("files", [])
        
        if file_paths:
            logger.info(f"[CHAT] Extracting text from {len(file_paths)} uploaded files...")
            
            for idx, file_path in enumerate(file_paths):
                # Get corresponding filename
                filename = file_names[idx] if idx < len(file_names) else Path(file_path).name
                
                # Extract text from file
                text = self.extract_text_from_file(file_path)
                
                if text and len(text.strip()) > 50:  # Only chunk if we got substantial text
                    chunks = self.chunk_document(text, metadata={
                        "source_file": filename,
                        "section": "FULL_DOCUMENT",
                        "file_path": file_path
                    })
                    all_chunks.extend(chunks)
                    logger.info(f"[CHAT] Created {len(chunks)} chunks from {filename}")
                else:
                    logger.warning(f"[CHAT] No text extracted from {filename}")
        
        # 2. Also include requirements (as fallback/supplement)
        requirements = rfp_data.get("requirements", [])
        if requirements:
            logger.info(f"[CHAT] Adding {len(requirements)} extracted requirements...")
            
            # Group requirements by section for better context
            section_texts = {}
            for req in requirements:
                section = req.get("section_ref", "UNKNOWN")
                text = req.get("text", "") or req.get("full_text", "")
                if text:
                    if section not in section_texts:
                        section_texts[section] = []
                    section_texts[section].append(f"[{req.get('req_id', 'REQ')}] {text}")
            
            # Chunk each section
            for section, texts in section_texts.items():
                combined_text = "\n\n".join(texts)
                chunks = self.chunk_document(combined_text, metadata={
                    "source_file": "extracted_requirements",
                    "section": section
                })
                all_chunks.extend(chunks)
        
        logger.info(f"[CHAT] Total chunks created for RFP: {len(all_chunks)}")
        
        if len(all_chunks) == 0:
            logger.warning("[CHAT] No chunks created! Check if files were uploaded and text was extracted.")
        
        return all_chunks
    
    # ============================================================================
    # RETRIEVAL
    # ============================================================================
    
    def retrieve_relevant_chunks(
        self, 
        question: str, 
        chunks: List[DocumentChunk],
        top_k: Optional[int] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve most relevant chunks for a question using keyword matching.
        
        Args:
            question: User's question
            chunks: All available document chunks
            top_k: Number of chunks to retrieve (default: self.max_chunks_to_retrieve)
            
        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if not chunks:
            return []
        
        top_k = top_k or self.max_chunks_to_retrieve
        question_lower = question.lower()
        
        # Extract keywords from question
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'what',
                     'when', 'where', 'who', 'how', 'why', 'which', 'this', 'that'}
        
        words = re.findall(r'\b\w+\b', question_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            chunk_text_lower = chunk.text.lower()
            score = 0
            
            # Keyword matching
            for keyword in keywords:
                count = chunk_text_lower.count(keyword)
                score += count * 2  # Weight by frequency
            
            # Boost for section mentions
            if any(section_word in question_lower for section_word in ['section', 'factor', 'volume']):
                # Extract section reference from question
                section_match = re.search(r'section\s+([a-z0-9\.]+)', question_lower)
                if section_match:
                    section_ref = section_match.group(1).upper()
                    chunk_section = chunk.metadata.get('section', '').upper()
                    if section_ref in chunk_section:
                        score += 20
            
            # Boost for exact phrase matches
            # Extract 2-3 word phrases
            for i in range(len(keywords) - 1):
                phrase = ' '.join(keywords[i:i+2])
                if phrase in chunk_text_lower:
                    score += 10
            
            scored_chunks.append((chunk, score))
        
        # Sort by score and return top K
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:top_k]
        
        # Filter out chunks with very low scores
        top_chunks = [(c, s) for c, s in top_chunks if s > 0]
        
        logger.info(f"[CHAT] Retrieved {len(top_chunks)} relevant chunks (scores: {[s for _, s in top_chunks[:5]]})")
        return top_chunks
    
    # ============================================================================
    # ANSWER GENERATION
    # ============================================================================
    
    def generate_answer(
        self,
        question: str,
        context_chunks: List[DocumentChunk],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate an answer using Claude API based on retrieved context.
        
        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks
            chat_history: Previous chat messages for context
            
        Returns:
            Tuple of (answer_text, sources_list)
        """
        if not context_chunks:
            return (
                "I don't have enough information in the uploaded RFP documents to answer that question. "
                "Could you try rephrasing or asking about a different aspect of the RFP?",
                []
            )
        
        # Build context from chunks (limit total length)
        context_parts = []
        total_length = 0
        sources = []
        
        for i, chunk in enumerate(context_chunks):
            chunk_text = chunk.text
            if total_length + len(chunk_text) > self.max_context_length:
                break
            
            context_parts.append(f"[Source {i+1}]\n{chunk_text}")
            sources.append({
                "id": chunk.id,
                "text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "section": chunk.metadata.get("section", "Unknown"),
                "page": chunk.metadata.get("page", "Unknown"),
                "file": chunk.metadata.get("source_file", "Unknown")
            })
            total_length += len(chunk_text)
        
        context = "\n\n".join(context_parts)
        
        # Build system prompt
        system_prompt = """You are an expert RFP (Request for Proposal) assistant helping proposal writers understand solicitation documents.

CRITICAL RULES:
1. Answer ONLY based on the provided context from the RFP documents
2. If the answer is not in the context, explicitly say "I don't have that information in the uploaded RFP documents"
3. Cite specific sections, pages, or document sources when possible
4. Be precise and actionable - provide information useful for proposal writing
5. If you're unsure, acknowledge uncertainty
6. Use clear, professional language
7. For page limits, deadlines, or requirements - be extremely precise

Your goal is to help proposal teams quickly find information and understand requirements."""

        # Build conversation messages
        messages = []
        
        # Add recent chat history for context (last 4 messages)
        if chat_history:
            for msg in chat_history[-4:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Add current question with context
        user_message = f"""Context from RFP documents:

{context}

Question: {question}

Please answer based only on the context above. If you cite information, reference the source number."""

        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Call Claude API
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.3,  # Lower temperature for more factual responses
                system=system_prompt,
                messages=messages
            )
            
            answer = response.content[0].text
            logger.info(f"[CHAT] Generated answer (length: {len(answer)})")
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"[CHAT] Error generating answer: {e}")
            return (
                f"I encountered an error while generating the answer: {str(e)}. "
                "Please try again or rephrase your question.",
                sources
            )
    
    # ============================================================================
    # CHAT ORCHESTRATION
    # ============================================================================
    
    def chat(
        self,
        question: str,
        document_chunks: List[DocumentChunk],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> ChatMessage:
        """
        Main chat function - retrieve context and generate answer.
        
        Args:
            question: User's question
            document_chunks: All document chunks for this RFP
            chat_history: Previous chat messages
            
        Returns:
            ChatMessage with assistant's response
        """
        logger.info(f"[CHAT] Processing question: {question[:100]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieved = self.retrieve_relevant_chunks(question, document_chunks)
        chunks_only = [chunk for chunk, score in retrieved]
        
        # Step 2: Generate answer
        answer, sources = self.generate_answer(question, chunks_only, chat_history)
        
        # Step 3: Create response message
        response = ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sources=sources
        )
        
        return response
