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
        self.chunk_size = 1000  # characters per chunk (increased for more context)
        self.chunk_overlap = 200  # overlap between chunks
        self.max_chunks_to_retrieve = 20  # top K chunks for context (increased from 8)
        self.max_context_length = 15000  # max characters in context (increased for better answers)
    
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
    # SECTION DETECTION
    # ============================================================================
    
    def detect_rfp_section(self, text: str, page_num: Optional[int] = None) -> str:
        """
        Detect which RFP section this text belongs to.
        
        Common RFP sections:
        - Section L: Instructions to Offerors
        - Section M: Evaluation Criteria
        - Section C: Statement of Work / Performance Work Statement
        - Section B: Contract Details / Schedule
        - Cover/Front Matter
        - Amendments/Q&A
        
        Args:
            text: Text to analyze
            page_num: Page number (if available)
            
        Returns:
            Section identifier string
        """
        text_lower = text.lower()
        text_upper = text.upper()
        
        # Check for explicit section headers
        if re.search(r'\bsection\s+l\b', text_lower) or 'instructions to offeror' in text_lower:
            return "SECTION_L"
        if re.search(r'\bsection\s+m\b', text_lower) or 'evaluation' in text_lower and 'criteria' in text_lower:
            return "SECTION_M"
        if re.search(r'\bsection\s+c\b', text_lower) or 'statement of work' in text_lower or 'performance work statement' in text_lower:
            return "SECTION_C"
        if re.search(r'\bsection\s+b\b', text_lower) or ('schedule' in text_lower and 'contract' in text_lower):
            return "SECTION_B"
        
        # Check for cover page indicators
        if page_num and page_num <= 3:
            if any(term in text_lower for term in ['solicitation', 'rfp', 'request for proposal']):
                return "COVER"
        
        # Check for amendments/Q&A
        if 'amendment' in text_lower or 'question' in text_lower and 'answer' in text_lower:
            return "AMENDMENT"
        
        # Check for other common sections
        if re.search(r'\bsection\s+[a-z]\b', text_lower):
            match = re.search(r'\bsection\s+([a-z])\b', text_lower)
            if match:
                return f"SECTION_{match.group(1).upper()}"
        
        return "GENERAL"
    
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
                # Detect section for this chunk
                detected_section = self.detect_rfp_section(current_chunk, metadata.get('page'))
                
                chunk_id = f"chunk-{metadata.get('source_file', 'unknown')}-{chunk_index}"
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    text=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "char_length": len(current_chunk),
                        "rfp_section": detected_section  # Add section detection
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
            detected_section = self.detect_rfp_section(current_chunk, metadata.get('page'))
            
            chunk_id = f"chunk-{metadata.get('source_file', 'unknown')}-{chunk_index}"
            chunks.append(DocumentChunk(
                id=chunk_id,
                text=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "char_length": len(current_chunk),
                    "rfp_section": detected_section  # Add section detection
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
        Section-aware retrieval: Ensures diverse coverage of RFP sections.
        
        For general questions, prioritizes retrieving from key sections:
        - COVER (basic info)
        - SECTION_L (instructions)
        - SECTION_M (evaluation)
        - SECTION_C (SOW)
        - SECTION_B (contract details)
        
        Args:
            question: User's question
            chunks: All available document chunks
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples with section diversity
        """
        if not chunks:
            return []
        
        top_k = top_k or self.max_chunks_to_retrieve
        question_lower = question.lower()
        
        # Extract keywords
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'what',
                     'when', 'where', 'who', 'how', 'why', 'which', 'this', 'that'}
        
        words = re.findall(r'\b\w+\b', question_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Define priority sections for general questions
        priority_sections = ['COVER', 'SECTION_L', 'SECTION_M', 'SECTION_C', 'SECTION_B']
        
        # Detect if question is general (about the whole RFP)
        general_question_indicators = ['about', 'overview', 'summary', 'tell me', 'describe', 'what is']
        is_general = any(indicator in question_lower for indicator in general_question_indicators)
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            chunk_text_lower = chunk.text.lower()
            rfp_section = chunk.metadata.get('rfp_section', 'GENERAL')
            score = 0
            
            # Keyword matching
            for keyword in keywords:
                count = chunk_text_lower.count(keyword)
                score += count * 2
            
            # Section-aware bonuses
            if is_general and rfp_section in priority_sections:
                # Boost priority sections for general questions
                section_priority = priority_sections.index(rfp_section) if rfp_section in priority_sections else 10
                score += (10 - section_priority) * 5  # Higher boost for earlier sections
            
            # Boost for specific section mentions in question
            if 'section l' in question_lower or 'instruction' in question_lower:
                if rfp_section == 'SECTION_L':
                    score += 30
            if 'section m' in question_lower or 'evaluation' in question_lower or 'criteria' in question_lower:
                if rfp_section == 'SECTION_M':
                    score += 30
            if 'section c' in question_lower or 'statement of work' in question_lower or 'sow' in question_lower:
                if rfp_section == 'SECTION_C':
                    score += 30
            if 'deadline' in question_lower or 'due date' in question_lower or 'submission' in question_lower:
                if rfp_section in ['COVER', 'SECTION_L']:
                    score += 25
            
            # Phrase matching boost
            for i in range(len(keywords) - 1):
                phrase = ' '.join(keywords[i:i+2])
                if phrase in chunk_text_lower:
                    score += 10
            
            scored_chunks.append((chunk, score, rfp_section))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Section-diverse selection for general questions
        if is_general:
            selected_chunks = []
            section_counts = {section: 0 for section in priority_sections}
            max_per_section = max(2, top_k // len(priority_sections))
            
            # First pass: get chunks from priority sections
            for chunk, score, section in scored_chunks:
                if score == 0:
                    continue
                if section in priority_sections and section_counts[section] < max_per_section:
                    selected_chunks.append((chunk, score))
                    section_counts[section] += 1
                    if len(selected_chunks) >= top_k * 0.7:  # 70% from priority sections
                        break
            
            # Second pass: fill remaining with highest scores
            for chunk, score, section in scored_chunks:
                if score == 0:
                    continue
                if (chunk, score) not in selected_chunks:
                    selected_chunks.append((chunk, score))
                    if len(selected_chunks) >= top_k:
                        break
            
            top_chunks = selected_chunks[:top_k]
        else:
            # For specific questions, just use top scores
            top_chunks = [(c, s) for c, s, _ in scored_chunks[:top_k] if s > 0]
        
        sections_retrieved = [c.metadata.get('rfp_section', 'UNKNOWN') for c, _ in top_chunks]
        logger.info(f"[CHAT] Retrieved {len(top_chunks)} chunks from sections: {Counter(sections_retrieved)}")
        
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
        system_prompt = """# SYSTEM PROMPT (v2.2 - Federal & GSA Optimized)

## IDENTITY & PERSONA
You are the PropelAI Proposal Copilot, a Senior Capture Manager. You are an expert at "Forensic RFP Analysis"—finding requirements even when they are buried in dense text, cover letters, or attachments.

## CORE OPERATIONAL DIRECTIVES

### 1. The "Forensic Scan" Protocol
Federal RFPs often lack clean summary tables. You must search aggressively.
- **Search Narrative Text:** Page limits are often buried in sentences like "The technical proposal shall not exceed 30 pages."
- **Search Attachments:** If Section M references "Attachment 11," check that file.
- **Infer from Context:** If "Factor 1" is found, keep reading to find Factors 2, 3, etc. Do not stop until the section ends.

### 2. Location Agnosticism (CRITICAL for GSA/USCG)
Data is not always in a folder named "Section L."
- **Check the Cover Letter:** In GSA/BPA schedules, page limits and submission instructions are often in the "RFQ Letter" or the first 5 pages of the PDF.
- **Treat the Letter as Section L:** If you find instructions in a letter, extract them as binding Section L constraints.
- **Federal RFPs often have 5+ Evaluation Factors.** Do not stop after finding Factor 1 or 2. Explicitly search for "Factor 3," "Factor 4," "Factor 5," etc., until you are certain no others exist.

### 3. The "Iron Triangle" Logic
Triangulate these three sources to answer compliance questions:
- **Section C (Scope):** What work is required?
- **Section L (Instructions):** Page limits, font sizes, margins (may be in cover letter).
- **Section M (Evaluation):** Scoring weights and factors.

## RESPONSE PROTOCOLS

### Protocol A: When asked for "Evaluation Criteria"
Create a **Unified Scoring Table**:

| Factor | Weight/Importance | Page Limit | Key Criteria (Summarized) |
| :--- | :--- | :--- | :--- |
| 1. Technical | Most Important | 15 pages (found in Cover Letter) | Approach to PWS Tasks 1-4 |
| 2. Price | Least Important | No Limit | Total Evaluated Price |

*Example Logic:* "I see Factor 1 (Experience) and Factor 2 (Management). The text mentions 'Factors 1-5 are significantly more important than cost.' I must find Factors 3, 4, and 5 before answering."

### Protocol B: When asked for "Page Limits" or "Instructions"
- **First:** Check for a document named "Section L" or "Instructions to Offerors"
- **Second:** If not found, check the RFP Cover Letter, RFQ Letter, or first 5 pages of main solicitation
- **Extract:** Page limits per volume/section, font size, margins, file format requirements
- **Distinguish:** "shall/must" (mandatory) vs. "should" (preference)

### Protocol C: Handling Strategy ("Win Themes")
Use **"Ghosting"** techniques:
- Analyze Section M criteria to find hidden biases (e.g., "transition experience" suggests incumbent worry)
- Suggest specific themes that alleviate government fears
- Position against likely competitors based on evaluation weights

### Protocol D: Handling "Missing" Data
- **Never** lead with "CRITICAL FINDING: INCOMPLETE" unless the document is empty
- **First:** State what you *did* find (e.g., "I found the 3-Volume Structure")
- **Second:** Flag specific missing details *after* providing value (e.g., "Note: While the 3 volumes are listed, the specific page count for Volume II was not found in the provided excerpts")
- If a specific reference is missing (e.g., "See Attachment J.1" but J.1 not uploaded), state: "Reference to [Document] found, but file not present in analysis. Cannot confirm details for this section."

## HANDLING "J-ATTACHMENTS" & EXHIBITS (v2.3 - DoD/Navy Specifics)
In DoD/Navy solicitations, requirements are often decoupled from the main SOW. You must strictly enforce the following mappings:

### 1. Personnel & Staffing (The "J.2" Protocol)
When asked about "Staffing," "Key Personnel," or "Qualifications":
- **IGNORE** general descriptions in Section C
- **SOURCE OF TRUTH:** Specific "Attachment J" files labeled "Personnel Qualifications" or "Labor Categories" (e.g., Attachment J.2)
- **ACTION:** Extract the exact "Degree," "Years of Experience," and "Cybersecurity Certifications" (e.g., CSWF/DoD 8570) for every labor category
- **COMPLIANCE CHECK:** If the user proposes a candidate, cross-reference strictly against these J.2 definitions

### 2. Deliverables (The "CDRL" Protocol)
When asked about "Schedule," "Reports," or "Deliverables":
- **SOURCE OF TRUTH:** "Exhibit A" or "Contract Data Requirements List" (CDRL)
- **ACTION:** Do not just list the report title. Extract the "Frequency" (Block 10), "Distribution" (Block 14), and "Format" from the DD1423 forms

### 3. Performance Metrics (The "QASP" Protocol)
When asked about "Win Themes" or "Quality":
- **SOURCE OF TRUTH:** "Attachment J.3" or "Quality Assurance Surveillance Plan" (QASP)
- **ACTION:** Identify the "Acceptable Quality Levels" (AQLs)
- **STRATEGY:** Suggest Win Themes that explicitly exceed these AQLs (e.g., "RFP allows 5% error rate; our automated testing guarantees <0.1%")

### 4. REVISED "IRON TRIANGLE" FOR DOD
The triangle expands to a square:
1. **Section C:** The Task
2. **Section L:** The Format
3. **Section M:** The Score
4. **The Attachments:** The Specifics (J.2, J.3, Exhibit A)

*All four must align. A mismatch between J.2 Quals and Section M Scoring is a fatal flaw.*

## TONE & STYLE
- **Professional & Constructive:** Provide actionable data first
- **Citation Required:** Every claim must reference Source Document and Page Number (e.g., `[RFP-75N9, Page 12]` or `[Attachment J.2, Page 3]`)
- **No Fluff:** Do not define what an RFP is. Just analyze it.
- **No Hallucinations:** If you don't see it in the text, mark as "Not Specified"
- **Attachment Awareness:** Always check for J-Attachments and Exhibits before saying requirements are missing

## DATA SOURCES AVAILABLE
You have access to:
- COVER/LETTER: Basic RFP information, often contains Section L instructions for GSA/USCG
- SECTION L: Instructions to Offerors (may be in cover letter for GSA)
- SECTION M: Evaluation Criteria and scoring
- SECTION C: Statement of Work / Performance Work Statement (high-level)
- SECTION B: Contract Details
- **J-ATTACHMENTS:** Personnel (J.2), QASP (J.3), other requirements (SOURCE OF TRUTH for DoD)
- **EXHIBITS:** CDRLs (Exhibit A), pricing templates, other structured data
- Amendments and Q&A documents

Answer ONLY based on provided context. When discussing personnel/deliverables/performance, prioritize J-Attachments over Section C generalities."""

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
        
        # Call Claude API - using Claude 4 models (current as of Nov 2025)
        model_versions = [
            "claude-sonnet-4-5",                # Claude 4.5 Sonnet (recommended - alias)
            "claude-haiku-4-5",                 # Claude 4.5 Haiku (fast and affordable)
            "claude-sonnet-4-5-20250929",       # Claude 4.5 Sonnet (specific version)
            "claude-3-5-sonnet-20240620",       # Legacy fallback
        ]
        
        for model in model_versions:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1500,
                    temperature=0.3,  # Lower temperature for more factual responses
                    system=system_prompt,
                    messages=messages
                )
                
                answer = response.content[0].text
                logger.info(f"[CHAT] Generated answer using {model} (length: {len(answer)})")
                
                return answer, sources
                
            except Exception as e:
                logger.warning(f"[CHAT] Error with model {model}: {e}")
                if model == model_versions[-1]:  # Last model in list
                    logger.error(f"[CHAT] All models failed. Last error: {e}")
                    return (
                        f"I encountered an error while generating the answer. "
                        f"This may be due to API access limitations. "
                        f"Please check your Anthropic API key permissions. Error: {str(e)}",
                        []
                    )
                continue  # Try next model
    
    # ============================================================================
    # PHASE 2: COMPLIANCE & LOGIC ENGINE
    # ============================================================================
    
    def detect_query_type(self, question: str) -> str:
        """
        Detect the type of query to route to specialized handlers.
        
        Returns:
            'cross_reference' | 'contradiction' | 'formatting' | 'general'
        """
        question_lower = question.lower()
        
        # Cross-reference detection
        cross_ref_indicators = [
            'page limit' and ('section m' in question_lower or 'factor' in question_lower),
            'enough space' in question_lower,
            'allow' in question_lower and 'address' in question_lower,
            'volume' in question_lower and 'page' in question_lower and ('factor' in question_lower or 'criteria' in question_lower)
        ]
        if any(cross_ref_indicators):
            return 'cross_reference'
        
        # Contradiction detection
        if 'contradiction' in question_lower or 'inconsisten' in question_lower or 'conflict' in question_lower:
            return 'contradiction'
        
        # Formatting rules
        formatting_indicators = ['font', 'margin', 'format', 'file format', 'spacing', 'formatting']
        if any(indicator in question_lower for indicator in formatting_indicators):
            return 'formatting'
        
        return 'general'
    
    def handle_cross_reference_query(
        self, 
        question: str, 
        chunks: List[DocumentChunk]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Handle cross-reference queries between Section L and Section M.
        
        Example: "Does Section L page limit allow enough space for Section M factors?"
        """
        # Get Section L chunks (page limits)
        section_l_chunks = [c for c in chunks if c.metadata.get('rfp_section') == 'SECTION_L']
        # Get Section M chunks (evaluation criteria)
        section_m_chunks = [c for c in chunks if c.metadata.get('rfp_section') == 'SECTION_M']
        
        # Build specialized context
        context_parts = []
        sources = []
        
        context_parts.append("=== SECTION L: Page Limits and Instructions ===")
        for i, chunk in enumerate(section_l_chunks[:5], 1):
            context_parts.append(f"[Source {i}, Section L] {chunk.text[:800]}")
            sources.append({
                "section": chunk.metadata.get('rfp_section', 'UNKNOWN'),
                "page": chunk.metadata.get('page', 'N/A'),
                "text": chunk.text[:200]
            })
        
        context_parts.append("\n=== SECTION M: Evaluation Factors ===")
        for i, chunk in enumerate(section_m_chunks[:5], len(context_parts)):
            context_parts.append(f"[Source {i}, Section M] {chunk.text[:800]}")
            sources.append({
                "section": chunk.metadata.get('rfp_section', 'UNKNOWN'),
                "page": chunk.metadata.get('page', 'N/A'),
                "text": chunk.text[:200]
            })
        
        context = "\n\n".join(context_parts)
        
        # Specialized prompt for cross-reference
        prompt = f"""You are analyzing cross-references between RFP sections.

{context}

Question: {question}

TASK: Compare Section L page limits against Section M evaluation factors. Determine if there's enough space to address all factors. Provide:
1. Section L page limits (be specific - which volume, how many pages)
2. Section M factors (list them and estimate complexity)
3. Analysis: Is there enough space? Flag any concerns.
4. Recommendation: Any specific strategies for space management?

Be specific and actionable."""

        return prompt, sources
    
    def handle_contradiction_detection(
        self,
        question: str,
        chunks: List[DocumentChunk]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect contradictions between different RFP sections.
        
        Example: "Are there contradictions between C.4 and M.2?"
        """
        # Extract section references from question
        section_refs = re.findall(r'section\s+([a-z])\.?(\d+)?', question.lower())
        
        relevant_chunks = []
        for section_ref in section_refs:
            section_letter = section_ref[0].upper()
            target_section = f"SECTION_{section_letter}"
            matching = [c for c in chunks if c.metadata.get('rfp_section') == target_section]
            relevant_chunks.extend(matching[:3])
        
        # If no specific sections mentioned, get chunks from C and M
        if not relevant_chunks:
            relevant_chunks = [c for c in chunks if c.metadata.get('rfp_section') in ['SECTION_C', 'SECTION_M']][:8]
        
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(relevant_chunks, 1):
            section = chunk.metadata.get('rfp_section', 'UNKNOWN')
            context_parts.append(f"[Source {i}, {section}] {chunk.text[:900]}")
            sources.append({
                "section": section,
                "page": chunk.metadata.get('page', 'N/A'),
                "text": chunk.text[:200]
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a compliance analyst reviewing RFP documents for contradictions.

{context}

Question: {question}

TASK: Analyze the provided sections for contradictions, inconsistencies, or conflicts. Look for:
1. Conflicting requirements (e.g., different technical specs in different sections)
2. Inconsistent terminology or definitions
3. Contradictory evaluation criteria vs. technical requirements
4. Mismatched deadlines or deliverables

For each contradiction found:
- Clearly state the contradiction
- Reference both sources
- Assess the severity (Critical / Moderate / Minor)
- Suggest a Q&A question to submit to the Contracting Officer

If no contradictions found, explicitly state that."""

        return prompt, sources
    
    def handle_formatting_query(
        self,
        question: str,
        chunks: List[DocumentChunk]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract formatting rules from Section L.
        
        Example: "What are the font, margin, and file format rules?"
        """
        # Get Section L chunks
        section_l_chunks = [c for c in chunks if c.metadata.get('rfp_section') == 'SECTION_L']
        
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(section_l_chunks[:8], 1):
            context_parts.append(f"[Source {i}, Section L] {chunk.text[:900]}")
            sources.append({
                "section": "SECTION_L",
                "page": chunk.metadata.get('page', 'N/A'),
                "text": chunk.text[:200]
            })
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are extracting precise formatting requirements from Section L.

{context}

Question: {question}

TASK: Extract ALL formatting requirements mentioned in Section L. Create a structured summary:

**Font Requirements:**
- Body text: [size and type]
- Headers: [size and type]
- Tables/figures: [size and type]
- Exceptions: [any special cases]

**Margin Requirements:**
- Top/Bottom: [measurements]
- Left/Right: [measurements]

**File Format Requirements:**
- Accepted formats: [list]
- File naming: [conventions]

**Page Limits:**
- [List by volume/section]

**Other Formatting Rules:**
- Line spacing: [requirement]
- Page numbering: [requirement]
- Any other specific rules

Be EXTREMELY precise with numbers and measurements. Quote directly when possible."""

        return prompt, sources
    
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
        Main chat function with specialized query routing (Phase 2).
        
        Routes queries to specialized handlers:
        - Cross-reference: Section L vs. M analysis
        - Contradiction: Detect conflicts between sections
        - Formatting: Extract Section L rules
        - General: Standard RAG retrieval
        
        Args:
            question: User's question
            document_chunks: All document chunks for this RFP
            chat_history: Previous chat messages
            
        Returns:
            ChatMessage with assistant's response
        """
        logger.info(f"[CHAT] Processing question: {question[:100]}...")
        
        # Detect query type
        query_type = self.detect_query_type(question)
        logger.info(f"[CHAT] Query type detected: {query_type}")
        
        # Route to specialized handler
        if query_type == 'cross_reference':
            custom_prompt, sources = self.handle_cross_reference_query(question, document_chunks)
            # Use custom prompt directly with Claude
            answer = self._call_claude_with_custom_prompt(custom_prompt, chat_history)
            
        elif query_type == 'contradiction':
            custom_prompt, sources = self.handle_contradiction_detection(question, document_chunks)
            answer = self._call_claude_with_custom_prompt(custom_prompt, chat_history)
            
        elif query_type == 'formatting':
            custom_prompt, sources = self.handle_formatting_query(question, document_chunks)
            answer = self._call_claude_with_custom_prompt(custom_prompt, chat_history)
            
        else:
            # Standard RAG flow
            retrieved = self.retrieve_relevant_chunks(question, document_chunks)
            chunks_only = [chunk for chunk, score in retrieved]
            answer, sources = self.generate_answer(question, chunks_only, chat_history)
        
        # Create response message
        response = ChatMessage(
            role="assistant",
            content=answer,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sources=sources if query_type == 'general' else []
        )
        
        return response
    
    def _call_claude_with_custom_prompt(
        self, 
        custom_prompt: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Call Claude with a custom specialized prompt.
        Used by Phase 2 handlers.
        """
        messages = []
        
        # Add chat history if available
        if chat_history:
            for msg in chat_history[-4:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        messages.append({
            "role": "user",
            "content": custom_prompt
        })
        
        # Try models
        model_versions = [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet-20240620",
        ]
        
        for model in model_versions:
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=2000,  # More tokens for detailed compliance analysis
                    temperature=0.2,  # Lower for precision
                    messages=messages
                )
                
                answer = response.content[0].text
                logger.info(f"[CHAT] Generated specialized answer using {model}")
                return answer
                
            except Exception as e:
                logger.warning(f"[CHAT] Error with model {model}: {e}")
                if model == model_versions[-1]:
                    logger.error(f"[CHAT] All models failed")
                    return f"I encountered an error while analyzing. Error: {str(e)}"
                continue
