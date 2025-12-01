"""
PropelAI Cycle 5: Requirement Extractor
Multi-pattern extraction with semantic classification

Extracts requirements from ALL sections, not just Section C

v3.0: Enhanced with VisibleThread-style dictionary and text cleaning
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .models import (
    RequirementNode, RequirementType, ConfidenceLevel, 
    RequirementStatus, SourceLocation, ParsedDocument, DocumentType
)
from .requirement_dictionary import get_requirement_dictionary, RequirementSeverity
from .text_cleaner import clean_text, clean_requirement_text


class RequirementExtractor:
    """
    Extract requirements from parsed documents
    
    Uses multiple extraction strategies:
    1. Keyword patterns (shall, must, required)
    2. Semantic patterns (by requirement type)
    3. Context analysis (section location)
    4. Entity extraction (CLINs, deliverables, dates)
    
    v2.1: Added quality filters to reduce noise
    - Minimum sentence length
    - Noise pattern filtering (TOC, headers, references)
    - Stronger semantic signals required
    - Duplicate/near-duplicate detection
    """
    
    # === QUALITY TUNING PARAMETERS ===
    MIN_SENTENCE_LENGTH = 100         # Minimum chars for a valid requirement (increased)
    MAX_SENTENCE_LENGTH = 1000        # Maximum chars (avoid capturing paragraphs)
    MIN_WORDS = 15                    # Minimum words in a requirement (increased)
    REQUIRE_ACTOR = True              # Require "contractor/offeror/government" for high confidence
    
    # Noise patterns to filter out (TOC, headers, boilerplate)
    NOISE_PATTERNS = [
        r"^SECTION\s+[A-Z]\s*[-–]\s*",                    # Section headers
        r"^ARTICLE\s+[A-Z]\.\d+",                         # Article headers  
        r"^TABLE\s+OF\s+CONTENTS",                        # TOC
        r"^\d+\s*$",                                      # Page numbers
        r"^[A-Z\s]+\.\.\.\.\.*\s*\d+$",                   # TOC entries with dots
        r"^(?:Page|Pg\.?)\s*\d+",                         # Page references
        r"^RFP\s+\d+",                                    # RFP number headers
        r"^REQUEST\s+FOR\s+PROPOSAL",                    # Document title
        r"^\s*\d+\s*\n",                                  # Standalone numbers
        r"^[A-Z][a-z]+\s+\d{1,2},\s+\d{4}",              # Dates as headers
        r"^ATTACHMENT\s+\d+.*RFP",                        # Attachment headers
        r"^\(continued\)",                                # Continuation markers
        r"^_{5,}",                                        # Underline separators
        r"^FAR\s+\d+\.\d+[-\d]*\s*$",                    # Bare FAR references
        r"^HHSAR\s+\d+\.\d+",                            # Bare HHSAR references
        r"^https?://",                                    # URLs alone
        r"^52\.\d{3}-\d+",                               # FAR clause numbers alone
        r"^\d{1,3}\s*$",                                  # Just numbers
        r"^[A-Z]\.\d+\s*$",                               # Just section refs (C.3.1)
        r"^PART\s+[IVX]+",                                # Part headers
        r"^SUPPLIES\s+OR\s+SERVICES",                     # Section B header
        r"^EVALUATION\s+FACTORS",                         # Section M header
        r"^INSTRUCTIONS.*OFFERORS",                       # Section L header
        r"^\(\s*[a-z]\s*\)\s*$",                          # Subparagraph markers
        r"^\s*[ivx]+\.\s*$",                              # Roman numeral lists
    ]
    
    # Boilerplate phrases that indicate non-requirement text
    BOILERPLATE_PATTERNS = [
        r"this\s+page\s+intentionally\s+left\s+blank",
        r"end\s+of\s+(?:section|document|attachment)",
        r"see\s+continuation\s+sheet",
        r"reserved\s*$",
        r"not\s+applicable\s*$",
        r"to\s+be\s+determined",
        r"^\s*n/a\s*$",
        r"^\s*tbd\s*$",
        r"incorporated\s+by\s+reference",
        r"as\s+prescribed\s+in",
        r"clause\s+is\s+incorporated",
        r"the\s+following\s+(?:clauses?|provisions?)\s+(?:are|is)\s+incorporated",
    ]
    
    # Mandatory requirement patterns
    MANDATORY_PATTERNS = [
        (r"\bshall\b", "shall"),
        (r"\bmust\b", "must"),
        (r"\bis\s+required\s+to\b", "required"),
        (r"\bare\s+required\s+to\b", "required"),
        (r"\bwill\s+be\s+required\b", "required"),
        (r"\bmandatory\b", "mandatory"),
        (r"\bis\s+responsible\s+for\b", "responsible"),
        (r"\bshall\s+be\s+responsible\b", "responsible"),
    ]
    
    # Conditional/optional patterns
    CONDITIONAL_PATTERNS = [
        (r"\bshould\b", "should"),
        (r"\bmay\b", "may"),
        (r"\bcan\b", "can"),
        (r"\bis\s+encouraged\b", "encouraged"),
        (r"\bis\s+recommended\b", "recommended"),
        (r"\boptional\b", "optional"),
    ]
    
    # Prohibition patterns
    PROHIBITION_PATTERNS = [
        (r"\bshall\s+not\b", "shall_not"),
        (r"\bmust\s+not\b", "must_not"),
        (r"\bwill\s+not\b", "will_not"),
        (r"\bprohibited\b", "prohibited"),
        (r"\bforbidden\b", "forbidden"),
        (r"\bnot\s+permitted\b", "not_permitted"),
    ]
    
    # Semantic patterns for classification
    SEMANTIC_PATTERNS = {
        RequirementType.PERFORMANCE: [
            r"contractor\s+shall\s+(?:provide|perform|maintain|ensure|support|develop|conduct|deliver)",
            r"contractor\s+must\s+(?:provide|perform|maintain|ensure|support|develop|conduct|deliver)",
            r"contractor\s+is\s+required\s+to",
            r"contractor\s+will\s+(?:provide|perform|be\s+responsible)",
        ],
        RequirementType.PROPOSAL_INSTRUCTION: [
            r"offeror[s]?\s+shall\s+(?:describe|provide|submit|include|demonstrate|address)",
            r"offeror[s]?\s+must\s+(?:describe|provide|submit|include|demonstrate|address)",
            r"proposal[s]?\s+(?:shall|must)\s+(?:include|contain|address|describe)",
            r"the\s+(?:technical|business)\s+proposal\s+(?:shall|must|should)",
            r"offeror[s]?\s+(?:should|may)\s+(?:describe|provide|include)",
            r"submit\s+(?:a|the)\s+(?:technical|management|staffing|cost)",
        ],
        RequirementType.EVALUATION_CRITERION: [
            r"government\s+will\s+(?:evaluate|assess|consider|review)",
            r"evaluation\s+(?:will|shall)\s+be\s+based\s+on",
            r"(?:will|shall)\s+be\s+evaluated\s+(?:on|based|against)",
            r"government\s+(?:may|will)\s+(?:award|select)",
            r"proposals?\s+will\s+be\s+(?:rated|scored|evaluated)",
            r"(?:most|more|less)\s+important\s+than",
        ],
        RequirementType.PERFORMANCE_METRIC: [
            r"performance\s+(?:will|shall)\s+be\s+(?:monitored|measured|assessed)",
            r"(?:threshold|target|objective|metric)[:\s]+\d+",
            r"acceptable\s+quality\s+level",
            r"\d+%\s+(?:on-time|accuracy|availability|uptime)",
            r"within\s+\d+\s+(?:hours|days|weeks)\s+of",
        ],
        RequirementType.DELIVERABLE: [
            r"(?:submit|deliver|provide)\s+(?:a|the|an)\s+(?:monthly|weekly|quarterly|final)\s+report",
            r"deliverable[s]?\s+(?:include|are|shall)",
            r"report\s+shall\s+be\s+(?:submitted|delivered|provided)",
            r"due\s+(?:within|by|on|no\s+later\s+than)",
        ],
        RequirementType.LABOR_REQUIREMENT: [
            r"\d+[,\d]*\s+(?:labor\s+)?hours",
            r"labor\s+(?:category|categories|mix|composition)",
            r"full-time\s+equivalent",
            r"key\s+personnel",
            r"(?:minimum|required)\s+(?:staff|personnel|FTE)",
        ],
        RequirementType.QUALIFICATION: [
            r"(?:must|shall)\s+be\s+(?:a\s+)?(?:small\s+business|8\(a\)|HUBZone|SDVOSB|WOSB)",
            r"(?:must|shall)\s+(?:have|possess|demonstrate)\s+(?:a\s+)?(?:clearance|certification|experience)",
            r"(?:minimum|required)\s+(?:qualifications?|experience|years)",
            r"certified\s+(?:in|as|by)",
        ],
        RequirementType.COMPLIANCE: [
            r"FAR\s+\d+\.\d+",
            r"DFARS\s+\d+\.\d+",
            r"HHSAR\s+\d+\.\d+",
            r"in\s+accordance\s+with",
            r"comply\s+with",
            r"compliant\s+with",
            r"Section\s+508",
        ],
        RequirementType.FORMAT: [
            r"\d+[\s-]*point\s+font",
            r"\d+[\s-]*inch\s+margin",
            r"(?:single|double)[\s-]*spaced?",
            r"page\s+limit\s+(?:of\s+)?\d+",
            r"maximum\s+(?:of\s+)?\d+\s+pages?",
            r"(?:PDF|Word|Excel)\s+format",
        ],
    }
    
    # Section reference patterns
    SECTION_REF_PATTERN = r"([A-Z])\.(\d+)(?:\.(\d+|[a-z]))?(?:\.(\d+|[a-z]))?"
    
    # Cross-reference patterns
    CROSS_REF_PATTERNS = [
        r"(?:see|refer\s+to|per|as\s+(?:specified|described|defined)\s+in)\s+(?:Section\s+)?([A-Z]\.[\d\.]+)",
        r"(?:Attachment|Exhibit)\s+(\d+|[A-Z])",
        r"FAR\s+(\d+\.\d+(?:-\d+)?)",
        r"DFARS\s+(\d+\.\d+(?:-\d+)?)",
        r"RO\s+([IVX]+)",
        r"Research\s+Outline\s+([IVX]+)",
    ]
    
    def __init__(self, include_context: bool = True, context_chars: int = 200, 
                 strict_mode: bool = True):
        """
        Initialize extractor
        
        Args:
            include_context: Whether to capture surrounding text
            context_chars: How much context to capture
            strict_mode: If True, apply stricter quality filters (recommended)
        """
        self.include_context = include_context
        self.context_chars = context_chars
        self.strict_mode = strict_mode
        self._compile_patterns()
        self._req_counter = 0
        self._seen_hashes = set()  # For duplicate detection
        
        # Load the VisibleThread-style requirement dictionary
        self.req_dictionary = get_requirement_dictionary()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns"""
        self.compiled_mandatory = [(re.compile(p, re.IGNORECASE), name) 
                                   for p, name in self.MANDATORY_PATTERNS]
        self.compiled_conditional = [(re.compile(p, re.IGNORECASE), name) 
                                     for p, name in self.CONDITIONAL_PATTERNS]
        self.compiled_prohibition = [(re.compile(p, re.IGNORECASE), name) 
                                     for p, name in self.PROHIBITION_PATTERNS]
        
        self.compiled_semantic = {
            req_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for req_type, patterns in self.SEMANTIC_PATTERNS.items()
        }
        
        self.compiled_crossref = [re.compile(p, re.IGNORECASE) for p in self.CROSS_REF_PATTERNS]
        
        # Compile noise and boilerplate patterns
        self.compiled_noise = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                               for p in self.NOISE_PATTERNS]
        self.compiled_boilerplate = [re.compile(p, re.IGNORECASE) 
                                     for p in self.BOILERPLATE_PATTERNS]
    
    def _is_noise(self, sentence: str) -> bool:
        """Check if sentence is noise (TOC, header, boilerplate)"""
        sentence_stripped = sentence.strip()
        
        # Check length constraints
        if len(sentence_stripped) < self.MIN_SENTENCE_LENGTH:
            return True
        if len(sentence_stripped) > self.MAX_SENTENCE_LENGTH:
            return True
        
        # Check word count
        words = sentence_stripped.split()
        if len(words) < self.MIN_WORDS:
            return True
        
        # Check noise patterns
        for pattern in self.compiled_noise:
            if pattern.search(sentence_stripped):
                return True
        
        # Check boilerplate patterns
        for pattern in self.compiled_boilerplate:
            if pattern.search(sentence_stripped):
                return True
        
        # Check for excessive special characters (likely garbled text)
        special_char_ratio = sum(1 for c in sentence_stripped if not c.isalnum() and c != ' ') / max(len(sentence_stripped), 1)
        if special_char_ratio > 0.3:
            return True
        
        # Check for TOC-like patterns (multiple dots followed by number)
        if re.search(r'\.{3,}\s*\d+\s*$', sentence_stripped):
            return True
        
        # Filter out mostly uppercase text (headers, titles)
        uppercase_ratio = sum(1 for c in sentence_stripped if c.isupper()) / max(len(sentence_stripped.replace(' ', '')), 1)
        if uppercase_ratio > 0.6:
            return True
        
        # Filter out lines that are mostly numbers/punctuation
        alpha_ratio = sum(1 for c in sentence_stripped if c.isalpha()) / max(len(sentence_stripped), 1)
        if alpha_ratio < 0.5:
            return True
        
        # Filter out clause listing text (e.g., "52.xxx-x Title")
        if re.match(r'^52\.\d{3}[-\d]*\s+', sentence_stripped):
            return True
        
        # Filter out pure reference sentences
        if re.match(r'^(?:See|Refer to|Per|As stated in|In accordance with)\s+(?:Section|Article|Attachment|FAR|DFARS)', 
                    sentence_stripped, re.IGNORECASE):
            if len(sentence_stripped) < 200:  # Short references
                return True
        
        return False
    
    def _has_actor(self, sentence: str) -> bool:
        """Check if sentence has a clear actor (contractor, offeror, government)"""
        actors = [
            r'\b(?:contractor|vendor|offeror|proposer)\b',
            r'\b(?:government|agency|contracting\s+officer|cor)\b',
            r'\bthe\s+(?:contractor|vendor|offeror|government)\b',
        ]
        sentence_lower = sentence.lower()
        return any(re.search(actor, sentence_lower) for actor in actors)
    
    def _is_duplicate(self, text: str) -> bool:
        """Check if we've already seen this requirement"""
        import hashlib
        text_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]
        if text_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(text_hash)
        return False

    def extract_from_document(self, doc: ParsedDocument) -> List[RequirementNode]:
        """
        Extract all requirements from a parsed document
        
        Returns:
            List of RequirementNode objects
        """
        requirements = []
        self._seen_hashes.clear()  # Reset duplicate detection per document
        
        # Clean the document text first (remove HTML entities, normalize)
        cleaned_text = clean_text(doc.full_text)
        
        # Split into sentences for processing
        sentences = self._split_into_sentences(cleaned_text)
        
        for i, sentence in enumerate(sentences):
            # Apply quality filters in strict mode
            if self.strict_mode:
                # Skip noise (TOC, headers, boilerplate)
                if self._is_noise(sentence):
                    continue
                
                # Skip duplicates
                if self._is_duplicate(sentence):
                    continue
            else:
                # Basic filter for non-strict mode
                if len(sentence.strip()) < 20:
                    continue
            
            # Check for requirement indicators
            req_type, keyword_match = self._classify_sentence(sentence)
            
            if req_type:
                # In strict mode, require actor for high confidence
                has_actor = self._has_actor(sentence)
                
                # Skip conditional requirements without actors in strict mode
                if self.strict_mode and keyword_match in ["should", "may", "can"]:
                    if not has_actor:
                        continue
                
                # Create requirement node
                req = self._create_requirement_node(
                    sentence=sentence,
                    sentence_index=i,
                    sentences=sentences,
                    doc=doc,
                    req_type=req_type,
                    keyword_match=keyword_match,
                )
                
                # Adjust confidence based on actor presence
                if has_actor and keyword_match in ["shall", "must", "required"]:
                    req.confidence = ConfidenceLevel.HIGH
                elif has_actor:
                    req.confidence = ConfidenceLevel.MEDIUM
                else:
                    req.confidence = ConfidenceLevel.MEDIUM if keyword_match in ["shall", "must"] else ConfidenceLevel.LOW
                
                requirements.append(req)
        
        # Also extract from sections specifically (but avoid duplicates)
        for section_id, section_text in doc.sections.items():
            section_reqs = self._extract_from_section(section_id, section_text, doc)
            
            # Merge, avoiding duplicates
            for new_req in section_reqs:
                if not self._is_duplicate(new_req.text):
                    requirements.append(new_req)
        
        return requirements
    
    def _classify_sentence(self, sentence: str) -> Tuple[Optional[RequirementType], Optional[str]]:
        """
        Classify a sentence and determine if it's a requirement
        
        Returns:
            (RequirementType or None, matched_keyword or None)
        """
        sentence_lower = sentence.lower()
        
        # First check semantic patterns (more specific)
        for req_type, patterns in self.compiled_semantic.items():
            for pattern in patterns:
                if pattern.search(sentence_lower):
                    # Find the keyword that matched
                    for regex, keyword in self.compiled_mandatory + self.compiled_conditional:
                        if regex.search(sentence_lower):
                            return req_type, keyword
                    return req_type, "semantic"
        
        # Check prohibition patterns
        for regex, keyword in self.compiled_prohibition:
            if regex.search(sentence_lower):
                return RequirementType.PROHIBITION, keyword
        
        # Check mandatory patterns
        for regex, keyword in self.compiled_mandatory:
            if regex.search(sentence_lower):
                return RequirementType.PERFORMANCE, keyword  # Default type for shall/must
        
        # Check conditional patterns (lower priority)
        for regex, keyword in self.compiled_conditional:
            if regex.search(sentence_lower):
                return RequirementType.PERFORMANCE, keyword
        
        return None, None
    
    def _create_requirement_node(
        self,
        sentence: str,
        sentence_index: int,
        sentences: List[str],
        doc: ParsedDocument,
        req_type: RequirementType,
        keyword_match: str,
    ) -> RequirementNode:
        """Create a RequirementNode from a sentence"""
        self._req_counter += 1
        
        # Generate ID based on document type
        type_prefix = doc.document_type.value[:3].upper()
        req_id = f"REQ-{type_prefix}-{self._req_counter:04d}"
        
        # Get context
        context_before = ""
        context_after = ""
        
        if self.include_context:
            if sentence_index > 0:
                context_before = sentences[sentence_index - 1][-self.context_chars:]
            if sentence_index < len(sentences) - 1:
                context_after = sentences[sentence_index + 1][:self.context_chars]
        
        # Find page number
        page_num = self._find_page_number(sentence, doc)
        
        # Find section reference
        section_id = self._extract_section_ref(sentence, doc)
        
        # Extract keywords
        keywords = self._extract_keywords(sentence)
        
        # Extract entities (CLINs, dates, references)
        entities = self._extract_entities(sentence)
        
        # Extract cross-references
        references = self._extract_cross_references(sentence)
        
        # Determine confidence
        confidence = self._assess_confidence(sentence, keyword_match, doc)
        
        # Create source location
        source = SourceLocation(
            document_name=doc.filename,
            document_type=doc.document_type,
            page_number=page_num,
            section_id=section_id,
        )
        
        # Clean the requirement text
        cleaned_sentence = clean_requirement_text(sentence)
        
        return RequirementNode(
            id=req_id,
            text=cleaned_sentence,
            requirement_type=req_type,
            confidence=confidence,
            source=source,
            context_before=context_before,
            context_after=context_after,
            keywords=keywords,
            entities=entities,
            references_to=references,
            extraction_method="regex",
        )
    
    def _extract_from_section(
        self, 
        section_id: str, 
        section_text: str, 
        doc: ParsedDocument
    ) -> List[RequirementNode]:
        """Extract requirements with section context"""
        requirements = []
        
        # Determine expected requirement type based on section
        section_letter = section_id.replace("section_", "").upper()
        
        default_type = {
            "C": RequirementType.PERFORMANCE,
            "L": RequirementType.PROPOSAL_INSTRUCTION,
            "M": RequirementType.EVALUATION_CRITERION,
            "B": RequirementType.LABOR_REQUIREMENT,
            "F": RequirementType.DELIVERABLE,
        }.get(section_letter, RequirementType.PERFORMANCE)
        
        sentences = self._split_into_sentences(section_text)
        
        for i, sentence in enumerate(sentences):
            # Apply quality filters
            if self.strict_mode:
                if self._is_noise(sentence):
                    continue
            else:
                if len(sentence.strip()) < 20:
                    continue
            
            req_type, keyword = self._classify_sentence(sentence)
            
            if req_type:
                # Use section-specific default if generic
                if req_type == RequirementType.PERFORMANCE:
                    req_type = default_type
                
                # In strict mode, skip conditionals without actors
                if self.strict_mode and keyword in ["should", "may", "can"]:
                    if not self._has_actor(sentence):
                        continue
                
                req = self._create_requirement_node(
                    sentence=sentence,
                    sentence_index=i,
                    sentences=sentences,
                    doc=doc,
                    req_type=req_type,
                    keyword_match=keyword or "",
                )
                
                # Override section ID with section context
                if req.source:
                    req.source.section_id = section_letter
                
                requirements.append(req)
        
        return requirements
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Handle common abbreviations
        text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '<DOT>', text)  # Abbreviations
        text = re.sub(r'(?<=\d)\.(?=\d)', '<DOT>', text)  # Numbers
        text = re.sub(r'(?<=\s[A-Z])\.(?=\s)', '<DOT>', text)  # Initials
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return sentences
    
    def _find_page_number(self, sentence: str, doc: ParsedDocument) -> int:
        """Find which page contains this sentence"""
        sentence_start = sentence[:50].lower()
        
        for i, page in enumerate(doc.pages):
            if sentence_start in page.lower():
                return i + 1
        
        return 0
    
    def _extract_section_ref(self, sentence: str, doc: ParsedDocument) -> str:
        """Extract section reference from context"""
        # Look for explicit section references
        match = re.search(self.SECTION_REF_PATTERN, sentence)
        if match:
            parts = [p for p in match.groups() if p]
            return ".".join(parts)
        
        return "UNSPEC"
    
    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract key terms from the requirement"""
        keywords = []
        
        # Technical terms
        tech_patterns = [
            r"(?:data|system|software|hardware|network|security|compliance|report|plan|document)",
            r"(?:training|support|maintenance|development|testing|implementation)",
            r"(?:monthly|weekly|quarterly|annual|daily)",
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, sentence.lower())
            keywords.extend(matches)
        
        # Proper nouns (capitalized words that aren't sentence starters)
        words = sentence.split()
        for i, word in enumerate(words[1:], 1):  # Skip first word
            if word[0].isupper() and len(word) > 2 and word.isalpha():
                keywords.append(word)
        
        return list(set(keywords))[:10]  # Limit to 10 keywords
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract named entities (CLINs, dates, amounts, references)"""
        entities = []
        
        # CLIN numbers
        clin_match = re.findall(r"CLIN\s*(\d+)", sentence, re.IGNORECASE)
        entities.extend([f"CLIN-{c}" for c in clin_match])
        
        # FAR/DFARS references
        far_match = re.findall(r"(FAR|DFARS)\s*(\d+\.\d+)", sentence, re.IGNORECASE)
        entities.extend([f"{f[0]}-{f[1]}" for f in far_match])
        
        # Dates
        date_match = re.findall(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", sentence)
        entities.extend(date_match)
        
        # Dollar amounts
        dollar_match = re.findall(r"\$[\d,]+(?:\.\d{2})?", sentence)
        entities.extend(dollar_match)
        
        # Percentages
        pct_match = re.findall(r"\d+(?:\.\d+)?%", sentence)
        entities.extend(pct_match)
        
        return entities
    
    def _extract_cross_references(self, sentence: str) -> List[str]:
        """Extract references to other sections/documents"""
        references = []
        
        for pattern in self.compiled_crossref:
            matches = pattern.findall(sentence)
            for match in matches:
                if isinstance(match, tuple):
                    references.append("-".join(str(m) for m in match if m))
                else:
                    references.append(str(match))
        
        return references
    
    def _assess_confidence(
        self, 
        sentence: str, 
        keyword: str, 
        doc: ParsedDocument
    ) -> ConfidenceLevel:
        """Assess extraction confidence"""
        # High confidence indicators
        if keyword in ["shall", "must", "required"]:
            if any(phrase in sentence.lower() for phrase in ["contractor shall", "offeror shall", "government will"]):
                return ConfidenceLevel.HIGH
        
        # Medium confidence
        if keyword in ["should", "may", "can"]:
            return ConfidenceLevel.MEDIUM
        
        # Document type affects confidence
        if doc.document_type in [DocumentType.MAIN_SOLICITATION, DocumentType.STATEMENT_OF_WORK]:
            return ConfidenceLevel.HIGH
        
        return ConfidenceLevel.MEDIUM
    
    def _is_duplicate_node(self, new_req: RequirementNode, existing: List[RequirementNode]) -> bool:
        """Check if requirement node is a duplicate of existing nodes"""
        for req in existing:
            if req.text_hash == new_req.text_hash:
                return True
            # Also check for high text similarity
            if self._text_similarity(req.text, new_req.text) > 0.9:
                return True
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def reset_counter(self):
        """Reset requirement ID counter and duplicate tracking"""
        self._req_counter = 0
        self._seen_hashes.clear()
