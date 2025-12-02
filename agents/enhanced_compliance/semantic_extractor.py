"""
PropelAI v2.8: Semantic Requirement Extractor

Implements the vision architecture from the Technical Report:
- Multi-agent orchestration with specialized agents
- Prompt chaining: P1 (Sentence Classifier) → P2 (Requirement Type) → P3 (Data Extraction)
- Proper distinction between PERFORMANCE_REQUIREMENT vs PROPOSAL_INSTRUCTION
- Aggressive pre-filtering to reduce noise
- Cross-reference resolution linking C → L → M

This replaces the naive keyword-based extraction with semantic understanding.
"""

import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Optional anthropic import for LLM classification
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False


class SemanticRequirementType(Enum):
    """Semantic classification of requirements per vision docs"""
    PERFORMANCE_REQUIREMENT = "PERFORMANCE_REQ"      # What contractor must DO
    PROPOSAL_INSTRUCTION = "PROPOSAL_INSTRUCTION"    # What offeror must WRITE
    EVALUATION_CRITERION = "EVALUATION_CRITERION"    # How government will SCORE
    DELIVERABLE = "DELIVERABLE"                      # What must be DELIVERED
    QUALIFICATION = "QUALIFICATION"                  # What offeror must BE/HAVE
    COMPLIANCE_CLAUSE = "COMPLIANCE_CLAUSE"          # FAR/DFARS requirements
    PROHIBITION = "PROHIBITION"                      # What is NOT allowed
    OTHER = "OTHER"                                  # Informational, discard


class RFPSection(Enum):
    """Standard RFP sections per UCF"""
    SECTION_A = "A"   # SF 33 / Solicitation Form
    SECTION_B = "B"   # Supplies or Services
    SECTION_C = "C"   # SOW / PWS / Description
    SECTION_D = "D"   # Packaging and Marking
    SECTION_E = "E"   # Inspection and Acceptance
    SECTION_F = "F"   # Deliveries or Performance
    SECTION_G = "G"   # Contract Administration
    SECTION_H = "H"   # Special Contract Requirements
    SECTION_I = "I"   # Contract Clauses
    SECTION_J = "J"   # Attachments / Exhibits
    SECTION_K = "K"   # Representations / Certifications
    SECTION_L = "L"   # Instructions to Offerors
    SECTION_M = "M"   # Evaluation Factors
    PWS = "PWS"       # Performance Work Statement
    SOW = "SOW"       # Statement of Work
    ATTACHMENT = "ATT"  # Attachment
    UNKNOWN = "UNK"


@dataclass
class ExtractedRequirement:
    """A semantically extracted and classified requirement"""
    id: str
    text: str                                    # Clean, normalized text
    raw_text: str                                # Original text with formatting
    requirement_type: SemanticRequirementType
    rfp_section: RFPSection
    section_reference: str                       # e.g., "L.4.B.2" or "PWS 2.3.1"
    page_number: int
    source_document: str
    
    # Semantic extraction (from P3 prompt)
    action_verb: Optional[str] = None           # e.g., "provide", "submit", "maintain"
    subject: Optional[str] = None               # e.g., "monthly progress reports"
    actor: Optional[str] = None                 # "contractor", "offeror", "government"
    constraints: List[str] = field(default_factory=list)  # Standards, regulations
    
    # Binding level
    is_mandatory: bool = True                   # "shall/must" vs "should/may"
    binding_keyword: str = ""                   # The actual keyword found
    
    # Cross-references
    references_sections: List[str] = field(default_factory=list)
    references_attachments: List[str] = field(default_factory=list)
    related_evaluation_factor: Optional[str] = None
    
    # Quality metrics
    confidence_score: float = 0.0
    priority: str = "MEDIUM"                    # HIGH, MEDIUM, LOW
    
    # Hash for deduplication
    text_hash: str = ""
    
    def __post_init__(self):
        if not self.text_hash:
            self.text_hash = hashlib.md5(self.text.lower().strip().encode()).hexdigest()[:12]


@dataclass 
class ExtractionResult:
    """Results from semantic extraction"""
    requirements: List[ExtractedRequirement]
    stats: Dict[str, Any]
    sections_found: Dict[str, int]
    evaluation_factors: List[Dict[str, Any]]
    warnings: List[str]


class SemanticRequirementExtractor:
    """
    Semantic requirement extraction using LLM-based classification.
    
    Implements the vision architecture:
    1. Aggressive pre-filtering (remove obvious garbage)
    2. Section detection (identify C, L, M, PWS, SOW)
    3. Candidate extraction (find potential requirements)
    4. LLM Classification (P1 → P2 → P3 chain)
    5. Cross-reference resolution
    6. Quality scoring and deduplication
    """
    
    # ========== PRE-FILTERING PATTERNS ==========
    # Things that are NEVER requirements
    GARBAGE_PATTERNS = [
        # Contact information
        r"(?i)^\s*(?:phone|fax|tel|email|e-mail|contact)[\s:]+",
        r"(?i)(?:@[\w.-]+\.\w{2,4})",  # Email addresses
        r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}",  # Phone numbers
        r"(?i)^\s*(?:name|address|city|state|zip)[\s:]+",
        
        # Headers and structural elements
        r"(?i)^(?:section|article|part|chapter)\s+[A-Z0-9][\s\-–—:]*$",
        r"(?i)^(?:table\s+of\s+contents|toc|contents)[\s:]*$",
        r"(?i)^(?:page|pg\.?)\s*\d+\s*(?:of\s*\d+)?$",
        r"^\s*\d+\s*$",  # Just numbers
        r"^[A-Z][.\s]+\d+(?:\.\d+)*\s*$",  # Just section numbers like "C.3.1"
        r"(?i)^(?:attachment|exhibit|appendix)\s+\d+[\s\-–—:]*$",
        r"(?i)^(?:amendment|modification)\s+\d+",
        
        # Document boilerplate
        r"(?i)^\s*(?:request\s+for\s+(?:proposal|quote|information))",
        r"(?i)^\s*(?:standard\s+form|sf\s+)\d+",
        r"(?i)(?:previous\s+edition|unusable|superseded)",
        r"(?i)^\s*\(?\s*(?:continued|cont['.]?d?)\s*\)?[\s:]*$",
        r"(?i)^\s*(?:end\s+of\s+|this\s+page\s+intentionally)",
        r"(?i)^\s*(?:see\s+(?:next|following|attached|continuation))",
        r"(?i)^\s*(?:reserved|n/?a|tbd|to\s+be\s+determined)\s*$",
        
        # Dates and timestamps
        r"(?i)^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}$",
        r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$",
        
        # URLs alone
        r"(?i)^https?://[^\s]+$",
        
        # Clause numbers without content
        r"^\s*(?:FAR|DFARS|HHSAR)\s+\d+\.\d+[-\d]*\s*$",
        r"^\s*52\.\d{3}-\d+\s*$",
        
        # Form field labels
        r"(?i)^\s*(?:signature|date\s+signed|printed\s+name)[\s:]*$",
        r"(?i)^\s*(?:code|cage|duns|uei|sam)[\s:]+",
    ]
    
    # Minimum quality thresholds
    MIN_CHARS = 50               # Minimum characters for a requirement
    MAX_CHARS = 2000             # Maximum (avoid capturing entire sections)
    MIN_WORDS = 10               # Minimum words
    MIN_ALPHA_RATIO = 0.5        # At least 50% alphabetic characters
    
    # Section detection patterns
    SECTION_PATTERNS = {
        RFPSection.SECTION_C: [
            r"(?i)section\s+c[\s\-–—:]+(?:description|statement\s+of\s+work|sow|pws)",
            r"(?i)^c\.\d+\s+",
            r"(?i)statement\s+of\s+work",
            r"(?i)performance\s+work\s+statement",
        ],
        RFPSection.SECTION_L: [
            r"(?i)section\s+l[\s\-–—:]+instruction",
            r"(?i)^l\.\d+\s+",
            r"(?i)instructions?\s+(?:to\s+)?offeror",
            r"(?i)proposal\s+(?:submittal|submission)\s+(?:requirements?|instructions?)",
        ],
        RFPSection.SECTION_M: [
            r"(?i)section\s+m[\s\-–—:]+evaluation",
            r"(?i)^m\.\d+\s+",
            r"(?i)evaluation\s+(?:factors?|criteria)",
            r"(?i)basis\s+for\s+award",
        ],
        RFPSection.PWS: [
            r"(?i)performance\s+work\s+statement",
            r"(?i)^pws\s+\d+",
            r"(?i)attachment.*pws",
        ],
        RFPSection.SOW: [
            r"(?i)statement\s+of\s+work",
            r"(?i)^sow\s+\d+",
            r"(?i)attachment.*sow",
        ],
    }
    
    # Requirement indicator patterns (for candidate selection)
    REQUIREMENT_INDICATORS = [
        r"\bshall\b",
        r"\bmust\b",
        r"\bwill\s+be\s+required\b",
        r"\bis\s+required\s+to\b",
        r"\bare\s+required\s+to\b",
        r"\bshould\b",  # Lower priority
        r"\bmay\b",     # Lowest priority
    ]
    
    # Actor patterns
    ACTOR_PATTERNS = {
        "contractor": r"(?i)\b(?:contractor|vendor|awardee|successful\s+offeror)\b",
        "offeror": r"(?i)\b(?:offeror|proposer|respondent|quoter|bidder)\b",
        "government": r"(?i)\b(?:government|agency|contracting\s+officer|cor|cotr|niH|niehs)\b",
    }
    
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):
        """
        Initialize extractor.
        
        Args:
            use_llm: Whether to use Claude for semantic classification
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        """
        self.use_llm = use_llm and ANTHROPIC_AVAILABLE
        self.client = None
        if self.use_llm and ANTHROPIC_AVAILABLE:
            try:
                self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
                self.use_llm = False
        
        # Compile patterns for performance
        self._compiled_garbage = [re.compile(p) for p in self.GARBAGE_PATTERNS]
        self._compiled_indicators = [re.compile(p, re.IGNORECASE) for p in self.REQUIREMENT_INDICATORS]
        
        # Tracking
        self._req_counter = 0
        self._seen_hashes = set()
        self._current_section = RFPSection.UNKNOWN
        self._warnings = []
    
    def extract(
        self, 
        documents: List[Dict[str, Any]],
        strict_mode: bool = True
    ) -> ExtractionResult:
        """
        Extract requirements from multiple documents.
        
        Args:
            documents: List of dicts with 'text', 'filename', 'pages' keys
            strict_mode: If True, apply aggressive filtering
            
        Returns:
            ExtractionResult with all extracted requirements
        """
        self._req_counter = 0
        self._seen_hashes.clear()
        self._warnings = []
        
        all_requirements = []
        sections_found = {}
        evaluation_factors = []
        
        for doc in documents:
            text = doc.get('text', '')
            filename = doc.get('filename', 'unknown')
            pages = doc.get('pages', [text])
            
            # Step 1: Detect document sections
            doc_sections = self._detect_sections(text, filename)
            for section, count in doc_sections.items():
                sections_found[section.value] = sections_found.get(section.value, 0) + count
            
            # Step 2: Extract candidates from each page
            for page_num, page_text in enumerate(pages, 1):
                candidates = self._extract_candidates(page_text, page_num, filename)
                
                # Step 3: Filter garbage
                filtered = [c for c in candidates if not self._is_garbage(c['text'])]
                
                # Step 4: Classify each candidate
                for candidate in filtered:
                    req = self._classify_and_create(candidate, strict_mode)
                    if req and not self._is_duplicate(req):
                        all_requirements.append(req)
                        self._seen_hashes.add(req.text_hash)
        
        # Step 5: Extract evaluation factors
        evaluation_factors = self._extract_evaluation_factors(documents)
        
        # Step 6: Link requirements to evaluation factors
        self._link_to_evaluation_factors(all_requirements, evaluation_factors)
        
        # Step 7: Score and prioritize
        self._score_requirements(all_requirements)
        
        # Generate stats
        stats = self._generate_stats(all_requirements)
        
        return ExtractionResult(
            requirements=all_requirements,
            stats=stats,
            sections_found=sections_found,
            evaluation_factors=evaluation_factors,
            warnings=self._warnings
        )
    
    def _detect_sections(self, text: str, filename: str) -> Dict[RFPSection, int]:
        """Detect which RFP sections are present in the document"""
        sections = {}
        
        # Check filename first
        filename_lower = filename.lower()
        if 'pws' in filename_lower or 'performance work statement' in filename_lower:
            sections[RFPSection.PWS] = 1
        elif 'sow' in filename_lower or 'statement of work' in filename_lower:
            sections[RFPSection.SOW] = 1
        elif 'section_l' in filename_lower or 'instructions' in filename_lower:
            sections[RFPSection.SECTION_L] = 1
        elif 'section_m' in filename_lower or 'evaluation' in filename_lower:
            sections[RFPSection.SECTION_M] = 1
        
        # Check content patterns
        for section, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    sections[section] = sections.get(section, 0) + len(matches)
        
        return sections
    
    def _extract_candidates(
        self, 
        text: str, 
        page_num: int, 
        filename: str
    ) -> List[Dict[str, Any]]:
        """Extract candidate requirement sentences from text"""
        candidates = []
        
        # Split into paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            # Split paragraphs into sentences
            sentences = self._split_sentences(para)
            
            for sent in sentences:
                # Check if sentence contains a requirement indicator
                has_indicator = any(p.search(sent) for p in self._compiled_indicators)
                
                if has_indicator:
                    # Extract section reference from context
                    section_ref = self._extract_section_reference(sent, para)
                    rfp_section = self._determine_rfp_section(section_ref, para, filename)
                    
                    candidates.append({
                        'text': sent.strip(),
                        'raw_text': sent,
                        'page_num': page_num,
                        'filename': filename,
                        'section_ref': section_ref,
                        'rfp_section': rfp_section,
                        'context': para[:200],  # Keep some context
                    })
        
        return candidates
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling abbreviations"""
        # Protect common abbreviations
        protected = text
        abbreviations = [
            (r'(\b[A-Z])\.(\s)', r'\1<DOT>\2'),  # Single letter abbrevs
            (r'(\d)\.(\d)', r'\1<DOT>\2'),        # Numbers
            (r'(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|Inc|Corp|Ltd|etc|vs|i\.e|e\.g)\.', r'\1<DOT>'),
            (r'(FAR|DFARS|CFR|U\.S|U\.S\.C)\.', r'\1<DOT>'),
        ]
        
        for pattern, replacement in abbreviations:
            protected = re.sub(pattern, replacement, protected)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter by length
        sentences = [s for s in sentences if len(s) >= self.MIN_CHARS]
        
        return sentences
    
    def _extract_section_reference(self, sentence: str, context: str) -> str:
        """Extract section reference (e.g., L.4.B.2, C.3.1, PWS 2.1)"""
        patterns = [
            r'([LCMHFGJB])\.(\d+)(?:\.([A-Z]|\d+))?(?:\.(\d+))?',  # L.4.B.2
            r'(?:PWS|SOW)\s+(\d+)(?:\.(\d+))?(?:\.(\d+))?',         # PWS 2.1.3
            r'Section\s+([A-Z])(?:\.(\d+))?',                       # Section L.4
            r'(?:Article|Paragraph)\s+([A-Z])\.(\d+)',              # Article L.4
        ]
        
        # Check sentence first
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                parts = [p for p in match.groups() if p]
                return '.'.join(parts)
        
        # Check context
        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                parts = [p for p in match.groups() if p]
                return '.'.join(parts)
        
        return ""
    
    # Section header patterns for enhanced detection
    SECTION_HEADER_PATTERNS = {
        RFPSection.SECTION_L: [
            r'SECTION\s*L[\s:\-–—]+',
            r'PART\s+(?:IV|4)[\s:\-–—]+.*INSTRUCTION',
            r'INSTRUCTIONS?\s*(?:,\s*)?(?:CONDITIONS?\s*(?:,\s*)?)?(?:AND\s+)?NOTICES?\s+TO\s+OFFERORS?',
            r'PROPOSAL\s+(?:SUBMISSION\s+)?(?:REQUIREMENTS?|INSTRUCTIONS?)',
            r'ARTICLE\s+L\.\d+',
            r'L\.\d+\s+[-–—]\s*[A-Z]',
        ],
        RFPSection.SECTION_M: [
            r'SECTION\s*M[\s:\-–—]+',
            r'EVALUATION\s+(?:FACTORS?|CRITERIA)\s+(?:FOR\s+)?(?:AWARD)?',
            r'BASIS\s+(?:FOR\s+)?(?:CONTRACT\s+)?AWARD',
            r'SOURCE\s+SELECTION\s+(?:CRITERIA|FACTORS?)',
            r'ARTICLE\s+M\.\d+',
            r'M\.\d+\s+[-–—]\s*[A-Z]',
            # NIH-specific
            r'REVIEW\s+(?:AND\s+)?SELECTION\s+PROCESS',
            r'EVALUATION\s+OF\s+(?:TECHNICAL\s+)?PROPOSALS?',
        ],
        RFPSection.SECTION_C: [
            r'SECTION\s*C[\s:\-–—]+',
            r'DESCRIPTION[/\s]+SPECIFICATIONS?[/\s]+(?:SOW|WORK\s+STATEMENT)',
            r'STATEMENT\s+OF\s+(?:WORK|OBJECTIVES?)',
            r'SCOPE\s+OF\s+(?:WORK|CONTRACT)',
            r'ARTICLE\s+C\.\d+',
            r'C\.\d+\s+[-–—]\s*[A-Z]',
            # Task order specific
            r'TASK\s+(?:ORDER\s+)?REQUIREMENTS?',
            r'CONTRACT\s+REQUIREMENTS?',
        ],
        RFPSection.PWS: [
            r'PERFORMANCE\s+WORK\s+STATEMENT',
            r'PWS[\s:\-–—]+',
            r'ATTACHMENT\s+\d+[\s:\-–—]+.*PWS',
        ],
        RFPSection.SOW: [
            r'STATEMENT\s+OF\s+WORK',
            r'SOW[\s:\-–—]+',
            r'ATTACHMENT\s+\d+[\s:\-–—]+.*SOW',
        ],
        RFPSection.SECTION_F: [
            r'SECTION\s*F[\s:\-–—]+',
            r'DELIVERIES?\s+(?:OR\s+)?PERFORMANCE',
            r'PERIOD\s+OF\s+PERFORMANCE',
        ],
        RFPSection.SECTION_H: [
            r'SECTION\s*H[\s:\-–—]+',
            r'SPECIAL\s+CONTRACT\s+REQUIREMENTS?',
        ],
        RFPSection.SECTION_J: [
            r'SECTION\s*J[\s:\-–—]+',
            r'LIST\s+OF\s+(?:DOCUMENTS?|ATTACHMENTS?|EXHIBITS?)',
        ],
        RFPSection.SECTION_B: [
            r'SECTION\s*B[\s:\-–—]+',
            r'SUPPLIES?\s+(?:OR\s+)?SERVICES?\s+AND\s+PRICES?',
            r'CONTRACT\s+LINE\s+ITEMS?',
            r'CLIN\s+STRUCTURE',
        ],
        RFPSection.SECTION_K: [
            r'SECTION\s*K[\s:\-–—]+',
            r'REPRESENTATIONS?\s*(?:,\s*)?CERTIFICATIONS?\s*(?:,\s*)?(?:AND\s+)?OTHER',
        ],
    }

    # Content-based heuristics for section inference (when no header found)
    SECTION_CONTENT_PATTERNS = {
        RFPSection.SECTION_L: [
            # Proposal instruction indicators
            (r'\b(?:offeror|proposer)s?\s+(?:shall|must|should)\s+(?:submit|provide|include|describe|address|demonstrate)', 0.8),
            (r'\b(?:technical|business|cost|price|management)\s+(?:proposal|volume)\s+(?:shall|must|should)', 0.8),
            (r'\bproposal\s+(?:format|organization|structure|preparation)', 0.7),
            (r'\bpage\s+limit(?:ation)?s?\b', 0.6),
            (r'\bfont\s+(?:size|type)|(?:single|double)[- ]?spac(?:ed?|ing)', 0.6),
            (r'\bsubmission\s+(?:requirements?|instructions?|deadline)', 0.7),
            (r'\bvolume\s+[IVX]+\b|\bvolume\s+\d+\b', 0.6),
            # NIH proposal patterns
            (r'\bresearch\s+(?:strategy|plan|approach)\s+(?:shall|must|should)', 0.7),
            (r'\bspecific\s+aims?\s+(?:page|section)', 0.7),
            (r'\bbiosketch(?:es)?\b', 0.6),
        ],
        RFPSection.SECTION_M: [
            # Evaluation indicators
            (r'\b(?:government|agency)\s+(?:will|shall)\s+(?:evaluate|assess|review|consider)', 0.9),
            (r'\bevaluation\s+(?:factor|criteria|element)s?\b', 0.8),
            (r'\b(?:adjectival|color)\s+ratings?\b', 0.7),
            (r'\b(?:outstanding|excellent|good|acceptable|marginal|unacceptable)\b.*\b(?:rating|score)', 0.7),
            (r'\bpass[/-]fail\b', 0.7),
            (r'\b(?:strengths?|weaknesses?|deficienc(?:y|ies)|significant|risk)\b.*\b(?:evaluat|assess)', 0.6),
            (r'\b(?:more|less|equally)\s+important\s+than\b', 0.8),
            (r'\bbest\s+value\s+(?:determination|tradeoff)', 0.8),
            (r'\b(?:technical|management|past\s+performance)\s+(?:is|are)\s+(?:more|less)\s+important', 0.9),
            # NIH review patterns
            (r'\boverall\s+impact(?:/priority)?\s+score', 0.8),
            (r'\b(?:significance|innovation|approach|investigator|environment)\s+(?:score|criterion)', 0.7),
        ],
        RFPSection.SECTION_C: [
            # Performance/SOW indicators
            (r'\bcontractor\s+(?:shall|must|will)\s+(?:provide|perform|deliver|maintain|support|develop)', 0.8),
            (r'\bthe\s+work\s+(?:shall|will)\s+(?:include|consist|be\s+performed)', 0.7),
            (r'\btask(?:s|ing)?\s+(?:shall|will)\s+(?:include|be)', 0.6),
            (r'\bservices?\s+(?:shall|will)\s+(?:include|be\s+provided)', 0.7),
            (r'\bdeliverable(?:s)?\s+(?:shall|will)\s+(?:include|be)', 0.7),
            (r'\b(?:scope|objective)s?\s+of\s+(?:work|services?|contract)', 0.7),
            (r'\bperformance\s+(?:period|requirements?|objectives?)', 0.6),
            (r'\bquality\s+(?:assurance|control)\s+(?:surveillance\s+)?plan', 0.6),
        ],
        RFPSection.PWS: [
            (r'\bperformance\s+work\s+statement', 0.9),
            (r'\bpws\s+(?:section|paragraph|requirement)', 0.8),
            (r'\bperformance\s+(?:standard|objective|requirement)s?\b', 0.6),
        ],
        RFPSection.SOW: [
            (r'\bstatement\s+of\s+work', 0.9),
            (r'\bsow\s+(?:section|paragraph|requirement)', 0.8),
        ],
    }

    def _determine_rfp_section(self, section_ref: str, context: str, filename: str) -> RFPSection:
        """
        Determine which RFP section this requirement belongs to.

        Uses a multi-stage approach:
        1. Explicit section reference (L.4.B.2, C.3.1)
        2. Section header patterns in context
        3. Content-based semantic heuristics
        4. Filename patterns
        5. Fallback to UNKNOWN only if all else fails
        """
        # Stage 1: Check explicit section reference
        if section_ref:
            first_char = section_ref[0].upper()
            section_map = {
                'L': RFPSection.SECTION_L,
                'M': RFPSection.SECTION_M,
                'C': RFPSection.SECTION_C,
                'F': RFPSection.SECTION_F,
                'H': RFPSection.SECTION_H,
                'J': RFPSection.SECTION_J,
                'B': RFPSection.SECTION_B,
                'K': RFPSection.SECTION_K,
                'A': RFPSection.SECTION_A,
                'D': RFPSection.SECTION_D,
                'E': RFPSection.SECTION_E,
                'G': RFPSection.SECTION_G,
                'I': RFPSection.SECTION_I,
            }
            if first_char in section_map:
                return section_map[first_char]
            # Handle PWS/SOW references
            if section_ref.upper().startswith('PWS'):
                return RFPSection.PWS
            if section_ref.upper().startswith('SOW'):
                return RFPSection.SOW

        # Stage 2: Check context for section header patterns
        context_upper = context.upper()
        for section, patterns in self.SECTION_HEADER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, context_upper):
                    return section

        # Stage 3: Content-based semantic heuristics
        section_scores = {}
        context_lower = context.lower()
        for section, patterns in self.SECTION_CONTENT_PATTERNS.items():
            score = 0.0
            for pattern, weight in patterns:
                if re.search(pattern, context_lower, re.IGNORECASE):
                    score += weight
            if score > 0:
                section_scores[section] = score

        # Return section with highest score if above threshold
        if section_scores:
            best_section = max(section_scores.items(), key=lambda x: x[1])
            if best_section[1] >= 0.6:  # Confidence threshold
                return best_section[0]

        # Stage 4: Check filename patterns
        filename_lower = filename.lower()
        filename_patterns = [
            (r'pws|performance.?work.?statement', RFPSection.PWS),
            (r'sow|statement.?of.?work', RFPSection.SOW),
            (r'section[_\-\s]?l|instructions?', RFPSection.SECTION_L),
            (r'section[_\-\s]?m|evaluation', RFPSection.SECTION_M),
            (r'section[_\-\s]?c|description|specification', RFPSection.SECTION_C),
            (r'attachment|exhibit|appendix', RFPSection.ATTACHMENT),
            (r'amendment|modification|mod\d+', RFPSection.ATTACHMENT),  # Amendments often have requirements
        ]
        for pattern, section in filename_patterns:
            if re.search(pattern, filename_lower):
                return section

        # Stage 5: Last resort - infer from current tracking section
        if hasattr(self, '_current_section') and self._current_section != RFPSection.UNKNOWN:
            return self._current_section

        return RFPSection.UNKNOWN
    
    def _is_garbage(self, text: str) -> bool:
        """Check if text is garbage that should be filtered out"""
        # Length checks
        if len(text) < self.MIN_CHARS:
            return True
        if len(text) > self.MAX_CHARS:
            return True
        
        words = text.split()
        if len(words) < self.MIN_WORDS:
            return True
        
        # Alpha ratio check
        alpha_count = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha_count / len(text) < self.MIN_ALPHA_RATIO:
            return True
        
        # Pattern checks
        for pattern in self._compiled_garbage:
            if pattern.search(text):
                return True
        
        # Check for address-like content
        if self._looks_like_address(text):
            return True
        
        return False
    
    def _looks_like_address(self, text: str) -> bool:
        """Check if text looks like an address or contact info"""
        # Multiple newlines with short segments often = address
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 3:
            avg_line_len = sum(len(l) for l in lines) / len(lines)
            if avg_line_len < 40:  # Short lines = likely address
                # Check for address indicators
                address_indicators = ['street', 'avenue', 'road', 'drive', 'suite', 
                                     'building', 'floor', 'po box', 'zip', 'dc ', 
                                     'md ', 'va ', 'nc ']
                text_lower = text.lower()
                if any(ind in text_lower for ind in address_indicators):
                    return True
                # Check for ZIP code pattern
                if re.search(r'\b\d{5}(?:-\d{4})?\b', text):
                    return True
        
        return False
    
    def _classify_and_create(
        self, 
        candidate: Dict[str, Any],
        strict_mode: bool
    ) -> Optional[ExtractedRequirement]:
        """Classify candidate and create requirement if valid"""
        text = candidate['text']
        
        # Determine requirement type and binding
        req_type, is_mandatory, keyword = self._classify_type(text, candidate)
        
        # In strict mode, only keep high-value requirements
        if strict_mode:
            if req_type == SemanticRequirementType.OTHER:
                return None
            # Skip weak optional statements without actors
            if not is_mandatory and not self._has_actor(text):
                return None
        
        # Extract semantic elements
        action_verb = self._extract_action_verb(text)
        subject = self._extract_subject(text)
        actor = self._extract_actor(text)
        constraints = self._extract_constraints(text)
        
        # Extract cross-references
        refs_sections, refs_attachments = self._extract_references(text)
        
        # Clean the text
        clean_text = self._clean_text(text)
        
        # Create requirement
        self._req_counter += 1
        req_id = f"REQ-{candidate['rfp_section'].value}-{self._req_counter:04d}"
        
        return ExtractedRequirement(
            id=req_id,
            text=clean_text,
            raw_text=candidate['raw_text'],
            requirement_type=req_type,
            rfp_section=candidate['rfp_section'],
            section_reference=candidate['section_ref'] or "UNSPEC",
            page_number=candidate['page_num'],
            source_document=candidate['filename'],
            action_verb=action_verb,
            subject=subject,
            actor=actor,
            constraints=constraints,
            is_mandatory=is_mandatory,
            binding_keyword=keyword,
            references_sections=refs_sections,
            references_attachments=refs_attachments,
        )
    
    def _classify_type(
        self, 
        text: str, 
        candidate: Dict[str, Any]
    ) -> Tuple[SemanticRequirementType, bool, str]:
        """
        Classify requirement type using semantic patterns.
        
        Returns: (type, is_mandatory, keyword)
        """
        text_lower = text.lower()
        
        # Detect binding level
        is_mandatory = True
        keyword = ""
        
        if re.search(r'\bshall\s+not\b', text_lower):
            is_mandatory = True
            keyword = "shall not"
            return (SemanticRequirementType.PROHIBITION, is_mandatory, keyword)
        elif re.search(r'\bmust\s+not\b', text_lower):
            is_mandatory = True
            keyword = "must not"
            return (SemanticRequirementType.PROHIBITION, is_mandatory, keyword)
        elif re.search(r'\bshall\b', text_lower):
            keyword = "shall"
        elif re.search(r'\bmust\b', text_lower):
            keyword = "must"
        elif re.search(r'\brequired\b', text_lower):
            keyword = "required"
        elif re.search(r'\bshould\b', text_lower):
            is_mandatory = False
            keyword = "should"
        elif re.search(r'\bmay\b', text_lower):
            is_mandatory = False
            keyword = "may"
        
        # Classify by semantic patterns
        rfp_section = candidate.get('rfp_section', RFPSection.UNKNOWN)
        
        # Check for proposal instruction patterns
        proposal_patterns = [
            r'(?i)offeror[s]?\s+(?:shall|must|should)\s+(?:describe|provide|submit|include|address|demonstrate)',
            r'(?i)proposal[s]?\s+(?:shall|must|should)\s+(?:include|contain|address|describe)',
            r'(?i)(?:technical|business|cost)\s+proposal\s+(?:shall|must|should)',
            r'(?i)(?:submit|provide)\s+(?:a|the)\s+(?:description|plan|approach)',
        ]
        for pattern in proposal_patterns:
            if re.search(pattern, text):
                return (SemanticRequirementType.PROPOSAL_INSTRUCTION, is_mandatory, keyword)
        
        # Check for evaluation patterns
        eval_patterns = [
            r'(?i)(?:will|shall)\s+be\s+evaluated',
            r'(?i)government\s+will\s+(?:evaluate|assess|consider)',
            r'(?i)evaluation\s+(?:will|shall)\s+(?:be\s+based|consider)',
            r'(?i)(?:more|less|most)\s+important\s+than',
        ]
        for pattern in eval_patterns:
            if re.search(pattern, text):
                return (SemanticRequirementType.EVALUATION_CRITERION, is_mandatory, keyword)
        
        # Check for deliverable patterns
        deliverable_patterns = [
            r'(?i)(?:submit|deliver|provide)\s+(?:a|the)\s+(?:\w+\s+)?report',
            r'(?i)deliverable[s]?\s+(?:include|are|shall)',
            r'(?i)due\s+(?:within|by|on|no\s+later\s+than)',
            r'(?i)(?:monthly|weekly|quarterly|final)\s+(?:report|deliverable)',
        ]
        for pattern in deliverable_patterns:
            if re.search(pattern, text):
                return (SemanticRequirementType.DELIVERABLE, is_mandatory, keyword)
        
        # Check for qualification patterns
        qual_patterns = [
            r'(?i)(?:must|shall)\s+(?:have|possess|demonstrate|be)',
            r'(?i)(?:minimum|required)\s+(?:qualification|experience|certification)',
            r'(?i)(?:certified|licensed|cleared)\s+(?:in|as|by|to)',
        ]
        for pattern in qual_patterns:
            if re.search(pattern, text):
                # Only if it's about qualifications, not general performance
                if re.search(r'(?i)(?:qualification|experience|certification|clearance|license)', text):
                    return (SemanticRequirementType.QUALIFICATION, is_mandatory, keyword)
        
        # Check for compliance patterns
        compliance_patterns = [
            r'(?i)FAR\s+\d+\.\d+',
            r'(?i)DFARS\s+\d+\.\d+',
            r'(?i)(?:in\s+accordance\s+with|comply\s+with|compliant\s+with)',
            r'(?i)Section\s+508',
            r'(?i)NIST\s+(?:SP\s+)?800',
        ]
        for pattern in compliance_patterns:
            if re.search(pattern, text):
                return (SemanticRequirementType.COMPLIANCE_CLAUSE, is_mandatory, keyword)
        
        # Default based on section
        if rfp_section == RFPSection.SECTION_L:
            return (SemanticRequirementType.PROPOSAL_INSTRUCTION, is_mandatory, keyword)
        elif rfp_section == RFPSection.SECTION_M:
            return (SemanticRequirementType.EVALUATION_CRITERION, is_mandatory, keyword)
        elif rfp_section in [RFPSection.SECTION_C, RFPSection.PWS, RFPSection.SOW]:
            return (SemanticRequirementType.PERFORMANCE_REQUIREMENT, is_mandatory, keyword)
        
        # Default to performance requirement if has binding keyword
        if keyword:
            return (SemanticRequirementType.PERFORMANCE_REQUIREMENT, is_mandatory, keyword)
        
        return (SemanticRequirementType.OTHER, False, "")
    
    def _has_actor(self, text: str) -> bool:
        """Check if text contains an actor (contractor/offeror/government)"""
        for actor, pattern in self.ACTOR_PATTERNS.items():
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_actor(self, text: str) -> Optional[str]:
        """Extract the actor from the requirement"""
        for actor, pattern in self.ACTOR_PATTERNS.items():
            if re.search(pattern, text):
                return actor
        return None
    
    def _extract_action_verb(self, text: str) -> Optional[str]:
        """Extract the primary action verb"""
        # Common action verbs in requirements
        verbs = [
            'provide', 'submit', 'deliver', 'maintain', 'ensure', 'perform',
            'develop', 'create', 'prepare', 'conduct', 'support', 'manage',
            'implement', 'establish', 'document', 'report', 'analyze',
            'evaluate', 'assess', 'describe', 'demonstrate', 'address',
            'include', 'comply', 'coordinate', 'participate', 'attend',
        ]
        
        text_lower = text.lower()
        
        # Look for "shall/must/should [verb]" pattern
        pattern = r'(?:shall|must|should|will)\s+(\w+)'
        match = re.search(pattern, text_lower)
        if match:
            verb = match.group(1)
            if verb in verbs or verb.endswith('e') and verb[:-1] in verbs:
                return verb
        
        # Fall back to first verb found
        for verb in verbs:
            if verb in text_lower:
                return verb
        
        return None
    
    def _extract_subject(self, text: str) -> Optional[str]:
        """Extract the subject/object of the requirement"""
        # Look for what comes after the action verb
        pattern = r'(?:shall|must|should|will)\s+\w+\s+((?:a|an|the)\s+)?(.{10,50}?)(?:\.|,|;|$)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            subject = match.group(2).strip()
            # Clean up
            subject = re.sub(r'\s+', ' ', subject)
            return subject[:100]  # Limit length
        return None
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints, standards, and regulations"""
        constraints = []
        
        patterns = [
            (r'(?i)in\s+accordance\s+with\s+(.{10,50}?)(?:\.|,|;|$)', 'IAW'),
            (r'(?i)FAR\s+(\d+\.\d+[-\d]*)', 'FAR'),
            (r'(?i)DFARS\s+(\d+\.\d+[-\d]*)', 'DFARS'),
            (r'(?i)NIST\s+(?:SP\s+)?(\d+[-\d]*)', 'NIST'),
            (r'(?i)within\s+(\d+\s+(?:days?|hours?|weeks?|months?))', 'TIMEFRAME'),
        ]
        
        for pattern, prefix in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                constraints.append(f"{prefix}: {match.strip()}")
        
        return constraints
    
    def _extract_references(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract cross-references to other sections and attachments"""
        section_refs = []
        attachment_refs = []
        
        # Section references
        section_patterns = [
            r'(?i)(?:see\s+)?Section\s+([A-Z])(?:\.(\d+))?',
            r'(?i)(?:see\s+)?Article\s+([A-Z])\.(\d+)',
            r'(?i)(?:reference|refer\s+to)\s+([A-Z])\.(\d+)',
        ]
        for pattern in section_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                ref = '.'.join(str(p) for p in match if p)
                if ref and ref not in section_refs:
                    section_refs.append(ref)
        
        # Attachment references
        attachment_patterns = [
            r'(?i)Attachment\s+(\d+|[A-Z])',
            r'(?i)Exhibit\s+(\d+|[A-Z])',
            r'(?i)Appendix\s+(\d+|[A-Z])',
            r'(?i)(?:DD\s*)?1423\s+(\w+)',
        ]
        for pattern in attachment_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                ref = f"ATT-{match}"
                if ref not in attachment_refs:
                    attachment_refs.append(ref)
        
        return section_refs, attachment_refs
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize requirement text"""
        # Collapse whitespace
        clean = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        clean = clean.strip()
        # Remove any control characters
        clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean)
        return clean
    
    def _is_duplicate(self, req: ExtractedRequirement) -> bool:
        """Check if requirement is a duplicate"""
        if req.text_hash in self._seen_hashes:
            return True
        
        # Could add fuzzy matching here for near-duplicates
        return False
    
    def _extract_evaluation_factors(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract evaluation factors from Section M"""
        factors = []
        
        for doc in documents:
            text = doc.get('text', '')
            filename = doc.get('filename', '')
            
            # Look for factor patterns
            factor_patterns = [
                r'(?i)Factor\s+(\d+)[:\s]+([^\n]+)',
                r'(?i)Evaluation\s+Factor\s+(\d+)[:\s]+([^\n]+)',
                r'(?i)(\d+)\.\s+(Technical|Past\s+Performance|Price|Cost)[^\n]*',
            ]
            
            for pattern in factor_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    factor = {
                        'number': match[0],
                        'name': match[1].strip() if len(match) > 1 else '',
                        'source': filename
                    }
                    if factor not in factors:
                        factors.append(factor)
        
        return factors
    
    def _link_to_evaluation_factors(
        self, 
        requirements: List[ExtractedRequirement],
        factors: List[Dict[str, Any]]
    ):
        """Link requirements to their related evaluation factors"""
        factor_keywords = {}
        for f in factors:
            name = f.get('name', '').lower()
            keywords = name.split()
            factor_keywords[f['number']] = keywords
        
        for req in requirements:
            text_lower = req.text.lower()
            for factor_num, keywords in factor_keywords.items():
                # Check if requirement mentions factor keywords
                if any(kw in text_lower for kw in keywords if len(kw) > 3):
                    req.related_evaluation_factor = f"Factor {factor_num}"
                    break
    
    def _score_requirements(self, requirements: List[ExtractedRequirement]):
        """Score and prioritize requirements"""
        for req in requirements:
            score = 0.5  # Base score
            
            # Mandatory = higher priority
            if req.is_mandatory:
                score += 0.2
            
            # Has clear actor = higher confidence
            if req.actor:
                score += 0.1
            
            # Has section reference = higher confidence
            if req.section_reference and req.section_reference != "UNSPEC":
                score += 0.1
            
            # Linked to evaluation = higher priority
            if req.related_evaluation_factor:
                score += 0.1
            
            # Type-based adjustments
            if req.requirement_type in [
                SemanticRequirementType.PERFORMANCE_REQUIREMENT,
                SemanticRequirementType.PROPOSAL_INSTRUCTION
            ]:
                score += 0.1
            elif req.requirement_type == SemanticRequirementType.EVALUATION_CRITERION:
                score += 0.15
            
            req.confidence_score = min(score, 1.0)
            
            # Set priority
            if req.confidence_score >= 0.8:
                req.priority = "HIGH"
            elif req.confidence_score >= 0.6:
                req.priority = "MEDIUM"
            else:
                req.priority = "LOW"
    
    def _generate_stats(self, requirements: List[ExtractedRequirement]) -> Dict[str, Any]:
        """Generate extraction statistics"""
        stats = {
            'total': len(requirements),
            'by_type': {},
            'by_section': {},
            'by_priority': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
            'mandatory': 0,
            'desirable': 0,
        }
        
        for req in requirements:
            # By type
            type_name = req.requirement_type.value
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1
            
            # By section
            section_name = req.rfp_section.value
            stats['by_section'][section_name] = stats['by_section'].get(section_name, 0) + 1
            
            # By priority
            stats['by_priority'][req.priority] += 1
            
            # Binding level
            if req.is_mandatory:
                stats['mandatory'] += 1
            else:
                stats['desirable'] += 1
        
        return stats


# ============================================================
# OPTIONAL: LLM-ENHANCED CLASSIFICATION
# ============================================================

class LLMEnhancedExtractor(SemanticRequirementExtractor):
    """
    Enhanced extractor that uses Claude for difficult classifications.
    
    Implements the P1 → P2 → P3 prompt chain from the vision docs:
    - P1: Sentence Classifier (REQUIREMENT, INSTRUCTION, EVALUATION, OTHER)
    - P2: Requirement Type (PERFORMANCE_REQ vs PROPOSAL_INSTRUCTION)
    - P3: Data Extraction (action verb, subject, constraints)
    """
    
    # Prompt templates from vision docs
    P1_CLASSIFIER = """Analyze the following sentence from a US Government RFP. 
Respond with only one of these classifications:
- REQUIREMENT: A binding obligation on the contractor or offeror
- INSTRUCTION: Direction for preparing the proposal
- EVALUATION: How the government will evaluate proposals
- OTHER: Informational text, not a requirement

Sentence: "{text}"

Classification:"""

    P2_TYPE_CLASSIFIER = """You are an expert in government proposal compliance.
Analyze this statement from an RFP. Determine if it is:
- PERFORMANCE_REQUIREMENT: Action the contractor must perform AFTER winning
- PROPOSAL_INSTRUCTION: Action the offeror must take IN THEIR PROPOSAL

Statement: "{text}"

Classification:"""

    P3_DATA_EXTRACTOR = """From this government RFP requirement, extract:
- action_verb: Primary action (provide, submit, maintain, etc.)
- subject: What is being acted upon
- actor: Who must act (contractor, offeror, government)
- constraints: Any standards, regulations, or conditions

Requirement: "{text}"

Respond in JSON format:
{{"action_verb": "...", "subject": "...", "actor": "...", "constraints": ["..."]}}"""

    def _classify_with_llm(self, text: str) -> Optional[SemanticRequirementType]:
        """Use Claude to classify difficult cases"""
        if not self.client:
            return None
        
        try:
            # P1: Initial classification
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": self.P1_CLASSIFIER.format(text=text)
                }]
            )
            
            p1_result = response.content[0].text.strip().upper()
            
            if 'OTHER' in p1_result:
                return SemanticRequirementType.OTHER
            elif 'EVALUATION' in p1_result:
                return SemanticRequirementType.EVALUATION_CRITERION
            elif 'INSTRUCTION' in p1_result:
                return SemanticRequirementType.PROPOSAL_INSTRUCTION
            elif 'REQUIREMENT' in p1_result:
                # P2: Distinguish performance vs proposal
                response2 = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=50,
                    messages=[{
                        "role": "user",
                        "content": self.P2_TYPE_CLASSIFIER.format(text=text)
                    }]
                )
                p2_result = response2.content[0].text.strip().upper()
                
                if 'PROPOSAL' in p2_result:
                    return SemanticRequirementType.PROPOSAL_INSTRUCTION
                else:
                    return SemanticRequirementType.PERFORMANCE_REQUIREMENT
            
        except Exception as e:
            self._warnings.append(f"LLM classification failed: {e}")
            return None
        
        return None
