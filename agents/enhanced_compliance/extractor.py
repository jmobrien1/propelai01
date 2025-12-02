"""
PropelAI Cycle 5: Requirement Extractor
Multi-pattern extraction with semantic classification

Extracts requirements from ALL sections, not just Section C
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .models import (
    RequirementNode, RequirementType, ConfidenceLevel, 
    RequirementStatus, SourceLocation, ParsedDocument, DocumentType
)


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
    
    # ============================================================================
    # ENHANCED KEYWORD DICTIONARY v2.0
    # Per accuracy.txt: Broader coverage, contextual patterns, category labels
    # ============================================================================

    # Requirement verb synonyms - expanded from base forms
    REQUIREMENT_VERBS = {
        # Primary action verbs with synonyms
        "provide": ["provide", "furnish", "supply", "deliver", "give", "offer"],
        "perform": ["perform", "execute", "conduct", "carry out", "accomplish", "complete"],
        "maintain": ["maintain", "sustain", "preserve", "keep", "uphold", "continue"],
        "ensure": ["ensure", "guarantee", "assure", "verify", "confirm", "certify"],
        "support": ["support", "assist", "aid", "help", "facilitate", "enable"],
        "develop": ["develop", "create", "design", "build", "construct", "establish"],
        "submit": ["submit", "present", "deliver", "provide", "furnish", "send"],
        "include": ["include", "contain", "incorporate", "comprise", "encompass"],
        "describe": ["describe", "explain", "detail", "outline", "specify", "document"],
        "demonstrate": ["demonstrate", "show", "prove", "establish", "evidence", "illustrate"],
        "comply": ["comply", "adhere", "conform", "follow", "observe", "meet"],
        "implement": ["implement", "deploy", "execute", "install", "establish", "put in place"],
        "manage": ["manage", "oversee", "supervise", "direct", "coordinate", "administer"],
        "report": ["report", "document", "record", "log", "notify", "inform"],
        "train": ["train", "educate", "instruct", "teach", "prepare", "qualify"],
    }

    # Contextual patterns with confidence weights
    # Higher weight = stronger requirement indicator
    # Format: (pattern, keyword_label, confidence_weight, binding_level)
    CONTEXTUAL_MANDATORY_PATTERNS = [
        # Strongest indicators (weight 1.0) - explicit actor + shall/must
        (r"\bcontractor\s+shall\s+(?:provide|perform|maintain|ensure|support|develop|deliver|implement|manage)", "contractor_shall", 1.0, "MANDATORY"),
        (r"\bcontractor\s+must\s+(?:provide|perform|maintain|ensure|support|develop|deliver|implement|manage)", "contractor_must", 1.0, "MANDATORY"),
        (r"\bofferor\s+shall\s+(?:submit|provide|include|describe|demonstrate|address|explain)", "offeror_shall", 1.0, "MANDATORY"),
        (r"\bofferor\s+must\s+(?:submit|provide|include|describe|demonstrate|address|explain)", "offeror_must", 1.0, "MANDATORY"),
        (r"\bgovernment\s+(?:will|shall)\s+(?:evaluate|assess|review|consider|rate|score)", "government_will", 1.0, "EVALUATION"),

        # Strong indicators (weight 0.9) - shall/must with context
        (r"\bshall\s+(?:be\s+)?(?:provided|performed|maintained|delivered|submitted|included)", "shall_passive", 0.9, "MANDATORY"),
        (r"\bmust\s+(?:be\s+)?(?:provided|performed|maintained|delivered|submitted|included)", "must_passive", 0.9, "MANDATORY"),
        (r"\bis\s+(?:required|mandatory)\s+(?:to|that)", "is_required", 0.9, "MANDATORY"),
        (r"\bare\s+(?:required|mandatory)\s+(?:to|that)", "are_required", 0.9, "MANDATORY"),

        # Medium-strong indicators (weight 0.8) - shall/must alone
        (r"\bshall\s+(?:provide|perform|submit|include|deliver|maintain|ensure|develop)", "shall_verb", 0.8, "MANDATORY"),
        (r"\bmust\s+(?:provide|perform|submit|include|deliver|maintain|ensure|develop)", "must_verb", 0.8, "MANDATORY"),
        (r"\bwill\s+be\s+required\s+to\b", "will_be_required", 0.8, "MANDATORY"),

        # Standard mandatory (weight 0.7)
        (r"\bshall\b", "shall", 0.7, "MANDATORY"),
        (r"\bmust\b", "must", 0.7, "MANDATORY"),
        (r"\brequired\s+to\b", "required_to", 0.7, "MANDATORY"),
        (r"\bmandatory\b", "mandatory", 0.7, "MANDATORY"),
        (r"\bresponsible\s+for\b", "responsible_for", 0.7, "MANDATORY"),

        # Prohibition patterns (weight 0.9)
        (r"\bshall\s+not\b", "shall_not", 0.9, "PROHIBITION"),
        (r"\bmust\s+not\b", "must_not", 0.9, "PROHIBITION"),
        (r"\bwill\s+not\s+(?:be\s+)?(?:allowed|permitted|accepted)", "will_not", 0.9, "PROHIBITION"),
        (r"\bprohibited\b", "prohibited", 0.9, "PROHIBITION"),
        (r"\bforbidden\b", "forbidden", 0.9, "PROHIBITION"),
        (r"\bnot\s+(?:permitted|allowed|acceptable)\b", "not_permitted", 0.9, "PROHIBITION"),
        (r"\bunder\s+no\s+circumstances\b", "no_circumstances", 0.9, "PROHIBITION"),
    ]

    CONTEXTUAL_CONDITIONAL_PATTERNS = [
        # Highly desirable (weight 0.6)
        (r"\bshould\s+(?:provide|perform|submit|include|deliver|describe)", "should_verb", 0.6, "HIGHLY_DESIRABLE"),
        (r"\bofferor\s+should\b", "offeror_should", 0.6, "HIGHLY_DESIRABLE"),
        (r"\bcontractor\s+should\b", "contractor_should", 0.6, "HIGHLY_DESIRABLE"),
        (r"\bstrongly\s+(?:recommended|encouraged|suggested)", "strongly_recommended", 0.6, "HIGHLY_DESIRABLE"),
        (r"\bhighly\s+(?:recommended|desirable|preferred)", "highly_recommended", 0.6, "HIGHLY_DESIRABLE"),

        # Desirable (weight 0.5)
        (r"\bshould\b", "should", 0.5, "DESIRABLE"),
        (r"\brecommended\b", "recommended", 0.5, "DESIRABLE"),
        (r"\bencouraged\b", "encouraged", 0.5, "DESIRABLE"),
        (r"\bpreferred\b", "preferred", 0.5, "DESIRABLE"),
        (r"\bdesirable\b", "desirable", 0.5, "DESIRABLE"),

        # Optional (weight 0.4)
        (r"\bmay\s+(?:provide|submit|include|choose|elect)", "may_verb", 0.4, "OPTIONAL"),
        (r"\bcan\s+(?:provide|submit|include|choose|opt)", "can_verb", 0.4, "OPTIONAL"),
        (r"\bmay\b", "may", 0.4, "OPTIONAL"),
        (r"\boptional\b", "optional", 0.4, "OPTIONAL"),
        (r"\bat\s+(?:the\s+)?(?:offeror'?s?|contractor'?s?)\s+discretion", "discretion", 0.4, "OPTIONAL"),
    ]

    # Section-specific pattern adjustments
    # Some words have different meanings in different sections
    SECTION_PATTERN_ADJUSTMENTS = {
        "L": {
            # In Section L, these are proposal instructions, not contract requirements
            "boost_patterns": [
                r"\bproposal\s+(?:shall|must|should)",
                r"\bvolume\s+(?:shall|must|should)",
                r"\bofferor\s+(?:shall|must|should)",
                r"\bpage\s+limit",
                r"\bformat\s+(?:shall|must|should)",
            ],
            "reduce_patterns": [
                # "submit" in Section L is about the proposal, not a deliverable
                r"\bsubmit\s+(?:to|by|before)",
            ],
        },
        "M": {
            # In Section M, these indicate evaluation criteria
            "boost_patterns": [
                r"\b(?:will|shall)\s+be\s+evaluated",
                r"\bevaluation\s+(?:factor|criteria)",
                r"\bscoring\b",
                r"\b(?:strengths?|weaknesses?|deficienc)",
                r"\b(?:more|less|equally)\s+important",
            ],
            "reduce_patterns": [],
        },
        "C": {
            # In Section C, focus on contractor performance requirements
            "boost_patterns": [
                r"\bcontractor\s+(?:shall|must|will)",
                r"\bthe\s+work\s+(?:shall|will)",
                r"\bservices?\s+(?:shall|will)",
                r"\bdeliverable",
            ],
            "reduce_patterns": [],
        },
    }

    # Hierarchical requirement categories
    # Maps category names to related keywords for grouping
    REQUIREMENT_CATEGORIES = {
        "DOCUMENTATION": {
            "keywords": ["manual", "report", "documentation", "guide", "procedure", "plan", "document", "record"],
            "patterns": [r"\b(?:monthly|weekly|quarterly|annual|final)\s+report", r"\bdocumentation\s+(?:shall|must)"],
        },
        "PERSONNEL": {
            "keywords": ["personnel", "staff", "team", "employee", "worker", "resource", "FTE"],
            "patterns": [r"\bkey\s+personnel", r"\blabor\s+(?:category|hour)", r"\bstaffing\s+(?:plan|level)"],
        },
        "SECURITY": {
            "keywords": ["security", "clearance", "classified", "cybersecurity", "FISMA", "FedRAMP"],
            "patterns": [r"\b(?:secret|top\s+secret|ts/sci)\s+clearance", r"\bsecurity\s+(?:requirement|control)"],
        },
        "QUALITY": {
            "keywords": ["quality", "QA", "QC", "assurance", "control", "inspection", "testing"],
            "patterns": [r"\bquality\s+(?:assurance|control)", r"\binspection\s+(?:and\s+)?acceptance"],
        },
        "SCHEDULE": {
            "keywords": ["schedule", "timeline", "milestone", "deadline", "delivery", "date", "period"],
            "patterns": [r"\bperiod\s+of\s+performance", r"\bdelivery\s+(?:date|schedule)", r"\bno\s+later\s+than"],
        },
        "COST": {
            "keywords": ["cost", "price", "budget", "funding", "invoice", "payment", "rate"],
            "patterns": [r"\bcost\s+(?:proposal|estimate)", r"\bpricing\s+(?:structure|schedule)"],
        },
        "COMPLIANCE": {
            "keywords": ["FAR", "DFARS", "compliance", "regulation", "clause", "provision", "statute"],
            "patterns": [r"\bFAR\s+\d+\.\d+", r"\bDFARS\s+\d+\.\d+", r"\bin\s+accordance\s+with"],
        },
        "TECHNICAL": {
            "keywords": ["technical", "system", "software", "hardware", "technology", "solution", "architecture"],
            "patterns": [r"\btechnical\s+(?:approach|solution|requirement)", r"\bsystem\s+(?:design|architecture)"],
        },
    }

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

    # Legacy patterns for backward compatibility
    # (Kept for code that may reference these directly)
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

    CONDITIONAL_PATTERNS = [
        (r"\bshould\b", "should"),
        (r"\bmay\b", "may"),
        (r"\bcan\b", "can"),
        (r"\bis\s+encouraged\b", "encouraged"),
        (r"\bis\s+recommended\b", "recommended"),
        (r"\boptional\b", "optional"),
    ]

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
        
        # Split into sentences for processing
        sentences = self._split_into_sentences(doc.full_text)
        
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
    
    def _classify_sentence(self, sentence: str, section_id: str = "") -> Tuple[Optional[RequirementType], Optional[str]]:
        """
        Classify a sentence and determine if it's a requirement.

        Enhanced with contextual patterns and section-aware adjustments.

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

        # Check using contextual patterns with confidence weights
        best_match = self._classify_with_contextual_patterns(sentence_lower, section_id)
        if best_match:
            return best_match

        # Fall back to legacy patterns for backward compatibility
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

    def _classify_with_contextual_patterns(
        self,
        sentence_lower: str,
        section_id: str = ""
    ) -> Optional[Tuple[RequirementType, str]]:
        """
        Classify using contextual patterns with confidence weights.

        Returns the highest-confidence match, or None if no match.
        """
        best_weight = 0.0
        best_keyword = None
        best_binding = None

        # Check mandatory contextual patterns
        for pattern, keyword, weight, binding in self.CONTEXTUAL_MANDATORY_PATTERNS:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                # Apply section-specific adjustments
                adjusted_weight = self._apply_section_adjustment(weight, sentence_lower, section_id)
                if adjusted_weight > best_weight:
                    best_weight = adjusted_weight
                    best_keyword = keyword
                    best_binding = binding

        # Check conditional contextual patterns (only if no mandatory found)
        if best_weight < 0.7:  # Only consider conditional if no strong mandatory
            for pattern, keyword, weight, binding in self.CONTEXTUAL_CONDITIONAL_PATTERNS:
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    adjusted_weight = self._apply_section_adjustment(weight, sentence_lower, section_id)
                    if adjusted_weight > best_weight:
                        best_weight = adjusted_weight
                        best_keyword = keyword
                        best_binding = binding

        if best_keyword and best_weight >= 0.4:  # Minimum threshold
            # Map binding level to requirement type
            req_type = self._binding_to_requirement_type(best_binding, sentence_lower)
            return req_type, best_keyword

        return None

    def _apply_section_adjustment(
        self,
        base_weight: float,
        sentence_lower: str,
        section_id: str
    ) -> float:
        """
        Apply section-specific weight adjustments.

        Some patterns have different significance in different sections.
        """
        if not section_id:
            return base_weight

        section_upper = section_id.upper().replace("SECTION_", "")

        adjustments = self.SECTION_PATTERN_ADJUSTMENTS.get(section_upper, {})

        # Check boost patterns
        for pattern in adjustments.get("boost_patterns", []):
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return min(base_weight + 0.15, 1.0)  # Boost but cap at 1.0

        # Check reduce patterns
        for pattern in adjustments.get("reduce_patterns", []):
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return max(base_weight - 0.2, 0.0)  # Reduce but floor at 0.0

        return base_weight

    def _binding_to_requirement_type(self, binding: str, sentence_lower: str) -> RequirementType:
        """
        Map binding level to requirement type, considering sentence content.
        """
        # Special handling for evaluation language
        if binding == "EVALUATION":
            return RequirementType.EVALUATION_CRITERION

        # Special handling for prohibitions
        if binding == "PROHIBITION":
            return RequirementType.PROHIBITION

        # Check for specific content indicators
        if re.search(r'\bproposal\b.*\b(?:shall|must|should)', sentence_lower):
            return RequirementType.PROPOSAL_INSTRUCTION

        if re.search(r'\b(?:deliverable|report|document)\b', sentence_lower):
            return RequirementType.DELIVERABLE

        if re.search(r'\b(?:qualification|experience|certification|clearance)\b', sentence_lower):
            return RequirementType.QUALIFICATION

        if re.search(r'\b(?:FAR|DFARS|comply|compliance)\b', sentence_lower):
            return RequirementType.COMPLIANCE

        # Default based on binding level
        if binding in ["MANDATORY", "HIGHLY_DESIRABLE", "DESIRABLE", "OPTIONAL"]:
            return RequirementType.PERFORMANCE

        return RequirementType.PERFORMANCE

    def get_requirement_category(self, sentence: str) -> Optional[str]:
        """
        Determine the category of a requirement for grouping.

        Returns the category name or None if no category matches.
        """
        sentence_lower = sentence.lower()

        for category, config in self.REQUIREMENT_CATEGORIES.items():
            # Check keywords
            for keyword in config.get("keywords", []):
                if keyword.lower() in sentence_lower:
                    return category

            # Check patterns
            for pattern in config.get("patterns", []):
                if re.search(pattern, sentence_lower, re.IGNORECASE):
                    return category

        return None

    def get_binding_level(self, sentence: str) -> str:
        """
        Determine the binding level of a requirement.

        Returns: MANDATORY, HIGHLY_DESIRABLE, DESIRABLE, OPTIONAL, or INFORMATIONAL
        """
        sentence_lower = sentence.lower()

        # Check mandatory patterns first (highest priority)
        for pattern, keyword, weight, binding in self.CONTEXTUAL_MANDATORY_PATTERNS:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return binding

        # Check conditional patterns
        for pattern, keyword, weight, binding in self.CONTEXTUAL_CONDITIONAL_PATTERNS:
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return binding

        return "INFORMATIONAL"
    
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
        
        return RequirementNode(
            id=req_id,
            text=sentence.strip(),
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
    
    # Additional patterns for section reference extraction
    SECTION_REF_EXTENDED_PATTERNS = [
        r'([A-M])\.(\d+)(?:\.(\d+|[a-z]))?(?:\.(\d+|[a-z]))?',  # L.4.B.2
        r'(?:PWS|SOW)\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?',          # PWS 2.1.3
        r'(?:Section|Article|Paragraph)\s+([A-M])(?:\.(\d+))?',  # Section L.4
        r'(?:SECTION\s+)?([A-M])\s*[-–—]\s*',                     # SECTION L -
    ]

    def _extract_section_ref(self, sentence: str, doc: ParsedDocument) -> str:
        """
        Extract section reference from sentence and context.

        Uses multiple strategies:
        1. Look for explicit section references in the sentence
        2. Check surrounding context for section markers
        3. Infer from document section if available
        """
        # Strategy 1: Look for explicit section references in sentence
        match = re.search(self.SECTION_REF_PATTERN, sentence)
        if match:
            parts = [p for p in match.groups() if p]
            return ".".join(parts)

        # Strategy 2: Try extended patterns
        for pattern in self.SECTION_REF_EXTENDED_PATTERNS:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                parts = [p for p in match.groups() if p]
                if parts:
                    return ".".join(parts)

        # Strategy 3: Check document sections for position-based assignment
        if doc and doc.sections:
            # Find which section this sentence belongs to
            sentence_pos = doc.full_text.find(sentence[:50])  # Use first 50 chars
            if sentence_pos >= 0:
                for section_id, section_text in doc.sections.items():
                    section_start = doc.full_text.find(section_text)
                    if section_start >= 0 and section_start <= sentence_pos < section_start + len(section_text):
                        return section_id.replace("section_", "").upper()

        # Strategy 4: Infer from requirement content
        sentence_lower = sentence.lower()
        content_indicators = [
            (r'\b(?:offeror|proposer)s?\s+(?:shall|must|should)', 'L'),
            (r'\bproposal\s+(?:shall|must|should)', 'L'),
            (r'\b(?:government|agency)\s+(?:will|shall)\s+(?:evaluate|assess)', 'M'),
            (r'\bevaluation\s+(?:factor|criteria)', 'M'),
            (r'\bcontractor\s+(?:shall|must|will)\s+(?:provide|perform|deliver)', 'C'),
            (r'\bthe\s+work\s+(?:shall|will)', 'C'),
        ]
        for pattern, section in content_indicators:
            if re.search(pattern, sentence_lower):
                return section

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
