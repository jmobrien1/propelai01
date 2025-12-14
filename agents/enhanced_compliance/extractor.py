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
    MAX_SENTENCE_LENGTH = 500         # Maximum chars (reduced from 1000 to avoid multi-shall bundles)
    MIN_WORDS = 15                    # Minimum words in a requirement (increased)
    REQUIRE_ACTOR = True              # Require "contractor/offeror/government" for high confidence

    # Obligation words that must be present for a valid requirement
    OBLIGATION_WORDS = ['shall', 'must', 'will', 'required', 'should', 'may', 'can']
    
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
        # Added from audit recommendations
        r"^See\s+(?:Section|Attachment|Exhibit)",         # Cross-reference only
        r"^Note:",                                         # Notes as headers
        r"^(?:Refer|Reference)\s+to",                     # Reference-only lines
        r"^End\s+of\s+(?:Section|Document|Clause)",       # Section terminators
        r"^[A-Z][A-Z\s]{0,30}:$",                         # All-caps labels ending with colon
        r"^\[.*\]$",                                       # Bracketed placeholders
        r"^TBD\s*$",                                       # TBD placeholders
        r"^N/A\s*$",                                       # N/A placeholders
        r"^Reserved\s*\.?$",                               # Reserved sections
        r"^This\s+(?:section|page)\s+intentionally",      # Intentionally blank
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

    # ============================================================================
    # SECTION-SPECIFIC EXTRACTION RULES v1.0
    # Per accuracy.txt: Different extraction behaviors per section
    # ============================================================================

    SECTION_EXTRACTION_CONFIG = {
        # Section B: Supplies/Services and Prices
        "B": {
            "description": "Supplies/Services and Prices/Costs",
            "primary_types": [RequirementType.LABOR_REQUIREMENT, RequirementType.DELIVERABLE],
            "secondary_types": [RequirementType.PERFORMANCE],
            "default_type": RequirementType.LABOR_REQUIREMENT,
            "confidence_boost": 0.1,  # Known section adds confidence
            "min_sentence_length": 80,  # Allow shorter CLIN descriptions
            "expected_actors": ["contractor", "vendor", "offeror"],
            "key_patterns": [
                r"\bCLIN\s*\d+",
                r"\bline\s+item\s*\d*",
                r"\bunit\s+price\b",
                r"\b(?:FFP|T&M|CPFF|CPIF|CPAF)\b",  # Contract types
                r"\blabor\s+(?:hour|category|rate)",
                r"\b(?:option\s+(?:year|period)|base\s+(?:year|period))\b",
            ],
            "entity_focus": ["clin", "price", "labor_category", "contract_type"],
            "noise_patterns": [
                r"^CLIN\s*$",  # Bare CLIN without details
                r"^\$[\d,]+\s*$",  # Just a price
            ],
        },

        # Section C: Statement of Work / Description
        "C": {
            "description": "Description/Specifications/Statement of Work",
            "primary_types": [RequirementType.PERFORMANCE, RequirementType.PERFORMANCE_METRIC],
            "secondary_types": [RequirementType.DELIVERABLE, RequirementType.QUALIFICATION],
            "default_type": RequirementType.PERFORMANCE,
            "confidence_boost": 0.15,  # SOW is high-value for requirements
            "min_sentence_length": 100,
            "expected_actors": ["contractor", "vendor", "government"],
            "key_patterns": [
                r"\bcontractor\s+(?:shall|must|will)\s+(?:provide|perform|maintain|ensure|develop)",
                r"\bthe\s+work\s+(?:shall|will)\s+(?:include|consist|encompass)",
                r"\bservices?\s+(?:shall|will)\s+(?:include|be\s+provided)",
                r"\b(?:task|subtask)\s+\d+",
                r"\bperformance\s+(?:standard|requirement|objective)",
                r"\bquality\s+(?:assurance|control|level)",
            ],
            "entity_focus": ["task", "deliverable", "metric", "standard"],
            "noise_patterns": [
                r"^\d+\.\d+\s+[A-Z][A-Z\s]+$",  # Section headers like "3.1 SCOPE"
            ],
        },

        # Section D: Packaging and Marking
        "D": {
            "description": "Packaging and Marking",
            "primary_types": [RequirementType.COMPLIANCE, RequirementType.PERFORMANCE],
            "secondary_types": [RequirementType.DELIVERABLE],
            "default_type": RequirementType.COMPLIANCE,
            "confidence_boost": 0.05,
            "min_sentence_length": 80,
            "expected_actors": ["contractor", "vendor"],
            "key_patterns": [
                r"\bpackag(?:e|ing)\s+(?:shall|must|will)",
                r"\bmark(?:ed|ing)\s+(?:shall|must|will|with)",
                r"\blabel(?:ed|ing)\s+(?:shall|must|will)",
                r"\bMIL[-\s]?STD",
            ],
            "entity_focus": ["standard", "specification"],
            "noise_patterns": [],
        },

        # Section E: Inspection and Acceptance
        "E": {
            "description": "Inspection and Acceptance",
            "primary_types": [RequirementType.PERFORMANCE_METRIC, RequirementType.COMPLIANCE],
            "secondary_types": [RequirementType.PERFORMANCE],
            "default_type": RequirementType.PERFORMANCE_METRIC,
            "confidence_boost": 0.1,
            "min_sentence_length": 80,
            "expected_actors": ["contractor", "government", "COR", "contracting officer"],
            "key_patterns": [
                r"\binspection\s+(?:shall|will)\s+be",
                r"\bacceptance\s+(?:criteria|shall|will)",
                r"\bquality\s+(?:assurance|control)\s+(?:shall|must)",
                r"\b(?:reject|rejection|defect|deficiency)",
                r"\b(?:AQL|acceptable\s+quality\s+level)",
            ],
            "entity_focus": ["quality_metric", "inspection_point", "acceptance_criteria"],
            "noise_patterns": [],
        },

        # Section F: Deliveries or Performance
        "F": {
            "description": "Deliveries or Performance",
            "primary_types": [RequirementType.DELIVERABLE, RequirementType.PERFORMANCE],
            "secondary_types": [RequirementType.PERFORMANCE_METRIC],
            "default_type": RequirementType.DELIVERABLE,
            "confidence_boost": 0.1,
            "min_sentence_length": 80,
            "expected_actors": ["contractor", "vendor"],
            "key_patterns": [
                r"\b(?:deliver|delivery)\s+(?:shall|will|date|schedule)",
                r"\b(?:due|deadline)\s+(?:date|within|by)",
                r"\bperiod\s+of\s+performance",
                r"\b(?:POP|PoP)\b",
                r"\b(?:option\s+period|base\s+period)",
                r"\b(?:no\s+later\s+than|NLT)\b",
            ],
            "entity_focus": ["date", "schedule", "milestone", "deliverable"],
            "noise_patterns": [
                r"^F\.O\.B\.\s*",  # FOB headers
            ],
        },

        # Section G: Contract Administration Data
        "G": {
            "description": "Contract Administration Data",
            "primary_types": [RequirementType.COMPLIANCE, RequirementType.PERFORMANCE],
            "secondary_types": [],
            "default_type": RequirementType.COMPLIANCE,
            "confidence_boost": 0.05,
            "min_sentence_length": 80,
            "expected_actors": ["contractor", "COR", "contracting officer", "CO", "ACO"],
            "key_patterns": [
                r"\b(?:invoice|payment)\s+(?:shall|must|will)",
                r"\b(?:COR|contracting\s+officer)\s+(?:shall|will|is)",
                r"\b(?:submit|submission)\s+(?:to|through)",
                r"\belectronic\s+(?:invoicing|payment)",
            ],
            "entity_focus": ["contact", "invoice_instruction", "submission_point"],
            "noise_patterns": [],
        },

        # Section H: Special Contract Requirements
        "H": {
            "description": "Special Contract Requirements",
            "primary_types": [RequirementType.COMPLIANCE, RequirementType.QUALIFICATION],
            "secondary_types": [RequirementType.PERFORMANCE],
            "default_type": RequirementType.COMPLIANCE,
            "confidence_boost": 0.1,
            "min_sentence_length": 100,
            "expected_actors": ["contractor", "vendor", "offeror"],
            "key_patterns": [
                r"\b(?:security|clearance)\s+(?:shall|must|requirement)",
                r"\b(?:insurance|bonding)\s+(?:shall|must|requirement)",
                r"\b(?:key\s+personnel|staffing)\s+(?:shall|must)",
                r"\b(?:OCONUS|overseas|travel)\b",
                r"\b(?:organizational\s+conflict\s+of\s+interest|OCI)\b",
                r"\b(?:subcontract|teaming)\s+(?:shall|must)",
            ],
            "entity_focus": ["clearance", "certification", "special_requirement"],
            "noise_patterns": [],
        },

        # Section I: Contract Clauses
        "I": {
            "description": "Contract Clauses",
            "primary_types": [RequirementType.COMPLIANCE],
            "secondary_types": [],
            "default_type": RequirementType.COMPLIANCE,
            "confidence_boost": -0.05,  # Mostly boilerplate FAR clauses
            "min_sentence_length": 120,  # Require longer text for clauses
            "expected_actors": ["contractor", "government"],
            "key_patterns": [
                r"\bFAR\s+\d+\.\d+",
                r"\bDFARS\s+\d+\.\d+",
                r"\bHHSAR\s+\d+\.\d+",
                r"\b52\.\d{3}-\d+",
            ],
            "entity_focus": ["far_clause", "regulation"],
            "noise_patterns": [
                r"^52\.\d{3}-\d+\s+[A-Z][A-Za-z\s]+$",  # Just clause number and title
                r"^\(.*\)$",  # Just a date in parentheses
            ],
        },

        # Section J: Attachments/Exhibits
        "J": {
            "description": "Attachments/Exhibits",
            "primary_types": [RequirementType.DELIVERABLE, RequirementType.PERFORMANCE],
            "secondary_types": [RequirementType.FORMAT],
            "default_type": RequirementType.DELIVERABLE,
            "confidence_boost": 0.05,
            "min_sentence_length": 80,
            "expected_actors": ["contractor", "offeror"],
            "key_patterns": [
                r"\b(?:attachment|exhibit)\s+\d+",
                r"\b(?:appendix|annex)\s+[A-Z\d]",
                r"\b(?:use|complete|fill)\s+(?:this|the)\s+(?:form|template)",
            ],
            "entity_focus": ["attachment", "form", "template"],
            "noise_patterns": [
                r"^(?:Attachment|Exhibit|Appendix)\s+[A-Z\d]+\s*$",  # Just attachment label
            ],
        },

        # Section K: Representations and Certifications
        "K": {
            "description": "Representations and Certifications",
            "primary_types": [RequirementType.QUALIFICATION, RequirementType.COMPLIANCE],
            "secondary_types": [],
            "default_type": RequirementType.QUALIFICATION,
            "confidence_boost": 0.05,
            "min_sentence_length": 100,
            "expected_actors": ["offeror", "contractor"],
            "key_patterns": [
                r"\b(?:represent|certif(?:y|ies|ication))\s+(?:that|the)",
                r"\b(?:small\s+business|8\(a\)|HUBZone|SDVOSB|WOSB)\b",
                r"\b(?:ownership|control|affiliation)\b",
                r"\b(?:debarred|suspended|ineligible)\b",
            ],
            "entity_focus": ["certification", "representation", "business_status"],
            "noise_patterns": [
                r"^\[\s*\]\s*(?:Yes|No)",  # Checkbox items alone
            ],
        },

        # Section L: Instructions, Conditions, Notices
        "L": {
            "description": "Instructions, Conditions, and Notices to Offerors",
            "primary_types": [RequirementType.PROPOSAL_INSTRUCTION, RequirementType.FORMAT],
            "secondary_types": [RequirementType.QUALIFICATION],
            "default_type": RequirementType.PROPOSAL_INSTRUCTION,
            "confidence_boost": 0.15,  # Section L is critical for proposals
            "min_sentence_length": 80,
            "expected_actors": ["offeror", "proposer", "vendor"],
            "key_patterns": [
                r"\bofferor\s+(?:shall|must|should)\s+(?:submit|provide|include|describe|demonstrate|address)",
                r"\bproposal\s+(?:shall|must|should)\s+(?:include|contain|address|describe|demonstrate)",
                r"\b(?:page\s+limit|font\s+size|margin|format)\b",
                r"\bvolume\s+\d*\s*(?:shall|must|should)",
                r"\b(?:technical|management|past\s+performance|cost|price)\s+(?:proposal|volume)",
                r"\b(?:oral\s+presentation|demonstration|site\s+visit)\b",
            ],
            "entity_focus": ["page_limit", "format_requirement", "submission_deadline", "volume"],
            "noise_patterns": [
                r"^L\.\d+\s+[A-Z][A-Z\s]+$",  # Just section numbers
            ],
        },

        # Section M: Evaluation Factors
        "M": {
            "description": "Evaluation Factors for Award",
            "primary_types": [RequirementType.EVALUATION_CRITERION],
            "secondary_types": [RequirementType.PERFORMANCE_METRIC],
            "default_type": RequirementType.EVALUATION_CRITERION,
            "confidence_boost": 0.15,  # Section M is critical for understanding award
            "min_sentence_length": 80,
            "expected_actors": ["government", "agency", "offeror"],
            "key_patterns": [
                r"\b(?:will|shall)\s+be\s+evaluated",
                r"\bevaluation\s+(?:factor|criteria|subfactor)",
                r"\b(?:technical|management|past\s+performance|cost|price)\s+(?:factor|volume|proposal)",
                r"\b(?:adjectival|color|rating|score)\s+(?:rating|scale|criteria)",
                r"\b(?:strength|weakness|deficiency|significant|acceptable|unacceptable)\b",
                r"\b(?:more|less|equally)\s+important\s+than\b",
                r"\b(?:tradeoff|best\s+value|LPTA|lowest\s+price)\b",
            ],
            "entity_focus": ["evaluation_factor", "weight", "rating", "award_methodology"],
            "noise_patterns": [],
        },

        # PWS: Performance Work Statement
        "PWS": {
            "description": "Performance Work Statement",
            "primary_types": [RequirementType.PERFORMANCE, RequirementType.PERFORMANCE_METRIC],
            "secondary_types": [RequirementType.DELIVERABLE, RequirementType.QUALIFICATION],
            "default_type": RequirementType.PERFORMANCE,
            "confidence_boost": 0.15,
            "min_sentence_length": 100,
            "expected_actors": ["contractor", "vendor"],
            "key_patterns": [
                r"\bcontractor\s+(?:shall|must|will)",
                r"\bperformance\s+(?:standard|requirement|objective|metric)",
                r"\b(?:task|subtask|work\s+area)\s+\d+",
                r"\bdeliverable\s+(?:shall|must|include)",
            ],
            "entity_focus": ["task", "deliverable", "metric", "standard"],
            "noise_patterns": [],
        },

        # SOW: Statement of Work
        "SOW": {
            "description": "Statement of Work",
            "primary_types": [RequirementType.PERFORMANCE, RequirementType.DELIVERABLE],
            "secondary_types": [RequirementType.PERFORMANCE_METRIC, RequirementType.QUALIFICATION],
            "default_type": RequirementType.PERFORMANCE,
            "confidence_boost": 0.15,
            "min_sentence_length": 100,
            "expected_actors": ["contractor", "vendor"],
            "key_patterns": [
                r"\bcontractor\s+(?:shall|must|will)",
                r"\bthe\s+work\s+(?:shall|will)",
                r"\b(?:task|subtask)\s+\d+",
            ],
            "entity_focus": ["task", "deliverable", "requirement"],
            "noise_patterns": [],
        },

        # Default/Unknown sections
        "UNKNOWN": {
            "description": "Unknown or Unspecified Section",
            "primary_types": [RequirementType.PERFORMANCE],
            "secondary_types": [],
            "default_type": RequirementType.PERFORMANCE,
            "confidence_boost": -0.1,  # Unknown section reduces confidence
            "min_sentence_length": 100,
            "expected_actors": ["contractor", "offeror", "government"],
            "key_patterns": [],
            "entity_focus": [],
            "noise_patterns": [],
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

                # Skip sentences without obligation words (reduces false positives by ~25%)
                has_obligation, obligation_word = self._has_obligation_word(sentence)
                if not has_obligation:
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
    
    # ============================================================================
    # CONFIDENCE THRESHOLDS FOR REVIEW FLAGGING
    # Per accuracy.txt: Flag borderline items for human review
    # ============================================================================

    # Confidence score thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8      # Above this = high confidence, no review
    REVIEW_THRESHOLD = 0.6               # Below this = needs review
    LOW_CONFIDENCE_THRESHOLD = 0.4       # Below this = low confidence

    def _create_requirement_node(
        self,
        sentence: str,
        sentence_index: int,
        sentences: List[str],
        doc: ParsedDocument,
        req_type: RequirementType,
        keyword_match: str,
    ) -> RequirementNode:
        """Create a RequirementNode from a sentence with confidence scoring."""
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

        # Calculate confidence score (numeric)
        confidence_score = self._calculate_confidence_score(sentence, keyword_match, section_id, doc)

        # Determine confidence level from score
        confidence = self._score_to_confidence_level(confidence_score)

        # Determine if review is needed and why
        needs_review, review_reason = self._determine_review_needed(
            sentence, keyword_match, confidence_score, section_id, req_type
        )

        # Get binding level and category
        binding_level = self.get_binding_level(sentence)
        category = self.get_requirement_category(sentence) or ""

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
            confidence_score=confidence_score,
            needs_review=needs_review,
            review_reason=review_reason,
            binding_level=binding_level,
            category=category,
            source=source,
            context_before=context_before,
            context_after=context_after,
            keywords=keywords,
            entities=entities,
            references_to=references,
            extraction_method="regex",
        )

    def _calculate_confidence_score(
        self,
        sentence: str,
        keyword_match: str,
        section_id: str,
        doc: ParsedDocument
    ) -> float:
        """
        Calculate a numeric confidence score (0.0-1.0) for the requirement.

        Factors considered:
        - Keyword strength (shall > must > should > may)
        - Actor presence (contractor, offeror, government)
        - Section context (appropriate section for requirement type)
        - Section-specific pattern matching
        - Document type (main solicitation vs attachment)
        - Pattern match quality
        """
        score = 0.5  # Base score

        sentence_lower = sentence.lower()

        # Factor 1: Keyword strength
        keyword_scores = {
            "contractor_shall": 0.35,
            "contractor_must": 0.35,
            "offeror_shall": 0.35,
            "offeror_must": 0.35,
            "government_will": 0.35,
            "shall_passive": 0.30,
            "must_passive": 0.30,
            "shall_verb": 0.25,
            "must_verb": 0.25,
            "shall": 0.20,
            "must": 0.20,
            "required": 0.20,
            "mandatory": 0.20,
            "responsible": 0.15,
            "should_verb": 0.10,
            "should": 0.08,
            "recommended": 0.05,
            "encouraged": 0.05,
            "may_verb": 0.0,
            "may": -0.05,
            "optional": -0.05,
        }
        score += keyword_scores.get(keyword_match, 0.0)

        # Factor 2: Actor presence (use section-aware actor check)
        section_config = self._get_section_config(section_id) if section_id else None
        if section_config:
            if self._has_section_actor(sentence, section_config):
                score += 0.15
        elif self._has_actor(sentence):
            score += 0.15

        # Factor 3: Section appropriateness with section-specific boost
        section_upper = section_id.upper() if section_id else ""
        if section_config:
            # Apply section-specific confidence boost from config
            section_boost = section_config.get("confidence_boost", 0.0)
            score += section_boost
        elif section_upper in ["L", "M", "C", "PWS", "SOW"]:
            score += 0.10  # Known section adds confidence
        elif section_upper in ["UNSPEC", "UNK", "UNKNOWN", ""]:
            score -= 0.10  # Unknown section reduces confidence

        # Factor 4: Section-specific pattern matching
        if section_config and self._matches_section_patterns(sentence, section_config):
            score += 0.10  # Bonus for matching section-specific patterns

        # Factor 5: Document type
        if doc.document_type in [DocumentType.MAIN_SOLICITATION, DocumentType.STATEMENT_OF_WORK]:
            score += 0.05
        elif doc.document_type == DocumentType.AMENDMENT:
            score += 0.0  # Neutral
        elif doc.document_type == DocumentType.ATTACHMENT:
            score -= 0.05

        # Factor 6: Content quality indicators
        # Longer, more detailed requirements are often more confident
        word_count = len(sentence.split())
        if word_count >= 20:
            score += 0.05
        elif word_count < 10:
            score -= 0.05

        # Presence of specific action verbs adds confidence
        if re.search(r'\b(?:provide|perform|deliver|submit|maintain|ensure|develop|implement)\b', sentence_lower):
            score += 0.05

        # Normalize to 0.0-1.0 range
        return max(0.0, min(1.0, score))

    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to ConfidenceLevel enum."""
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif score >= self.REVIEW_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _determine_review_needed(
        self,
        sentence: str,
        keyword_match: str,
        confidence_score: float,
        section_id: str,
        req_type: RequirementType
    ) -> Tuple[bool, str]:
        """
        Determine if a requirement needs human review.

        Returns:
            (needs_review: bool, reason: str)
        """
        reasons = []

        # Reason 1: Low confidence score
        if confidence_score < self.REVIEW_THRESHOLD:
            reasons.append(f"Low confidence score ({confidence_score:.2f})")

        # Reason 2: Conditional/optional keywords without clear context
        if keyword_match in ["should", "may", "can", "recommended", "encouraged", "optional"]:
            if not self._has_actor(sentence):
                reasons.append("Optional/conditional keyword without clear actor")

        # Reason 3: Unknown or unspecified section
        section_upper = section_id.upper() if section_id else ""
        if section_upper in ["UNSPEC", "UNK", "UNKNOWN", ""]:
            reasons.append("Section could not be determined")

        # Reason 4: Potential ambiguity in requirement type
        sentence_lower = sentence.lower()
        # Check for mixed signals (e.g., both "shall" and "may" in same sentence)
        has_mandatory = bool(re.search(r'\b(?:shall|must)\b', sentence_lower))
        has_optional = bool(re.search(r'\b(?:may|should|can)\b', sentence_lower))
        if has_mandatory and has_optional:
            reasons.append("Mixed mandatory/optional language")

        # Reason 5: Very short requirement text
        if len(sentence.split()) < 8:
            reasons.append("Very short requirement text")

        # Reason 6: Complex sentence structure (multiple clauses)
        clause_indicators = len(re.findall(r'\b(?:and|or|but|however|unless|except|provided that)\b', sentence_lower))
        if clause_indicators >= 3:
            reasons.append("Complex sentence with multiple clauses")

        # Reason 7: Reference to external document without context
        if re.search(r'\b(?:see|refer to|per|as specified in)\b', sentence_lower):
            if len(sentence) < 150:  # Short sentence that's mostly a reference
                reasons.append("Reference to external document - may need verification")

        # Determine if review is needed
        needs_review = len(reasons) > 0

        # Create review reason string
        review_reason = "; ".join(reasons) if reasons else ""

        return needs_review, review_reason
    
    def _get_section_config(self, section_id: str) -> Dict[str, Any]:
        """
        Get the section-specific extraction configuration.

        Args:
            section_id: Section identifier (e.g., "L", "M", "C", "PWS")

        Returns:
            Configuration dictionary for the section
        """
        # Normalize section ID
        section_key = section_id.replace("section_", "").upper().strip()

        # Handle special cases
        if section_key in ["UNSPEC", "UNK", ""]:
            section_key = "UNKNOWN"

        # Look up first letter for UCF sections like "L.4.B"
        if len(section_key) > 1 and section_key[0].isalpha() and section_key[1] == ".":
            section_key = section_key[0]

        return self.SECTION_EXTRACTION_CONFIG.get(
            section_key,
            self.SECTION_EXTRACTION_CONFIG["UNKNOWN"]
        )

    def _is_section_noise(self, sentence: str, section_config: Dict[str, Any]) -> bool:
        """
        Check if sentence is noise according to section-specific rules.

        Args:
            sentence: The text to check
            section_config: Section-specific configuration

        Returns:
            True if sentence should be filtered out
        """
        # First apply general noise check
        if self._is_noise(sentence):
            return True

        # Apply section-specific noise patterns
        sentence_stripped = sentence.strip()
        for pattern in section_config.get("noise_patterns", []):
            if re.search(pattern, sentence_stripped, re.IGNORECASE):
                return True

        # Check against section-specific minimum length
        min_length = section_config.get("min_sentence_length", self.MIN_SENTENCE_LENGTH)
        if len(sentence_stripped) < min_length:
            return True

        return False

    def _has_section_actor(self, sentence: str, section_config: Dict[str, Any]) -> bool:
        """
        Check if sentence has an actor expected for this section.

        Args:
            sentence: The text to check
            section_config: Section-specific configuration

        Returns:
            True if sentence contains expected actor
        """
        sentence_lower = sentence.lower()
        expected_actors = section_config.get("expected_actors", [])

        for actor in expected_actors:
            # Build flexible pattern for actor
            actor_pattern = rf'\b{re.escape(actor)}\b'
            if re.search(actor_pattern, sentence_lower):
                return True

        # Fall back to general actor check
        return self._has_actor(sentence)

    def _matches_section_patterns(self, sentence: str, section_config: Dict[str, Any]) -> bool:
        """
        Check if sentence matches any section-specific key patterns.

        Args:
            sentence: The text to check
            section_config: Section-specific configuration

        Returns:
            True if sentence matches section patterns
        """
        sentence_lower = sentence.lower()
        for pattern in section_config.get("key_patterns", []):
            if re.search(pattern, sentence_lower, re.IGNORECASE):
                return True
        return False

    def _determine_section_requirement_type(
        self,
        sentence: str,
        detected_type: RequirementType,
        section_config: Dict[str, Any]
    ) -> RequirementType:
        """
        Determine the most appropriate requirement type for this section.

        Uses section context to refine generic types.

        Args:
            sentence: The requirement text
            detected_type: Type detected from keyword/semantic patterns
            section_config: Section-specific configuration

        Returns:
            Refined requirement type
        """
        primary_types = section_config.get("primary_types", [])
        secondary_types = section_config.get("secondary_types", [])
        default_type = section_config.get("default_type", RequirementType.PERFORMANCE)

        # If detected type is a primary type for this section, keep it
        if detected_type in primary_types:
            return detected_type

        # If detected type is a secondary type, keep it
        if detected_type in secondary_types:
            return detected_type

        # For generic PERFORMANCE type, use section default
        if detected_type == RequirementType.PERFORMANCE:
            return default_type

        # Check sentence content against key patterns to determine type
        sentence_lower = sentence.lower()

        # Map patterns to types
        type_indicators = [
            (r'\b(?:deliverable|report|submit.*report|documentation)\b', RequirementType.DELIVERABLE),
            (r'\b(?:metric|threshold|target|acceptable\s+quality)\b', RequirementType.PERFORMANCE_METRIC),
            (r'\b(?:proposal|offeror|volume)\s+(?:shall|must|should)', RequirementType.PROPOSAL_INSTRUCTION),
            (r'\b(?:evaluate|evaluation|rating|scoring)\b', RequirementType.EVALUATION_CRITERION),
            (r'\b(?:clearance|qualification|certification|experience|years)\b', RequirementType.QUALIFICATION),
            (r'\b(?:FAR|DFARS|comply|compliance|in\s+accordance)\b', RequirementType.COMPLIANCE),
            (r'\b(?:labor|FTE|staff|personnel|hour)\b', RequirementType.LABOR_REQUIREMENT),
            (r'\b(?:page|font|margin|format|limit)\b', RequirementType.FORMAT),
        ]

        for pattern, req_type in type_indicators:
            if re.search(pattern, sentence_lower):
                if req_type in primary_types or req_type in secondary_types:
                    return req_type

        # Default to section default type
        return default_type

    def _extract_from_section(
        self,
        section_id: str,
        section_text: str,
        doc: ParsedDocument
    ) -> List[RequirementNode]:
        """
        Extract requirements with section-specific context and rules.

        Uses SECTION_EXTRACTION_CONFIG for section-aware extraction.
        """
        requirements = []

        # Get section-specific configuration
        section_letter = section_id.replace("section_", "").upper()
        section_config = self._get_section_config(section_letter)

        sentences = self._split_into_sentences(section_text)

        for i, sentence in enumerate(sentences):
            # Apply section-specific quality filters
            if self.strict_mode:
                if self._is_section_noise(sentence, section_config):
                    continue

                # Skip sentences without obligation words
                has_obligation, _ = self._has_obligation_word(sentence)
                if not has_obligation:
                    continue
            else:
                min_len = section_config.get("min_sentence_length", 20)
                if len(sentence.strip()) < min_len:
                    continue

            # Classify sentence with section context
            req_type, keyword = self._classify_sentence(sentence, section_letter)

            if req_type:
                # Refine type based on section rules
                req_type = self._determine_section_requirement_type(
                    sentence, req_type, section_config
                )

                # Check for section-appropriate actor
                has_actor = self._has_section_actor(sentence, section_config)

                # In strict mode, handle conditionals
                if self.strict_mode and keyword in ["should", "may", "can"]:
                    # Skip conditionals without actors unless they match section patterns
                    if not has_actor and not self._matches_section_patterns(sentence, section_config):
                        continue

                req = self._create_requirement_node(
                    sentence=sentence,
                    sentence_index=i,
                    sentences=sentences,
                    doc=doc,
                    req_type=req_type,
                    keyword_match=keyword or "",
                )

                # Apply section-specific confidence boost
                if req.confidence_score:
                    boost = section_config.get("confidence_boost", 0.0)
                    req.confidence_score = max(0.0, min(1.0, req.confidence_score + boost))

                    # Recalculate confidence level
                    req.confidence = self._score_to_confidence_level(req.confidence_score)

                # Add section pattern match bonus
                if self._matches_section_patterns(sentence, section_config):
                    req.confidence_score = min(1.0, req.confidence_score + 0.05)
                    if has_actor:
                        req.confidence_score = min(1.0, req.confidence_score + 0.05)

                # Override section ID with section context
                if req.source:
                    req.source.section_id = section_letter

                requirements.append(req)

        return requirements
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences and then split multi-shall paragraphs"""
        # Handle common abbreviations
        text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '<DOT>', text)  # Abbreviations
        text = re.sub(r'(?<=\d)\.(?=\d)', '<DOT>', text)  # Numbers
        text = re.sub(r'(?<=\s[A-Z])\.(?=\s)', '<DOT>', text)  # Initials

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Apply multi-shall splitting to each sentence
        result = []
        for sentence in sentences:
            split_sentences = self._split_multi_shall(sentence)
            result.extend(split_sentences)

        return result

    def _split_multi_shall(self, text: str) -> List[str]:
        """
        Split paragraphs containing multiple shall/must statements into separate requirements.

        This addresses the audit finding that ~100 requirements per matrix contain
        multiple "shall" statements bundled together.

        Args:
            text: Potentially multi-shall paragraph

        Returns:
            List of individual requirements (may be single item if no split needed)
        """
        # Count obligation words in text
        obligation_pattern = r'\b(?:shall|must|will\s+be\s+required)\b'
        obligation_matches = list(re.finditer(obligation_pattern, text, re.IGNORECASE))

        # If only 0-1 obligation words, no split needed
        if len(obligation_matches) <= 1:
            return [text]

        # Strategy 1: Split on sentence boundaries (period + space + capital letter)
        # This handles: "Sentence one shall X. Sentence two shall Y."
        sentence_split_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        parts = re.split(sentence_split_pattern, text)

        if len(parts) > 1:
            # Filter parts that contain obligation words
            requirements = []
            pending_context = ""

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Check if this part has an obligation word
                if re.search(obligation_pattern, part, re.IGNORECASE):
                    # Prepend any pending context
                    if pending_context:
                        part = pending_context + " " + part
                        pending_context = ""
                    requirements.append(part)
                elif len(requirements) > 0 and len(part) < 150:
                    # Short clause without obligation word - could be context for next
                    pending_context = part
                elif len(requirements) > 0:
                    # Append to previous requirement
                    requirements[-1] = requirements[-1] + " " + part

            if requirements and len(requirements) > 1:
                return requirements

        # Strategy 2: Split on semicolons (common in legal text)
        # This handles: "The fee shall be X; the percent shall be Y; payment shall be Z."
        if len(obligation_matches) > 1:
            semicolon_parts = text.split(';')
            if len(semicolon_parts) > 1:
                requirements = []
                for part in semicolon_parts:
                    part = part.strip()
                    if part and re.search(obligation_pattern, part, re.IGNORECASE):
                        # Add period for proper sentence structure
                        if not part.endswith('.'):
                            part = part + '.'
                        requirements.append(part)

                if len(requirements) > 1:
                    return requirements

        # Strategy 3: Split on "shall" boundaries as last resort
        # This handles run-on sentences: "...shall X and shall Y and shall Z"
        if len(obligation_matches) >= 3:
            # Find positions of each "shall"
            shall_positions = [m.start() for m in obligation_matches]

            # Try to split between shall statements if they're far apart
            if shall_positions[-1] - shall_positions[0] > 200:
                requirements = []
                for i, pos in enumerate(shall_positions):
                    # Find the start of this clause (go back to previous punctuation)
                    start = pos
                    while start > 0 and text[start - 1] not in '.;':
                        start -= 1

                    # Find the end (next shall or end of text)
                    if i < len(shall_positions) - 1:
                        end = shall_positions[i + 1]
                        # Go back to find punctuation before next shall
                        while end > pos and text[end - 1] not in '.;':
                            end -= 1
                    else:
                        end = len(text)

                    clause = text[start:end].strip()
                    if clause and len(clause) > 50:
                        requirements.append(clause)

                if len(requirements) > 1:
                    return requirements

        # Return original if no successful split
        return [text]

    def _has_obligation_word(self, text: str) -> Tuple[bool, str]:
        """
        Check if text contains an obligation word.

        This addresses the audit finding that 24-30% of extractions lack
        obligation words (false positives).

        Args:
            text: Requirement text to check

        Returns:
            Tuple of (has_obligation: bool, matched_word: str or "informational")
        """
        text_lower = text.lower()
        for word in self.OBLIGATION_WORDS:
            if word in text_lower:
                return True, word
        return False, "informational"
    
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
