# PropelAI Edge Case Enhancement Plan
## Based on USCA25Q0053 RFQ Analysis

**Date**: December 2024  
**Version**: 4.1 Planning Document  
**Target**: Address non-standard RFP/RFQ formats with multi-document solicitations

---

## Executive Summary

The USCA25Q0053 RFQ analysis reveals critical gaps in PropelAI v4.0's ability to handle:
1. **Multi-document solicitations** (10+ separate files)
2. **Non-UCF formats** (RFQs that don't follow Section L-M-C structure)
3. **Hidden compliance rules** (structural requirements in "RFP Letter" or transmittal docs)
4. **Volume-based page limits** (different limits for different volumes)
5. **Critical formatting constraints** (font, margins, content restrictions)

**Current State**: PropelAI v4.0 correctly detects non-standard format ("appears to be a GSA Schedule, BPA, or non-standard RFP") but **fails to extract critical submission requirements** from auxiliary documents.

**The Gap**: Like "receiving a custom house blueprint where the critical safety codes are hidden on a napkin in the mailbox, not in the main contract binder." PropelAI reads the binder but misses the napkin.

---

## Problem Statement

### Current PropelAI v4.0 Behavior (USCA25Q0053 Test)

**✅ What Works:**
- Detects non-standard format
- Extracts 88 requirements from main solicitation
- Identifies 56 mandatory requirements (SHALL/MUST)
- Categorizes into Technical (6) and Section M Alignment (30)
- Correctly notes "No Section L requirements found"

**❌ What's Missing:**
1. **Page limits not extracted**: 30 pages (Vol I), 2 pages/ref (Vol II), 10 pages (Vol III)
2. **Formatting rules not captured**: 11pt Times New Roman, 1" margins
3. **Critical compliance flag not detected**: "Price ONLY in Volume III" rule
4. **Multi-document correlation**: RFP Letter requirements not linked to main solicitation
5. **Volume allocation**: Requirements not tagged by target volume (I, II, or III)
6. **Excel questionnaire not parsed**: J.2 Requirements Questionnaire.xlsx not ingested
7. **Amendment integration**: /0004 updates not cross-referenced with base requirements

### Risk Assessment

**High Risk Gaps** (Immediate Disqualification):
- Missing page limits → Non-compliant proposal structure
- Price information in Vol I or II → Rule violation, instant rejection
- Non-compliant formatting → Deemed "unacceptable"

**Medium Risk Gaps** (Scoring Impact):
- Incomplete requirements (missing J.2 Excel) → Lower technical score
- No volume allocation → Poor proposal organization
- Missing amendment updates → Outdated deliverables

---

## Enhancement Plan: PropelAI v4.1 "Multi-Doc RFP Handler"

### Phase 1: Multi-Document Ingestion & Correlation (Priority 1)

#### 1.1 Document Bundle Detection

**Current**: Single RFP upload → process  
**New**: Multiple files upload → detect bundle → correlate

**Implementation**:
```
Enhanced upload flow:
1. User uploads 10+ files
2. System detects:
   - Main solicitation (USCA25Q0053.pdf)
   - RFP Letter (transmittal/instructions)
   - Amendments (USCA25Q0053 /0004.pdf)
   - Attachments (J.1, J.2, J.3)
   - Requirements docs
3. Classify each file:
   - Type: Base | Amendment | Attachment | Instruction
   - Priority: Critical | High | Medium | Low
4. Create document hierarchy
```

**New Component**: `bundle_detector.py`

**Features**:
- Filename pattern matching (Amendment, Attachment J.X, RFP Letter)
- Content-based classification (keywords: "AMENDMENT", "hereby incorporated", "submission instructions")
- Relationship mapping (which docs supersede others)

#### 1.2 Auxiliary Document Parser

**Current**: Only parses main solicitation body  
**New**: Parses ALL document types with specialized extractors

**Implementation**:

**A. RFP Letter Extractor** (NEW)
```python
class RFPLetterExtractor:
    """
    Extract submission instructions from RFP Letters/Transmittals.
    
    Targets:
    - Volume structure (e.g., "three volumes")
    - Page limits (e.g., "30 pages", "2 pages per reference")
    - Formatting rules (e.g., "11-point Times New Roman")
    - Critical constraints (e.g., "Price ONLY in Volume III")
    - Due dates
    - Contact information
    """
    
    def detect_volume_structure(text) -> List[Dict]:
        # Extract: "Volume I - Technical Approach"
        # Returns: [{"volume": "I", "name": "Technical Approach", "page_limit": 30}]
    
    def extract_formatting_rules(text) -> Dict:
        # Extract: Font size, font type, margins, line spacing
        # Returns: {"font": "Times New Roman", "size": 11, "margins": "1 inch"}
    
    def detect_compliance_traps(text) -> List[Dict]:
        # Extract: "ONLY in Volume III", "shall not exceed"
        # Returns: [{"rule": "Price ONLY in Volume III", "severity": "CRITICAL"}]
```

**B. Excel Questionnaire Parser** (ENHANCE)
```python
class ExcelQuestionnaireParser:
    """
    Parse Excel-based requirements (J.2 style).
    
    Current: Basic Excel parsing exists
    Enhancement: Detect questionnaire format, extract as requirements
    """
    
    def detect_questionnaire_format(sheet) -> bool:
        # Check for columns: Requirement | Yes/No | Explanation
    
    def extract_questionnaire_requirements(sheet) -> List[Dict]:
        # Each row = 1 requirement
        # Returns: [{"req_id": "Q-001", "text": "...", "response_required": True}]
```

**C. Amendment Delta Extractor** (ENHANCE)
```python
class AmendmentExtractor:
    """
    Extract changes from amendments.
    
    Current: Basic amendment tracking exists
    Enhancement: Structured change extraction
    """
    
    def extract_changes(amendment_text, base_rfp) -> Dict:
        # Detect:
        # - New requirements added
        # - Requirements deleted
        # - Due date changes
        # - Document list updates
        # Returns: {"added": [...], "deleted": [...], "modified": [...]}
```

#### 1.3 Document Correlation Engine

**New Component**: `document_correlator.py`

**Purpose**: Link requirements across multiple documents

```python
class DocumentCorrelator:
    """
    Correlate requirements from multiple sources.
    
    Example:
    - Main solicitation defines work (Section C/PWS)
    - J.1 Attachment provides detailed capabilities
    - J.2 Questionnaire adds granular requirements
    - RFP Letter defines submission structure
    - Amendment updates due dates
    
    Output: Unified requirements list with source tracking
    """
    
    def correlate_documents(doc_bundle) -> CorrelatedRFP:
        # Create master requirements list
        # Track: source_doc, source_page, priority, volume_target
```

---

### Phase 2: Volume-Based Compliance Tracking (Priority 1)

#### 2.1 Volume Allocation Logic

**Current**: Requirements tagged by Section (L, M, C)  
**New**: Requirements tagged by Target Volume (I, II, III)

**Enhancement to `/app/agents/enhanced_compliance/section_aware_extractor.py`**:

```python
def allocate_requirement_to_volume(requirement, rfp_metadata) -> str:
    """
    Determine which volume this requirement should appear in.
    
    Logic:
    - Technical/PWS/SOW requirements → Volume I (Technical Approach)
    - Past performance requirements → Volume II (Past Performance)
    - Pricing/cost requirements → Volume III (Price)
    
    Uses:
    - Keywords: "technical approach", "management approach", "past performance", "price", "cost"
    - Section mapping: C/PWS → Vol I, M (eval) → Vol I, L (submission) → varies
    - RFP Letter instructions
    """
    
volume_allocation = {
    "volume": "I",  # or "II" or "III"
    "volume_name": "Technical Approach",
    "page_limit": 30,
    "page_budget_used": 0  # Track consumption
}
```

#### 2.2 Page Budget Tracker

**New Feature**: Track page limit consumption during outline generation

```python
class PageBudgetTracker:
    """
    Monitor page limit usage across volumes.
    
    Example:
    - Volume I: 30 pages allocated
      - Executive Summary: 2 pages
      - Technical Approach: 18 pages
      - Management Approach: 8 pages
      - Remaining: 2 pages
    """
    
    def track_allocation(volume, section, estimated_pages):
        # Update budget
        # Warn if exceeding limit
        # Suggest content trimming
```

---

### Phase 3: Formatting Rules Extraction (Priority 2)

#### 3.1 Formatting Metadata Capture

**New Field in RFP Data Model**:

```python
class FormattingConstraints(BaseModel):
    font_family: str = "Times New Roman"
    font_size_min: int = 11
    font_size_max: Optional[int] = None
    margins_min: str = "1 inch"
    line_spacing: Optional[str] = None
    page_orientation: str = "Portrait"
    header_footer_allowed: bool = True
    color_allowed: bool = True
    
    # Volume-specific overrides
    volume_overrides: Dict[str, Dict] = {}
```

**Extraction Logic**:
```python
def extract_formatting_requirements(rfp_letter_text) -> FormattingConstraints:
    """
    Parse formatting rules from RFP Letter.
    
    Patterns:
    - "11-point Times New Roman font"
    - "margins of not less than one (1) inch"
    - "single-spaced"
    - "8.5 x 11 inch paper"
    """
```

#### 3.2 Compliance Checker

**New Feature**: Validate proposal documents against formatting constraints

```python
class FormattingComplianceChecker:
    """
    Check if proposal drafts meet formatting requirements.
    
    Future feature: Parse Word/PDF draft and verify:
    - Font compliance
    - Margin compliance
    - Page count within limits
    """
```

---

### Phase 4: Critical Compliance Flag System (Priority 1)

#### 4.1 Compliance Rule Detection

**New Component**: `compliance_rule_detector.py`

**Purpose**: Identify non-negotiable submission rules that are "instant disqualification" if violated

```python
class ComplianceRuleDetector:
    """
    Detect critical compliance rules (traps).
    
    Examples:
    - "Price information shall ONLY appear in Volume III"
    - "Proposals exceeding page limits will not be evaluated"
    - "Late submissions will not be accepted"
    - "Offerors must be registered in SAM.gov"
    """
    
    RULE_PATTERNS = {
        "price_isolation": [
            "price.*only.*volume",
            "pricing.*shall not.*appear",
            "cost.*limited to.*volume iii"
        ],
        "page_limit_enforcement": [
            "exceeding.*page limit.*not evaluated",
            "proposals.*over.*pages.*rejected"
        ],
        "mandatory_registration": [
            "must be registered.*sam",
            "registration.*sam.gov.*required"
        ]
    }
    
    def detect_rules(rfp_documents) -> List[ComplianceRule]:
        """
        Scan all documents for compliance traps.
        
        Returns:
        [
            {
                "rule_id": "CR-001",
                "severity": "CRITICAL",  # CRITICAL | HIGH | MEDIUM
                "category": "price_isolation",
                "text": "Pricing shall be fully explained and stated ONLY in Volume III",
                "source": "RFP Letter, Page 3",
                "impact": "Proposal may be rejected if price appears in Vol I or II",
                "action": "Flag all price/cost mentions in non-Price volumes"
            }
        ]
        """
```

#### 4.2 Compliance Flag in CTM

**Enhancement to Compliance Matrix**:

```python
# Add new column: CRITICAL_FLAGS
{
    "requirement_id": "REQ-042",
    "text": "Describe pricing methodology",
    "section": "C.3.5",
    "volume": "III",
    "page": 15,
    "critical_flags": [
        {
            "flag_id": "CR-001",
            "rule": "Price ONLY in Volume III",
            "action": "Ensure this content stays in Vol III during drafting"
        }
    ]
}
```

#### 4.3 Real-Time Compliance Alerts

**Future Enhancement**: Alert users during drafting

```
⚠️ CRITICAL COMPLIANCE ALERT
You mentioned "cost" in Volume I - Section 2.3.
Rule: "Price information shall ONLY appear in Volume III"
Risk: Proposal may be rejected
Action: Move pricing discussion to Volume III
```

---

### Phase 5: Smart Outline Generator Enhancement (Priority 2)

#### 5.1 Volume-Based Outline Structure

**Current SOG Output**:
```
1. Executive Summary
2. Technical Approach
   2.1 Task 1
   2.2 Task 2
3. Management Approach
4. Past Performance
5. Price
```

**New SOG Output (Volume-Aware)**:
```
VOLUME I: TECHNICAL APPROACH (Page Limit: 30 pages)
├── Formatting: 11pt Times New Roman, 1" margins
├── Cover Sheet (not counted toward page limit)
├── 1. Executive Summary [Est: 2 pages]
├── 2. Technical Approach [Est: 18 pages]
│   ├── 2.1 Requirement C.3.1.1 - Virtual Range Architecture
│   └── 2.2 Requirement C.3.1.2 - Training Scenarios
├── 3. Management Approach [Est: 8 pages]
└── Page Budget: 28/30 used (✓ Compliant)

VOLUME II: PAST PERFORMANCE (Page Limit: 2 pages per reference)
├── Reference 1: USCG Intelligence Modernization [2 pages]
├── Reference 2: VA Cyber Training Platform [2 pages]
└── Reference 3: DOJ Network Security [2 pages]

VOLUME III: PRICE (Page Limit: 10 pages + Excel)
├── ⚠️ CRITICAL: Price information ONLY in this volume
├── 1. Pricing Methodology [Est: 3 pages]
├── 2. Cost Breakdown [Est: 5 pages]
├── 3. Assumptions [Est: 2 pages]
└── Attachment: J.3 Pricing Sheet.xlsx
```

#### 5.2 Outline Metadata

**Enhanced Outline Model**:

```python
class SmartOutlineNode:
    section_number: str  # "2.1"
    title: str  # "Technical Approach"
    volume: str  # "I", "II", or "III"
    volume_name: str  # "Technical Approach"
    page_limit: int  # 30
    estimated_pages: int  # 18
    
    # Requirements this section addresses
    requirements: List[str]  # ["REQ-001", "REQ-005"]
    
    # Compliance flags
    critical_flags: List[ComplianceRule]
    
    # Formatting constraints
    formatting: FormattingConstraints
    
    # Evaluation criteria this maps to
    evaluation_factor: str  # "Criteria 1: Technical Approach"
```

---

### Phase 6: Requirements Questionnaire Integration (Priority 2)

#### 6.1 Excel Requirements Parsing

**Enhance**: `/app/agents/enhanced_compliance/excel_parser.py`

**Current**: Parses pricing spreadsheets  
**New**: Parse requirements questionnaires (J.2 style)

```python
class RequirementsQuestionnaireParser:
    """
    Parse Excel files with requirements in Q&A format.
    
    Expected columns:
    - Req # or ID
    - Requirement Description
    - Yes/No or Compliant/Non-Compliant
    - Explanation/Response
    - Reference (where addressed in proposal)
    """
    
    def detect_questionnaire(sheet) -> bool:
        # Check if Excel follows Q&A pattern
        # Look for columns: requirement, response, explanation
    
    def parse_questionnaire(sheet) -> List[QuestionnaireRequirement]:
        """
        Returns:
        [
            {
                "req_id": "J.2-001",
                "category": "Technical",
                "text": "Does solution support Red Team vs Blue Team scenarios?",
                "response_required": True,
                "response_type": "yes_no_explain",
                "source": "J.2 Requirements Questionnaire.xlsx, Row 5",
                "volume": "I",  # Where to address in proposal
                "page_estimate": 0.25  # Est pages needed to respond
            }
        ]
        """
```

#### 6.2 Questionnaire Integration in CTM

**New Requirement Type**: `QUESTIONNAIRE`

```python
{
    "requirement_id": "REQ-Q-005",
    "requirement_type": "QUESTIONNAIRE",
    "source": "J.2 Requirements Questionnaire.xlsx",
    "question": "Does solution support Red Team vs Blue Team scenarios?",
    "response_format": "Yes/No + Explanation",
    "volume": "I",
    "evaluation_criteria": "Criteria 1: Technical Approach"
}
```

---

### Phase 7: Amendment Delta Tracking (Priority 3)

#### 7.1 Amendment Change Log

**Enhancement**: `/app/agents/enhanced_compliance/amendment_processor.py`

**New Feature**: Structured change tracking

```python
class AmendmentChangeLog:
    amendment_number: str  # "/0004"
    amendment_date: str
    changes: List[AmendmentChange]
    
class AmendmentChange:
    change_type: str  # "ADDED" | "DELETED" | "MODIFIED"
    affected_item: str  # "Due Date" | "Requirement REQ-042" | "Attachment J.1"
    old_value: Optional[str]
    new_value: str
    impact: str  # "CRITICAL" | "MEDIUM" | "LOW"
    affected_volumes: List[str]  # ["I", "III"]
```

**Example Output**:
```
AMENDMENT /0004 CHANGE SUMMARY:
✓ Due date extended: Dec 1 → Dec 3, 2025
⚠️ Attachment J.1 REPLACED: Old deleted, new version added
✓ Technical Volume may include 8-page Technical Solution document
⚠️ Business Volume requirements added: UEI, SAM.gov evidence
```

#### 7.2 Conflict Detection

**Enhanced Conflict Detection**:

```python
def detect_amendment_conflicts(amendments) -> List[Conflict]:
    """
    Detect if multiple amendments contradict each other.
    
    Example:
    - Amendment /0002: "Due date is Nov 30"
    - Amendment /0004: "Due date is Dec 3"
    → Flag: /0004 supersedes /0002
    """
```

---

## Implementation Roadmap

### Sprint 1: Core Multi-Doc Infrastructure (2 weeks)

**Goals**:
- ✅ Implement `bundle_detector.py`
- ✅ Create `document_correlator.py`
- ✅ Add multi-file upload UI
- ✅ Test with USCA25Q0053 bundle (10 files)

**Deliverables**:
- Multi-document upload working
- Document classification functional
- Hierarchy view in UI

---

### Sprint 2: RFP Letter Extraction (2 weeks)

**Goals**:
- ✅ Implement `RFPLetterExtractor`
- ✅ Extract volume structure, page limits, formatting rules
- ✅ Display in Smart Outline Generator
- ✅ Test with 5 different RFP Letters

**Deliverables**:
- Page limits appear in outline
- Formatting constraints captured
- Volume structure detected

---

### Sprint 3: Compliance Flag System (1 week)

**Goals**:
- ✅ Implement `ComplianceRuleDetector`
- ✅ Add critical_flags to CTM
- ✅ Display flags in UI (red alerts)
- ✅ Test "Price ONLY in Volume III" detection

**Deliverables**:
- Critical compliance rules extracted
- Flags visible in compliance matrix
- Alert system functional

---

### Sprint 4: Volume Allocation & Budget Tracking (1 week)

**Goals**:
- ✅ Add volume allocation to requirements
- ✅ Implement `PageBudgetTracker`
- ✅ Update outline to show page budgets
- ✅ Test with volume-based proposals

**Deliverables**:
- Requirements tagged by volume
- Page budget tracking working
- Over-limit warnings functional

---

### Sprint 5: Excel Questionnaire Parser (1 week)

**Goals**:
- ✅ Enhance `excel_parser.py` for questionnaires
- ✅ Parse J.2 style Excel files
- ✅ Integrate questionnaire requirements into CTM
- ✅ Test with 3 different questionnaire formats

**Deliverables**:
- Excel questionnaires parsed
- Questions appear as requirements
- Response format captured

---

### Sprint 6: Amendment Delta Tracking (1 week)

**Goals**:
- ✅ Implement structured change extraction
- ✅ Create amendment change log UI
- ✅ Add conflict detection
- ✅ Test with USCA25Q0053 /0004 amendment

**Deliverables**:
- Amendment changes tracked
- Change summary displayed
- Conflicts detected and flagged

---

### Sprint 7: Integration Testing & Polish (1 week)

**Goals**:
- ✅ End-to-end test with USCA25Q0053 (10 files)
- ✅ Verify all requirements extracted (main + J.1 + J.2 + RFP Letter)
- ✅ Verify outline includes page limits and formatting
- ✅ Verify compliance flags appear
- ✅ Bug fixes and UX improvements

**Deliverables**:
- Full USCA25Q0053 test passes
- All 88+ requirements captured
- Page limits in outline
- Price isolation flag detected

---

## Testing Plan

### Test Case 1: USCA25Q0053 (Cyber Range RFQ)

**Input Files**:
1. USCA25Q0053.pdf (main solicitation)
2. RFP Letter (submission instructions)
3. USCA25Q0053 /0004 Amendment.pdf
4. Attachment J.1 Required Capabilities Document.pdf
5. Attachment J.2 Requirements Questionnaire.xlsx
6. Attachment J.3 Pricing Sheet.xlsx
7-10. Additional clauses/documents

**Expected Output**:

**Compliance Matrix**:
- ✅ 88+ requirements extracted from ALL sources
- ✅ Volume allocation: I, II, or III
- ✅ Excel questionnaire requirements included (J.2)
- ✅ Critical flag: "Price ONLY in Volume III"
- ✅ Amendment /0004 changes integrated

**Smart Outline**:
```
VOLUME I: TECHNICAL APPROACH (30 pages max)
└── Formatting: 11pt Times New Roman, 1" margins
└── Requirements: REQ-001 to REQ-056 (Technical/PWS)

VOLUME II: PAST PERFORMANCE (2 pages per ref)
└── Requirements: REQ-057 to REQ-068 (M.2 references)

VOLUME III: PRICE (10 pages + Excel)
└── ⚠️ CRITICAL: Price information ONLY here
└── Requirements: REQ-069 to REQ-074 (Pricing)
└── Attachment: J.3 Pricing Sheet.xlsx
```

**Amendment Change Log**:
```
Amendment /0004 Changes:
- Due date: Dec 1 → Dec 3, 2025
- J.1 replaced with new version
- 8-page Technical Solution document allowed in Vol I
- Business Volume requirements added
```

---

### Test Case 2: Standard FAR 15 RFP (Control)

**Purpose**: Verify new features don't break existing functionality

**Input**: Single PDF with Section L, M, C

**Expected**: Works as before (backward compatible)

---

### Test Case 3: GSA Schedule RFQ

**Purpose**: Test RFP Letter extraction with Task Order format

**Input**: 
- GSA RFQ.pdf
- RFQ Letter.pdf (instructions)

**Expected**:
- Instructions extracted from RFQ Letter
- Task Order format detected
- No Section L/M/C (non-UCF)

---

## Success Metrics

### Functional Metrics

| Metric | Current v4.0 | Target v4.1 |
|--------|--------------|-------------|
| **Multi-document handling** | ❌ Single file only | ✅ 10+ files |
| **Page limit extraction** | ❌ 0% | ✅ 95%+ |
| **Formatting rule extraction** | ❌ 0% | ✅ 90%+ |
| **Critical compliance flags** | ❌ 0 detected | ✅ 3-5 per RFP |
| **Volume allocation** | ❌ N/A | ✅ 100% of reqs |
| **Excel questionnaire parsing** | ⚠️ Basic | ✅ Advanced |
| **Amendment change tracking** | ⚠️ Basic | ✅ Structured |

### Quality Metrics

| Metric | Target |
|--------|--------|
| **Requirement extraction recall** | 95%+ (no missing reqs) |
| **Precision** | 90%+ (no false requirements) |
| **Processing time** | <2 min for 10-file bundle |
| **User error rate** | <5% (missing critical rules) |

---

## Risk Assessment

### High Risk Items

**1. Parsing Complexity**
- **Risk**: RFP Letters have no standard format
- **Mitigation**: Pattern matching + LLM extraction fallback
- **Contingency**: Human-in-the-loop review for critical flags

**2. Performance**
- **Risk**: 10+ files = slow processing
- **Mitigation**: Parallel document parsing, async operations
- **Contingency**: Background processing with status updates

**3. Backward Compatibility**
- **Risk**: Breaking existing single-file workflows
- **Mitigation**: Feature flags, gradual rollout
- **Contingency**: v4.0 mode toggle

### Medium Risk Items

**1. Excel Format Variability**
- **Risk**: Questionnaires may have custom formats
- **Mitigation**: Flexible column detection
- **Contingency**: Manual mapping UI for unknown formats

**2. Amendment Complexity**
- **Risk**: Complex amendment chains (5+ amendments)
- **Mitigation**: Chronological ordering, conflict detection
- **Contingency**: Amendment summary review screen

---

## UI/UX Enhancements

### Upload Flow

**Before**:
```
[Drag file here]
→ Upload
→ Process
```

**After**:
```
[Drag MULTIPLE files here]
→ System detects bundle
→ Show document classification:
   ✓ Main Solicitation (USCA25Q0053.pdf)
   ✓ RFP Letter (instructions)
   ✓ Amendment /0004
   ✓ Attachments (J.1, J.2, J.3)
→ User confirms or adjusts classification
→ Process bundle
```

### Compliance Matrix View

**New Columns**:
- **Volume**: I, II, or III
- **Page Budget**: Contribution to page limit
- **Critical Flags**: 🚩 icon for compliance traps
- **Source Doc**: Which file the requirement came from

### Smart Outline View

**New Features**:
- Volume headers with page limits
- Page budget tracker (visual bar)
- Formatting rule callout box
- Critical compliance alerts at top

---

## Documentation Updates

### For Users

**New Guide**: "Working with Multi-Document RFPs"
- How to upload document bundles
- Understanding document classification
- Reviewing compliance flags
- Managing page budgets

### For Developers

**New Architecture Docs**:
- `bundle_detector.py` design
- `document_correlator.py` logic
- Volume allocation algorithm
- Compliance rule patterns

---

## Future Enhancements (v4.2+)

### AI-Powered Document Classification

**Replace**: Pattern matching  
**With**: LLM-based classification

```python
# Use Claude to classify documents
prompt = f"""
You are analyzing a government solicitation bundle.
Classify this document: {filename}

Content preview: {first_500_chars}

Is this:
A) Main solicitation
B) RFP Letter/Instructions
C) Amendment
D) Attachment (J.X)
E) Requirements document
F) Other

Provide classification and confidence score.
"""
```

### Natural Language Compliance Rule Detection

**Replace**: Regex patterns  
**With**: LLM extraction

```python
# Use Claude to find compliance traps
prompt = """
Analyze this RFP Letter for critical compliance rules that 
could result in proposal rejection if violated.

Examples:
- Page limit enforcement
- Content restrictions (e.g., price only in specific volume)
- Mandatory registrations
- Formatting requirements
- Due date penalties

Extract all such rules with severity level.
"""
```

### Smart Volume Optimization

**Feature**: AI suggests optimal page allocation

```python
# Analyze requirements density
# Suggest: "Task 2.3 requires 8 pages but you allocated 5"
# Auto-balance: Move content between sections to fit limits
```

---

## Conclusion

The USCA25Q0053 RFQ reveals that **PropelAI v4.0's single-document assumption is insufficient** for real-world government contracting. 

**The Enhancement Plan addresses**:
1. ✅ Multi-document bundle handling
2. ✅ Auxiliary document parsing (RFP Letters, Excel)
3. ✅ Volume-based compliance tracking
4. ✅ Critical rule detection (compliance traps)
5. ✅ Page budget management
6. ✅ Amendment integration

**Implementation**: 9 weeks (7 sprints)

**Expected Outcome**: PropelAI v4.1 successfully processes complex, multi-document solicitations like USCA25Q0053, extracting 100% of requirements (main + attachments + instructions) and preventing compliance failures through proactive flagging of critical submission rules.

**Metaphor Resolution**: We're teaching PropelAI to not just read the main blueprint but also check the mailbox, the sticky notes, and the amendment pile—ensuring no critical "napkin instructions" are ever missed.

---

**Next Step**: Review and approve plan → Begin Sprint 1 implementation

