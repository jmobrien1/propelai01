# As-Built Documentation: PropelAI Annotated Outline Generator v2.12

**Document Version:** 1.0
**Date:** December 14, 2024
**Author:** Claude Code / PropelAI Engineering
**System:** PropelAI Enhanced Compliance Module

---

## 1. Executive Summary

This document describes the implementation of the Smart Proposal Outline Generator and associated fixes completed during the December 2024 development cycle. The primary goal was to transform a generic template-based outline generator into a content-aware system that populates actual RFP requirements into the annotated outline.

### Key Achievements
- Fixed outline generator to create proper volume/section structure
- Implemented Section L/M/C content population with semantic matching
- Fixed P0 constraint type extraction for compliance gate visibility
- Enhanced binding level detection for Section M requirements
- Improved evaluation factor weighting and adjectival rating extraction

---

## 2. System Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PropelAI RFP Processing                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │ PDF/Document     │───▶│ Section-Aware    │                  │
│  │ Ingestion        │    │ Extractor        │                  │
│  └──────────────────┘    └────────┬─────────┘                  │
│                                   │                             │
│                    ┌──────────────┴──────────────┐             │
│                    ▼                              ▼             │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │ Compliance Matrix    │    │ Smart Outline        │          │
│  │ Generator (CTM)      │    │ Generator            │          │
│  └──────────┬───────────┘    └──────────┬───────────┘          │
│             │                            │                      │
│             ▼                            ▼                      │
│  ┌──────────────────────┐    ┌──────────────────────┐          │
│  │ Excel Exporter       │    │ Annotated Outline    │          │
│  │ (best_practices_ctm) │    │ Exporter (JS/Python) │          │
│  └──────────────────────┘    └──────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Files Modified

| File | Purpose | Changes Made |
|------|---------|--------------|
| `smart_outline_generator.py` | Outline generation logic | Volume/section creation, content population, P0 extraction |
| `section_aware_extractor.py` | RFP parsing & extraction | Binding level detection, formatting constraint extraction |
| `best_practices_ctm.py` | Compliance matrix export | Section M matrix columns, priority/binding level display |
| `annotated_outline_exporter.js` | Word document generation | Section L/M/C content rendering, P0 constraint table |
| `annotated_outline_exporter.py` | Python wrapper for JS | Data transformation, JSON serialization |

---

## 3. Detailed Implementation

### 3.1 Volume and Section Creation

#### Problem Statement
The outline generator created volumes without sections, causing the JS exporter to fall back to generic placeholder content.

#### Solution Implemented

**File:** `smart_outline_generator.py`

Three functions were updated to create sections within volumes:

```python
def _create_default_volumes(self, rfp_format: str, section_m: List[Dict]) -> List[ProposalVolume]:
    """
    Create default volumes if none were extracted.
    CRITICAL: Must include sections within each volume so that
    _populate_section_content() has sections to populate.
    """
    tech_sections = [
        ProposalSection(id="SEC-TECH-1", name="Technical Approach"),
        ProposalSection(id="SEC-TECH-2", name="Methodology and Solution Design"),
        ProposalSection(id="SEC-TECH-3", name="Innovation and Technical Excellence"),
    ]
    # ... additional section definitions for mgmt, pp, staffing, cost

    return [
        ProposalVolume(
            id="VOL-TECH", name="Technical Proposal",
            volume_type=VolumeType.TECHNICAL, order=0,
            sections=tech_sections  # <-- KEY: sections now included
        ),
        # ... additional volumes
    ]
```

**Section Templates by Volume Type:**

| Volume Type | Default Sections |
|-------------|-----------------|
| TECHNICAL | Technical Approach, Methodology and Solution, Technical Excellence |
| MANAGEMENT | Management Approach, Project Schedule, Risk Management, Quality Assurance |
| PAST_PERFORMANCE | Past Performance Overview, Contract Reference 1/2/3 |
| COST_PRICE | Cost Summary, Pricing Methodology |
| STAFFING | Staffing Plan, Key Personnel, Organization Chart |
| SMALL_BUSINESS | Small Business Participation, Subcontracting Plan |

### 3.2 Content Population (Iron Triangle Mapping)

#### Requirements (from outline.rtf)
- Section L → Structure & submission instructions
- Section M → Evaluation criteria & scoring weights
- Section C/PWS → Technical requirements

#### Implementation

**File:** `smart_outline_generator.py` - `_populate_section_content()`

```python
def _populate_section_content(
    self,
    volumes: List[ProposalVolume],
    section_l: List[Dict],
    section_m: List[Dict],
    technical: List[Dict],
    eval_factors: List[EvaluationFactor]
):
    """
    Populate proposal sections with actual RFP content.
    This is critical for the annotated outline - without actual content,
    the outline is just a generic template.
    """
    # Factor profiles for semantic matching
    factor_profiles = {
        "experience": {
            "keywords": ["experience", "years", "demonstrated", "prior", "relevant",
                        "comparable", "similar", "contracts", "projects", "track record"],
            "factor_nums": ["1"]
        },
        "management": {
            "keywords": ["management", "project", "program", "schedule", "milestone",
                        "deliverable", "coordination", "oversight", "control", "monitor",
                        "quality", "risk", "communication", "stakeholder"],
            "factor_nums": ["2"]
        },
        # ... additional profiles for technical, personnel, facilities, past_performance
    }
```

**Matching Algorithm:**
1. Detect factor type from section name using keyword matching
2. Score each requirement for relevance to the factor
3. Populate `section.requirements` with Section L content
4. Populate `section.eval_criteria` with Section M content
5. Map technical requirements using relevance scoring

### 3.3 P0 Constraint Type Extraction

#### Problem Statement
P0 constraints showed "UNKNOWN" type despite descriptions being extracted correctly.

#### Root Cause
Data flows through JSON serialization where field names change:
- Python dataclass: `constraint_type`
- JSON output: `type`
- Reading back: code only checked `constraint_type`

#### Solution Implemented

**File:** `smart_outline_generator.py` - `_extract_p0_constraints()`

```python
def get_field(obj, field: str, default: str = '', alt_field: str = None) -> str:
    """Get field from either dict or object, with optional alternate field name"""
    if isinstance(obj, dict):
        # Try primary field first, then alternate
        value = obj.get(field) or (obj.get(alt_field) if alt_field else None)
        return value if value else default
    # For objects, use getattr
    value = getattr(obj, field, None) or (getattr(obj, alt_field, None) if alt_field else None)
    return value if value else default

# Usage:
ctype = get_field(c, 'constraint_type', 'UNKNOWN', alt_field='type')
```

**Extracted Constraint Types:**

| Type | Description Pattern |
|------|-------------------|
| PAGE_LIMIT | "Page limit: X pages", "Section: X pages maximum" |
| FONT_SIZE | "Font size: no smaller than X-point" |
| FONT_FAMILY | "Font family: Times New Roman" |
| MARGIN | "Margins: X-inch margins all around" |
| FILE_FORMAT | "File format: Microsoft Excel/Adobe PDF" |
| SUBMISSION_METHOD | "Electronic submission only via..." |
| DEADLINE | "Proposals must be received by deadline" |
| RATING_THRESHOLD | "Must not receive 'Unacceptable' rating" |

### 3.4 Binding Level Detection Enhancement

#### Implementation

**File:** `section_aware_extractor.py`

Added new evaluation keyword patterns:

```python
EVALUATION_KEYWORDS = [
    # Original patterns
    r'\bwill\s+be\s+evaluat',
    r'\bwill\s+be\s+assessed',
    # ... existing patterns ...

    # NEW patterns added:
    r'\bfactor\s+\d+',
    r'\bphase\s+[ivx\d]+\b',
    r'\bproposal(?:s)?\s+(?:will|shall)\s+be',
    r'\bofferor(?:s)?\s+(?:will|shall|must)',
    r'\b(?:acceptable|unacceptable|marginal|outstanding)\b',
    r'\b(?:point|score)\s*(?:value|s|ing)',
    r'\bevaluat(?:e|ion|ing|ed|or)\b',
    r'\b(?:go|no)[/-]?go\b',
    r'\bpass[/-]?fail\b',
]
```

Added fallback for Section M items:
```python
if is_evaluation_section:
    return BindingLevel.HIGHLY_DESIRABLE, "evaluation"
```

---

## 4. Data Flow

### 4.1 Extraction to Outline Flow

```
RFP Documents (PDF/DOCX)
         │
         ▼
┌────────────────────────┐
│ SectionAwareExtractor  │
│ - Parse documents      │
│ - Classify sections    │
│ - Extract requirements │
│ - Detect binding level │
│ - Extract constraints  │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ ExtractionResult       │
│ - requirements[]       │
│ - formatting_constraints│
│ - volume_structure     │
│ - evaluation_subfactors│
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ SmartOutlineGenerator  │
│ - Detect RFP format    │
│ - Create volumes       │
│ - Create sections      │
│ - Populate content     │
│ - Extract P0 constraints│
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ to_json()              │
│ - Serialize outline    │
│ - Format for JS export │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ AnnotatedOutlineExporter│
│ (Python wrapper)       │
│ - Build input JSON     │
│ - Invoke Node.js       │
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ annotated_outline_     │
│ exporter.js            │
│ - Generate Word doc    │
│ - Apply color coding   │
│ - Create tables        │
└──────────┬─────────────┘
           │
           ▼
    Annotated_Outline.docx
```

### 4.2 JSON Data Structure

```json
{
  "rfpTitle": "RFP Title",
  "solicitationNumber": "TBD",
  "dueDate": "TBD",
  "volumes": [
    {
      "id": "VOL-TECH",
      "name": "Technical Proposal",
      "type": "technical",
      "page_limit": null,
      "sections": [
        {
          "id": "SEC-F1",
          "title": "Factor 1: Experience",
          "name": "Factor 1: Experience",
          "requirements": ["Section L instruction text..."],
          "eval_criteria": ["Section M evaluation text..."],
          "content_requirements": ["..."]
        }
      ]
    }
  ],
  "evaluationFactors": [
    {
      "id": "EVAL-F1",
      "name": "Factor 1: Experience",
      "weight": "TBD",
      "importance": "Most Important"
    }
  ],
  "formatRequirements": {
    "font": "Times New Roman",
    "font_size": 12,
    "p0_constraints": [
      {
        "type": "PAGE_LIMIT",
        "description": "Page limit: 8 pages",
        "applies_to": "Volume 1",
        "consequence": "Excess pages will NOT be read"
      }
    ]
  }
}
```

---

## 5. Output Specifications

### 5.1 Annotated Outline Document Structure

1. **Cover Page**
   - Title: "ANNOTATED PROPOSAL OUTLINE"
   - RFP Title
   - Solicitation Number
   - Due Date
   - Company Name

2. **Submission & Format Requirements**
   - Due date, submission method
   - Font, margins, page size
   - P0 Constraints table (disqualification risks)

3. **Color Code Legend**
   | Color | Source | Purpose |
   |-------|--------|---------|
   | RED | Section L | Instructions |
   | BLUE | Section M | Evaluation criteria |
   | PURPLE | Section C/PWS | Technical requirements |
   | GREEN | Win Themes | Discriminators |
   | ORANGE | Proof Points | Evidence needed |
   | BLUE-GRAY | Graphics | Planned visuals |

4. **Evaluation Factors Summary**
   - Factor name, weight, key criteria

5. **Adjectival Rating Definitions** (for OASIS+ task orders)
   - Exceptional, Very Good, Good, Unacceptable

6. **Volume Outlines**
   - Each volume with page limits
   - Each section with:
     - Section L instructions (RED)
     - Section M evaluation criteria (BLUE)
     - Section C/PWS requirements (PURPLE)
     - Win Themes placeholder (GREEN)
     - Proof Points placeholder (ORANGE)
     - Graphics placeholder (BLUE-GRAY)
     - Content writing area

### 5.2 Compliance Matrix Structure

| Sheet | Content |
|-------|---------|
| Cover Sheet | RFP metadata, statistics |
| Section L Compliance | Instructions with priority |
| Section M Alignment | Eval criteria with binding level |
| Technical Requirements | PWS/SOW requirements |
| All Requirements | Complete list |
| Cross-References | Requirement linkages |

---

## 6. Testing Results

### 6.1 Test Case: Air Force 24x7 NOC RFP

**Input:** Placement Procedures document with:
- 3 volumes (Technical, Cost/Price, Contract Documentation)
- 2 technical subfactors (Management Approach, Infrastructure Approach)
- Page limits per section
- Adjectival rating scale

**Output Validation:**

| Feature | Expected | Actual | Status |
|---------|----------|--------|--------|
| P0 Constraints Type | PAGE_LIMIT, FONT_SIZE, etc. | PAGE_LIMIT, FONT_SIZE, etc. | ✅ PASS |
| Section L Content | RFP instruction text | Populated | ✅ PASS |
| Section M Content | Evaluation criteria | Populated | ✅ PASS |
| Adjectival Ratings | Exceptional/Very Good/Good/Unacceptable | Populated | ✅ PASS |
| Evaluation Weighting | SF1 > SF2 >> Cost | Captured | ✅ PASS |
| Format Requirements | Times New Roman 12pt | Extracted | ✅ PASS |
| Binding Level | Mandatory/Highly Desirable | Detected | ✅ PASS |

---

## 7. Known Limitations

### 7.1 Current Limitations

1. **Solicitation Number Extraction**: Not reliably extracted from all RFP formats
2. **Due Date Extraction**: Date parsing works for common formats but may miss non-standard formats
3. **Duplicate Content**: Factor 1 and Factor 2 sections may have overlapping L/M/C content when semantic matching isn't sufficiently differentiated
4. **Page Limit to Volume Mapping**: Page limits extracted but not always correctly mapped to specific volumes

### 7.2 Edge Cases Not Fully Handled

1. Non-UCF format RFPs (some state/local formats)
2. RFPs with unusual section numbering
3. Scanned PDFs without proper OCR
4. RFPs with page limits specified in attachments only

---

## 8. Dependencies

### 8.1 Python Dependencies
- `openpyxl` - Excel generation
- `dataclasses` - Data modeling
- `re` - Regular expression matching
- `typing` - Type hints

### 8.2 Node.js Dependencies
- `docx` - Word document generation
- `fs` - File system operations

### 8.3 System Requirements
- Python 3.8+
- Node.js 14+
- 512MB RAM minimum for document processing

---

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-14 | Claude Code | Initial as-built documentation |

---

## 10. Appendix: Code Commit History

```
6068aa1 - Fix outline generator to create proper sections within volumes
caa85dc - Fix P0 constraint type extraction for dict/object compatibility
4435763 - Fix P0 constraint type extraction (alt_field for 'type' key)
6b5c8a0 - Latest production build
```
