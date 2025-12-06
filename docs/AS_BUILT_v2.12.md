# PropelAI Enhanced Compliance Module - As-Built Documentation

**Version:** 2.12
**Date:** December 2024
**Status:** Production Ready

---

## Executive Summary

PropelAI's Enhanced Compliance Module is an autonomous RFP analysis system that extracts requirements from federal solicitations and generates structured compliance artifacts. Version 2.12 introduces specialized support for OASIS+ Task Orders with P0 constraint tracking, mandatory artifact management, and adjectival rating awareness.

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PropelAI API Layer                          │
│                         (api/main.py)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Document        │  │ Section-Aware   │  │ Smart Outline       │ │
│  │ Structure       │  │ Extractor       │  │ Generator (SOG)     │ │
│  │ Parser          │  │                 │  │                     │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘ │
│           │                    │                      │            │
│           ▼                    ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Extraction Result                         │   │
│  │  - StructuredRequirement[]                                   │   │
│  │  - FormattingConstraint[] (P0)                               │   │
│  │  - VolumeStructure[]                                         │   │
│  │  - EvaluationSubFactor[]                                     │   │
│  │  - AdjectivalRatings{}                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │ Best Practices  │  │ Annotated       │  │ Excel Export        │ │
│  │ CTM Exporter    │  │ Outline         │  │ (Compliance Matrix) │ │
│  │ (.xlsx)         │  │ Exporter (.docx)│  │                     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
agents/enhanced_compliance/
├── __init__.py                      # Module exports
├── section_aware_extractor.py       # Core extraction engine (v2.12)
├── document_structure.py            # RFP structure parser
├── smart_outline_generator.py       # SOG with OASIS+ support
├── annotated_outline_exporter.py    # Python wrapper for DOCX export
├── annotated_outline_exporter.js    # Node.js DOCX generator
├── best_practices_ctm.py            # Compliance matrix exporter
├── excel_export.py                  # Legacy Excel export
├── bundle_detector.py               # Multi-document RFP detection
└── outline_generator.py             # Legacy outline generator
```

---

## Feature Inventory

### 1. RFP Processing Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rfp` | POST | Create new RFP entry |
| `/api/rfp/{id}/upload` | POST | Upload RFP documents |
| `/api/rfp/{id}/process-best-practices` | POST | Extract requirements (v2.9+) |
| `/api/rfp/{id}/compliance-matrix` | GET | Download Excel compliance matrix |
| `/api/rfp/{id}/outline` | GET/POST | Generate proposal outline |
| `/api/rfp/{id}/outline/export` | GET | Download annotated outline (.docx) |

### 2. Document Structure Analysis

**Capabilities:**
- UCF Section Detection (L, M, C/PWS/SOW)
- Solicitation Number Extraction (NIH, DoD, GSA formats)
- OASIS+ Task Order Detection
- Placement Procedures Recognition

**Supported Solicitation Patterns:**
```python
# NIH Format
r"75N\d{5}[A-Z]\d{5}"                    # 75N96025R00004

# DoD Air Force (with/without hyphens)
r"FA\d{4}-\d{2}-[RQ]-\d{4}"              # FA8806-25-R-B003
r"FA\d{6}[RQ][A-Z]?\d{3,}"               # FA880625RB003

# DoD Army
r"W\d{3}[A-Z]{2}\d{2}[RQ]\d{4,}"         # W912HQ25R0001

# DoD Navy
r"N\d{5}\d{2}[RQ]\d+"                    # N0017826R30020003
```

### 3. Requirement Extraction

**Data Model: `StructuredRequirement`**
```python
@dataclass
class StructuredRequirement:
    id: str                          # Unique identifier
    rfp_reference: str               # Original RFP reference (L.4.B.2)
    requirement_text: str            # Full requirement text
    binding_level: BindingLevel      # MANDATORY, HIGHLY_DESIRABLE, DESIRABLE
    source_section: UCFSection       # L, M, C, etc.
    source_document: str             # Source filename
    compliance_approach: str         # Suggested approach
    evidence_needed: str             # Required evidence
    category: str                    # L_COMPLIANCE, EVALUATION, TECHNICAL
```

**Binding Level Detection:**
- **Mandatory (SHALL/MUST)**: 90% confidence
- **Highly Desirable (SHOULD)**: 85% confidence
- **Desirable (MAY)**: 75% confidence

### 4. OASIS+ Task Order Support (v2.12)

**P0 Constraint Extraction:**
```python
@dataclass
class FormattingConstraint:
    constraint_type: str    # PAGE_LIMIT, FONT_SIZE, MARGIN, etc.
    description: str        # Human-readable description
    value: str              # Numeric or text value
    applies_to: str         # Volume/section affected
    consequence: str        # What happens if violated
    priority: str = "P0"    # Always P0 for pass/fail constraints
```

**Extracted Constraint Types:**
| Type | Pattern Examples | Consequence |
|------|-----------------|-------------|
| PAGE_LIMIT | "Executive Summary 1", "10 pages maximum" | Excess pages NOT read |
| FONT_SIZE | "no smaller than 12-point" | Disqualification |
| FONT_FAMILY | "Times New Roman" | Disqualification |
| MARGIN | "one-inch margins all around" | Disqualification |
| FILE_FORMAT | "MS Word 2016, MS Excel 2016" | Non-compliant |
| SUBMISSION_METHOD | "Electronic submission only" | Ineligible |
| DEADLINE | "Late proposals" | Ineligible |
| RATING_THRESHOLD | "Unacceptable = ineligible" | Disqualification |

**Volume Structure:**
```python
@dataclass
class VolumeStructure:
    volume_number: int              # 1, 2, 3
    volume_name: str                # "Technical", "Cost/Price", etc.
    required_content: List[str]     # Required sections
    page_limit: Optional[int]       # If specified
    subfactors: List[str]           # Evaluation subfactors
```

**Adjectival Ratings:**
```python
adjectival_ratings = {
    "Exceptional": "Clearly exceeds minimum requirements, contains multiple strengths...",
    "Very Good": "Clearly exceeds minimum requirements, contains at least one strength...",
    "Good": "Clearly meets the minimum requirements...",
    "Unacceptable": "Does not clearly meet the minimum requirements..."
}
```

**Mandatory Artifacts (Volume 3):**
```python
@dataclass
class MandatoryArtifact:
    artifact_id: str                # Unique ID
    name: str                       # "SF1449", "DD-254"
    description: str                # Instructions
    far_reference: Optional[str]    # FAR 52.204-7
    form_number: Optional[str]      # Form identifier
    is_pass_fail: bool = True       # Always pass/fail
```

### 5. Smart Outline Generator (SOG)

**Features:**
- Evaluation Factor Weighting: `SF1 > SF2 >> Cost/Price`
- P0 Constraint Integration
- Mandatory Artifact Tracking
- Win Theme Placeholders
- Adjectival Rating Definitions

**Entry Points:**
```python
# Standard generation
outline = generator.generate_from_compliance_matrix(
    section_l_requirements,
    section_m_requirements,
    technical_requirements,
    stats
)

# OASIS+ enhanced generation
outline = generator.generate_with_oasis_data(
    section_l_requirements,
    section_m_requirements,
    technical_requirements,
    stats,
    oasis_task_order_data={
        "formatting_constraints": [...],
        "adjectival_ratings": {...},
        "volume_structure": [...],
        "evaluation_subfactors": [...]
    }
)
```

### 6. Annotated Outline Export

**Document Sections:**
1. Cover Page (Title, Solicitation #, Due Date)
2. Submission & Format Requirements
3. **P0 Constraints Table** (v2.12)
4. Color Code Legend
5. Evaluation Factors Summary
6. **Adjectival Rating Definitions** (v2.12)
7. Volume 1: Technical Proposal
8. Volume 2: Cost/Price Proposal
9. **Volume 3: Contract Documentation** (v2.12)

**Color Coding:**
| Color | Section | Hex Code |
|-------|---------|----------|
| Orange | Section L (Instructions) | #C65911 |
| Purple | Section M (Evaluation) | #7030A0 |
| Blue | Section C/PWS (Technical) | #2E75B6 |
| Green | Win Themes | #538135 |
| Yellow | Proof Points | #FFF2CC |

### 7. Compliance Matrix Export

**Excel Structure:**
| Sheet | Content |
|-------|---------|
| Cover Sheet | Solicitation #, Counts, Version |
| All Requirements | Complete requirement list |
| Section L Compliance | Instructions requirements |
| Section M Alignment | Evaluation requirements |
| Technical Requirements | C/PWS/SOW requirements |
| Attachments | Attachment requirements |

**Filename Convention:**
```
{SolicitationNumber}_{Agency}_ComplianceMatrix.xlsx
Example: FA880625RB003_AirForce_ComplianceMatrix.xlsx
```

---

## API Response Structures

### Best Practices Extraction Result

```json
{
  "status": "completed",
  "requirements": [...],
  "stats": {
    "total_requirements": 486,
    "by_section": {
      "L": 80,
      "M": 82,
      "C": 76,
      "ATTACHMENT": 75
    },
    "by_binding_level": {
      "Mandatory": 317,
      "Highly Desirable": 34,
      "Desirable": 62
    },
    "rfp_type": "OASIS_TASK_ORDER",
    "oasis_task_order": {
      "placement_procedures_source": "Attachment 2. Placement Procedures.pdf",
      "formatting_constraints": [...],
      "volume_structure": [...],
      "adjectival_ratings": {...}
    }
  }
}
```

### Outline JSON Structure

```json
{
  "volumes": [
    {
      "id": "VOL-1",
      "name": "Technical Volume",
      "page_limit": 21,
      "sections": [...]
    }
  ],
  "evaluation_factors": [
    {
      "name": "Management Approach",
      "weight": "More Important",
      "rating_scale": "Exceptional/Very Good/Good/Unacceptable"
    }
  ],
  "format_requirements": {
    "font": "Times New Roman",
    "font_size": "12-point",
    "margins": "1 inch",
    "p0_constraints": [...]
  },
  "mandatory_artifacts": [...],
  "adjectival_ratings": {...}
}
```

---

## Known Limitations

### Current Constraints

1. **PDF Text Extraction**: Relies on external PDF libraries; scanned PDFs require OCR
2. **Table Detection**: Limited ability to parse complex tables in PDFs
3. **Win Theme Population**: Placeholders only; requires manual entry or future AI integration
4. **Confidence Scores**: Calculated heuristically, not ML-based
5. **Multi-Volume RFPs**: Some edge cases with non-standard volume numbering

### OASIS+ Specific

1. **Adjectival Ratings**: Extracted from Placement Procedures; may miss custom definitions
2. **P0 Constraints**: Pattern-based extraction; unusual phrasing may be missed
3. **Volume 3 Artifacts**: Standard DoD forms; agency-specific forms may need manual addition

---

## Testing Verification

### Test RFP: FA880625RB003 (Air Force 24x7 NOC)

| Metric | Before v2.12 | After v2.12 |
|--------|--------------|-------------|
| Section L Requirements | 2 | 80 |
| Section M Requirements | 4 | 82 |
| Total Requirements | ~50 | 486 |
| P0 Constraints | 0 | 12+ |
| Solicitation Extracted | No | Yes |
| Filename Correct | No | Yes |
| Confidence Scores | 0% | 75-95% |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.9 | Nov 2024 | Best Practices CTM extraction |
| 2.10 | Nov 2024 | Smart Outline Generator |
| 2.11 | Nov 2024 | Annotated Outline Export |
| 2.12 | Dec 2024 | OASIS+ P0 constraints, Volume 3, Adjectival ratings |

---

## Dependencies

### Python
- FastAPI, Uvicorn
- python-docx (optional)
- openpyxl

### Node.js
- docx (DOCX generation)

### System
- Node.js 16+ (for annotated outline export)

---

## Deployment Notes

1. Ensure Node.js is installed for DOCX export functionality
2. Run `npm install docx` in the enhanced_compliance directory
3. API endpoints auto-detect OASIS+ task orders based on document content
4. Compliance matrix and annotated outline use consistent naming based on extracted solicitation number
