# PropelAI - Comprehensive Handoff Document

**Last Updated:** November 27, 2025  
**Current Version:** 2.9.0  
**Owner:** Mike (Principal at Thoughtworks, SLED Government Modernization)  
**Repository:** GitHub → Render deployment  

---

## 1. Executive Summary

PropelAI is an AI-powered proposal generation system for government contractors. It automates the extraction of requirements from complex federal RFP documents and generates Compliance Traceability Matrices (CTMs) that follow Shipley Process methodology.

**Core Value Proposition:**
- Reduces manual RFP shredding from days to minutes
- Produces evaluator-ready compliance matrices
- Handles multi-document RFP bundles (400+ pages)
- Tracks amendments and requirement changes

**Target Users:** Government contractors responding to federal, state, and local RFPs

---

## 2. Architecture Overview

### Technology Stack
- **Backend:** FastAPI (Python 3.11+)
- **Frontend:** React (single-file, no build step)
- **Deployment:** Render.com (free tier)
- **AI Integration:** Anthropic Claude API (optional LLM enhancement)
- **Document Processing:** PyMuPDF, python-docx, openpyxl

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (React)                           │
│                     /web/index.html                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                      /api/main.py                               │
│                                                                 │
│  Endpoints:                                                     │
│  - POST /api/rfp                    Create RFP project          │
│  - POST /api/rfp/{id}/upload        Upload documents            │
│  - POST /api/rfp/{id}/process       Legacy extraction           │
│  - POST /api/rfp/{id}/process-semantic    v2.8 extraction       │
│  - POST /api/rfp/{id}/process-best-practices  v2.9 extraction   │
│  - GET  /api/rfp/{id}/export        Download CTM Excel          │
│  - POST /api/rfp/{id}/amendments    Upload amendments           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Enhanced Compliance Agent Module                    │
│              /agents/enhanced_compliance/                        │
│                                                                 │
│  v2.9 Best Practices Pipeline:                                  │
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────┐ │
│  │ Document         │→ │ Section-Aware     │→ │ Best         │ │
│  │ Structure Parser │  │ Extractor         │  │ Practices    │ │
│  │                  │  │                   │  │ CTM Exporter │ │
│  └──────────────────┘  └───────────────────┘  └──────────────┘ │
│                                                                 │
│  Legacy Pipeline:                                               │
│  ┌──────────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Bundle       │→ │ Multi-    │→ │ Require- │→ │ Cross-Ref │  │
│  │ Detector     │  │ Format    │  │ ment     │  │ Resolver  │  │
│  │              │  │ Parser    │  │ Extractor│  │           │  │
│  └──────────────┘  └───────────┘  └──────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

```
propelai/
├── agents/
│   ├── __init__.py
│   ├── enhanced_compliance/           # Main extraction module
│   │   ├── __init__.py               # Exports all components
│   │   ├── models.py                 # Data models (DocumentType, RequirementNode, etc.)
│   │   ├── bundle_detector.py        # Classifies document bundles
│   │   ├── parser.py                 # MultiFormatParser (PDF, DOCX, XLSX)
│   │   ├── extractor.py              # Legacy requirement extractor
│   │   ├── resolver.py               # Cross-reference resolver
│   │   ├── agent.py                  # EnhancedComplianceAgent orchestrator
│   │   ├── amendment_processor.py    # Amendment tracking
│   │   ├── outline_generator.py      # Proposal outline generation
│   │   ├── excel_export.py           # Legacy Excel export
│   │   ├── excel_parser.py           # Parse existing CTM matrices
│   │   ├── semantic_extractor.py     # v2.8: Semantic extraction
│   │   ├── semantic_ctm_export.py    # v2.8: 19-column CTM export
│   │   ├── document_structure.py     # v2.9: RFP structure parser
│   │   ├── section_aware_extractor.py # v2.9: Section-aware extraction
│   │   └── best_practices_ctm.py     # v2.9: Best practices CTM export
│   ├── compliance_agent.py           # Original compliance agent (deprecated)
│   ├── drafting_agent.py             # Section drafting agent
│   ├── strategy_agent.py             # Win strategy agent
│   └── red_team_agent.py             # Red team review agent
│
├── api/
│   ├── __init__.py
│   └── main.py                       # FastAPI application
│
├── web/
│   └── index.html                    # Single-file React UI
│
├── requirements.txt                  # Python dependencies
├── Procfile                          # Render deployment config
└── HANDOFF_DOCUMENT.md              # This file
```

---

## 4. Version History

### v2.9.0 - Best Practices CTM (Current)
**Date:** November 27, 2025

**Key Changes:**
- New `document_structure.py` - Analyzes RFP structure BEFORE extraction
- New `section_aware_extractor.py` - Extracts by section, preserves RFP numbering
- New `best_practices_ctm.py` - Creates THREE distinct matrices (L/M/C)
- Web UI dropdown for extraction mode selection
- Default mode changed to "Best Practices"

**Best Practices Implemented:**
1. Anchor CTM to RFP structure exactly as evaluators expect
2. Extract every requirement - NEVER summarize
3. Separate L (Instructions) from C (Technical) from M (Evaluation)
4. Preserve RFP's own requirement IDs (L.4.B.2, C.3.1.a)
5. Use federal-level 5-6 column model

**CTM Output Structure:**
- Cover Sheet (summary and navigation)
- Section L Compliance (submission instructions)
- Technical Requirements (C/PWS/SOW)
- Section M Alignment (evaluation factors with discriminators)
- All Requirements (complete list)
- Cross-References (L→M→C linkages)

### v2.8.0 - Semantic Extraction
**Date:** November 26, 2025

**Key Changes:**
- `SemanticRequirementExtractor` with 7 requirement types
- 19-column CTM structure with strategic columns
- Garbage filtering to reduce noise
- Type classification (Performance, Proposal Instruction, Evaluation, etc.)
- 5-sheet Excel workbook

**Issues Identified:**
- 809 requirements from NIH RFP (high but legitimate)
- 43% requirements with "UNK" section (section detection gaps)
- No LOW priority items (scoring too aggressive)

### v2.7.0 and Earlier
- Basic keyword-based extraction
- Pattern matching for "shall", "must", "will"
- Simple Excel export
- Amendment processing foundation

---

## 5. Core Components Detail

### Document Structure Parser (`document_structure.py`)

**Purpose:** Analyzes RFP documents to identify section boundaries BEFORE extraction.

**Key Classes:**
- `UCFSection` - Enum for Uniform Contract Format sections (A-M)
- `SectionBoundary` - Detected section with page range and subsections
- `SubsectionBoundary` - Nested sections like L.4.B.2
- `AttachmentInfo` - Attachment metadata and type
- `DocumentStructure` - Complete structural analysis result
- `RFPStructureParser` - Main parser class

**Detection Patterns:**
```python
SECTION_PATTERNS = {
    UCFSection.SECTION_L: [
        r"SECTION\s+L[\s:\-–—]+",
        r"INSTRUCTIONS.*(?:CONDITIONS|NOTICES).*OFFERORS",
    ],
    UCFSection.SECTION_M: [
        r"SECTION\s+M[\s:\-–—]+",
        r"EVALUATION\s+FACTORS",
    ],
    # ... etc
}
```

### Section-Aware Extractor (`section_aware_extractor.py`)

**Purpose:** Extracts requirements while respecting document structure.

**Key Classes:**
- `RequirementCategory` - L_COMPLIANCE, TECHNICAL, EVALUATION, ADMINISTRATIVE, ATTACHMENT
- `BindingLevel` - Mandatory, Highly Desirable, Desirable, Informational
- `StructuredRequirement` - Requirement with full structural context
- `ExtractionResult` - Categorized requirements by type
- `SectionAwareExtractor` - Main extractor class

**Key Principles:**
1. NEVER rename RFP's own requirement references
2. Extract COMPLETE requirement text (paragraphs, not fragments)
3. Maintain clear separation of L/M/C requirements
4. Track cross-references for compliance mapping

**Binding Detection:**
```python
MANDATORY_KEYWORDS = [r'\bshall\b', r'\bmust\b', r'\brequired\s+to\b', ...]
SHOULD_KEYWORDS = [r'\bshould\b', ...]
MAY_KEYWORDS = [r'\bmay\b', r'\bcan\b', r'\bis\s+encouraged\b', ...]
```

### Best Practices CTM Exporter (`best_practices_ctm.py`)

**Purpose:** Creates evaluator-ready Excel workbook with three distinct matrices.

**Output Sheets:**
1. **Cover Sheet** - Summary stats and navigation
2. **Section L Compliance** - For internal use ensuring submission format compliance
3. **Technical Requirements** - Evaluator-facing, proving 100% requirement coverage
4. **Section M Alignment** - Maps evaluation factors to strengths and discriminators
5. **All Requirements** - Complete reference list
6. **Cross-References** - Document structure and L→M→C linkages

**Column Models:**

Section L Compliance:
| RFP Reference | Requirement Text | Source Page | Binding Level | Volume/Section | Compliance Status | Compliance Response | Evidence/Notes |

Technical Requirements:
| Req ID | Requirement Text | Source (PWS/SOW/C) | Page | Binding | Proposal Section | Compliance Status | How We Meet This | Evidence Required | Assigned Owner |

Section M Alignment:
| Evaluation Factor | Criterion Text | Page | Weight/Importance | Proposal Location | Our Strength | Discriminator | Proof Points | Risk/Gap |

---

## 6. API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check, returns version and component status |
| `/api/rfp` | POST | Create new RFP project |
| `/api/rfp/{id}` | GET | Get RFP details |
| `/api/rfp/{id}/upload` | POST | Upload documents (multipart/form-data) |
| `/api/rfp/{id}/process` | POST | Start legacy extraction |
| `/api/rfp/{id}/process-semantic` | POST | Start v2.8 semantic extraction |
| `/api/rfp/{id}/process-best-practices` | POST | Start v2.9 best practices extraction |
| `/api/rfp/{id}/status` | GET | Get processing status |
| `/api/rfp/{id}/requirements` | GET | Get extracted requirements |
| `/api/rfp/{id}/export` | GET | Download CTM Excel file |
| `/api/rfp/{id}/amendments` | POST | Upload amendment |
| `/api/rfp/{id}/amendments` | GET | Get amendment history |

### Processing Flow

```
1. POST /api/rfp           → Create project, get rfp_id
2. POST /api/rfp/{id}/upload  → Upload PDF/DOCX files
3. POST /api/rfp/{id}/process-best-practices → Start extraction
4. GET  /api/rfp/{id}/status  → Poll until completed
5. GET  /api/rfp/{id}/export  → Download CTM
```

---

## 7. Deployment

### Render.com Setup

**Service Type:** Web Service  
**Build Command:** `pip install -r requirements.txt`  
**Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`  
**Environment:** Python 3.11

### Deployment Steps

```bash
# From local machine
cd ~/Downloads/propelai

# Extract new version
unzip -o ~/Downloads/propelai_v2.9_best_practices.zip -d .

# Commit and push
git add -A
git commit -m "v2.9: Best Practices CTM with document structure analysis"
git push origin main

# Render auto-deploys from GitHub
# Check logs at: https://dashboard.render.com
```

### Common Deployment Issues

1. **Git not detecting changes**
   - Cause: Extracted to wrong directory
   - Fix: Ensure unzip uses `-d .` from within propelai directory

2. **Parser missing argument**
   - Cause: `parse_file()` called without `doc_type`
   - Fix: Added `infer_doc_type()` function in API

3. **ParsedDocument attribute error**
   - Cause: Using `.text` instead of `.full_text`
   - Fix: Updated to use `parsed.full_text`

---

## 8. Known Issues & Limitations

### Current Issues

1. **Section Detection Gaps**
   - ~43% of requirements map to "UNK" section
   - Root cause: Section header patterns don't match all RFP formats
   - Impact: Requirements correctly extracted but missing section context

2. **Priority Scoring Too Aggressive**
   - No LOW priority items being assigned
   - All items are HIGH or MEDIUM

3. **Subsection Detection**
   - Not finding all L.4.B.2 style references
   - Regex patterns need tuning for varied formatting

### Limitations

1. **No OCR Support** - Scanned PDFs not supported
2. **No Table Extraction** - Tables converted to text, structure lost
3. **Single Language** - English only
4. **No Cloud Storage** - Files stored in temp directory, lost on restart
5. **No User Authentication** - All users share same instance

---

## 9. Testing

### Local Testing

```python
# Test v2.9 imports
from agents.enhanced_compliance import (
    BEST_PRACTICES_AVAILABLE,
    RFPStructureParser,
    SectionAwareExtractor,
    BestPracticesCTMExporter,
)

# Test structure parsing
parser = RFPStructureParser()
structure = parser.parse_structure([{'text': '...', 'filename': 'test.pdf', 'pages': [...]}])
print(structure.sections.keys())

# Test extraction
extractor = SectionAwareExtractor()
result = extractor.extract([doc])
print(f"Found {len(result.all_requirements)} requirements")

# Test export
exporter = BestPracticesCTMExporter()
exporter.export(result, "/tmp/test.xlsx", "SOL-001", "Test RFP")
```

### Test RFP Used
- **NIH RFP 75N96025R00004** - Scientific support services
- 400+ pages across multiple documents
- Main solicitation + SOW attachment + 2 amendments
- v2.8 extracted 809 requirements (legitimate for this complexity)

---

## 10. Best Practices Reference

The system implements guidance from the embedded best practices document:

### Key Principles Implemented

1. **Anchor to RFP Structure** - Parse L/M/C sections before extracting
2. **Extract Verbatim** - Never summarize in CTM
3. **Separate L from C from M** - Three distinct matrices
4. **Preserve RFP IDs** - Use their numbering, not ours
5. **Federal 5-6 Column Model** - Standard compliance format
6. **Evaluator-Centric Language** - "We meet this by..." not "We intend to..."

### CTM Best Practices Checklist

- [ ] Section L Compliance Matrix (internal use)
- [ ] Technical Requirements Matrix (evaluator-facing)
- [ ] Section M Alignment Matrix (scoring criteria)
- [ ] Requirement IDs match RFP exactly
- [ ] Full verbatim text (no paraphrasing)
- [ ] Cross-references tracked
- [ ] Freeze header row
- [ ] Alternating row shading
- [ ] Mandatory items highlighted

---

## 11. Future Development Ideas

### Short-term Improvements

1. **Fix Section Detection** - Improve regex patterns, add ML classification
2. **Add Priority Scoring** - Implement LOW priority based on "may/should" language
3. **Subsection Linking** - Better L→M→C cross-reference resolution
4. **Table Extraction** - Parse requirement tables maintaining structure

### Medium-term Features

1. **LLM Enhancement** - Enable Claude API for semantic classification
2. **Requirement Consolidation** - Group related requirements from same clause
3. **Win Theme Integration** - Link requirements to win themes
4. **Team Collaboration** - Assign owners, track status

### Long-term Vision

1. **Full Proposal Generation** - Draft compliant responses per requirement
2. **Past Performance Matching** - Auto-suggest relevant experience
3. **Price Volume Integration** - Link requirements to CLIN structure
4. **Competitive Analysis** - Compare against known competitor approaches

---

## 12. Key Contacts & Resources

### Project Owner
- **Mike** - Principal at Thoughtworks, SLED Government Modernization
- Focus: Federal/state digital transformation, GovCon business development

### Resources
- **Deployment:** Render.com dashboard
- **Repository:** GitHub (linked to Render for auto-deploy)
- **Best Practices Doc:** Embedded in this codebase
- **Test Data:** NIH RFP 75N96025R00004

---

## 13. Quick Start for New Claude Session

When starting a new conversation, provide this context:

```
I'm working on PropelAI, an AI-powered RFP compliance matrix generator. 
Current version is 2.9 with Best Practices CTM extraction.

Key files:
- /agents/enhanced_compliance/document_structure.py - RFP structure parser
- /agents/enhanced_compliance/section_aware_extractor.py - Requirement extraction
- /agents/enhanced_compliance/best_practices_ctm.py - CTM export
- /api/main.py - FastAPI backend
- /web/index.html - React frontend

The system processes federal RFPs and generates evaluator-ready compliance 
matrices following Shipley methodology. It creates three distinct matrices:
1. Section L Compliance (submission instructions)
2. Technical Requirements (C/PWS/SOW)
3. Section M Alignment (evaluation factors)

Current issues:
- 43% of requirements mapping to "UNK" section (section detection needs tuning)
- No LOW priority items (scoring too aggressive)

I need help with: [specific task]
```

---

## Appendix A: Dependencies

```
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pydantic>=2.5.0
PyMuPDF>=1.23.0
python-docx>=1.1.0
openpyxl>=3.1.2
anthropic>=0.7.0
aiofiles>=23.2.1
```

---

## Appendix B: Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PORT` | Server port (set by Render) | Yes |
| `ANTHROPIC_API_KEY` | For LLM enhancement | No |

---

*End of Handoff Document*
