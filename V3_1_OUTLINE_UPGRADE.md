# PropelAI v3.1: Smart Outline Generator Upgrade

## 🎯 Problem Statement

The v3.0 Chat Agent successfully implemented router-based architecture with mode-specific protocols, but the Smart Outline Generator was still using generic "Chapter 1, Chapter 2" logic. This created **Asymmetric Intelligence** - the chat feature understood RFP types, but the outline generator didn't.

### Evidence from ao.rtf Analysis
The annotated outline output showed:
- **Spreadsheet RFPs**: Messy list of "Questionnaire" items without drafting space
- **State RFPs**: Original "Section 4.1.1" renumbered to "1.1", confusing evaluators
- **Federal RFPs**: PWS sections (C.1, C.2) listed as top-level headers instead of Volume I subsections

## ✅ v3.1 Solution: Router Integration

### Architecture Changes

**Before v3.1**: Single `generate_from_compliance_matrix()` method
**After v3.1**: Router dispatches to mode-specific generators

```
generate_from_compliance_matrix()
  └─> classify_rfp_type_from_requirements()
      ├─> MODE A: _generate_federal_outline()
      ├─> MODE B: _generate_sled_outline()
      ├─> MODE C: _generate_dod_outline()
      └─> MODE D: _generate_spreadsheet_outline()
```

### Implementation Details

#### 1. RFP Classification (`classify_rfp_type_from_requirements()`)
- **Input**: Section L, M, and Technical requirements + file names
- **Logic**: Same priority-based classification as Chat Agent Router
  1. Spreadsheet (Excel + questionnaire keywords)
  2. DoD Attachments (J.2, J.3, CDRL)
  3. SLED/State (numeric sections without FAR)
  4. Federal (Section L/M/C, FAR clauses)
- **Output**: `RFPType` enum

#### 2. MODE D: Spreadsheet/Questionnaire (`_generate_spreadsheet_outline()`)
**Problem**: Current code tries to create chapters for spreadsheet rows
**Solution**: Drafting Template format

**Output Structure**:
```
Volume: "Questionnaire Response Template"
  Section: "Row 5: [Requirement Text]"
    - DRAFTING INSTRUCTION: Write compliant 'YES' response, max 150 words
    - Requirement: [Full text from Column B]
    - [Empty Drafting Block]
  Section: "Row 6: [Requirement Text]"
    - ...
```

**Key Changes**:
- Flat list (no nested hierarchy)
- Each row becomes a "drafting task"
- Includes instruction, requirement, and space for answer
- No page limits (spreadsheets don't have them)

#### 3. MODE A: Federal/GSA (`_generate_federal_outline()`)
**Problem**: PWS sections (C.1, C.2) listed as top-level headers
**Solution**: Volume-Centric Hierarchy

**Output Structure**:
```
Volume I: Technical Proposal
  └─> Section C.1, C.2, C.3... (from PWS)
Volume II: Past Performance
  └─> Relevant Experience 1, 2, 3...
Volume III: Price
  └─> Pricing details
```

**Key Changes**:
- Enforces 3-volume structure
- All PWS/SOW sections move under Volume I
- Creates placeholder sections for Vol II, III

#### 4. MODE B: SLED/State (`_generate_sled_outline()`)
**Problem**: Renumbers "Section 4.1.1" to "1.1"
**Solution**: Strict Section Mirroring

**Output Structure**:
```
Volume: "Section 4" (kept as-is)
  └─> Section 4.1: [Name]
      └─> Section 4.1.1: [Subsection]
```

**Key Changes**:
- Preserves original section numbering
- Does NOT normalize to "1.1, 1.2"
- Groups by top-level section (Section 4, Section 2, etc.)

#### 5. MODE C: DoD/Attachment-Heavy (`_generate_dod_outline()`)
**Problem**: J-Attachments not prominently featured
**Solution**: Dedicated Attachment Sections

**Output Structure**:
```
Volume I: Technical Proposal
  └─> Attachment J.2: Personnel Qualifications (TOP)
  └─> [Standard PWS sections]
  └─> Attachment J.3: Quality Assurance (BOTTOM)
```

**Key Changes**:
- Inserts J.2 section at top of Volume I
- Adds J.3 section at end
- Uses federal outline as base

---

## 📁 Files Modified

### 1. `/app/agents/enhanced_compliance/smart_outline_generator.py` (Major)
**Changes**:
- Added `RFPType` import from Chat Agent
- Added `classify_rfp_type_from_requirements()` method
- Updated `__init__()` to accept `rfp_type` parameter
- Refactored `generate_from_compliance_matrix()` to use router
- Added 4 new mode-specific generators:
  - `_generate_federal_outline()`
  - `_generate_sled_outline()`
  - `_generate_dod_outline()`
  - `_generate_spreadsheet_outline()`

**Line Count**: ~100 lines added

---

## 🧪 Testing Checklist

### Backend Integration
- [x] Import `RFPType` from Chat Agent successfully
- [x] Classification router compiles without errors
- [x] Backend restarts successfully
- [ ] Test with actual RFP data (requires user upload)

### MODE D: Spreadsheet
- [ ] Upload USCA25Q0053 RFP
- [ ] Generate annotated outline
- [ ] Verify output shows "Drafting Template" structure
- [ ] Verify each row has drafting instruction
- [ ] Verify no chapters/volumes created

### MODE A: Federal
- [ ] Upload NIH or USCG RFP
- [ ] Generate annotated outline
- [ ] Verify Volume I, II, III structure
- [ ] Verify PWS sections are under Volume I
- [ ] Verify Past Performance placeholders exist

### MODE B: SLED
- [ ] Upload West Virginia RFP
- [ ] Generate annotated outline
- [ ] Verify "Section 4.1.1" remains "4.1.1" (not renumbered)
- [ ] Verify top-level sections preserved

### MODE C: DoD
- [ ] Upload DoD RFP with J-Attachments
- [ ] Generate annotated outline
- [ ] Verify J.2 appears at top of Volume I
- [ ] Verify J.3 appears at end

---

## 🎯 Expected Outcomes

### Quality Improvements

**Before v3.1**:
```
Generic Outline (All RFP Types):
1. Chapter 1: Introduction
2. Chapter 2: Technical Approach
3. Chapter 3: Management Plan
```

**After v3.1**:

**Spreadsheet**:
```
Drafting Template:
- Row 5: [Requirement] → Draft YES response
- Row 6: [Requirement] → Draft YES response
```

**Federal**:
```
Volume I: Technical
  - Section C.1: [Task]
  - Section C.2: [Task]
Volume II: Past Performance
Volume III: Price
```

**SLED**:
```
Section 4: Specifications
  - Section 4.1.1: [Requirement]
  - Section 4.2: [Requirement]
```

### User Impact

**For "Brenda" (Proposal Manager)**:
- Spreadsheet RFPs now provide **drafting checklist**, not confusing chapter outline
- Can copy-paste each drafting block directly into Excel
- No more manual restructuring required

**For "Charles" (Capture Manager)**:
- Federal RFPs show proper **3-volume structure** from day one
- State RFPs preserve **original section numbering** for evaluator clarity
- DoD RFPs highlight **J-Attachments** prominently

---

## 🔄 Backward Compatibility

### Existing RFPs
- Old outlines remain unchanged (stored in database)
- New outline generation requests use v3.1 logic
- No migration required

### API Compatibility
- `generate_from_compliance_matrix()` signature extended with optional `file_names` parameter
- Existing callers work without changes (default behavior preserved)
- New callers can pass `file_names` for better classification

---

## 📊 Success Metrics

### Quantitative
1. **Classification Accuracy**: Target 95%+ (same as Chat Agent)
2. **Outline Generation Time**: < 2 seconds
3. **User Satisfaction**: 4.5/5 stars for outline quality

### Qualitative
1. **Outline Usability**: Can users directly use the outline without manual restructuring?
2. **Section Clarity**: Do section names match RFP expectations?
3. **Format Compliance**: Does outline structure align with RFP format?

---

## 🚀 Integration with v3.0 Chat

### Synergy Benefits

**Consistent Classification**:
- Chat Agent classifies RFP → Stores `rfp_type`
- Outline Generator can use stored `rfp_type` (no re-classification)

**Shared Intelligence**:
- Both systems understand RFP types
- User experience is consistent across features

**Future Enhancement**:
- Pass `rfp_type` from Chat to Outline Generator
- Eliminate redundant classification

---

## 🛠️ Developer Notes

### Import Strategy
```python
# Try to import from Chat Agent
try:
    from agents.chat.rfp_chat_agent import RFPType
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False
    # Fallback: Define RFPType locally
    class RFPType(Enum):
        FEDERAL_STANDARD = "federal_standard"
        # ...
```

**Why**: Ensures outline generator works even if Chat Agent is unavailable.

### Classification Logic
```python
# Reuses same priority-based logic as Chat Router
1. Spreadsheet (Excel + keywords)
2. DoD (J-Attachments)
3. SLED (numeric sections, no FAR)
4. Federal (Section L/M/C)
```

**Why**: Consistency across features, proven classification algorithm.

---

## 📝 Next Steps

### Immediate (Post-v3.1)
1. User testing with real RFPs (all 4 modes)
2. Monitor classification accuracy
3. Collect feedback on outline quality

### Short-Term (v3.2)
1. Pass `rfp_type` from Chat to Outline (eliminate re-classification)
2. Add user feedback loop ("Was this outline structure correct?")
3. Fine-tune mode-specific templates

### Long-Term (v4.0)
1. Machine learning for section naming
2. Automated compliance checking
3. Visual outline editor with drag-and-drop

---

## ✅ Validation Status

- [x] Code compiles without errors
- [x] Backend restarts successfully
- [x] v3.0 Chat Agent imports work
- [x] Classification router implemented
- [x] 4 mode-specific generators implemented
- [ ] User testing with real RFPs (pending)

---

**Status**: 🟢 **IMPLEMENTATION COMPLETE - READY FOR USER TESTING**

**Version**: v3.1.0  
**Date**: November 30, 2025  
**Author**: E1 Agent - Emergent Labs  
**Related**: v3.0 Chat Agent (Router-Based Architecture)
