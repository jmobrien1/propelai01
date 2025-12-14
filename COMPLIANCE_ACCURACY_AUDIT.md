# PropelAI Compliance Accuracy Audit Report

**Date:** December 14, 2024
**Audited By:** Automated Analysis
**Scope:** 10 compliance matrices from production RFP processing
**Status:** ✅ FIXES IMPLEMENTED

---

## Executive Summary

PropelAI's compliance extraction achieves approximately **62-70% quality score** based on detected issues. While the system successfully extracts requirements with proper binding level classification (80% Mandatory), there are three critical issues impacting accuracy:

| Issue | Impact | Count | Priority | Status |
|-------|--------|-------|----------|--------|
| Multi-shall bundles | Under-counting requirements | 100-104 per matrix | **HIGH** | ✅ FIXED |
| Missing obligation words | False positives (noise) | 24-30% of extractions | **HIGH** | ✅ FIXED |
| Very long extractions | Requirement bundling | 25-30% of extractions | **MEDIUM** | ✅ FIXED |
| Enhanced noise patterns | Noise reduction | ~74 items | **LOW** | ✅ FIXED |
| Title extraction bugs | Wrong RFP titles | Various | **HIGH** | ✅ FIXED |

### Fixes Applied (December 14, 2024)

1. **Multi-shall splitting** - `_split_multi_shall()` method with 3 strategies
2. **Obligation word validation** - `_has_obligation_word()` filters non-requirements
3. **MAX_SENTENCE_LENGTH** reduced from 1000 to 500 chars
4. **Enhanced noise patterns** - 10 additional patterns added
5. **Title extraction** - Rejects conjunctions, articles, prepositions, partial sentences

---

## Audit Methodology

### Matrices Analyzed
```
1. FA880625RB003_AirForce_ComplianceMatrix.xlsx (486 requirements)
2. 75N96025R00004_NIH_ComplianceMatrix.xlsx (986 requirements)
3. 3RFP-A66C5F14_ComplianceMatrix.xlsx (529 requirements)
4. 4RFP-F42523C2_ComplianceMatrix.xlsx (529 requirements)
5. 5RFP-CD7053C7_ComplianceMatrix.xlsx (529 requirements)
6. 6RFP-CEAA0F9F_ComplianceMatrix.xlsx (529 requirements)
7. 1RFP-0D257679_ComplianceMatrix.xlsx (299 requirements)
8. 2RFP-0229F21E_ComplianceMatrix.xlsx (139 requirements)
9. 2RFP-B20C3E8D_ComplianceMatrix.xlsx (299 requirements)
10. 75N96025R00004_2_NIH_ComplianceMatrix.xlsx (986 requirements)
```

**Total Requirements Analyzed:** 4,811

### Metrics Measured
1. **Binding Level Distribution** - Are requirements correctly classified?
2. **Multi-shall Detection** - Are bundled requirements being split?
3. **Length Analysis** - Are extractions appropriately scoped?
4. **Obligation Word Presence** - Are non-requirements being captured?
5. **Duplicate Detection** - Are requirements being extracted multiple times?

---

## Findings

### 1. Binding Level Distribution ✅ GOOD

```
Mandatory:        80.2%
Desirable:        10.6%
Highly Desirable:  9.2%
```

**Assessment:** Binding level classification is working correctly. The 80% Mandatory rate aligns with expected RFP structure where most "shall" statements are binding.

---

### 2. Multi-Shall Bundles ⚠️ CRITICAL

**Issue:** Requirements containing multiple "shall" statements are being extracted as single items.

| Matrix | Multi-Shall Count | Impact |
|--------|-------------------|--------|
| NIH | 104 requirements | Each contains 2-5 "shall" statements |
| Air Force | 104 requirements | Under-counting by ~200-400 requirements |

**Example (NIH):**
```
[Contains 3 'shall' statements]
"The fixed fee for the Base Period of this contract is $TBD. The fixed fee
shall be paid in direct ratio to the level of effort expended; that is, the
percent of fee paid shall be equal to the percent of total effort expended.
Payment shall be subject to the withholding provisions..."
```

**Root Cause:** The extractor captures paragraph-level text without splitting on sentence boundaries.

**Recommendation:** Implement sentence-level splitting for paragraphs containing multiple obligation keywords.

---

### 3. Missing Obligation Words ⚠️ HIGH

**Issue:** 24-30% of extracted "requirements" contain no obligation word (shall/must/will/required).

| Matrix | No Obligation Word | Percentage |
|--------|-------------------|------------|
| NIH | 236 | 23.9% |
| Air Force | 144 | 29.6% |

**Assessment:** These extractions are likely:
- Informational context
- Headers/section titles captured as requirements
- Cross-references without binding language

**Recommendation:**
1. Add post-extraction filter for obligation word presence
2. Reclassify as "Informational" binding level
3. Separate into "Context" category in output

---

### 4. Very Long Extractions ⚠️ MEDIUM

**Issue:** 25-30% of extractions exceed 500 characters.

| Matrix | Long Extractions | Percentage |
|--------|-----------------|------------|
| NIH | 296 | 30% |
| Air Force | 125 | 26% |

**Root Cause:** MAX_SENTENCE_LENGTH in extractor.py is 1000 chars, allowing paragraph-level captures.

**Recommendation:** Reduce MAX_SENTENCE_LENGTH to 500 or implement paragraph splitting.

---

### 5. Duplicate Detection ✅ GOOD

**Finding:** No exact duplicates detected in sampled matrices.

**Assessment:** Deduplication logic (hash-based) is working correctly.

---

### 6. Noise Filtering ⚠️ LOW

**Issue:** ~74 items in NIH matrix match noise patterns (headers, page numbers, section markers).

**Root Cause:** NOISE_PATTERNS in extractor.py aren't catching all patterns.

**Recommendation:** Add patterns for:
- FAR clause references used as headers
- Attachment title lines
- Cross-reference-only lines

---

## Quality Score Calculation

```
Total Requirements: 986 (NIH sample)
Issues Detected:
  - Very long (>500 chars): 296
  - Multi-shall bundles: 104
  - Noise patterns: 74
  - No obligation word: 236 (overlap with above)

Unique Issues: ~370
Quality Score: 100 - (370/986 * 100) = 62.5%
```

---

## Recommendations by Priority

### Priority 1: Multi-Shall Splitting (HIGH IMPACT)

**File:** `agents/enhanced_compliance/extractor.py`

Add sentence-level splitting:
```python
def split_multi_shall(text: str) -> List[str]:
    """Split paragraphs containing multiple shall statements"""
    sentences = re.split(r'(?<=[.;])\s+', text)
    requirements = []
    for sentence in sentences:
        if re.search(r'\b(shall|must|will|required)\b', sentence, re.I):
            requirements.append(sentence.strip())
    return requirements if len(requirements) > 1 else [text]
```

### Priority 2: Obligation Word Filter (HIGH IMPACT)

Add post-extraction validation:
```python
def validate_requirement(text: str) -> Tuple[bool, str]:
    """Check if text contains obligation language"""
    obligation_words = ['shall', 'must', 'will', 'required', 'should', 'may']
    text_lower = text.lower()
    for word in obligation_words:
        if word in text_lower:
            return True, word
    return False, "informational"
```

### Priority 3: Reduce MAX_SENTENCE_LENGTH (MEDIUM IMPACT)

**File:** `agents/enhanced_compliance/extractor.py`

```python
MAX_SENTENCE_LENGTH = 500  # Reduced from 1000
```

### Priority 4: Enhanced Noise Patterns (LOW IMPACT)

Add to NOISE_PATTERNS:
```python
r"^See\s+(?:Section|Attachment|Exhibit)",  # Cross-reference only
r"^Note:",  # Notes as headers
r"^(?:Refer|Reference)\s+to",  # Reference-only lines
```

---

## Accuracy Targets

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Quality Score | 62.5% | 95% | 32.5% |
| Multi-shall bundles | 104/matrix | 0 | HIGH |
| False positives | 24-30% | <5% | HIGH |
| Max extraction length | 1000 | 500 | MEDIUM |

---

## Next Steps

1. ~~**Implement multi-shall splitting**~~ - ✅ COMPLETED
2. ~~**Add obligation word validation**~~ - ✅ COMPLETED
3. ~~**Reduce max length**~~ - ✅ COMPLETED
4. ~~**Confidence scoring**~~ - ✅ COMPLETED (December 14, 2024)
5. **Manual spot-check** - Verify 50 random requirements against source RFPs
6. **Ground truth validation** - Create validation framework with human-verified RFPs
7. **Source traceability** - Link requirements to source page/paragraph

---

## Confidence Scoring (Implemented December 14, 2024)

Each extracted requirement now includes a confidence score to flag items needing human review:

| Confidence Level | Score Range | Criteria |
|-----------------|-------------|----------|
| **HIGH** | 0.85-1.0 | Strong obligation word + identified actor + section reference |
| **MEDIUM** | 0.60-0.84 | Has obligation word but missing some context |
| **LOW** | 0.0-0.59 | Inferred from evaluation criteria or uncertain |

### Scoring Factors

1. **Binding keyword presence** (+0.20 for SHALL/MUST, +0.15 for SHOULD, +0.10 for MAY)
2. **Actor identification** (+0.15 for "contractor", "offeror", "government", etc.)
3. **Section/RFP reference** (+0.10 if present)
4. **Cross-references** (+0.05 if found)
5. **Length appropriateness** (+0.05 for 50-400 chars, -0.05 for >450 chars)
6. **Compliance gate** (+0.05 for pass/fail requirements)
7. **Sentence structure** (+0.05 for complete sentences)

### Usage

```python
from agents.enhanced_compliance.section_aware_extractor import extract_requirements_structured

result = extract_requirements_structured(documents)

# Get stats
print(f"High confidence: {result.stats['by_confidence']['high_pct']}%")
print(f"Needs human review: {result.stats['needs_human_review']}")

# Filter low-confidence for QA
for req in result.all_requirements:
    if req.confidence_level.value == "Low":
        print(f"[LOW] {req.full_text[:80]}...")
        print(f"  Reasons: {req.confidence_reasons}")
```

---

## Ground Truth Validation Framework (Implemented December 14, 2024)

A framework for validating extraction accuracy against human-verified datasets.

### Metrics Calculated

| Metric | Description |
|--------|-------------|
| **Precision** | % of extracted requirements that are correct (TP / (TP + FP)) |
| **Recall** | % of ground truth requirements that were extracted (TP / (TP + FN)) |
| **F1 Score** | Harmonic mean of precision and recall |
| **Binding Level Accuracy** | % of requirements with correct binding level |
| **Confidence Correlation** | Precision for HIGH vs LOW confidence items |

### Usage

```python
from agents.enhanced_compliance.ground_truth_validator import (
    GroundTruthValidator, validate_extraction
)
from agents.enhanced_compliance.section_aware_extractor import extract_requirements_structured

# Run extraction
result = extract_requirements_structured(documents)

# Validate against ground truth
validation = validate_extraction(result, "ground_truth/rfp_123.json")

# View metrics
print(f"Precision: {validation.precision:.2%}")
print(f"Recall: {validation.recall:.2%}")
print(f"F1 Score: {validation.f1_score:.2%}")
print(f"Missed requirements: {len(validation.missed_requirements)}")

# Generate detailed report
validator = GroundTruthValidator()
report = validator.generate_report(validation)
print(report)
```

### Creating Ground Truth Datasets

1. **From extraction results** (recommended for efficiency):
```python
validator = GroundTruthValidator()
template = validator.create_ground_truth_template(
    extraction_result=result,
    rfp_id="75N96025R00004",
    rfp_title="NIH Research Support Services",
    reviewer="analyst@company.com"
)
validator.save_ground_truth(template, "ground_truth/nih_draft.json")
# Then manually review and verify each requirement
```

2. **Manual creation**: Copy `ground_truth/SAMPLE_ground_truth.json` and populate

### Ground Truth JSON Schema

```json
{
  "rfp_id": "string",
  "rfp_title": "string",
  "source_files": ["file1.pdf", "file2.pdf"],
  "verified_by": ["reviewer@email.com"],
  "requirements": [
    {
      "id": "GT-XXX-0001",
      "rfp_reference": "L.4.B.1",
      "full_text": "The Contractor shall...",
      "binding_level": "Mandatory|Highly Desirable|Desirable|Informational",
      "category": "L_COMPLIANCE|TECHNICAL|EVALUATION|ADMINISTRATIVE|ATTACHMENT",
      "source_page": 42,
      "verified_by": "reviewer@email.com",
      "verification_date": "2024-12-14",
      "notes": "Optional notes"
    }
  ]
}
```

---

## Appendix: Code References

- Extractor: `agents/enhanced_compliance/extractor.py:36-68` (quality tuning params)
- Semantic extractor: `agents/enhanced_compliance/semantic_extractor.py`
- CTM extractor: `agents/enhanced_compliance/ctm_extractor.py`
- Section-aware extractor: `agents/enhanced_compliance/section_aware_extractor.py`
- Ground truth validator: `agents/enhanced_compliance/ground_truth_validator.py`
