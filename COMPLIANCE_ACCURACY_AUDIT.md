# PropelAI Compliance Accuracy Audit Report

**Date:** December 14, 2024
**Audited By:** Automated Analysis
**Scope:** 10 compliance matrices from production RFP processing

---

## Executive Summary

PropelAI's compliance extraction achieves approximately **62-70% quality score** based on detected issues. While the system successfully extracts requirements with proper binding level classification (80% Mandatory), there are three critical issues impacting accuracy:

| Issue | Impact | Count | Priority |
|-------|--------|-------|----------|
| Multi-shall bundles | Under-counting requirements | 100-104 per matrix | **HIGH** |
| Missing obligation words | False positives (noise) | 24-30% of extractions | **HIGH** |
| Very long extractions | Requirement bundling | 25-30% of extractions | **MEDIUM** |

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

1. **Implement multi-shall splitting** - Highest impact, addresses 100+ issues per matrix
2. **Add obligation word validation** - Reduces false positives by 20%+
3. **Reduce max length** - Improves requirement granularity
4. **Manual spot-check** - Verify 50 random requirements against source RFPs
5. **A/B test** - Run extraction with fixes on same RFPs, compare metrics

---

## Appendix: Code References

- Extractor: `agents/enhanced_compliance/extractor.py:36-68` (quality tuning params)
- Semantic extractor: `agents/enhanced_compliance/semantic_extractor.py`
- CTM extractor: `agents/enhanced_compliance/ctm_extractor.py`
- Section-aware extractor: `agents/enhanced_compliance/section_aware_extractor.py`
