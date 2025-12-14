# PropelAI Strategic Fix Plan

## Expert Panel Analysis

### The Core Problem Statement

The same 11 documents (DOD NOC FA880625RB003) produce wildly different results:
- **Run 1**: 199 Section C requirements
- **Run 2**: 4 Section C requirements

This is not a typo detection bug. This is a **fundamental architectural fragility**.

---

## Root Cause Analysis (Expert Panel Synthesis)

### MIT Mathematician's Diagnosis

The extraction pipeline contains a **discontinuous function**:

```
f(documents) → requirements
where ∂f/∂input → ∞ at critical thresholds
```

Specifically, at `document_structure.py:401`:
```python
if best_score > -50:  # This threshold creates a cliff
```

A score of -49 produces ZERO Section C requirements.
A score of -51 produces ALL Section C requirements.

**This is mathematically unstable.** Small variations in document formatting, text extraction order, or regex matching cause the system to fall off the cliff.

### SaaS Architects' Diagnosis

1. **No observability**: When Section C detection fails, there's no logging or alert
2. **No fallback**: If scoring fails, the entire Section C is lost
3. **No validation**: The system doesn't verify that critical sections were found
4. **No idempotency**: Each run re-processes from scratch with potential variations

### AI/ML Engineers' Diagnosis

The current approach inverts the reliability pyramid:

```
WRONG:                      RIGHT:
  ┌──────────────┐            ┌──────────────┐
  │  Fuzzy Logic │            │   Fallback   │
  │ (fragile)    │            │   Defaults   │
  ├──────────────┤            ├──────────────┤
  │  Regex       │            │   Multi-Pass │
  │  Patterns    │            │   Validation │
  ├──────────────┤            ├──────────────┤
  │  Threshold   │            │   Robust     │
  │  Scoring     │            │   Detection  │
  └──────────────┘            └──────────────┘
    Base: Brittle               Base: Solid
```

### Government Proposal Executives' Diagnosis

**The business impact is catastrophic:**

| Missing Section | Impact |
|-----------------|--------|
| Section C (SOW) | Cannot scope the work → Cannot price → Cannot win |
| Section L | Don't know format requirements → Non-compliant → Rejected |
| Section M | Don't know evaluation criteria → Poor proposal focus → Lose |

Missing 95% of Section C requirements (4 vs 199) means **the tool is worse than useless** - it provides false confidence.

### Contrarian Thinker's Challenge

> "You're patching symptoms. The disease is your architecture."

The fundamental flaw: **Detection and extraction are conflated.**

Current flow:
```
Document → Detect Sections → Extract from Sections → Output
             ↑ FAIL HERE = Cascade failure
```

The system should be:
```
Document → Extract ALL Candidates → Classify Candidates → Validate → Output
                                          ↑ FAIL HERE = Graceful degradation
```

---

## Strategic Solution Architecture

### Principle 1: Defense in Depth

Never depend on a single detection mechanism. Layer defenses:

```
Layer 1: Filename patterns (fast, obvious cases)
Layer 2: Content header detection (regex for "STATEMENT OF WORK")
Layer 3: Structural patterns (section numbering like "1.0 SCOPE")
Layer 4: Keyword density analysis (high "shall/must" = requirements doc)
Layer 5: Fallback inclusion (unknown docs → manual review queue)
```

### Principle 2: Fail Safe, Not Fail Silent

```python
# WRONG (current):
if best_score > -50:
    include_section(section)
# If score is -51, section vanishes silently

# RIGHT:
if best_score > 50:
    include_section(section, confidence="HIGH")
elif best_score > -50:
    include_section(section, confidence="MEDIUM")
else:
    include_section(section, confidence="LOW")
    add_warning(f"Section {section} detected with low confidence. Manual review recommended.")
# Nothing vanishes. User is informed.
```

### Principle 3: Extract First, Classify Later

```python
# WRONG (current):
def extract():
    structure = analyze_rfp_structure(docs)  # Detection
    if SECTION_C in structure:               # Gate
        requirements = extract_from_section(SECTION_C)  # Extraction
    # If detection fails, extraction never happens

# RIGHT:
def extract():
    # Pass 1: Extract ALL potential requirements from ALL docs
    candidates = []
    for doc in docs:
        candidates.extend(extract_potential_requirements(doc))

    # Pass 2: Classify each candidate
    for req in candidates:
        req.section = classify_section(req)
        req.confidence = compute_confidence(req)

    # Pass 3: Validate and warn
    validate_coverage(candidates)  # Ensure no major sections missing
```

### Principle 4: Deterministic Reproducibility

```python
# Add to every extraction run:
extraction_metadata = {
    "version": "2.10",
    "document_hashes": [hash(doc) for doc in docs],
    "extraction_timestamp": datetime.utcnow(),
    "random_seed": 42,  # If any randomness used
}

# Cache results keyed by document hashes
cache_key = hash(tuple(sorted(document_hashes)))
if cache_key in extraction_cache:
    return extraction_cache[cache_key]
```

---

## Implementation Plan

### Phase 1: Immediate Stabilization (Critical Path)

**Goal**: Same inputs MUST produce same outputs.

| Task | File | Change |
|------|------|--------|
| 1.1 | `document_structure.py:401` | Remove hard threshold. Use graduated confidence levels. |
| 1.2 | `document_structure.py` | Add `detect_section_c_comprehensive()` with multiple detection strategies |
| 1.3 | `section_aware_extractor.py` | Add fallback: if Section C has <10 requirements, scan ALL docs for SOW content |
| 1.4 | `main.py` | Add validation: log warning if any UCF section has 0 requirements |

### Phase 2: Extract-First Architecture

**Goal**: Never lose requirements due to classification failures.

| Task | Description |
|------|-------------|
| 2.1 | Create `universal_extractor.py` that extracts ALL paragraphs with binding language |
| 2.2 | Create `section_classifier.py` that classifies extracted requirements post-hoc |
| 2.3 | Add confidence scores to every requirement |
| 2.4 | Create "low confidence" queue for manual review |

### Phase 3: Validation & Observability

**Goal**: Know when extraction quality is poor.

| Task | Description |
|------|-------------|
| 3.1 | Add extraction quality metrics (requirements per page, section coverage) |
| 3.2 | Add anomaly detection (if Section C < 5% of requirements, flag for review) |
| 3.3 | Add comparison mode (diff against previous extraction of same docs) |
| 3.4 | Add ground truth validation (compare against annotated test set) |

### Phase 4: Robustness Testing

**Goal**: Prove the system is stable.

| Task | Description |
|------|-------------|
| 4.1 | Run same documents 10 times, verify identical outputs |
| 4.2 | Permute document upload order, verify identical outputs |
| 4.3 | Introduce filename variations, verify stable detection |
| 4.4 | Test against ground truth, achieve >90% recall |

---

## Specific Code Changes Required

### Change 1: Comprehensive Section C Detection

**File**: `document_structure.py`

Replace the fragile scoring with multi-layer detection:

```python
def _detect_section_c_comprehensive(self, text: str, documents: List[Dict]) -> SectionCResult:
    """
    Multi-layer Section C detection. Never returns "not found" - returns confidence.
    """
    results = []

    # Layer 1: Explicit "SECTION C" header
    matches = re.finditer(r'SECTION\s+C[\s:\-–—]+', text, re.IGNORECASE)
    for m in matches:
        if not self._is_toc_entry(text, m.start()):
            results.append(SectionCCandidate(
                start=m.start(),
                method="explicit_header",
                confidence=0.9
            ))

    # Layer 2: SOW/PWS document in attachments
    for doc in documents:
        if self._is_sow_document(doc):
            results.append(SectionCCandidate(
                start=0,  # Entire document
                method="sow_attachment",
                confidence=0.95,
                source_document=doc['filename']
            ))

    # Layer 3: Content with "Statement of Work" header
    sow_header = re.search(r'STATEMENT\s+OF\s+WORK', text, re.IGNORECASE)
    if sow_header:
        results.append(SectionCCandidate(
            start=sow_header.start(),
            method="sow_header_in_text",
            confidence=0.85
        ))

    # Layer 4: High-density requirement paragraphs
    high_density_regions = self._find_requirement_dense_regions(text)
    for region in high_density_regions:
        results.append(SectionCCandidate(
            start=region.start,
            method="requirement_density",
            confidence=0.7
        ))

    # NEVER return empty - always return best candidate with confidence
    if not results:
        # Fallback: entire document as low-confidence Section C
        results.append(SectionCCandidate(
            start=0,
            method="fallback_entire_doc",
            confidence=0.3
        ))

    return max(results, key=lambda r: r.confidence)
```

### Change 2: Extraction Fallback

**File**: `section_aware_extractor.py`

Add fallback when Section C detection is weak:

```python
def extract(self, documents: List[Dict]) -> ExtractionResult:
    structure = analyze_rfp_structure(documents)
    result = ExtractionResult()

    # Extract from detected sections
    for section in structure.sections.values():
        reqs = self._extract_from_section(section)
        result.add_requirements(reqs)

    # CRITICAL: Validate Section C coverage
    section_c_count = len([r for r in result.all_requirements
                          if r.parent_section == UCFSection.SECTION_C])

    if section_c_count < 10:
        # Section C detection likely failed - use fallback
        self._log_warning(f"Only {section_c_count} Section C requirements found. "
                         "Running comprehensive SOW scan.")

        for doc in documents:
            if self._contains_sow_indicators(doc['text']):
                fallback_reqs = self._extract_all_requirements(doc['text'])
                for req in fallback_reqs:
                    req.parent_section = UCFSection.SECTION_C
                    req.detection_method = "fallback_sow_scan"
                    req.confidence = "MEDIUM"
                result.add_requirements(fallback_reqs)

    return result
```

### Change 3: Validation Checkpoint

**File**: `main.py`

Add extraction quality validation:

```python
def validate_extraction_quality(result: ExtractionResult, rfp_id: str):
    """Validate extraction didn't silently fail."""
    issues = []

    # Check Section C coverage
    section_c = [r for r in result.all_requirements
                 if r.parent_section == UCFSection.SECTION_C]
    if len(section_c) < 10:
        issues.append(f"LOW_SECTION_C: Only {len(section_c)} technical requirements found")

    # Check Section L coverage
    section_l = [r for r in result.all_requirements
                 if r.parent_section == UCFSection.SECTION_L]
    if len(section_l) < 5:
        issues.append(f"LOW_SECTION_L: Only {len(section_l)} proposal instructions found")

    # Check for SOW detection
    if not result.sow_detected:
        issues.append("NO_SOW: Statement of Work not detected in any document")

    if issues:
        logging.warning(f"RFP {rfp_id} extraction quality issues: {issues}")
        # Store issues for UI display
        rfp_store.set_quality_warnings(rfp_id, issues)

    return len(issues) == 0
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Reproducibility | Non-deterministic | 100% same output for same input |
| Section C Recall | 30.1% (31/103) | >90% |
| Silent Failures | Many | Zero (all failures logged with warnings) |
| Confidence Reporting | None | Every requirement has confidence score |
| Validation Coverage | None | Automated checks against ground truth |

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Stabilization | 2-3 days | Deterministic extraction, no silent failures |
| Phase 2: Extract-First | 1 week | New architecture with universal extraction |
| Phase 3: Observability | 3-4 days | Quality metrics and anomaly detection |
| Phase 4: Validation | 2-3 days | Proven stability with ground truth testing |

---

## Decision Required

Before proceeding, confirm:

1. **Scope**: Full architectural refactor (Phases 1-4) or immediate stabilization only (Phase 1)?
2. **Breaking Changes**: OK to change output format to include confidence scores?
3. **Ground Truth**: Use my 103-requirement annotation for DOD NOC as validation baseline?

