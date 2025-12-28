# PropelAI Master Architect Implementation Plan
## Based on s.rtf Blueprint & PEARL Framework

**Version:** 1.0
**Date:** 2025-12-28
**Status:** Approved for Implementation

---

## 1. Problem Statement

The current Annotated Outline Exporter produces **skeleton templates** rather than **RFP-specific blueprints**:

| Issue | Current Behavior | Required Behavior |
|-------|-----------------|-------------------|
| Volume Cloning | Vol 2, 3, 4 duplicate Vol 1 headers | Volume-specific templates (PP, Cost, Staffing) |
| Requirements | Placeholders "[Review Section L...]" | Actual extracted L/M/C requirements injected |
| Win Themes | "[Add differentiators to Company Library]" | Auto-generated from Company Library + RFP analysis |
| Page Allocations | Generic or missing | Calculated from Section M weights |
| Traceability | None | Every section linked to specific L/M/C refs |

---

## 2. Four-Phase Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: IRON TRIANGLE ANALYSIS                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ Section C   │ ↔ │ Section L   │ ↔ │ Section M   │               │
│  │ (What)      │   │ (How)       │   │ (Scoring)   │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         └─────────────────┼─────────────────┘                       │
│                           ▼                                         │
│              ┌─────────────────────────┐                            │
│              │   COMPLIANCE MATRIX     │                            │
│              │ ReqID | L Ref | M Ref   │                            │
│              └───────────┬─────────────┘                            │
└──────────────────────────┼──────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: ANNOTATED OUTLINE                        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ SmartOutlineGenerator.generate_volume_specific_structure()      ││
│  │  ├─ Volume 1: Technical (from M Factor analysis)                ││
│  │  ├─ Volume 2: Past Performance (PPQ template)                   ││
│  │  ├─ Volume 3: Cost/Price (CLIN/Labor Rate template)             ││
│  │  └─ Volume 4: Staffing (Org Chart/Resume template)              ││
│  └─────────────────────────────────────────────────────────────────┘│
│                           │                                          │
│                           ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ RequirementInjector.inject_requirements_by_section()            ││
│  │  ├─ Match L reqs to outline sections                            ││
│  │  ├─ Match M factors to section annotations                      ││
│  │  └─ Match C tasks to proof point placeholders                   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: F-B-P DRAFTING                          │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ DraftingAgent.generate_section_content()                        ││
│  │                                                                  ││
│  │  For each section:                                               ││
│  │  1. Query Company Library for relevant capabilities              ││
│  │  2. Structure as Feature → Benefit → Proof                       ││
│  │  3. Insert [PLACEHOLDER] for missing proofs (no hallucination)  ││
│  │  4. Apply ghosting language from CompetitorAnalyzer              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: RED TEAM REVIEW                         │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ RedTeamAgent.evaluate_section()                                  ││
│  │  ├─ Score: Blue (Outstanding) / Green (Good) / Yellow (Marginal)││
│  │  ├─ Identify missing proof points                                ││
│  │  ├─ Flag unsupported claims                                      ││
│  │  └─ Suggest rewrites for "Marginal" → "Outstanding"              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Tasks

### Task 1: Fix Volume Cloning Issue
**File:** `agents/enhanced_compliance/smart_outline_generator.py`

```python
# NEW: Volume-specific templates
VOLUME_TEMPLATES = {
    "technical": {
        "sections": ["Executive Summary", "Technical Approach", "Management Approach"],
        "default_allocation": 0.6  # 60% of page budget
    },
    "past_performance": {
        "sections": ["Relevance Summary", "Contract 1", "Contract 2", "Contract 3"],
        "default_allocation": 0.15
    },
    "cost_price": {
        "sections": ["Cost Narrative", "Basis of Estimate", "CLIN Summary"],
        "default_allocation": 0.15
    },
    "staffing": {
        "sections": ["Organizational Structure", "Key Personnel", "Staffing Plan"],
        "default_allocation": 0.10
    }
}
```

**Changes Required:**
1. Add `VOLUME_TEMPLATES` constant
2. Modify `_create_default_volumes()` to use volume-specific templates
3. Add `_detect_volume_type_from_name()` method
4. Remove clone logic that copies Vol 1 structure to other volumes

### Task 2: Implement Requirement Injection
**File:** `agents/enhanced_compliance/requirement_injector.py` (NEW)

```python
class RequirementInjector:
    """
    Injects extracted L/M/C requirements into outline sections.
    Replaces placeholders with actual RFP text.
    """

    def inject_requirements(
        self,
        outline: ProposalOutline,
        requirements: List[Dict],
        compliance_matrix: List[Dict]
    ) -> ProposalOutline:
        """
        Match requirements to sections based on:
        1. Section name keywords
        2. Section L references
        3. Evaluation factor mappings
        """
        for volume in outline.volumes:
            for section in volume.sections:
                # Find matching requirements
                section.requirements = self._find_matching_requirements(
                    section, requirements, compliance_matrix
                )
                # Find matching eval criteria
                section.eval_criteria = self._find_matching_eval_criteria(
                    section, requirements
                )
        return outline
```

### Task 3: Integrate Company Library for Win Themes
**File:** `agents/enhanced_compliance/smart_outline_generator.py`

```python
def _generate_win_themes(
    self,
    section: ProposalSection,
    company_library_data: Dict,
    evaluation_factors: List[EvaluationFactor]
) -> List[str]:
    """
    Generate win themes by matching:
    - Company capabilities to evaluation factors
    - Past performance to section requirements
    - Differentiators to hot buttons
    """
    themes = []

    # Match capabilities to this section's requirements
    for cap in company_library_data.get("capabilities", []):
        if self._capability_matches_section(cap, section):
            themes.append(f"Our {cap['name']} delivers {cap['description']}")

    # If no matches, return actionable placeholder
    if not themes:
        themes.append("[WIN THEME: Query Company Library for relevant differentiators]")

    return themes
```

### Task 4: Calculate Page Allocations from Section M Weights
**File:** `agents/enhanced_compliance/smart_outline_generator.py`

```python
def _calculate_page_allocation(
    self,
    section: ProposalSection,
    eval_factors: List[EvaluationFactor],
    total_pages: int
) -> int:
    """
    Allocate pages based on Section M evaluation weights.

    Formula: section_pages = total_pages * (factor_weight / total_weight)
    """
    # Find matching evaluation factor
    matching_factor = None
    for factor in eval_factors:
        if self._factor_matches_section(factor, section):
            matching_factor = factor
            break

    if matching_factor and matching_factor.weight:
        # Parse weight (e.g., "Most Important" = 0.35, "Important" = 0.25)
        weight = self._parse_importance_weight(matching_factor.weight)
        return max(1, int(total_pages * weight))

    # Default: equal distribution
    return max(1, total_pages // len(sections))
```

### Task 5: Add Traceability IDs to Annotations
**File:** `agents/enhanced_compliance/annotated_outline_exporter.js`

```javascript
// NEW: Add traceability annotations
function buildRequirementAnnotation(req, sectionId) {
    return new Paragraph({
        children: [
            new TextRun({
                text: `[${req.id}] `,
                bold: true,
                color: COLORS.SECTION_L
            }),
            new TextRun({ text: req.text }),
            new TextRun({
                text: ` → Addresses M.${req.evalFactorRef}`,
                italic: true,
                color: COLORS.SECTION_M
            })
        ]
    });
}
```

### Task 6: API Endpoint for 4-Phase Workflow
**File:** `api/main.py`

```python
@app.post("/api/rfp/{rfp_id}/master-architect")
async def run_master_architect_workflow(
    rfp_id: str,
    phase: Optional[int] = None  # None = run all, 1-4 = specific phase
):
    """
    Execute the Master Architect 4-phase workflow:

    Phase 1: Iron Triangle Analysis → Compliance Matrix
    Phase 2: Generate RFP-Specific Annotated Outline
    Phase 3: F-B-P Drafting (requires Company Library)
    Phase 4: Red Team Review & Scoring

    Returns phase results with option to pause/resume between phases.
    """
```

---

## 4. File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `smart_outline_generator.py` | MODIFY | Add volume templates, win theme generation, page allocation |
| `requirement_injector.py` | CREATE | New class for L/M/C injection |
| `annotated_outline_exporter.js` | MODIFY | Add traceability annotations |
| `annotated_outline_exporter.py` | MODIFY | Pass compliance matrix to JS |
| `api/main.py` | MODIFY | Add `/master-architect` endpoint |

---

## 5. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Volume-specific structure | ❌ Cloned | ✅ Unique per volume type |
| Requirements injected | ❌ Placeholders | ✅ Actual L/M/C text |
| Win themes | ❌ Generic | ✅ From Company Library |
| Page allocations | ❌ Missing | ✅ Based on M weights |
| Traceability | ❌ None | ✅ ReqID → L Ref → M Ref |
| Red Team scoring | ❌ None | ✅ Blue/Green/Yellow ratings |

---

## 6. Estimated Effort

| Phase | Task | Complexity | Priority |
|-------|------|------------|----------|
| 1 | Fix Volume Cloning | Medium | P0 - Critical |
| 1 | Requirement Injection | Medium | P0 - Critical |
| 2 | Win Theme Generation | Low | P1 - High |
| 2 | Page Allocation | Low | P1 - High |
| 3 | Traceability Annotations | Low | P2 - Medium |
| 4 | Master Architect API | Medium | P2 - Medium |

---

## 7. Next Steps

1. **Immediate:** Fix volume cloning in `smart_outline_generator.py`
2. **Immediate:** Create `requirement_injector.py`
3. **Short-term:** Integrate Company Library for win themes
4. **Short-term:** Add page allocation calculation
5. **Medium-term:** Build `/master-architect` orchestration endpoint
