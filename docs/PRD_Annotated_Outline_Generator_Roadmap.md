# Product Requirements Document: PropelAI Annotated Outline Generator Roadmap

**Document Version:** 1.0
**Date:** December 14, 2024
**Author:** Claude Code / PropelAI Engineering
**Status:** Draft for Review

---

## 1. Executive Summary

This PRD outlines the future roadmap for the PropelAI Annotated Outline Generator, building on the v2.12 implementation completed in December 2024. The roadmap prioritizes features that will increase win rates, reduce proposal development time, and improve compliance accuracy.

### Vision Statement
Transform the Annotated Outline Generator from a document creation tool into an intelligent proposal development platform that guides users from RFP analysis through compliant, compelling proposal content.

---

## 2. Current State Assessment

### 2.1 Capabilities Delivered (v2.12)
- Volume/section structure creation from RFP analysis
- Section L/M/C content population via semantic matching
- P0 constraint extraction and display
- Binding level detection for requirements
- Evaluation factor weighting and adjectival rating extraction
- Word document export with color-coded content

### 2.2 Known Gaps
| Gap | Impact | Priority |
|-----|--------|----------|
| Solicitation number extraction unreliable | Manual entry required | Medium |
| Due date parsing incomplete | May miss non-standard formats | Medium |
| Duplicate content across factors | Requires manual cleanup | High |
| Page limit to volume mapping incomplete | Incorrect allocations | High |
| No cross-reference linkage display | Evaluator compliance unclear | Medium |

---

## 3. Roadmap Phases

## Phase 1: Foundation Improvements (Q1 2025)

### 3.1 Enhanced RFP Metadata Extraction

**Feature:** Intelligent Solicitation Metadata Parser

**Problem:** Solicitation numbers, due dates, and agency information are not reliably extracted from all RFP formats.

**Requirements:**
- P1.1: Extract solicitation number from standard locations (cover page, header, SF-33)
- P1.2: Parse due dates in multiple formats (MM/DD/YYYY, Month DD, YYYY, ISO 8601)
- P1.3: Identify contracting office and agency from document metadata
- P1.4: Extract set-aside type (8(a), HUBZone, WOSB, SDVOSB, Full & Open)
- P1.5: Detect contract type (FFP, T&M, CPFF, IDIQ task order)

**Acceptance Criteria:**
- 95% accuracy on solicitation number extraction for UCF format RFPs
- 90% accuracy on due date parsing
- Fallback to manual entry with clear prompts when confidence is low

---

### 3.2 Improved Content Deduplication

**Feature:** Semantic Deduplication Engine

**Problem:** Section L/M/C content may appear in multiple factors when semantic matching isn't sufficiently differentiated, requiring manual cleanup.

**Requirements:**
- P1.6: Implement TF-IDF scoring for requirement-to-factor relevance
- P1.7: Track assigned requirements to prevent duplicate assignment
- P1.8: Use factor hierarchy to resolve assignment conflicts
- P1.9: Provide confidence scores for each requirement mapping
- P1.10: Allow manual override with persistence

**Acceptance Criteria:**
- Less than 5% content duplication across factors
- Clear audit trail of assignment decisions
- One-click reassignment capability

---

### 3.3 Page Allocation Intelligence

**Feature:** Smart Page Budget Calculator

**Problem:** Page limits are extracted but not correctly mapped to specific volumes and sections.

**Requirements:**
- P1.11: Parse page limits from Section L volume descriptions
- P1.12: Map page limits to corresponding volumes by name matching
- P1.13: Calculate section-level page recommendations based on evaluation weights
- P1.14: Display page budget warnings when allocations exceed limits
- P1.15: Support "excluding" clauses (e.g., "25 pages excluding resumes")

**Acceptance Criteria:**
- Correct volume-level page limits for 90% of standard UCF RFPs
- Section recommendations sum to volume limit with 10% margin
- Visual indicators for over/under allocation

---

## Phase 2: Compliance Intelligence (Q2 2025)

### 3.4 Compliance Requirement Traceability

**Feature:** Bidirectional Requirement Tracing

**Problem:** No clear linkage between proposal sections and source requirements for evaluator compliance verification.

**Requirements:**
- P2.1: Assign unique IDs to all extracted requirements
- P2.2: Generate cross-reference matrix (requirement → proposal section)
- P2.3: Generate reverse matrix (proposal section → requirements addressed)
- P2.4: Calculate compliance coverage percentage per section
- P2.5: Identify orphan requirements (extracted but not mapped)
- P2.6: Identify thin sections (mapped but low requirement coverage)

**Acceptance Criteria:**
- 100% of mandatory requirements tracked
- Compliance matrix exportable to Excel
- Real-time coverage dashboard in UI

---

### 3.5 Automated Compliance Verification

**Feature:** Pre-Submission Compliance Checker

**Problem:** P0 constraints and formatting requirements are displayed but not automatically verified against proposal content.

**Requirements:**
- P2.7: Parse submitted proposal document
- P2.8: Verify page counts against limits
- P2.9: Verify font and margin compliance via document inspection
- P2.10: Check for required section headings
- P2.11: Validate file format requirements
- P2.12: Generate compliance checklist with pass/fail/warning status

**Acceptance Criteria:**
- Detects 95% of P0 violations before submission
- Clear remediation guidance for each violation
- Exportable compliance report for capture lead review

---

### 3.6 Enhanced Section M Analysis

**Feature:** Evaluation Strategy Advisor

**Problem:** Evaluation factor importance and adjectival rating definitions are extracted but not synthesized into actionable strategy guidance.

**Requirements:**
- P2.13: Calculate relative importance scores from textual descriptions
- P2.14: Map adjectival ratings to specific evidence requirements
- P2.15: Generate "What Evaluators Want to See" guidance per factor
- P2.16: Identify discriminator opportunities based on evaluation weights
- P2.17: Suggest proof point types for each rating level

**Acceptance Criteria:**
- Importance ranking matches human BD expert assessment 85% of time
- At least 3 actionable suggestions per evaluation factor
- Guidance tied to specific RFP language (with citations)

---

## Phase 3: Content Development Assistance (Q3 2025)

### 3.7 Win Theme Integration

**Feature:** Win Theme Management System

**Problem:** Win themes are placeholders in the outline with no connection to capture intelligence or differentiation strategy.

**Requirements:**
- P3.1: Accept win themes from capture plan import
- P3.2: Map win themes to relevant proposal sections
- P3.3: Generate theme reinforcement suggestions per section
- P3.4: Track theme coverage across proposal
- P3.5: Ensure each volume has at least one primary theme

**Acceptance Criteria:**
- Win themes visible in each relevant section
- Coverage report shows theme distribution
- No volume without assigned themes

---

### 3.8 Proof Point Library Integration

**Feature:** Evidence Repository Connection

**Problem:** Proof points are generic placeholders rather than company-specific evidence.

**Requirements:**
- P3.6: Connect to corporate past performance database
- P3.7: Match past performance to current requirement areas
- P3.8: Suggest relevant metrics and quantified achievements
- P3.9: Track proof point usage across proposal
- P3.10: Prevent proof point overuse (same stat in multiple sections)

**Acceptance Criteria:**
- At least 2 relevant proof point suggestions per technical section
- Proof points include quantified metrics
- Usage tracking prevents redundancy

---

### 3.9 Graphics Planning Assistant

**Feature:** Visual Communication Planner

**Problem:** Graphics placeholders are generic without guidance on what visuals would be most effective.

**Requirements:**
- P3.11: Analyze section content for visualization opportunities
- P3.12: Suggest graphic types (process flow, org chart, comparison table, etc.)
- P3.13: Generate action caption templates based on section requirements
- P3.14: Calculate page impact of graphics
- P3.15: Provide graphic complexity estimates for production planning

**Acceptance Criteria:**
- At least 1 graphic suggestion per major section
- Action caption follows "statement + so what + proof" format
- Page estimates accurate to ±0.25 pages

---

## Phase 4: AI-Assisted Content Generation (Q4 2025)

### 3.10 Intelligent Section Drafting

**Feature:** AI Proposal Writing Assistant

**Problem:** Annotated outline provides structure but actual content writing is manual.

**Requirements:**
- P4.1: Generate compliant section drafts based on L/M/C mapping
- P4.2: Incorporate win themes into generated content
- P4.3: Embed proof points naturally into narrative
- P4.4: Match client terminology and writing style
- P4.5: Provide multiple draft variations for author selection
- P4.6: Track compliance coverage in generated content

**Acceptance Criteria:**
- Generated drafts address 90% of section requirements
- Content passes plagiarism check
- Authors rate drafts as "useful starting point" 80% of time

---

### 3.11 Compliance-Aware Editing

**Feature:** Real-Time Compliance Editor

**Problem:** Authors may introduce compliance issues while editing generated or manual content.

**Requirements:**
- P4.7: Monitor content edits for compliance impact
- P4.8: Alert when requirement coverage drops below threshold
- P4.9: Suggest language to restore compliance
- P4.10: Track "shall" statement response completeness
- P4.11: Validate page count during editing

**Acceptance Criteria:**
- Real-time alerts within 2 seconds of edit
- Compliance suggestions restore coverage 90% of time
- No false positive alerts in normal editing

---

### 3.12 Executive Summary Generator

**Feature:** Automated Executive Summary

**Problem:** Executive summaries require synthesizing entire proposal into compelling narrative.

**Requirements:**
- P4.12: Extract key themes from all volumes
- P4.13: Synthesize discriminators and proof points
- P4.14: Generate summary aligned with evaluation factors
- P4.15: Ensure summary fits page allocation
- P4.16: Provide multiple versions (technical, executive, client-focused)

**Acceptance Criteria:**
- Summary covers all evaluation factors
- Fits within page limit
- Capture lead rates as "submission ready" 70% of time

---

## Phase 5: Enterprise Capabilities (2026+)

### 3.13 Multi-User Collaboration

**Feature:** Team Proposal Workspace

**Requirements:**
- P5.1: Real-time collaborative editing
- P5.2: Section assignment and ownership tracking
- P5.3: Review and approval workflows
- P5.4: Version control with diff view
- P5.5: Comment and annotation system

---

### 3.14 Historical Analytics

**Feature:** Win/Loss Intelligence

**Requirements:**
- P5.6: Track proposal outcomes by approach patterns
- P5.7: Identify winning discriminators and themes
- P5.8: Benchmark section strategies against win rates
- P5.9: Agency-specific preference learning
- P5.10: Competitor analysis integration

---

### 3.15 Multi-Format Support

**Feature:** Extended RFP Format Handling

**Requirements:**
- P5.11: State and local government formats
- P5.12: Commercial RFP formats
- P5.13: International procurement standards
- P5.14: Non-PDF source document handling
- P5.15: OCR enhancement for scanned documents

---

## 4. Technical Architecture Evolution

### 4.1 Current Architecture
```
PDF → Python Extractor → Smart Outline Generator → JSON → Node.js Exporter → DOCX
```

### 4.2 Target Architecture (Phase 4+)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PropelAI Platform                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Document     │   │ NLP Engine   │   │ Knowledge    │                │
│  │ Intelligence │◄──┤ (LLM-based)  │◄──┤ Graph        │                │
│  │ Service      │   │              │   │ (Past Perf)  │                │
│  └──────┬───────┘   └──────────────┘   └──────────────┘                │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────┐              │
│  │              Compliance Intelligence Core             │              │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │              │
│  │  │ Extractor  │  │ Matcher    │  │ Validator  │     │              │
│  │  │ Pipeline   │  │ Pipeline   │  │ Pipeline   │     │              │
│  │  └────────────┘  └────────────┘  └────────────┘     │              │
│  └──────────────────────────────────────────────────────┘              │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────┐              │
│  │              Content Generation Engine                │              │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │              │
│  │  │ Outline    │  │ Draft      │  │ Review     │     │              │
│  │  │ Builder    │  │ Generator  │  │ Assistant  │     │              │
│  │  └────────────┘  └────────────┘  └────────────┘     │              │
│  └──────────────────────────────────────────────────────┘              │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────────────┐              │
│  │              Output Services                          │              │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │              │
│  │  │ DOCX       │  │ PDF        │  │ Web Editor │     │              │
│  │  │ Exporter   │  │ Generator  │  │ (React)    │     │              │
│  │  └────────────┘  └────────────┘  └────────────┘     │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Key Technical Investments

| Phase | Investment Area | Purpose |
|-------|----------------|---------|
| 1 | Enhanced NLP pipeline | Better metadata extraction |
| 2 | Graph database | Requirement traceability |
| 3 | Knowledge base integration | Proof point and past performance |
| 4 | LLM integration | Content generation |
| 5 | Real-time collaboration | Multi-user workspace |

---

## 5. Success Metrics

### 5.1 Phase 1 Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Metadata extraction accuracy | 60% | 95% |
| Content duplication rate | 20% | <5% |
| Page allocation accuracy | 50% | 90% |

### 5.2 Phase 2 Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Compliance coverage visibility | Manual | 100% automated |
| P0 violation detection | Display only | 95% auto-detect |
| Evaluator guidance generation | None | 3+ per factor |

### 5.3 Phase 3-4 Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Time to first draft | Manual | 50% reduction |
| Win theme coverage | Manual tracking | Automated 100% |
| Author satisfaction | N/A | 80% "useful" rating |

### 5.4 Business Impact Metrics
| Metric | Target |
|--------|--------|
| Proposal development time reduction | 40% |
| Compliance issue reduction | 75% |
| Win rate improvement | 15% increase |

---

## 6. Dependencies and Risks

### 6.1 Dependencies
| Dependency | Phase | Mitigation |
|------------|-------|------------|
| LLM API availability | 4 | Multi-provider support |
| Past performance database | 3 | Manual entry fallback |
| User adoption | All | Training program |
| Document parsing accuracy | 1 | OCR enhancement |

### 6.2 Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM hallucination in content | Medium | High | Human review gates |
| Performance at scale | Medium | Medium | Caching, async processing |
| RFP format variability | High | Medium | Continuous pattern updates |
| User resistance to AI content | Medium | Medium | Transparency, control |

---

## 7. Stakeholder Requirements

### 7.1 Capture Managers
- Clear compliance coverage visibility
- Win theme integration
- Page allocation recommendations

### 7.2 Proposal Writers
- Structured starting points
- Requirement context in-line
- Easy section navigation

### 7.3 Volume Leads
- Cross-section consistency tools
- Page budget management
- Review workflows

### 7.4 Quality Reviewers
- Compliance verification tools
- Traceability matrices
- Annotation capabilities

---

## 8. Implementation Priorities

### 8.1 Immediate (Next 30 Days)
1. Fix remaining metadata extraction edge cases
2. Implement content deduplication
3. Complete page allocation mapping

### 8.2 Short-Term (60-90 Days)
1. Bidirectional requirement tracing
2. Compliance coverage dashboard
3. P0 violation pre-check

### 8.3 Medium-Term (Q2-Q3)
1. Win theme management
2. Proof point integration
3. Graphics planning

### 8.4 Long-Term (Q4+)
1. AI content generation
2. Collaborative editing
3. Analytics and learning

---

## 9. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| Annotated Outline | Master template for proposal development with color-coded requirements |
| P0 Constraint | Pass/fail requirement causing disqualification if violated |
| UCF | Uniform Contract Format - standard federal RFP structure |
| Iron Triangle | Section L (instructions), M (evaluation), C (requirements) mapping |
| Adjectival Rating | Qualitative evaluation scale (Exceptional, Very Good, etc.) |
| Discriminator | Differentiating feature that provides competitive advantage |
| Proof Point | Quantified evidence supporting capability claims |

### Appendix B: Reference Documents

- AS_BUILT_Annotated_Outline_Generator.md - Current implementation details
- outline.rtf - Original requirements specification
- OASIS+ ordering procedures - Task order format reference

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-14 | Claude Code | Initial PRD roadmap |

---

*Document Status: Ready for stakeholder review*
