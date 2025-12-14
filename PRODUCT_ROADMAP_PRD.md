# PropelAI Product Requirements Document (PRD)
# Strategic Roadmap & Future Development

**Document Version:** 1.0
**Last Updated:** December 14, 2024
**Author:** Product Team
**Status:** Strategic Planning

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Market Context](#2-market-context)
3. [User Personas](#3-user-personas)
4. [Current State Assessment](#4-current-state-assessment)
5. [Strategic Priorities](#5-strategic-priorities)
6. [Phase 2: Trust Foundation](#6-phase-2-trust-foundation)
7. [Phase 3: Color Team Workflows](#7-phase-3-color-team-workflows)
8. [Phase 4: Red Team Simulation](#8-phase-4-red-team-simulation)
9. [Phase 5: Full Proposal Generation](#9-phase-5-full-proposal-generation)
10. [Integration Requirements](#10-integration-requirements)
11. [Technical Architecture Evolution](#11-technical-architecture-evolution)
12. [Go-to-Market Strategy](#12-go-to-market-strategy)
13. [Success Metrics](#13-success-metrics)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Appendix: Feature Specifications](#15-appendix-feature-specifications)

---

## 1. Executive Summary

### 1.1 Vision

PropelAI will become the **Autonomous Proposal Operating System** for government contractors—transforming proposal development from a labor-intensive craft into a systematic, AI-augmented process that produces higher win rates with lower effort.

### 1.2 Mission

Reduce the time and cost of government proposal development by 70% while improving compliance accuracy to 99%+ and enabling capture teams to pursue more opportunities with existing headcount.

### 1.3 Phase 1 Achievements

| Capability | Status | Accuracy |
|------------|--------|----------|
| RFP Document Parsing | ✅ Complete | 95%+ |
| Section L/M/C Extraction | ✅ Complete | 85%+ |
| Compliance Matrix Generation | ✅ Complete | 90%+ |
| Annotated Outline Generation | ✅ Complete | 85%+ |
| Multi-model Integration (Claude/Gemini/GPT-4) | ✅ Architecture | Ready |
| Amendment Processing | ✅ Complete | 80%+ |

### 1.4 Strategic Imperative

**The trust gate must be passed before feature expansion:**

> "Compliance MUST be deterministic—100% accuracy required before teams trust anything else."
> — Compass Market Research

A single missed requirement disqualifies proposals. PropelAI must prove compliance accuracy BEFORE users will trust drafting, strategy, or other AI-assisted features.

### 1.5 Competitive Window

The market is consolidating rapidly:
- **Vultron:** $27M raised, 72% drafting time reduction claimed
- **Deltek Dela:** Launching November 2025
- **Unanet:** Acquired GovPro AI (November 2024)

**PropelAI must establish market position within 12-18 months.**

---

## 2. Market Context

### 2.1 Addressable Market

| Metric | Value | Source |
|--------|-------|--------|
| Total Addressable Market (TAM) | $5.6B+ | Compass Research |
| Serviceable Addressable Market (SAM) | $2.5-3.8B | AI proposal tools |
| CAGR | 8-14% | Industry growth |
| Federal Contract Spending (2027 proj.) | $7.5T | Federal market |

### 2.2 Market Adoption Trends

| Metric | Current | Growth |
|--------|---------|--------|
| GovCon firms using AI | 33% | +20% YoY |
| Firms planning AI expansion | 59% | Next 12 months |
| AI-reported time savings | 70-92% | First draft generation |
| Opportunity increase | +30% | Without headcount increase |

### 2.3 Competitive Landscape

| Competitor | Positioning | Funding | Key Differentiator |
|------------|-------------|---------|-------------------|
| **Vultron** | AI-native proposal platform | $27M | Speed, end-to-end |
| **Deltek GovWin IQ** | Market intelligence + AI | Established | Data, relationships |
| **GovDash** | Compliance automation | Series A | Compliance focus |
| **Unanet (GovPro AI)** | ERP-integrated | Acquired | Financial integration |
| **VisibleThread** | Content analytics | Established | Readability, compliance |

### 2.4 PropelAI Differentiation Opportunity

| Opportunity | Description | Status |
|-------------|-------------|--------|
| **Compliance Accuracy** | 99%+ deterministic extraction | Building |
| **Source Traceability** | Every claim linked to source | Planned |
| **Color Team Integration** | AI-augmented review workflows | Planned |
| **Red Team Simulation** | SSEB evaluation modeling | Planned |
| **Open Architecture** | Multi-model, not vendor locked | Complete |

---

## 3. User Personas

### 3.1 Primary Personas

#### Capture Manager
| Attribute | Detail |
|-----------|--------|
| **Role** | Strategy architect, customer relationship owner |
| **Goals** | Win opportunities, build relationships, develop strategy |
| **Pain Points** | Insufficient intel, partner friction, rapid turnover |
| **AI Opportunities** | Competitor monitoring, capture plan generation, PWin modeling |
| **Non-Negotiable** | Human control of strategic decisions |
| **Tenure** | 6-9 months (declining) |

#### Proposal Manager
| Attribute | Detail |
|-----------|--------|
| **Role** | Central coordinator, compliance owner |
| **Goals** | 100% compliant, on-time submissions |
| **Pain Points** | SME unavailability, last-minute amendments, compliance gaps |
| **AI Opportunities** | Auto CTM generation, compliance alerts, schedule management |
| **Non-Negotiable** | Explainable AI with audit trails |
| **Primary Metric** | Compliance score (must be 100%) |

#### Technical Volume Lead
| Attribute | Detail |
|-----------|--------|
| **Role** | Solution architect, SME coordinator |
| **Goals** | Technical excellence, solution differentiation |
| **Pain Points** | Technical inaccuracies, cost-technical misalignment |
| **AI Opportunities** | Content generation, consistency checking, graphics |
| **Non-Negotiable** | SME validation of all technical claims |
| **Primary Metric** | Technical score, discriminator strength |

#### Subject Matter Expert (SME)
| Attribute | Detail |
|-----------|--------|
| **Role** | Domain expert, content contributor |
| **Goals** | Contribute expertise efficiently, return to operations |
| **Pain Points** | Time away from primary job, blank-page writing |
| **AI Opportunities** | Talk-to-proposal, knowledge capture, draft generation |
| **Non-Negotiable** | Accurate representation of their expertise |
| **Time Available** | 20% of work time searching for information |

#### Compliance Analyst
| Attribute | Detail |
|-----------|--------|
| **Role** | Gatekeeper, requirement tracker |
| **Goals** | Zero compliance failures, requirement coverage |
| **Pain Points** | Manual tracking, format violations, missed requirements |
| **AI Opportunities** | Auto-extraction, real-time monitoring, format validation |
| **Non-Negotiable** | Deterministic accuracy (100% required) |
| **Primary Metric** | Zero rejected submissions |

#### Executive Reviewer
| Attribute | Detail |
|-----------|--------|
| **Role** | Final authority, strategic alignment |
| **Goals** | Quality assurance, corporate risk management |
| **Pain Points** | Limited review time, missing competitive context |
| **AI Opportunities** | Document summarization, competitive intel synthesis |
| **Non-Negotiable** | Confidence in AI-assisted content quality |
| **Review Time** | Gold Team: 2-4 hours typical |

### 3.2 Persona Trust Hierarchy

```
                    TRUST EARNED PROGRESSIVELY

Level 1: COMPLIANCE (Must be 100% first)
    ├── Requirement extraction
    ├── Compliance matrix accuracy
    └── Format verification
         │
         ▼
Level 2: RESEARCH & ANALYSIS
    ├── Opportunity identification
    ├── Competitor analysis
    └── Historical data synthesis
         │
         ▼
Level 3: FIRST DRAFT GENERATION
    ├── Boilerplate sections
    ├── Past performance summaries
    └── Standard responses
         │
         ▼
Level 4: STRATEGIC ASSISTANCE
    ├── Win theme suggestions
    ├── Discriminator identification
    └── PWin modeling
         │
         ▼
Level 5: REVIEW & SCORING (Never autonomous)
    ├── AI-assisted quality review
    ├── Competitive scoring simulation
    └── Final human approval ALWAYS
```

---

## 4. Current State Assessment

### 4.1 Phase 1 Capabilities

| Feature | Maturity | Notes |
|---------|----------|-------|
| PDF/DOCX/XLSX Parsing | Production | 95%+ accuracy |
| Section L/M/C Detection | Production | 85%+ accuracy |
| Requirement Extraction | Production | 43% need section context improvement |
| Binding Level Classification | Production | Mandatory/Should/May detection |
| CTM Generation | Production | Multi-sheet, evaluator-ready |
| Annotated Outline | Production | Requirements mapped to sections |
| Amendment Processing | Beta | Change tracking functional |
| Agency Detection | Production | NIH, DoD, VA, GSA, DHS |

### 4.2 Known Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| Section detection ~43% "UNK" | Missing context for requirements | P1 |
| No source traceability | Can't verify AI claims | P1 |
| No real-time gap analysis | Static compliance only | P2 |
| No color team integration | Workflow disconnect | P2 |
| No persistent storage | Data lost on restart | P2 |
| No user authentication | Single-user only | P2 |
| No Microsoft integration | Adoption friction | P3 |

### 4.3 Technical Debt

| Item | Impact | Effort to Fix |
|------|--------|---------------|
| In-memory storage | No persistence | Medium |
| Single-threaded API | No concurrency | Medium |
| No test coverage for edge cases | Quality risk | Medium |
| Hardcoded limits | Inflexible | Low |
| No logging/monitoring | Operational blindness | Low |

---

## 5. Strategic Priorities

### 5.1 Priority Framework

Based on Compass research, priorities must follow the **Trust Hierarchy**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIORITY 1: TRUST FOUNDATION                  │
│                                                                  │
│  "A single missed requirement disqualifies proposals"            │
│                                                                  │
│  • Compliance accuracy audit (validate 99%+ extraction)          │
│  • Source traceability (every claim → source)                    │
│  • Audit trail (AI-generated vs human-written)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRIORITY 2: WORKFLOW VALUE                    │
│                                                                  │
│  "Most proposals fail due to poor or missing Pink Teams"         │
│                                                                  │
│  • Real-time gap analysis dashboard                              │
│  • Color team workflow support (Pink, Red, Gold)                 │
│  • Comment consolidation and prioritization                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PRIORITY 3: DIFFERENTIATORS                   │
│                                                                  │
│  "Competitive window closing—Vultron, Deltek advancing"          │
│                                                                  │
│  • Red Team simulation (SSEB scoring model)                      │
│  • Content generation (first drafts)                             │
│  • Microsoft ecosystem integration                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Roadmap Timeline

| Phase | Focus | Duration | Key Deliverables |
|-------|-------|----------|------------------|
| **Phase 2** | Trust Foundation | Q1 2025 | Compliance audit, source traceability |
| **Phase 3** | Color Team Workflows | Q2 2025 | Gap analysis, Pink Team support |
| **Phase 4** | Red Team Simulation | Q3 2025 | SSEB modeling, scoring |
| **Phase 5** | Full Generation | Q4 2025+ | Content drafting, past performance |

---

## 6. Phase 2: Trust Foundation

### 6.1 Overview

**Objective:** Prove compliance accuracy (the trust gate) and add source traceability.

**Duration:** Q1 2025 (12 weeks)

**Success Criteria:**
- 99%+ requirement extraction accuracy on 20+ RFP test set
- Zero false negatives (missed mandatory requirements)
- Every AI-generated paragraph links to source

### 6.2 Feature: Compliance Accuracy Audit

#### 6.2.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| CA-001 | System shall extract 99%+ of all mandatory (SHALL/MUST) requirements | P1 |
| CA-002 | System shall have zero false negatives for compliance-critical items | P1 |
| CA-003 | System shall detect conflicting requirements (e.g., page limit conflicts) | P1 |
| CA-004 | System shall auto-detect pass/fail compliance gates | P1 |
| CA-005 | System shall provide extraction confidence scores per requirement | P2 |
| CA-006 | System shall generate accuracy report comparing to manual shred | P2 |

#### 6.2.2 Acceptance Criteria

```gherkin
Feature: Compliance Accuracy

  Scenario: Mandatory requirement extraction
    Given an RFP with 100 mandatory requirements
    When the system processes the RFP
    Then at least 99 requirements shall be extracted
    And all compliance gates shall be flagged

  Scenario: Page limit conflict detection
    Given an RFP with conflicting page allocations
    When the system analyzes the RFP
    Then a conflict alert shall be raised
    And draft Q&A language shall be suggested
```

#### 6.2.3 Implementation Notes

1. Create test corpus of 20+ RFPs with manual requirement counts
2. Implement comparison tool: PropelAI extraction vs. manual
3. Add "Conflict Detection" module for mathematical conflicts
4. Add "Compliance Gate" flag for pass/fail requirements

### 6.3 Feature: Source Traceability

#### 6.3.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| ST-001 | Every AI-generated paragraph shall link to RFP section source | P1 |
| ST-002 | Every claim shall link to content library source (if used) | P1 |
| ST-003 | System shall provide confidence score for each generated item | P1 |
| ST-004 | System shall distinguish AI-generated vs human-written content | P1 |
| ST-005 | System shall provide "why" explanations for compliance ratings | P2 |
| ST-006 | System shall maintain audit trail for all changes | P2 |

#### 6.3.2 User Stories

**US-ST-01: View Requirement Source**
> As a Proposal Manager, I want to see the exact RFP location for each extracted requirement, so I can verify accuracy and cite the source in my response.

**US-ST-02: Track Content Origin**
> As a Compliance Analyst, I want to know whether content was AI-generated or human-written, so I can apply appropriate verification.

**US-ST-03: Understand Confidence**
> As a Volume Lead, I want to see confidence scores on AI suggestions, so I know where to focus verification effort.

#### 6.3.3 Data Model Extension

```python
@dataclass
class SourceTrace:
    """Traceability for any generated content"""
    content_id: str              # Unique identifier
    source_type: SourceType      # RFP, LIBRARY, AI_GENERATED
    rfp_reference: Optional[str] # L.4.B.2
    page_number: Optional[int]   # Source page
    character_range: Optional[Tuple[int, int]]  # Exact location
    library_source: Optional[str] # Content library item ID
    confidence_score: float      # 0.0-1.0
    generation_method: str       # "extraction", "generation", "human"
    audit_trail: List[AuditEntry] # Change history
    verification_status: str     # "unverified", "verified", "rejected"
    verified_by: Optional[str]   # User who verified
```

### 6.4 Feature: Conflict Detection

#### 6.4.1 Conflict Types

| Type | Example | Detection Method |
|------|---------|------------------|
| **Page Limit Conflict** | 8 pages allocated, 20 required | Sum vs. limit |
| **Date Conflict** | Due date before questions deadline | Date comparison |
| **Reference Conflict** | Section L says 10, Section M says 5 | Cross-reference |
| **Format Conflict** | Font requirement differs across sections | Pattern match |

#### 6.4.2 Output: Risks & Conflicts Report

```
┌─────────────────────────────────────────────────────────────────┐
│              RFP RISKS & CONFLICTS REPORT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ⚠️ CRITICAL CONFLICTS                                          │
│  ─────────────────────                                           │
│  1. Page Limit Conflict (Section L, Page 45)                     │
│     - Volume limit: 8 pages                                      │
│     - SF1 allocation: 10 pages                                   │
│     - SF2 allocation: 10 pages                                   │
│     - TOTAL REQUIRED: 20 pages                                   │
│     - RECOMMENDATION: Submit Q&A to CO                           │
│     - DRAFT Q&A: "Section L specifies an 8-page limit for       │
│       the Technical Volume, while Sub-Factors 1 and 2 are       │
│       allocated 10 pages each. Please clarify..."               │
│                                                                  │
│  ⚡ COMPLIANCE GATES                                             │
│  ─────────────────                                               │
│  1. CMMI-SVC Certification (Section M, Page 12)                 │
│     - PASS/FAIL: "failure to provide proof... will result       │
│       in immediate disqualification"                            │
│     - ACTION: Obtain certification before submission            │
│                                                                  │
│  2. OCI Mitigation Plan (Section L.7, Page 23)                  │
│     - PASS/FAIL: "Proposals without OCI plan will not be        │
│       evaluated"                                                │
│     - ACTION: Prepare OCI mitigation plan                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.5 Phase 2 Deliverables

| Deliverable | Description | Acceptance |
|-------------|-------------|------------|
| Accuracy Test Suite | 20+ RFPs with verified requirement counts | Manual validation |
| Accuracy Dashboard | Visual comparison: PropelAI vs. manual | <1% variance |
| Source Trace UI | Click requirement → see exact source | All requirements linked |
| Confidence Scores | 0-100% confidence per requirement | Displayed in UI |
| Conflict Detector | Auto-detect page/date/reference conflicts | Zero missed in test set |
| Compliance Gate Flags | Identify pass/fail requirements | 100% gate detection |
| Audit Trail | Track all content changes | Full history |

---

## 7. Phase 3: Color Team Workflows

### 7.1 Overview

**Objective:** Integrate with proposal color team review process.

**Duration:** Q2 2025 (12 weeks)

**Success Criteria:**
- AI pre-review identifies 80%+ of issues before human reviewers
- Comment consolidation reduces review reconciliation time by 50%
- Real-time compliance dashboard shows live requirement coverage

### 7.2 Feature: Real-Time Gap Analysis

#### 7.2.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| GA-001 | Dashboard shall show requirement coverage % as writing progresses | P1 |
| GA-002 | Dashboard shall highlight unaddressed requirements | P1 |
| GA-003 | Dashboard shall show page limit status per section | P1 |
| GA-004 | Dashboard shall update in real-time (polling or websocket) | P2 |
| GA-005 | Dashboard shall flag requirements approaching deadline | P2 |

#### 7.2.2 Dashboard Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│                REAL-TIME COMPLIANCE DASHBOARD                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Overall Coverage: ████████████░░░░░░░░ 67%                     │
│                                                                  │
│  By Volume:                                                      │
│  ─────────                                                       │
│  Technical     ██████████████░░░░░░ 72%   [View Gaps]           │
│  Management    ████████████████░░░░ 82%   [View Gaps]           │
│  Past Perf     ██████████░░░░░░░░░░ 51%   [View Gaps] ⚠️        │
│  Cost          ░░░░░░░░░░░░░░░░░░░░  0%   [Not Started]        │
│                                                                  │
│  Page Status:                                                    │
│  ────────────                                                    │
│  Section 1.0   8/10 pages   ████████░░ OK                       │
│  Section 2.0   12/10 pages  ████████████ ⚠️ OVER BY 2           │
│  Section 3.0   4/15 pages   ████░░░░░░░░░░░ Under               │
│                                                                  │
│  ⚠️ URGENT: 23 mandatory requirements not yet addressed         │
│  [View List]                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Feature: Pink Team AI Pre-Review

#### 7.3.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| PT-001 | System shall identify compliance gaps before human review | P1 |
| PT-002 | System shall flag unsupported claims ("We will..." without proof) | P1 |
| PT-003 | System shall calculate readability scores (Flesch-Kincaid) | P2 |
| PT-004 | System shall check win theme presence across sections | P2 |
| PT-005 | System shall generate prioritized issue list | P1 |

#### 7.3.2 Pre-Review Report

```
┌─────────────────────────────────────────────────────────────────┐
│              PINK TEAM AI PRE-REVIEW REPORT                      │
│              Technical Volume v0.8                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SUMMARY                                                         │
│  ────────                                                        │
│  Compliance Score: 78%                                           │
│  Readability: Grade 14.2 (Target: 12)                           │
│  Win Theme Coverage: 3/5 themes present                         │
│  Unsupported Claims: 12 found                                   │
│                                                                  │
│  🔴 CRITICAL ISSUES (Must Fix)                                   │
│  ─────────────────────────────                                   │
│  1. Missing response to L.4.B.2 (Mandatory)                     │
│     Section 1.3 does not address transition planning            │
│                                                                  │
│  2. Unsupported claim (Page 8, Para 2)                          │
│     "We will deliver on time and within budget"                 │
│     → Add: specific proof point, past performance reference     │
│                                                                  │
│  🟡 RECOMMENDED IMPROVEMENTS                                     │
│  ─────────────────────────────                                   │
│  1. Readability (Page 12)                                       │
│     Flesch-Kincaid: 16.8 → Target: 12                           │
│     Consider: shorter sentences, active voice                   │
│                                                                  │
│  2. Win Theme "Innovation" not found                            │
│     Sections 2.1, 2.3, 2.4 should reference innovation          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 Feature: Comment Consolidation

#### 7.4.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| CC-001 | System shall aggregate comments from multiple reviewers | P1 |
| CC-002 | System shall identify duplicate/conflicting comments | P1 |
| CC-003 | System shall prioritize comments by impact | P1 |
| CC-004 | System shall track comment resolution status | P2 |
| CC-005 | System shall generate disposition matrix | P2 |

#### 7.4.2 Consolidated View

```
┌─────────────────────────────────────────────────────────────────┐
│              COMMENT CONSOLIDATION - Section 2.1                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GROUPED COMMENT: Technical Approach Clarity                     │
│  Priority: HIGH (3 reviewers)                                    │
│  Status: ⬜ Unresolved                                           │
│                                                                  │
│  Reviewers:                                                      │
│  • John S: "Need more detail on implementation approach"         │
│  • Mary T: "Technical solution unclear - expand"                 │
│  • Bob K: "How does this meet the SOW requirement?"             │
│                                                                  │
│  AI RECOMMENDATION:                                              │
│  Expand Section 2.1.3 to include:                               │
│  - Implementation timeline                                       │
│  - Resource allocation                                           │
│  - Link to SOW 3.2.1                                            │
│                                                                  │
│  [Mark Resolved] [Assign Owner] [Dismiss]                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5 Phase 3 Deliverables

| Deliverable | Description | Acceptance |
|-------------|-------------|------------|
| Gap Analysis Dashboard | Real-time compliance coverage | Live updates |
| Page Counter | Per-section page tracking | Auto-calculate |
| Pink Team Pre-Review | AI issue identification | 80%+ issue detection |
| Readability Scoring | Flesch-Kincaid per section | Accurate to ±0.5 |
| Win Theme Checker | Theme presence detection | Configurable themes |
| Comment Consolidator | Multi-reviewer aggregation | Duplicate detection |
| Disposition Matrix | Comment tracking | Full lifecycle |

---

## 8. Phase 4: Red Team Simulation

### 8.1 Overview

**Objective:** Simulate government SSEB evaluation methodology.

**Duration:** Q3 2025 (12 weeks)

**Success Criteria:**
- AI scoring correlates with actual debrief results (within 1 rating level)
- Strength/weakness identification matches post-award feedback
- Color team efficiency improved by 40%

### 8.2 Feature: SSEB Scoring Simulation

#### 8.2.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SS-001 | System shall score proposals against Section M criteria | P1 |
| SS-002 | System shall use adjectival ratings matching agency methodology | P1 |
| SS-003 | System shall identify strengths, weaknesses, deficiencies, risks | P1 |
| SS-004 | System shall provide narrative evaluation per factor | P2 |
| SS-005 | System shall compare to competitor baseline (when available) | P2 |

#### 8.2.2 SSEB Rating Scale

```python
class SSEBRating(Enum):
    OUTSTANDING = "Outstanding"      # Exceeds requirements, no weaknesses
    GOOD = "Good"                    # Meets requirements, minor weaknesses
    ACCEPTABLE = "Acceptable"        # Meets minimum, some weaknesses
    MARGINAL = "Marginal"            # Does not meet some requirements
    UNACCEPTABLE = "Unacceptable"    # Major deficiencies
```

#### 8.2.3 Evaluation Report

```
┌─────────────────────────────────────────────────────────────────┐
│              RED TEAM EVALUATION SIMULATION                      │
│              Proposal: 75N96025R00004                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  OVERALL ASSESSMENT: GOOD                                        │
│                                                                  │
│  Factor 1: Technical Approach (40%)                              │
│  Rating: GOOD                                                    │
│                                                                  │
│  STRENGTHS:                                                      │
│  S-1: Innovative AI-assisted methodology exceeds SOW 3.2        │
│  S-2: Strong past performance on similar NIH contracts          │
│                                                                  │
│  WEAKNESSES:                                                     │
│  W-1: Limited detail on Phase 2 transition (L.4.B.2)            │
│  W-2: Staffing plan lacks named personnel for Year 2            │
│                                                                  │
│  DEFICIENCIES:                                                   │
│  D-1: None identified                                           │
│                                                                  │
│  RISKS:                                                          │
│  R-1: Key personnel retention (2 of 3 are new hires)            │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  Factor 2: Management Approach (30%)                             │
│  Rating: ACCEPTABLE                                              │
│  ...                                                             │
│                                                                  │
│  RECOMMENDATIONS:                                                │
│  1. Expand Phase 2 transition plan (address W-1)                │
│  2. Add named personnel for Year 2 positions (address W-2)      │
│  3. Include retention strategy for key staff (mitigate R-1)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Feature: Unsupported Claim Detection

#### 8.3.1 Claim Types

| Type | Pattern | Required Proof |
|------|---------|----------------|
| **Capability Claim** | "We can/will/are able to..." | Past performance, certification |
| **Experience Claim** | "We have done/delivered..." | Contract reference, metrics |
| **Quality Claim** | "Best-in-class/industry-leading..." | Third-party validation |
| **Timeline Claim** | "Within X days/weeks..." | Schedule, resource plan |

#### 8.3.2 Detection & Recommendation

```
┌─────────────────────────────────────────────────────────────────┐
│              UNSUPPORTED CLAIM ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ⚠️ CLAIM: "We will deliver the system on time and within       │
│            budget, as we have done on all previous contracts."  │
│                                                                  │
│  Location: Section 2.1, Page 8, Paragraph 3                     │
│  Type: Capability + Experience Claim                            │
│  Confidence: HIGH (this will be flagged by evaluators)          │
│                                                                  │
│  RECOMMENDATION:                                                 │
│  Replace with substantiated version:                            │
│                                                                  │
│  "We will deliver the system within the 18-month schedule,      │
│  supported by our track record of on-time delivery:             │
│  • NIH Contract 75N12345: Delivered 2 months early              │
│  • VA Contract 36C789: 100% on-time milestone completion        │
│  • HHS Contract 75P456: Under budget by 8%                      │
│                                                                  │
│  Our proven methodology (Section 2.3) and dedicated team        │
│  (Section 4.2) mitigate schedule risk."                         │
│                                                                  │
│  [Apply Suggestion] [Edit] [Dismiss]                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 Phase 4 Deliverables

| Deliverable | Description | Acceptance |
|-------------|-------------|------------|
| SSEB Scoring Engine | Factor-by-factor evaluation | Matches methodology |
| Strength/Weakness Identifier | Auto-detect S/W/D/R | 70%+ accuracy |
| Adjectival Rating | 5-level rating system | Configurable per agency |
| Evaluation Narrative | Per-factor assessment | Readable, actionable |
| Claim Detector | Find unsupported claims | <10% false positives |
| Proof Point Suggester | Recommend substantiation | Relevant suggestions |

---

## 9. Phase 5: Full Proposal Generation

### 9.1 Overview

**Objective:** AI-assisted content generation with human-in-the-loop.

**Duration:** Q4 2025 and beyond

**Success Criteria:**
- 70-80% complete first drafts in minutes vs. hours
- SME time reduced by 60%
- Quality matches human-written baselines

### 9.2 Feature: First Draft Generation

#### 9.2.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FD-001 | System shall generate section drafts from requirements | P1 |
| FD-002 | System shall pull from content library for boilerplate | P1 |
| FD-003 | System shall incorporate past performance data | P1 |
| FD-004 | System shall maintain source traceability for all content | P1 |
| FD-005 | System shall respect page limits in generation | P2 |
| FD-006 | System shall match proposal style guide | P2 |

#### 9.2.2 Generation Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Requirements   │────▶│  Content        │────▶│  First Draft    │
│  (Annotated     │     │  Library        │     │  Generation     │
│   Outline)      │     │  Search         │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Final          │◀────│  Human          │◀────│  Draft with     │
│  Content        │     │  Refinement     │     │  Source Traces  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 9.3 Feature: SME "Talk to Proposal"

#### 9.3.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| TP-001 | SMEs shall be able to provide input via natural language | P1 |
| TP-002 | System shall convert spoken/typed input to proposal prose | P1 |
| TP-003 | System shall ask clarifying questions to fill gaps | P2 |
| TP-004 | System shall maintain SME attribution | P1 |

#### 9.3.2 Interview Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              SME INTERVIEW: Technical Approach                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PropelAI: "Section 2.3 needs to address requirement C.3.2.1:  │
│            'The contractor shall implement a continuous         │
│            monitoring system.' Can you describe our approach?"  │
│                                                                  │
│  SME: "Yeah, so we use Splunk for log aggregation combined     │
│        with custom dashboards. We've done this on the VA        │
│        contract. Response time is under 15 minutes for          │
│        critical alerts. We have 24x7 NOC coverage."            │
│                                                                  │
│  PropelAI: "Great. What's the team structure for the NOC?"     │
│                                                                  │
│  SME: "Three shifts, 4 people per shift. Tier 1 handles        │
│        initial triage, escalates to Tier 2 for complex issues. │
│        We have a 95% first-contact resolution rate."           │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  GENERATED DRAFT:                                                │
│                                                                  │
│  "[Company] will implement a continuous monitoring system       │
│  using Splunk for centralized log aggregation, integrated with  │
│  custom dashboards for real-time visibility. Our 24x7 Network  │
│  Operations Center, staffed across three shifts with 4 analysts │
│  each, ensures response times under 15 minutes for critical     │
│  alerts. Our tiered support model (Tier 1 triage, Tier 2        │
│  escalation) achieves a 95% first-contact resolution rate,      │
│  proven on [VA Contract Reference]."                            │
│                                                                  │
│  Source: SME interview (John Smith, 12/15/2025)                 │
│                                                                  │
│  [Accept] [Edit] [Re-interview]                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 Feature: Past Performance Matching

#### 9.4.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| PP-001 | System shall search past performance database by relevance | P1 |
| PP-002 | System shall match requirements to past projects | P1 |
| PP-003 | System shall generate past performance summaries | P2 |
| PP-004 | System shall highlight gaps in experience | P2 |

### 9.5 Phase 5 Deliverables

| Deliverable | Description | Acceptance |
|-------------|-------------|------------|
| Draft Generator | Section-by-section generation | 70%+ complete drafts |
| Content Library | Searchable boilerplate database | Relevance ranking |
| SME Interview | Natural language input | Prose conversion |
| Past Performance Matcher | Requirement-to-project matching | Relevance scoring |
| Style Guide Enforcer | Formatting compliance | Configurable rules |

---

## 10. Integration Requirements

### 10.1 Microsoft Ecosystem (Critical)

> "Tools that don't connect to Word/SharePoint face rejection regardless of AI capability."
> — Compass Research

| Integration | Priority | Capability |
|-------------|----------|------------|
| **Word Plugin** | P1 | Two-way editing, requirement lookup |
| **SharePoint** | P1 | Content library connectivity |
| **Teams** | P2 | Notifications, collaboration |
| **Outlook** | P3 | Email tracking, Q&A management |

### 10.2 GovCon Systems

| System | Priority | Purpose |
|--------|----------|---------|
| **SAM.gov** | P2 | Opportunity monitoring |
| **GovWin IQ** | P3 | Market intelligence |
| **Costpoint** | P3 | Financial integration |
| **Deltek Vision** | P3 | Project data |

### 10.3 API Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PropelAI API Gateway                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  REST API (Current)                                             │
│  ├── /api/rfp/*           Core functionality                    │
│  ├── /api/content/*       Content library                       │
│  └── /api/integration/*   External systems                      │
│                                                                  │
│  GraphQL API (Planned)                                          │
│  └── /graphql             Flexible queries                      │
│                                                                  │
│  Webhooks (Planned)                                             │
│  ├── Processing complete                                        │
│  ├── Review comments                                            │
│  └── Deadline alerts                                            │
│                                                                  │
│  Office Add-in API (Planned)                                    │
│  └── Word/Excel/Outlook plugins                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Technical Architecture Evolution

### 11.1 Current vs. Target Architecture

| Aspect | Current | Target |
|--------|---------|--------|
| Storage | In-memory | PostgreSQL + S3 |
| Authentication | None | OAuth 2.0 / SSO |
| API | Single-threaded | Async + workers |
| Deployment | Render free tier | AWS/Azure production |
| Caching | None | Redis |
| Search | Linear | Elasticsearch |
| AI Models | API calls | Managed + local hybrid |

### 11.2 Target Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PropelAI Production Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Web App    │  │ Word Plugin  │  │   Teams Bot  │  │  Mobile App  │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │             │
│         └────────────────┬┴─────────────────┴────────────────┘             │
│                          │                                                  │
│                          ▼                                                  │
│                   ┌─────────────┐                                           │
│                   │ API Gateway │  (Kong / AWS API Gateway)                │
│                   │ + Auth      │                                           │
│                   └──────┬──────┘                                           │
│                          │                                                  │
│         ┌────────────────┼────────────────┐                                │
│         │                │                │                                │
│         ▼                ▼                ▼                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │  API Server │  │  API Server │  │  API Server │  (Kubernetes pods)     │
│  │  (FastAPI)  │  │  (FastAPI)  │  │  (FastAPI)  │                        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                        │
│         │                │                │                                │
│         └────────────────┼────────────────┘                                │
│                          │                                                  │
│         ┌────────────────┼────────────────┐                                │
│         │                │                │                                │
│         ▼                ▼                ▼                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│  │ PostgreSQL  │  │    Redis    │  │ Elasticsearch│                       │
│  │  (RDS)      │  │  (Cache)    │  │  (Search)   │                        │
│  └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Background Workers (Celery)                     │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐         │   │
│  │  │ Document  │  │ Extraction│  │ AI Model  │  │ Export    │         │   │
│  │  │ Parser    │  │ Worker    │  │ Worker    │  │ Worker    │         │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         S3 (Document Storage)                        │   │
│  │  ├── /uploads/{org}/{rfp_id}/                                       │   │
│  │  ├── /outputs/{org}/{rfp_id}/                                       │   │
│  │  └── /content-library/{org}/                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Multi-Model AI Architecture

PropelAI uses a heterogeneous model mesh:

| Model | Provider | Use Case | Cost |
|-------|----------|----------|------|
| Claude 3.5 Sonnet | Anthropic | Complex extraction, analysis | $$ |
| GPT-4 Turbo | OpenAI | Generation, summarization | $$ |
| Gemini Pro | Google | Fast classification | $ |
| Local (Llama) | Self-hosted | Sensitive data, high volume | Fixed |

---

## 12. Go-to-Market Strategy

### 12.1 Target Segments

| Segment | Revenue | Entry Strategy | Price Point |
|---------|---------|----------------|-------------|
| **Small Business** | <$10M | Usage-based, quick ROI | $99-299/mo |
| **Mid-Tier** | $25-100M | Reference customers, case studies | $500-2,000/mo |
| **Primes** | >$100M | FedRAMP path, on-prem option | Enterprise |

### 12.2 Competitive Positioning

```
                        HIGH
                          │
                          │        ┌─────────────┐
         Accuracy         │        │  PropelAI   │
                          │        │  (Target)   │
                          │        └─────────────┘
                          │
                          │  ┌─────────────┐
                          │  │ VisibleThread│
                          │  └─────────────┘
                          │
                          │        ┌─────────────┐
                          │        │   Vultron   │
                          │        └─────────────┘
                          │
                          │            ┌─────────────┐
                          │            │   GovDash   │
                          │            └─────────────┘
                          │
                        LOW
                          └────────────────────────────────────▶
                                                              HIGH
                               Feature Completeness
```

### 12.3 Differentiation Messages

| For Segment | Key Message |
|-------------|-------------|
| **Compliance Analysts** | "100% requirement extraction—guaranteed" |
| **Proposal Managers** | "Real-time compliance dashboard" |
| **Capture Managers** | "Win probability modeling" |
| **Executives** | "30% more bids with same headcount" |

---

## 13. Success Metrics

### 13.1 Product Metrics

| Metric | Current | Phase 2 Target | Phase 5 Target |
|--------|---------|----------------|----------------|
| Extraction accuracy | 85% | 99% | 99.5% |
| Processing time | 5-10 min | 3-5 min | 2-3 min |
| User adoption | N/A | 50 orgs | 500 orgs |
| NPS | N/A | 40 | 60 |

### 13.2 Business Metrics

| Metric | Phase 2 | Phase 3 | Phase 5 |
|--------|---------|---------|---------|
| MRR | $10K | $50K | $500K |
| Customers | 20 | 100 | 500 |
| Retention | 80% | 85% | 90% |

### 13.3 Customer Outcomes

| Outcome | Baseline | Target |
|---------|----------|--------|
| RFP shred time | 2-5 days | 30 minutes |
| Compliance errors | 5-10% | <1% |
| First draft time | 7+ hours | 30 minutes |
| Opportunities pursued | Baseline | +30% |
| Win rate improvement | Baseline | +5-10% |

---

## 14. Risks & Mitigations

### 14.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM hallucinations | High | High | Source traceability, human verification |
| Model API costs | Medium | Medium | Local model fallback, caching |
| Scale limitations | Medium | High | Async processing, cloud infrastructure |
| Data security | Medium | High | Encryption, FedRAMP path |

### 14.2 Market Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Competitor advancement | High | High | Differentiate on accuracy |
| Enterprise sales cycle | High | Medium | Land-and-expand model |
| Regulatory changes | Low | Medium | Monitor OMB/FAR updates |
| AI skepticism | Medium | Medium | Transparency, audit trails |

### 14.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Key person dependency | High | High | Documentation, knowledge sharing |
| Support scalability | Medium | Medium | Self-service, automation |
| Quality consistency | Medium | High | Testing, monitoring |

---

## 15. Appendix: Feature Specifications

### 15.1 Phase 2 Features (Detailed)

#### Feature: Compliance Accuracy Dashboard

**User Story:**
> As a Proposal Manager, I want to see accuracy metrics for PropelAI's extraction compared to manual shreds, so I can trust the system's output.

**Acceptance Criteria:**
- Display extraction accuracy percentage
- Show breakdown by requirement type
- Highlight any missed mandatory requirements
- Provide comparison to manual baseline

**Technical Notes:**
- Requires test corpus with verified requirement counts
- Store baseline comparisons in database
- Calculate accuracy per RFP and aggregate

---

#### Feature: Source Traceability UI

**User Story:**
> As a Compliance Analyst, I want to click on any requirement and see exactly where it came from in the RFP, so I can verify accuracy.

**Acceptance Criteria:**
- Click requirement → show source document
- Highlight exact text in original
- Show page number and section reference
- Display confidence score

**Technical Notes:**
- Store character offsets during extraction
- PDF viewer integration needed
- Consider PDF.js for in-browser viewing

---

#### Feature: Conflict Detection Engine

**User Story:**
> As a Capture Manager, I want the system to automatically detect conflicts in the RFP (like page limits that don't add up), so I can submit clarifying questions before they become problems.

**Acceptance Criteria:**
- Detect mathematical conflicts (page limits)
- Detect date conflicts
- Detect reference conflicts
- Generate draft Q&A questions
- Display in dedicated "Risks & Conflicts" section

**Technical Notes:**
- Parse page allocation tables
- Compare sums against stated limits
- Extract dates and validate timeline logic
- Store Q&A templates by conflict type

---

### 15.2 API Specifications (Planned)

#### POST /api/rfp/{id}/verify-accuracy

**Purpose:** Compare PropelAI extraction to manual baseline

**Request:**
```json
{
  "manual_counts": {
    "section_l": 45,
    "technical": 120,
    "evaluation": 30,
    "mandatory": 89
  }
}
```

**Response:**
```json
{
  "accuracy": {
    "overall": 0.97,
    "section_l": 0.98,
    "technical": 0.96,
    "evaluation": 0.97
  },
  "discrepancies": [
    {
      "category": "technical",
      "expected": 120,
      "actual": 115,
      "variance": -5
    }
  ],
  "missed_mandatory": []
}
```

---

#### GET /api/rfp/{id}/conflicts

**Purpose:** Get detected conflicts and risks

**Response:**
```json
{
  "conflicts": [
    {
      "type": "page_limit",
      "severity": "critical",
      "description": "Volume limit 8 pages, allocations total 20",
      "locations": [
        {"section": "L", "page": 45},
        {"section": "L", "page": 46}
      ],
      "draft_qa": "Section L specifies an 8-page limit..."
    }
  ],
  "compliance_gates": [
    {
      "requirement_id": "L.7.2",
      "text": "CMMI-SVC certification required",
      "consequence": "immediate disqualification",
      "action_required": "Obtain certification before submission"
    }
  ]
}
```

---

### 15.3 Data Migration Plan

#### From In-Memory to PostgreSQL

**Phase 1: Schema Design**
```sql
-- Core tables
CREATE TABLE organizations (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    created_at TIMESTAMP
);

CREATE TABLE rfp_projects (
    id UUID PRIMARY KEY,
    org_id UUID REFERENCES organizations(id),
    name VARCHAR(255),
    solicitation_number VARCHAR(100),
    agency VARCHAR(100),
    status VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE requirements (
    id UUID PRIMARY KEY,
    rfp_id UUID REFERENCES rfp_projects(id),
    rfp_reference VARCHAR(100),
    generated_id VARCHAR(100),
    full_text TEXT,
    category VARCHAR(50),
    binding_level VARCHAR(50),
    source_section VARCHAR(10),
    page_number INTEGER,
    is_compliance_gate BOOLEAN,
    source_trace JSONB,
    created_at TIMESTAMP
);

CREATE TABLE source_traces (
    id UUID PRIMARY KEY,
    requirement_id UUID REFERENCES requirements(id),
    source_type VARCHAR(50),
    document_name VARCHAR(255),
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    confidence FLOAT,
    verified_by UUID,
    verified_at TIMESTAMP
);
```

**Phase 2: Migration Script**
- Export in-memory data to JSON
- Transform to relational format
- Load into PostgreSQL
- Verify data integrity

**Phase 3: Dual-Write Period**
- Write to both memory and database
- Validate consistency
- Switch read operations to database
- Deprecate in-memory store

---

*End of Product Requirements Document*
