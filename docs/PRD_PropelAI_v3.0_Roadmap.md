# Product Requirements Document: PropelAI v3.0 Roadmap

**Document Version:** 3.1
**Date:** December 14, 2024
**Author:** PropelAI Engineering
**Status:** Strategic Planning
**Reference:** PropelAI Long-Form Generation Strategy

---

## 1. Executive Summary

### Vision Statement
Transform PropelAI from an RFP shredding tool into a complete **Autonomous Proposal Operating System (APOS)** that handles the full proposal lifecycle—from opportunity identification through submission—with minimal human intervention while maintaining government-grade compliance and audit trails.

### Strategic Goals
1. **Accuracy:** Achieve 95%+ requirement extraction accuracy
2. **Automation:** Reduce proposal development time by 60%
3. **Win Rate:** Increase client win rates by 20%
4. **Compliance:** Zero P0 violations in submitted proposals
5. **Scale:** Support 100+ concurrent proposals
6. **Density:** Generate 35-100+ page compliant responses (not summaries)

### Current State (v2.12)
- RFP shredding with 85%+ accuracy (best practices pipeline)
- 19-column CTM generation (Shipley methodology)
- Annotated outline generation with page allocations
- Amendment tracking and version control
- Basic win strategy and drafting capabilities
- OASIS+ module in development

---

## 2. The Brevity Bottleneck: Core Technical Challenge

### 2.1 Problem Statement

The government contracting sector operates on a fundamental paradox: while the volume of required documentation is immense (often 50-100+ pages per response), the tolerance for hallucination or superficiality is **zero**. Current LLMs have demonstrated capability for fluency but systemic failure in **compliance density**.

### 2.2 Root Cause: Summarization Bias

Modern LLMs are architecturally biased against long-form, detailed content due to their training:

| Training Factor | Effect | Impact on Proposals |
|-----------------|--------|-------------------|
| **RLHF Alignment** | Human labelers prefer concise answers | Models learn brevity = quality |
| **Length-Penalizing Reward Models** | Token generation "value" drops as length increases | Models truncate to minimize error probability |
| **Hallucination Avoidance** | Models default to "safe" generic answers | Specific, winning answers are avoided |
| **Mean Convergence** | Probability distributions favor generic output | "Industry-standard practices" vs. "NIST 800-171 Rev 2 with Splunk Enterprise" |

### 2.3 Why Simple Prompting Fails

Prompt engineering attempts ("write a long response," "be detailed") produce **verbose fluff**, not additional **substance**. The model stretches the same information over more words rather than retrieving new information. The solution is not linguistic (prompting) but **architectural** (orchestration and retrieval).

### 2.4 The Solution: Stateful Multi-Agent Cognitive Architecture

PropelAI v3.0 must transcend the single-prompt paradigm with:

1. **Cyclic Workflows:** Draft → Critique → Expand loops that mechanically force content expansion
2. **Graph of Records:** Replace naive RAG with relationship-aware retrieval
3. **Heterogeneous Model Mesh:** Right model for right task
4. **Criteria-Eval Gating:** Don't stop when sentence ends; stop when compliance is achieved

---

## 3. Roadmap Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PropelAI Strategic Roadmap                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2025 Q1          2025 Q2          2025 Q3          2025 Q4          2026+  │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ Phase 1  │───▶│ Phase 2  │───▶│ Phase 3  │───▶│ Phase 4  │───▶│Phase 5│ │
│  │Foundation│    │Compliance│    │ Content  │    │Full APOS │    │Scale  │ │
│  │& Ingestion│   │& Memory  │    │ Engine   │    │Orchestr. │    │       │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│                                                                             │
│  Key Deliverables:                                                          │
│  • Gemini 1M     • Graph of      • Draft-Critique• LangGraph   • Multi-    │
│  • 95% extract   • Records (GoR) • -Expand Loop  • Cycles      • tenant    │
│  • Metadata      • Criteria-Eval • Fine-tuned    • ToT Strategy• API       │
│  • Deduplication • Traceability  • Writer Model  • Human-loop  • Partners  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase 1: Foundation & Intelligent Ingestion (Q1 2025)

### 4.1 Massive Context Ingestion (The Librarian)

**Problem:** Standard 128K context windows are insufficient for large RFPs (300+ pages with attachments). Critical cross-references between sections are missed.

**Solution:** Deploy Gemini 1.5 Pro (1M token context) as the "Librarian" agent for lossless RFP ingestion.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P1.1 | Implement Gemini 1.5 Pro integration for full RFP ingestion | P0 |
| P1.2 | Load entire RFP package (Solicitation + Attachments + Q&A) in single context | P0 |
| P1.3 | Enable "global attention" - detect that C.4 is modified by footnote in Attachment J.8 | P0 |
| P1.4 | Generate comprehensive Compliance Matrix from full-context analysis | P1 |
| P1.5 | Extract all cross-document relationships in single pass | P1 |

**Success Metrics:**
- Process 300+ page RFPs without chunking
- Detect 95% of cross-document references
- Build complete requirement graph in single inference

---

### 4.2 Enhanced Metadata Extraction

**Problem:** Solicitation numbers, due dates, and agency information are not reliably extracted from all RFP formats.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P1.6 | Extract solicitation number from SF-33, SF-1449, cover pages | P0 |
| P1.7 | Parse due dates in all common formats (MM/DD/YYYY, ISO, text) | P0 |
| P1.8 | Identify contracting office and agency automatically | P1 |
| P1.9 | Detect set-aside type (8(a), HUBZone, WOSB, SDVOSB, Full & Open) | P1 |
| P1.10 | Extract contract type (FFP, T&M, CPFF, IDIQ) | P1 |
| P1.11 | Parse NAICS codes and size standards | P1 |
| P1.12 | Identify incumbent contractor when mentioned | P2 |

**Success Metrics:**
- 95% accuracy on solicitation number extraction
- 90% accuracy on due date parsing
- Fallback to manual entry with clear prompts when confidence < 80%

---

### 4.3 Content Deduplication Engine

**Problem:** Requirements appear in multiple sections/factors due to overlapping semantic matches.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P1.13 | Implement TF-IDF scoring for requirement-to-section relevance | P0 |
| P1.14 | Track assigned requirements to prevent duplicate assignment | P0 |
| P1.15 | Use section hierarchy to resolve conflicts | P1 |
| P1.16 | Provide confidence scores for each mapping | P1 |
| P1.17 | Allow manual override with persistence | P1 |

**Success Metrics:**
- Less than 5% content duplication across factors
- Clear audit trail of assignment decisions

---

### 4.4 Extraction Pipeline Optimization

**Problem:** Current 85% accuracy leaves gaps that require manual cleanup.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P1.18 | Add context window expansion for edge-case requirements | P0 |
| P1.19 | Implement confidence calibration based on training data | P1 |
| P1.20 | Add pattern library for common non-UCF formats | P1 |
| P1.21 | Improve table extraction with structure preservation | P1 |

**Success Metrics:**
- Achieve 95% extraction accuracy on UCF format RFPs
- 90% accuracy on non-standard formats

---

## 5. Phase 2: Compliance Excellence & Graph Memory (Q2 2025)

### 5.1 Graph of Records (GoR) Implementation

**Problem:** Standard RAG breaks documents into small chunks and retrieves by semantic similarity, destroying structural context. This leads to hallucinations when the model blends unrelated content.

**Solution:** Implement Graph of Records methodology where nodes represent information units and edges represent relationships.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Graph of Records (GoR)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐         "Satisfies"          ┌─────────────┐             │
│   │ Requirement │ ─────────────────────────────▶│ Past Perf   │             │
│   │   C.3.1.2   │                               │  Project A  │             │
│   └─────────────┘                               └─────────────┘             │
│         │                                              │                    │
│         │ "Evaluated By"                               │ "Proves"           │
│         ▼                                              ▼                    │
│   ┌─────────────┐         "Contradicts"         ┌─────────────┐            │
│   │ Eval Factor │ ◀─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│   Resume    │            │
│   │    M.2      │                               │  John Doe   │            │
│   └─────────────┘                               └─────────────┘             │
│         │                                              │                    │
│         │ "Instructed By"                              │ "Supports"         │
│         ▼                                              ▼                    │
│   ┌─────────────┐         "Refers To"           ┌─────────────┐            │
│   │ Instruction │ ─────────────────────────────▶│  Generated  │            │
│   │    L.4      │                               │   Output    │            │
│   └─────────────┘                               └─────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P2.1 | Implement graph database (Neo4j) for requirement relationships | P0 |
| P2.2 | Create node types: Requirement, PastPerformance, Resume, Deliverable, Output | P0 |
| P2.3 | Create edge types: Satisfies, EvaluatedBy, InstructedBy, RefersTo, Contradicts | P0 |
| P2.4 | Enable graph traversal during generation (not just text retrieval) | P0 |
| P2.5 | Add generated outputs as new nodes in graph for cross-volume consistency | P1 |
| P2.6 | Detect contradictions between Technical Volume and Cost Volume | P1 |
| P2.7 | Enable agents to "retrieve" decisions from earlier agents via graph | P1 |

**Success Metrics:**
- 100% of requirements linked to source sections
- Cost Volume reflects Technical Volume decisions (via graph)
- 15% improvement in coherence metrics

---

### 5.2 Criteria-Eval Compliance Framework

**Problem:** Current evaluation checks for fluency, not completeness. A missing requirement is a fatal error.

**Solution:** Transform evaluation from subjective ("Is this good?") to objective ("Is this complete?") using Criteria-Eval methodology.

**Workflow:**
```
Step 1: Criteria Extraction (Planner Agent)
        ↓
        JSON: [{"id": "SHALL-001", "text": "Staffing plan", "mandatory": true}, ...]
        ↓
Step 2: Generation (Drafter Agent)
        ↓
Step 3: Verification (Evaluator Agent - LLM-as-Judge)
        ↓
        Scan text for each JSON criterion
        ↓
Step 4: Scoring
        ↓
        Coverage: 85% → FAIL (mandatory not met)
        ↓
Step 5: Gating
        ↓
        If < 100% mandatory: Trigger revision loop
```

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P2.8 | Implement automated criteria extraction from RFP sections | P0 |
| P2.9 | Generate JSON checklist of atomic requirements per section | P0 |
| P2.10 | Implement LLM-as-Judge with "hostile SSB member" persona | P0 |
| P2.11 | Calculate coverage scores per section | P1 |
| P2.12 | Gate progression on 100% mandatory requirement coverage | P1 |
| P2.13 | Generate specific missing-item feedback for revision loops | P1 |

**Success Metrics:**
- 100% of mandatory requirements verified before section completion
- Zero "missing shall statement" issues in final output
- Automated scoring matches human reviewer 90% of time

---

### 5.3 Bidirectional Requirement Traceability

**Problem:** No clear linkage between proposal sections and source requirements for evaluator compliance verification.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P2.14 | Assign unique IDs to all extracted requirements | P0 |
| P2.15 | Generate cross-reference matrix (requirement → section) | P0 |
| P2.16 | Generate reverse matrix (section → requirements) | P0 |
| P2.17 | Calculate compliance coverage percentage per section | P1 |
| P2.18 | Identify orphan requirements (extracted but unmapped) | P1 |
| P2.19 | Export traceability matrix to Excel | P1 |

**Success Metrics:**
- 100% of mandatory requirements tracked
- Real-time coverage dashboard
- Exportable compliance matrix

---

### 5.4 Automated Compliance Verification

**Problem:** P0 constraints are displayed but not automatically verified against proposal content.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P2.20 | Parse submitted proposal document for verification | P0 |
| P2.21 | Verify page counts against limits | P0 |
| P2.22 | Verify font and margin compliance via document inspection | P1 |
| P2.23 | Check for required section headings | P1 |
| P2.24 | Generate compliance checklist with pass/fail/warning | P1 |
| P2.25 | Pre-submission compliance gate | P1 |

**Success Metrics:**
- 95% P0 violation detection before submission
- Zero P0 violations in production-submitted proposals

---

### 5.5 Enhanced Section M Analysis

**Problem:** Evaluation factor importance and rating definitions need synthesis into actionable guidance.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P2.26 | Calculate relative importance from textual descriptions | P0 |
| P2.27 | Map adjectival ratings to specific evidence requirements | P1 |
| P2.28 | Generate "What Evaluators Want to See" guidance per factor | P1 |
| P2.29 | Identify discriminator opportunities based on weights | P1 |

**Success Metrics:**
- Importance ranking matches BD expert 85% of time
- 3+ actionable suggestions per evaluation factor

---

## 6. Phase 3: Content Generation Engine (Q3 2025)

### 6.1 Draft-Critique-Expand Loop (The Blue Team)

**Problem:** Linear chains ("fire-and-forget") cannot self-correct. If the model summarizes instead of detailing, the output is finalized with errors.

**Solution:** Implement cyclic workflows using LangGraph where the system loops until compliance criteria are met.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Draft-Critique-Expand Loop                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                          │
│   │   Planner   │ ──── Generates "Requirement Checklist" (JSON)            │
│   │   (Claude)  │                                                          │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                          │
│   │   Drafter   │ ──── Generates initial text from plan + context          │
│   │  (Fine-tuned│                                                          │
│   │   GPT-4)    │                                                          │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                          │
│   │  Critique   │ ──── Reviews against Requirement Checklist               │
│   │   (Claude)  │      Identifies: missing "shall", lack of substantiation,│
│   │ "Blue Team" │      excessive brevity                                   │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐     ┌─────────────────────────────────────────────────┐  │
│   │  Decision   │────▶│ If Compliant: Move to Final Polish              │  │
│   │    Edge     │     │ If Non-Compliant: Update State with specific    │  │
│   └─────────────┘     │   feedback → Loop back to Drafter               │  │
│          │            └─────────────────────────────────────────────────┘  │
│          │                                                                  │
│          └──────────── (Loop until 100% coverage OR max iterations)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P3.1 | Implement LangGraph cyclic workflow for section generation | P0 |
| P3.2 | Create Planner node that generates JSON requirement checklist | P0 |
| P3.3 | Create Drafter node that generates text from plan + retrieved context | P0 |
| P3.4 | Create Critique node ("Blue Team") that reviews against checklist | P0 |
| P3.5 | Implement Decision Edge with compliance gating | P0 |
| P3.6 | Maintain Critique History in state to prevent repeated mistakes | P1 |
| P3.7 | Set maximum loop iterations (5) with human escalation | P1 |
| P3.8 | Track token costs per loop for optimization | P2 |

**Success Metrics:**
- Generated drafts address 95% of section requirements (vs. 60% baseline)
- Average 2.3 loops to compliance
- Critique feedback is actionable and specific

---

### 6.2 Heterogeneous Model Strategy (Model Mesh)

**Problem:** No single model excels at all tasks. Using GPT-4 for everything wastes its strengths and exposes its weaknesses.

**Solution:** Deploy a router architecture that dispatches tasks to specialized models.

**Model Roles:**

| Model | Role | Rationale | Tasks |
|-------|------|-----------|-------|
| **Gemini 1.5 Pro** | The Librarian | 1M token context | Full RFP ingestion, cross-doc analysis, Compliance Matrix generation |
| **Claude 3.5 Sonnet** | The Architect/Critic | Superior instruction-following, less sycophantic | Planner, Critique (Blue Team), Orchestrator |
| **GPT-4 (Fine-tuned)** | The Writer | Calibrated for length; fine-tuned on winning proposals | Prose generation, section drafting |
| **Llama 3 (Local)** | The Verifier | Fast, cheap, private | Quick compliance checks, formatting validation |

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P3.9 | Implement model router with task-based dispatch | P0 |
| P3.10 | Integrate Gemini 1.5 Pro for ingestion tasks | P0 |
| P3.11 | Integrate Claude 3.5 Sonnet for planning/critique | P0 |
| P3.12 | Fine-tune GPT-4 variant on winning proposal corpus | P0 |
| P3.13 | Apply post-hoc reward calibration to remove length penalty | P1 |
| P3.14 | Implement fallback routing when primary model unavailable | P1 |
| P3.15 | Track per-model token costs and latency | P2 |

**Success Metrics:**
- 40% reduction in per-section generation cost (via appropriate model selection)
- Fine-tuned model produces 3x longer output with same quality
- Zero "safe utterance" summaries in technical sections

---

### 6.3 Hierarchical Context Injection

**Problem:** Context must be layered: immediate constraints + historical assets + formatting rules.

**Solution:** Inject context at three levels for each section.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Hierarchical Context Injection                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Level 1: THE LAW (Immediate Constraints)                                  │
│   ├── Full text of Section C.3 requirements                                 │
│   ├── Relevant Section L instructions                                       │
│   └── Section M evaluation criteria                                         │
│                                                                             │
│   Level 2: THE CAPABILITY (Historical Assets)                               │
│   ├── Top 5 "Gold Standard" past proposals related to C.3 (via GoR)        │
│   ├── Relevant past performance narratives                                  │
│   └── Key personnel resumes                                                 │
│                                                                             │
│   Level 3: THE FORM (Style and Compliance)                                  │
│   ├── "Proposal-ese" style guide                                            │
│   ├── Win themes to incorporate                                             │
│   └── P0 formatting constraints                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P3.16 | Implement 3-level context injection for each section | P0 |
| P3.17 | Retrieve historical assets via Graph of Records (not vector similarity) | P0 |
| P3.18 | Inject style guide and win themes at Level 3 | P1 |
| P3.19 | Track which context items were actually used in generation | P2 |

**Success Metrics:**
- All generated content cites specific RFP requirements
- Past performance is relevant (not just semantically similar)
- Win themes appear in 80% of technical sections

---

### 6.4 Win Theme Management System

**Problem:** Win themes are placeholders with no connection to capture intelligence.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P3.20 | Accept win themes from capture plan import | P0 |
| P3.21 | Map win themes to relevant proposal sections via GoR | P0 |
| P3.22 | Generate theme reinforcement suggestions per section | P1 |
| P3.23 | Track theme coverage across proposal | P1 |
| P3.24 | Ensure each volume has at least one primary theme | P1 |

**Success Metrics:**
- Win themes visible in each relevant section
- Coverage report shows theme distribution
- No volume without assigned themes

---

### 6.5 Past Performance Library Integration

**Problem:** Proof points are generic rather than company-specific.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P3.25 | Connect to corporate past performance database | P0 |
| P3.26 | Match past performance to current requirements via GoR edges | P1 |
| P3.27 | Suggest relevant metrics and achievements | P1 |
| P3.28 | Track proof point usage to prevent overuse | P1 |
| P3.29 | Auto-generate past performance narratives | P2 |

**Success Metrics:**
- 2+ relevant proof points per technical section
- Proof points include quantified metrics
- Usage tracking prevents redundancy

---

## 7. Phase 4: Full APOS Orchestration (Q4 2025)

### 7.1 LangGraph State Machine

**Problem:** Current orchestration is linear and stateless. Complex proposals require branching, looping, and persistent memory.

**Solution:** Full LangGraph implementation with state persistence, cyclic workflows, and human-in-the-loop checkpoints.

**State Schema:**
```python
class ProposalState(TypedDict):
    # Global Context (persists across all sections)
    win_themes: List[str]
    price_to_win: float
    teaming_partners: List[str]

    # Local Context (per-section)
    current_section: str
    section_requirements: List[Dict]

    # Critique History (prevents repeated mistakes)
    critique_log: List[Dict]  # {"section": "C.3", "issue": "missing staffing", "resolved": True}

    # Graph State
    generated_nodes: List[str]  # IDs of generated outputs in GoR

    # Compliance State
    coverage_scores: Dict[str, float]  # {"C.3": 0.85, "C.4": 1.0}
```

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P4.1 | Implement complete LangGraph state machine | P0 |
| P4.2 | State persistence with PostgreSQL checkpointing | P0 |
| P4.3 | Implement cyclic workflows for Draft-Critique-Expand | P0 |
| P4.4 | Maintain Global/Local/Critique state separation | P0 |
| P4.5 | Automatic phase transitions based on completion | P1 |
| P4.6 | Parallel agent execution where possible | P1 |
| P4.7 | Rollback capability to previous checkpoints | P2 |

**Success Metrics:**
- End-to-end processing without manual intervention
- State recovery from any failure point
- Processing time < 4 hours for complete proposal

---

### 7.2 Tree of Thoughts (ToT) for Strategic Sections

**Problem:** Linear drafting for Executive Summary or Technical Strategy yields generic output.

**Solution:** Implement Tree of Thoughts prompting for high-value sections.

**Workflow:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Tree of Thoughts (ToT)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Task: Define Risk Management Approach                                     │
│                                                                             │
│                    ┌──────────────────────────────┐                        │
│                    │         Root Task            │                        │
│                    │   "Risk Management Strategy" │                        │
│                    └──────────────┬───────────────┘                        │
│                                   │                                         │
│            ┌──────────────────────┼──────────────────────┐                 │
│            ▼                      ▼                      ▼                  │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│   │   Branch 1:     │   │   Branch 2:     │   │   Branch 3:     │         │
│   │  Aggressive/    │   │   Moderate/     │   │  Conservative/  │         │
│   │   Low-Cost      │   │   Balanced      │   │ High-Assurance  │         │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘         │
│            │                     │                      │                   │
│            └─────────────────────┼──────────────────────┘                  │
│                                  ▼                                          │
│                    ┌──────────────────────────────┐                        │
│                    │      Self-Evaluation         │                        │
│                    │   Score against Section M    │                        │
│                    │   evaluation criteria        │                        │
│                    └──────────────┬───────────────┘                        │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                        │
│                    │     Select Best Branch       │                        │
│                    │     Expand to Full Draft     │                        │
│                    └──────────────────────────────┘                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P4.8 | Implement ToT prompting for Executive Summary | P0 |
| P4.9 | Implement ToT for Technical Management Strategy | P0 |
| P4.10 | Generate 3 distinct approach branches per strategic section | P1 |
| P4.11 | Self-evaluate branches against Section M criteria | P1 |
| P4.12 | Select and expand winning branch automatically | P1 |
| P4.13 | Provide human override for branch selection | P2 |

**Success Metrics:**
- ToT improves strategic section quality by 40% (vs. linear)
- Selected branch aligns with Section M 90% of time
- Executive Summary rated "submission ready" 70% of time

---

### 7.3 Human-in-the-Loop Checkpoints

**Problem:** AI decisions need human oversight at critical junctures.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P4.14 | Define mandatory review checkpoints (strategy, draft, final) | P0 |
| P4.15 | Pause workflow for human approval | P0 |
| P4.16 | Capture and incorporate human feedback | P1 |
| P4.17 | Resume from checkpoint with modifications | P1 |
| P4.18 | Track all human interventions in audit log | P1 |
| P4.19 | Support multiple reviewer roles (Capture Lead, Volume Lead, BD) | P2 |

**Success Metrics:**
- Zero automated submissions without human approval
- Complete audit trail of decisions
- Feedback incorporated into future processing

---

### 7.4 Proposal Assembly and Submission

**Problem:** Final proposal assembly and submission is manual.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P4.20 | Assemble all volumes into final package | P1 |
| P4.21 | Generate table of contents and cross-references | P1 |
| P4.22 | Apply final formatting per P0 constraints | P1 |
| P4.23 | Generate compliance certification checklist | P1 |
| P4.24 | Package for submission (ZIP, naming conventions) | P1 |
| P4.25 | SAM.gov integration for electronic submission | P2 |

**Success Metrics:**
- Complete submission package in one click
- All P0 constraints verified before packaging

---

### 7.5 Red Team Enhancement

**Problem:** Red team evaluation needs deeper integration with remediation workflow.

**Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| P4.26 | Per-section color scoring with rationale | P0 |
| P4.27 | Specific remediation actions per finding | P1 |
| P4.28 | Re-score after remediation (via LangGraph loop) | P1 |
| P4.29 | Historical scoring trends | P2 |
| P4.30 | Pink team / Gold team differentiated reviews | P2 |

**Success Metrics:**
- All sections receive color scores
- 90% of findings have actionable remediation
- Re-score confirms improvement

---

## 8. Phase 5: Enterprise Scale (2026+)

### 8.1 Multi-User Collaboration

| ID | Requirement | Priority |
|----|-------------|----------|
| P5.1 | Real-time collaborative editing | P1 |
| P5.2 | Section assignment and ownership | P1 |
| P5.3 | Review and approval workflows | P1 |
| P5.4 | Version control with diff view | P1 |
| P5.5 | Role-based access control | P1 |

### 8.2 Multi-Tenant Architecture

| ID | Requirement | Priority |
|----|-------------|----------|
| P5.6 | Tenant isolation for data security | P0 |
| P5.7 | Per-tenant customization | P1 |
| P5.8 | Usage-based billing integration | P1 |
| P5.9 | Tenant-specific model fine-tuning | P2 |

### 8.3 Analytics and Learning

| ID | Requirement | Priority |
|----|-------------|----------|
| P5.10 | Win/loss tracking by approach patterns | P1 |
| P5.11 | Identify winning discriminators | P1 |
| P5.12 | Agency-specific preference learning | P2 |
| P5.13 | Competitor analysis integration | P2 |

### 8.4 API Platform

| ID | Requirement | Priority |
|----|-------------|----------|
| P5.14 | Public REST API with documentation | P1 |
| P5.15 | Webhook notifications | P1 |
| P5.16 | SDK for common languages | P2 |
| P5.17 | Partner integration marketplace | P2 |

---

## 9. Technical Architecture Evolution

### 9.1 Current Architecture (v2.12)

```
PDF → Python Extractor → Smart Outline Generator → JSON → Node.js Exporter → DOCX
```

### 9.2 Target Architecture (v3.0+)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PropelAI v3.0 Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     Model Mesh (Heterogeneous)                      │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │    │
│  │  │ Gemini 1.5 │  │ Claude 3.5 │  │  GPT-4     │  │  Llama 3   │   │    │
│  │  │   Pro      │  │  Sonnet    │  │(Fine-tuned)│  │  (Local)   │   │    │
│  │  │ Librarian  │  │ Architect  │  │   Writer   │  │  Verifier  │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    LangGraph Orchestration                          │    │
│  │  ┌──────────────────────────────────────────────────────────────┐ │    │
│  │  │                 Cyclic Workflows                              │ │    │
│  │  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │ │    │
│  │  │  │ Planner │───▶│ Drafter │───▶│Critique │───▶│Decision │   │ │    │
│  │  │  └─────────┘    └─────────┘    └─────────┘    └────┬────┘   │ │    │
│  │  │                       ▲                            │        │ │    │
│  │  │                       └────────────────────────────┘        │ │    │
│  │  │                          (Loop until compliant)             │ │    │
│  │  └──────────────────────────────────────────────────────────────┘ │    │
│  │                                                                    │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │    │
│  │  │  ToT       │  │ Criteria   │  │  Human     │  │   State    │  │    │
│  │  │ Strategic  │  │   Eval     │  │  Review    │  │ Checkpoint │  │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Memory & Retrieval Layer                         │    │
│  │  ┌────────────────────────────────────────────────────────────┐   │    │
│  │  │                 Graph of Records (GoR)                      │   │    │
│  │  │   Requirements ←→ Past Performance ←→ Resumes ←→ Outputs   │   │    │
│  │  └────────────────────────────────────────────────────────────┘   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │    │
│  │  │ Neo4j    │  │ Vector   │  │PostgreSQL│  │  Object  │          │    │
│  │  │ (Graph)  │  │ Store    │  │ (State)  │  │ Storage  │          │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Technology Investments by Phase

| Phase | Investment | Purpose |
|-------|------------|---------|
| 1 | Gemini 1.5 Pro integration | 1M token RFP ingestion |
| 2 | Neo4j + GoR implementation | Relationship-aware retrieval |
| 2 | Criteria-Eval framework | Objective compliance scoring |
| 3 | LangGraph cyclic workflows | Draft-Critique-Expand loops |
| 3 | Model fine-tuning pipeline | Remove brevity bias |
| 4 | ToT implementation | Strategic section quality |
| 4 | Full state machine | End-to-end orchestration |
| 5 | Multi-tenant infrastructure | Enterprise scale |

---

## 10. Success Metrics by Phase

### Phase 1 Metrics

| Metric | Current | Target |
|--------|---------|--------|
| RFP ingestion coverage | Chunked | Full document (1M tokens) |
| Cross-document reference detection | 50% | 95% |
| Metadata extraction accuracy | 60% | 95% |
| Extraction accuracy | 85% | 95% |

### Phase 2 Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Compliance coverage visibility | Manual | 100% automated |
| Requirement traceability | None | 100% bidirectional |
| P0 violation detection | Display only | 95% detect |
| Criteria-Eval accuracy | N/A | 90% match human |

### Phase 3 Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Section requirement coverage | 60% | 95% |
| Average draft loops to compliance | N/A | 2.3 |
| Generated content length | Summary | Full (3x baseline) |
| Fine-tuned model quality | N/A | No brevity bias |
| Author satisfaction | N/A | 80% "useful starting point" |

### Phase 4 Metrics

| Metric | Current | Target |
|--------|---------|--------|
| End-to-end automation | Partial | 90% automated |
| Processing time | 2-5 days | <4 hours |
| ToT strategy quality improvement | N/A | 40% better |
| Human review efficiency | N/A | 2 hours per proposal |

### Phase 5 Metrics

| Metric | Target |
|--------|--------|
| Concurrent proposals | 100+ |
| User capacity | 1000+ |
| Win rate improvement | 20% |
| Customer NPS | 50+ |

---

## 11. Cost Model

### Per-Proposal Token Estimates

| Component | Tokens | Model | Cost/1M | Est. Cost |
|-----------|--------|-------|---------|-----------|
| RFP Ingestion | 500K | Gemini 1.5 Pro | $3.50 | $1.75 |
| Planning (10 sections) | 200K | Claude 3.5 | $15.00 | $3.00 |
| Drafting (10 sections × 2 loops) | 600K | GPT-4 (fine-tuned) | $30.00 | $18.00 |
| Critique (10 sections × 2 loops) | 400K | Claude 3.5 | $15.00 | $6.00 |
| Verification | 100K | Llama 3 (local) | $0.00 | $0.00 |
| **Total per proposal** | | | | **~$29** |

### Monthly Infrastructure

| Component | Phase 1-2 | Phase 3-4 | Phase 5 |
|-----------|-----------|-----------|---------|
| Compute (cloud) | $5K | $15K | $50K |
| LLM API | $10K | $30K | $100K |
| Neo4j (Graph DB) | $2K | $5K | $15K |
| PostgreSQL | $1K | $3K | $10K |
| Storage | $1K | $3K | $10K |
| **Total** | **$19K** | **$56K** | **$185K** |

---

## 12. Dependencies and Risks

### Dependencies

| Dependency | Phase | Impact | Mitigation |
|------------|-------|--------|------------|
| Gemini 1.5 Pro availability | 1 | High | Fallback to chunked ingestion |
| Claude API stability | 2-4 | High | Multi-provider support |
| Fine-tuning data quality | 3 | High | Curated winning proposal corpus |
| Neo4j expertise | 2 | Medium | Training + consultant |
| LangGraph maturity | 4 | Medium | Early adoption, contribute fixes |

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Critique loops don't converge | Medium | High | Max iterations + human escalation |
| Fine-tuned model still brief | Medium | High | Reward calibration + prompt tuning |
| GoR graph becomes too large | Medium | Medium | Pruning strategies, caching |
| Token costs exceed budget | High | Medium | Model selection optimization |
| LLM hallucination | Medium | High | Citations, verification loops |
| Compliance failure | Low | Very High | Criteria-Eval gating |

---

## 13. Implementation Priorities

### Immediate (30 Days)
1. Gemini 1.5 Pro integration for full-context ingestion
2. Basic LangGraph workflow (linear, not cyclic)
3. Criteria extraction prototype

### Short-Term (60-90 Days)
1. Graph of Records schema and Neo4j setup
2. Criteria-Eval framework
3. Claude integration for critique agent
4. First cyclic workflow (single section)

### Medium-Term (Q2-Q3)
1. Full Draft-Critique-Expand loop
2. Model fine-tuning pipeline
3. Hierarchical context injection
4. Multi-section orchestration

### Long-Term (Q4+)
1. Tree of Thoughts for strategic sections
2. Complete state machine
3. Human-in-the-loop checkpoints
4. Proposal assembly automation

---

## 14. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| APOS | Autonomous Proposal Operating System |
| Brevity Bottleneck | LLM tendency to summarize rather than elaborate due to RLHF training |
| CTM | Compliance Traceability Matrix |
| Criteria-Eval | Methodology converting subjective quality to objective completeness |
| GoR | Graph of Records - relationship-aware retrieval replacing naive RAG |
| Iron Triangle | Section L + M + C mapping |
| LangGraph | Framework for stateful, cyclic LLM workflows |
| Model Mesh | Heterogeneous multi-model architecture |
| P0 Constraint | Pass/fail requirement causing disqualification |
| RLHF | Reinforcement Learning from Human Feedback |
| ToT | Tree of Thoughts - branching reasoning for strategic decisions |
| UCF | Uniform Contract Format |

### Appendix B: Research References

| Topic | Source | Key Finding |
|-------|--------|-------------|
| Summarization Bias | RLHF Research | Reward models penalize length |
| LangGraph | LangChain Docs | Enables cyclic, stateful workflows |
| Graph of Records | Arxiv | 15% improvement in coherence |
| Tree of Thoughts | Arxiv | 4% → 74% success rate improvement |
| Gemini Context | Google AI | 1M token window enables global attention |
| Claude Instruction Following | Anthropic | Superior to GPT-4 for critique |

### Appendix C: Integration Roadmap

| System | Phase | Integration Type |
|--------|-------|------------------|
| Neo4j | 2 | Graph database for GoR |
| Gemini API | 1 | LLM for ingestion |
| Claude API | 2 | LLM for planning/critique |
| OpenAI API | 3 | LLM for generation (fine-tuned) |
| SAM.gov | 4 | API (submission) |
| Deltek | 5 | API (opportunities) |

---

## 15. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-14 | PropelAI Engineering | Initial PRD |
| 3.0 | 2024-12-14 | PropelAI Engineering | Full system roadmap |
| 3.1 | 2024-12-14 | PropelAI Engineering | Integrated Long-Form Generation Strategy: Brevity Bottleneck analysis, LangGraph cycles, GoR, Model Mesh, ToT, Criteria-Eval |

---

**Document Status:** Ready for Executive Review
**Next Review Date:** January 2025
**Reference Document:** PropelAI Long-Form Generation Strategy.rtf
