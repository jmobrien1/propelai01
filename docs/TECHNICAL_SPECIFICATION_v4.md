# Technical Specification: PropelAI v4.0 Refactor (Stateful Agentic Architecture)

## 1. Architectural Vision & Constraints

**Objective:** Transition PropelAI from a linear extraction pipeline (v3.3) to a Stateful Multi-Agent Cognitive Architecture (v4.0).

**Core Philosophy:** The system must adhere to the "Iron Triangle" of GovCon logic, linking Section L (Instructions), Section M (Evaluation Factors), and Section C (Statement of Work).

**The Trust Gate:** The system must prioritize deterministic compliance (extraction accuracy) over probabilistic generation. No generative features are permitted without source traceability.

---

## 2. Phase 1: The "Trust Gate" (Source Traceability Engine) - COMPLETE ✓

**Goal:** Bridge the "Trust Gap" by providing mathematical proof of extraction accuracy via visual overlays.

### 2.1 Data Model Refactoring
**File:** `agents/enhanced_compliance/document_structure.py`

**Requirement:** The current StructuredRequirement class captures text but lacks geospatial grounding. You must implement the SourceCoordinate dataclass to support the frontend highlight overlay.

- **New Dataclass: SourceCoordinate**
  - Attributes:
    - `source_document_id` (str): UUID of the source PDF.
    - `page_index` (int): 1-based index of the page.
    - `visual_rects` (List[BoundingBox]): A list of rectangles, as requirements often span multiple lines or columns.

- **New Dataclass: BoundingBox**
  - `x` (float): Left coordinate (normalized 0.0–1.0).
  - `y` (float): Top coordinate (normalized 0.0–1.0).
  - `width` (float): Normalized width.
  - `height` (float): Normalized height.

- **Normalization Logic:** PDF coordinates (often bottom-left origin, points) must be converted to Web coordinates (top-left origin, percentage) to ensure the frontend react-pdf-highlighter component renders correctly regardless of screen resolution.

### 2.2 Parser Upgrade (Dual-Engine Strategy)
**File:** `agents/enhanced_compliance/parser.py`

**Current State:** Uses pypdf, which is excellent for text extraction but struggles with precise character-level bounding boxes.

**Refactor Plan:**
1. **Integrate pdfplumber or PyMuPDF:** Implement a secondary pass specifically for coordinate extraction.
2. **Mapping Logic:** When SectionAwareExtractor identifies a requirement string (e.g., "The contractor shall..."), the parser must locate that exact string in the pdfplumber layout object and retrieve the (x0, top, x1, bottom) tuples for every character in the sequence.
3. **Fallback Strategy:** Keep pypdf as the primary text extractor for speed. Only invoke the layout-aware parser when generating the SourceCoordinate object to optimize performance.

### 2.3 API Traceability Endpoint
**File:** `api/main.py`

**New Endpoint:** `GET /api/rfp/{rfp_id}/requirements/{req_id}/source`
- **Behavior:** Lazy-loads coordinates. When a user clicks a requirement in the UI, this endpoint returns the specific JSON payload required by the frontend overlay.
- **Response Schema:** Must return `boundingRect` (union of all rects for scrolling) and `rects` (individual line highlights).

---

## 3. Phase 2: The "Iron Triangle" Logic Engine - COMPLETE ✓

**Goal:** Move from "Shredding" to "Reasoning" by modeling the dependencies between RFP sections.

### 3.1 The Strategy Agent
**File:** `agents/strategy_agent.py` (New File)

**Architecture:** This is a LangGraph Node that accepts the ComplianceMatrix and RFPStructure as state.

**Logic Flow:**
1. **Decomposition:** Ingest Section M (Evaluation Factors). Extract scoring weights (e.g., "Technical is significantly more important than Past Performance").
2. **Cross-Walking:** Map Section M factors to Section C (SOW) paragraphs.
   - **Validation Rule:** If Section M lists a "Sub Factor 2: Infrastructure Approach," the agent must verify that Section L allows for a corresponding proposal volume or section. If Section L is silent, flag a "Structure Conflict".
3. **Conflict Detection:** Implement logic to detect page count conflicts (e.g., Section L limits Volume 1 to 20 pages, but the sum of required sub-sections implies 30 pages).

### 3.2 Strategic Data Models
**File:** `agents/enhanced_compliance/document_structure.py`

- **New Dataclass: WinTheme**
  - Attributes: `theme_id`, `discriminator` ("What we have that they don't"), `benefit_statement` ("How this helps the client"), `proof_points` (List of IDs from Company Library).

- **New Dataclass: CompetitorProfile**
  - Attributes: `name`, `known_weaknesses`, `likely_solution_approach`. This enables "Ghosting" strategies (subtle critiques of competitors).

---

## 4. Phase 3: The Drafting Agent (PEARL Framework Implementation) - COMPLETE ✓

**Goal:** Implement "Planning and Executing Actions over Long Documents" to prevent LLM hallucination and loss of focus.

### 4.1 LangGraph Orchestration
**File:** `agents/drafting_agent.py`

**Architecture:** Replace linear LLMChain with a StateGraph. State Schema: ProposalState.
- Attributes: `section_id`, `requirements` (List), `win_themes` (List), `draft_plan` (PEARL Plan), `draft_content` (str), `revision_count` (int).

### 4.2 PEARL Workflow Implementation (F-B-P Framework)

You must strictly implement the three stages of the PEARL framework defined in the research:

1. **Node A: Action Mining (The Architect)**
   - **Task:** Do not write text. Instead, decompose the drafting task into a set of discrete actions.
   - **Prompt Strategy:** "Analyze the requirements for this section. Identify necessary actions such as FIND_PROOF(Cybersecurity), CHECK_PAGE_LIMIT(Section_L), or SYNTHESIZE_THEME(Innovation)".

2. **Node B: Plan Generation (The Planner)**
   - **Task:** Generate a pseudocode plan.
   - **Output:**
     ```
     1. constraints = EXTRACT_CONSTRAINTS(Section_L)
     2. evidence = RETRIEVE_PROOF(Company_Library, "Agile Methodology")
     3. theme = GET_WIN_THEME(Section_M, "Factor 1")
     4. DRAFT_CONTENT(constraints, evidence, theme)
     ```
   - **Why:** Research shows this intermediate planning step significantly improves reasoning over long documents compared to zero-shot drafting.

3. **Node C: Plan Execution (The Writer)**
   - **Task:** Execute the plan. The writer is constrained to use only the evidence retrieved in Step 2.
   - **Constraint:** Use the Feature-Benefit-Proof (F-B-P) framework for all argumentative text.

4. **Node D: Red Team Critique (The Loop)**
   - **Task:** Score the draft against Section M criteria.
   - **Conditional Edge:** `if score < 80: return "DraftingNode" else: return END`.

---

## 5. Phase 4: Persistence Layer (PostgreSQL & pgvector) - COMPLETE ✓

**Goal:** Enable long-running, interruptible agent workflows and semantic search.

### 5.1 Database Architecture

**Constraint:** Do not use SQLite. Use PostgreSQL with the pgvector extension.

**Reasoning:**
1. **Vector Search:** The "Company Library" (RAG) requires high-dimensional vector storage to find "relevant past performance" matching RFP requirements. SQLite cannot handle this natively at scale.
2. **LangGraph Checkpointing:** To allow a user to pause a proposal (e.g., "Reviewing Draft") and resume days later, the graph state must be serialized to a robust backend. Postgres provides the concurrency and reliability needed for this state management.

### 5.2 Implementation Tasks
1. **Dockerize:** Add `pgvector/pgvector:pg16` to the `docker-compose.yml`.
2. **Schema:** Create tables for projects, requirements (with JSONB for coordinates), and proposal_states (for LangGraph checkpoints).
3. **Migration:** Update `persistence.py` to use SQLAlchemy or AsyncPG instead of local file I/O.

---

## Summary of Immediate Actions for the Agent

1. ✓ **Read:** AS_BUILT_TDD.md (Current State) and PropelAI v4.0 Architecture Specification.docx (Target State).
2. ✓ **Install:** pdfplumber for Phase 1.
3. ✓ **Execute Phase 1:** Modify parser.py to extract bounding boxes and populate SourceCoordinate.
4. ✓ **Execute Phase 2:** Build StrategyAgent to map L-M-C dependencies.
5. ✓ **Execute Phase 3:** Implement StateGraph with PEARL nodes (Plan -> Execute).
6. ✓ **Execute Phase 4:** PostgreSQL + pgvector for Company Library semantic search.

---

## Implementation Status

| Phase | Component | File(s) | Status |
|-------|-----------|---------|--------|
| 1 | BoundingBox dataclass | `models.py` | ✓ Complete |
| 1 | SourceCoordinate dataclass | `models.py` | ✓ Complete |
| 1 | PDFCoordinateExtractor | `pdf_coordinate_extractor.py` | ✓ Complete |
| 1 | Source endpoint | `api/main.py` | ✓ Complete |
| 2 | StrategyAgent | `strategy_agent.py` | ✓ Complete |
| 2 | CompetitorAnalyzer | `strategy_agent.py` | ✓ Complete |
| 2 | GhostingLanguageGenerator | `strategy_agent.py` | ✓ Complete |
| 2 | WinTheme dataclass | `document_structure.py` | ✓ Complete |
| 3 | DraftingWorkflow (LangGraph) | `drafting_workflow.py` | ✓ Complete |
| 3 | F-B-P Framework nodes | `drafting_workflow.py` | ✓ Complete |
| 4 | PostgreSQL + pgvector | `docker-compose.yml`, `init.sql` | ✓ Complete |
| 4 | VectorStore | `api/vector_store.py` | ✓ Complete |
| 4 | Company Library tables | `api/database.py` | ✓ Complete |
| 4 | Embedding generation | `api/vector_store.py` | ✓ Complete |

---

*Last Updated: December 2024*
*All v4.0 phases complete and deployed to Render.com*
