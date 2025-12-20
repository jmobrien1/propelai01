# PropelAI v4.0 Implementation Plan

**Version:** 4.0 (Agentic Architecture)
**Created:** December 20, 2025
**Status:** Implementation Roadmap

---

## Executive Summary

PropelAI v4.0 transforms the platform from a document processing tool into a fully agentic proposal development system. The architecture introduces four major phases:

1. **Trust Gate** - Source traceability with PDF coordinate tracking
2. **Strategy Engine** - Enhanced win theme development with Iron Triangle
3. **Drafting Agent** - LangGraph-powered F-B-P content generation
4. **Word Plugin** - Native Microsoft Word integration

---

## Phase 1: Trust Gate (Source Traceability)

### Overview
The Trust Gate ensures every extracted requirement can be traced back to its exact location in the source PDF, enabling one-click verification.

### 1.1 SourceCoordinate Model

**File:** `agents/enhanced_compliance/models.py`

**Task 1.1.1: Create SourceCoordinate dataclass**
```python
@dataclass
class SourceCoordinate:
    document_id: str           # Unique document identifier
    page_number: int           # 1-indexed page number
    bounding_box: BoundingBox  # PDF coordinates
    text_snippet: str          # Extracted text (first 200 chars)
    extraction_method: str     # "pdfplumber" | "pypdf" | "ocr"
    confidence: float          # 0.0 - 1.0
```

**Dependencies:** None
**Complexity:** Low
**Priority:** P0 (Foundation)

---

**Task 1.1.2: Create BoundingBox dataclass**
```python
@dataclass
class BoundingBox:
    x0: float    # Left edge (PDF points)
    y0: float    # Bottom edge (PDF points)
    x1: float    # Right edge (PDF points)
    y1: float    # Top edge (PDF points)
    page_width: float
    page_height: float

    def to_css_percent(self) -> Dict[str, float]:
        """Convert to CSS percentage positioning for overlay"""
        return {
            "left": (self.x0 / self.page_width) * 100,
            "top": (1 - self.y1 / self.page_height) * 100,
            "width": ((self.x1 - self.x0) / self.page_width) * 100,
            "height": ((self.y1 - self.y0) / self.page_height) * 100,
        }
```

**Dependencies:** None
**Complexity:** Low
**Priority:** P0 (Foundation)

---

### 1.2 PDF Coordinate Extraction

**File:** `agents/enhanced_compliance/pdf_coordinate_extractor.py` (NEW)

**Task 1.2.1: Implement PDFCoordinateExtractor class**
- Install pdfplumber for coordinate extraction: `pip install pdfplumber`
- Extract text with bounding boxes per word/line
- Handle multi-column layouts
- Support OCR fallback with Tesseract coordinates

```python
class PDFCoordinateExtractor:
    def extract_with_coordinates(self, pdf_path: str) -> List[TextBlock]:
        """Extract text blocks with bounding box coordinates"""

    def find_text_location(self, pdf_path: str, search_text: str) -> Optional[SourceCoordinate]:
        """Find exact location of text in PDF"""

    def highlight_requirement(self, pdf_path: str, requirement_text: str) -> SourceCoordinate:
        """Locate requirement and return coordinates for UI highlighting"""
```

**Dependencies:** Task 1.1.1, Task 1.1.2
**Complexity:** Medium
**Priority:** P0

---

**Task 1.2.2: Integrate coordinates into requirement extraction**

**File:** `agents/enhanced_compliance/resilient_extractor.py`

- Modify `_extract_chunk()` to capture coordinates
- Store SourceCoordinate in requirement dict
- Add `source_coordinates: List[SourceCoordinate]` field

**Dependencies:** Task 1.2.1
**Complexity:** Medium
**Priority:** P0

---

### 1.3 Frontend PDF Viewer with Highlighting

**File:** `web/index.html`

**Task 1.3.1: Integrate PDF.js viewer**
- Add PDF.js library (Mozilla)
- Create side-by-side layout: Requirements List | PDF Viewer
- Implement page navigation

**Complexity:** Medium
**Priority:** P1

---

**Task 1.3.2: Implement highlight overlay system**
- On requirement click, navigate to page
- Draw semi-transparent highlight box using CSS positioning
- Animate scroll-to-highlight

```javascript
function highlightRequirement(sourceCoordinate) {
    const { page_number, bounding_box } = sourceCoordinate;
    pdfViewer.navigateToPage(page_number);
    const cssPos = boundingBoxToCSS(bounding_box);
    createHighlightOverlay(cssPos);
}
```

**Dependencies:** Task 1.3.1, Task 1.2.2
**Complexity:** Medium
**Priority:** P1

---

### 1.4 API Endpoints

**File:** `api/main.py`

**Task 1.4.1: Add coordinate retrieval endpoint**
```python
@app.get("/api/requirements/{req_id}/source")
async def get_requirement_source(req_id: str) -> SourceCoordinate:
    """Get source coordinates for a requirement"""
```

**Task 1.4.2: Add PDF page image endpoint**
```python
@app.get("/api/documents/{doc_id}/page/{page_num}/image")
async def get_page_image(doc_id: str, page_num: int) -> FileResponse:
    """Return rendered PDF page as image for overlay"""
```

**Dependencies:** Task 1.2.2
**Complexity:** Low
**Priority:** P1

---

### Phase 1 Deliverables Summary

| Task | Description | Complexity | Priority |
|------|-------------|------------|----------|
| 1.1.1 | SourceCoordinate model | Low | P0 |
| 1.1.2 | BoundingBox model | Low | P0 |
| 1.2.1 | PDFCoordinateExtractor class | Medium | P0 |
| 1.2.2 | Integrate into extraction pipeline | Medium | P0 |
| 1.3.1 | PDF.js viewer integration | Medium | P1 |
| 1.3.2 | Highlight overlay system | Medium | P1 |
| 1.4.1 | Source coordinate API | Low | P1 |
| 1.4.2 | Page image API | Low | P1 |

**Total Tasks:** 8
**Critical Path:** 1.1.1 → 1.1.2 → 1.2.1 → 1.2.2 → 1.4.1

---

## Phase 2: Strategy Engine (Iron Triangle)

### Overview
Enhance the Strategy Agent to implement the "Iron Triangle" methodology: Win Themes, Discriminators, and Ghosting Language.

### 2.1 Iron Triangle Data Models

**File:** `agents/strategy_agent.py`

**Task 2.1.1: Enhanced WinTheme model**
```python
@dataclass
class WinTheme:
    id: str
    theme_headline: str              # "We reduce risk through..."
    theme_narrative: str             # 2-3 sentence expansion
    discriminators: List[Discriminator]
    proof_points: List[ProofPoint]
    linked_eval_criteria: List[str]  # Section M factor IDs
    page_allocation: float           # Suggested % of volume
    priority: int                    # 1-5 ranking
```

**Dependencies:** None
**Complexity:** Low
**Priority:** P0

---

**Task 2.1.2: Create Discriminator model**
```python
@dataclass
class Discriminator:
    id: str
    category: DiscriminatorType      # TECHNICAL | MANAGEMENT | PAST_PERF | PRICE
    claim: str                       # "Our patented AI reduces review time by 40%"
    evidence_type: str               # "case_study" | "metric" | "testimonial"
    evidence_source: str             # Document reference
    ghosting_angle: Optional[str]    # How this positions against competitors
```

**Dependencies:** None
**Complexity:** Low
**Priority:** P0

---

**Task 2.1.3: Create GhostingStrategy model**
```python
@dataclass
class GhostingStrategy:
    competitor_weakness: str         # What competitor lacks
    our_strength: str                # Our counter-positioning
    language_template: str           # "Unlike solutions that..., our approach..."
    eval_criteria_link: str          # Which Section M factor this addresses
    subtlety_level: int              # 1-5 (1=very subtle, 5=direct comparison)
```

**Dependencies:** None
**Complexity:** Low
**Priority:** P1

---

### 2.2 Strategy Analysis Pipeline

**File:** `agents/strategy_agent.py`

**Task 2.2.1: Implement Section M factor analyzer**
```python
def analyze_evaluation_factors(self, section_m_text: str) -> List[EvalFactor]:
    """
    Parse Section M to extract:
    - Factor names and weights
    - Subfactor hierarchy
    - Evaluation approach (adjectival, point-based, etc.)
    - Key phrases indicating priorities
    """
```

**Dependencies:** Task 2.1.1
**Complexity:** Medium
**Priority:** P0

---

**Task 2.2.2: Implement win theme generator**
```python
def generate_win_themes(
    self,
    eval_factors: List[EvalFactor],
    company_capabilities: List[Capability],
    competitor_intel: Optional[List[CompetitorProfile]]
) -> List[WinTheme]:
    """
    Generate ranked win themes based on:
    1. Evaluation factor weights
    2. Company strengths alignment
    3. Competitor gap analysis
    """
```

**Dependencies:** Task 2.2.1
**Complexity:** High
**Priority:** P0

---

**Task 2.2.3: Implement ghosting language generator**
```python
def generate_ghosting_language(
    self,
    win_themes: List[WinTheme],
    competitors: List[CompetitorProfile]
) -> Dict[str, List[GhostingStrategy]]:
    """
    Generate subtle competitive positioning language
    Maps each win theme to ghosting opportunities
    """
```

**Dependencies:** Task 2.1.3, Task 2.2.2
**Complexity:** Medium
**Priority:** P1

---

### 2.3 Competitive Intelligence Module

**File:** `agents/enhanced_compliance/competitor_analyzer.py` (NEW)

**Task 2.3.1: Create CompetitorAnalyzer class**
```python
class CompetitorAnalyzer:
    def analyze_from_rfp(self, rfp_text: str) -> List[str]:
        """Identify likely competitors from incumbency, past awards"""

    def build_profile(self, competitor_name: str) -> CompetitorProfile:
        """Build competitor profile from public sources"""

    def identify_ghosting_opportunities(
        self,
        our_strengths: List[str],
        competitor: CompetitorProfile
    ) -> List[GhostingStrategy]:
        """Find areas where we can subtly de-position competitor"""
```

**Dependencies:** Task 2.1.3
**Complexity:** High
**Priority:** P2

---

### 2.4 Strategy Output Integration

**File:** `api/main.py`

**Task 2.4.1: Add strategy analysis endpoint**
```python
@app.post("/api/rfp/{rfp_id}/strategy")
async def generate_strategy(rfp_id: str, config: StrategyConfig) -> StrategyOutput:
    """
    Returns:
    - Ranked win themes with discriminators
    - Suggested page allocations per theme
    - Ghosting language library
    - Storyboard outline
    """
```

**Dependencies:** Task 2.2.2
**Complexity:** Medium
**Priority:** P1

---

**Task 2.4.2: Add strategy to outline generator**

**File:** `agents/enhanced_compliance/smart_outline_generator.py`

- Integrate win themes into volume structure
- Add theme annotations to sections
- Include page allocation recommendations

**Dependencies:** Task 2.4.1
**Complexity:** Medium
**Priority:** P1

---

### Phase 2 Deliverables Summary

| Task | Description | Complexity | Priority |
|------|-------------|------------|----------|
| 2.1.1 | Enhanced WinTheme model | Low | P0 |
| 2.1.2 | Discriminator model | Low | P0 |
| 2.1.3 | GhostingStrategy model | Low | P1 |
| 2.2.1 | Section M analyzer | Medium | P0 |
| 2.2.2 | Win theme generator | High | P0 |
| 2.2.3 | Ghosting language generator | Medium | P1 |
| 2.3.1 | CompetitorAnalyzer class | High | P2 |
| 2.4.1 | Strategy API endpoint | Medium | P1 |
| 2.4.2 | Outline integration | Medium | P1 |

**Total Tasks:** 9
**Critical Path:** 2.1.1 → 2.2.1 → 2.2.2 → 2.4.1 → 2.4.2

---

## Phase 3: Drafting Agent (F-B-P Framework)

### Overview
Transform the Drafting Agent into a LangGraph-powered content generator using the Feature-Benefit-Proof (F-B-P) framework.

### 3.1 F-B-P Content Model

**File:** `agents/drafting_agent.py`

**Task 3.1.1: Create FBP content models**
```python
@dataclass
class Feature:
    id: str
    description: str              # What we offer
    technical_detail: str         # How it works
    linked_requirement: str       # Requirement ID this addresses

@dataclass
class Benefit:
    id: str
    statement: str                # Why it matters to customer
    quantified_impact: Optional[str]  # "Reduces cost by 30%"
    eval_criteria_link: str       # Section M factor

@dataclass
class Proof:
    id: str
    proof_type: ProofType         # PAST_PERFORMANCE | CASE_STUDY | METRIC | TESTIMONIAL
    source_document: str          # Citation source
    source_coordinate: SourceCoordinate  # Exact location
    summary: str                  # Brief description

@dataclass
class FBPBlock:
    """A complete Feature-Benefit-Proof content block"""
    feature: Feature
    benefit: Benefit
    proofs: List[Proof]
    generated_narrative: str      # LLM-generated prose
    word_count: int
    compliance_score: float       # How well it addresses requirement
```

**Dependencies:** Phase 1 (SourceCoordinate)
**Complexity:** Medium
**Priority:** P0

---

### 3.2 LangGraph Drafting Workflow

**File:** `agents/drafting_workflow.py` (NEW)

**Task 3.2.1: Create DraftingGraph state schema**
```python
class DraftingState(TypedDict):
    requirement: Dict                    # Current requirement to address
    win_theme: Optional[WinTheme]        # Applicable win theme
    company_library: List[Dict]          # Available evidence
    fbp_blocks: List[FBPBlock]           # Generated content
    draft_text: str                      # Current draft
    revision_count: int                  # Iteration counter
    quality_scores: Dict[str, float]     # Compliance, clarity, etc.
    human_feedback: Optional[str]        # User corrections
```

**Dependencies:** Task 3.1.1
**Complexity:** Medium
**Priority:** P0

---

**Task 3.2.2: Implement drafting graph nodes**
```python
def build_drafting_graph() -> StateGraph:
    builder = StateGraph(DraftingState)

    # Node: Research evidence
    builder.add_node("research", research_node)

    # Node: Generate F-B-P structure
    builder.add_node("structure_fbp", structure_fbp_node)

    # Node: Draft narrative
    builder.add_node("draft", draft_node)

    # Node: Quality check
    builder.add_node("quality_check", quality_check_node)

    # Node: Human review
    builder.add_node("human_review", human_review_node)

    # Node: Revise based on feedback
    builder.add_node("revise", revise_node)

    # Conditional routing
    builder.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {"pass": "human_review", "revise": "revise", "research_more": "research"}
    )
```

**Dependencies:** Task 3.2.1
**Complexity:** High
**Priority:** P0

---

**Task 3.2.3: Implement research node**
```python
def research_node(state: DraftingState) -> DraftingState:
    """
    Query company library for evidence supporting the requirement:
    - Past performance citations
    - Resume highlights
    - Case studies
    - Capability statements

    Uses semantic search to find relevant proof points
    """
```

**Dependencies:** Task 3.2.1
**Complexity:** Medium
**Priority:** P0

---

**Task 3.2.4: Implement F-B-P structuring node**
```python
def structure_fbp_node(state: DraftingState) -> DraftingState:
    """
    Given requirement and evidence, structure into F-B-P blocks:
    1. Identify feature that addresses requirement
    2. Articulate benefit to customer
    3. Link proof points from research
    """
```

**Dependencies:** Task 3.2.3
**Complexity:** Medium
**Priority:** P0

---

**Task 3.2.5: Implement draft generation node**
```python
def draft_node(state: DraftingState) -> DraftingState:
    """
    Generate narrative prose from F-B-P structure:
    - Apply voice style (formal, persuasive, technical)
    - Embed citations inline
    - Respect word count limits
    - Incorporate win theme language
    """
```

**Dependencies:** Task 3.2.4
**Complexity:** High
**Priority:** P0

---

**Task 3.2.6: Implement quality check node**
```python
def quality_check_node(state: DraftingState) -> DraftingState:
    """
    Score draft on multiple dimensions:
    - Compliance: Does it address the requirement?
    - Clarity: Is it readable and well-structured?
    - Citation coverage: Are all claims supported?
    - Word count: Within limits?
    - Theme alignment: Does it reinforce win themes?
    """
```

**Dependencies:** Task 3.2.5
**Complexity:** Medium
**Priority:** P1

---

### 3.3 Human-in-the-Loop Integration

**File:** `agents/drafting_workflow.py`

**Task 3.3.1: Implement human review checkpoint**
```python
def human_review_node(state: DraftingState) -> DraftingState:
    """
    Pause workflow for human review:
    - Present draft with tracked changes
    - Collect feedback/edits
    - Resume workflow with feedback
    """
```

**Dependencies:** Task 3.2.6
**Complexity:** Medium
**Priority:** P1

---

**Task 3.3.2: Implement revision node**
```python
def revise_node(state: DraftingState) -> DraftingState:
    """
    Apply human feedback to draft:
    - Parse edit instructions
    - Regenerate affected sections
    - Preserve approved content
    - Track revision history
    """
```

**Dependencies:** Task 3.3.1
**Complexity:** Medium
**Priority:** P1

---

### 3.4 Company Library Integration

**File:** `agents/enhanced_compliance/company_library.py` (NEW)

**Task 3.4.1: Create CompanyLibrary class**
```python
class CompanyLibrary:
    """
    Vector store of company evidence for proposal reuse:
    - Past performance citations
    - Resume/personnel qualifications
    - Case studies
    - Capability statements
    - Boilerplate sections
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def search(self, query: str, filters: Dict) -> List[Evidence]:
        """Semantic search for relevant evidence"""

    def add_document(self, doc: Document, doc_type: str) -> None:
        """Index new evidence document"""

    def get_past_performance(self, keywords: List[str]) -> List[PastPerformance]:
        """Retrieve relevant past performance citations"""
```

**Dependencies:** None
**Complexity:** High
**Priority:** P1

---

### 3.5 Drafting API Endpoints

**File:** `api/main.py`

**Task 3.5.1: Add drafting workflow endpoint**
```python
@app.post("/api/rfp/{rfp_id}/draft")
async def start_drafting(
    rfp_id: str,
    section_id: str,
    config: DraftingConfig
) -> DraftingSession:
    """
    Start drafting workflow for a section:
    - Returns session ID for status polling
    - Streams progress updates via SSE
    """
```

**Task 3.5.2: Add draft revision endpoint**
```python
@app.post("/api/draft/{session_id}/feedback")
async def submit_feedback(session_id: str, feedback: UserFeedback) -> DraftingState:
    """Submit human feedback to resume drafting workflow"""
```

**Dependencies:** Task 3.2.2
**Complexity:** Medium
**Priority:** P1

---

### Phase 3 Deliverables Summary

| Task | Description | Complexity | Priority |
|------|-------------|------------|----------|
| 3.1.1 | F-B-P content models | Medium | P0 |
| 3.2.1 | DraftingState schema | Medium | P0 |
| 3.2.2 | Drafting graph structure | High | P0 |
| 3.2.3 | Research node | Medium | P0 |
| 3.2.4 | F-B-P structuring node | Medium | P0 |
| 3.2.5 | Draft generation node | High | P0 |
| 3.2.6 | Quality check node | Medium | P1 |
| 3.3.1 | Human review checkpoint | Medium | P1 |
| 3.3.2 | Revision node | Medium | P1 |
| 3.4.1 | CompanyLibrary class | High | P1 |
| 3.5.1 | Drafting API endpoint | Medium | P1 |
| 3.5.2 | Feedback API endpoint | Medium | P1 |

**Total Tasks:** 12
**Critical Path:** 3.1.1 → 3.2.1 → 3.2.2 → 3.2.3 → 3.2.4 → 3.2.5 → 3.5.1

---

## Phase 4: Word Plugin (Office Integration)

### Overview
Create a Microsoft Word add-in that enables proposal writers to work natively in Word while leveraging PropelAI's capabilities.

### 4.1 Office Add-in Foundation

**Directory:** `word-plugin/` (NEW)

**Task 4.1.1: Initialize Office Add-in project**
```bash
# Using Yeoman generator
npm install -g yo generator-office
yo office --projectType taskpane --name PropelAI --host Word
```

Creates:
- `word-plugin/manifest.xml` - Add-in manifest
- `word-plugin/src/taskpane/` - React taskpane UI
- `word-plugin/src/commands/` - Ribbon commands

**Dependencies:** None
**Complexity:** Low
**Priority:** P0

---

**Task 4.1.2: Configure manifest for PropelAI**
```xml
<OfficeApp>
  <Id>propelai-word-addin</Id>
  <DisplayName>PropelAI Proposal Assistant</DisplayName>
  <Hosts>
    <Host Name="Document"/>
  </Hosts>
  <Requirements>
    <Sets>
      <Set Name="WordApi" MinVersion="1.3"/>
    </Sets>
  </Requirements>
  <DefaultSettings>
    <SourceLocation DefaultValue="https://propelai.app/word-plugin/"/>
  </DefaultSettings>
  <Permissions>ReadWriteDocument</Permissions>
</OfficeApp>
```

**Dependencies:** Task 4.1.1
**Complexity:** Low
**Priority:** P0

---

### 4.2 Taskpane UI Components

**File:** `word-plugin/src/taskpane/components/`

**Task 4.2.1: Create RequirementPanel component**
```typescript
interface RequirementPanelProps {
    requirements: Requirement[];
    onRequirementClick: (req: Requirement) => void;
}

const RequirementPanel: React.FC<RequirementPanelProps> = ({ requirements, onRequirementClick }) => {
    // Display requirements grouped by section
    // Click to insert compliance response
    // Show completion status
};
```

**Dependencies:** Task 4.1.1
**Complexity:** Medium
**Priority:** P1

---

**Task 4.2.2: Create ComplianceChecker component**
```typescript
interface ComplianceCheckerProps {
    documentContent: string;
    requirements: Requirement[];
}

const ComplianceChecker: React.FC<ComplianceCheckerProps> = (props) => {
    // Real-time compliance checking
    // Highlight addressed/unaddressed requirements
    // Show compliance score
};
```

**Dependencies:** Task 4.2.1
**Complexity:** High
**Priority:** P1

---

**Task 4.2.3: Create DraftAssistant component**
```typescript
const DraftAssistant: React.FC = () => {
    // Select requirement to address
    // Show F-B-P suggestions
    // Insert generated content at cursor
    // Track what's been drafted
};
```

**Dependencies:** Task 4.2.1
**Complexity:** High
**Priority:** P1

---

### 4.3 Word JavaScript API Integration

**File:** `word-plugin/src/services/wordApi.ts`

**Task 4.3.1: Implement document reader service**
```typescript
class WordDocumentService {
    async getDocumentContent(): Promise<string> {
        return Word.run(async (context) => {
            const body = context.document.body;
            body.load("text");
            await context.sync();
            return body.text;
        });
    }

    async getSelectedText(): Promise<string> {
        return Word.run(async (context) => {
            const selection = context.document.getSelection();
            selection.load("text");
            await context.sync();
            return selection.text;
        });
    }
}
```

**Dependencies:** Task 4.1.1
**Complexity:** Medium
**Priority:** P0

---

**Task 4.3.2: Implement content insertion service**
```typescript
class ContentInsertionService {
    async insertAtCursor(content: string, format: InsertFormat): Promise<void> {
        return Word.run(async (context) => {
            const selection = context.document.getSelection();
            selection.insertText(content, Word.InsertLocation.replace);
            await context.sync();
        });
    }

    async insertWithFormatting(richContent: RichContent): Promise<void> {
        // Insert with headers, bullets, tables
    }

    async insertCitation(citation: Citation): Promise<void> {
        // Insert as footnote or inline reference
    }
}
```

**Dependencies:** Task 4.3.1
**Complexity:** Medium
**Priority:** P1

---

**Task 4.3.3: Implement compliance highlighting**
```typescript
class ComplianceHighlighter {
    async highlightRequirementCoverage(
        requirements: Requirement[],
        coverageMap: Map<string, TextRange>
    ): Promise<void> {
        // Highlight text that addresses requirements
        // Color-code by requirement type
        // Add comments linking to requirement IDs
    }

    async findRequirementMatches(requirement: Requirement): Promise<TextRange[]> {
        return Word.run(async (context) => {
            const results = context.document.body.search(requirement.keywords);
            results.load("text");
            await context.sync();
            return results.items;
        });
    }
}
```

**Dependencies:** Task 4.3.1
**Complexity:** High
**Priority:** P2

---

### 4.4 PropelAI API Client

**File:** `word-plugin/src/services/propelaiClient.ts`

**Task 4.4.1: Create PropelAI API client**
```typescript
class PropelAIClient {
    private baseUrl: string;
    private authToken: string;

    async getRequirements(rfpId: string): Promise<Requirement[]> {}
    async getOutline(rfpId: string): Promise<Outline> {}
    async generateDraft(requirementId: string): Promise<DraftContent> {}
    async checkCompliance(content: string, requirements: Requirement[]): Promise<ComplianceReport> {}
    async getWinThemes(rfpId: string): Promise<WinTheme[]> {}
}
```

**Dependencies:** None
**Complexity:** Medium
**Priority:** P0

---

**Task 4.4.2: Implement authentication flow**
```typescript
class AuthService {
    async login(credentials: Credentials): Promise<AuthToken> {}
    async refreshToken(): Promise<AuthToken> {}
    async getStoredToken(): Promise<AuthToken | null> {
        // Use Office.context.roamingSettings for persistence
    }
}
```

**Dependencies:** Task 4.4.1
**Complexity:** Medium
**Priority:** P1

---

### 4.5 Real-time Sync

**File:** `word-plugin/src/services/syncService.ts`

**Task 4.5.1: Implement document sync service**
```typescript
class DocumentSyncService {
    private websocket: WebSocket;

    async syncWithServer(documentId: string): Promise<void> {
        // Push document changes to PropelAI
        // Pull compliance updates
        // Merge conflicts
    }

    onDocumentChange(callback: (changes: Change[]) => void): void {
        // Listen for Word document changes
        // Debounce and batch updates
    }
}
```

**Dependencies:** Task 4.4.1
**Complexity:** High
**Priority:** P2

---

### 4.6 Ribbon Commands

**File:** `word-plugin/src/commands/commands.ts`

**Task 4.6.1: Implement ribbon commands**
```typescript
// Ribbon button handlers
function onCheckCompliance(event: Office.AddinCommands.Event) {
    // Run compliance check on current document
}

function onInsertOutline(event: Office.AddinCommands.Event) {
    // Insert proposal outline at cursor
}

function onGenerateDraft(event: Office.AddinCommands.Event) {
    // Generate draft for selected requirement
}

function onOpenTaskpane(event: Office.AddinCommands.Event) {
    // Open PropelAI taskpane
}
```

**Dependencies:** Task 4.3.1, Task 4.4.1
**Complexity:** Medium
**Priority:** P1

---

### Phase 4 Deliverables Summary

| Task | Description | Complexity | Priority |
|------|-------------|------------|----------|
| 4.1.1 | Initialize Office Add-in | Low | P0 |
| 4.1.2 | Configure manifest | Low | P0 |
| 4.2.1 | RequirementPanel component | Medium | P1 |
| 4.2.2 | ComplianceChecker component | High | P1 |
| 4.2.3 | DraftAssistant component | High | P1 |
| 4.3.1 | Document reader service | Medium | P0 |
| 4.3.2 | Content insertion service | Medium | P1 |
| 4.3.3 | Compliance highlighting | High | P2 |
| 4.4.1 | PropelAI API client | Medium | P0 |
| 4.4.2 | Authentication flow | Medium | P1 |
| 4.5.1 | Document sync service | High | P2 |
| 4.6.1 | Ribbon commands | Medium | P1 |

**Total Tasks:** 12
**Critical Path:** 4.1.1 → 4.1.2 → 4.3.1 → 4.4.1 → 4.2.1 → 4.6.1

---

## Implementation Priority Matrix

### P0 - Foundation (Must Have)
| Phase | Task | Description |
|-------|------|-------------|
| 1 | 1.1.1 | SourceCoordinate model |
| 1 | 1.1.2 | BoundingBox model |
| 1 | 1.2.1 | PDFCoordinateExtractor |
| 1 | 1.2.2 | Pipeline integration |
| 2 | 2.1.1 | WinTheme model |
| 2 | 2.1.2 | Discriminator model |
| 2 | 2.2.1 | Section M analyzer |
| 2 | 2.2.2 | Win theme generator |
| 3 | 3.1.1 | F-B-P models |
| 3 | 3.2.1-5 | Drafting graph |
| 4 | 4.1.1-2 | Office Add-in init |
| 4 | 4.3.1 | Word document reader |
| 4 | 4.4.1 | API client |

### P1 - Core Features (Should Have)
| Phase | Task | Description |
|-------|------|-------------|
| 1 | 1.3.1-2 | PDF viewer UI |
| 1 | 1.4.1-2 | Coordinate APIs |
| 2 | 2.1.3 | GhostingStrategy model |
| 2 | 2.2.3 | Ghosting generator |
| 2 | 2.4.1-2 | Strategy API |
| 3 | 3.2.6 | Quality check |
| 3 | 3.3.1-2 | Human review |
| 3 | 3.4.1 | Company Library |
| 3 | 3.5.1-2 | Drafting API |
| 4 | 4.2.1-3 | Taskpane UI |
| 4 | 4.3.2 | Content insertion |
| 4 | 4.4.2 | Auth flow |
| 4 | 4.6.1 | Ribbon commands |

### P2 - Advanced Features (Nice to Have)
| Phase | Task | Description |
|-------|------|-------------|
| 2 | 2.3.1 | CompetitorAnalyzer |
| 4 | 4.3.3 | Compliance highlighting |
| 4 | 4.5.1 | Real-time sync |

---

## Dependency Graph

```
PHASE 1 (Trust Gate)
1.1.1 SourceCoordinate ─┬─→ 1.2.1 PDFCoordinateExtractor ─→ 1.2.2 Pipeline ─→ 1.4.1 API
1.1.2 BoundingBox ──────┘                                                      ↓
                                                              1.3.1 PDF.js ─→ 1.3.2 Highlight

PHASE 2 (Strategy Engine)
2.1.1 WinTheme ────→ 2.2.1 Section M Analyzer ─→ 2.2.2 Theme Generator ─→ 2.4.1 API ─→ 2.4.2 Outline
2.1.2 Discriminator ─┘                                       ↓
2.1.3 Ghosting ──────────────────────────────────→ 2.2.3 Ghosting Generator
                                                             ↓
                                               2.3.1 CompetitorAnalyzer (P2)

PHASE 3 (Drafting Agent)
1.1.1 SourceCoordinate ─→ 3.1.1 FBP Models ─→ 3.2.1 State ─→ 3.2.2 Graph ─→ 3.2.3 Research
                                                                            ↓
                                                          3.2.4 Structure ─→ 3.2.5 Draft ─→ 3.2.6 Quality
                                                                                            ↓
                                                                    3.3.1 Human Review ─→ 3.3.2 Revise
                                                                            ↓
                                                                    3.5.1 API ─→ 3.5.2 Feedback
                                                                            ↑
                                                          3.4.1 CompanyLibrary

PHASE 4 (Word Plugin)
4.1.1 Init ─→ 4.1.2 Manifest ─→ 4.3.1 Reader ─→ 4.4.1 API Client ─→ 4.2.1 RequirementPanel
                                       ↓                ↓                     ↓
                               4.3.2 Insertion ←── 4.4.2 Auth         4.2.2 ComplianceChecker
                                       ↓                                      ↓
                               4.3.3 Highlight                       4.2.3 DraftAssistant
                                                                              ↓
                                                                     4.6.1 Ribbon Commands
                                                                              ↓
                                                                     4.5.1 Sync Service
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| PDF coordinate extraction fails for scanned PDFs | High | OCR fallback with Tesseract coordinate mapping |
| LangGraph workflow complexity | Medium | Start with linear flow, add conditionals incrementally |
| Word API limitations | Medium | Feature detection, graceful degradation |
| Real-time sync conflicts | Low | Operational transformation or last-write-wins |
| Competitor data quality | Low | Human review checkpoint for intel |

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] 95% of requirements have valid source coordinates
- [ ] PDF highlight click navigates to correct location
- [ ] Extraction time increase < 20%

### Phase 2 Success Criteria
- [ ] Win themes generated for 100% of RFPs with Section M
- [ ] User approval rate > 70% for generated themes
- [ ] Ghosting language passes compliance review

### Phase 3 Success Criteria
- [ ] Draft generation < 30 seconds per section
- [ ] Citation coverage > 80% of claims
- [ ] Human revision rate < 3 iterations

### Phase 4 Success Criteria
- [ ] Add-in loads in < 3 seconds
- [ ] Compliance check runs in < 5 seconds
- [ ] Content insertion maintains formatting

---

## Recommended Implementation Order

1. **Sprint 1-2: Trust Gate Foundation**
   - Tasks 1.1.1, 1.1.2, 1.2.1, 1.2.2
   - Deliverable: Requirements with source coordinates

2. **Sprint 3-4: Strategy Engine Core**
   - Tasks 2.1.1, 2.1.2, 2.2.1, 2.2.2, 2.4.1
   - Deliverable: Win theme API endpoint

3. **Sprint 5-7: Drafting Agent**
   - Tasks 3.1.1, 3.2.1-5, 3.4.1, 3.5.1
   - Deliverable: F-B-P drafting workflow

4. **Sprint 8-9: Word Plugin MVP**
   - Tasks 4.1.1-2, 4.3.1, 4.4.1, 4.2.1, 4.6.1
   - Deliverable: Basic Word integration

5. **Sprint 10-12: Polish & Integration**
   - All P1 and P2 tasks
   - Full end-to-end testing
   - Performance optimization

---

## Appendix A: Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF coordinate library | pdfplumber | Better bounding box support than pypdf |
| Vector store | Chroma (local) / Pinecone (cloud) | Good LangChain integration |
| Office Add-in framework | Yeoman + React | Official Microsoft tooling |
| Real-time communication | WebSocket | Lower latency than SSE for bidirectional |
| LLM for drafting | Gemini 1.5 Pro | Best balance of quality/cost for long-form |

---

## Appendix B: File Changes Summary

### New Files
- `agents/enhanced_compliance/pdf_coordinate_extractor.py`
- `agents/enhanced_compliance/competitor_analyzer.py`
- `agents/enhanced_compliance/company_library.py`
- `agents/drafting_workflow.py`
- `word-plugin/` (entire directory)

### Modified Files
- `agents/enhanced_compliance/models.py` (add SourceCoordinate, BoundingBox)
- `agents/enhanced_compliance/resilient_extractor.py` (coordinate integration)
- `agents/strategy_agent.py` (enhanced models, new methods)
- `agents/drafting_agent.py` (F-B-P integration)
- `agents/enhanced_compliance/smart_outline_generator.py` (strategy integration)
- `api/main.py` (new endpoints)
- `web/index.html` (PDF viewer)
- `requirements.txt` (pdfplumber, new deps)

---

*Document generated: December 20, 2025*
*PropelAI Implementation Team*
