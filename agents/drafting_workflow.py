"""
PropelAI v4.0: LangGraph Drafting Workflow

This module implements the agentic drafting workflow using LangGraph.
The workflow orchestrates content generation using the F-B-P (Feature-Benefit-Proof)
framework with human-in-the-loop review capabilities.

Key Components:
- DraftingState: TypedDict for workflow state management
- Research Node: Queries company library for evidence
- Structure Node: Builds F-B-P blocks from requirements
- Draft Node: Generates narrative prose
- Quality Check Node: Scores draft on multiple dimensions
- Human Review Node: Pause point for human feedback
- Revise Node: Incorporates feedback into draft

Usage:
    from agents.drafting_workflow import build_drafting_graph, run_drafting_workflow

    graph = build_drafting_graph()
    result = run_drafting_workflow(
        requirement={"id": "REQ-001", "text": "..."},
        win_theme={"headline": "..."},
        company_library=company_lib
    )
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any, Literal, TypedDict, Annotated
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

# LangGraph imports (graceful fallback if not available)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    MemorySaver = None

# PostgreSQL checkpointer for workflow persistence
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_CHECKPOINTER_AVAILABLE = True
except ImportError:
    POSTGRES_CHECKPOINTER_AVAILABLE = False
    PostgresSaver = None

# Import F-B-P models from drafting agent
try:
    from .drafting_agent import (
        Feature, Benefit, Proof, FBPBlock, ProofType,
    )
    FBP_MODELS_AVAILABLE = True
except ImportError:
    FBP_MODELS_AVAILABLE = False
    # Define minimal fallback classes
    class ProofType(str, Enum):
        PAST_PERFORMANCE = "past_performance"
        CASE_STUDY = "case_study"
        METRIC = "metric"
        TESTIMONIAL = "testimonial"
        CERTIFICATION = "certification"
        METHODOLOGY = "methodology"

    @dataclass
    class Feature:
        description: str
        technical_detail: str = ""
        linked_requirement: str = ""

    @dataclass
    class Benefit:
        statement: str
        quantified_impact: Optional[str] = None
        eval_criteria_link: Optional[str] = None

    @dataclass
    class Proof:
        proof_type: ProofType
        source_document: str
        summary: str
        confidence: float = 1.0

    @dataclass
    class FBPBlock:
        feature: Feature
        benefit: Benefit
        proofs: List[Proof] = field(default_factory=list)
        generated_narrative: str = ""
        compliance_score: float = 0.0


# ============================================================================
# Drafting State
# ============================================================================

class QualityScores(TypedDict, total=False):
    """Quality assessment scores for draft content"""
    compliance: float      # Does it address the requirement?
    clarity: float         # Is it readable and well-structured?
    citation_coverage: float  # Are claims supported by evidence?
    word_count_ratio: float   # Actual/target word count
    theme_alignment: float    # Does it reinforce win themes?
    overall: float         # Weighted average


class DraftingState(TypedDict, total=False):
    """
    State schema for the drafting workflow.

    This state is passed between nodes and accumulates data
    as the workflow progresses.
    """
    # Input requirements
    requirement_id: str
    requirement_text: str
    requirement_section: str

    # Win theme context
    win_theme_headline: str
    win_theme_narrative: str
    discriminators: List[Dict]

    # Evidence from company library
    evidence: List[Dict]
    past_performance: List[Dict]
    key_personnel: List[Dict]

    # F-B-P content blocks
    fbp_blocks: List[Dict]

    # Generated draft
    draft_text: str
    draft_word_count: int
    target_word_count: int

    # Quality metrics
    quality_scores: QualityScores

    # Human feedback loop
    revision_count: int
    human_feedback: Optional[str]
    approved: bool

    # Workflow metadata
    workflow_id: str
    started_at: str
    completed_at: Optional[str]
    current_node: str
    error: Optional[str]


# ============================================================================
# LLM Integration
# ============================================================================

def get_llm_client():
    """Get LLM client for content generation"""
    # Try Gemini first (best for long-form content)
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            return ("gemini", genai.GenerativeModel("gemini-1.5-pro"))
        except Exception:
            pass

    # Fall back to Claude
    claude_key = os.environ.get("ANTHROPIC_API_KEY")
    if claude_key:
        try:
            import anthropic
            return ("claude", anthropic.Anthropic(api_key=claude_key))
        except Exception:
            pass

    # Fall back to OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            return ("openai", openai.OpenAI(api_key=openai_key))
        except Exception:
            pass

    return (None, None)


def generate_with_llm(prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Generate content using available LLM"""
    provider, client = get_llm_client()

    if not client:
        return None

    try:
        if provider == "gemini":
            response = client.generate_content(prompt)
            return response.text

        elif provider == "claude":
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif provider == "openai":
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"LLM generation error ({provider}): {e}")
        return None

    return None


# ============================================================================
# Workflow Nodes
# ============================================================================

def research_node(state: DraftingState) -> DraftingState:
    """
    Research Node: Query company library for evidence supporting the requirement.

    Searches for:
    - Past performance citations
    - Resume highlights
    - Case studies
    - Capability statements

    Uses semantic search to find relevant proof points.
    """
    state["current_node"] = "research"

    requirement_text = state.get("requirement_text", "")

    # Extract keywords for search
    keywords = _extract_keywords(requirement_text)

    # Mock evidence (in production, this queries the company library)
    evidence = []
    past_performance = []
    key_personnel = []

    # Try to import and use actual company library
    try:
        from agents.enhanced_compliance.company_library import CompanyLibrary
        import tempfile
        library = CompanyLibrary(str(tempfile.gettempdir()) + "/propelai_company_library")

        # Search for relevant content
        search_results = library.search(" ".join(keywords))

        for result in search_results[:10]:
            content_type = result.get("type")
            content = result.get("content", {})

            if content_type == "capability":
                evidence.append({
                    "type": "capability",
                    "name": content.get("name", ""),
                    "description": content.get("description", ""),
                    "relevance_score": result.get("score", 0)
                })
            elif content_type == "past_performance":
                past_performance.append({
                    "project_name": content.get("project_name", ""),
                    "client": content.get("client", ""),
                    "description": content.get("description", ""),
                    "outcomes": content.get("outcomes", [])
                })
            elif content_type == "key_personnel":
                key_personnel.append({
                    "name": content.get("name", ""),
                    "title": content.get("title", ""),
                    "summary": content.get("summary", ""),
                    "skills": content.get("skills", [])
                })
            elif content_type == "differentiator":
                evidence.append({
                    "type": "differentiator",
                    "title": content.get("title", ""),
                    "description": content.get("description", ""),
                    "relevance_score": result.get("score", 0)
                })
    except Exception as e:
        # Library not available - use mock data
        pass

    state["evidence"] = evidence
    state["past_performance"] = past_performance
    state["key_personnel"] = key_personnel

    return state


def structure_fbp_node(state: DraftingState) -> DraftingState:
    """
    Structure F-B-P Node: Create Feature-Benefit-Proof blocks from requirement and evidence.

    For each requirement:
    1. Identify the feature that addresses it
    2. Articulate the benefit to the customer
    3. Link proof points from research phase
    """
    state["current_node"] = "structure_fbp"

    requirement_text = state.get("requirement_text", "")
    evidence = state.get("evidence", [])
    past_performance = state.get("past_performance", [])
    win_theme = state.get("win_theme_headline", "")

    # Build F-B-P blocks
    fbp_blocks = []

    # Try LLM-powered structuring
    if requirement_text:
        prompt = f"""Analyze this RFP requirement and structure a Feature-Benefit-Proof response.

REQUIREMENT:
{requirement_text}

WIN THEME TO REINFORCE:
{win_theme}

AVAILABLE EVIDENCE:
{json.dumps(evidence[:5], indent=2)}

PAST PERFORMANCE:
{json.dumps(past_performance[:3], indent=2)}

Create a structured F-B-P block with:
1. FEATURE: What specific capability/approach addresses this requirement?
2. BENEFIT: What value does this provide to the customer? (quantify if possible)
3. PROOF: What evidence supports this? (reference past performance or capabilities)

Return as JSON:
{{
    "feature": {{"description": "...", "technical_detail": "..."}},
    "benefit": {{"statement": "...", "quantified_impact": "..."}},
    "proofs": [{{"type": "past_performance|case_study|metric|certification", "summary": "..."}}]
}}
"""

        llm_response = generate_with_llm(prompt)
        if llm_response:
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{[\s\S]*\}', llm_response)
                if json_match:
                    fbp_data = json.loads(json_match.group())
                    fbp_blocks.append(fbp_data)
            except Exception:
                pass

    # Fallback: Create template-based F-B-P block
    if not fbp_blocks:
        fbp_blocks.append({
            "feature": {
                "description": f"Our approach addresses: {requirement_text[:100]}...",
                "technical_detail": "Leveraging proven methodologies and expert personnel"
            },
            "benefit": {
                "statement": "Reduces risk and ensures compliance with requirements",
                "quantified_impact": None
            },
            "proofs": [
                {
                    "type": "past_performance",
                    "summary": past_performance[0].get("description", "Demonstrated success in similar engagements")[:200] if past_performance else "Proven track record"
                }
            ] if past_performance or True else []
        })

    state["fbp_blocks"] = fbp_blocks
    return state


def draft_node(state: DraftingState) -> DraftingState:
    """
    Draft Node: Generate narrative prose from F-B-P structure.

    Applies:
    - Formal, persuasive voice
    - Inline citations
    - Word count limits
    - Win theme language reinforcement
    """
    state["current_node"] = "draft"

    fbp_blocks = state.get("fbp_blocks", [])
    requirement_text = state.get("requirement_text", "")
    win_theme = state.get("win_theme_headline", "")
    target_words = state.get("target_word_count", 250)

    # Try LLM-powered draft generation
    if fbp_blocks:
        prompt = f"""Generate a formal proposal response using the Feature-Benefit-Proof structure.

REQUIREMENT TO ADDRESS:
{requirement_text}

F-B-P STRUCTURE:
{json.dumps(fbp_blocks, indent=2)}

WIN THEME TO WEAVE IN:
{win_theme}

INSTRUCTIONS:
1. Write in formal, persuasive proposal voice
2. Lead with understanding of the requirement
3. Present the feature/approach clearly
4. Emphasize benefits with quantification where possible
5. Cite proof points inline using [Reference X] format
6. Target approximately {target_words} words
7. Do NOT use first-person pronouns (use "the team" or company name)

Generate the proposal response paragraph(s):
"""

        draft_text = generate_with_llm(prompt, max_tokens=1500)

        if not draft_text:
            # Fallback to template
            draft_text = _generate_template_draft(fbp_blocks, requirement_text, win_theme)
    else:
        draft_text = _generate_template_draft(fbp_blocks, requirement_text, win_theme)

    state["draft_text"] = draft_text
    state["draft_word_count"] = len(draft_text.split())

    return state


def quality_check_node(state: DraftingState) -> DraftingState:
    """
    Quality Check Node: Score draft on multiple dimensions.

    Evaluates:
    - Compliance: Does it address the requirement?
    - Clarity: Is it readable and well-structured?
    - Citation coverage: Are all claims supported?
    - Word count: Within limits?
    - Theme alignment: Does it reinforce win themes?
    """
    state["current_node"] = "quality_check"

    draft_text = state.get("draft_text", "")
    requirement_text = state.get("requirement_text", "")
    target_words = state.get("target_word_count", 250)
    actual_words = state.get("draft_word_count", 0)
    win_theme = state.get("win_theme_headline", "")

    scores: QualityScores = {}

    # Word count ratio
    if target_words > 0:
        scores["word_count_ratio"] = min(actual_words / target_words, 1.5)
    else:
        scores["word_count_ratio"] = 1.0

    # Basic compliance check (keyword overlap)
    req_keywords = set(_extract_keywords(requirement_text))
    draft_keywords = set(_extract_keywords(draft_text))
    if req_keywords:
        overlap = len(req_keywords & draft_keywords) / len(req_keywords)
        scores["compliance"] = min(overlap * 1.5, 1.0)  # Scale up
    else:
        scores["compliance"] = 0.5

    # Citation coverage (look for reference markers)
    import re
    citations = re.findall(r'\[.*?\]|\([A-Z].*?\)', draft_text)
    scores["citation_coverage"] = min(len(citations) * 0.25, 1.0)

    # Clarity (sentence length, paragraph structure)
    sentences = draft_text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    # Optimal sentence length is 15-25 words
    if 15 <= avg_sentence_length <= 25:
        scores["clarity"] = 1.0
    elif 10 <= avg_sentence_length <= 30:
        scores["clarity"] = 0.8
    else:
        scores["clarity"] = 0.6

    # Theme alignment
    theme_keywords = set(_extract_keywords(win_theme))
    if theme_keywords:
        theme_overlap = len(theme_keywords & draft_keywords) / len(theme_keywords)
        scores["theme_alignment"] = min(theme_overlap * 2, 1.0)
    else:
        scores["theme_alignment"] = 0.5

    # Overall score (weighted average)
    weights = {
        "compliance": 0.35,
        "clarity": 0.20,
        "citation_coverage": 0.20,
        "word_count_ratio": 0.10,
        "theme_alignment": 0.15
    }

    overall = sum(scores.get(k, 0.5) * v for k, v in weights.items())
    scores["overall"] = round(overall, 3)

    state["quality_scores"] = scores

    return state


def human_review_node(state: DraftingState) -> DraftingState:
    """
    Human Review Node: Checkpoint for human feedback.

    This node processes human feedback that was provided when the workflow
    was resumed. The workflow pauses BEFORE this node (via interrupt_before)
    and resumes with human feedback in the state.

    Workflow:
    1. Graph pauses before this node (interrupt_before=["human_review"])
    2. Human reviews draft via API/UI
    3. Human submits feedback via resume_workflow()
    4. This node processes the feedback and sets approved status
    """
    state["current_node"] = "human_review"

    # Check if human feedback was provided (via resume_workflow)
    human_feedback = state.get("human_feedback")

    if human_feedback:
        # Parse feedback to determine action
        feedback_lower = human_feedback.lower().strip()

        if feedback_lower in ("approved", "approve", "lgtm", "ok", "accept", "good"):
            state["approved"] = True
        elif feedback_lower.startswith("reject"):
            # Rejection with optional reason
            state["approved"] = False
        else:
            # Any other feedback is considered revision request
            state["approved"] = False
    else:
        # No feedback yet - workflow should have paused before reaching here
        # If we got here without feedback, it's likely running without interrupt
        # Log warning and don't approve
        print("Warning: human_review_node reached without feedback. "
              "Ensure interrupt_before_human_review=True in build_drafting_graph()")
        state["approved"] = False

    return state


def revise_node(state: DraftingState) -> DraftingState:
    """
    Revise Node: Apply human feedback to improve draft.

    Incorporates:
    - Edit instructions
    - Specific text changes
    - Additional requirements
    """
    state["current_node"] = "revise"

    feedback = state.get("human_feedback")
    draft_text = state.get("draft_text", "")
    revision_count = state.get("revision_count", 0)

    if feedback and not state.get("approved", False):
        # Try LLM-powered revision
        prompt = f"""Revise this proposal draft based on the feedback.

CURRENT DRAFT:
{draft_text}

FEEDBACK:
{feedback}

Instructions:
1. Address all feedback points
2. Maintain the F-B-P structure
3. Keep the same approximate length
4. Preserve citations and references

Revised draft:
"""

        revised_text = generate_with_llm(prompt, max_tokens=1500)

        if revised_text:
            state["draft_text"] = revised_text
            state["draft_word_count"] = len(revised_text.split())

    state["revision_count"] = revision_count + 1

    return state


# ============================================================================
# Routing Logic
# ============================================================================

def route_after_quality(state: DraftingState) -> Literal["human_review", "revise", "research"]:
    """
    Route after quality check based on scores and revision count.
    """
    quality_scores = state.get("quality_scores", {})
    overall = quality_scores.get("overall", 0)
    revision_count = state.get("revision_count", 0)

    # If quality is high, go to human review
    if overall >= 0.75:
        return "human_review"

    # If citation coverage is very low, research more
    if quality_scores.get("citation_coverage", 0) < 0.25 and revision_count < 2:
        return "research"

    # Otherwise, revise
    if revision_count < 3:  # Max 3 revision cycles
        return "revise"

    # Fallback to human review after max revisions
    return "human_review"


def route_after_human_review(state: DraftingState) -> Literal["revise", "end"]:
    """
    Route after human review based on approval status.
    """
    if state.get("approved", False):
        return "end"

    revision_count = state.get("revision_count", 0)
    if revision_count >= 5:  # Hard limit on revisions
        return "end"

    return "revise"


# ============================================================================
# Checkpointer Factory
# ============================================================================

def get_checkpointer(connection_string: Optional[str] = None):
    """
    Get a checkpointer for workflow state persistence.

    Priority:
    1. PostgreSQL (if connection string provided and available)
    2. MemorySaver (in-memory fallback)

    Args:
        connection_string: PostgreSQL connection string (optional)

    Returns:
        A checkpointer instance for LangGraph
    """
    # Try PostgreSQL first
    if connection_string and POSTGRES_CHECKPOINTER_AVAILABLE:
        try:
            # Get connection string from environment if not provided
            conn_str = connection_string or os.environ.get("DATABASE_URL")
            if conn_str:
                # PostgresSaver expects a connection pool
                checkpointer = PostgresSaver.from_conn_string(conn_str)
                print("Using PostgreSQL checkpointer for workflow persistence")
                return checkpointer
        except Exception as e:
            print(f"PostgreSQL checkpointer failed, falling back to memory: {e}")

    # Fallback to memory
    if MemorySaver:
        print("Using in-memory checkpointer (state will not persist across restarts)")
        return MemorySaver()

    return None


# Global checkpointer instance (lazy initialization)
_checkpointer = None


def get_or_create_checkpointer() -> Optional[Any]:
    """Get or create the global checkpointer instance."""
    global _checkpointer
    if _checkpointer is None:
        db_url = os.environ.get("DATABASE_URL")
        _checkpointer = get_checkpointer(db_url)
    return _checkpointer


# ============================================================================
# Graph Builder
# ============================================================================

def build_drafting_graph(
    checkpointer: Optional[Any] = None,
    interrupt_before_human_review: bool = True
) -> Optional["StateGraph"]:
    """
    Build the LangGraph drafting workflow.

    Args:
        checkpointer: Optional checkpointer for state persistence.
                     If None, uses get_or_create_checkpointer().
        interrupt_before_human_review: If True, workflow pauses at human_review
                                       for actual human input (proper HITL).

    Returns:
        StateGraph configured for proposal drafting, or None if LangGraph unavailable
    """
    if not LANGGRAPH_AVAILABLE:
        print("Warning: LangGraph not available. Install with: pip install langgraph")
        return None

    # Create state graph
    builder = StateGraph(DraftingState)

    # Add nodes
    builder.add_node("research", research_node)
    builder.add_node("structure_fbp", structure_fbp_node)
    builder.add_node("draft", draft_node)
    builder.add_node("quality_check", quality_check_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("revise", revise_node)

    # Set entry point
    builder.set_entry_point("research")

    # Add edges (linear flow)
    builder.add_edge("research", "structure_fbp")
    builder.add_edge("structure_fbp", "draft")
    builder.add_edge("draft", "quality_check")

    # Conditional routing after quality check
    builder.add_conditional_edges(
        "quality_check",
        route_after_quality,
        {
            "human_review": "human_review",
            "revise": "revise",
            "research": "research"
        }
    )

    # Conditional routing after human review
    builder.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "revise": "revise",
            "end": END
        }
    )

    # Revise always goes back to quality check
    builder.add_edge("revise", "quality_check")

    # Get checkpointer
    if checkpointer is None:
        checkpointer = get_or_create_checkpointer()

    # Compile with checkpointer and optional interrupt
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    # Enable HITL: interrupt BEFORE human_review so the workflow pauses
    # and waits for actual human input
    if interrupt_before_human_review:
        compile_kwargs["interrupt_before"] = ["human_review"]

    return builder.compile(**compile_kwargs)


# ============================================================================
# Workflow Runner
# ============================================================================

def run_drafting_workflow(
    requirement: Dict[str, Any],
    win_theme: Optional[Dict[str, Any]] = None,
    target_word_count: int = 250,
    company_library: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the complete drafting workflow for a requirement.

    Args:
        requirement: Dict with 'id', 'text', 'section' fields
        win_theme: Optional dict with 'headline', 'narrative', 'discriminators'
        target_word_count: Target length for the draft
        company_library: Optional CompanyLibrary instance

    Returns:
        Dict with final draft, quality scores, and metadata
    """
    # Build initial state
    initial_state: DraftingState = {
        "workflow_id": str(uuid.uuid4())[:8],
        "requirement_id": requirement.get("id", ""),
        "requirement_text": requirement.get("text", ""),
        "requirement_section": requirement.get("section", ""),
        "win_theme_headline": (win_theme or {}).get("headline", ""),
        "win_theme_narrative": (win_theme or {}).get("narrative", ""),
        "discriminators": (win_theme or {}).get("discriminators", []),
        "target_word_count": target_word_count,
        "evidence": [],
        "past_performance": [],
        "key_personnel": [],
        "fbp_blocks": [],
        "draft_text": "",
        "draft_word_count": 0,
        "quality_scores": {},
        "revision_count": 0,
        "human_feedback": None,
        "approved": False,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "current_node": "start",
        "error": None
    }

    # Build graph
    graph = build_drafting_graph()

    if graph:
        try:
            # Run the graph
            final_state = graph.invoke(initial_state)
            final_state["completed_at"] = datetime.now().isoformat()
            return final_state
        except Exception as e:
            initial_state["error"] = str(e)
            return initial_state
    else:
        # Fallback: run nodes sequentially without LangGraph
        state = initial_state
        try:
            state = research_node(state)
            state = structure_fbp_node(state)
            state = draft_node(state)
            state = quality_check_node(state)
            state["approved"] = True  # Auto-approve in fallback mode
            state["completed_at"] = datetime.now().isoformat()
        except Exception as e:
            state["error"] = str(e)

        return state


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_keywords(text: str) -> List[str]:
    """Extract keywords from text for matching"""
    import re
    # Remove common words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them',
        'we', 'our', 'us', 'you', 'your', 'i', 'my', 'me', 'he', 'she',
        'his', 'her', 'him', 'which', 'who', 'whom', 'what', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'any'
    }

    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Filter and return unique keywords
    keywords = [w for w in words if w not in stop_words]
    return list(dict.fromkeys(keywords))[:30]  # Dedupe and limit


def _generate_template_draft(
    fbp_blocks: List[Dict],
    requirement_text: str,
    win_theme: str
) -> str:
    """Generate a template-based draft when LLM is unavailable"""

    if not fbp_blocks:
        return f"""[COMPANY NAME] fully understands and is prepared to meet this requirement.

{requirement_text[:150]}...

Our approach leverages proven methodologies and experienced personnel to deliver compliant solutions. We have successfully delivered similar capabilities on previous engagements, demonstrating our ability to meet and exceed customer expectations.

{f'This aligns with our commitment to {win_theme}.' if win_theme else ''}
"""

    fbp = fbp_blocks[0]
    feature = fbp.get("feature", {})
    benefit = fbp.get("benefit", {})
    proofs = fbp.get("proofs", [])

    draft_parts = []

    # Opening - acknowledge requirement
    draft_parts.append(
        f"[COMPANY NAME] understands the importance of this requirement and offers a proven approach. "
    )

    # Feature
    if feature.get("description"):
        draft_parts.append(feature["description"] + ". ")
        if feature.get("technical_detail"):
            draft_parts.append(feature["technical_detail"] + ". ")

    # Benefit
    if benefit.get("statement"):
        draft_parts.append(f"This approach {benefit['statement'].lower()}. ")
        if benefit.get("quantified_impact"):
            draft_parts.append(f"({benefit['quantified_impact']}) ")

    # Proof
    if proofs:
        proof = proofs[0]
        draft_parts.append(
            f"Our capability is demonstrated by {proof.get('summary', 'proven past performance')}. "
        )

    # Theme tie-in
    if win_theme:
        draft_parts.append(f"This reinforces our commitment to {win_theme}.")

    return "".join(draft_parts)


# ============================================================================
# Workflow Management Functions
# ============================================================================

def get_workflow_status(
    workflow_id: str,
    graph: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the current status of a drafting workflow.

    Args:
        workflow_id: The workflow ID (thread_id for LangGraph)
        graph: Optional compiled graph. If None, builds a new one.

    Returns:
        Dict with workflow status, current node, and state snapshot
    """
    if graph is None:
        graph = build_drafting_graph()

    if graph is None:
        return None

    try:
        # Get current state from checkpointer
        config = {"configurable": {"thread_id": workflow_id}}
        state = graph.get_state(config)

        if state is None or state.values is None:
            return None

        values = state.values
        return {
            "workflow_id": workflow_id,
            "status": "paused" if state.next else "completed",
            "current_node": values.get("current_node", "unknown"),
            "next_nodes": list(state.next) if state.next else [],
            "draft_preview": (values.get("draft_text", "")[:500] + "...")
                            if values.get("draft_text") else None,
            "quality_scores": values.get("quality_scores", {}),
            "revision_count": values.get("revision_count", 0),
            "approved": values.get("approved", False),
            "started_at": values.get("started_at"),
            "error": values.get("error"),
        }
    except Exception as e:
        return {"workflow_id": workflow_id, "status": "error", "error": str(e)}


def resume_workflow(
    workflow_id: str,
    human_feedback: str,
    graph: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Resume a paused drafting workflow with human feedback.

    Args:
        workflow_id: The workflow ID (thread_id for LangGraph)
        human_feedback: Feedback from human reviewer. Options:
            - "approve" / "approved" / "lgtm": Accept the draft
            - "reject: <reason>": Reject the draft
            - Any other text: Request revision with this feedback

        graph: Optional compiled graph. If None, builds a new one.

    Returns:
        Dict with updated workflow state after resuming

    Example:
        # Approve a draft
        resume_workflow("wf-123", "approved")

        # Request revision
        resume_workflow("wf-123", "Please add more specific metrics")
    """
    if graph is None:
        graph = build_drafting_graph()

    if graph is None:
        return {"error": "LangGraph not available"}

    try:
        config = {"configurable": {"thread_id": workflow_id}}

        # Get current state
        current_state = graph.get_state(config)
        if current_state is None or current_state.values is None:
            return {"error": f"Workflow {workflow_id} not found"}

        # Check if workflow is paused
        if not current_state.next:
            return {"error": "Workflow is not paused", "status": "completed"}

        # Update state with human feedback
        graph.update_state(
            config,
            {"human_feedback": human_feedback},
            as_node="quality_check"  # Inject as if coming from quality_check
        )

        # Resume the workflow (it will continue from the interrupt point)
        result = None
        for event in graph.stream(None, config):
            result = event

        # Get final state
        final_state = graph.get_state(config)
        if final_state and final_state.values:
            values = final_state.values
            return {
                "workflow_id": workflow_id,
                "status": "paused" if final_state.next else "completed",
                "current_node": values.get("current_node"),
                "approved": values.get("approved", False),
                "draft_text": values.get("draft_text", ""),
                "quality_scores": values.get("quality_scores", {}),
                "revision_count": values.get("revision_count", 0),
                "completed_at": values.get("completed_at"),
            }

        return {"workflow_id": workflow_id, "status": "unknown"}

    except Exception as e:
        return {"error": str(e), "workflow_id": workflow_id}


def list_pending_workflows(graph: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    List all workflows waiting for human review.

    Note: This requires a PostgreSQL checkpointer to work across restarts.
    With MemorySaver, only workflows from the current session are visible.

    Args:
        graph: Optional compiled graph.

    Returns:
        List of workflow status dicts for paused workflows
    """
    # This would query the checkpoints table for paused workflows
    # For now, return empty list as this requires database query
    # TODO: Implement when PostgreSQL is configured
    return []


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "DraftingState",
    "QualityScores",
    "build_drafting_graph",
    "run_drafting_workflow",
    "research_node",
    "structure_fbp_node",
    "draft_node",
    "quality_check_node",
    "human_review_node",
    "revise_node",
    "LANGGRAPH_AVAILABLE",
    # Checkpointing
    "get_checkpointer",
    "get_or_create_checkpointer",
    "POSTGRES_CHECKPOINTER_AVAILABLE",
    # Workflow management
    "get_workflow_status",
    "resume_workflow",
    "list_pending_workflows",
]
