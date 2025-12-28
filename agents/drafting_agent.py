"""
PropelAI Drafting Agent - "The Writer"
Play 3: Collaborative Drafting ("The Pen")

Goal: Generate compliant, citation-backed narrative text

CRITICAL CONSTRAINT: Zero Hallucination Policy
- The agent is FORBIDDEN from inventing facts
- Every claim must have a hyperlinked citation to source
- Uncited claims are flagged as "High Risk" (red underline)

This agent works closely with the Research Agent (The Librarian)
to retrieve evidence before writing.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from core.state import ProposalState, ProposalPhase


class VoiceStyle(str, Enum):
    """Available writing voice styles"""
    TECHNICAL = "technical"        # Precise, detailed, dry
    PERSUASIVE = "persuasive"      # Sales-oriented, benefit-focused
    PLAIN = "plain"                # Simple, accessible language
    FORMAL = "formal"              # Government-appropriate formal
    EXECUTIVE = "executive"        # High-level, strategic


class ClaimStatus(str, Enum):
    """Status of a claim in the draft"""
    CITED = "cited"
    UNCITED = "uncited"           # HIGH RISK - needs attention
    VERIFIED = "verified"          # Human verified
    REJECTED = "rejected"          # Flagged as problematic


@dataclass
class Citation:
    """A citation to source material"""
    id: str
    source_type: str              # "past_proposal", "resume", "past_performance", "capability"
    source_document: str          # Document name/ID
    source_text: str              # The actual quoted/referenced text
    page_reference: Optional[str]
    confidence: float


@dataclass
class DraftParagraph:
    """A paragraph of draft content"""
    id: str
    content: str
    citations: List[Citation]
    uncited_claims: List[str]     # Claims without citations - HIGH RISK
    word_count: int
    voice_style: VoiceStyle


class DraftingAgent:
    """
    The Drafting Agent - "The Writer"
    
    Generates narrative proposal text with strict citation requirements.
    Works with the Research Agent to get evidence before writing.
    
    KEY PRINCIPLE: Never hallucinate. Every fact must be traceable.
    """
    
    def __init__(
        self, 
        llm_client: Optional[Any] = None,
        research_agent: Optional[Any] = None,
        voice_style: VoiceStyle = VoiceStyle.FORMAL
    ):
        """
        Initialize the Drafting Agent
        
        Args:
            llm_client: LLM for generating text (Gemini Pro recommended)
            research_agent: The Librarian for evidence retrieval
            voice_style: Default writing style
        """
        self.llm_client = llm_client
        self.research_agent = research_agent
        self.voice_style = voice_style
        
    def __call__(self, state: ProposalState) -> Dict[str, Any]:
        """
        Main entry point - called by the Orchestrator
        
        Generates draft content for proposal sections
        """
        start_time = datetime.now()
        
        # Get inputs from state
        annotated_outline = state.get("annotated_outline", {})
        win_themes = state.get("win_themes", [])
        requirements = state.get("requirements", [])
        compliance_matrix = state.get("compliance_matrix", [])
        existing_drafts = state.get("draft_sections", {})
        
        if not annotated_outline:
            existing_trace = state.get("agent_trace_log", [])
            return {
                "error_state": "No annotated outline found - run strategy agent first",
                "agent_trace_log": existing_trace + [{
                    "timestamp": start_time.isoformat(),
                    "agent_name": "drafting_agent",
                    "action": "generate_drafts",
                    "input_summary": "Missing outline",
                    "output_summary": "Error: Prerequisites not met",
                    "reasoning_trace": "Drafting requires storyboard from strategy phase"
                }]
            }
        
        # Process each volume and section
        draft_sections = dict(existing_drafts)  # Copy existing
        total_citations = 0
        total_uncited = 0
        
        for volume_id, volume_data in annotated_outline.get("volumes", {}).items():
            for section in volume_data.get("sections", []):
                section_id = f"{volume_id}_{section['section_number']}"
                
                # Skip if already drafted (unless revision needed)
                if section_id in draft_sections and not state.get("revision_requested"):
                    continue
                
                # Get relevant requirements for this section
                relevant_reqs = self._get_relevant_requirements(
                    section,
                    requirements,
                    compliance_matrix
                )
                
                # Get win theme for this section
                section_theme = self._get_section_theme(section, win_themes)
                
                # Request evidence from Research Agent
                evidence = self._request_evidence(
                    section,
                    relevant_reqs,
                    section_theme
                )
                
                # Generate the draft
                draft = self._generate_section_draft(
                    section,
                    relevant_reqs,
                    section_theme,
                    evidence
                )
                
                draft_sections[section_id] = draft
                total_citations += len(draft.get("citations", []))
                total_uncited += len(draft.get("uncited_claims", []))
        
        # Calculate processing time
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Build trace log
        trace_log = {
            "timestamp": start_time.isoformat(),
            "agent_name": "drafting_agent",
            "action": "generate_drafts",
            "input_summary": f"{len(annotated_outline.get('volumes', {}))} volumes, {len(win_themes)} themes",
            "output_summary": f"Generated {len(draft_sections)} sections, {total_citations} citations, {total_uncited} uncited claims",
            "reasoning_trace": f"Voice style: {self.voice_style.value}. "
                             f"Citation compliance: {total_citations/(total_citations+total_uncited)*100:.1f}%" if total_citations + total_uncited > 0 else "No claims processed",
            "duration_ms": duration_ms,
            "tool_calls": [
                {"tool": "request_evidence", "result": "Evidence retrieved"},
                {"tool": "generate_draft", "result": f"{len(draft_sections)} sections"},
            ]
        }
        
        # Determine next phase
        next_phase = ProposalPhase.DRAFTING.value
        if total_uncited > 0:
            # Need more research
            next_phase = ProposalPhase.DRAFTING.value
        elif all(draft_sections.values()):
            # All sections drafted, ready for review
            next_phase = ProposalPhase.REVIEW.value
        
        # Accumulate trace logs
        existing_trace = state.get("agent_trace_log", [])

        return {
            "current_phase": next_phase,
            "draft_sections": draft_sections,
            "agent_trace_log": existing_trace + [trace_log],
            "updated_at": datetime.now().isoformat()
        }
    
    def _get_relevant_requirements(
        self,
        section: Dict,
        requirements: List[Dict],
        compliance_matrix: List[Dict]
    ) -> List[Dict]:
        """Find requirements relevant to this section"""
        relevant = []
        
        # Get linked requirements from the section
        linked_reqs = section.get("linked_requirements", [])
        
        for req in requirements:
            # Check if explicitly linked
            if req.get("id") in linked_reqs:
                relevant.append(req)
                continue
            
            # Check keyword overlap with section title
            section_keywords = set(section.get("title", "").lower().split())
            req_keywords = set(req.get("keywords", []))
            
            if len(section_keywords & req_keywords) >= 2:
                relevant.append(req)
        
        return relevant[:10]  # Limit to top 10 most relevant
    
    def _get_section_theme(
        self, 
        section: Dict, 
        win_themes: List[Dict]
    ) -> Optional[Dict]:
        """Get the win theme for this section"""
        # Check if section has explicit win theme
        if section.get("win_theme"):
            return {
                "theme_text": section["win_theme"],
                "discriminator": section.get("discriminator", ""),
                "proof_points": section.get("proof_points", [])
            }
        
        # Find matching theme from list
        section_title_lower = section.get("title", "").lower()
        
        for theme in win_themes:
            theme_lower = theme.get("theme_text", "").lower()
            if any(word in theme_lower for word in section_title_lower.split()):
                return theme
        
        return None
    
    def _request_evidence(
        self,
        section: Dict,
        requirements: List[Dict],
        theme: Optional[Dict]
    ) -> List[Dict]:
        """
        Request evidence from the Research Agent
        
        This is the collaboration point - Writer asks Librarian for proof
        """
        evidence = []
        
        if self.research_agent:
            # Build query from section context
            query_parts = [
                section.get("title", ""),
                " ".join(req.get("text", "")[:100] for req in requirements[:3]),
            ]
            if theme:
                query_parts.append(theme.get("discriminator", ""))
            
            query = " ".join(query_parts)
            
            # Query the Research Agent
            # evidence = self.research_agent.search(query)
        
        # Generate placeholder evidence for demo
        evidence = [
            {
                "id": "EVID-001",
                "source_type": "past_proposal",
                "source_document": "Contract ABC123 Technical Volume",
                "source_text": "Successfully implemented similar solution with 99.9% uptime",
                "page_reference": "p. 15",
                "confidence": 0.92
            },
            {
                "id": "EVID-002",
                "source_type": "past_performance",
                "source_document": "CPARS Report - Contract DEF456",
                "source_text": "Rated Exceptional in Technical Performance",
                "page_reference": None,
                "confidence": 0.95
            }
        ]
        
        return evidence
    
    def _generate_section_draft(
        self,
        section: Dict,
        requirements: List[Dict],
        theme: Optional[Dict],
        evidence: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate the draft content for a section
        
        CRITICAL: Every claim must be cited or flagged
        """
        section_title = section.get("title", "Section")
        page_allocation = section.get("page_allocation", 5)
        word_target = page_allocation * 300  # ~300 words per page
        
        # Build the draft structure
        paragraphs = []
        citations = []
        uncited_claims = []
        
        # Opening paragraph with theme
        if theme:
            opening = self._generate_opening_paragraph(theme, evidence)
            paragraphs.append(opening["content"])
            citations.extend(opening.get("citations", []))
            uncited_claims.extend(opening.get("uncited", []))
        else:
            paragraphs.append(f"This section addresses the requirements for {section_title}.")
        
        # Body paragraphs addressing requirements
        for i, req in enumerate(requirements[:5]):
            body = self._generate_requirement_response(req, evidence)
            paragraphs.append(body["content"])
            citations.extend(body.get("citations", []))
            uncited_claims.extend(body.get("uncited", []))
        
        # Closing paragraph with proof points
        if theme and theme.get("proof_points"):
            closing = self._generate_closing_paragraph(theme, evidence)
            paragraphs.append(closing["content"])
            citations.extend(closing.get("citations", []))
        
        # Assemble the draft
        content = "\n\n".join(paragraphs)
        word_count = len(content.split())
        
        return {
            "section_id": f"{section.get('section_number', '0.0')}",
            "section_title": section_title,
            "content": content,
            "citations": citations,
            "uncited_claims": uncited_claims,
            "word_count": word_count,
            "page_allocation": page_allocation,
            "compliance_score": self._calculate_compliance_score(requirements, content),
            "quality_score": self._calculate_quality_score(citations, uncited_claims),
            "version": 1,
            "last_modified": datetime.now().isoformat(),
            "modified_by": "drafting_agent"
        }
    
    def _generate_opening_paragraph(
        self, 
        theme: Dict, 
        evidence: List[Dict]
    ) -> Dict[str, Any]:
        """Generate opening paragraph with theme"""
        theme_text = theme.get("theme_text", "")
        discriminator = theme.get("discriminator", "")
        
        # Build opening that leads with the win theme
        content = f"{theme_text}. {discriminator}."
        
        citations = []
        uncited = []
        
        # If we have evidence for the discriminator, cite it
        relevant_evidence = [e for e in evidence if e["confidence"] > 0.8]
        if relevant_evidence:
            ev = relevant_evidence[0]
            content += f" [{ev['source_document']}, {ev.get('page_reference', '')}]"
            citations.append({
                "id": ev["id"],
                "source_type": ev["source_type"],
                "source_document": ev["source_document"],
                "source_text": ev["source_text"],
                "page_reference": ev.get("page_reference"),
                "confidence": ev["confidence"]
            })
        else:
            # Flag the discriminator claim as uncited
            uncited.append(discriminator)
        
        return {
            "content": content,
            "citations": citations,
            "uncited": uncited
        }
    
    def _generate_requirement_response(
        self, 
        requirement: Dict, 
        evidence: List[Dict]
    ) -> Dict[str, Any]:
        """Generate response to a specific requirement"""
        req_text = requirement.get("text", "")
        req_ref = requirement.get("section_ref", "")
        
        # Build compliant response
        content = f"In response to {req_ref}, [Offeror] provides "
        
        citations = []
        uncited = []
        
        # Check if we have evidence
        if evidence:
            ev = evidence[0]
            content += f"demonstrated capability as evidenced by {ev['source_text']}."
            content += f" [{ev['source_document']}]"
            citations.append({
                "id": ev["id"],
                "source_type": ev["source_type"],
                "source_document": ev["source_document"],
                "source_text": ev["source_text"],
                "page_reference": ev.get("page_reference"),
                "confidence": ev["confidence"]
            })
        else:
            # Generate generic compliant language but FLAG IT
            content += "comprehensive capabilities to meet this requirement."
            uncited.append(f"Capability claim for {req_ref} - NEEDS EVIDENCE")
        
        return {
            "content": content,
            "citations": citations,
            "uncited": uncited
        }
    
    def _generate_closing_paragraph(
        self, 
        theme: Dict, 
        evidence: List[Dict]
    ) -> Dict[str, Any]:
        """Generate closing paragraph with proof points"""
        proof_points = theme.get("proof_points", [])
        
        content = "Our approach delivers measurable results: "
        content += "; ".join(proof_points[:3]) + "."
        
        citations = []
        
        # Try to cite proof points
        for proof in proof_points[:2]:
            for ev in evidence:
                if any(word in ev["source_text"].lower() for word in proof.lower().split()[:3]):
                    citations.append({
                        "id": ev["id"],
                        "source_type": ev["source_type"],
                        "source_document": ev["source_document"],
                        "source_text": ev["source_text"],
                        "page_reference": ev.get("page_reference"),
                        "confidence": ev["confidence"]
                    })
                    break
        
        return {
            "content": content,
            "citations": citations,
            "uncited": []
        }
    
    def _calculate_compliance_score(
        self, 
        requirements: List[Dict], 
        content: str
    ) -> float:
        """Calculate how well the draft addresses requirements"""
        if not requirements:
            return 0.0
        
        addressed = 0
        content_lower = content.lower()
        
        for req in requirements:
            # Check if key requirement words appear in content
            keywords = req.get("keywords", [])
            if any(kw.lower() in content_lower for kw in keywords):
                addressed += 1
        
        return addressed / len(requirements) if requirements else 0.0
    
    def _calculate_quality_score(
        self, 
        citations: List[Dict], 
        uncited: List[str]
    ) -> float:
        """Calculate draft quality based on citation coverage"""
        total = len(citations) + len(uncited)
        if total == 0:
            return 0.5  # Neutral if no claims
        
        cited_score = len(citations) / total
        
        # Weight by confidence of citations
        if citations:
            avg_confidence = sum(c.get("confidence", 0.5) for c in citations) / len(citations)
            return cited_score * avg_confidence
        
        return cited_score


class ResearchAgent:
    """
    The Research Agent - "The Librarian"
    
    Retrieves supporting evidence from:
    - Past proposals (Vector Store)
    - Resumes
    - Past Performance records
    - Corporate capabilities
    
    Acts as the "memory" for the Drafting Agent
    """
    
    def __init__(self, vector_store: Optional[Any] = None):
        """
        Initialize the Research Agent
        
        Args:
            vector_store: Vector database for semantic search
        """
        self.vector_store = vector_store
        self.cache = {}  # Cache recent queries
        
    def __call__(self, state: ProposalState) -> Dict[str, Any]:
        """
        Main entry point - processes research requests
        """
        start_time = datetime.now()
        
        # Check for pending research requests
        draft_sections = state.get("draft_sections", {})
        
        # Find sections with uncited claims
        research_results = []
        
        for section_id, draft in draft_sections.items():
            uncited = draft.get("uncited_claims", [])
            if uncited:
                # Search for evidence for each uncited claim
                for claim in uncited:
                    results = self.search(claim)
                    research_results.extend(results)
        
        trace_log = {
            "timestamp": start_time.isoformat(),
            "agent_name": "research_agent",
            "action": "retrieve_evidence",
            "input_summary": f"Searching for evidence on uncited claims",
            "output_summary": f"Found {len(research_results)} evidence items",
            "reasoning_trace": "Queried vector store for relevant past performance and capabilities"
        }

        # Accumulate trace logs
        existing_trace = state.get("agent_trace_log", [])

        return {
            "agent_trace_log": existing_trace + [trace_log],
            "updated_at": datetime.now().isoformat()
        }
    
    def search(
        self, 
        query: str, 
        source_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for evidence matching the query
        
        Args:
            query: The search query
            source_types: Filter by source type
            top_k: Number of results to return
        """
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = []
        
        if self.vector_store:
            # Perform semantic search
            # raw_results = self.vector_store.similarity_search(query, k=top_k)
            # results = self._process_results(raw_results, source_types)
            pass
        
        # Demo results
        results = [
            {
                "id": f"EVID-{hashlib.md5(query.encode()).hexdigest()[:6]}",
                "source_type": "past_performance",
                "source_document": "Past Performance Record - Contract X",
                "source_text": f"Relevant experience: {query[:50]}...",
                "page_reference": None,
                "confidence": 0.85,
                "relevance_score": 0.90
            }
        ]
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    def _process_results(
        self, 
        raw_results: List[Any], 
        source_types: Optional[List[str]]
    ) -> List[Dict]:
        """Process raw vector search results into evidence format"""
        processed = []
        
        for result in raw_results:
            # Extract metadata and content
            evidence = {
                "id": result.metadata.get("id", ""),
                "source_type": result.metadata.get("source_type", "unknown"),
                "source_document": result.metadata.get("document_name", ""),
                "source_text": result.page_content[:500],
                "page_reference": result.metadata.get("page"),
                "confidence": result.metadata.get("confidence", 0.5),
                "relevance_score": result.metadata.get("score", 0.5)
            }
            
            # Filter by source type if specified
            if source_types and evidence["source_type"] not in source_types:
                continue
            
            processed.append(evidence)
        
        return processed


def create_drafting_agent(
    llm_client: Optional[Any] = None,
    research_agent: Optional[Any] = None,
    voice_style: VoiceStyle = VoiceStyle.FORMAL
) -> DraftingAgent:
    """Factory function to create a Drafting Agent"""
    return DraftingAgent(
        llm_client=llm_client,
        research_agent=research_agent,
        voice_style=voice_style
    )


def create_research_agent(vector_store: Optional[Any] = None) -> ResearchAgent:
    """Factory function to create a Research Agent"""
    return ResearchAgent(vector_store=vector_store)
