"""
PropelAI v6.0 Research Agent
The Librarian - provides verified evidence from internal and external sources.
"""

import uuid
import json
from typing import List, Dict, Any, Optional

from swarm.agents.base import BaseAgent, AgentConfig
from swarm.state import ProposalState, Evidence


RESEARCH_SYSTEM_PROMPT = """You are the Research Agent for PropelAI, a fact-finding and evidence retrieval system.

Your mission is to provide VERIFIED, CITABLE evidence for the Drafting Agent.

Sources you access:
1. INTERNAL: Company Library (past proposals, resumes, capabilities, past performance)
2. EXTERNAL: SAM.gov, FPDS.gov, public contract databases

TRACEABILITY IS MANDATORY:
- Every piece of evidence must include source_id, page_number, and exact snippet
- Never fabricate or hallucinate evidence
- If you cannot find evidence, return empty results (let human verify)

Evidence Quality Standards:
- Past Performance: Contract number, value, agency, period of performance
- Resumes: Name, clearance level, years of experience, relevant certifications
- Capabilities: Specific technologies, certifications, methodologies

Output JSON format:
{
  "evidence": [
    {
      "id": "EV-001",
      "source_type": "past_performance",
      "source_id": "contract-abc-123",
      "source_name": "HHS IT Modernization",
      "page_number": null,
      "snippet_text": "Successfully delivered $12M IT modernization...",
      "relevance_score": 0.92,
      "citation_text": "(See Past Performance: HHS IT Modernization, Contract ABC-123)"
    }
  ],
  "search_summary": {
    "query": "...",
    "sources_searched": [...],
    "total_results": 15,
    "top_results_returned": 5
  }
}"""


class ResearchAgent(BaseAgent):
    """
    The Research Agent retrieves verified evidence from multiple sources.

    Key capabilities:
    - Internal RAG search (Company Library)
    - External API integration (SAM.gov, FPDS)
    - Evidence validation and citation formatting
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ResearchAgent",
                model="gemini-1.5-pro",
                temperature=0.1,  # Very low for factual retrieval
                max_tokens=4096,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
                tools=["rag_search", "sam_gov_api", "fpds_api"],
            )
        super().__init__(config)

        # Register tools
        self.register_tool("rag_search", self._rag_search)
        self.register_tool("external_search", self._external_search)

    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        Execute evidence gathering for the proposal.
        """
        requirements = state.get("requirements", [])
        win_themes = state.get("win_themes", [])

        if not requirements and not win_themes:
            self._log_action(state, "observation", "No requirements or themes to research")
            return state

        # Gather evidence for requirements
        self._log_action(state, "thought", "Gathering evidence for requirements")
        req_evidence = await self._gather_requirement_evidence(requirements)

        # Gather evidence for win themes
        self._log_action(state, "thought", "Gathering evidence for win themes")
        theme_evidence = await self._gather_theme_evidence(win_themes)

        # Merge and dedupe
        all_evidence = self._merge_evidence(req_evidence, theme_evidence)

        # Update state
        state["evidence_pool"] = [e.to_dict() if hasattr(e, 'to_dict') else e for e in all_evidence]

        self._log_action(
            state,
            "output",
            f"Gathered {len(all_evidence)} evidence items"
        )

        return state

    async def _gather_requirement_evidence(
        self,
        requirements: List[Dict],
    ) -> List[Evidence]:
        """
        Search for evidence relevant to each requirement.
        """
        evidence = []

        # Group similar requirements to reduce queries
        requirement_groups = self._group_by_topic(requirements)

        for topic, reqs in requirement_groups.items():
            # Build search query from requirements
            query = self._build_search_query(reqs)

            # Search internal library
            internal_results = await self._rag_search(query)
            evidence.extend(internal_results)

        return evidence

    async def _gather_theme_evidence(
        self,
        win_themes: List[Dict],
    ) -> List[Evidence]:
        """
        Search for evidence to support win themes and proof points.
        """
        evidence = []

        for theme in win_themes:
            # Search for each proof point
            proof_points = theme.get("proof_points", [])

            for proof in proof_points:
                results = await self._rag_search(proof)
                evidence.extend(results)

        return evidence

    async def _rag_search(self, query: str, limit: int = 5) -> List[Evidence]:
        """
        Search the Company Library using RAG.
        """
        # In production, this would call the RAG search service
        # from rag.search import RAGSearch, SearchQuery, SearchMode
        # search = RAGSearch()
        # results = await search.search(SearchQuery(query=query, mode=SearchMode.HYBRID))

        # For now, return mock evidence structure
        self._log_action(
            {"agent_trace_log": []},
            "action",
            f"RAG search: {query[:100]}",
            tool_calls=[{"tool": "rag_search", "query": query}],
        )

        # Placeholder - in production, actual RAG results
        return []

    async def _external_search(self, query: str) -> List[Evidence]:
        """
        Search external government databases (SAM.gov, FPDS).
        """
        # In production, this would call external APIs
        # SAM.gov API: https://api.sam.gov/opportunities/v2/search
        # FPDS API: https://www.fpds.gov/ezsearch

        self._log_action(
            {"agent_trace_log": []},
            "action",
            f"External search: {query[:100]}",
            tool_calls=[{"tool": "external_search", "query": query}],
        )

        return []

    def _group_by_topic(self, requirements: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group requirements by similar topics to reduce search queries.
        """
        # Simple grouping by section
        groups = {}
        for req in requirements:
            section = req.get("section_ref", "general")[:1]  # First letter (L, M, C)
            if section not in groups:
                groups[section] = []
            groups[section].append(req)
        return groups

    def _build_search_query(self, requirements: List[Dict]) -> str:
        """
        Build a search query from a group of requirements.
        """
        # Extract key terms from requirements
        texts = [r.get("text", "") for r in requirements[:5]]
        combined = " ".join(texts)

        # Extract important terms (simplified)
        import re
        words = re.findall(r'\b[A-Za-z]{4,}\b', combined)
        unique_words = list(set(words))[:10]

        return " ".join(unique_words)

    def _merge_evidence(
        self,
        *evidence_lists: List[Evidence],
    ) -> List[Evidence]:
        """
        Merge and deduplicate evidence from multiple sources.
        """
        seen_ids = set()
        merged = []

        for evidence_list in evidence_lists:
            for e in evidence_list:
                e_id = e.id if hasattr(e, 'id') else e.get('id', '')
                if e_id not in seen_ids:
                    seen_ids.add(e_id)
                    merged.append(e)

        # Sort by relevance
        merged.sort(
            key=lambda x: x.relevance_score if hasattr(x, 'relevance_score') else x.get('relevance_score', 0),
            reverse=True
        )

        return merged

    async def verify_citation(
        self,
        citation: str,
        state: ProposalState,
    ) -> Optional[Evidence]:
        """
        Verify that a citation exists in our evidence pool.
        """
        evidence_pool = state.get("evidence_pool", [])

        for e in evidence_pool:
            e_id = e.get("id", "") if isinstance(e, dict) else e.id
            if e_id in citation:
                return e

        # Citation not found - search for it
        results = await self._rag_search(citation, limit=1)
        return results[0] if results else None
