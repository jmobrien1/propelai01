"""
PropelAI Cycle 5: Cross-Reference Resolver
Builds graph edges between requirements across documents

Creates the Requirements Graph with:
- Section C → Section M links (performance → evaluation)
- Section C → Section L links (performance → proposal instructions)
- CDRL → Section C links (deliverables → requirements)
- Research Outline → Labor links (NIH-specific)
- Amendment changes → Original requirements
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

from .models import (
    RequirementNode, RequirementType, DocumentType, 
    ParsedDocument, RFPBundle, ConfidenceLevel
)


class CrossReferenceResolver:
    """
    Build relationships between requirements across documents
    
    Creates a Requirements Graph where:
    - Nodes = Requirements
    - Edges = Relationships (references, evaluates, instructs, etc.)
    """
    
    # Patterns for matching requirements across sections
    IRON_TRIANGLE_KEYWORDS = {
        # Technical keywords that should appear in C, L, and M
        "approach": ["technical approach", "management approach", "staffing approach"],
        "past_performance": ["past performance", "relevant experience", "similar work"],
        "cost": ["cost", "price", "pricing", "budget"],
        "key_personnel": ["key personnel", "staffing", "personnel qualifications", "resumes"],
        "schedule": ["schedule", "timeline", "milestones", "deliverables"],
        "quality": ["quality", "quality assurance", "quality control", "QASP"],
        "security": ["security", "clearance", "classified", "CUI", "CMMC"],
    }
    
    # Section relationships
    SECTION_RELATIONSHIPS = {
        ("C", "L"): "instructed_by",      # C requirements → L instructions
        ("C", "M"): "evaluated_by",        # C requirements → M criteria
        ("L", "M"): "evaluated_by",        # L instructions → M criteria
        ("B", "C"): "clin_reference",      # B CLINs → C requirements
    }
    
    def __init__(self):
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)
    
    def resolve_references(
        self,
        requirements: List[RequirementNode],
        parsed_docs: Dict[str, ParsedDocument],
    ) -> Dict[str, RequirementNode]:
        """
        Build the requirements graph by resolving cross-references
        
        Args:
            requirements: List of extracted requirements
            parsed_docs: Dict of parsed documents
            
        Returns:
            Dict mapping requirement ID to RequirementNode (the graph)
        """
        # Build graph as dict
        graph: Dict[str, RequirementNode] = {req.id: req for req in requirements}
        
        # Step 1: Build keyword index for matching
        self._build_keyword_index(requirements)
        
        # Step 2: Resolve explicit references (e.g., "See Section C.3.1")
        self._resolve_explicit_references(graph)
        
        # Step 3: Build Iron Triangle links (C ↔ L ↔ M)
        self._build_iron_triangle_links(graph)
        
        # Step 4: Link CDRL/deliverables to requirements
        self._link_deliverables(graph, parsed_docs)
        
        # Step 5: Link evaluation criteria to requirements
        self._link_evaluation_criteria(graph)
        
        # Step 6: NIH-specific: Link Research Outlines to labor/CLINs
        self._link_research_outlines(graph, parsed_docs)
        
        return graph
    
    def _build_keyword_index(self, requirements: List[RequirementNode]):
        """Build inverted index of keywords → requirement IDs"""
        self.keyword_index.clear()
        
        for req in requirements:
            # Index by keywords
            for keyword in req.keywords:
                self.keyword_index[keyword.lower()].append(req.id)
            
            # Index by entities
            for entity in req.entities:
                self.keyword_index[entity.lower()].append(req.id)
            
            # Index by significant words
            words = re.findall(r'\b[a-z]{4,}\b', req.text.lower())
            for word in set(words):
                self.keyword_index[word].append(req.id)
    
    def _resolve_explicit_references(self, graph: Dict[str, RequirementNode]):
        """Resolve explicit cross-references mentioned in requirement text"""
        for req_id, req in graph.items():
            for ref in req.references_to:
                # Try to find the referenced requirement
                target_id = self._find_requirement_by_reference(ref, graph)
                if target_id and target_id != req_id:
                    # Add bidirectional edge
                    if target_id not in req.references_to:
                        req.references_to.append(target_id)
                    
                    target = graph.get(target_id)
                    if target and req_id not in target.referenced_by:
                        target.referenced_by.append(req_id)
    
    def _find_requirement_by_reference(
        self, 
        reference: str, 
        graph: Dict[str, RequirementNode]
    ) -> Optional[str]:
        """Find a requirement that matches the reference string"""
        reference_lower = reference.lower()
        
        for req_id, req in graph.items():
            # Check if reference matches section ID
            if req.source and req.source.section_id:
                if reference_lower in req.source.section_id.lower():
                    return req_id
            
            # Check if reference matches requirement ID
            if reference_lower in req_id.lower():
                return req_id
        
        return None
    
    def _build_iron_triangle_links(self, graph: Dict[str, RequirementNode]):
        """
        Build links between Section C, L, and M requirements
        
        The "Iron Triangle":
        - C = What contractor must DO (performance requirements)
        - L = What offeror must WRITE (proposal instructions)
        - M = How government will EVALUATE (evaluation criteria)
        
        A well-linked proposal addresses all three for each topic.
        """
        # Group requirements by section
        by_section: Dict[str, List[RequirementNode]] = defaultdict(list)
        
        for req in graph.values():
            if req.source and req.source.section_id:
                section = req.source.section_id[0].upper()  # First letter: C, L, M
                by_section[section].append(req)
            elif req.source and req.source.document_type:
                # Infer section from document type
                if req.source.document_type == DocumentType.STATEMENT_OF_WORK:
                    by_section["C"].append(req)
        
        # Link C requirements to L instructions
        for c_req in by_section.get("C", []):
            for l_req in by_section.get("L", []):
                similarity = self._compute_topic_similarity(c_req, l_req)
                if similarity > 0.3:  # Threshold for linking
                    c_req.instructed_by.append(l_req.id)
                    l_req.references_to.append(c_req.id)
        
        # Link C requirements to M criteria
        for c_req in by_section.get("C", []):
            for m_req in by_section.get("M", []):
                similarity = self._compute_topic_similarity(c_req, m_req)
                if similarity > 0.3:
                    c_req.evaluated_by.append(m_req.id)
                    m_req.references_to.append(c_req.id)
        
        # Link L instructions to M criteria
        for l_req in by_section.get("L", []):
            for m_req in by_section.get("M", []):
                similarity = self._compute_topic_similarity(l_req, m_req)
                if similarity > 0.3:
                    l_req.evaluated_by.append(m_req.id)
    
    def _compute_topic_similarity(
        self, 
        req1: RequirementNode, 
        req2: RequirementNode
    ) -> float:
        """Compute topic similarity between two requirements"""
        # Keyword overlap
        keywords1 = set(k.lower() for k in req1.keywords)
        keywords2 = set(k.lower() for k in req2.keywords)
        
        if not keywords1 or not keywords2:
            # Fall back to text similarity
            return self._text_similarity(req1.text, req2.text)
        
        overlap = keywords1 & keywords2
        union = keywords1 | keywords2
        
        keyword_sim = len(overlap) / len(union) if union else 0
        
        # Iron Triangle topic matching
        topic_match = 0
        for topic, phrases in self.IRON_TRIANGLE_KEYWORDS.items():
            in_req1 = any(p in req1.text.lower() for p in phrases)
            in_req2 = any(p in req2.text.lower() for p in phrases)
            if in_req1 and in_req2:
                topic_match = 0.4  # Boost for shared topic
                break
        
        return keyword_sim * 0.6 + topic_match
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity between texts"""
        words1 = set(re.findall(r'\b[a-z]{4,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-z]{4,}\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        overlap = words1 & words2
        union = words1 | words2
        
        return len(overlap) / len(union)
    
    def _link_deliverables(
        self, 
        graph: Dict[str, RequirementNode],
        parsed_docs: Dict[str, ParsedDocument],
    ):
        """Link CDRL/deliverable requirements to performance requirements"""
        deliverables = [
            req for req in graph.values() 
            if req.requirement_type == RequirementType.DELIVERABLE
        ]
        
        performance_reqs = [
            req for req in graph.values()
            if req.requirement_type == RequirementType.PERFORMANCE
        ]
        
        for deliv in deliverables:
            # Find performance requirements that this deliverable addresses
            deliv_keywords = set(deliv.keywords)
            
            for perf in performance_reqs:
                perf_keywords = set(perf.keywords)
                overlap = deliv_keywords & perf_keywords
                
                if len(overlap) >= 2:  # At least 2 shared keywords
                    deliv.references_to.append(perf.id)
                    perf.deliverable_for = deliv.id
    
    def _link_evaluation_criteria(self, graph: Dict[str, RequirementNode]):
        """Link evaluation criteria to the requirements they evaluate"""
        eval_criteria = [
            req for req in graph.values()
            if req.requirement_type == RequirementType.EVALUATION_CRITERION
        ]
        
        # Common evaluation factor patterns
        factor_patterns = {
            "technical": r"technical\s+(?:approach|capability|understanding)",
            "management": r"management\s+(?:approach|plan|capability)",
            "past_performance": r"past\s+performance|relevant\s+experience",
            "price": r"price|cost|pricing",
            "staffing": r"key\s+personnel|staffing|qualifications",
        }
        
        for eval_req in eval_criteria:
            eval_text = eval_req.text.lower()
            
            # Identify which factor this is
            matched_factor = None
            for factor, pattern in factor_patterns.items():
                if re.search(pattern, eval_text):
                    matched_factor = factor
                    break
            
            if not matched_factor:
                continue
            
            # Find requirements related to this factor
            for req_id, req in graph.items():
                if req.requirement_type == RequirementType.EVALUATION_CRITERION:
                    continue
                
                req_text = req.text.lower()
                
                # Check if requirement relates to this evaluation factor
                if re.search(factor_patterns[matched_factor], req_text):
                    if eval_req.id not in req.evaluated_by:
                        req.evaluated_by.append(eval_req.id)
                    if req_id not in eval_req.references_to:
                        eval_req.references_to.append(req_id)
    
    def _link_research_outlines(
        self,
        graph: Dict[str, RequirementNode],
        parsed_docs: Dict[str, ParsedDocument],
    ):
        """
        NIH-specific: Link Research Outlines to related requirements
        
        Research Outlines (RO I, RO II, etc.) define specific research projects
        that need to be linked to:
        - Labor hour requirements
        - CLIN references
        - Deliverables
        """
        # Find requirements from Research Outline documents
        ro_requirements = []
        for req in graph.values():
            if req.source and req.source.document_type == DocumentType.RESEARCH_OUTLINE:
                ro_requirements.append(req)
            elif req.research_outline:
                ro_requirements.append(req)
        
        # Find labor/CLIN requirements
        labor_reqs = [
            req for req in graph.values()
            if req.requirement_type == RequirementType.LABOR_REQUIREMENT
        ]
        
        # Look for RO references in labor requirements
        for labor_req in labor_reqs:
            # Check for "RO I", "RO II", "Research Outline I" patterns
            ro_matches = re.findall(r"RO\s*([IVX]+|\d+)|Research\s+Outline\s+([IVX]+|\d+)", 
                                    labor_req.text, re.IGNORECASE)
            
            for match in ro_matches:
                ro_id = match[0] or match[1]
                
                # Find the corresponding RO requirement
                for ro_req in ro_requirements:
                    if ro_id in (ro_req.research_outline or ""):
                        labor_req.research_outline = ro_id
                        labor_req.references_to.append(ro_req.id)
                        ro_req.referenced_by.append(labor_req.id)
                        break
    
    def get_graph_statistics(self, graph: Dict[str, RequirementNode]) -> Dict[str, Any]:
        """Calculate statistics about the requirements graph"""
        stats = {
            "total_requirements": len(graph),
            "by_type": defaultdict(int),
            "by_section": defaultdict(int),
            "by_confidence": defaultdict(int),
            "total_edges": 0,
            "avg_edges_per_node": 0.0,
            "orphan_requirements": 0,
            "fully_linked_requirements": 0,
        }
        
        total_edges = 0
        
        for req in graph.values():
            # Count by type
            stats["by_type"][req.requirement_type.value] += 1
            
            # Count by section
            if req.source and req.source.section_id:
                section = req.source.section_id[0] if req.source.section_id else "UNKNOWN"
                stats["by_section"][section] += 1
            
            # Count by confidence
            stats["by_confidence"][req.confidence.value] += 1
            
            # Count edges
            edge_count = (
                len(req.references_to) + 
                len(req.referenced_by) + 
                len(req.evaluated_by) + 
                len(req.instructed_by)
            )
            total_edges += edge_count
            
            if edge_count == 0:
                stats["orphan_requirements"] += 1
            elif edge_count >= 3:
                stats["fully_linked_requirements"] += 1
        
        stats["total_edges"] = total_edges // 2  # Edges are bidirectional
        stats["avg_edges_per_node"] = total_edges / len(graph) if graph else 0
        
        # Convert defaultdicts to regular dicts
        stats["by_type"] = dict(stats["by_type"])
        stats["by_section"] = dict(stats["by_section"])
        stats["by_confidence"] = dict(stats["by_confidence"])
        
        return stats
    
    def find_gaps(self, graph: Dict[str, RequirementNode]) -> Dict[str, List[str]]:
        """
        Identify gaps in the requirements coverage
        
        Returns:
            Dict with gap categories and list of requirement IDs
        """
        gaps = {
            "performance_without_evaluation": [],    # C reqs not linked to M
            "performance_without_instruction": [],   # C reqs not linked to L
            "evaluation_without_requirements": [],   # M criteria with no C/L links
            "deliverables_without_requirements": [], # Deliverables not linked to reqs
            "orphan_requirements": [],               # No links at all
        }
        
        for req_id, req in graph.items():
            # Check for orphans
            has_links = (
                req.references_to or 
                req.referenced_by or 
                req.evaluated_by or 
                req.instructed_by
            )
            if not has_links:
                gaps["orphan_requirements"].append(req_id)
            
            # Check Iron Triangle gaps
            if req.requirement_type == RequirementType.PERFORMANCE:
                if not req.evaluated_by:
                    gaps["performance_without_evaluation"].append(req_id)
                if not req.instructed_by:
                    gaps["performance_without_instruction"].append(req_id)
            
            elif req.requirement_type == RequirementType.EVALUATION_CRITERION:
                if not req.references_to:
                    gaps["evaluation_without_requirements"].append(req_id)
            
            elif req.requirement_type == RequirementType.DELIVERABLE:
                if not req.references_to:
                    gaps["deliverables_without_requirements"].append(req_id)
        
        return gaps
