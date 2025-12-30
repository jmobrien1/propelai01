"""
PropelAI v5.0: Requirements Graph with NetworkX DAG
Iron Triangle dependency mapping and orphan detection

Implements FR-2.2: Dependency Mapping with NetworkX DAG
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from .models import RequirementNode, RequirementType, DocumentType


class EdgeType(Enum):
    """Types of relationships between requirements"""
    REFERENCES = "references"           # Generic reference
    INSTRUCTS = "instructs"             # L instructs how to address C
    EVALUATES = "evaluates"             # M evaluates C or L content
    DELIVERS = "delivers"               # Deliverable fulfills requirement
    PARENT_OF = "parent_of"             # Hierarchical parent
    AMENDS = "amends"                   # Amendment modifies original
    CONFLICTS = "conflicts"             # Conflicting requirements


class NodeSection(Enum):
    """Iron Triangle section classification"""
    SECTION_C = "C"     # Performance/SOW - What contractor must DO
    SECTION_L = "L"     # Instructions - What offeror must WRITE
    SECTION_M = "M"     # Evaluation - How government will EVALUATE
    OTHER = "OTHER"     # Attachments, forms, etc.


@dataclass
class GraphEdge:
    """An edge in the requirements graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrphanReport:
    """Report of orphaned requirements"""
    orphan_id: str
    section: NodeSection
    requirement_type: RequirementType
    reason: str
    suggestion: str


@dataclass
class GraphAnalysis:
    """Complete analysis of the requirements graph"""
    total_nodes: int
    total_edges: int
    orphan_count: int
    orphans: List[OrphanReport]
    section_counts: Dict[str, int]
    edge_type_counts: Dict[str, int]
    connected_components: int
    iron_triangle_coverage: Dict[str, float]
    critical_path: List[str]
    dependency_depth: int


class RequirementsDAG:
    """
    Directed Acyclic Graph for requirements dependency mapping.

    Implements the Iron Triangle logic:
    - Section C (Performance) requirements
    - Section L (Instructions) guide how to address C
    - Section M (Evaluation) criteria score the response

    A complete requirement should have:
    - C linked to L (how to write about it)
    - C linked to M (how it will be scored)
    - L linked to M (what the instruction addresses)
    """

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for RequirementsDAG. "
                "Install with: pip install networkx>=3.0"
            )

        self.graph: nx.DiGraph = nx.DiGraph()
        self._section_index: Dict[NodeSection, Set[str]] = defaultdict(set)
        self._type_index: Dict[RequirementType, Set[str]] = defaultdict(set)

    def add_requirement(self, req: RequirementNode) -> None:
        """Add a requirement node to the graph"""
        section = self._classify_section(req)

        self.graph.add_node(
            req.id,
            text=req.text[:200],  # Truncate for memory
            section=section.value,
            req_type=req.requirement_type.value,
            confidence=req.confidence.value,
            source_page=req.source.page_number if req.source else None,
            source_doc=req.source.document_name if req.source else None,
        )

        # Update indexes
        self._section_index[section].add(req.id)
        self._type_index[req.requirement_type].add(req.id)

    def add_requirements(self, requirements: List[RequirementNode]) -> None:
        """Add multiple requirements to the graph"""
        for req in requirements:
            self.add_requirement(req)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        **metadata
    ) -> bool:
        """
        Add a directed edge between requirements.

        Returns False if adding edge would create a cycle.
        """
        if source_id not in self.graph or target_id not in self.graph:
            return False

        # Check for cycle before adding
        if self._would_create_cycle(source_id, target_id):
            return False

        self.graph.add_edge(
            source_id,
            target_id,
            edge_type=edge_type.value,
            weight=weight,
            **metadata
        )
        return True

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge source -> target would create a cycle"""
        if source == target:
            return True

        # Check if there's already a path from target to source
        try:
            return nx.has_path(self.graph, target, source)
        except nx.NetworkXError:
            return False

    def _classify_section(self, req: RequirementNode) -> NodeSection:
        """Classify requirement into Iron Triangle section"""
        # Check source section ID first
        if req.source and req.source.section_id:
            section_id = req.source.section_id.upper()
            if section_id.startswith('C') or section_id.startswith('SOW'):
                return NodeSection.SECTION_C
            elif section_id.startswith('L'):
                return NodeSection.SECTION_L
            elif section_id.startswith('M'):
                return NodeSection.SECTION_M

        # Infer from document type
        if req.source and req.source.document_type:
            if req.source.document_type == DocumentType.STATEMENT_OF_WORK:
                return NodeSection.SECTION_C

        # Infer from requirement type
        if req.requirement_type == RequirementType.PERFORMANCE:
            return NodeSection.SECTION_C
        elif req.requirement_type == RequirementType.PROPOSAL_INSTRUCTION:
            return NodeSection.SECTION_L
        elif req.requirement_type == RequirementType.EVALUATION_CRITERION:
            return NodeSection.SECTION_M

        return NodeSection.OTHER

    def build_iron_triangle_edges(
        self,
        similarity_threshold: float = 0.3
    ) -> int:
        """
        Automatically build Iron Triangle edges based on content similarity.

        Links:
        - C -> L (performance instructed by proposal instructions)
        - C -> M (performance evaluated by criteria)
        - L -> M (instructions evaluated by criteria)

        Returns number of edges created.
        """
        edges_created = 0

        c_nodes = list(self._section_index[NodeSection.SECTION_C])
        l_nodes = list(self._section_index[NodeSection.SECTION_L])
        m_nodes = list(self._section_index[NodeSection.SECTION_M])

        # C -> L links (L instructs C)
        for c_id in c_nodes:
            c_text = self.graph.nodes[c_id].get('text', '')
            for l_id in l_nodes:
                l_text = self.graph.nodes[l_id].get('text', '')
                sim = self._text_similarity(c_text, l_text)
                if sim >= similarity_threshold:
                    if self.add_edge(l_id, c_id, EdgeType.INSTRUCTS, weight=sim):
                        edges_created += 1

        # C -> M links (M evaluates C)
        for c_id in c_nodes:
            c_text = self.graph.nodes[c_id].get('text', '')
            for m_id in m_nodes:
                m_text = self.graph.nodes[m_id].get('text', '')
                sim = self._text_similarity(c_text, m_text)
                if sim >= similarity_threshold:
                    if self.add_edge(m_id, c_id, EdgeType.EVALUATES, weight=sim):
                        edges_created += 1

        # L -> M links (M evaluates L)
        for l_id in l_nodes:
            l_text = self.graph.nodes[l_id].get('text', '')
            for m_id in m_nodes:
                m_text = self.graph.nodes[m_id].get('text', '')
                sim = self._text_similarity(l_text, m_text)
                if sim >= similarity_threshold:
                    if self.add_edge(m_id, l_id, EdgeType.EVALUATES, weight=sim):
                        edges_created += 1

        return edges_created

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity between texts"""
        import re
        words1 = set(re.findall(r'\b[a-z]{4,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-z]{4,}\b', text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def find_orphans(self) -> List[OrphanReport]:
        """
        Find orphaned requirements that lack proper Iron Triangle links.

        Orphan types:
        - C without L: Performance requirement with no instruction link
        - C without M: Performance requirement with no evaluation link
        - L without M: Instruction with no evaluation link
        - M without C/L: Evaluation criterion that doesn't reference anything
        """
        orphans = []

        for node_id in self.graph.nodes():
            node = self.graph.nodes[node_id]
            section = NodeSection(node.get('section', 'OTHER'))
            req_type = RequirementType(node.get('req_type', 'performance'))

            in_edges = list(self.graph.in_edges(node_id, data=True))
            out_edges = list(self.graph.out_edges(node_id, data=True))

            # Check Section C requirements
            if section == NodeSection.SECTION_C:
                has_l_link = any(
                    self.graph.nodes[e[0]].get('section') == 'L'
                    for e in in_edges
                )
                has_m_link = any(
                    self.graph.nodes[e[0]].get('section') == 'M'
                    for e in in_edges
                )

                if not has_l_link:
                    orphans.append(OrphanReport(
                        orphan_id=node_id,
                        section=section,
                        requirement_type=req_type,
                        reason="Section C requirement has no Section L instruction link",
                        suggestion="Add proposal instruction that addresses this requirement"
                    ))

                if not has_m_link:
                    orphans.append(OrphanReport(
                        orphan_id=node_id,
                        section=section,
                        requirement_type=req_type,
                        reason="Section C requirement is not linked to any evaluation criterion",
                        suggestion="Verify this requirement is covered by Section M criteria"
                    ))

            # Check Section L requirements
            elif section == NodeSection.SECTION_L:
                has_m_link = any(
                    self.graph.nodes[e[0]].get('section') == 'M'
                    for e in in_edges
                )

                if not has_m_link:
                    orphans.append(OrphanReport(
                        orphan_id=node_id,
                        section=section,
                        requirement_type=req_type,
                        reason="Section L instruction is not linked to evaluation criteria",
                        suggestion="Verify this instruction aligns with Section M factors"
                    ))

            # Check Section M requirements
            elif section == NodeSection.SECTION_M:
                targets_anything = len(out_edges) > 0

                if not targets_anything:
                    orphans.append(OrphanReport(
                        orphan_id=node_id,
                        section=section,
                        requirement_type=req_type,
                        reason="Evaluation criterion does not link to any requirements",
                        suggestion="Map this criterion to relevant C/L requirements"
                    ))

            # Completely isolated nodes
            if len(in_edges) == 0 and len(out_edges) == 0:
                if node_id not in [o.orphan_id for o in orphans]:
                    orphans.append(OrphanReport(
                        orphan_id=node_id,
                        section=section,
                        requirement_type=req_type,
                        reason="Requirement is completely isolated (no links)",
                        suggestion="Review and link to related requirements"
                    ))

        return orphans

    def get_topological_order(self) -> List[str]:
        """Get requirements in topological order (dependency order)"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, return nodes in arbitrary order
            return list(self.graph.nodes())

    def get_critical_path(self) -> List[str]:
        """Find the longest dependency chain (critical path)"""
        if len(self.graph) == 0:
            return []

        try:
            return nx.dag_longest_path(self.graph)
        except nx.NetworkXUnfeasible:
            return []

    def get_dependency_depth(self) -> int:
        """Get the maximum depth of dependencies"""
        path = self.get_critical_path()
        return len(path) if path else 0

    def get_requirements_for_section(
        self,
        section: NodeSection
    ) -> List[str]:
        """Get all requirement IDs for a section"""
        return list(self._section_index.get(section, set()))

    def get_linked_requirements(
        self,
        req_id: str,
        direction: str = "both"
    ) -> Dict[str, List[str]]:
        """
        Get requirements linked to the given requirement.

        Args:
            req_id: Requirement ID to query
            direction: "in", "out", or "both"

        Returns:
            Dict with edge types as keys and lists of requirement IDs as values
        """
        if req_id not in self.graph:
            return {}

        result = defaultdict(list)

        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(req_id, data=True):
                edge_type = data.get('edge_type', 'references')
                result[f"in_{edge_type}"].append(source)

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(req_id, data=True):
                edge_type = data.get('edge_type', 'references')
                result[f"out_{edge_type}"].append(target)

        return dict(result)

    def analyze(self) -> GraphAnalysis:
        """Perform comprehensive graph analysis"""
        orphans = self.find_orphans()

        # Count edges by type
        edge_type_counts = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'references')
            edge_type_counts[edge_type] += 1

        # Section counts
        section_counts = {
            section.value: len(nodes)
            for section, nodes in self._section_index.items()
        }

        # Connected components
        try:
            components = nx.number_weakly_connected_components(self.graph)
        except:
            components = 1

        # Iron Triangle coverage
        iron_triangle_coverage = self._calculate_iron_triangle_coverage()

        return GraphAnalysis(
            total_nodes=len(self.graph.nodes()),
            total_edges=len(self.graph.edges()),
            orphan_count=len(orphans),
            orphans=orphans,
            section_counts=section_counts,
            edge_type_counts=dict(edge_type_counts),
            connected_components=components,
            iron_triangle_coverage=iron_triangle_coverage,
            critical_path=self.get_critical_path(),
            dependency_depth=self.get_dependency_depth(),
        )

    def _calculate_iron_triangle_coverage(self) -> Dict[str, float]:
        """Calculate coverage percentages for Iron Triangle"""
        c_nodes = self._section_index[NodeSection.SECTION_C]
        l_nodes = self._section_index[NodeSection.SECTION_L]
        m_nodes = self._section_index[NodeSection.SECTION_M]

        coverage = {
            "c_with_l": 0.0,
            "c_with_m": 0.0,
            "l_with_m": 0.0,
            "overall": 0.0,
        }

        if not c_nodes:
            return coverage

        # C nodes with L links
        c_with_l = sum(
            1 for c_id in c_nodes
            if any(
                self.graph.nodes[e[0]].get('section') == 'L'
                for e in self.graph.in_edges(c_id)
            )
        )
        coverage["c_with_l"] = c_with_l / len(c_nodes) if c_nodes else 0.0

        # C nodes with M links
        c_with_m = sum(
            1 for c_id in c_nodes
            if any(
                self.graph.nodes[e[0]].get('section') == 'M'
                for e in self.graph.in_edges(c_id)
            )
        )
        coverage["c_with_m"] = c_with_m / len(c_nodes) if c_nodes else 0.0

        # L nodes with M links
        l_with_m = sum(
            1 for l_id in l_nodes
            if any(
                self.graph.nodes[e[0]].get('section') == 'M'
                for e in self.graph.in_edges(l_id)
            )
        )
        coverage["l_with_m"] = l_with_m / len(l_nodes) if l_nodes else 0.0

        # Overall coverage
        coverage["overall"] = (
            coverage["c_with_l"] + coverage["c_with_m"] + coverage["l_with_m"]
        ) / 3.0

        return coverage

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format"""
        nodes = []
        for node_id in self.graph.nodes():
            node_data = dict(self.graph.nodes[node_id])
            node_data["id"] = node_id
            nodes.append(node_data)

        edges = []
        for source, target, data in self.graph.edges(data=True):
            edge_data = dict(data)
            edge_data["source"] = source
            edge_data["target"] = target
            edges.append(edge_data)

        analysis = self.analyze()

        return {
            "nodes": nodes,
            "edges": edges,
            "analysis": {
                "total_nodes": analysis.total_nodes,
                "total_edges": analysis.total_edges,
                "orphan_count": analysis.orphan_count,
                "section_counts": analysis.section_counts,
                "edge_type_counts": analysis.edge_type_counts,
                "connected_components": analysis.connected_components,
                "iron_triangle_coverage": analysis.iron_triangle_coverage,
                "dependency_depth": analysis.dependency_depth,
            },
            "orphans": [
                {
                    "id": o.orphan_id,
                    "section": o.section.value,
                    "type": o.requirement_type.value,
                    "reason": o.reason,
                    "suggestion": o.suggestion,
                }
                for o in analysis.orphans
            ],
        }

    @classmethod
    def from_requirements(
        cls,
        requirements: List[RequirementNode],
        auto_link: bool = True,
        similarity_threshold: float = 0.3
    ) -> "RequirementsDAG":
        """
        Factory method to create DAG from requirement nodes.

        Args:
            requirements: List of RequirementNode objects
            auto_link: Whether to automatically build Iron Triangle edges
            similarity_threshold: Threshold for automatic linking

        Returns:
            Populated RequirementsDAG
        """
        dag = cls()
        dag.add_requirements(requirements)

        if auto_link:
            dag.build_iron_triangle_edges(similarity_threshold)

        return dag
