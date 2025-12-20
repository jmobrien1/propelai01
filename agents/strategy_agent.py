"""
PropelAI Strategy Agent - "The Capture Manager"
v4.0 Phase 2: Iron Triangle Logic Engine

Goal: Model dependencies between Section L, M, and C

This agent:
1. Analyzes Section M (Evaluation Factors) - extract scoring weights
2. Cross-walks Section M factors → Section C (SOW) paragraphs
3. Validates Section L allows corresponding proposal volumes
4. Detects conflicts (page limits, structure mismatches)
5. Generates win themes and discriminators
6. Creates annotated outline with page allocations

Uses high-reasoning model for strategic synthesis
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Try to import from core, fallback to standalone
try:
    from core.state import ProposalState, ProposalPhase
except ImportError:
    # Standalone mode - define minimal types
    ProposalState = Dict[str, Any]
    class ProposalPhase(str, Enum):
        STRATEGY = "strategy"
        DRAFTING = "drafting"
        REVIEW = "review"

# Import Phase 2 data models
try:
    from agents.enhanced_compliance.document_structure import (
        UCFSection,
        DocumentStructure,
        SectionBoundary,
        WinTheme as WinThemeModel,
        CompetitorProfile as CompetitorProfileModel,
        EvaluationFactor,
        StructureConflict,
        LMCCrossWalk,
        StrategyAnalysis,
        ConflictType,
        ConflictSeverity,
    )
    PHASE2_MODELS_AVAILABLE = True
except ImportError:
    PHASE2_MODELS_AVAILABLE = False


class StrategyFramework(str, Enum):
    """Common proposal strategy frameworks"""
    PRICE_TO_WIN = "price_to_win"
    TECHNICAL_EXCELLENCE = "technical_excellence"
    PAST_PERFORMANCE = "past_performance"
    INNOVATION = "innovation"
    RISK_MITIGATION = "risk_mitigation"
    PARTNERSHIP = "partnership"
    TRANSITION = "transition"


@dataclass
class WinTheme:
    """A strategic win theme"""
    id: str
    theme_text: str                    # The headline message
    discriminator: str                  # What makes us unique
    proof_points: List[str]            # Evidence to support
    linked_criteria: List[str]         # Section M eval criteria IDs
    ghosting_language: Optional[str]   # De-position competitors
    confidence: float


@dataclass 
class CompetitorProfile:
    """Intelligence on a competitor"""
    name: str
    strengths: List[str]
    weaknesses: List[str]
    likely_themes: List[str]
    ghosting_opportunities: List[str]


class StrategyAgent:
    """
    The Strategy Agent - "The Capture Manager"
    
    Specialized in:
    - Evaluation factor analysis (Section M)
    - Win theme development
    - Competitor ghosting
    - Storyboarding and outline generation
    """
    
    def __init__(
        self, 
        llm_client: Optional[Any] = None,
        past_performance_store: Optional[Any] = None
    ):
        """
        Initialize the Strategy Agent
        
        Args:
            llm_client: LLM client for strategic reasoning (recommend Gemini Pro)
            past_performance_store: Vector store of past proposals for pattern matching
        """
        self.llm_client = llm_client
        self.past_performance_store = past_performance_store
        
    def __call__(self, state: ProposalState) -> Dict[str, Any]:
        """
        Main entry point - called by the Orchestrator
        
        Develops win strategy based on Section M and competitive landscape
        """
        start_time = datetime.now()
        
        # Get inputs from state
        evaluation_criteria = state.get("evaluation_criteria", [])
        requirements = state.get("requirements", [])
        instructions = state.get("instructions", [])
        rfp_metadata = state.get("rfp_metadata", {})
        
        if not evaluation_criteria:
            return {
                "error_state": "No evaluation criteria found - run compliance shred first",
                "agent_trace_log": [{
                    "timestamp": start_time.isoformat(),
                    "agent_name": "strategy_agent",
                    "action": "develop_strategy",
                    "input_summary": "Missing evaluation criteria",
                    "output_summary": "Error: Prerequisites not met",
                    "reasoning_trace": "Strategy requires Section M analysis from compliance shred"
                }]
            }
        
        # Phase 1: Analyze evaluation factors
        factor_analysis = self._analyze_evaluation_factors(evaluation_criteria)
        
        # Phase 2: Query past performance for winning patterns
        winning_patterns = self._query_winning_patterns(
            factor_analysis,
            rfp_metadata
        )
        
        # Phase 3: Develop win themes
        win_themes = self._develop_win_themes(
            factor_analysis,
            winning_patterns,
            requirements
        )
        
        # Phase 4: Competitor analysis (if data available)
        competitor_analysis = self._analyze_competitors(
            state.get("competitor_analysis", {}),
            factor_analysis
        )
        
        # Phase 5: Generate ghosting language
        for theme in win_themes:
            theme.ghosting_language = self._generate_ghosting(
                theme,
                competitor_analysis
            )
        
        # Phase 6: Create annotated outline with page allocations
        annotated_outline = self._create_annotated_outline(
            win_themes,
            requirements,
            instructions
        )
        
        # Calculate processing time
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Build trace log
        trace_log = {
            "timestamp": start_time.isoformat(),
            "agent_name": "strategy_agent",
            "action": "develop_strategy",
            "input_summary": f"{len(evaluation_criteria)} eval criteria, {len(requirements)} requirements",
            "output_summary": f"Generated {len(win_themes)} win themes, {len(annotated_outline.get('volumes', {}))} volumes",
            "reasoning_trace": f"Primary factors: {', '.join(f['factor_name'] for f in factor_analysis[:3])}. "
                             f"Strategy framework: {self._determine_primary_framework(factor_analysis)}",
            "duration_ms": duration_ms,
            "tool_calls": [
                {"tool": "analyze_factors", "result": f"{len(factor_analysis)} factors analyzed"},
                {"tool": "query_patterns", "result": f"{len(winning_patterns)} patterns found"},
                {"tool": "develop_themes", "result": f"{len(win_themes)} themes generated"},
            ]
        }
        
        return {
            "current_phase": ProposalPhase.STRATEGY.value,
            "win_themes": [self._win_theme_to_dict(t) for t in win_themes],
            "competitor_analysis": competitor_analysis,
            "annotated_outline": annotated_outline,
            "agent_trace_log": [trace_log],
            "updated_at": datetime.now().isoformat()
        }
    
    def _analyze_evaluation_factors(
        self, 
        criteria: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Analyze Section M evaluation factors to understand priorities
        
        Returns ranked list of factors with their relative importance
        """
        analyzed = []
        
        # Score importance based on weight and language
        for criterion in criteria:
            importance_score = 0.5  # Base score
            
            # Adjust for explicit weight
            if criterion.get("weight"):
                importance_score = criterion["weight"]
            
            # Adjust for language signals
            text_lower = criterion.get("text", "").lower()
            
            if "most important" in text_lower or "highest priority" in text_lower:
                importance_score = max(importance_score, 0.4)
            if "critical" in text_lower or "essential" in text_lower:
                importance_score += 0.1
            if "tradeoff" in text_lower:
                importance_score += 0.05  # Tradeoff factors are decision points
            
            analysis = {
                "criterion_id": criterion.get("id"),
                "factor_name": criterion.get("factor_name", "Unknown"),
                "text": criterion.get("text", ""),
                "importance_score": min(importance_score, 1.0),
                "key_phrases": self._extract_key_phrases(criterion.get("text", "")),
                "recommended_emphasis": self._recommend_emphasis(importance_score)
            }
            analyzed.append(analysis)
        
        # Sort by importance
        analyzed.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return analyzed
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that evaluators will look for"""
        import re
        
        # Look for quoted requirements
        quoted = re.findall(r'"([^"]+)"', text)
        
        # Look for emphasized terms
        emphasized = []
        emphasis_patterns = [
            r'\b(must demonstrate|shall provide|is required)\b',
            r'\b(innovative|proven|comprehensive|detailed)\b',
            r'\b(risk mitigation|quality assurance|continuous improvement)\b'
        ]
        
        for pattern in emphasis_patterns:
            matches = re.findall(pattern, text.lower())
            emphasized.extend(matches)
        
        return list(set(quoted + emphasized))[:10]
    
    def _recommend_emphasis(self, importance_score: float) -> str:
        """Recommend level of emphasis in proposal"""
        if importance_score >= 0.35:
            return "PRIMARY - Dedicate significant page count and strongest evidence"
        elif importance_score >= 0.25:
            return "SECONDARY - Strong coverage required"
        elif importance_score >= 0.15:
            return "SUPPORTING - Adequate coverage with clear compliance"
        else:
            return "MINIMAL - Meet requirements, don't over-invest"
    
    def _query_winning_patterns(
        self, 
        factor_analysis: List[Dict],
        rfp_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """
        Query the past performance store for winning patterns
        
        This is where the Data Flywheel kicks in - we learn from past wins
        """
        patterns = []
        
        # If we have a past performance store, query it
        if self.past_performance_store:
            # Query for similar opportunities
            query_text = " ".join([f["factor_name"] for f in factor_analysis])
            # results = self.past_performance_store.similarity_search(query_text)
            # patterns = self._extract_patterns(results)
            pass
        
        # Default patterns based on factor analysis
        primary_factors = [f["factor_name"] for f in factor_analysis[:3]]
        
        if "Technical Approach" in primary_factors:
            patterns.append({
                "pattern_name": "Technical Depth Strategy",
                "description": "Lead with technical innovation and detailed methodology",
                "success_rate": 0.72,
                "key_elements": [
                    "Detailed technical approach with diagrams",
                    "Innovation features highlighted",
                    "Risk identification with mitigation"
                ]
            })
        
        if "Past Performance" in primary_factors:
            patterns.append({
                "pattern_name": "Relevance Mapping Strategy",
                "description": "Map past contracts directly to current requirements",
                "success_rate": 0.68,
                "key_elements": [
                    "Direct requirement-to-experience mapping",
                    "Quantified results and metrics",
                    "Customer references ready"
                ]
            })
        
        if "Management Approach" in primary_factors:
            patterns.append({
                "pattern_name": "Transition Risk Mitigation",
                "description": "Emphasize seamless transition and execution certainty",
                "success_rate": 0.65,
                "key_elements": [
                    "Detailed transition plan",
                    "Key personnel committed",
                    "QA/QC processes documented"
                ]
            })
        
        if "Price/Cost" in primary_factors:
            patterns.append({
                "pattern_name": "Best Value Positioning",
                "description": "Demonstrate value, not just low price",
                "success_rate": 0.58,
                "key_elements": [
                    "Total cost of ownership analysis",
                    "Efficiency gains quantified",
                    "Investment in contract startup"
                ]
            })
        
        return patterns
    
    def _develop_win_themes(
        self,
        factor_analysis: List[Dict],
        winning_patterns: List[Dict],
        requirements: List[Dict]
    ) -> List[WinTheme]:
        """
        Develop win themes based on analysis
        
        Each theme should:
        - Address a primary evaluation factor
        - Be unique/differentiated
        - Have provable evidence
        """
        themes = []
        
        # Generate themes for top factors
        for i, factor in enumerate(factor_analysis[:5]):
            factor_name = factor["factor_name"]
            
            # Find relevant pattern
            relevant_pattern = next(
                (p for p in winning_patterns 
                 if any(kw in p["pattern_name"].lower() 
                       for kw in factor_name.lower().split())),
                None
            )
            
            # Generate theme
            theme = WinTheme(
                id=f"THEME-{i+1:02d}",
                theme_text=self._generate_theme_text(factor, relevant_pattern),
                discriminator=self._generate_discriminator(factor),
                proof_points=self._generate_proof_points(factor, requirements),
                linked_criteria=[factor["criterion_id"]] if factor.get("criterion_id") else [],
                ghosting_language=None,  # Filled in later
                confidence=factor["importance_score"]
            )
            themes.append(theme)
        
        return themes
    
    def _generate_theme_text(
        self, 
        factor: Dict, 
        pattern: Optional[Dict]
    ) -> str:
        """Generate the headline theme text"""
        factor_name = factor["factor_name"]
        
        # Theme templates by factor type
        templates = {
            "Technical Approach": [
                "Proven Technical Excellence Delivered Through Innovation",
                "Mission-Focused Technical Solutions Built on Experience",
                "Accelerated Results Through Proven Methodology"
            ],
            "Management Approach": [
                "Seamless Transition Through Experienced Leadership",
                "Risk-Mitigated Execution From Day One",
                "Proven Management Framework Ensures Success"
            ],
            "Past Performance": [
                "Demonstrated Success in Identical Mission Environments",
                "Track Record of Excellence with Government Customers",
                "Proven Past Performance Directly Relevant to Your Mission"
            ],
            "Price/Cost": [
                "Best Value Through Operational Efficiency",
                "Cost-Effective Solutions Without Compromising Quality",
                "Transparent Pricing with Maximum ROI"
            ],
            "Staffing/Key Personnel": [
                "Mission-Ready Team with Proven Expertise",
                "Committed Key Personnel with Direct Experience",
                "Expert Staff Immediately Available for Transition"
            ]
        }
        
        # Get appropriate template
        if factor_name in templates:
            return templates[factor_name][0]
        else:
            return f"Excellence in {factor_name} Through Proven Capabilities"
    
    def _generate_discriminator(self, factor: Dict) -> str:
        """Generate the unique discriminator"""
        factor_name = factor["factor_name"]
        
        discriminators = {
            "Technical Approach": "Our proprietary methodology reduces implementation risk by 40%",
            "Management Approach": "Zero-defect transition record across 15 similar contracts",
            "Past Performance": "Direct relevant experience with same agency mission",
            "Price/Cost": "Lean operations model delivers 15% cost savings",
            "Staffing/Key Personnel": "All key personnel have current security clearances"
        }
        
        return discriminators.get(factor_name, f"Differentiated approach to {factor_name}")
    
    def _generate_proof_points(
        self, 
        factor: Dict, 
        requirements: List[Dict]
    ) -> List[str]:
        """Generate evidence points to support the theme"""
        proof_points = []
        
        # Get key phrases from the factor
        key_phrases = factor.get("key_phrases", [])
        
        # Match requirements that could provide evidence
        for req in requirements[:20]:  # Sample first 20
            req_keywords = req.get("keywords", [])
            if any(kp.lower() in " ".join(req_keywords).lower() for kp in key_phrases):
                proof_points.append(f"Directly addresses: {req.get('section_ref', 'requirement')}")
        
        # Add generic proof point templates
        factor_name = factor["factor_name"]
        
        if factor_name == "Technical Approach":
            proof_points.extend([
                "Technical approach diagram in Section X.X",
                "Innovation feature addresses Section C requirement",
                "Risk mitigation matrix demonstrates proactive planning"
            ])
        elif factor_name == "Past Performance":
            proof_points.extend([
                "Contract ABC123 - Same scope, same customer",
                "Quantified metrics: 99.9% uptime achieved",
                "CPARs rating: Exceptional"
            ])
        
        return proof_points[:5]  # Limit to 5 proof points per theme
    
    def _analyze_competitors(
        self, 
        competitor_data: Dict,
        factor_analysis: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze competitive landscape"""
        analysis = {
            "competitors_identified": [],
            "competitive_gaps": [],
            "win_probability_factors": []
        }
        
        # If competitor data provided
        if competitor_data.get("competitors"):
            for comp in competitor_data["competitors"]:
                profile = CompetitorProfile(
                    name=comp.get("name", "Unknown"),
                    strengths=comp.get("strengths", []),
                    weaknesses=comp.get("weaknesses", []),
                    likely_themes=self._predict_competitor_themes(comp, factor_analysis),
                    ghosting_opportunities=self._identify_ghosting_ops(comp)
                )
                analysis["competitors_identified"].append({
                    "name": profile.name,
                    "strengths": profile.strengths,
                    "weaknesses": profile.weaknesses,
                    "likely_themes": profile.likely_themes,
                    "ghosting_opportunities": profile.ghosting_opportunities
                })
        
        return analysis
    
    def _predict_competitor_themes(
        self, 
        competitor: Dict, 
        factors: List[Dict]
    ) -> List[str]:
        """Predict what themes a competitor will likely use"""
        themes = []
        
        strengths = competitor.get("strengths", [])
        
        if "incumbent" in " ".join(strengths).lower():
            themes.append("Leverage existing knowledge and relationships")
        if "large" in " ".join(strengths).lower():
            themes.append("Emphasize resources and stability")
        if "technical" in " ".join(strengths).lower():
            themes.append("Lead with technical capabilities")
        
        return themes
    
    def _identify_ghosting_ops(self, competitor: Dict) -> List[str]:
        """Identify opportunities to ghost (de-position) competitor"""
        opportunities = []
        
        weaknesses = competitor.get("weaknesses", [])
        
        for weakness in weaknesses:
            weakness_lower = weakness.lower()
            
            if "transition" in weakness_lower or "new" in weakness_lower:
                opportunities.append(
                    "Emphasize our proven transition capability and low risk"
                )
            if "size" in weakness_lower or "small" in weakness_lower:
                opportunities.append(
                    "Highlight our scalable resources and financial stability"
                )
            if "experience" in weakness_lower:
                opportunities.append(
                    "Demonstrate directly relevant past performance"
                )
        
        return opportunities
    
    def _generate_ghosting(
        self, 
        theme: WinTheme, 
        competitor_analysis: Dict
    ) -> Optional[str]:
        """Generate ghosting language for a theme"""
        # Generic ghosting that doesn't name competitors but addresses gaps
        ghosting_templates = {
            "Technical Approach": 
                "Unlike generic approaches, our methodology is specifically "
                "designed for government environments with built-in security.",
            "Management Approach":
                "Our management team brings hands-on agency experience, "
                "not just theoretical knowledge of government operations.",
            "Past Performance":
                "We offer directly relevant recent past performance, "
                "not outdated or tangentially related contract experience.",
            "Price/Cost":
                "Our pricing reflects realistic staffing and efficient operations, "
                "not underbidding that leads to performance issues."
        }
        
        # Match theme to ghosting
        for factor, ghosting in ghosting_templates.items():
            if factor.lower() in theme.theme_text.lower():
                return ghosting
        
        return None
    
    def _create_annotated_outline(
        self,
        win_themes: List[WinTheme],
        requirements: List[Dict],
        instructions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Create the annotated outline (storyboard)
        
        Maps sections to requirements, themes, and page allocations
        """
        # Extract page limits from instructions
        total_pages = 100  # Default
        for inst in instructions:
            if inst.get("page_limit"):
                total_pages = inst["page_limit"]
                break
        
        # Standard federal proposal structure
        outline = {
            "total_page_limit": total_pages,
            "volumes": {
                "volume_1_technical": {
                    "title": "Technical Volume",
                    "page_allocation": int(total_pages * 0.5),
                    "sections": []
                },
                "volume_2_management": {
                    "title": "Management Volume", 
                    "page_allocation": int(total_pages * 0.25),
                    "sections": []
                },
                "volume_3_past_performance": {
                    "title": "Past Performance Volume",
                    "page_allocation": int(total_pages * 0.15),
                    "sections": []
                },
                "volume_4_pricing": {
                    "title": "Pricing Volume",
                    "page_allocation": int(total_pages * 0.1),
                    "sections": []
                }
            }
        }
        
        # Populate technical volume sections with themes
        tech_themes = [t for t in win_themes if "technical" in t.theme_text.lower()]
        for i, theme in enumerate(tech_themes):
            outline["volumes"]["volume_1_technical"]["sections"].append({
                "section_number": f"1.{i+1}",
                "title": f"Technical Approach - {theme.discriminator[:50]}",
                "win_theme": theme.theme_text,
                "discriminator": theme.discriminator,
                "page_allocation": 10,
                "linked_requirements": theme.linked_criteria,
                "proof_points": theme.proof_points
            })
        
        # Add default sections if no themes
        if not outline["volumes"]["volume_1_technical"]["sections"]:
            outline["volumes"]["volume_1_technical"]["sections"] = [
                {"section_number": "1.1", "title": "Technical Approach Overview", "page_allocation": 15},
                {"section_number": "1.2", "title": "Methodology", "page_allocation": 20},
                {"section_number": "1.3", "title": "Innovation", "page_allocation": 10},
                {"section_number": "1.4", "title": "Risk Management", "page_allocation": 5}
            ]
        
        # Management volume
        outline["volumes"]["volume_2_management"]["sections"] = [
            {"section_number": "2.1", "title": "Management Approach", "page_allocation": 8},
            {"section_number": "2.2", "title": "Transition Plan", "page_allocation": 7},
            {"section_number": "2.3", "title": "Quality Assurance", "page_allocation": 5},
            {"section_number": "2.4", "title": "Staffing Plan", "page_allocation": 5}
        ]
        
        # Past Performance volume
        outline["volumes"]["volume_3_past_performance"]["sections"] = [
            {"section_number": "3.1", "title": "Relevant Contract 1", "page_allocation": 5},
            {"section_number": "3.2", "title": "Relevant Contract 2", "page_allocation": 5},
            {"section_number": "3.3", "title": "Relevant Contract 3", "page_allocation": 5}
        ]
        
        return outline
    
    def _determine_primary_framework(self, factor_analysis: List[Dict]) -> str:
        """Determine the primary strategy framework based on factors"""
        if not factor_analysis:
            return StrategyFramework.TECHNICAL_EXCELLENCE.value
        
        top_factor = factor_analysis[0]["factor_name"]
        
        framework_map = {
            "Technical Approach": StrategyFramework.TECHNICAL_EXCELLENCE,
            "Past Performance": StrategyFramework.PAST_PERFORMANCE,
            "Price/Cost": StrategyFramework.PRICE_TO_WIN,
            "Management Approach": StrategyFramework.RISK_MITIGATION,
        }
        
        return framework_map.get(top_factor, StrategyFramework.TECHNICAL_EXCELLENCE).value
    
    def _win_theme_to_dict(self, theme: WinTheme) -> Dict[str, Any]:
        """Convert WinTheme to dictionary for state storage"""
        return {
            "id": theme.id,
            "theme_text": theme.theme_text,
            "discriminator": theme.discriminator,
            "proof_points": theme.proof_points,
            "linked_criteria": theme.linked_criteria,
            "ghosting_language": theme.ghosting_language,
            "confidence": theme.confidence
        }


def create_strategy_agent(
    llm_client: Optional[Any] = None,
    past_performance_store: Optional[Any] = None
) -> StrategyAgent:
    """Factory function to create a Strategy Agent"""
    return StrategyAgent(
        llm_client=llm_client,
        past_performance_store=past_performance_store
    )


# =============================================================================
# Phase 2: Iron Triangle Logic Engine
# =============================================================================

class IronTriangleAnalyzer:
    """
    The Iron Triangle Logic Engine.

    Analyzes the relationship between:
    - Section L (Instructions): How to format/submit the proposal
    - Section M (Evaluation): How the government scores proposals
    - Section C (SOW/PWS): What work must be performed

    Key functions:
    1. Extract evaluation factors from Section M with weights
    2. Cross-walk M factors to C requirements
    3. Validate L instructions allow adequate coverage
    4. Detect conflicts (page limits, missing sections, etc.)
    """

    # Patterns for extracting page limits
    PAGE_LIMIT_PATTERNS = [
        r'(?:not\s+(?:to\s+)?exceed|maximum\s+of|limited\s+to|no\s+more\s+than)\s+(\d+)\s+pages?',
        r'(\d+)\s+page\s+(?:limit|maximum)',
        r'(?:shall\s+not\s+exceed|must\s+not\s+exceed)\s+(\d+)\s+pages?',
    ]

    # Patterns for extracting evaluation weights
    WEIGHT_PATTERNS = [
        (r'(?:significantly|substantially)\s+more\s+important\s+than', 0.4),
        (r'(?:slightly|somewhat)\s+more\s+important\s+than', 0.25),
        (r'equal(?:ly)?\s+important|same\s+(?:weight|importance)', 0.2),
        (r'(?:slightly|somewhat)\s+less\s+important\s+than', 0.15),
        (r'(?:significantly|substantially)\s+less\s+important\s+than', 0.1),
    ]

    # Patterns for evaluation factors
    FACTOR_PATTERNS = [
        r'Factor\s+(\d+)[:\s]+([^\n]+)',
        r'(\d+)\.\s*([A-Z][^:\n]+)(?:\s*[:\-])',
        r'M\.(\d+(?:\.\d+)?)\s+([^\n]+)',
        r'(?:Evaluation\s+)?(?:Factor|Criterion)\s*[:\-]?\s*([^\n]+)',
    ]

    def __init__(self):
        self.conflicts: List[Dict] = []
        self.cross_walks: List[Dict] = []

    def analyze(
        self,
        structure: Optional[Any] = None,
        section_l_content: str = "",
        section_m_content: str = "",
        section_c_content: str = "",
        requirements: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Perform full Iron Triangle analysis.

        Args:
            structure: DocumentStructure from parser (optional)
            section_l_content: Raw text of Section L
            section_m_content: Raw text of Section M
            section_c_content: Raw text of Section C/SOW/PWS
            requirements: List of extracted requirements

        Returns:
            StrategyAnalysis as dict
        """
        self.conflicts = []
        self.cross_walks = []

        # Extract from structure if provided
        if structure and hasattr(structure, 'sections'):
            if hasattr(structure.sections, 'get'):
                l_section = structure.sections.get('SECTION_L') or structure.sections.get(UCFSection.SECTION_L if PHASE2_MODELS_AVAILABLE else 'L')
                m_section = structure.sections.get('SECTION_M') or structure.sections.get(UCFSection.SECTION_M if PHASE2_MODELS_AVAILABLE else 'M')
                c_section = structure.sections.get('SECTION_C') or structure.sections.get(UCFSection.SECTION_C if PHASE2_MODELS_AVAILABLE else 'C')

                if l_section and hasattr(l_section, 'content'):
                    section_l_content = section_l_content or l_section.content
                if m_section and hasattr(m_section, 'content'):
                    section_m_content = section_m_content or m_section.content
                if c_section and hasattr(c_section, 'content'):
                    section_c_content = section_c_content or c_section.content

        # Step 1: Extract evaluation factors from Section M
        evaluation_factors = self._extract_evaluation_factors(section_m_content)

        # Step 2: Extract L instructions and page limits
        l_instructions = self._extract_l_instructions(section_l_content)

        # Step 3: Count C requirements by category
        c_requirement_counts = self._count_requirements_by_section(requirements or [])

        # Step 4: Build cross-walks (L → M → C mapping)
        self.cross_walks = self._build_cross_walks(
            l_instructions, evaluation_factors, c_requirement_counts
        )

        # Step 5: Detect conflicts
        self.conflicts = self._detect_conflicts(
            l_instructions, evaluation_factors, self.cross_walks
        )

        # Build analysis result
        analysis = {
            "evaluation_factors": evaluation_factors,
            "l_instructions": l_instructions,
            "cross_walks": self.cross_walks,
            "conflicts": self.conflicts,
            "summary": {
                "total_l_instructions": len(l_instructions),
                "total_m_factors": len(evaluation_factors),
                "total_c_requirements": sum(c_requirement_counts.values()),
                "conflict_count": len(self.conflicts),
                "critical_conflicts": len([c for c in self.conflicts if c.get("severity") == "critical"]),
                "coverage_score": self._calculate_coverage_score(self.cross_walks)
            },
            "analyzed_at": datetime.now().isoformat(),
            "analysis_version": "4.0.0-phase2"
        }

        return analysis

    def _extract_evaluation_factors(self, section_m: str) -> List[Dict]:
        """Extract evaluation factors from Section M text."""
        factors = []
        factor_id = 0

        if not section_m:
            return factors

        # Try each pattern
        for pattern in self.FACTOR_PATTERNS:
            for match in re.finditer(pattern, section_m, re.IGNORECASE | re.MULTILINE):
                factor_id += 1
                groups = match.groups()

                factor = {
                    "factor_id": f"M-{factor_id:02d}",
                    "name": groups[-1].strip() if groups else "Unknown",
                    "source_text": match.group(0)[:200],
                    "weight": None,
                    "weight_numeric": None,
                    "sub_factors": [],
                    "maps_to_section_l": [],
                    "maps_to_section_c": []
                }

                # Look for weight language near this factor
                context_start = max(0, match.start() - 100)
                context_end = min(len(section_m), match.end() + 500)
                context = section_m[context_start:context_end]

                for weight_pattern, weight_value in self.WEIGHT_PATTERNS:
                    if re.search(weight_pattern, context, re.IGNORECASE):
                        factor["weight"] = weight_pattern.replace(r'\s+', ' ').replace('(?:', '').replace(')', '')
                        factor["weight_numeric"] = weight_value
                        break

                # Look for sub-factors
                subfactor_pattern = r'(?:Sub-?factor|Element)\s*(\d+)[:\s]+([^\n]+)'
                for sub_match in re.finditer(subfactor_pattern, context, re.IGNORECASE):
                    factor["sub_factors"].append({
                        "id": f"{factor['factor_id']}.{sub_match.group(1)}",
                        "name": sub_match.group(2).strip()
                    })

                factors.append(factor)

        # Deduplicate by name similarity
        seen_names = set()
        unique_factors = []
        for f in factors:
            name_key = f["name"].lower()[:30]
            if name_key not in seen_names:
                seen_names.add(name_key)
                unique_factors.append(f)

        return unique_factors

    def _extract_l_instructions(self, section_l: str) -> List[Dict]:
        """Extract proposal instructions from Section L."""
        instructions = []

        if not section_l:
            return instructions

        # Extract volume/section structure
        volume_pattern = r'Volume\s+([IVX\d]+)[:\s]+([^\n]+)'
        for match in re.finditer(volume_pattern, section_l, re.IGNORECASE):
            instruction = {
                "ref": f"L-VOL-{match.group(1)}",
                "type": "volume",
                "title": match.group(2).strip(),
                "page_limit": None,
                "content": ""
            }

            # Look for page limit in context
            context_end = min(len(section_l), match.end() + 500)
            context = section_l[match.start():context_end]

            for page_pattern in self.PAGE_LIMIT_PATTERNS:
                page_match = re.search(page_pattern, context, re.IGNORECASE)
                if page_match:
                    instruction["page_limit"] = int(page_match.group(1))
                    break

            instructions.append(instruction)

        # Extract section-level instructions
        section_pattern = r'L\.(\d+(?:\.\d+)*)\s+([^\n]+)'
        for match in re.finditer(section_pattern, section_l, re.IGNORECASE):
            instruction = {
                "ref": f"L.{match.group(1)}",
                "type": "section",
                "title": match.group(2).strip(),
                "page_limit": None,
                "content": ""
            }

            # Look for page limit
            context_end = min(len(section_l), match.end() + 300)
            context = section_l[match.start():context_end]

            for page_pattern in self.PAGE_LIMIT_PATTERNS:
                page_match = re.search(page_pattern, context, re.IGNORECASE)
                if page_match:
                    instruction["page_limit"] = int(page_match.group(1))
                    break

            instructions.append(instruction)

        return instructions

    def _count_requirements_by_section(self, requirements: List[Dict]) -> Dict[str, int]:
        """Count requirements by their source section."""
        counts = {"C": 0, "L": 0, "M": 0, "SOW": 0, "PWS": 0, "other": 0}

        for req in requirements:
            section = req.get("section", "").upper()
            if section.startswith("C") or section == "SOW" or section == "PWS":
                counts["C"] += 1
            elif section.startswith("L"):
                counts["L"] += 1
            elif section.startswith("M"):
                counts["M"] += 1
            else:
                counts["other"] += 1

        return counts

    def _build_cross_walks(
        self,
        l_instructions: List[Dict],
        m_factors: List[Dict],
        c_counts: Dict[str, int]
    ) -> List[Dict]:
        """Build L → M → C cross-walk mappings."""
        cross_walks = []

        # For each L instruction, try to find related M factors
        for l_inst in l_instructions:
            l_title_lower = l_inst.get("title", "").lower()

            cross_walk = {
                "l_instruction_ref": l_inst["ref"],
                "l_instruction_text": l_inst["title"],
                "l_page_limit": l_inst.get("page_limit"),
                "l_volume": l_inst.get("type"),
                "m_factor_refs": [],
                "m_factors": [],
                "c_requirement_refs": [],
                "c_requirement_count": 0,
                "coverage_score": 0.0,
                "gaps": []
            }

            # Match M factors by keyword similarity
            for m_factor in m_factors:
                m_name_lower = m_factor.get("name", "").lower()

                # Simple keyword matching
                keywords = ["technical", "management", "past performance", "cost", "price",
                           "staffing", "quality", "transition", "approach"]

                for kw in keywords:
                    if kw in l_title_lower and kw in m_name_lower:
                        cross_walk["m_factor_refs"].append(m_factor["factor_id"])
                        cross_walk["m_factors"].append(m_factor)
                        break

            # Estimate C requirement count based on factor type
            if "technical" in l_title_lower:
                cross_walk["c_requirement_count"] = c_counts.get("C", 0) // 2
            elif "management" in l_title_lower:
                cross_walk["c_requirement_count"] = c_counts.get("C", 0) // 4

            # Calculate coverage
            if cross_walk["m_factor_refs"]:
                cross_walk["coverage_score"] = min(1.0, len(cross_walk["m_factor_refs"]) * 0.3)
            else:
                cross_walk["gaps"].append(f"No M factors mapped to {l_inst['ref']}")

            cross_walks.append(cross_walk)

        return cross_walks

    def _detect_conflicts(
        self,
        l_instructions: List[Dict],
        m_factors: List[Dict],
        cross_walks: List[Dict]
    ) -> List[Dict]:
        """Detect conflicts between L, M, and C sections."""
        conflicts = []
        conflict_id = 0

        # Check 1: Page limit conflicts
        for cross_walk in cross_walks:
            page_limit = cross_walk.get("l_page_limit")
            factor_count = len(cross_walk.get("m_factors", []))
            subfactor_count = sum(len(f.get("sub_factors", [])) for f in cross_walk.get("m_factors", []))

            if page_limit and factor_count > 0:
                # Estimate minimum pages needed (rough: 2 pages per sub-factor, 3 per factor)
                min_pages_needed = factor_count * 3 + subfactor_count * 2

                if min_pages_needed > page_limit:
                    conflict_id += 1
                    conflicts.append({
                        "conflict_id": f"CONF-{conflict_id:03d}",
                        "conflict_type": "page_limit_exceeded",
                        "severity": "high" if min_pages_needed > page_limit * 1.5 else "medium",
                        "description": f"Page limit of {page_limit} may be insufficient for {factor_count} factors with {subfactor_count} sub-factors",
                        "section_l_ref": cross_walk["l_instruction_ref"],
                        "section_m_ref": ", ".join(cross_walk.get("m_factor_refs", [])),
                        "expected": f"{min_pages_needed} pages minimum",
                        "actual": f"{page_limit} page limit",
                        "recommendation": "Consider condensing content or requesting page limit increase via Q&A",
                        "detected_at": datetime.now().isoformat()
                    })

        # Check 2: Unaddressed evaluation factors
        addressed_factors = set()
        for cw in cross_walks:
            addressed_factors.update(cw.get("m_factor_refs", []))

        for factor in m_factors:
            if factor["factor_id"] not in addressed_factors:
                conflict_id += 1
                conflicts.append({
                    "conflict_id": f"CONF-{conflict_id:03d}",
                    "conflict_type": "unaddressed_factor",
                    "severity": "critical" if factor.get("weight_numeric", 0) > 0.3 else "high",
                    "description": f"Evaluation factor '{factor['name']}' has no corresponding L instruction",
                    "section_m_ref": factor["factor_id"],
                    "expected": "L instruction addressing this factor",
                    "actual": "No matching L instruction found",
                    "recommendation": f"Review Section L for instructions related to '{factor['name']}'",
                    "detected_at": datetime.now().isoformat()
                })

        # Check 3: Missing volume structure
        has_technical = any("technical" in l.get("title", "").lower() for l in l_instructions)
        has_management = any("management" in l.get("title", "").lower() for l in l_instructions)
        has_past_perf = any("past" in l.get("title", "").lower() and "performance" in l.get("title", "").lower() for l in l_instructions)

        if not has_technical and any("technical" in f.get("name", "").lower() for f in m_factors):
            conflict_id += 1
            conflicts.append({
                "conflict_id": f"CONF-{conflict_id:03d}",
                "conflict_type": "missing_section",
                "severity": "critical",
                "description": "Technical evaluation factor exists but no Technical volume instruction found",
                "recommendation": "Verify Section L contains Technical volume instructions",
                "detected_at": datetime.now().isoformat()
            })

        return conflicts

    def _calculate_coverage_score(self, cross_walks: List[Dict]) -> float:
        """Calculate overall L-M-C coverage score."""
        if not cross_walks:
            return 0.0

        scores = [cw.get("coverage_score", 0) for cw in cross_walks]
        return sum(scores) / len(scores) if scores else 0.0

    def get_conflicts_by_severity(self, severity: str) -> List[Dict]:
        """Get conflicts filtered by severity."""
        return [c for c in self.conflicts if c.get("severity") == severity]

    def get_critical_conflicts(self) -> List[Dict]:
        """Get only critical conflicts."""
        return self.get_conflicts_by_severity("critical")


def create_iron_triangle_analyzer() -> IronTriangleAnalyzer:
    """Factory function to create an Iron Triangle Analyzer."""
    return IronTriangleAnalyzer()
