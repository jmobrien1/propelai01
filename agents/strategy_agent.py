"""
PropelAI Strategy Agent - "The Capture Manager"
Play 2: The Strategy Engine (Win Themes)

Goal: Define *why* we win before writing a single word

This agent:
1. Analyzes Section M (Evaluation Factors)
2. Queries past bid strategies from the Agent-Trace database
3. Performs competitor ghosting analysis
4. Generates win themes and discriminators
5. Creates annotated outline with page allocations

Uses high-reasoning model (Gemini 1.5 Pro or Claude) for strategic synthesis
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.state import ProposalState, ProposalPhase


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
            existing_trace = state.get("agent_trace_log", [])
            return {
                "error_state": "No evaluation criteria found - run compliance shred first",
                "agent_trace_log": existing_trace + [{
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
        
        # Accumulate trace logs
        existing_trace = state.get("agent_trace_log", [])

        return {
            "current_phase": ProposalPhase.STRATEGY.value,
            "win_themes": [self._win_theme_to_dict(t) for t in win_themes],
            "competitor_analysis": competitor_analysis,
            "annotated_outline": annotated_outline,
            "agent_trace_log": existing_trace + [trace_log],
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


class CompetitorAnalyzer:
    """
    Competitive Analysis Engine

    Analyzes competitive landscape for proposal strategy:
    - Identifies likely competitors
    - Generates ghosting language
    - Maps competitor weaknesses to our strengths
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the Competitor Analyzer

        Args:
            use_llm: Whether to use LLM for advanced analysis
        """
        self.use_llm = use_llm

    def analyze_competitive_landscape(
        self,
        rfp_data: Dict[str, Any],
        known_competitors: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze competitive landscape for an RFP

        Args:
            rfp_data: RFP data including requirements and evaluation criteria
            known_competitors: List of known competitor profiles

        Returns:
            Competitive analysis with ghosting library
        """
        known_competitors = known_competitors or []

        # Build competitor profiles
        profiles = []
        for comp in known_competitors:
            profile = CompetitorProfile(
                name=comp.get("name", "Unknown"),
                strengths=comp.get("strengths", []),
                weaknesses=comp.get("weaknesses", []),
                likely_themes=self._infer_themes(comp),
                ghosting_opportunities=self._identify_ghosting(comp)
            )
            profiles.append(profile)

        # Generate ghosting library
        ghosting_library = self._build_ghosting_library(profiles, rfp_data)

        # Identify win opportunities
        opportunities = self._identify_opportunities(profiles, rfp_data)

        return {
            "competitor_count": len(profiles),
            "competitors": [
                {
                    "name": p.name,
                    "strengths": p.strengths,
                    "weaknesses": p.weaknesses,
                    "likely_themes": p.likely_themes,
                    "ghosting_opportunities": p.ghosting_opportunities
                }
                for p in profiles
            ],
            "ghosting_library": ghosting_library,
            "win_opportunities": opportunities,
            "analysis_date": datetime.now().isoformat()
        }

    def _infer_themes(self, competitor: Dict[str, Any]) -> List[str]:
        """Infer likely win themes a competitor will use"""
        themes = []
        strengths = competitor.get("strengths", [])

        if any("incumbent" in str(s).lower() for s in strengths):
            themes.append("Continuity and proven performance")
        if any("price" in str(s).lower() or "cost" in str(s).lower() for s in strengths):
            themes.append("Competitive pricing")
        if any("innovation" in str(s).lower() or "technology" in str(s).lower() for s in strengths):
            themes.append("Technical innovation")
        if any("experience" in str(s).lower() or "years" in str(s).lower() for s in strengths):
            themes.append("Deep domain experience")

        return themes if themes else ["Standard compliance approach"]

    def _identify_ghosting(self, competitor: Dict[str, Any]) -> List[str]:
        """Identify ghosting opportunities for a competitor"""
        opportunities = []
        weaknesses = competitor.get("weaknesses", [])

        for weakness in weaknesses:
            weakness_lower = weakness.lower()
            if "transition" in weakness_lower:
                opportunities.append("Emphasize seamless transition capability")
            if "size" in weakness_lower or "capacity" in weakness_lower:
                opportunities.append("Highlight scalable resources and depth")
            if "innovation" in weakness_lower:
                opportunities.append("Lead with modern, innovative approaches")
            if "past performance" in weakness_lower:
                opportunities.append("Showcase directly relevant experience")
            if "key personnel" in weakness_lower:
                opportunities.append("Feature committed, named key personnel")

        return opportunities

    def _build_ghosting_library(
        self,
        profiles: List[CompetitorProfile],
        rfp_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Build library of ghosting language"""
        library = []

        # Common ghosting phrases by weakness type
        ghosting_templates = {
            "transition": {
                "phrase": "Our team brings Day 1 readiness with no learning curve",
                "context": "When competitors lack transition experience"
            },
            "innovation": {
                "phrase": "Unlike legacy approaches, our modern methodology...",
                "context": "When competitors use dated methods"
            },
            "personnel": {
                "phrase": "Our named key personnel are committed and available",
                "context": "When competitors have turnover concerns"
            },
            "scalability": {
                "phrase": "With [X] employees nationwide, we scale to meet surge demands",
                "context": "When competitors have capacity limits"
            },
            "incumbent_risk": {
                "phrase": "Fresh perspective unencumbered by legacy processes",
                "context": "Counter incumbent's 'we've always done it this way'"
            }
        }

        # Add applicable ghosting phrases
        all_weaknesses = []
        for profile in profiles:
            all_weaknesses.extend(profile.weaknesses)

        for weakness in all_weaknesses:
            weakness_lower = weakness.lower()
            for key, template in ghosting_templates.items():
                if key in weakness_lower or any(k in weakness_lower for k in key.split("_")):
                    if template not in library:
                        library.append(template)

        # Always include some standard differentiators
        library.append({
            "phrase": "Our proven track record of on-time, on-budget delivery",
            "context": "Standard differentiator"
        })

        return library

    def _identify_opportunities(
        self,
        profiles: List[CompetitorProfile],
        rfp_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify win opportunities based on competitive analysis"""
        opportunities = []

        # Collect all competitor weaknesses
        all_weaknesses = set()
        for profile in profiles:
            all_weaknesses.update(profile.weaknesses)

        # Map weaknesses to opportunities
        if any("transition" in w.lower() for w in all_weaknesses):
            opportunities.append({
                "opportunity": "Transition Excellence",
                "description": "Competitors lack transition experience - emphasize our proven transition methodology",
                "priority": "high"
            })

        if any("incumbent" in w.lower() for w in all_weaknesses):
            opportunities.append({
                "opportunity": "Fresh Perspective",
                "description": "Incumbent complacency - position as innovative challenger with new ideas",
                "priority": "medium"
            })

        if any("personnel" in w.lower() or "staff" in w.lower() for w in all_weaknesses):
            opportunities.append({
                "opportunity": "Key Personnel Commitment",
                "description": "Competitor personnel concerns - showcase committed, named staff",
                "priority": "high"
            })

        # Default opportunity
        if not opportunities:
            opportunities.append({
                "opportunity": "Compliance and Value",
                "description": "Standard competitive position - focus on clear compliance and best value",
                "priority": "medium"
            })

        return opportunities


class GhostingLanguageGenerator:
    """
    Generates ghosting language to subtly de-position competitors.

    Ghosting is the art of highlighting your strengths in a way that
    draws attention to competitor weaknesses without naming them directly.
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the Ghosting Language Generator

        Args:
            use_llm: Whether to use LLM for advanced language generation
        """
        self.use_llm = use_llm

        # Standard ghosting templates by category
        self.templates = {
            "incumbent_risk": [
                "Our fresh perspective brings innovative solutions unconstrained by legacy approaches",
                "We bring proven best practices refined across multiple similar engagements",
                "Our team is specifically assembled for this opportunity with no competing priorities",
            ],
            "transition": [
                "Our Day 1 readiness ensures zero disruption to mission operations",
                "Our proven transition methodology has achieved 100% on-time transitions",
                "We commit dedicated transition resources with no learning curve",
            ],
            "personnel": [
                "Our named key personnel are committed and available for immediate assignment",
                "We maintain deep bench strength ensuring continuity through any personnel changes",
                "Our team members average [X] years of directly relevant experience",
            ],
            "innovation": [
                "Our modern, cloud-native architecture enables rapid capability deployment",
                "We leverage cutting-edge technologies proven in production environments",
                "Our agile methodology delivers continuous improvement throughout performance",
            ],
            "scale": [
                "With [X] employees nationwide, we scale resources to meet surge demands",
                "Our national presence ensures local expertise wherever needed",
                "We maintain excess capacity specifically for rapid response requirements",
            ],
            "past_performance": [
                "Our directly relevant experience spans [X] similar contracts",
                "We have achieved [metric] across all comparable engagements",
                "Our customer satisfaction ratings consistently exceed [X]%",
            ],
        }

    def generate_ghosting_language(
        self,
        competitor_weaknesses: List[str],
        our_strengths: List[str],
        evaluation_factors: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate ghosting language based on competitor weaknesses and our strengths.

        Args:
            competitor_weaknesses: List of identified competitor weaknesses
            our_strengths: List of our relevant strengths
            evaluation_factors: Optional list of evaluation factors for targeting

        Returns:
            List of ghosting phrases with context
        """
        ghosting_phrases = []

        for weakness in competitor_weaknesses:
            weakness_lower = weakness.lower()

            # Match weakness to template category
            for category, templates in self.templates.items():
                if any(kw in weakness_lower for kw in category.split("_")):
                    # Select best template
                    phrase = templates[0]  # In production, use LLM to select/customize

                    ghosting_phrases.append({
                        "phrase": phrase,
                        "targets_weakness": weakness,
                        "category": category,
                        "usage_context": f"Use when addressing {category.replace('_', ' ')} in proposal",
                    })
                    break

        # Add strength-based phrases
        for strength in our_strengths[:3]:  # Top 3 strengths
            ghosting_phrases.append({
                "phrase": f"Our proven {strength.lower()} delivers measurable results",
                "targets_weakness": "general",
                "category": "strength_highlight",
                "usage_context": "General discriminator language",
            })

        return ghosting_phrases

    def get_templates(self) -> Dict[str, List[str]]:
        """Get all available ghosting templates"""
        return self.templates

    def add_custom_template(self, category: str, phrases: List[str]):
        """Add custom ghosting templates for a category"""
        if category in self.templates:
            self.templates[category].extend(phrases)
        else:
            self.templates[category] = phrases
