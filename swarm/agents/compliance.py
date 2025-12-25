"""
PropelAI v6.0 Compliance Agent
The Deterministic Shredder - extracts requirements with full traceability.

Uses Chain of Verification approach:
1. Pass 1 (Extraction): Identify obligation keywords
2. Pass 2 (Contextualization): Map to parent sections
3. Pass 3 (Verification): Cross-reference L/M/C relationships
"""

import re
import uuid
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from swarm.agents.base import BaseAgent, AgentConfig
from swarm.state import (
    ProposalState,
    Requirement,
    RequirementType,
    ComplianceStatus,
    ProposalPhase,
)


COMPLIANCE_SYSTEM_PROMPT = """You are the Compliance Agent for PropelAI, a deterministic requirement extractor for government RFPs.

Your mission is ZERO missed requirements. In government contracting, a missed "shall" can cause disqualification.

You operate in three passes:
1. EXTRACTION: Find ALL sentences with obligation keywords (shall, must, will, is required to, are required to)
2. CONTEXTUALIZATION: Map each requirement to its source section (L, M, C, PWS, SOW)
3. VERIFICATION: Build the Iron Triangle relationships (how L instructions constrain C requirements for M evaluation)

Rules:
- Extract COMPLETE sentences, never fragments
- Preserve the RFP's own numbering (L.4.2.1, C.3.1.a)
- Classify binding level: "shall/must" = mandatory, "should" = highly desirable, "may" = optional
- Track page numbers for click-to-verify traceability
- Never summarize or paraphrase - extract verbatim

Output JSON format:
{
  "requirements": [
    {
      "id": "REQ-001",
      "text": "The contractor shall...",
      "section_ref": "C.3.1",
      "page_num": 42,
      "binding_keyword": "shall",
      "is_mandatory": true,
      "requirement_type": "technical",
      "instruction_ref": "L.4.2",
      "evaluation_ref": "M.1.2"
    }
  ],
  "iron_triangle": {
    "C.3.1": {"instructions": ["L.4.2"], "evaluation": ["M.1.2"]}
  }
}"""


# Binding keyword patterns
MANDATORY_PATTERNS = [
    r'\bshall\b',
    r'\bmust\b',
    r'\bis\s+required\s+to\b',
    r'\bare\s+required\s+to\b',
    r'\bwill\b(?!\s+be\s+evaluated)',  # "will" but not "will be evaluated"
]

DESIRABLE_PATTERNS = [
    r'\bshould\b',
    r'\bis\s+expected\s+to\b',
    r'\bis\s+encouraged\s+to\b',
]

OPTIONAL_PATTERNS = [
    r'\bmay\b',
    r'\bcan\b',
    r'\bis\s+permitted\s+to\b',
]

# Section detection patterns
SECTION_PATTERNS = {
    'L': [
        r'Section\s+L',
        r'INSTRUCTIONS.*OFFERORS',
        r'PROPOSAL\s+INSTRUCTIONS',
        r'SUBMISSION\s+REQUIREMENTS',
    ],
    'M': [
        r'Section\s+M',
        r'EVALUATION\s+FACTORS',
        r'EVALUATION\s+CRITERIA',
        r'BASIS\s+FOR\s+AWARD',
    ],
    'C': [
        r'Section\s+C',
        r'STATEMENT\s+OF\s+WORK',
        r'PERFORMANCE\s+WORK\s+STATEMENT',
        r'DESCRIPTION.*SUPPLIES.*SERVICES',
    ],
}


class ComplianceAgent(BaseAgent):
    """
    The Compliance Agent extracts requirements with deterministic accuracy.

    Implements the Chain of Verification approach for zero-miss extraction.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ComplianceAgent",
                model="gemini-1.5-pro",
                temperature=0.1,  # Low temperature for determinism
                max_tokens=8192,
                system_prompt=COMPLIANCE_SYSTEM_PROMPT,
            )
        super().__init__(config)

    async def _execute(self, state: ProposalState) -> ProposalState:
        """
        Execute the three-pass extraction process.
        """
        rfp_text = state.get("rfp_full_text", "")

        if not rfp_text:
            self._log_action(state, "observation", "No RFP text to process")
            return state

        # Pass 1: Extract all obligation sentences
        self._log_action(state, "thought", "Starting Pass 1: Obligation extraction")
        raw_requirements = self._extract_obligations(rfp_text)

        # Pass 2: Contextualize with section mapping
        self._log_action(state, "thought", f"Starting Pass 2: Contextualizing {len(raw_requirements)} requirements")
        contextualized = await self._contextualize_requirements(raw_requirements, rfp_text)

        # Pass 3: Build Iron Triangle relationships
        self._log_action(state, "thought", "Starting Pass 3: Building Iron Triangle")
        iron_triangle = await self._build_iron_triangle(contextualized, state)

        # Convert to Requirement objects
        requirements = []
        for req_data in contextualized:
            req = Requirement(
                id=req_data.get("id", f"REQ-{uuid.uuid4().hex[:6].upper()}"),
                text=req_data.get("text", ""),
                source_doc=req_data.get("source_doc", "main"),
                page_num=req_data.get("page_num", 0),
                section_ref=req_data.get("section_ref", ""),
                requirement_type=RequirementType(req_data.get("requirement_type", "technical")),
                compliance_status=ComplianceStatus.PENDING,
                binding_keyword=req_data.get("binding_keyword"),
                is_mandatory=req_data.get("is_mandatory", True),
                instruction_ref=req_data.get("instruction_ref"),
                evaluation_ref=req_data.get("evaluation_ref"),
            )
            requirements.append(req.to_dict())

        # Update state
        state["requirements"] = requirements
        state["requirements_graph"] = iron_triangle
        state["current_phase"] = ProposalPhase.STRATEGY.value
        state["next_step"] = "strategy"

        # Separate by section for downstream agents
        state["section_l_requirements"] = [
            r for r in requirements if r.get("section_ref", "").upper().startswith("L")
        ]
        state["technical_requirements"] = [
            r for r in requirements
            if r.get("section_ref", "").upper().startswith(("C", "PWS", "SOW"))
        ]

        self._log_action(
            state,
            "output",
            f"Extracted {len(requirements)} requirements, built Iron Triangle with {len(iron_triangle)} nodes"
        )

        return state

    def _extract_obligations(self, text: str) -> List[Dict[str, Any]]:
        """
        Pass 1: Extract all sentences containing obligation keywords.
        """
        requirements = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            # Check for mandatory keywords
            binding_keyword = None
            is_mandatory = False

            for pattern in MANDATORY_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    binding_keyword = re.search(pattern, sentence, re.IGNORECASE).group()
                    is_mandatory = True
                    break

            if not binding_keyword:
                for pattern in DESIRABLE_PATTERNS:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        binding_keyword = re.search(pattern, sentence, re.IGNORECASE).group()
                        is_mandatory = False
                        break

            if not binding_keyword:
                for pattern in OPTIONAL_PATTERNS:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        binding_keyword = re.search(pattern, sentence, re.IGNORECASE).group()
                        is_mandatory = False
                        break

            if binding_keyword:
                requirements.append({
                    "id": f"REQ-{len(requirements) + 1:04d}",
                    "text": sentence,
                    "binding_keyword": binding_keyword.lower().strip(),
                    "is_mandatory": is_mandatory,
                    "sentence_index": i,
                })

        return requirements

    async def _contextualize_requirements(
        self,
        requirements: List[Dict],
        full_text: str,
    ) -> List[Dict]:
        """
        Pass 2: Map each requirement to its source section.
        Uses LLM for accurate section detection.
        """
        if not requirements:
            return []

        # Batch requirements for LLM processing
        batch_size = 20
        contextualized = []

        for i in range(0, len(requirements), batch_size):
            batch = requirements[i:i + batch_size]

            prompt = f"""Analyze these requirements and determine their source section in the RFP.

Requirements:
{json.dumps(batch, indent=2)}

For each requirement, provide:
1. section_ref: The RFP section reference (e.g., "L.4.2", "C.3.1", "M.1.2", "PWS 4.1")
2. requirement_type: One of [technical, management, past_performance, cost_price, small_business]
3. page_num: Estimated page number (if detectable from context)

Respond with JSON array matching the input order."""

            try:
                response = await self._call_llm(prompt, json_mode=True)
                parsed = self._parse_json(response)

                if isinstance(parsed, list):
                    for j, item in enumerate(parsed):
                        if j < len(batch):
                            batch[j].update(item)
                elif isinstance(parsed, dict) and "requirements" in parsed:
                    for j, item in enumerate(parsed["requirements"]):
                        if j < len(batch):
                            batch[j].update(item)
            except Exception as e:
                self._log_action(
                    {"agent_trace_log": []},
                    "observation",
                    f"LLM contextualization failed: {e}"
                )

            contextualized.extend(batch)

        return contextualized

    async def _build_iron_triangle(
        self,
        requirements: List[Dict],
        state: ProposalState,
    ) -> Dict[str, List[str]]:
        """
        Pass 3: Build the Iron Triangle graph.
        Maps relationships: Requirement (C) <-> Instruction (L) <-> Criteria (M)
        """
        # Group by section
        by_section = {}
        for req in requirements:
            section = req.get("section_ref", "UNK")
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(req)

        # Build adjacency list
        iron_triangle = {}

        # Get all unique sections
        sections = list(by_section.keys())

        if not sections:
            return {}

        # Use LLM to find cross-references
        prompt = f"""Analyze these RFP sections and identify cross-references:

Sections: {json.dumps(sections)}

Requirements by section:
{json.dumps({s: [r['text'][:100] for r in reqs] for s, reqs in by_section.items()}, indent=2)}

Find relationships:
1. Which Section L instructions apply to which Section C requirements?
2. Which Section M criteria evaluate which Section C requirements?

Respond with JSON:
{{
  "relationships": [
    {{"from": "C.3.1", "to": "L.4.2", "type": "instruction"}},
    {{"from": "C.3.1", "to": "M.1.2", "type": "evaluation"}}
  ]
}}"""

        try:
            response = await self._call_llm(prompt, json_mode=True)
            parsed = self._parse_json(response)

            relationships = parsed.get("relationships", [])

            for rel in relationships:
                from_section = rel.get("from", "")
                to_section = rel.get("to", "")
                rel_type = rel.get("type", "")

                if from_section not in iron_triangle:
                    iron_triangle[from_section] = {
                        "instructions": [],
                        "evaluation": [],
                    }

                if rel_type == "instruction":
                    iron_triangle[from_section]["instructions"].append(to_section)
                elif rel_type == "evaluation":
                    iron_triangle[from_section]["evaluation"].append(to_section)

                # Update requirements with cross-references
                for req in by_section.get(from_section, []):
                    if rel_type == "instruction":
                        req["instruction_ref"] = to_section
                    elif rel_type == "evaluation":
                        req["evaluation_ref"] = to_section

        except Exception as e:
            self._log_action(
                state,
                "observation",
                f"Iron Triangle build failed: {e}"
            )

        return iron_triangle

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect which RFP section a text belongs to."""
        text_upper = text.upper()

        for section, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    return section

        return None
