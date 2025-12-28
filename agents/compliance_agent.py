"""
PropelAI Compliance Agent - "The Paralegal"
Play 1: The Intelligent Shred (RFP Ingestion)

Goal: Reduce RFP analysis time from 3 days to <1 hour

This agent:
1. Ingests and structures RFP documents (PDF/DOCX)
2. Identifies the "Iron Triangle": Requirements (C), Instructions (L), Evaluation (M)
3. Extracts 'shall' statements and mandatory requirements
4. Builds the Requirements Traceability Matrix (RTM)
5. Generates the Compliance Matrix

Uses Gemini Flash for cost-efficient extraction (Model Cascading strategy)
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import hashlib

from core.state import ProposalState, ProposalPhase, ComplianceStatus


# Regex patterns for requirement extraction
REQUIREMENT_PATTERNS = {
    "mandatory": [
        r"\bshall\b",
        r"\bmust\b", 
        r"\bis required to\b",
        r"\bare required to\b",
        r"\bwill be required\b",
        r"\bmandatory\b",
        r"\bis responsible for\b",
    ],
    "conditional": [
        r"\bshould\b",
        r"\bmay\b",
        r"\bcan\b",
        r"\bis encouraged\b",
        r"\bis recommended\b",
    ],
    "prohibition": [
        r"\bshall not\b",
        r"\bmust not\b",
        r"\bwill not\b",
        r"\bprohibited\b",
        r"\bforbidden\b",
    ]
}

# Section identifiers
SECTION_PATTERNS = {
    "section_c": [
        r"section\s*c[\s:\-]+",
        r"statement\s+of\s+work",
        r"sow",
        r"performance\s+work\s+statement",
        r"pws",
        r"scope\s+of\s+work",
        r"technical\s+requirements",
    ],
    "section_l": [
        r"section\s*l[\s:\-]+",
        r"instructions\s+to\s+offerors",
        r"proposal\s+instructions",
        r"submission\s+requirements",
        r"format\s+requirements",
        r"page\s+limit",
    ],
    "section_m": [
        r"section\s*m[\s:\-]+",
        r"evaluation\s+criteria",
        r"evaluation\s+factors",
        r"basis\s+for\s+award",
        r"source\s+selection",
        r"scoring\s+criteria",
    ]
}


@dataclass
class ExtractedRequirement:
    """A requirement extracted from the RFP"""
    id: str
    text: str
    section_type: str           # "C", "L", "M", or "other"
    section_ref: str            # e.g., "C.4.2.1"
    requirement_type: str       # "mandatory", "conditional", "prohibition"
    keywords: List[str]
    context_before: str         # Surrounding text for context
    context_after: str
    confidence: float           # Extraction confidence score


class ComplianceAgent:
    """
    The Compliance Agent - "The Paralegal"
    
    Specialized in:
    - RFP structure recognition
    - Requirement extraction (shall/must statements)
    - Iron Triangle mapping (C -> L -> M)
    - Compliance matrix generation
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the Compliance Agent
        
        Args:
            llm_client: Optional LLM client (Gemini Flash recommended for cost efficiency)
        """
        self.llm_client = llm_client
        self.requirement_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for performance"""
        compiled = {}
        for category, patterns in REQUIREMENT_PATTERNS.items():
            compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def __call__(self, state: ProposalState) -> Dict[str, Any]:
        """
        Main entry point - called by the Orchestrator
        
        Processes the RFP and extracts requirements
        """
        start_time = datetime.now()
        
        rfp_text = state.get("rfp_raw_text", "")
        
        if not rfp_text:
            existing_trace = state.get("agent_trace_log", [])
            return {
                "error_state": "No RFP content to process",
                "agent_trace_log": existing_trace + [{
                    "timestamp": start_time.isoformat(),
                    "agent_name": "compliance_agent",
                    "action": "shred_rfp",
                    "input_summary": "Empty RFP text",
                    "output_summary": "Error: No content",
                    "reasoning_trace": "Cannot process empty RFP"
                }]
            }
        
        # Phase 1: Segment the document into sections
        sections = self._segment_document(rfp_text)
        
        # Phase 2: Extract requirements from each section
        requirements = []
        for section_type, section_text in sections.items():
            extracted = self._extract_requirements(section_text, section_type)
            requirements.extend(extracted)
        
        # Phase 3: Build the requirements graph (dependencies)
        requirements_graph = self._build_requirements_graph(requirements)
        
        # Phase 4: Generate compliance matrix
        compliance_matrix = self._generate_compliance_matrix(requirements)
        
        # Phase 5: Extract evaluation criteria from Section M
        evaluation_criteria = self._extract_evaluation_criteria(sections.get("section_m", ""))
        
        # Phase 6: Extract instructions from Section L
        instructions = self._extract_instructions(sections.get("section_l", ""))
        
        # Calculate processing time
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Build trace log
        trace_log = {
            "timestamp": start_time.isoformat(),
            "agent_name": "compliance_agent",
            "action": "shred_rfp",
            "input_summary": f"RFP with {len(rfp_text)} characters",
            "output_summary": f"Extracted {len(requirements)} requirements, {len(evaluation_criteria)} eval criteria",
            "reasoning_trace": f"Segmented into {len(sections)} sections. Built requirements graph with {len(requirements_graph)} nodes.",
            "duration_ms": duration_ms,
            "tool_calls": [
                {"tool": "segment_document", "result": f"{len(sections)} sections"},
                {"tool": "extract_requirements", "result": f"{len(requirements)} requirements"},
                {"tool": "build_graph", "result": f"{len(requirements_graph)} nodes"},
            ]
        }
        
        # Accumulate trace logs
        existing_trace = state.get("agent_trace_log", [])

        return {
            "current_phase": ProposalPhase.SHRED.value,
            "requirements": [self._requirement_to_dict(r) for r in requirements],
            "instructions": instructions,
            "evaluation_criteria": evaluation_criteria,
            "requirements_graph": requirements_graph,
            "compliance_matrix": compliance_matrix,
            "rfp_metadata": {
                "total_requirements": len(requirements),
                "mandatory_count": sum(1 for r in requirements if r.requirement_type == "mandatory"),
                "sections_found": list(sections.keys()),
                "processed_at": start_time.isoformat(),
            },
            "agent_trace_log": existing_trace + [trace_log],
            "updated_at": datetime.now().isoformat()
        }
    
    def _segment_document(self, text: str) -> Dict[str, str]:
        """
        Segment the RFP into Section C, L, M, and other sections
        
        Uses pattern matching to identify section boundaries
        """
        sections = {
            "section_c": "",
            "section_l": "",
            "section_m": "",
            "other": ""
        }
        
        # Split into paragraphs for processing
        paragraphs = text.split("\n\n")
        current_section = "other"
        current_content = []
        
        for para in paragraphs:
            para_lower = para.lower()
            
            # Check for section markers
            new_section = None
            for section_name, patterns in SECTION_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, para_lower):
                        new_section = section_name
                        break
                if new_section:
                    break
            
            # If we found a new section, save current and switch
            if new_section and new_section != current_section:
                if current_content:
                    sections[current_section] += "\n\n".join(current_content) + "\n\n"
                current_section = new_section
                current_content = [para]
            else:
                current_content.append(para)
        
        # Don't forget the last section
        if current_content:
            sections[current_section] += "\n\n".join(current_content)
        
        return sections
    
    def _extract_requirements(
        self, 
        text: str, 
        section_type: str
    ) -> List[ExtractedRequirement]:
        """
        Extract requirements (shall/must statements) from text
        
        Returns list of ExtractedRequirement objects
        """
        requirements = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for i, sentence in enumerate(sentences):
            # Check against all requirement patterns
            for req_type, patterns in self.requirement_patterns.items():
                for pattern in patterns:
                    if pattern.search(sentence):
                        # Found a requirement
                        req_id = self._generate_requirement_id(sentence, section_type)
                        
                        # Get context
                        context_before = sentences[i-1] if i > 0 else ""
                        context_after = sentences[i+1] if i < len(sentences)-1 else ""
                        
                        # Extract section reference (e.g., "C.4.2.1")
                        section_ref = self._extract_section_reference(sentence, context_before)
                        
                        # Extract keywords
                        keywords = self._extract_keywords(sentence)
                        
                        req = ExtractedRequirement(
                            id=req_id,
                            text=sentence.strip(),
                            section_type=section_type.replace("section_", "").upper(),
                            section_ref=section_ref,
                            requirement_type=req_type,
                            keywords=keywords,
                            context_before=context_before[:200],
                            context_after=context_after[:200],
                            confidence=0.85  # Base confidence, could be enhanced with LLM
                        )
                        requirements.append(req)
                        break  # Don't double-count
                else:
                    continue
                break
        
        return requirements
    
    def _extract_section_reference(self, sentence: str, context: str) -> str:
        """Extract section reference number (e.g., 'C.4.2.1')"""
        # Pattern for section references
        ref_pattern = r'\b([A-Z]\.\d+(?:\.\d+)*)\b'
        
        # Check sentence first
        match = re.search(ref_pattern, sentence)
        if match:
            return match.group(1)
        
        # Check context
        match = re.search(ref_pattern, context)
        if match:
            return match.group(1)
        
        return "UNREF"
    
    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract important keywords from requirement text"""
        # Remove common words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "shall", "should",
            "may", "might", "must", "can", "could", "would", "of", "to", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "and", "but", "or", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "now", "here", "there",
            "when", "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "any",
            "contractor", "offeror", "government", "agency", "required"
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        keywords = [w for w in words if w not in stopwords]
        
        # Return unique keywords
        return list(set(keywords))[:10]
    
    def _generate_requirement_id(self, text: str, section: str) -> str:
        """Generate a unique ID for a requirement"""
        hash_input = f"{section}:{text[:100]}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        prefix = section.replace("section_", "").upper()
        return f"REQ-{prefix}-{hash_value}"
    
    def _build_requirements_graph(
        self, 
        requirements: List[ExtractedRequirement]
    ) -> Dict[str, List[str]]:
        """
        Build a dependency graph linking requirements
        
        Maps Section C requirements to Section L instructions and Section M criteria
        """
        graph = {}
        
        # Group requirements by section
        by_section = {"C": [], "L": [], "M": [], "OTHER": []}
        for req in requirements:
            section = req.section_type if req.section_type in by_section else "OTHER"
            by_section[section].append(req)
        
        # For each Section C requirement, find related L and M items
        for c_req in by_section["C"]:
            linked = []
            c_keywords = set(c_req.keywords)
            
            # Find related Section L instructions
            for l_req in by_section["L"]:
                l_keywords = set(l_req.keywords)
                overlap = c_keywords & l_keywords
                if len(overlap) >= 2:  # At least 2 shared keywords
                    linked.append(l_req.id)
            
            # Find related Section M criteria
            for m_req in by_section["M"]:
                m_keywords = set(m_req.keywords)
                overlap = c_keywords & m_keywords
                if len(overlap) >= 2:
                    linked.append(m_req.id)
            
            graph[c_req.id] = linked
        
        return graph
    
    def _generate_compliance_matrix(
        self, 
        requirements: List[ExtractedRequirement]
    ) -> List[Dict[str, Any]]:
        """
        Generate the Compliance Matrix (Excel-ready format)
        
        Columns: Requirement ID, Text, Section Ref, Type, Status, Response, Owner
        """
        matrix = []
        
        for req in requirements:
            row = {
                "requirement_id": req.id,
                "requirement_text": req.text,
                "section_reference": req.section_ref,
                "section_type": req.section_type,
                "requirement_type": req.requirement_type,
                "keywords": req.keywords,
                "compliance_status": ComplianceStatus.NOT_STARTED.value,
                "compliant_response": "",
                "assigned_owner": "",
                "assigned_section": "",
                "notes": "",
                "confidence": req.confidence
            }
            matrix.append(row)
        
        return matrix
    
    def _extract_evaluation_criteria(self, section_m_text: str) -> List[Dict[str, Any]]:
        """Extract evaluation criteria from Section M"""
        criteria = []
        
        if not section_m_text:
            return criteria
        
        # Common evaluation factor patterns
        factor_patterns = [
            r"(?:factor|criterion)\s*(\d+)[:\s]+([^\n]+)",
            r"(?:technical|management|past\s+performance|price|cost)[:\s]+",
            r"\b(most\s+important|equally\s+important|descending\s+order)\b"
        ]
        
        # Split into paragraphs
        paragraphs = section_m_text.split("\n\n")
        
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            
            # Check if this paragraph describes an evaluation factor
            if any(kw in para_lower for kw in ["technical", "management", "past performance", "price", "cost", "factor", "criterion"]):
                criterion = {
                    "id": f"EVAL-{i:03d}",
                    "text": para.strip()[:500],
                    "section_ref": f"M.{i+1}",
                    "factor_name": self._identify_factor_name(para),
                    "weight": self._extract_weight(para),
                    "subfactors": []
                }
                criteria.append(criterion)
        
        return criteria
    
    def _identify_factor_name(self, text: str) -> str:
        """Identify the evaluation factor name"""
        text_lower = text.lower()
        
        if "technical" in text_lower:
            if "approach" in text_lower:
                return "Technical Approach"
            return "Technical"
        elif "management" in text_lower:
            return "Management Approach"
        elif "past performance" in text_lower:
            return "Past Performance"
        elif "price" in text_lower or "cost" in text_lower:
            return "Price/Cost"
        elif "staffing" in text_lower or "key personnel" in text_lower:
            return "Staffing/Key Personnel"
        else:
            return "Other"
    
    def _extract_weight(self, text: str) -> Optional[float]:
        """Extract relative weight of an evaluation factor"""
        text_lower = text.lower()
        
        # Look for percentage weights
        match = re.search(r'(\d+)\s*%', text)
        if match:
            return float(match.group(1)) / 100
        
        # Look for relative importance statements
        if "most important" in text_lower or "highest priority" in text_lower:
            return 0.4
        elif "equally important" in text_lower:
            return 0.33
        elif "less important" in text_lower or "secondary" in text_lower:
            return 0.2
        
        return None
    
    def _extract_instructions(self, section_l_text: str) -> List[Dict[str, Any]]:
        """Extract submission instructions from Section L"""
        instructions = []
        
        if not section_l_text:
            return instructions
        
        # Split into paragraphs
        paragraphs = section_l_text.split("\n\n")
        
        for i, para in enumerate(paragraphs):
            para_lower = para.lower()
            
            # Check for instruction-like content
            instruction_keywords = ["page", "volume", "submit", "format", "font", "margin", "limit", "requirement", "include", "provide"]
            
            if any(kw in para_lower for kw in instruction_keywords):
                instruction = {
                    "id": f"INST-{i:03d}",
                    "text": para.strip()[:500],
                    "section_ref": f"L.{i+1}",
                    "instruction_type": self._classify_instruction(para),
                    "page_limit": self._extract_page_limit(para),
                    "format_requirements": self._extract_format_requirements(para)
                }
                instructions.append(instruction)
        
        return instructions
    
    def _classify_instruction(self, text: str) -> str:
        """Classify the type of instruction"""
        text_lower = text.lower()
        
        if "page" in text_lower and ("limit" in text_lower or "maximum" in text_lower):
            return "page_limit"
        elif "format" in text_lower or "font" in text_lower or "margin" in text_lower:
            return "formatting"
        elif "submit" in text_lower or "delivery" in text_lower:
            return "submission"
        elif "volume" in text_lower:
            return "volume_structure"
        else:
            return "general"
    
    def _extract_page_limit(self, text: str) -> Optional[int]:
        """Extract page limit from instruction text"""
        match = re.search(r'(\d+)\s*(?:page|pg)s?(?:\s+(?:maximum|limit))?', text.lower())
        if match:
            return int(match.group(1))
        return None
    
    def _extract_format_requirements(self, text: str) -> Dict[str, Any]:
        """Extract formatting requirements (font, margins, etc.)"""
        requirements = {}
        text_lower = text.lower()
        
        # Font size
        match = re.search(r'(\d+)\s*(?:-?\s*)?point', text_lower)
        if match:
            requirements["font_size"] = int(match.group(1))
        
        # Margins
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:inch|")\s*margin', text_lower)
        if match:
            requirements["margin_inches"] = float(match.group(1))
        
        # Line spacing
        if "single" in text_lower and "spac" in text_lower:
            requirements["line_spacing"] = "single"
        elif "double" in text_lower and "spac" in text_lower:
            requirements["line_spacing"] = "double"
        
        return requirements
    
    def _requirement_to_dict(self, req: ExtractedRequirement) -> Dict[str, Any]:
        """Convert ExtractedRequirement to dictionary for state storage"""
        return {
            "id": req.id,
            "text": req.text,
            "section_type": req.section_type,
            "section_ref": req.section_ref,
            "requirement_type": req.requirement_type,
            "keywords": req.keywords,
            "context_before": req.context_before,
            "context_after": req.context_after,
            "confidence": req.confidence
        }


# Convenience function for creating the agent
def create_compliance_agent(llm_client: Optional[Any] = None) -> ComplianceAgent:
    """Factory function to create a Compliance Agent"""
    return ComplianceAgent(llm_client=llm_client)
