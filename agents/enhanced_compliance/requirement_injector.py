"""
PropelAI: Requirement Injector v1.0
Master Architect Plan - P0 Implementation

Injects extracted L/M/C requirements into outline sections,
replacing placeholders with actual RFP text.

Based on s.rtf blueprint:
- Match requirements to sections based on keywords
- Link Section L instructions to proposal sections
- Map Section M evaluation criteria for scoring emphasis
- Connect Section C/PWS tasks to proof point placeholders
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class InjectedRequirement:
    """A requirement matched to a section"""
    id: str
    text: str
    source_section: str  # L, M, or C
    section_ref: Optional[str] = None
    match_score: float = 0.0
    match_reason: str = ""


class RequirementInjector:
    """
    Injects extracted L/M/C requirements into outline sections.

    Replaces generic placeholders like "[Review Section L and add specific
    requirements for this section]" with actual extracted requirements.
    """

    def __init__(self):
        # Section keyword mappings for matching
        self.section_keywords = {
            "executive_summary": [
                "executive summary", "overview", "understanding", "summary",
                "introduction", "offeror", "capability statement"
            ],
            "technical_approach": [
                "technical approach", "methodology", "solution", "design",
                "implementation", "technical proposal", "approach"
            ],
            "management_approach": [
                "management", "project management", "program management",
                "quality", "schedule", "risk", "oversight"
            ],
            "past_performance": [
                "past performance", "experience", "prior contract", "reference",
                "similar work", "relevant experience", "cpars"
            ],
            "staffing": [
                "staffing", "personnel", "key personnel", "resume", "staff",
                "qualifications", "team", "organizational"
            ],
            "cost_price": [
                "cost", "price", "pricing", "labor rate", "clin",
                "basis of estimate", "budget"
            ],
            "transition": [
                "transition", "phase-in", "takeover", "startup",
                "knowledge transfer"
            ],
            "quality_assurance": [
                "quality", "qa", "qc", "quality assurance", "quality control",
                "inspection", "metrics"
            ],
            "small_business": [
                "small business", "subcontracting", "mentor", "protege",
                "participation goal"
            ],
        }

        # Category mappings from CTM extraction
        self.category_to_source = {
            "L_COMPLIANCE": "L",
            "EVALUATION": "M",
            "TECHNICAL": "C",
            "PWS": "C",
            "SOW": "C",
            "INSPECTION": "M",  # Often evaluation-related
        }

    def inject_requirements(
        self,
        outline_data: Dict[str, Any],
        requirements: List[Dict[str, Any]],
        compliance_matrix: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Inject requirements into outline sections.

        Args:
            outline_data: Proposal outline from SmartOutlineGenerator.to_json()
            requirements: List of requirements from CTM extraction
            compliance_matrix: Optional compliance matrix for traceability

        Returns:
            Updated outline_data with requirements injected into sections
        """
        # Categorize requirements by source section
        l_requirements = self._filter_by_source(requirements, "L")
        m_requirements = self._filter_by_source(requirements, "M")
        c_requirements = self._filter_by_source(requirements, "C")

        # Process each volume
        volumes = outline_data.get("volumes", [])
        for volume in volumes:
            volume_type = volume.get("volume_type") or volume.get("type", "technical")

            # Process each section in the volume
            sections = volume.get("sections", [])
            for section in sections:
                section_name = section.get("name", "") or section.get("title", "")

                # Match and inject requirements
                matched_l = self._match_requirements_to_section(
                    l_requirements, section_name, volume_type
                )
                matched_m = self._match_requirements_to_section(
                    m_requirements, section_name, volume_type
                )
                matched_c = self._match_requirements_to_section(
                    c_requirements, section_name, volume_type
                )

                # Update section with matched requirements
                section["l_requirements"] = matched_l
                section["m_requirements"] = matched_m
                section["c_requirements"] = matched_c

                # Also update the generic requirements field
                all_matched = matched_l + matched_m + matched_c
                if all_matched:
                    section["requirements"] = [r["text"] for r in all_matched[:10]]
                    section["has_injected_requirements"] = True

        return outline_data

    def _filter_by_source(
        self,
        requirements: List[Dict],
        source: str
    ) -> List[Dict]:
        """Filter requirements by source section (L, M, or C)"""
        result = []

        for req in requirements:
            # Check explicit section field
            section = (req.get("section") or "").upper()
            category = req.get("category", "")

            # Map category to source
            mapped_source = self.category_to_source.get(category, "")

            if section == source or mapped_source == source:
                result.append(req)
            elif source == "L" and category == "L_COMPLIANCE":
                result.append(req)
            elif source == "M" and category in ["EVALUATION", "INSPECTION"]:
                result.append(req)
            elif source == "C" and category in ["TECHNICAL", "PWS", "SOW"]:
                result.append(req)

        return result

    def _match_requirements_to_section(
        self,
        requirements: List[Dict],
        section_name: str,
        volume_type: str
    ) -> List[Dict]:
        """
        Match requirements to a specific section based on keywords.

        Returns requirements sorted by match score (best matches first).
        """
        if not requirements or not section_name:
            return []

        section_lower = section_name.lower()
        volume_lower = volume_type.lower() if volume_type else ""

        matched = []

        for req in requirements:
            text = req.get("text", "") or req.get("full_text", "") or ""
            text_lower = text.lower()

            score = 0.0
            reasons = []

            # Check for section name keywords in requirement text
            for section_key, keywords in self.section_keywords.items():
                if any(kw in section_lower for kw in keywords):
                    # This section matches a known type
                    for kw in keywords:
                        if kw in text_lower:
                            score += 0.3
                            reasons.append(f"keyword '{kw}'")
                            break

            # Check for volume type alignment
            if volume_lower in text_lower:
                score += 0.2
                reasons.append(f"volume type '{volume_lower}'")

            # Check for explicit section mentions
            if section_lower in text_lower:
                score += 0.5
                reasons.append("exact section match")

            # Past performance volume gets PP requirements
            if "past" in volume_lower and "past" in text_lower:
                score += 0.4
                reasons.append("past performance alignment")

            # Cost/price volume gets cost requirements
            if ("cost" in volume_lower or "price" in volume_lower):
                if "cost" in text_lower or "price" in text_lower:
                    score += 0.4
                    reasons.append("cost/price alignment")

            # Technical volume gets technical requirements
            if "technical" in volume_lower:
                if any(kw in text_lower for kw in ["approach", "methodology", "solution", "technical"]):
                    score += 0.3
                    reasons.append("technical alignment")

            # Management gets management requirements
            if "management" in volume_lower or "management" in section_lower:
                if any(kw in text_lower for kw in ["management", "schedule", "risk", "quality"]):
                    score += 0.3
                    reasons.append("management alignment")

            # Only include if there's a meaningful match
            if score > 0.2:
                matched.append({
                    "id": req.get("id", ""),
                    "text": text[:500],  # Truncate long requirements
                    "source": req.get("section", ""),
                    "category": req.get("category", ""),
                    "score": score,
                    "match_reason": ", ".join(reasons[:3])
                })

        # Sort by score and return top matches
        matched.sort(key=lambda x: x["score"], reverse=True)
        return matched[:5]  # Top 5 matches per section

    def create_traceability_matrix(
        self,
        outline_data: Dict[str, Any],
        requirements: List[Dict]
    ) -> List[Dict]:
        """
        Create a traceability matrix linking requirements to proposal sections.

        Returns list of mappings: Req ID | Requirement Text | L Ref | M Ref | Proposal Section
        """
        matrix = []

        volumes = outline_data.get("volumes", [])
        for volume in volumes:
            volume_name = volume.get("name", "")

            for section in volume.get("sections", []):
                section_name = section.get("name", "") or section.get("title", "")
                proposal_section = f"{volume_name} > {section_name}"

                # Add L requirements
                for req in section.get("l_requirements", []):
                    matrix.append({
                        "req_id": req.get("id", ""),
                        "requirement_text": req.get("text", "")[:200],
                        "l_ref": req.get("source", "L"),
                        "m_ref": "",
                        "proposal_section": proposal_section,
                        "match_score": req.get("score", 0)
                    })

                # Add M requirements
                for req in section.get("m_requirements", []):
                    matrix.append({
                        "req_id": req.get("id", ""),
                        "requirement_text": req.get("text", "")[:200],
                        "l_ref": "",
                        "m_ref": req.get("source", "M"),
                        "proposal_section": proposal_section,
                        "match_score": req.get("score", 0)
                    })

        return matrix


# =============================================================================
# Factory Function
# =============================================================================

def create_requirement_injector() -> RequirementInjector:
    """Create a new RequirementInjector instance"""
    return RequirementInjector()


def inject_requirements_into_outline(
    outline_data: Dict[str, Any],
    requirements: List[Dict],
    compliance_matrix: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Convenience function to inject requirements into an outline.

    Args:
        outline_data: Outline from SmartOutlineGenerator.to_json()
        requirements: Requirements from CTM extraction
        compliance_matrix: Optional compliance matrix

    Returns:
        Updated outline with injected requirements
    """
    injector = RequirementInjector()
    return injector.inject_requirements(outline_data, requirements, compliance_matrix)
