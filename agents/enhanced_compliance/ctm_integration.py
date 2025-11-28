"""
PropelAI CTM Integration Module v3.0
Bridges the enhanced CTM data models with the existing extraction pipeline

This module provides:
1. Adapter to convert legacy requirement dicts to EnhancedRequirement objects
2. Post-processing to enrich existing CTM output with v3.0 metadata
3. API endpoint helpers for the new fields

Author: PropelAI Team
Version: 3.0.0
Date: November 28, 2025
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from .ctm_data_models import (
    EnhancedRequirement,
    ComplianceMatrix,
    ScoringType,
    ResponseFormat,
    RequirementType,
    RFPSection,
    ComplianceStatus,
    PageLimit,
    FormattingRequirement,
)
from .ctm_extractor import EnhancedCTMExtractor


# =============================================================================
# LEGACY ADAPTER
# =============================================================================

class LegacyRequirementAdapter:
    """
    Adapts legacy requirement dictionaries to EnhancedRequirement objects.
    
    The existing best_practices_ctm.py outputs requirements as dictionaries.
    This adapter converts them to the new EnhancedRequirement format while
    enriching them with v3.0 metadata.
    """
    
    def __init__(self, rfp_format: str = "STANDARD_UCF"):
        """
        Initialize the adapter.
        
        Args:
            rfp_format: The detected RFP format (NIH_FACTOR, GSA_BPA, etc.)
        """
        self.extractor = EnhancedCTMExtractor(rfp_format)
        self.rfp_format = rfp_format
    
    def convert_requirement(
        self,
        legacy_req: Dict[str, Any],
        context: str = ""
    ) -> EnhancedRequirement:
        """
        Convert a legacy requirement dict to EnhancedRequirement.
        
        Args:
            legacy_req: Legacy requirement dictionary with keys like:
                - requirement_text or text
                - section_reference or section_ref
                - rfp_section or section
                - priority or priority_score
                - requirement_type or type
            context: Additional context for extraction
        
        Returns:
            EnhancedRequirement with all v3.0 fields populated
        """
        # Extract text
        text = legacy_req.get('requirement_text') or legacy_req.get('text', '')
        
        # Extract section reference
        section_ref = (
            legacy_req.get('section_reference') or 
            legacy_req.get('section_ref') or 
            legacy_req.get('reference', '')
        )
        
        # Map RFP section
        section_str = (
            legacy_req.get('rfp_section') or 
            legacy_req.get('section', 'OTHER')
        )
        rfp_section = self._map_rfp_section(section_str)
        
        # Use extractor to get enhanced metadata
        enhanced = self.extractor.extract_all_metadata(
            requirement_text=text,
            section_reference=section_ref,
            rfp_section=rfp_section,
            context=context
        )
        
        # Preserve legacy fields that might be better than extraction
        if 'priority' in legacy_req or 'priority_score' in legacy_req:
            legacy_priority = legacy_req.get('priority') or legacy_req.get('priority_score')
            if legacy_priority and isinstance(legacy_priority, (int, float)):
                enhanced.priority_score = int(legacy_priority)
        
        if 'requirement_type' in legacy_req or 'type' in legacy_req:
            legacy_type = legacy_req.get('requirement_type') or legacy_req.get('type')
            if legacy_type:
                enhanced.requirement_type = self._map_requirement_type(legacy_type)
        
        # Copy source tracking
        enhanced.source_document = legacy_req.get('source_document') or legacy_req.get('source_file')
        enhanced.source_page = legacy_req.get('source_page') or legacy_req.get('page')
        
        # Copy any existing ID
        if 'id' in legacy_req:
            enhanced.id = legacy_req['id']
        
        # Copy notes
        if 'notes' in legacy_req:
            enhanced.notes = legacy_req['notes']
        
        return enhanced
    
    def convert_batch(
        self,
        legacy_requirements: List[Dict[str, Any]],
        section_context: str = ""
    ) -> List[EnhancedRequirement]:
        """
        Convert a batch of legacy requirements.
        
        Args:
            legacy_requirements: List of legacy requirement dicts
            section_context: Context about the section these came from
        
        Returns:
            List of EnhancedRequirement objects
        """
        return [
            self.convert_requirement(req, section_context)
            for req in legacy_requirements
        ]
    
    def _map_rfp_section(self, section_str: str) -> RFPSection:
        """Map string to RFPSection enum."""
        section_str = str(section_str).upper().strip()
        
        # Direct mapping
        mapping = {
            'A': RFPSection.SECTION_A,
            'B': RFPSection.SECTION_B,
            'C': RFPSection.SECTION_C,
            'D': RFPSection.SECTION_D,
            'E': RFPSection.SECTION_E,
            'F': RFPSection.SECTION_F,
            'G': RFPSection.SECTION_G,
            'H': RFPSection.SECTION_H,
            'I': RFPSection.SECTION_I,
            'J': RFPSection.SECTION_J,
            'K': RFPSection.SECTION_K,
            'L': RFPSection.SECTION_L,
            'M': RFPSection.SECTION_M,
            'SECTION_L': RFPSection.SECTION_L,
            'SECTION_M': RFPSection.SECTION_M,
            'SECTION_C': RFPSection.SECTION_C,
        }
        
        return mapping.get(section_str, RFPSection.OTHER)
    
    def _map_requirement_type(self, type_str: str) -> RequirementType:
        """Map string to RequirementType enum."""
        type_str = str(type_str).lower().strip()
        
        mapping = {
            'technical': RequirementType.TECHNICAL,
            'management': RequirementType.MANAGEMENT,
            'past_performance': RequirementType.PAST_PERFORMANCE,
            'past performance': RequirementType.PAST_PERFORMANCE,
            'key_personnel': RequirementType.KEY_PERSONNEL,
            'key personnel': RequirementType.KEY_PERSONNEL,
            'cost': RequirementType.COST_PRICE,
            'price': RequirementType.COST_PRICE,
            'cost_price': RequirementType.COST_PRICE,
            'formatting': RequirementType.FORMATTING,
            'administrative': RequirementType.ADMINISTRATIVE,
            'security': RequirementType.SECURITY,
            'transition': RequirementType.TRANSITION,
            'compliance': RequirementType.COMPLIANCE,
        }
        
        return mapping.get(type_str, RequirementType.OTHER)


# =============================================================================
# CTM ENRICHMENT
# =============================================================================

class CTMEnricher:
    """
    Enriches existing CTM output with v3.0 metadata.
    
    This can be used as a post-processor for the existing pipeline.
    """
    
    def __init__(self, rfp_format: str = "STANDARD_UCF"):
        self.adapter = LegacyRequirementAdapter(rfp_format)
        self.rfp_format = rfp_format
    
    def enrich_ctm_output(
        self,
        ctm_data: Dict[str, Any],
        rfp_id: str,
        rfp_name: str
    ) -> ComplianceMatrix:
        """
        Enrich a CTM output dictionary with v3.0 metadata.
        
        Args:
            ctm_data: The existing CTM output with keys like:
                - section_l_requirements
                - section_m_requirements
                - technical_requirements
                - stats
            rfp_id: RFP identifier
            rfp_name: RFP name/title
        
        Returns:
            ComplianceMatrix with all requirements enriched
        """
        matrix = ComplianceMatrix(
            rfp_id=rfp_id,
            rfp_name=rfp_name,
            rfp_format=self.rfp_format
        )
        
        # Process Section L requirements
        section_l = ctm_data.get('section_l_requirements', [])
        for req in section_l:
            enhanced = self.adapter.convert_requirement(
                req, 
                context="Section L - Instructions to Offerors"
            )
            enhanced.rfp_section = RFPSection.SECTION_L
            matrix.add_requirement(enhanced)
        
        # Process Section M requirements
        section_m = ctm_data.get('section_m_requirements', [])
        for req in section_m:
            enhanced = self.adapter.convert_requirement(
                req,
                context="Section M - Evaluation Factors"
            )
            enhanced.rfp_section = RFPSection.SECTION_M
            matrix.add_requirement(enhanced)
        
        # Process technical/Section C requirements
        technical = ctm_data.get('technical_requirements', [])
        for req in technical:
            enhanced = self.adapter.convert_requirement(
                req,
                context="Section C - Statement of Work"
            )
            enhanced.rfp_section = RFPSection.SECTION_C
            matrix.add_requirement(enhanced)
        
        # Extract global formatting if present in stats
        stats = ctm_data.get('stats', {})
        if 'formatting' in stats:
            matrix.global_formatting = FormattingRequirement(**stats['formatting'])
        
        # Calculate total points
        matrix.total_max_points = sum(
            r.max_points for r in matrix.requirements 
            if r.max_points is not None
        )
        
        return matrix
    
    def add_v3_fields_to_dict(
        self,
        ctm_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add v3.0 fields directly to an existing CTM dictionary.
        
        This is useful when you want to enhance the output without
        changing the data structure.
        
        Args:
            ctm_data: Existing CTM dictionary
        
        Returns:
            Enhanced dictionary with v3.0 fields added to each requirement
        """
        result = ctm_data.copy()
        
        for section_key in ['section_l_requirements', 'section_m_requirements', 'technical_requirements']:
            if section_key in result:
                enhanced_reqs = []
                for req in result[section_key]:
                    enhanced = self.adapter.convert_requirement(req)
                    # Merge new fields into existing dict
                    req_with_v3 = {**req, **enhanced.to_dict()}
                    enhanced_reqs.append(req_with_v3)
                result[section_key] = enhanced_reqs
        
        # Add v3.0 stats
        if 'stats' not in result:
            result['stats'] = {}
        
        all_reqs = (
            result.get('section_l_requirements', []) +
            result.get('section_m_requirements', []) +
            result.get('technical_requirements', [])
        )
        
        result['stats']['v3_summary'] = {
            'pass_fail_count': sum(1 for r in all_reqs if r.get('scoring_type') == 'pass_fail'),
            'weighted_count': sum(1 for r in all_reqs if r.get('scoring_type') == 'weighted'),
            'qualitative_count': sum(1 for r in all_reqs if r.get('scoring_type') == 'qualitative'),
            'total_max_points': sum(r.get('max_points', 0) or 0 for r in all_reqs),
            'future_diligence_count': sum(1 for r in all_reqs if r.get('future_diligence_required')),
            'evidence_required_count': sum(1 for r in all_reqs if r.get('evidence_location_required')),
            'high_risk_count': sum(1 for r in all_reqs if r.get('disqualification_risk') == 'HIGH'),
        }
        
        return result


# =============================================================================
# API HELPERS
# =============================================================================

def format_ctm_for_api(matrix: ComplianceMatrix) -> Dict[str, Any]:
    """
    Format a ComplianceMatrix for API response.
    
    Returns a clean JSON structure optimized for the frontend.
    """
    # Group requirements by section
    by_section = {}
    for section in RFPSection:
        reqs = matrix.get_by_section(section)
        if reqs:
            by_section[section.value] = [r.to_dict() for r in reqs]
    
    # Create summary for quick display
    summary = {
        'total_requirements': len(matrix.requirements),
        'pass_fail_requirements': len(matrix.get_pass_fail_requirements()),
        'scored_requirements': len(matrix.get_scored_requirements()),
        'total_possible_points': matrix.total_max_points,
        'high_value_requirements': len(matrix.get_high_value_requirements(50)),
        'future_diligence_items': len(matrix.get_future_diligence_items()),
        'evidence_required_items': len(matrix.get_evidence_required_items()),
    }
    
    # Create risk assessment
    risk_assessment = {
        'high_risk_items': [
            {
                'id': r.id,
                'text': r.requirement_text[:100] + '...' if len(r.requirement_text) > 100 else r.requirement_text,
                'section': r.section_reference,
                'risk': r.disqualification_risk
            }
            for r in matrix.requirements
            if r.disqualification_risk == 'HIGH'
        ],
        'medium_risk_items': [
            {
                'id': r.id,
                'text': r.requirement_text[:100] + '...' if len(r.requirement_text) > 100 else r.requirement_text,
                'section': r.section_reference,
                'risk': r.disqualification_risk
            }
            for r in matrix.requirements
            if r.disqualification_risk == 'MEDIUM'
        ],
    }
    
    # Create points breakdown
    points_breakdown = {}
    for r in matrix.requirements:
        if r.max_points and r.evaluation_factor_name:
            factor = r.evaluation_factor_name
            if factor not in points_breakdown:
                points_breakdown[factor] = 0
            points_breakdown[factor] += r.max_points
    
    return {
        'rfp_id': matrix.rfp_id,
        'rfp_name': matrix.rfp_name,
        'rfp_format': matrix.rfp_format,
        'summary': summary,
        'risk_assessment': risk_assessment,
        'points_breakdown': points_breakdown,
        'requirements_by_section': by_section,
        'stats': matrix.stats,
        'created_at': matrix.created_at.isoformat(),
        'updated_at': matrix.updated_at.isoformat(),
    }


def format_requirement_for_outline(req: EnhancedRequirement) -> Dict[str, Any]:
    """
    Format a requirement for use by the Smart Outline Generator.
    
    Returns only the fields needed for outline generation.
    """
    return {
        'id': req.id,
        'text': req.requirement_text,
        'section_ref': req.section_reference,
        'scoring_type': req.scoring_type.value,
        'max_points': req.max_points,
        'content_depth_multiplier': req.content_depth_multiplier,
        'response_format': req.response_format.value,
        'page_limit': req.page_limit.to_dict() if req.page_limit else None,
        'evaluation_factor': req.evaluation_factor_name,
        'is_mandatory': req.is_mandatory,
        'is_scored': req.is_scored,
    }


def get_content_allocation_guidance(matrix: ComplianceMatrix) -> Dict[str, Any]:
    """
    Generate content allocation guidance for the drafting agent.
    
    Based on the NLM analysis: higher point sections should receive
    proportionally more narrative depth.
    """
    # Get all scored requirements
    scored = matrix.get_scored_requirements()
    if not scored:
        return {'guidance': 'No scored requirements found', 'allocations': []}
    
    # Calculate total points
    total_points = sum(r.max_points or 0 for r in scored)
    if total_points == 0:
        return {'guidance': 'No point values found', 'allocations': []}
    
    # Generate allocation percentages
    allocations = []
    for req in sorted(scored, key=lambda r: r.max_points or 0, reverse=True):
        if req.max_points:
            percentage = (req.max_points / total_points) * 100
            allocations.append({
                'id': req.id,
                'section_ref': req.section_reference,
                'max_points': req.max_points,
                'percentage_of_total': round(percentage, 1),
                'content_depth_multiplier': round(req.content_depth_multiplier, 2),
                'recommended_emphasis': 'HIGH' if percentage > 15 else 'MEDIUM' if percentage > 5 else 'STANDARD',
            })
    
    return {
        'total_points': total_points,
        'guidance': f'Allocate narrative depth proportionally across {len(scored)} scored sections',
        'allocations': allocations,
    }
