"""
Strategic Compliance Mapper
Maps requirements to proposal sections, win themes, and evidence

Based on professional compliance matrix best practices from
VisibleThread and FedProposalExperts
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from .models import RequirementNode, RequirementType


class ProposalVolume(Enum):
    """Standard proposal volume structure"""
    VOLUME_1_TECHNICAL = "Volume I - Technical Approach"
    VOLUME_2_MANAGEMENT = "Volume II - Management Approach"
    VOLUME_3_PAST_PERFORMANCE = "Volume III - Past Performance"
    VOLUME_4_COST_PRICE = "Volume IV - Cost/Price Proposal"
    VOLUME_5_ADMIN = "Volume V - Administrative/Forms"


class WinTheme(Enum):
    """Common win themes for government proposals"""
    TECHNICAL_EXCELLENCE = "Technical Excellence & Innovation"
    PROVEN_PERFORMANCE = "Proven Past Performance"
    SUPERIOR_MANAGEMENT = "Superior Management & Quality"
    COST_EFFECTIVENESS = "Cost Efficiency & Value"
    SECURITY_COMPLIANCE = "Robust Security & Compliance"
    QUALIFIED_PERSONNEL = "Highly Qualified Team"
    AGILE_DELIVERY = "Agile & Responsive Delivery"
    RISK_MITIGATION = "Proactive Risk Management"


class StrategicMapper:
    """
    Maps requirements to strategic proposal elements
    
    Provides:
    - Proposal volume/section mapping
    - Win theme suggestions
    - Evidence requirements
    - Response strategies
    """
    
    # Map requirement types to proposal volumes
    REQUIREMENT_TO_VOLUME = {
        RequirementType.PERFORMANCE: ProposalVolume.VOLUME_1_TECHNICAL,
        RequirementType.DELIVERABLE: ProposalVolume.VOLUME_1_TECHNICAL,
        RequirementType.PERFORMANCE_METRIC: ProposalVolume.VOLUME_1_TECHNICAL,
        RequirementType.LABOR_REQUIREMENT: ProposalVolume.VOLUME_2_MANAGEMENT,
        RequirementType.QUALIFICATION: ProposalVolume.VOLUME_3_PAST_PERFORMANCE,
        RequirementType.PROPOSAL_INSTRUCTION: ProposalVolume.VOLUME_5_ADMIN,
        RequirementType.FORMAT: ProposalVolume.VOLUME_5_ADMIN,
        RequirementType.COMPLIANCE: ProposalVolume.VOLUME_5_ADMIN,
        RequirementType.EVALUATION_CRITERION: ProposalVolume.VOLUME_1_TECHNICAL,
        RequirementType.PROHIBITION: ProposalVolume.VOLUME_5_ADMIN,
    }
    
    # Map requirement types to win themes
    REQUIREMENT_TO_WIN_THEME = {
        RequirementType.PERFORMANCE: WinTheme.TECHNICAL_EXCELLENCE,
        RequirementType.TECHNICAL: WinTheme.TECHNICAL_EXCELLENCE,
        RequirementType.SECURITY: WinTheme.SECURITY_COMPLIANCE,
        RequirementType.DELIVERABLE: WinTheme.AGILE_DELIVERY,
        RequirementType.QUALITY: WinTheme.SUPERIOR_MANAGEMENT,
        RequirementType.LABOR_REQUIREMENT: WinTheme.QUALIFIED_PERSONNEL,
        RequirementType.MANAGEMENT: WinTheme.SUPERIOR_MANAGEMENT,
        RequirementType.STAFFING: WinTheme.QUALIFIED_PERSONNEL,
        RequirementType.PAST_PERFORMANCE: WinTheme.PROVEN_PERFORMANCE,
        RequirementType.QUALIFICATION: WinTheme.QUALIFIED_PERSONNEL,
        RequirementType.PRICING: WinTheme.COST_EFFECTIVENESS,
        RequirementType.PAYMENT: WinTheme.COST_EFFECTIVENESS,
        RequirementType.COMPLIANCE: WinTheme.SECURITY_COMPLIANCE,
        RequirementType.EVALUATION_CRITERION: WinTheme.TECHNICAL_EXCELLENCE,
    }
    
    def __init__(self):
        """Initialize strategic mapper"""
        pass
    
    def map_requirement(self, requirement: RequirementNode) -> Dict[str, str]:
        """
        Map a requirement to all strategic elements
        
        Returns:
            Dict with:
            - proposal_volume
            - proposal_section
            - win_theme
            - response_strategy
            - evidence_required
            - proof_points
        """
        # Get base mappings
        volume = self._map_to_volume(requirement)
        win_theme = self._map_to_win_theme(requirement)
        
        # Generate specific section within volume
        section = self._generate_section_mapping(requirement, volume)
        
        # Generate response strategy
        response_strategy = self._generate_response_strategy(requirement)
        
        # Determine evidence requirements
        evidence = self._determine_evidence(requirement)
        
        # Generate proof points
        proof_points = self._generate_proof_points(requirement)
        
        return {
            'proposal_volume': volume,
            'proposal_section': section,
            'win_theme': win_theme,
            'response_strategy': response_strategy,
            'evidence_required': evidence,
            'proof_points': proof_points,
        }
    
    def _map_to_volume(self, req: RequirementNode) -> str:
        """Map requirement to proposal volume"""
        req_type = req.requirement_type
        
        # Check for specific keywords that override default mapping
        text_lower = req.text.lower()
        
        # Past performance indicators
        if any(keyword in text_lower for keyword in [
            'past performance', 'previous contract', 'similar work',
            'experience', 'references', 'prior project'
        ]):
            return ProposalVolume.VOLUME_3_PAST_PERFORMANCE.value
        
        # Management indicators
        if any(keyword in text_lower for keyword in [
            'manage', 'management plan', 'quality assurance',
            'project manager', 'team lead', 'organizational'
        ]):
            return ProposalVolume.VOLUME_2_MANAGEMENT.value
        
        # Cost/pricing indicators
        if any(keyword in text_lower for keyword in [
            'price', 'cost', 'pricing', 'payment', 'invoice',
            'rate', 'fee'
        ]):
            return ProposalVolume.VOLUME_4_COST_PRICE.value
        
        # Use requirement type mapping as default
        volume = self.REQUIREMENT_TO_VOLUME.get(
            req_type,
            ProposalVolume.VOLUME_1_TECHNICAL
        )
        
        return volume.value
    
    def _generate_section_mapping(self, req: RequirementNode, volume: str) -> str:
        """Generate specific section within volume"""
        text_lower = req.text.lower()
        
        if "Volume I" in volume:
            # Technical volume sections
            if 'security' in text_lower or 'cybersecurity' in text_lower:
                return "Section 3: Security Approach"
            elif 'architecture' in text_lower or 'design' in text_lower:
                return "Section 2: Technical Architecture"
            elif 'migration' in text_lower or 'transition' in text_lower:
                return "Section 4: Transition/Migration Plan"
            elif 'testing' in text_lower or 'quality' in text_lower:
                return "Section 5: Quality Assurance & Testing"
            else:
                return "Section 1: Technical Solution Overview"
        
        elif "Volume II" in volume:
            # Management volume sections
            if 'staff' in text_lower or 'personnel' in text_lower or 'team' in text_lower:
                return "Section 2: Staffing Plan"
            elif 'quality' in text_lower or 'qa' in text_lower:
                return "Section 3: Quality Management"
            elif 'risk' in text_lower:
                return "Section 4: Risk Management"
            else:
                return "Section 1: Management Approach"
        
        elif "Volume III" in volume:
            # Past performance sections
            if 'similar' in text_lower or 'relevant' in text_lower:
                return "Section 1: Relevant Past Performance"
            else:
                return "Section 2: Contract References"
        
        elif "Volume IV" in volume:
            return "Cost/Price Proposal"
        
        else:
            return "Administrative Section"
    
    def _map_to_win_theme(self, req: RequirementNode) -> str:
        """Map requirement to win theme"""
        # Use requirement type mapping
        theme = self.REQUIREMENT_TO_WIN_THEME.get(
            req.requirement_type,
            WinTheme.TECHNICAL_EXCELLENCE
        )
        
        return theme.value
    
    def _generate_response_strategy(self, req: RequirementNode) -> str:
        """Generate response strategy based on requirement"""
        text_lower = req.text.lower()
        req_type = req.requirement_type
        
        # Check for mandatory keywords
        is_mandatory = any(word in text_lower for word in ['shall', 'must', 'required'])
        
        # Generate strategy based on requirement type
        if req_type == RequirementType.TECHNICAL or req_type == RequirementType.PERFORMANCE:
            if 'security' in text_lower:
                return "Demonstrate security certifications and compliance framework. Provide security architecture diagrams."
            elif 'migration' in text_lower or 'transition' in text_lower:
                return "Present detailed migration plan with phases, timeline, and risk mitigation. Include similar past migrations."
            elif 'agile' in text_lower or 'devops' in text_lower:
                return "Describe agile/DevOps methodology with tools, processes, and team structure. Show metrics from past projects."
            else:
                return "Explain technical approach with architecture, implementation plan, and risk mitigation. Provide relevant examples."
        
        elif req_type == RequirementType.PAST_PERFORMANCE:
            return "Provide 3-5 relevant contract references with similar scope, size, and complexity. Include metrics and client contact info."
        
        elif req_type == RequirementType.STAFFING or req_type == RequirementType.LABOR_REQUIREMENT:
            return "Present team structure, key personnel resumes, and labor mix. Highlight relevant certifications and experience."
        
        elif req_type == RequirementType.MANAGEMENT:
            return "Outline management approach, quality processes, and governance structure. Show proven methodologies."
        
        elif req_type == RequirementType.QUALIFICATION:
            return "Provide certifications, qualifications, and documentation. Include copies of certificates and credentials."
        
        elif req_type == RequirementType.DELIVERABLE:
            return "List all deliverables with format, frequency, and acceptance criteria. Provide sample templates if applicable."
        
        elif req_type == RequirementType.SECURITY:
            return "Demonstrate security posture with certifications (FedRAMP, FISMA, etc.), controls, and monitoring approach."
        
        elif req_type == RequirementType.COMPLIANCE:
            return "Confirm compliance and provide supporting documentation. Reference specific FAR/DFARS clauses as applicable."
        
        elif req_type == RequirementType.PRICING:
            return "Provide detailed pricing breakdown by CLIN/labor category. Justify rates with market data if requested."
        
        else:
            if is_mandatory:
                return "Provide clear statement of compliance with supporting evidence and documentation."
            else:
                return "Address requirement with relevant approach and supporting examples."
    
    def _determine_evidence(self, req: RequirementNode) -> str:
        """Determine what evidence is required"""
        text_lower = req.text.lower()
        req_type = req.requirement_type
        
        evidence_list = []
        
        # Certification/qualification requirements
        if any(word in text_lower for word in ['certified', 'certification', 'accredited', 'iso', 'cmmi']):
            evidence_list.append("Certification documents (ISO, CMMI, etc.)")
        
        if 'clearance' in text_lower or 'security clearance' in text_lower:
            evidence_list.append("Security clearance documentation")
        
        # Past performance
        if req_type == RequirementType.PAST_PERFORMANCE:
            evidence_list.append("3-5 relevant contract references")
            evidence_list.append("Client contact information")
            evidence_list.append("Performance metrics and outcomes")
        
        # Staffing/personnel
        if req_type == RequirementType.STAFFING or 'resume' in text_lower or 'key personnel' in text_lower:
            evidence_list.append("Resumes of key personnel")
            evidence_list.append("Professional certifications")
            evidence_list.append("Org chart")
        
        # Technical capabilities
        if req_type == RequirementType.TECHNICAL:
            evidence_list.append("Architecture diagrams")
            evidence_list.append("Technical approach documentation")
            evidence_list.append("Sample deliverables")
        
        # Security requirements
        if 'security' in text_lower or req_type == RequirementType.SECURITY:
            evidence_list.append("Security plan")
            evidence_list.append("FedRAMP/FISMA compliance docs")
            evidence_list.append("Security controls matrix")
        
        # Management
        if req_type == RequirementType.MANAGEMENT:
            evidence_list.append("Management plan")
            evidence_list.append("Quality assurance plan")
            evidence_list.append("Risk management plan")
        
        # Financial/pricing
        if req_type == RequirementType.PRICING:
            evidence_list.append("Detailed cost breakdown")
            evidence_list.append("Rate justification")
        
        # Compliance
        if req_type == RequirementType.COMPLIANCE:
            evidence_list.append("Compliance certification")
            evidence_list.append("Supporting documentation")
        
        # Default
        if not evidence_list:
            evidence_list.append("Written response with supporting documentation")
        
        return "; ".join(evidence_list)
    
    def _generate_proof_points(self, req: RequirementNode) -> str:
        """Generate specific proof points/discriminators"""
        text_lower = req.text.lower()
        req_type = req.requirement_type
        
        proof_points = []
        
        # Performance metrics
        if any(word in text_lower for word in ['metric', 'measure', 'kpi', 'performance']):
            proof_points.append("Quantifiable performance metrics from similar projects")
        
        # Innovation
        if any(word in text_lower for word in ['innovative', 'modern', 'latest', 'cutting-edge']):
            proof_points.append("Industry-leading tools and methodologies")
        
        # Experience
        if any(word in text_lower for word in ['experience', 'years', 'similar']):
            proof_points.append("X years of relevant experience in similar environments")
        
        # Quality
        if 'quality' in text_lower or req_type == RequirementType.QUALITY:
            proof_points.append("Quality certifications (ISO 9001, CMMI Level 3+)")
        
        # Security
        if 'security' in text_lower:
            proof_points.append("FedRAMP authorized or ATO certification")
            proof_points.append("Zero security incidents track record")
        
        # Agile/Modern
        if 'agile' in text_lower or 'devops' in text_lower:
            proof_points.append("Mature DevSecOps pipelines with automated testing")
        
        # Team qualifications
        if any(word in text_lower for word in ['staff', 'personnel', 'team']):
            proof_points.append("Average X years of experience per team member")
            proof_points.append("Relevant certifications (PMP, AWS, CISSP, etc.)")
        
        # Cost effectiveness
        if req_type == RequirementType.PRICING:
            proof_points.append("Competitive rates with demonstrated value")
        
        # Default based on requirement type
        if not proof_points:
            if req_type in [RequirementType.TECHNICAL, RequirementType.PERFORMANCE]:
                proof_points.append("Proven technical solution with successful implementations")
            elif req_type == RequirementType.PAST_PERFORMANCE:
                proof_points.append("Excellent past performance ratings")
            elif req_type == RequirementType.MANAGEMENT:
                proof_points.append("Mature management processes with ISO/CMMI certification")
        
        # Ensure we have at least one proof point
        if not proof_points:
            proof_points.append("Demonstrated capability and relevant experience")
        
        return "; ".join(proof_points)


# Singleton instance
_mapper = None


def get_strategic_mapper() -> StrategicMapper:
    """Get the singleton strategic mapper"""
    global _mapper
    if _mapper is None:
        _mapper = StrategicMapper()
    return _mapper
