"""
PropelAI v6.0 CMMC Compliance Layer
Cybersecurity Maturity Model Certification Level 2 compliance for DoD proposals.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CMMCLevel(str, Enum):
    """CMMC certification levels"""
    LEVEL_1 = "Level 1"  # Foundational (17 practices)
    LEVEL_2 = "Level 2"  # Advanced (110 practices from NIST 800-171)
    LEVEL_3 = "Level 3"  # Expert (110+ additional practices)


class CMMCDomain(str, Enum):
    """CMMC 2.0 Domains"""
    AC = "Access Control"
    AT = "Awareness and Training"
    AU = "Audit and Accountability"
    CA = "Assessment, Authorization, and Monitoring"
    CM = "Configuration Management"
    IA = "Identification and Authentication"
    IR = "Incident Response"
    MA = "Maintenance"
    MP = "Media Protection"
    PE = "Physical Protection"
    PS = "Personnel Security"
    RA = "Risk Assessment"
    SC = "System and Communications Protection"
    SI = "System and Information Integrity"


class ComplianceStatus(str, Enum):
    """Compliance status for each practice"""
    COMPLIANT = "Compliant"
    PARTIALLY_COMPLIANT = "Partially Compliant"
    NOT_COMPLIANT = "Not Compliant"
    NOT_APPLICABLE = "Not Applicable"
    PENDING_REVIEW = "Pending Review"


@dataclass
class CMMCPractice:
    """Individual CMMC practice/requirement"""
    practice_id: str  # e.g., AC.L2-3.1.1
    domain: CMMCDomain
    level: CMMCLevel
    title: str
    description: str
    assessment_objectives: List[str] = field(default_factory=list)
    nist_mapping: Optional[str] = None  # e.g., "3.1.1"
    status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    evidence: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CMMCAssessment:
    """CMMC compliance assessment for a proposal"""
    proposal_id: str
    level: CMMCLevel
    practices: Dict[str, CMMCPractice] = field(default_factory=dict)
    overall_status: ComplianceStatus = ComplianceStatus.PENDING_REVIEW
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# CMMC Level 2 Practices (subset - key practices for proposals)
CMMC_LEVEL_2_PRACTICES = [
    # Access Control (AC)
    CMMCPractice(
        practice_id="AC.L2-3.1.1",
        domain=CMMCDomain.AC,
        level=CMMCLevel.LEVEL_2,
        title="Authorized Access Control",
        description="Limit system access to authorized users, processes acting on behalf of authorized users, and devices.",
        nist_mapping="3.1.1",
        assessment_objectives=[
            "Authorized users are identified",
            "Processes acting on behalf of authorized users are identified",
            "Devices authorized to connect are identified",
            "System access is limited to authorized users, processes, and devices",
        ],
    ),
    CMMCPractice(
        practice_id="AC.L2-3.1.2",
        domain=CMMCDomain.AC,
        level=CMMCLevel.LEVEL_2,
        title="Transaction & Function Control",
        description="Limit system access to the types of transactions and functions that authorized users are permitted to execute.",
        nist_mapping="3.1.2",
        assessment_objectives=[
            "Types of transactions and functions authorized users can execute are defined",
            "System access is limited to defined transactions and functions",
        ],
    ),
    CMMCPractice(
        practice_id="AC.L2-3.1.3",
        domain=CMMCDomain.AC,
        level=CMMCLevel.LEVEL_2,
        title="CUI Flow Control",
        description="Control the flow of CUI in accordance with approved authorizations.",
        nist_mapping="3.1.3",
        assessment_objectives=[
            "Information flow control policies are defined",
            "Methods and enforcement mechanisms are identified",
            "CUI flow is controlled per approved authorizations",
        ],
    ),
    # Awareness and Training (AT)
    CMMCPractice(
        practice_id="AT.L2-3.2.1",
        domain=CMMCDomain.AT,
        level=CMMCLevel.LEVEL_2,
        title="Role-Based Risk Awareness",
        description="Ensure that managers, systems administrators, and users are made aware of security risks.",
        nist_mapping="3.2.1",
        assessment_objectives=[
            "Security risks associated with activities are identified",
            "Personnel are made aware of risks",
            "Training covers policies, procedures, and rules of behavior",
        ],
    ),
    CMMCPractice(
        practice_id="AT.L2-3.2.2",
        domain=CMMCDomain.AT,
        level=CMMCLevel.LEVEL_2,
        title="Role-Based Training",
        description="Ensure that personnel are trained to carry out their assigned security-related duties.",
        nist_mapping="3.2.2",
        assessment_objectives=[
            "Security-related duties are identified",
            "Training is provided before access is granted",
            "Training is provided when required by changes",
            "Training is updated and provided periodically",
        ],
    ),
    # Audit and Accountability (AU)
    CMMCPractice(
        practice_id="AU.L2-3.3.1",
        domain=CMMCDomain.AU,
        level=CMMCLevel.LEVEL_2,
        title="System Auditing",
        description="Create and retain system audit logs and records.",
        nist_mapping="3.3.1",
        assessment_objectives=[
            "Audit logs are created",
            "Audit logs contain required content",
            "Audit logs are retained as specified",
        ],
    ),
    CMMCPractice(
        practice_id="AU.L2-3.3.2",
        domain=CMMCDomain.AU,
        level=CMMCLevel.LEVEL_2,
        title="User Accountability",
        description="Ensure that actions of individual system users can be uniquely traced.",
        nist_mapping="3.3.2",
        assessment_objectives=[
            "User actions can be uniquely traced",
            "Users can be held accountable for their actions",
        ],
    ),
    # Configuration Management (CM)
    CMMCPractice(
        practice_id="CM.L2-3.4.1",
        domain=CMMCDomain.CM,
        level=CMMCLevel.LEVEL_2,
        title="System Baselining",
        description="Establish and maintain baseline configurations and inventories.",
        nist_mapping="3.4.1",
        assessment_objectives=[
            "Baseline configurations are established",
            "Baseline configurations are maintained",
            "System inventories are established",
            "System inventories are maintained",
        ],
    ),
    CMMCPractice(
        practice_id="CM.L2-3.4.2",
        domain=CMMCDomain.CM,
        level=CMMCLevel.LEVEL_2,
        title="Security Configuration Enforcement",
        description="Establish and enforce security configuration settings.",
        nist_mapping="3.4.2",
        assessment_objectives=[
            "Security configuration settings are established",
            "Security configuration settings are enforced",
        ],
    ),
    # Identification and Authentication (IA)
    CMMCPractice(
        practice_id="IA.L2-3.5.1",
        domain=CMMCDomain.IA,
        level=CMMCLevel.LEVEL_2,
        title="Identification",
        description="Identify system users, processes, and devices.",
        nist_mapping="3.5.1",
        assessment_objectives=[
            "Users are identified",
            "Processes acting on behalf of users are identified",
            "Devices are identified",
        ],
    ),
    CMMCPractice(
        practice_id="IA.L2-3.5.2",
        domain=CMMCDomain.IA,
        level=CMMCLevel.LEVEL_2,
        title="Authentication",
        description="Authenticate identities of users, processes, or devices.",
        nist_mapping="3.5.2",
        assessment_objectives=[
            "User identities are authenticated",
            "Process identities are authenticated",
            "Device identities are authenticated",
        ],
    ),
    # Incident Response (IR)
    CMMCPractice(
        practice_id="IR.L2-3.6.1",
        domain=CMMCDomain.IR,
        level=CMMCLevel.LEVEL_2,
        title="Incident Handling",
        description="Establish an operational incident-handling capability.",
        nist_mapping="3.6.1",
        assessment_objectives=[
            "Incident handling capability is established",
            "Preparation activities are performed",
            "Detection and analysis activities are performed",
            "Containment activities are performed",
            "Recovery activities are performed",
        ],
    ),
    # Media Protection (MP)
    CMMCPractice(
        practice_id="MP.L2-3.8.1",
        domain=CMMCDomain.MP,
        level=CMMCLevel.LEVEL_2,
        title="Media Protection",
        description="Protect paper and digital media containing CUI.",
        nist_mapping="3.8.1",
        assessment_objectives=[
            "Paper media containing CUI is protected",
            "Digital media containing CUI is protected",
        ],
    ),
    # Personnel Security (PS)
    CMMCPractice(
        practice_id="PS.L2-3.9.1",
        domain=CMMCDomain.PS,
        level=CMMCLevel.LEVEL_2,
        title="Screen Individuals",
        description="Screen individuals prior to authorizing access.",
        nist_mapping="3.9.1",
        assessment_objectives=[
            "Screening criteria are established",
            "Individuals are screened before access is granted",
        ],
    ),
    # System and Communications Protection (SC)
    CMMCPractice(
        practice_id="SC.L2-3.13.1",
        domain=CMMCDomain.SC,
        level=CMMCLevel.LEVEL_2,
        title="Boundary Protection",
        description="Monitor, control, and protect communications at system boundaries.",
        nist_mapping="3.13.1",
        assessment_objectives=[
            "Communications at system boundaries are monitored",
            "Communications at system boundaries are controlled",
            "Communications at system boundaries are protected",
        ],
    ),
    CMMCPractice(
        practice_id="SC.L2-3.13.8",
        domain=CMMCDomain.SC,
        level=CMMCLevel.LEVEL_2,
        title="Data at Rest",
        description="Implement cryptographic mechanisms to prevent unauthorized disclosure of CUI at rest.",
        nist_mapping="3.13.8",
        assessment_objectives=[
            "CUI at rest is identified",
            "Cryptographic mechanisms are implemented",
            "Unauthorized disclosure is prevented",
        ],
    ),
    # System and Information Integrity (SI)
    CMMCPractice(
        practice_id="SI.L2-3.14.1",
        domain=CMMCDomain.SI,
        level=CMMCLevel.LEVEL_2,
        title="Flaw Remediation",
        description="Identify, report, and correct system flaws in a timely manner.",
        nist_mapping="3.14.1",
        assessment_objectives=[
            "Flaws are identified",
            "Flaws are reported",
            "Flaws are corrected in a timely manner",
        ],
    ),
    CMMCPractice(
        practice_id="SI.L2-3.14.2",
        domain=CMMCDomain.SI,
        level=CMMCLevel.LEVEL_2,
        title="Malicious Code Protection",
        description="Provide protection from malicious code at designated locations.",
        nist_mapping="3.14.2",
        assessment_objectives=[
            "Designated locations for malicious code protection are identified",
            "Malicious code protection is provided at designated locations",
        ],
    ),
]


class CMMCComplianceChecker:
    """
    Checks proposal content for CMMC compliance indicators.

    Uses pattern matching and keyword analysis to identify:
    - CMMC requirements in RFPs
    - Company's compliance claims
    - Gaps between requirements and claims
    """

    # Keywords indicating CMMC/CUI requirements
    CMMC_KEYWORDS = [
        r"cmmc",
        r"cybersecurity maturity model",
        r"controlled unclassified information",
        r"cui",
        r"nist\s*800-171",
        r"nist\s*sp\s*800-171",
        r"dfars\s*252\.204-7012",
        r"dfars\s*252\.204-7019",
        r"dfars\s*252\.204-7020",
        r"dfars\s*252\.204-7021",
        r"federal contract information",
        r"fci",
        r"system security plan",
        r"ssp",
        r"plan of action",
        r"poa&m",
        r"poam",
    ]

    # Domain-specific keywords
    DOMAIN_KEYWORDS = {
        CMMCDomain.AC: ["access control", "authorization", "least privilege", "remote access"],
        CMMCDomain.AT: ["security training", "awareness", "phishing", "social engineering"],
        CMMCDomain.AU: ["audit", "logging", "accountability", "monitoring"],
        CMMCDomain.CM: ["configuration management", "baseline", "change control"],
        CMMCDomain.IA: ["authentication", "multi-factor", "mfa", "password", "identity"],
        CMMCDomain.IR: ["incident response", "breach", "security incident"],
        CMMCDomain.MP: ["media protection", "sanitization", "encryption at rest"],
        CMMCDomain.PS: ["personnel security", "background check", "screening"],
        CMMCDomain.SC: ["encryption", "cryptographic", "boundary protection", "firewall"],
        CMMCDomain.SI: ["malware", "antivirus", "patch management", "vulnerability"],
    }

    def __init__(self):
        self.practices = {p.practice_id: p for p in CMMC_LEVEL_2_PRACTICES}

    def detect_cmmc_requirements(self, text: str) -> Dict[str, Any]:
        """
        Detect CMMC requirements in RFP text.

        Returns:
            Dict with detected level, domains, and specific requirements
        """
        text_lower = text.lower()

        # Check for CMMC keywords
        cmmc_mentioned = any(
            re.search(kw, text_lower) for kw in self.CMMC_KEYWORDS
        )

        if not cmmc_mentioned:
            return {
                "cmmc_required": False,
                "level": None,
                "domains": [],
                "requirements": [],
            }

        # Detect level
        level = CMMCLevel.LEVEL_2  # Default for most DoD contracts
        if re.search(r"cmmc\s*(level\s*)?3", text_lower):
            level = CMMCLevel.LEVEL_3
        elif re.search(r"cmmc\s*(level\s*)?1", text_lower):
            level = CMMCLevel.LEVEL_1

        # Detect mentioned domains
        mentioned_domains = set()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    mentioned_domains.add(domain)
                    break

        # Extract specific requirements
        requirements = []
        for practice in self.practices.values():
            # Check if practice domain is mentioned
            if practice.domain in mentioned_domains:
                requirements.append({
                    "practice_id": practice.practice_id,
                    "title": practice.title,
                    "domain": practice.domain.value,
                })

        return {
            "cmmc_required": True,
            "level": level.value,
            "domains": [d.value for d in mentioned_domains],
            "requirements": requirements,
        }

    def assess_compliance(
        self,
        proposal_id: str,
        proposal_text: str,
        company_capabilities: Optional[Dict[str, Any]] = None,
    ) -> CMMCAssessment:
        """
        Assess proposal compliance with CMMC requirements.

        Args:
            proposal_id: Proposal identifier
            proposal_text: Full proposal text
            company_capabilities: Optional company capability data

        Returns:
            CMMCAssessment with status, gaps, and recommendations
        """
        assessment = CMMCAssessment(
            proposal_id=proposal_id,
            level=CMMCLevel.LEVEL_2,
        )

        text_lower = proposal_text.lower()

        # Check each practice
        compliant_count = 0
        partial_count = 0
        gaps = []
        recommendations = []

        for practice_id, practice in self.practices.items():
            # Create copy of practice for assessment
            assessed = CMMCPractice(
                practice_id=practice.practice_id,
                domain=practice.domain,
                level=practice.level,
                title=practice.title,
                description=practice.description,
                assessment_objectives=practice.assessment_objectives,
                nist_mapping=practice.nist_mapping,
            )

            # Check for keywords related to this practice's domain
            domain_keywords = self.DOMAIN_KEYWORDS.get(practice.domain, [])
            keywords_found = sum(1 for kw in domain_keywords if kw in text_lower)

            # Check for explicit claims
            explicit_claim = any([
                practice_id.lower() in text_lower,
                practice.nist_mapping and practice.nist_mapping in text_lower,
            ])

            # Determine status
            if explicit_claim:
                assessed.status = ComplianceStatus.COMPLIANT
                compliant_count += 1
            elif keywords_found >= 2:
                assessed.status = ComplianceStatus.PARTIALLY_COMPLIANT
                partial_count += 1
            elif keywords_found >= 1:
                assessed.status = ComplianceStatus.PENDING_REVIEW
            else:
                assessed.status = ComplianceStatus.NOT_COMPLIANT
                gaps.append(f"{practice_id}: {practice.title}")
                recommendations.append(
                    f"Address {practice.domain.value} requirement: {practice.description}"
                )

            assessment.practices[practice_id] = assessed

        # Determine overall status
        total = len(self.practices)
        if compliant_count == total:
            assessment.overall_status = ComplianceStatus.COMPLIANT
        elif compliant_count + partial_count >= total * 0.8:
            assessment.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            assessment.overall_status = ComplianceStatus.NOT_COMPLIANT

        assessment.gaps = gaps
        assessment.recommendations = recommendations[:10]  # Limit recommendations

        return assessment

    def generate_compliance_matrix(
        self,
        assessment: CMMCAssessment,
    ) -> List[Dict[str, Any]]:
        """Generate a compliance matrix from assessment"""
        matrix = []

        for practice_id, practice in assessment.practices.items():
            matrix.append({
                "practice_id": practice_id,
                "domain": practice.domain.value,
                "title": practice.title,
                "nist_ref": practice.nist_mapping,
                "status": practice.status.value,
                "evidence": practice.evidence,
                "notes": practice.notes,
            })

        # Sort by domain and practice ID
        matrix.sort(key=lambda x: (x["domain"], x["practice_id"]))

        return matrix

    def get_gap_report(self, assessment: CMMCAssessment) -> str:
        """Generate a gap analysis report"""
        lines = [
            f"# CMMC Level 2 Gap Analysis",
            f"Proposal: {assessment.proposal_id}",
            f"Overall Status: {assessment.overall_status.value}",
            "",
            "## Summary",
            f"- Total Practices: {len(assessment.practices)}",
            f"- Compliant: {sum(1 for p in assessment.practices.values() if p.status == ComplianceStatus.COMPLIANT)}",
            f"- Partially Compliant: {sum(1 for p in assessment.practices.values() if p.status == ComplianceStatus.PARTIALLY_COMPLIANT)}",
            f"- Not Compliant: {sum(1 for p in assessment.practices.values() if p.status == ComplianceStatus.NOT_COMPLIANT)}",
            "",
            "## Gaps",
        ]

        for gap in assessment.gaps:
            lines.append(f"- {gap}")

        lines.extend([
            "",
            "## Recommendations",
        ])

        for rec in assessment.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_cmmc_checker() -> CMMCComplianceChecker:
    """Create a new CMMC compliance checker"""
    return CMMCComplianceChecker()
