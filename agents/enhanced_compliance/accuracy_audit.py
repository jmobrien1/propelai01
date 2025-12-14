"""
PropelAI Compliance Accuracy Audit Framework

This module provides tools to validate extraction accuracy against
manual requirement counts and identify gaps in the extraction process.

Usage:
    from agents.enhanced_compliance.accuracy_audit import AccuracyAuditor

    auditor = AccuracyAuditor()

    # After processing an RFP
    report = auditor.audit_extraction(
        extraction_result=result,
        manual_counts={
            'section_l': 45,
            'technical': 120,
            'evaluation': 30,
            'mandatory': 89
        }
    )

    print(report.accuracy_score)
    print(report.missed_requirements)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .section_aware_extractor import (
    ExtractionResult,
    StructuredRequirement,
    RequirementCategory,
    BindingLevel
)


class AuditSeverity(Enum):
    """Severity level for audit findings"""
    CRITICAL = "critical"      # Missed mandatory requirement
    HIGH = "high"              # Significant accuracy gap
    MEDIUM = "medium"          # Minor accuracy gap
    LOW = "low"                # Informational
    INFO = "info"              # Diagnostic info


@dataclass
class AuditFinding:
    """Individual finding from accuracy audit"""
    severity: AuditSeverity
    category: str
    title: str
    description: str
    recommendation: str
    evidence: Optional[str] = None
    location: Optional[str] = None


@dataclass
class ConflictFinding:
    """Detected conflict in RFP requirements"""
    conflict_type: str          # "page_limit", "date_conflict", "reference_conflict"
    severity: AuditSeverity
    description: str
    locations: List[Dict[str, Any]]
    draft_qa: str               # Draft Q&A for Contracting Officer
    recommendation: str


@dataclass
class ComplianceGate:
    """Pass/fail requirement that can cause disqualification"""
    requirement_id: str
    full_text: str
    gate_type: str             # "certification", "experience", "format", "content"
    consequence: str           # What happens if not met
    source_location: str
    page_number: int
    action_required: str


@dataclass
class AccuracyReport:
    """Complete accuracy audit report"""
    # Timestamp
    audit_date: str = field(default_factory=lambda: datetime.now().isoformat())
    rfp_id: str = ""

    # Accuracy metrics
    overall_accuracy: float = 0.0
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)

    # Counts comparison
    propelai_counts: Dict[str, int] = field(default_factory=dict)
    manual_counts: Dict[str, int] = field(default_factory=dict)
    variance: Dict[str, int] = field(default_factory=dict)

    # Findings
    findings: List[AuditFinding] = field(default_factory=list)
    conflicts: List[ConflictFinding] = field(default_factory=list)
    compliance_gates: List[ComplianceGate] = field(default_factory=list)

    # Quality metrics
    section_detection_rate: float = 0.0
    requirements_with_rfp_ids: int = 0
    requirements_with_page_numbers: int = 0
    duplicate_count: int = 0

    # Summary
    passed: bool = False
    summary: str = ""


class AccuracyAuditor:
    """
    Audits extraction accuracy and identifies quality issues.

    Key responsibilities:
    1. Compare extraction results to manual counts
    2. Detect conflicts in RFP requirements
    3. Identify compliance gates
    4. Generate actionable recommendations
    """

    # Patterns for compliance gate detection
    COMPLIANCE_GATE_PATTERNS = [
        (r'failure\s+to\s+(?:provide|submit|demonstrate|comply|meet).*(?:will|shall)\s+result\s+in\s+(?:immediate\s+)?(?:disqualification|rejection|elimination|removal)', 'disqualification'),
        (r'(?:will\s+)?not\s+be\s+(?:evaluated|considered|accepted|reviewed)\s+(?:for\s+award)?', 'not_evaluated'),
        (r'proposal\s+(?:will|shall)\s+be\s+(?:deemed\s+)?(?:non-?compliant|non-?responsive|unacceptable)', 'non_compliant'),
        (r'(?:mandatory|pass[\/\-]fail)\s+(?:requirement|criteria|gate)', 'pass_fail'),
        (r'must\s+(?:provide|submit|demonstrate|include).*(?:proof|evidence|certification)', 'certification'),
        (r'(?:lack|absence|failure)\s+of.*(?:will|shall).*(?:disqualif|reject|eliminat)', 'disqualification'),
    ]

    # Patterns for page limit conflict detection
    PAGE_LIMIT_PATTERNS = [
        r'(?:page|pg)\s*limit[:\s]+(\d+)',
        r'(?:not\s+(?:to\s+)?exceed|maximum\s+of|limited\s+to)\s*(\d+)\s*(?:pages?|pgs?)',
        r'(\d+)\s*(?:pages?|pgs?)\s*(?:maximum|max|limit)',
    ]

    # Patterns for page allocation detection
    PAGE_ALLOCATION_PATTERNS = [
        r'(?:factor|section|volume)\s*\d+[:\s]+(\d+)\s*(?:pages?|pgs?)',
        r'(?:SF|Sub\s*Factor)\s*\d+[:\s]*(\d+)\s*(?:pages?|pgs?)',
    ]

    def __init__(self):
        self.findings = []

    def audit_extraction(
        self,
        extraction_result: ExtractionResult,
        manual_counts: Optional[Dict[str, int]] = None,
        rfp_id: str = ""
    ) -> AccuracyReport:
        """
        Perform comprehensive accuracy audit.

        Args:
            extraction_result: Results from SectionAwareExtractor
            manual_counts: Optional manual requirement counts for comparison
            rfp_id: RFP identifier

        Returns:
            AccuracyReport with findings and recommendations
        """
        report = AccuracyReport(rfp_id=rfp_id)

        # Build PropelAI counts
        report.propelai_counts = {
            'total': len(extraction_result.all_requirements),
            'section_l': len(extraction_result.section_l_requirements),
            'technical': len(extraction_result.technical_requirements),
            'evaluation': len(extraction_result.evaluation_requirements),
            'attachment': len(extraction_result.attachment_requirements),
            'administrative': len(extraction_result.administrative_requirements),
            'mandatory': sum(1 for r in extraction_result.all_requirements
                           if r.binding_level == BindingLevel.MANDATORY),
            'highly_desirable': sum(1 for r in extraction_result.all_requirements
                                   if r.binding_level == BindingLevel.HIGHLY_DESIRABLE),
            'desirable': sum(1 for r in extraction_result.all_requirements
                           if r.binding_level == BindingLevel.DESIRABLE),
        }

        # Compare to manual counts if provided
        if manual_counts:
            report.manual_counts = manual_counts
            report.variance = {}
            report.accuracy_by_category = {}

            for key in manual_counts:
                propelai_count = report.propelai_counts.get(key, 0)
                manual_count = manual_counts[key]
                report.variance[key] = propelai_count - manual_count

                if manual_count > 0:
                    # Accuracy = 1 - (variance / manual_count), capped at 0-100%
                    accuracy = max(0, 1 - abs(report.variance[key]) / manual_count)
                    report.accuracy_by_category[key] = round(accuracy * 100, 1)

            # Calculate overall accuracy (weighted by mandatory)
            if 'mandatory' in manual_counts and manual_counts['mandatory'] > 0:
                mandatory_accuracy = report.accuracy_by_category.get('mandatory', 0)
                total_accuracy = report.accuracy_by_category.get('total', 0)
                report.overall_accuracy = round((mandatory_accuracy * 0.7 + total_accuracy * 0.3), 1)
            else:
                report.overall_accuracy = report.accuracy_by_category.get('total', 0)

        # Quality metrics
        report.section_detection_rate = self._calculate_section_detection_rate(extraction_result)
        report.requirements_with_rfp_ids = sum(
            1 for r in extraction_result.all_requirements
            if not r.rfp_reference.startswith('TW-')
        )
        report.requirements_with_page_numbers = sum(
            1 for r in extraction_result.all_requirements
            if r.page_number > 0
        )

        # Detect compliance gates
        report.compliance_gates = self._detect_compliance_gates(extraction_result)

        # Detect conflicts
        if extraction_result.structure:
            report.conflicts = self._detect_conflicts(extraction_result)

        # Generate findings
        report.findings = self._generate_findings(report, extraction_result)

        # Summary
        report.passed = report.overall_accuracy >= 95 if manual_counts else True
        report.summary = self._generate_summary(report)

        return report

    def _calculate_section_detection_rate(self, result: ExtractionResult) -> float:
        """Calculate percentage of requirements with known section context"""
        if not result.all_requirements:
            return 0.0

        # Count requirements that are NOT in "unknown" section
        known_section = sum(
            1 for r in result.all_requirements
            if r.source_section is not None and r.source_section.value != 'UNK'
        )

        return round((known_section / len(result.all_requirements)) * 100, 1)

    def _detect_compliance_gates(self, result: ExtractionResult) -> List[ComplianceGate]:
        """Detect pass/fail compliance gates in requirements"""
        gates = []

        for req in result.all_requirements:
            text_lower = req.full_text.lower()

            for pattern, gate_type in self.COMPLIANCE_GATE_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Extract the consequence
                    consequence_match = re.search(
                        r'(will\s+(?:result\s+in|be|not\s+be)[^.]+\.)',
                        req.full_text, re.IGNORECASE
                    )
                    consequence = consequence_match.group(1) if consequence_match else "Non-compliance may result in disqualification"

                    # Determine action required
                    action = self._determine_action(req.full_text, gate_type)

                    gates.append(ComplianceGate(
                        requirement_id=req.rfp_reference or req.generated_id,
                        full_text=req.full_text[:500],
                        gate_type=gate_type,
                        consequence=consequence,
                        source_location=f"Section {req.source_section.value}" if req.source_section else "Unknown",
                        page_number=req.page_number,
                        action_required=action
                    ))
                    break  # Only one gate per requirement

        return gates

    def _determine_action(self, text: str, gate_type: str) -> str:
        """Determine action required for a compliance gate"""
        text_lower = text.lower()

        if 'certif' in text_lower:
            return "Obtain and submit required certification before proposal deadline"
        elif 'oci' in text_lower or 'conflict of interest' in text_lower:
            return "Prepare and submit Organizational Conflict of Interest mitigation plan"
        elif 'past performance' in text_lower:
            return "Ensure past performance references meet minimum requirements"
        elif 'experience' in text_lower:
            return "Verify experience requirements are met with documented evidence"
        elif 'page' in text_lower or 'format' in text_lower:
            return "Strictly adhere to formatting and page limit requirements"
        else:
            return "Review requirement and ensure full compliance before submission"

    def _detect_conflicts(self, result: ExtractionResult) -> List[ConflictFinding]:
        """Detect conflicts in RFP requirements"""
        conflicts = []

        # Combine all requirement text for analysis
        all_text = "\n".join(r.full_text for r in result.all_requirements)

        # Detect page limit conflicts
        page_conflict = self._detect_page_limit_conflict(all_text, result)
        if page_conflict:
            conflicts.append(page_conflict)

        # Detect date conflicts (due date vs questions deadline, etc.)
        date_conflict = self._detect_date_conflicts(all_text, result)
        if date_conflict:
            conflicts.append(date_conflict)

        return conflicts

    def _detect_page_limit_conflict(
        self,
        text: str,
        result: ExtractionResult
    ) -> Optional[ConflictFinding]:
        """Detect conflicts between stated page limits and allocations"""
        # Find stated page limit
        total_limit = None
        limit_location = None

        for pattern in self.PAGE_LIMIT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    total_limit = int(match.group(1))
                    # Find which requirement contains this
                    for req in result.all_requirements:
                        if match.group(0) in req.full_text:
                            limit_location = {
                                'section': req.source_section.value if req.source_section else 'UNK',
                                'page': req.page_number,
                                'text': match.group(0)
                            }
                            break
                    break
                except ValueError:
                    continue

        if not total_limit:
            return None

        # Find page allocations
        allocations = []
        total_allocated = 0

        for pattern in self.PAGE_ALLOCATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    pages = int(match.group(1))
                    allocations.append({
                        'text': match.group(0),
                        'pages': pages
                    })
                    total_allocated += pages
                except ValueError:
                    continue

        # Check for conflict
        if total_allocated > total_limit:
            return ConflictFinding(
                conflict_type="page_limit",
                severity=AuditSeverity.CRITICAL,
                description=f"Page limit conflict detected: {total_allocated} pages allocated but limit is {total_limit} pages",
                locations=[limit_location] if limit_location else [],
                draft_qa=self._generate_page_limit_qa(total_limit, total_allocated, allocations),
                recommendation=f"Submit Q&A to Contracting Officer to clarify page limit. Do not assume {total_limit} is a typo."
            )

        return None

    def _generate_page_limit_qa(
        self,
        limit: int,
        allocated: int,
        allocations: List[Dict]
    ) -> str:
        """Generate draft Q&A for page limit conflict"""
        allocation_text = ", ".join(
            f"{a['text']}" for a in allocations[:5]
        )

        return f"""Question: Section L specifies a {limit}-page limit for the Technical Volume. However, the individual section/factor allocations ({allocation_text}) total {allocated} pages. Please clarify whether:
(a) The overall page limit should be {allocated} pages to accommodate all required sections, or
(b) The individual allocations should be reduced proportionally to fit within {limit} pages, or
(c) The Government will provide revised page allocations.

This clarification is necessary to ensure our proposal addresses all evaluation factors within the specified format requirements."""

    def _detect_date_conflicts(
        self,
        text: str,
        result: ExtractionResult
    ) -> Optional[ConflictFinding]:
        """Detect conflicts in dates (due date before Q&A deadline, etc.)"""
        # This would require date parsing - simplified for now
        # Look for obvious conflicts like "questions due after proposals"

        date_patterns = [
            r'(?:questions?|inquiries?)\s+(?:due|deadline)[:\s]+([^\n]+)',
            r'(?:proposal|offer)\s+(?:due|deadline)[:\s]+([^\n]+)',
        ]

        # Simplified - would need actual date parsing for full implementation
        return None

    def _generate_findings(
        self,
        report: AccuracyReport,
        result: ExtractionResult
    ) -> List[AuditFinding]:
        """Generate audit findings based on analysis"""
        findings = []

        # Check accuracy gaps
        if report.manual_counts:
            for category, variance in report.variance.items():
                if variance < 0:  # PropelAI found fewer than manual
                    severity = AuditSeverity.CRITICAL if category == 'mandatory' else AuditSeverity.HIGH
                    findings.append(AuditFinding(
                        severity=severity,
                        category="accuracy",
                        title=f"Extraction gap in {category}",
                        description=f"PropelAI extracted {abs(variance)} fewer {category} requirements than manual count",
                        recommendation=f"Review {category} requirements for missed items. Check section detection patterns."
                    ))

        # Check section detection
        if report.section_detection_rate < 80:
            findings.append(AuditFinding(
                severity=AuditSeverity.HIGH,
                category="quality",
                title="Low section detection rate",
                description=f"Only {report.section_detection_rate}% of requirements have known section context",
                recommendation="Improve section header detection patterns. Check for non-standard RFP format."
            ))

        # Check RFP ID preservation
        total = len(result.all_requirements)
        if total > 0:
            id_rate = (report.requirements_with_rfp_ids / total) * 100
            if id_rate < 50:
                findings.append(AuditFinding(
                    severity=AuditSeverity.MEDIUM,
                    category="quality",
                    title="Low RFP ID preservation",
                    description=f"Only {id_rate:.1f}% of requirements preserve RFP's own reference IDs",
                    recommendation="Review RFP reference detection patterns (L.4.B.2, C.3.1.a, etc.)"
                ))

        # Check for compliance gates
        if report.compliance_gates:
            findings.append(AuditFinding(
                severity=AuditSeverity.CRITICAL,
                category="compliance",
                title=f"{len(report.compliance_gates)} compliance gates detected",
                description="Pass/fail requirements that could cause disqualification",
                recommendation="Review all compliance gates immediately. Assign owners and track completion."
            ))

        # Check for conflicts
        critical_conflicts = [c for c in report.conflicts if c.severity == AuditSeverity.CRITICAL]
        if critical_conflicts:
            findings.append(AuditFinding(
                severity=AuditSeverity.CRITICAL,
                category="conflict",
                title=f"{len(critical_conflicts)} critical conflicts detected",
                description="RFP contains conflicting requirements that need clarification",
                recommendation="Submit Q&A questions to Contracting Officer before proposal deadline"
            ))

        return findings

    def _generate_summary(self, report: AccuracyReport) -> str:
        """Generate human-readable summary"""
        lines = []

        if report.manual_counts:
            lines.append(f"Overall Accuracy: {report.overall_accuracy}%")
            if report.overall_accuracy >= 95:
                lines.append("✅ PASSED accuracy threshold (95%)")
            else:
                lines.append("❌ FAILED accuracy threshold (95%)")

        lines.append(f"Total Requirements: {report.propelai_counts.get('total', 0)}")
        lines.append(f"Section Detection Rate: {report.section_detection_rate}%")

        critical_count = sum(1 for f in report.findings if f.severity == AuditSeverity.CRITICAL)
        if critical_count > 0:
            lines.append(f"⚠️ {critical_count} CRITICAL findings require attention")

        if report.compliance_gates:
            lines.append(f"🚨 {len(report.compliance_gates)} compliance gates detected")

        if report.conflicts:
            lines.append(f"⚡ {len(report.conflicts)} conflicts detected")

        return "\n".join(lines)

    def export_report_json(self, report: AccuracyReport) -> Dict[str, Any]:
        """Export report as JSON-serializable dict"""
        return {
            'audit_date': report.audit_date,
            'rfp_id': report.rfp_id,
            'overall_accuracy': report.overall_accuracy,
            'accuracy_by_category': report.accuracy_by_category,
            'propelai_counts': report.propelai_counts,
            'manual_counts': report.manual_counts,
            'variance': report.variance,
            'section_detection_rate': report.section_detection_rate,
            'requirements_with_rfp_ids': report.requirements_with_rfp_ids,
            'passed': report.passed,
            'summary': report.summary,
            'findings': [
                {
                    'severity': f.severity.value,
                    'category': f.category,
                    'title': f.title,
                    'description': f.description,
                    'recommendation': f.recommendation
                }
                for f in report.findings
            ],
            'conflicts': [
                {
                    'type': c.conflict_type,
                    'severity': c.severity.value,
                    'description': c.description,
                    'draft_qa': c.draft_qa,
                    'recommendation': c.recommendation
                }
                for c in report.conflicts
            ],
            'compliance_gates': [
                {
                    'requirement_id': g.requirement_id,
                    'gate_type': g.gate_type,
                    'consequence': g.consequence,
                    'page_number': g.page_number,
                    'action_required': g.action_required
                }
                for g in report.compliance_gates
            ]
        }


# Convenience function
def audit_extraction(
    extraction_result: ExtractionResult,
    manual_counts: Optional[Dict[str, int]] = None,
    rfp_id: str = ""
) -> AccuracyReport:
    """
    Convenience function to audit extraction accuracy.

    Usage:
        from agents.enhanced_compliance.accuracy_audit import audit_extraction

        report = audit_extraction(
            extraction_result=result,
            manual_counts={'mandatory': 89, 'total': 156}
        )

        if not report.passed:
            print("Accuracy audit failed!")
            for finding in report.findings:
                print(f"  {finding.severity.value}: {finding.title}")
    """
    auditor = AccuracyAuditor()
    return auditor.audit_extraction(extraction_result, manual_counts, rfp_id)
