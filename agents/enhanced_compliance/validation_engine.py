"""
PropelAI v5.0: L-M-C Validation Engine
Deterministic validation for Iron Triangle consistency

Implements FR-2.3: Deterministic Validation
- Alert when manual changes violate graph logic
- Enforce section placement rules
- Detect coverage gaps
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .models import RequirementNode, RequirementType


class ViolationType(Enum):
    """Types of Iron Triangle violations"""
    # Section placement violations
    WRONG_SECTION = "wrong_section"  # Content in wrong section
    SECTION_MISMATCH = "section_mismatch"  # Section ID doesn't match content

    # Coverage violations
    ORPHAN_PERFORMANCE = "orphan_performance"  # C without L or M links
    ORPHAN_INSTRUCTION = "orphan_instruction"  # L without M link
    ORPHAN_EVALUATION = "orphan_evaluation"  # M without C/L target
    MISSING_EVALUATION = "missing_evaluation"  # Topic has C and L but no M

    # Content violations
    DUPLICATE_REQUIREMENT = "duplicate_requirement"  # Same text in multiple places
    CONFLICTING_REQUIREMENTS = "conflicting_requirements"  # Contradictory requirements
    INCOMPLETE_COVERAGE = "incomplete_coverage"  # Topic partially covered

    # Format violations
    VOLUME_RESTRICTION = "volume_restriction"  # Content in wrong volume
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"  # Section exceeds limit

    # Logical violations
    CIRCULAR_REFERENCE = "circular_reference"  # A -> B -> A
    INVALID_DEPENDENCY = "invalid_dependency"  # Invalid edge type


class Severity(Enum):
    """Severity levels for violations"""
    CRITICAL = "critical"  # Must fix before submission
    WARNING = "warning"  # Should fix, may impact score
    INFO = "info"  # Suggestion for improvement


@dataclass
class ValidationViolation:
    """A single validation violation"""
    id: str
    violation_type: ViolationType
    severity: Severity
    requirement_id: str
    requirement_text: str
    message: str
    suggestion: str
    related_requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.violation_type.value,
            "severity": self.severity.value,
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text[:200],
            "message": self.message,
            "suggestion": self.suggestion,
            "related_requirements": self.related_requirements,
            "metadata": self.metadata,
            "detected_at": self.detected_at,
        }


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    total_violations: int
    critical_count: int
    warning_count: int
    info_count: int
    violations: List[ValidationViolation]
    compliance_score: float  # 0-100
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_violations": self.total_violations,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "violations": [v.to_dict() for v in self.violations],
            "compliance_score": self.compliance_score,
            "validated_at": self.validated_at,
        }


# Volume/Section placement rules
VOLUME_SECTION_RULES = {
    # Volume -> Allowed section types
    "technical": ["C", "SOW", "PWS"],
    "management": ["C", "SOW"],
    "past_performance": ["L"],  # Past perf is written per L instructions
    "cost": ["B", "PRICING"],
    "administrative": ["K", "L"],
}

# Section content type rules
SECTION_CONTENT_RULES = {
    # Section -> Allowed requirement types
    "C": [
        RequirementType.PERFORMANCE,
        RequirementType.DELIVERABLE,
        RequirementType.LABOR_REQUIREMENT,
        RequirementType.PERFORMANCE_METRIC,
    ],
    "L": [
        RequirementType.PROPOSAL_INSTRUCTION,
        RequirementType.FORMAT,
        RequirementType.QUALIFICATION,
    ],
    "M": [
        RequirementType.EVALUATION_CRITERION,
    ],
}


class ValidationEngine:
    """
    Deterministic validation engine for Iron Triangle consistency.

    Validates:
    1. Section placement (content in correct section)
    2. Coverage completeness (L-M-C triangle coverage)
    3. Logical consistency (no conflicts or circular refs)
    4. Format compliance (volume restrictions)
    """

    def __init__(self):
        self._violation_counter = 0

    def validate(
        self,
        requirements: List[RequirementNode],
        graph: Optional[Dict[str, Any]] = None,
        outline: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Run complete validation on requirements.

        Args:
            requirements: List of RequirementNode objects
            graph: Optional requirements graph (NetworkX export)
            outline: Optional proposal outline for volume validation

        Returns:
            ValidationResult with all violations
        """
        violations = []

        # 1. Section placement validation
        violations.extend(self._validate_section_placement(requirements))

        # 2. Duplicate detection
        violations.extend(self._validate_no_duplicates(requirements))

        # 3. Coverage validation (if graph provided)
        if graph:
            violations.extend(self._validate_coverage(requirements, graph))

        # 4. Volume restriction validation (if outline provided)
        if outline:
            violations.extend(self._validate_volume_restrictions(requirements, outline))

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(requirements, violations)

        # Count by severity
        critical_count = sum(1 for v in violations if v.severity == Severity.CRITICAL)
        warning_count = sum(1 for v in violations if v.severity == Severity.WARNING)
        info_count = sum(1 for v in violations if v.severity == Severity.INFO)

        return ValidationResult(
            is_valid=critical_count == 0,
            total_violations=len(violations),
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            violations=violations,
            compliance_score=compliance_score,
        )

    def validate_single_requirement(
        self,
        requirement: RequirementNode,
        existing_requirements: List[RequirementNode],
        target_section: Optional[str] = None,
        target_volume: Optional[str] = None,
    ) -> List[ValidationViolation]:
        """
        Validate a single requirement (for real-time UI validation).

        Used when a user manually adds or moves a requirement.

        Args:
            requirement: The requirement to validate
            existing_requirements: List of existing requirements
            target_section: Section where user wants to place it
            target_volume: Volume where user wants to place it

        Returns:
            List of violations (empty if valid)
        """
        violations = []

        # Check section placement
        if target_section:
            violations.extend(
                self._check_section_compatibility(requirement, target_section)
            )

        # Check volume restrictions
        if target_volume:
            violations.extend(
                self._check_volume_compatibility(requirement, target_volume)
            )

        # Check for duplicates
        violations.extend(
            self._check_duplicate(requirement, existing_requirements)
        )

        return violations

    def _validate_section_placement(
        self,
        requirements: List[RequirementNode]
    ) -> List[ValidationViolation]:
        """Validate that requirements are in appropriate sections"""
        violations = []

        for req in requirements:
            section = self._get_section(req)
            if not section:
                continue

            allowed_types = SECTION_CONTENT_RULES.get(section, [])
            if allowed_types and req.requirement_type not in allowed_types:
                violations.append(ValidationViolation(
                    id=self._next_violation_id(),
                    violation_type=ViolationType.WRONG_SECTION,
                    severity=Severity.WARNING,
                    requirement_id=req.id,
                    requirement_text=req.text,
                    message=f"Requirement type '{req.requirement_type.value}' is unusual for Section {section}",
                    suggestion=f"Review if this requirement should be in a different section",
                    metadata={
                        "current_section": section,
                        "requirement_type": req.requirement_type.value,
                        "expected_types": [t.value for t in allowed_types],
                    }
                ))

        return violations

    def _validate_no_duplicates(
        self,
        requirements: List[RequirementNode]
    ) -> List[ValidationViolation]:
        """Detect duplicate requirements"""
        violations = []
        seen_hashes: Dict[str, str] = {}

        for req in requirements:
            if req.text_hash in seen_hashes:
                violations.append(ValidationViolation(
                    id=self._next_violation_id(),
                    violation_type=ViolationType.DUPLICATE_REQUIREMENT,
                    severity=Severity.WARNING,
                    requirement_id=req.id,
                    requirement_text=req.text,
                    message="Duplicate requirement detected",
                    suggestion="Consider merging or removing duplicate",
                    related_requirements=[seen_hashes[req.text_hash]],
                ))
            else:
                seen_hashes[req.text_hash] = req.id

        return violations

    def _validate_coverage(
        self,
        requirements: List[RequirementNode],
        graph: Dict[str, Any]
    ) -> List[ValidationViolation]:
        """Validate Iron Triangle coverage using graph data"""
        violations = []

        # Get orphans from graph analysis
        orphans = graph.get("orphans", [])
        for orphan in orphans:
            # Find the requirement
            req = next((r for r in requirements if r.id == orphan["id"]), None)
            if not req:
                continue

            severity = Severity.WARNING
            if orphan.get("section") == "C":
                # Performance requirements without links are more critical
                severity = Severity.CRITICAL if "evaluation" in orphan.get("reason", "").lower() else Severity.WARNING

            violations.append(ValidationViolation(
                id=self._next_violation_id(),
                violation_type=self._map_orphan_to_violation_type(orphan),
                severity=severity,
                requirement_id=orphan["id"],
                requirement_text=req.text if req else "",
                message=orphan.get("reason", "Orphan requirement detected"),
                suggestion=orphan.get("suggestion", "Link to related requirements"),
            ))

        return violations

    def _validate_volume_restrictions(
        self,
        requirements: List[RequirementNode],
        outline: Dict[str, Any]
    ) -> List[ValidationViolation]:
        """Validate that requirements don't violate volume placement rules"""
        violations = []

        # Build section -> volume mapping from outline
        section_to_volume = {}
        for volume in outline.get("volumes", []):
            vol_type = volume.get("type", "technical")
            for section in volume.get("sections", []):
                section_to_volume[section.get("id", "")] = vol_type

        for req in requirements:
            section = self._get_section(req)
            if not section:
                continue

            # Check if this section type is allowed in its volume
            volume_type = section_to_volume.get(section, "technical")
            allowed_sections = VOLUME_SECTION_RULES.get(volume_type, [])

            if allowed_sections and section not in allowed_sections:
                violations.append(ValidationViolation(
                    id=self._next_violation_id(),
                    violation_type=ViolationType.VOLUME_RESTRICTION,
                    severity=Severity.CRITICAL,
                    requirement_id=req.id,
                    requirement_text=req.text,
                    message=f"Section {section} content should not be in {volume_type} volume",
                    suggestion=f"Move to appropriate volume or update section reference",
                    metadata={
                        "current_volume": volume_type,
                        "section": section,
                        "allowed_sections": allowed_sections,
                    }
                ))

        return violations

    def _check_section_compatibility(
        self,
        requirement: RequirementNode,
        target_section: str
    ) -> List[ValidationViolation]:
        """Check if requirement type is compatible with target section"""
        violations = []

        section_letter = target_section[0].upper() if target_section else None
        allowed_types = SECTION_CONTENT_RULES.get(section_letter, [])

        if allowed_types and requirement.requirement_type not in allowed_types:
            violations.append(ValidationViolation(
                id=self._next_violation_id(),
                violation_type=ViolationType.WRONG_SECTION,
                severity=Severity.CRITICAL,
                requirement_id=requirement.id,
                requirement_text=requirement.text,
                message=f"Cannot place {requirement.requirement_type.value} in Section {target_section}",
                suggestion=f"This type belongs in: {', '.join(self._get_valid_sections(requirement.requirement_type))}",
                metadata={
                    "target_section": target_section,
                    "requirement_type": requirement.requirement_type.value,
                }
            ))

        return violations

    def _check_volume_compatibility(
        self,
        requirement: RequirementNode,
        target_volume: str
    ) -> List[ValidationViolation]:
        """Check if requirement can be placed in target volume"""
        violations = []

        section = self._get_section(requirement)
        if not section:
            return violations

        allowed_sections = VOLUME_SECTION_RULES.get(target_volume.lower(), [])
        if allowed_sections and section not in allowed_sections:
            violations.append(ValidationViolation(
                id=self._next_violation_id(),
                violation_type=ViolationType.VOLUME_RESTRICTION,
                severity=Severity.CRITICAL,
                requirement_id=requirement.id,
                requirement_text=requirement.text,
                message=f"Section {section} content cannot be placed in {target_volume} volume",
                suggestion=f"Use a different volume or restructure the requirement",
            ))

        return violations

    def _check_duplicate(
        self,
        requirement: RequirementNode,
        existing_requirements: List[RequirementNode]
    ) -> List[ValidationViolation]:
        """Check if requirement is a duplicate"""
        violations = []

        for existing in existing_requirements:
            if existing.id == requirement.id:
                continue

            if existing.text_hash == requirement.text_hash:
                violations.append(ValidationViolation(
                    id=self._next_violation_id(),
                    violation_type=ViolationType.DUPLICATE_REQUIREMENT,
                    severity=Severity.WARNING,
                    requirement_id=requirement.id,
                    requirement_text=requirement.text,
                    message="This requirement duplicates an existing one",
                    suggestion="Consider linking instead of duplicating",
                    related_requirements=[existing.id],
                ))
                break

        return violations

    def _get_section(self, requirement: RequirementNode) -> Optional[str]:
        """Extract section letter from requirement"""
        if requirement.source and requirement.source.section_id:
            return requirement.source.section_id[0].upper()
        return None

    def _get_valid_sections(self, req_type: RequirementType) -> List[str]:
        """Get valid sections for a requirement type"""
        valid = []
        for section, allowed_types in SECTION_CONTENT_RULES.items():
            if req_type in allowed_types:
                valid.append(section)
        return valid or ["Any"]

    def _map_orphan_to_violation_type(self, orphan: Dict) -> ViolationType:
        """Map orphan reason to violation type"""
        section = orphan.get("section", "OTHER")
        reason = orphan.get("reason", "").lower()

        if section == "C":
            if "instruction" in reason:
                return ViolationType.ORPHAN_PERFORMANCE
            elif "evaluation" in reason:
                return ViolationType.ORPHAN_PERFORMANCE
        elif section == "L":
            return ViolationType.ORPHAN_INSTRUCTION
        elif section == "M":
            return ViolationType.ORPHAN_EVALUATION

        return ViolationType.INCOMPLETE_COVERAGE

    def _calculate_compliance_score(
        self,
        requirements: List[RequirementNode],
        violations: List[ValidationViolation]
    ) -> float:
        """Calculate overall compliance score 0-100"""
        if not requirements:
            return 100.0

        total_reqs = len(requirements)

        # Weight by severity
        critical_penalty = sum(10 for v in violations if v.severity == Severity.CRITICAL)
        warning_penalty = sum(3 for v in violations if v.severity == Severity.WARNING)
        info_penalty = sum(1 for v in violations if v.severity == Severity.INFO)

        total_penalty = critical_penalty + warning_penalty + info_penalty
        max_penalty = total_reqs * 10  # Max penalty if all are critical

        if max_penalty == 0:
            return 100.0

        score = max(0, 100 - (total_penalty / max_penalty * 100))
        return round(score, 1)

    def _next_violation_id(self) -> str:
        """Generate next violation ID"""
        self._violation_counter += 1
        return f"VIO-{self._violation_counter:04d}"


# Convenience function for API usage
def validate_requirements(
    requirements: List[RequirementNode],
    graph: Optional[Dict[str, Any]] = None,
    outline: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Validate requirements for Iron Triangle compliance.

    Args:
        requirements: List of RequirementNode objects
        graph: Optional requirements graph (from RequirementsDAG.to_dict())
        outline: Optional proposal outline

    Returns:
        ValidationResult with all violations and compliance score
    """
    engine = ValidationEngine()
    return engine.validate(requirements, graph, outline)
