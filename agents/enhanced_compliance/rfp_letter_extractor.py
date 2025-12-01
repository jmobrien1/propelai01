"""
RFP Letter Extractor
Phase 4.1 - Sprint 2

Extracts critical submission instructions from RFP Letters and transmittal documents.
Targets: Volume structure, page limits, formatting rules, compliance traps.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class FormattingConstraint(Enum):
    """Types of formatting constraints"""
    FONT_FAMILY = "font_family"
    FONT_SIZE = "font_size"
    MARGINS = "margins"
    LINE_SPACING = "line_spacing"
    PAGE_SIZE = "page_size"
    ORIENTATION = "orientation"


@dataclass
class VolumeStructure:
    """Represents a proposal volume definition"""
    volume_id: str  # "I", "II", "III"
    volume_name: str  # "Technical Approach"
    page_limit: Optional[int] = None
    page_limit_text: Optional[str] = None  # "30 pages" or "2 pages per reference"
    content_description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'volume_id': self.volume_id,
            'volume_name': self.volume_name,
            'page_limit': self.page_limit,
            'page_limit_text': self.page_limit_text,
            'content_description': self.content_description
        }


@dataclass
class FormattingRule:
    """Represents a formatting requirement"""
    rule_type: FormattingConstraint
    value: str
    source_text: str
    mandatory: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'rule_type': self.rule_type.value,
            'value': self.value,
            'source_text': self.source_text,
            'mandatory': self.mandatory
        }


@dataclass
class ComplianceFlag:
    """Critical compliance rule that could cause rejection"""
    flag_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM
    category: str  # price_isolation, page_limit_enforcement, mandatory_registration, etc.
    rule_text: str
    source_location: str
    impact: str
    action_required: str
    
    def to_dict(self) -> Dict:
        return {
            'flag_id': self.flag_id,
            'severity': self.severity,
            'category': self.category,
            'rule_text': self.rule_text,
            'source_location': self.source_location,
            'impact': self.impact,
            'action_required': self.action_required
        }


class RFPLetterExtractor:
    """
    Extracts submission instructions from RFP Letters.
    
    Targets:
    - Volume structure ("three volumes")
    - Page limits ("30 pages", "2 pages per reference")
    - Formatting rules ("11-point Times New Roman")
    - Critical constraints ("Price ONLY in Volume III")
    - Due dates
    - Submission instructions
    """
    
    # Patterns for volume detection
    VOLUME_PATTERNS = [
        # "Volume I - Technical Approach"
        r'volume\s+([IVX]+|[123])\s*[-–—:]\s*([^.\n]{5,50})',
        # "Volume I: Technical Approach"
        r'volume\s+([IVX]+|[123]):\s*([^.\n]{5,50})',
        # "Technical Volume (Volume I)"
        r'([^.\n]{5,50})\s*\(?volume\s+([IVX]+|[123])\)?',
    ]
    
    # Patterns for page limits
    PAGE_LIMIT_PATTERNS = [
        # "shall not exceed 30 pages"
        r'shall not exceed\s+(\d+)\s+pages?',
        # "limited to 30 pages"
        r'limited to\s+(\d+)\s+pages?',
        # "30 pages maximum"
        r'(\d+)\s+pages?\s+maximum',
        # "not to exceed 30 pages"
        r'not to exceed\s+(\d+)\s+pages?',
        # "2 pages per reference"
        r'(\d+)\s+pages?\s+per\s+(reference|project|contract)',
        # "30-page limit"
        r'(\d+)[-–]page\s+limit',
    ]
    
    # Patterns for font requirements
    FONT_PATTERNS = [
        # "11-point Times New Roman"
        r'(\d+)[-–]?point\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        # "Times New Roman, 11 point"
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,?\s*(\d+)\s*point',
        # "font size of 11 points"
        r'font size of\s+(\d+)\s+points?',
    ]
    
    # Patterns for margin requirements
    MARGIN_PATTERNS = [
        # "margins of not less than one (1) inch"
        r'margins?\s+of\s+not less than\s+(?:one\s*\(?1\)?|\d+)\s*inch',
        # "1-inch margins"
        r'(\d+)[-–]inch\s+margins?',
        # "margins shall be 1 inch"
        r'margins?\s+shall be\s+(\d+)\s*inch',
    ]
    
    # Patterns for critical compliance flags
    COMPLIANCE_FLAG_PATTERNS = {
        'price_isolation': [
            r'price.*only.*volume\s+([IVX]+)',
            r'pricing.*shall\s+(?:only\s+)?(?:be\s+)?(?:in|appear(?:\s+in)?)\s+volume\s+([IVX]+)',
            r'cost.*limited to.*volume\s+([IVX]+)',
            r'volume\s+([IVX]+).*(?:only|exclusively).*price',
        ],
        'page_limit_enforcement': [
            r'exceed(?:ing)?\s+(?:the\s+)?page limit.*(?:will|shall)\s+not be evaluated',
            r'proposals?.*over.*pages?.*(?:will be|shall be)?\s*rejected',
            r'(?:will|shall)\s+not.*evaluate.*exceed.*page',
        ],
        'mandatory_registration': [
            r'must be registered.*sam\.gov',
            r'registration.*sam\.gov.*(?:is\s+)?(?:required|mandatory)',
            r'sam\.gov.*registration.*prior to\s+award',
        ],
        'late_submission': [
            r'late\s+(?:submissions?|proposals?).*(?:will|shall)\s+not.*accept',
            r'(?:will|shall)\s+not.*(?:accept|consider).*late',
        ],
        'content_restriction': [
            r'shall not.*(?:include|contain|reference).*(?:in|within)\s+volume\s+([IVX]+)',
            r'volume\s+([IVX]+).*shall not.*(?:include|contain)',
        ]
    }
    
    def __init__(self):
        self.volumes: List[VolumeStructure] = []
        self.formatting_rules: List[FormattingRule] = []
        self.compliance_flags: List[ComplianceFlag] = []
        self.metadata: Dict = {}
    
    def extract_from_text(self, text: str, filename: str = "RFP Letter") -> Dict:
        """
        Extract all submission instructions from RFP Letter text.
        
        Args:
            text: Full text of RFP Letter
            filename: Source document name
        
        Returns:
            Dict with volumes, formatting, flags, and metadata
        """
        self.volumes = []
        self.formatting_rules = []
        self.compliance_flags = []
        self.metadata = {'source': filename}
        
        # Step 1: Extract volume structure
        self._extract_volumes(text)
        
        # Step 2: Extract page limits
        self._extract_page_limits(text)
        
        # Step 3: Extract formatting rules
        self._extract_formatting_rules(text)
        
        # Step 4: Detect compliance flags
        self._detect_compliance_flags(text, filename)
        
        # Step 5: Extract due dates
        self._extract_due_dates(text)
        
        return self.to_dict()
    
    def _extract_volumes(self, text: str):
        """Extract volume structure definitions"""
        text_lower = text.lower()
        
        for pattern in self.VOLUME_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 2:
                    vol_id = match.group(1).upper().strip()
                    vol_name = match.group(2).strip()
                    
                    # Normalize volume ID (1 -> I, 2 -> II, etc.)
                    vol_id = self._normalize_volume_id(vol_id)
                    
                    # Validate: Volume ID must be a valid Roman numeral or number
                    if vol_id not in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']:
                        continue
                    
                    # Clean volume name
                    vol_name = self._clean_volume_name(vol_name)
                    
                    # Skip if volume name is too short (likely a false positive)
                    if len(vol_name) < 3:
                        continue
                    
                    # Check if already exists
                    if not any(v.volume_id == vol_id for v in self.volumes):
                        volume = VolumeStructure(
                            volume_id=vol_id,
                            volume_name=vol_name
                        )
                        self.volumes.append(volume)
        
        # Sort by volume ID
        self.volumes.sort(key=lambda v: self._volume_sort_key(v.volume_id))
    
    def _normalize_volume_id(self, vol_id: str) -> str:
        """Convert volume ID to Roman numeral (1->I, 2->II, 3->III)"""
        roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V'}
        return roman_map.get(vol_id, vol_id)
    
    def _volume_sort_key(self, vol_id: str) -> int:
        """Get sort key for volume ID"""
        order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
        return order.get(vol_id, 99)
    
    def _clean_volume_name(self, name: str) -> str:
        """Clean volume name"""
        # Remove trailing punctuation
        name = name.rstrip('.,;:')
        # Capitalize first letter of each word
        name = ' '.join(word.capitalize() for word in name.split())
        return name
    
    def _extract_page_limits(self, text: str):
        """Extract page limits and associate with volumes"""
        # Split text into paragraphs for context
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_lower = para.lower()
            
            # Check for volume mentions in paragraph
            volume_in_para = None
            for vol in self.volumes:
                if f"volume {vol.volume_id.lower()}" in para_lower:
                    volume_in_para = vol
                    break
            
            # Extract page limits
            for pattern in self.PAGE_LIMIT_PATTERNS:
                matches = re.finditer(pattern, para, re.IGNORECASE)
                for match in matches:
                    page_limit_text = match.group(0)
                    
                    # Extract number
                    numbers = re.findall(r'\d+', match.group(0))
                    page_limit = int(numbers[0]) if numbers else None
                    
                    # Check if "per reference/project"
                    is_per_item = bool(re.search(r'per\s+(reference|project|contract)', match.group(0), re.I))
                    
                    if volume_in_para:
                        # Associate with specific volume
                        volume_in_para.page_limit = page_limit
                        volume_in_para.page_limit_text = page_limit_text
                        if is_per_item:
                            volume_in_para.page_limit_text += " (per item)"
                    else:
                        # General page limit (might be total)
                        self.metadata['general_page_limit'] = page_limit
                        self.metadata['general_page_limit_text'] = page_limit_text
    
    def _extract_formatting_rules(self, text: str):
        """Extract formatting requirements"""
        seen_rules = set()  # Track unique rules to avoid duplicates
        
        # Font family and size
        for pattern in self.FONT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'point' in match.group(0).lower():
                    # Extract size and family
                    size_match = re.search(r'(\d+)', match.group(0))
                    family_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', match.group(0))
                    
                    if size_match:
                        rule_key = ('font_size', f"{size_match.group(1)} pt")
                        if rule_key not in seen_rules:
                            rule = FormattingRule(
                                rule_type=FormattingConstraint.FONT_SIZE,
                                value=f"{size_match.group(1)} pt",
                                source_text=match.group(0).strip()
                            )
                            self.formatting_rules.append(rule)
                            seen_rules.add(rule_key)
                    
                    if family_match and family_match.group(1).lower() not in ['point', 'inch']:
                        rule_key = ('font_family', family_match.group(1))
                        if rule_key not in seen_rules:
                            rule = FormattingRule(
                                rule_type=FormattingConstraint.FONT_FAMILY,
                                value=family_match.group(1),
                                source_text=match.group(0).strip()
                            )
                            self.formatting_rules.append(rule)
                            seen_rules.add(rule_key)
        
        # Margins
        for pattern in self.MARGIN_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract margin size
                size_match = re.search(r'(\d+|one)', match.group(0), re.I)
                if size_match:
                    size = '1' if size_match.group(1).lower() == 'one' else size_match.group(1)
                    rule_key = ('margins', f"{size} inch")
                    if rule_key not in seen_rules:
                        rule = FormattingRule(
                            rule_type=FormattingConstraint.MARGINS,
                            value=f"{size} inch",
                            source_text=match.group(0).strip()
                        )
                        self.formatting_rules.append(rule)
                        seen_rules.add(rule_key)
        
        # Line spacing
        if re.search(r'single[-\s]spac', text, re.I):
            rule = FormattingRule(
                rule_type=FormattingConstraint.LINE_SPACING,
                value="single",
                source_text="single-spaced"
            )
            self.formatting_rules.append(rule)
        elif re.search(r'double[-\s]spac', text, re.I):
            rule = FormattingRule(
                rule_type=FormattingConstraint.LINE_SPACING,
                value="double",
                source_text="double-spaced"
            )
            self.formatting_rules.append(rule)
        
        # Page size
        if re.search(r'8\.5\s*x\s*11', text, re.I) or re.search(r'8-1/2\s*x\s*11', text, re.I):
            rule = FormattingRule(
                rule_type=FormattingConstraint.PAGE_SIZE,
                value="8.5 x 11 inches",
                source_text="8.5 x 11 inch paper"
            )
            self.formatting_rules.append(rule)
    
    def _detect_compliance_flags(self, text: str, filename: str):
        """Detect critical compliance rules"""
        flag_counter = 1
        
        for category, patterns in self.COMPLIANCE_FLAG_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Determine severity
                    severity = self._determine_severity(category, match.group(0))
                    
                    # Extract context (surrounding text)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]
                    
                    # Create flag
                    flag = ComplianceFlag(
                        flag_id=f"CF-{flag_counter:03d}",
                        severity=severity,
                        category=category,
                        rule_text=match.group(0),
                        source_location=filename,
                        impact=self._get_impact_text(category),
                        action_required=self._get_action_text(category)
                    )
                    
                    # Avoid duplicates
                    if not any(f.rule_text.lower() == flag.rule_text.lower() for f in self.compliance_flags):
                        self.compliance_flags.append(flag)
                        flag_counter += 1
    
    def _determine_severity(self, category: str, rule_text: str) -> str:
        """Determine severity level of compliance flag"""
        critical_categories = ['price_isolation', 'page_limit_enforcement', 'late_submission']
        
        if category in critical_categories:
            return 'CRITICAL'
        
        # Check for keywords indicating severity
        text_lower = rule_text.lower()
        if any(kw in text_lower for kw in ['reject', 'will not', 'shall not', 'disqualif']):
            return 'CRITICAL'
        elif any(kw in text_lower for kw in ['must', 'required', 'mandatory']):
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _get_impact_text(self, category: str) -> str:
        """Get impact description for category"""
        impacts = {
            'price_isolation': 'Proposal may be rejected if pricing appears in wrong volume',
            'page_limit_enforcement': 'Proposal may not be evaluated if page limit exceeded',
            'mandatory_registration': 'Cannot receive award without SAM.gov registration',
            'late_submission': 'Proposal will not be considered if submitted after deadline',
            'content_restriction': 'Violating content restrictions may result in non-compliance'
        }
        return impacts.get(category, 'May impact proposal evaluation or acceptance')
    
    def _get_action_text(self, category: str) -> str:
        """Get recommended action for category"""
        actions = {
            'price_isolation': 'Ensure all pricing/cost information only appears in designated volume',
            'page_limit_enforcement': 'Strictly adhere to page limits; trim content if necessary',
            'mandatory_registration': 'Verify SAM.gov registration is active before submission',
            'late_submission': 'Submit proposal well before deadline; plan for system delays',
            'content_restriction': 'Review volume allocation; ensure restricted content is excluded'
        }
        return actions.get(category, 'Review requirement carefully and ensure compliance')
    
    def _extract_due_dates(self, text: str):
        """Extract submission due dates"""
        # Common date patterns
        date_patterns = [
            r'due\s+(?:date\s+)?(?:is\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:must|shall)\s+be\s+(?:submitted|received)\s+(?:by\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'deadline\s+(?:is\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(1)
                self.metadata['due_date'] = date_text
                self.metadata['due_date_text'] = match.group(0)
                return  # Use first match
    
    def to_dict(self) -> Dict:
        """Serialize extraction results"""
        return {
            'volumes': [v.to_dict() for v in self.volumes],
            'formatting_rules': [r.to_dict() for r in self.formatting_rules],
            'compliance_flags': [f.to_dict() for f in self.compliance_flags],
            'metadata': self.metadata,
            'summary': {
                'total_volumes': len(self.volumes),
                'total_formatting_rules': len(self.formatting_rules),
                'total_compliance_flags': len(self.compliance_flags),
                'critical_flags': len([f for f in self.compliance_flags if f.severity == 'CRITICAL'])
            }
        }


def extract_rfp_letter(text: str, filename: str = "RFP Letter") -> Dict:
    """
    Convenience function for RFP Letter extraction.
    
    Args:
        text: Full text of RFP Letter
        filename: Source document name
    
    Returns:
        Dict with extracted submission instructions
    """
    extractor = RFPLetterExtractor()
    return extractor.extract_from_text(text, filename)
