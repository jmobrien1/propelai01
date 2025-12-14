"""
Enhanced Metadata Extractor for RFP Documents

Extracts critical RFP metadata using multiple strategies:
1. Regex patterns for common formats
2. Structured field detection (SF30, SF1449, etc.)
3. Context-aware extraction for complex cases

Extracted fields:
- Due date/time (with timezone handling)
- Contract type (FFP, CPFF, T&M, etc.)
- NAICS codes
- Set-aside status (small business, 8(a), etc.)
- PSC codes
- Estimated value/ceiling
- Place of performance
- Period of performance
- Point of contact information
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)


class ContractType(str, Enum):
    """Federal contract types"""
    FFP = "FFP"  # Firm Fixed Price
    FPIF = "FPIF"  # Fixed Price Incentive Firm
    FPAF = "FPAF"  # Fixed Price Award Fee
    FP_EPA = "FP-EPA"  # Fixed Price with Economic Price Adjustment
    CPFF = "CPFF"  # Cost Plus Fixed Fee
    CPIF = "CPIF"  # Cost Plus Incentive Fee
    CPAF = "CPAF"  # Cost Plus Award Fee
    CR = "CR"  # Cost Reimbursement
    TM = "T&M"  # Time and Materials
    LH = "LH"  # Labor Hour
    IDIQ = "IDIQ"  # Indefinite Delivery/Indefinite Quantity
    BPA = "BPA"  # Blanket Purchase Agreement
    BOA = "BOA"  # Basic Ordering Agreement
    HYBRID = "HYBRID"  # Multiple types
    UNKNOWN = "UNKNOWN"


class SetAsideType(str, Enum):
    """Small business set-aside categories"""
    NONE = "none"  # Full and open
    SB = "SB"  # Small Business
    SDB = "SDB"  # Small Disadvantaged Business
    EIGHT_A = "8(a)"  # 8(a) Program
    HUBZONE = "HUBZone"  # Historically Underutilized Business Zone
    SDVOSB = "SDVOSB"  # Service-Disabled Veteran-Owned
    VOSB = "VOSB"  # Veteran-Owned Small Business
    WOSB = "WOSB"  # Women-Owned Small Business
    EDWOSB = "EDWOSB"  # Economically Disadvantaged WOSB
    INDIAN = "Indian"  # Indian Economic Enterprise
    TRIBAL = "Tribal"  # Tribally-owned
    ALASKAN = "ANC"  # Alaska Native Corporation
    MULTIPLE = "Multiple"  # Multiple set-asides


@dataclass
class PointOfContact:
    """Contracting officer or point of contact"""
    name: str = ""
    title: str = ""
    email: str = ""
    phone: str = ""
    organization: str = ""


@dataclass
class DueDateInfo:
    """Detailed due date information"""
    date: Optional[date] = None
    time: Optional[str] = None  # "14:00" format
    timezone: str = "EST"  # Default to Eastern
    datetime_str: str = ""  # Original string
    is_approximate: bool = False  # "on or about"
    extended_by: Optional[str] = None  # Amendment reference if extended


@dataclass
class RFPMetadata:
    """
    Comprehensive RFP metadata extracted from solicitation documents.
    """
    # Core identifiers
    solicitation_number: str = ""
    title: str = ""
    agency: str = ""
    sub_agency: str = ""

    # Timeline
    issue_date: Optional[date] = None
    questions_due: Optional[DueDateInfo] = None
    proposals_due: Optional[DueDateInfo] = None
    site_visit_date: Optional[DueDateInfo] = None
    oral_presentation_dates: List[str] = field(default_factory=list)

    # Contract details
    contract_type: ContractType = ContractType.UNKNOWN
    contract_type_details: str = ""  # Additional context
    set_aside: SetAsideType = SetAsideType.NONE
    set_aside_details: str = ""

    # Classification codes
    naics_code: str = ""
    naics_description: str = ""
    size_standard: str = ""  # e.g., "$30M" or "500 employees"
    psc_code: str = ""  # Product Service Code
    psc_description: str = ""

    # Value and scope
    estimated_value: str = ""
    ceiling_value: str = ""
    base_period: str = ""
    option_periods: List[str] = field(default_factory=list)
    total_period: str = ""

    # Location
    place_of_performance: str = ""
    performance_locations: List[str] = field(default_factory=list)

    # Contacts
    contracting_officer: Optional[PointOfContact] = None
    contract_specialist: Optional[PointOfContact] = None
    technical_poc: Optional[PointOfContact] = None

    # Document info
    amendment_count: int = 0
    latest_amendment: str = ""
    incumbent: str = ""

    # Extraction confidence
    extraction_confidence: float = 0.0
    fields_extracted: List[str] = field(default_factory=list)
    fields_missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "solicitation_number": self.solicitation_number,
            "title": self.title,
            "agency": self.agency,
            "sub_agency": self.sub_agency,
            "issue_date": self.issue_date.isoformat() if self.issue_date else None,
            "proposals_due": {
                "date": self.proposals_due.date.isoformat() if self.proposals_due and self.proposals_due.date else None,
                "time": self.proposals_due.time if self.proposals_due else None,
                "timezone": self.proposals_due.timezone if self.proposals_due else "EST",
                "original": self.proposals_due.datetime_str if self.proposals_due else None,
            } if self.proposals_due else None,
            "contract_type": self.contract_type.value,
            "contract_type_details": self.contract_type_details,
            "set_aside": self.set_aside.value,
            "set_aside_details": self.set_aside_details,
            "naics_code": self.naics_code,
            "naics_description": self.naics_description,
            "size_standard": self.size_standard,
            "psc_code": self.psc_code,
            "estimated_value": self.estimated_value,
            "ceiling_value": self.ceiling_value,
            "base_period": self.base_period,
            "option_periods": self.option_periods,
            "place_of_performance": self.place_of_performance,
            "contracting_officer": {
                "name": self.contracting_officer.name,
                "email": self.contracting_officer.email,
                "phone": self.contracting_officer.phone,
            } if self.contracting_officer else None,
            "extraction_confidence": self.extraction_confidence,
            "fields_extracted": self.fields_extracted,
            "fields_missing": self.fields_missing,
        }


class RFPMetadataExtractor:
    """
    Extract comprehensive metadata from RFP documents.

    Uses layered extraction:
    1. Standard form field patterns (SF30, SF1449, SF33)
    2. Common RFP text patterns
    3. Contextual inference
    """

    # =========================================================================
    # SOLICITATION NUMBER PATTERNS
    # =========================================================================

    SOLICITATION_PATTERNS = [
        # Standard formats: ABC123-45-Q-6789
        r"(?:Solicitation|RFP|RFQ|RFI|IFB|BAA|OTA)[\s#:]+([A-Z0-9]{2,}-[\dA-Z]+-[A-Z]-[\dA-Z]+)",
        # NIH style: 75N96025R00004
        r"(?:Solicitation|RFP)[\s#:]+(\d{2}[A-Z]\d{5}[A-Z]\d{5})",
        # DoD style: W91278-25-R-0001
        r"([A-Z]\d{5}-\d{2}-[A-Z]-\d{4})",
        # GSA style: 47QFCA25R0001
        r"(\d{2}[A-Z]{4}\d{2}[A-Z]\d{4})",
        # Generic: number-number-letter-number
        r"(?:Solicitation|Contract|Award)[\s#:]+([A-Z0-9]+-\d+-[A-Z]-\d+)",
        # SF30 Block 1
        r"1\.\s*CONTRACT ID CODE[\s\S]*?(\d{2}[A-Z0-9]+[-/][A-Z0-9-]+)",
        # Simple patterns
        r"(?:Solicitation\s+Number|RFP\s+No\.?)[\s:]+([A-Z0-9][-A-Z0-9]+)",
    ]

    # =========================================================================
    # AGENCY PATTERNS
    # =========================================================================

    AGENCY_PATTERNS = [
        # Full names
        (r"(?:Department\s+of|Dept\.?\s+of)\s+([A-Z][A-Za-z\s]+?)(?:\s*\(|\s*,|\s+Office)", "department"),
        # Abbreviations
        (r"\b(DoD|DOD|HHS|NIH|NASA|DOE|DHS|DOJ|DOT|DOL|DOS|USDA|EPA|GSA|VA|SSA|SBA|USAID)\b", "abbreviation"),
        # Agency office patterns
        (r"(National Institutes? of Health|NIH)", "NIH"),
        (r"(Department of Defense|DoD)", "DoD"),
        (r"(General Services Administration|GSA)", "GSA"),
        (r"(National Aeronautics and Space Administration|NASA)", "NASA"),
        # SF30 Block 6 / SF1449 Block 5
        (r"(?:ISSUED BY|ADMINISTERED BY)[\s\S]{0,100}?([A-Z][A-Za-z\s,]+(?:Center|Office|Agency|Administration|Department))", "block"),
    ]

    # =========================================================================
    # DATE PATTERNS - Comprehensive
    # =========================================================================

    DATE_PATTERNS = [
        # Full format: January 15, 2025 at 2:00 PM EST
        r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\s*(?:at|by|@)?\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\s*((?:E|C|M|P)(?:S|D)T)?",
        # Format: 01/15/2025 14:00 EST
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s*(?:at|by|@)?\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\s*((?:E|C|M|P)(?:S|D)T)?",
        # Format: 2025-01-15T14:00:00
        r"(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}(?::\d{2})?))?",
        # Simple date: January 15, 2025
        r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})",
        # Simple date: 01/15/2025
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        # Written: fifteenth of January, 2025
        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})",
    ]

    DUE_DATE_CONTEXT = [
        r"(?:proposals?|offers?|quotes?|responses?)\s+(?:are\s+)?(?:due|must\s+be\s+(?:received|submitted)|shall\s+be\s+(?:received|submitted))",
        r"(?:closing|submission|response)\s+(?:date|deadline|time)",
        r"(?:submit|receive)\s+(?:proposals?|offers?|quotes?)\s+(?:by|no\s+later\s+than|NLT)",
        r"deadline\s+for\s+(?:submission|receipt|proposals?)",
    ]

    QUESTIONS_DUE_CONTEXT = [
        r"questions?\s+(?:are\s+)?(?:due|must\s+be\s+(?:submitted|received))",
        r"(?:submit|send)\s+questions?\s+(?:by|no\s+later\s+than)",
        r"deadline\s+for\s+questions?",
        r"inquiries?\s+(?:due|must\s+be\s+(?:submitted|received))",
    ]

    # =========================================================================
    # CONTRACT TYPE PATTERNS
    # =========================================================================

    CONTRACT_TYPE_PATTERNS = [
        (r"\bFirm[\s-]Fixed[\s-]Price\b", ContractType.FFP),
        (r"\bFFP\b", ContractType.FFP),
        (r"\bFixed[\s-]Price[\s-]Incentive[\s-]Firm\b", ContractType.FPIF),
        (r"\bFPIF\b", ContractType.FPIF),
        (r"\bFixed[\s-]Price[\s-]Award[\s-]Fee\b", ContractType.FPAF),
        (r"\bFPAF\b", ContractType.FPAF),
        (r"\bFixed[\s-]Price.*Economic[\s-]Price[\s-]Adjustment\b", ContractType.FP_EPA),
        (r"\bFP[\s-]EPA\b", ContractType.FP_EPA),
        (r"\bCost[\s-]Plus[\s-]Fixed[\s-]Fee\b", ContractType.CPFF),
        (r"\bCPFF\b", ContractType.CPFF),
        (r"\bCost[\s-]Plus[\s-]Incentive[\s-]Fee\b", ContractType.CPIF),
        (r"\bCPIF\b", ContractType.CPIF),
        (r"\bCost[\s-]Plus[\s-]Award[\s-]Fee\b", ContractType.CPAF),
        (r"\bCPAF\b", ContractType.CPAF),
        (r"\bCost[\s-]Reimbursement\b", ContractType.CR),
        (r"\bTime[\s-]and[\s-]Materials?\b", ContractType.TM),
        (r"\bT&M\b", ContractType.TM),
        (r"\bT\s*&\s*M\b", ContractType.TM),
        (r"\bLabor[\s-]Hour\b", ContractType.LH),
        (r"\bLH\b", ContractType.LH),
        (r"\b(?:IDIQ|ID/IQ)\b", ContractType.IDIQ),
        (r"\bIndefinite[\s-]Delivery[\s/-]Indefinite[\s-]Quantity\b", ContractType.IDIQ),
        (r"\bBPA\b", ContractType.BPA),
        (r"\bBlanket[\s-]Purchase[\s-]Agreement\b", ContractType.BPA),
        (r"\bBOA\b", ContractType.BOA),
        (r"\bBasic[\s-]Ordering[\s-]Agreement\b", ContractType.BOA),
    ]

    # =========================================================================
    # SET-ASIDE PATTERNS
    # =========================================================================

    SET_ASIDE_PATTERNS = [
        (r"\b8\s*\(\s*a\s*\)\b", SetAsideType.EIGHT_A),
        (r"\bHUBZone\b", SetAsideType.HUBZONE),
        (r"\bHistorically\s+Underutilized\s+Business\s+Zone\b", SetAsideType.HUBZONE),
        (r"\bSDVOSB\b", SetAsideType.SDVOSB),
        (r"\bService[\s-]Disabled\s+Veteran[\s-]Owned\b", SetAsideType.SDVOSB),
        (r"\bVOSB\b", SetAsideType.VOSB),
        (r"\bVeteran[\s-]Owned\s+Small\s+Business\b", SetAsideType.VOSB),
        (r"\bWOSB\b", SetAsideType.WOSB),
        (r"\bWomen[\s-]Owned\s+Small\s+Business\b", SetAsideType.WOSB),
        (r"\bEDWOSB\b", SetAsideType.EDWOSB),
        (r"\bEconomically\s+Disadvantaged\s+(?:Women[\s-]Owned|WOSB)\b", SetAsideType.EDWOSB),
        (r"\bSDB\b", SetAsideType.SDB),
        (r"\bSmall\s+Disadvantaged\s+Business\b", SetAsideType.SDB),
        (r"\bSmall\s+Business\s+Set[\s-]?Aside\b", SetAsideType.SB),
        (r"\bTotal\s+Small\s+Business\b", SetAsideType.SB),
        (r"\bFull\s+and\s+Open(?:\s+Competition)?\b", SetAsideType.NONE),
        (r"\bUnrestricted\b", SetAsideType.NONE),
    ]

    # =========================================================================
    # NAICS PATTERNS
    # =========================================================================

    NAICS_PATTERNS = [
        # Standard format with description
        r"NAICS(?:\s+Code)?[\s:]+(\d{6})\s*[-–—]\s*([^\n]+)",
        # Just the code
        r"NAICS(?:\s+Code)?[\s:]+(\d{6})",
        # In table format
        r"(\d{6})\s+(?:is\s+)?(?:the\s+)?(?:applicable\s+)?NAICS",
        # With size standard
        r"NAICS[\s:]+(\d{6})[\s\S]{0,50}?(?:Size\s+Standard|Threshold)[\s:]+(\$[\d,]+(?:\s*(?:M|Million))?|\d+\s+employees?)",
    ]

    PSC_PATTERNS = [
        r"(?:PSC|Product\s+Service\s+Code)[\s:]+([A-Z]\d{3})",
        r"([A-Z]\d{3})\s+[-–—]\s+(?:is\s+)?(?:the\s+)?(?:applicable\s+)?(?:PSC|Product\s+Service\s+Code)",
    ]

    # =========================================================================
    # VALUE PATTERNS
    # =========================================================================

    VALUE_PATTERNS = [
        r"(?:estimated|anticipated|approximate)\s+(?:contract\s+)?(?:value|amount|ceiling)[\s:]+(\$[\d,]+(?:\.\d{2})?(?:\s*(?:M|Million|B|Billion))?)",
        r"(?:ceiling|maximum)\s+(?:value|amount)[\s:]+(\$[\d,]+(?:\.\d{2})?(?:\s*(?:M|Million|B|Billion))?)",
        r"(?:contract\s+)?(?:value|amount)[\s:]+(?:not\s+to\s+exceed\s+)?(\$[\d,]+(?:\.\d{2})?(?:\s*(?:M|Million|B|Billion))?)",
        r"(\$[\d,]+(?:\.\d{2})?(?:\s*(?:M|Million|B|Billion))?)\s+(?:ceiling|maximum|estimated)",
    ]

    # =========================================================================
    # PERIOD OF PERFORMANCE PATTERNS
    # =========================================================================

    POP_PATTERNS = [
        r"(?:base\s+)?period\s+of\s+performance[\s:]+(\d+)\s+(?:year|month|day)s?",
        r"(\d+)[\s-]?year\s+base\s+period",
        r"base\s+period[\s:]+(\d+)\s+(?:year|month)s?",
        r"(\d+)\s+(?:one[\s-]?year|12[\s-]?month)\s+option\s+(?:year|period)s?",
    ]

    # =========================================================================
    # CONTACT PATTERNS
    # =========================================================================

    EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    PHONE_PATTERN = r"(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}"

    CONTACT_CONTEXT = [
        r"(?:Contracting\s+Officer|CO)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
        r"(?:Contract\s+Specialist)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
        r"(?:Point\s+of\s+Contact|POC)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
    ]

    def __init__(self):
        """Initialize the metadata extractor"""
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        self.compiled_solicitation = [
            re.compile(p, re.IGNORECASE) for p in self.SOLICITATION_PATTERNS
        ]
        self.compiled_agency = [
            (re.compile(p, re.IGNORECASE), name) for p, name in self.AGENCY_PATTERNS
        ]
        self.compiled_date = [
            re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS
        ]
        self.compiled_contract_type = [
            (re.compile(p, re.IGNORECASE), ct) for p, ct in self.CONTRACT_TYPE_PATTERNS
        ]
        self.compiled_set_aside = [
            (re.compile(p, re.IGNORECASE), sa) for p, sa in self.SET_ASIDE_PATTERNS
        ]

    def extract(self, text: str, filename: str = "") -> RFPMetadata:
        """
        Extract all metadata from RFP text.

        Args:
            text: Full text content of the RFP
            filename: Optional filename for context

        Returns:
            RFPMetadata object with extracted fields
        """
        metadata = RFPMetadata()
        fields_found = []

        # Extract each field type
        sol_num = self._extract_solicitation_number(text, filename)
        if sol_num:
            metadata.solicitation_number = sol_num
            fields_found.append("solicitation_number")

        agency, sub_agency = self._extract_agency(text)
        if agency:
            metadata.agency = agency
            metadata.sub_agency = sub_agency
            fields_found.append("agency")

        title = self._extract_title(text)
        if title:
            metadata.title = title
            fields_found.append("title")

        proposals_due = self._extract_due_date(text, "proposals")
        if proposals_due:
            metadata.proposals_due = proposals_due
            fields_found.append("proposals_due")

        questions_due = self._extract_due_date(text, "questions")
        if questions_due:
            metadata.questions_due = questions_due
            fields_found.append("questions_due")

        contract_type, ct_details = self._extract_contract_type(text)
        metadata.contract_type = contract_type
        metadata.contract_type_details = ct_details
        if contract_type != ContractType.UNKNOWN:
            fields_found.append("contract_type")

        set_aside, sa_details = self._extract_set_aside(text)
        metadata.set_aside = set_aside
        metadata.set_aside_details = sa_details
        if set_aside != SetAsideType.NONE:
            fields_found.append("set_aside")

        naics, naics_desc, size_std = self._extract_naics(text)
        if naics:
            metadata.naics_code = naics
            metadata.naics_description = naics_desc
            metadata.size_standard = size_std
            fields_found.append("naics_code")

        psc, psc_desc = self._extract_psc(text)
        if psc:
            metadata.psc_code = psc
            metadata.psc_description = psc_desc
            fields_found.append("psc_code")

        est_value, ceiling = self._extract_value(text)
        if est_value:
            metadata.estimated_value = est_value
            fields_found.append("estimated_value")
        if ceiling:
            metadata.ceiling_value = ceiling
            fields_found.append("ceiling_value")

        base_period, options = self._extract_pop(text)
        if base_period:
            metadata.base_period = base_period
            metadata.option_periods = options
            fields_found.append("period_of_performance")

        pop = self._extract_place_of_performance(text)
        if pop:
            metadata.place_of_performance = pop
            fields_found.append("place_of_performance")

        co = self._extract_contact(text, "contracting_officer")
        if co and co.name:
            metadata.contracting_officer = co
            fields_found.append("contracting_officer")

        # Calculate extraction confidence
        required_fields = [
            "solicitation_number", "title", "agency", "proposals_due",
            "contract_type", "naics_code"
        ]
        metadata.fields_extracted = fields_found
        metadata.fields_missing = [f for f in required_fields if f not in fields_found]
        metadata.extraction_confidence = len(fields_found) / max(len(required_fields), 1)

        return metadata

    def _extract_solicitation_number(self, text: str, filename: str = "") -> str:
        """Extract solicitation number"""
        # Try filename first (often contains sol number)
        if filename:
            for pattern in self.compiled_solicitation:
                match = pattern.search(filename)
                if match:
                    return match.group(1)

        # Search text
        for pattern in self.compiled_solicitation:
            match = pattern.search(text)
            if match:
                return match.group(1)

        return ""

    def _extract_agency(self, text: str) -> Tuple[str, str]:
        """Extract agency and sub-agency"""
        agency = ""
        sub_agency = ""

        for pattern, pattern_type in self.compiled_agency:
            match = pattern.search(text)
            if match:
                agency = match.group(1).strip()
                # Try to find sub-agency in context
                if pattern_type == "department":
                    sub_match = re.search(
                        rf"{re.escape(agency)}[\s,]+([A-Z][A-Za-z\s]+(?:Center|Office|Institute|Administration))",
                        text
                    )
                    if sub_match:
                        sub_agency = sub_match.group(1).strip()
                break

        return agency, sub_agency

    def _extract_title(self, text: str) -> str:
        """Extract solicitation title"""
        patterns = [
            r"(?:Title|Subject|Description)[\s:]+([^\n]+)",
            r"(?:Solicitation|RFP|Contract)\s+for[\s:]+([^\n]+)",
            r"(?:REQUEST FOR PROPOSAL|RFP)[\s\S]{0,50}?(?:for|:)\s+([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean up title
                title = re.sub(r'\s+', ' ', title)
                if len(title) > 10 and len(title) < 200:
                    return title

        return ""

    def _extract_due_date(self, text: str, date_type: str = "proposals") -> Optional[DueDateInfo]:
        """Extract due date with time and timezone"""
        # Build context pattern based on type
        if date_type == "proposals":
            context_patterns = self.DUE_DATE_CONTEXT
        elif date_type == "questions":
            context_patterns = self.QUESTIONS_DUE_CONTEXT
        else:
            context_patterns = self.DUE_DATE_CONTEXT

        # Find context and nearby dates
        for ctx_pattern in context_patterns:
            ctx_match = re.search(ctx_pattern, text, re.IGNORECASE)
            if ctx_match:
                # Search for date near this context (within 500 chars)
                search_area = text[max(0, ctx_match.start()-100):ctx_match.end()+500]

                for date_pattern in self.compiled_date:
                    date_match = date_pattern.search(search_area)
                    if date_match:
                        return self._parse_date_match(date_match, search_area)

        return None

    def _parse_date_match(self, match: re.Match, context: str) -> DueDateInfo:
        """Parse a date match into DueDateInfo"""
        info = DueDateInfo()
        groups = match.groups()

        # Parse date component
        date_str = groups[0] if groups else ""
        info.datetime_str = date_str

        try:
            # Try various date formats
            for fmt in ["%B %d, %Y", "%B %d %Y", "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"]:
                try:
                    info.date = datetime.strptime(date_str.strip(","), fmt).date()
                    break
                except ValueError:
                    continue
        except Exception:
            pass

        # Parse time if present
        if len(groups) > 1 and groups[1]:
            info.time = groups[1].strip()

        # Parse timezone if present
        if len(groups) > 2 and groups[2]:
            info.timezone = groups[2].strip()

        # Check for approximate language
        if re.search(r"on\s+or\s+about|approximately|around", context, re.IGNORECASE):
            info.is_approximate = True

        return info

    def _extract_contract_type(self, text: str) -> Tuple[ContractType, str]:
        """Extract contract type"""
        found_types = []

        for pattern, ct in self.compiled_contract_type:
            if pattern.search(text):
                found_types.append(ct)

        if not found_types:
            return ContractType.UNKNOWN, ""

        if len(found_types) > 1:
            # Multiple types found - likely hybrid
            if ContractType.IDIQ in found_types:
                # IDIQ is often paired with another type
                other = [ct for ct in found_types if ct != ContractType.IDIQ]
                if other:
                    details = f"IDIQ with {other[0].value} task orders"
                    return ContractType.IDIQ, details
            return ContractType.HYBRID, f"Multiple types: {', '.join(ct.value for ct in found_types)}"

        return found_types[0], ""

    def _extract_set_aside(self, text: str) -> Tuple[SetAsideType, str]:
        """Extract set-aside status"""
        found = []

        for pattern, sa in self.compiled_set_aside:
            if pattern.search(text):
                found.append(sa)

        if not found:
            return SetAsideType.NONE, ""

        if len(found) > 1:
            # Remove duplicates and NONE
            unique = list(set(sa for sa in found if sa != SetAsideType.NONE))
            if len(unique) > 1:
                return SetAsideType.MULTIPLE, f"Multiple set-asides: {', '.join(sa.value for sa in unique)}"
            elif unique:
                return unique[0], ""

        return found[0], ""

    def _extract_naics(self, text: str) -> Tuple[str, str, str]:
        """Extract NAICS code, description, and size standard"""
        for pattern in self.NAICS_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                naics = groups[0] if groups else ""
                description = groups[1].strip() if len(groups) > 1 and groups[1] else ""
                size_std = groups[2].strip() if len(groups) > 2 and groups[2] else ""

                # Try to find size standard separately if not in match
                if not size_std and naics:
                    size_match = re.search(
                        rf"{naics}[\s\S]{{0,100}}?(?:size\s+standard|threshold)[\s:]+(\$[\d,]+(?:\s*(?:M|Million))?|\d+\s+employees?)",
                        text, re.IGNORECASE
                    )
                    if size_match:
                        size_std = size_match.group(1)

                return naics, description, size_std

        return "", "", ""

    def _extract_psc(self, text: str) -> Tuple[str, str]:
        """Extract PSC code"""
        for pattern in self.PSC_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1), ""

        return "", ""

    def _extract_value(self, text: str) -> Tuple[str, str]:
        """Extract estimated and ceiling values"""
        estimated = ""
        ceiling = ""

        for pattern in self.VALUE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                context = text[max(0, match.start()-50):match.end()+50].lower()

                if "ceiling" in context or "maximum" in context:
                    ceiling = value
                else:
                    estimated = value

        return estimated, ceiling

    def _extract_pop(self, text: str) -> Tuple[str, List[str]]:
        """Extract period of performance"""
        base = ""
        options = []

        # Look for base period
        base_match = re.search(
            r"(?:base\s+)?period\s+of\s+performance[\s:]+(\d+)\s+(year|month|day)s?",
            text, re.IGNORECASE
        )
        if base_match:
            base = f"{base_match.group(1)} {base_match.group(2)}s"

        # Look for option periods
        option_matches = re.findall(
            r"(\d+)\s+(?:option\s+)?(?:year|period)s?\s+(?:option|optional)",
            text, re.IGNORECASE
        )
        if option_matches:
            options = [f"{m} year option" for m in option_matches]

        return base, options

    def _extract_place_of_performance(self, text: str) -> str:
        """Extract place of performance"""
        patterns = [
            r"(?:place|location)\s+of\s+performance[\s:]+([^\n]+)",
            r"(?:work|services?)\s+(?:will|shall)\s+be\s+performed\s+(?:at|in)[\s:]+([^\n]+)",
            r"performance\s+location[\s:]+([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up
                location = re.sub(r'\s+', ' ', location)
                if len(location) > 5:
                    return location[:200]  # Limit length

        return ""

    def _extract_contact(self, text: str, contact_type: str) -> PointOfContact:
        """Extract point of contact information"""
        poc = PointOfContact()

        # Find contact name
        if contact_type == "contracting_officer":
            name_match = re.search(
                r"(?:Contracting\s+Officer|CO)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
                text
            )
        else:
            name_match = re.search(
                r"(?:Contract\s+Specialist|Point\s+of\s+Contact|POC)[\s:]+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
                text
            )

        if name_match:
            poc.name = name_match.group(1)

            # Search for email and phone near the name
            search_area = text[name_match.start():min(name_match.end()+500, len(text))]

            email_match = re.search(self.EMAIL_PATTERN, search_area)
            if email_match:
                poc.email = email_match.group(0)

            phone_match = re.search(self.PHONE_PATTERN, search_area)
            if phone_match:
                poc.phone = phone_match.group(0)

        return poc


def extract_rfp_metadata(text: str, filename: str = "") -> RFPMetadata:
    """
    Convenience function to extract metadata from RFP text.

    Args:
        text: Full text content of the RFP
        filename: Optional filename for context

    Returns:
        RFPMetadata object with extracted fields
    """
    extractor = RFPMetadataExtractor()
    return extractor.extract(text, filename)
