"""
VisibleThread-Style Requirement Dictionary
Based on professional RFP compliance matrix standards

Maps requirement keywords to severity levels and search patterns
Following the VisibleThread Extended Dictionary approach
"""

from enum import Enum
from typing import List, Dict, Tuple
import re


class RequirementSeverity(Enum):
    """Requirement severity levels (color-coded)"""
    MANDATORY = "mandatory"      # Red - #ff0000 - shall/will/must
    STRONG = "strong"            # Orange - #ff7f00 - should/include/ensure
    POSSIBLE = "possible"        # Yellow - #ffa500 - may/might


class SearchType(Enum):
    """How to search for the term"""
    EXACT = "exact"              # Exact word match
    WILDCARD = "wildcard"        # Supports * wildcards
    REGEX = "regex"              # Full regex pattern


class RequirementKeyword:
    """A single requirement keyword with metadata"""
    
    def __init__(
        self,
        term: str,
        severity: RequirementSeverity,
        search_type: SearchType = SearchType.EXACT,
        case_sensitive: bool = False,
        description: str = "",
        category: str = "Requirements"
    ):
        self.term = term
        self.severity = severity
        self.search_type = search_type
        self.case_sensitive = case_sensitive
        self.description = description or "Needs to be checked in context"
        self.category = category
        
        # Build the regex pattern
        self.pattern = self._build_pattern()
    
    def _build_pattern(self) -> re.Pattern:
        """Build regex pattern from term and search type"""
        if self.search_type == SearchType.REGEX:
            pattern_str = self.term
        elif self.search_type == SearchType.WILDCARD:
            # Convert wildcard to regex: provid* -> provid\w*
            pattern_str = r'\b' + self.term.replace('*', r'\w*') + r'\b'
        else:  # EXACT
            pattern_str = r'\b' + re.escape(self.term) + r'\b'
        
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(pattern_str, flags)
    
    def search(self, text: str) -> List[Tuple[int, str]]:
        """
        Search for this keyword in text
        
        Returns:
            List of (position, matched_text) tuples
        """
        matches = []
        for match in self.pattern.finditer(text):
            matches.append((match.start(), match.group(0)))
        return matches


class RequirementDictionary:
    """
    Complete requirement dictionary for RFP analysis
    Based on VisibleThread Extended Dictionary
    """
    
    def __init__(self):
        self.keywords: List[RequirementKeyword] = []
        self._build_dictionary()
    
    def _build_dictionary(self):
        """Build the complete dictionary"""
        
        # ========================================
        # MANDATORY REQUIREMENTS (Red - #ff0000)
        # ========================================
        mandatory = [
            "will",
            "shall",
            "must",
        ]
        
        for term in mandatory:
            self.keywords.append(RequirementKeyword(
                term=term,
                severity=RequirementSeverity.MANDATORY,
                search_type=SearchType.EXACT,
                description="Mandatory requirement - must be addressed in proposal"
            ))
        
        # ========================================
        # STRONG REQUIREMENTS (Orange - #ff7f00)
        # ========================================
        
        # Exact matches
        strong_exact = [
            "should",
            "submit",
            "cannot",
            "indicate",
            "describe",
            "illustrate",
            "show",
            "document",
            "list",
            "apply",
            "applies",
            "compliant",
            "compulsion",
            "compulsory",
            "contractor",
            "duty",
            "prohibited",
            "provision",
            "unable",
            "offeror",
        ]
        
        for term in strong_exact:
            self.keywords.append(RequirementKeyword(
                term=term,
                severity=RequirementSeverity.STRONG,
                search_type=SearchType.EXACT,
                description="Strong requirement or compliance consideration"
            ))
        
        # Wildcard matches
        strong_wildcard = [
            "include*",       # includes, including
            "ensure*",        # ensures, ensuring
            "insure*",        # insures, insuring
            "assure*",        # assures, assuring
            "provid*",        # provide, provided, providing, provider
            "commit*",        # commit, commits, committed, commitment
            "compel*",        # compel, compels, compelling
            "consent*",       # consent, consents, consenting
            "enforce*",       # enforce, enforced, enforcement
            "fail*",          # fail, fails, failure
            "incorporate*",   # incorporate, incorporated, incorporating
            "necessitate*",   # necessitate, necessitates
            "oblige*",        # oblige, obliges, obligation, obligated
            "prohibit*",      # prohibit, prohibits, prohibited, prohibition
            "request*",       # request, requests, requested, requesting
            "responsib*",     # responsible, responsibility, responsibilities
            "require*",       # require, required, requirement, requirements
            "intend*",        # intend, intends, intended, intention
            "anticipate*",    # anticipate, anticipated, anticipation
            "assume*",        # assume, assumed, assumption
            "plan*",          # plan, plans, planned, planning
            "expect*",        # expect, expected, expectation
            "propose*",       # propose, proposed, proposal
        ]
        
        for term in strong_wildcard:
            self.keywords.append(RequirementKeyword(
                term=term,
                severity=RequirementSeverity.STRONG,
                search_type=SearchType.WILDCARD,
                description="Strong requirement or compliance consideration"
            ))
        
        # ========================================
        # POSSIBLE REQUIREMENTS (Yellow - #ffa500)
        # ========================================
        possible = [
            "may",
            "might",
            "can",
            "could",
        ]
        
        for term in possible:
            self.keywords.append(RequirementKeyword(
                term=term,
                severity=RequirementSeverity.POSSIBLE,
                search_type=SearchType.EXACT,
                description="Possible obligation or option"
            ))
        
        # ========================================
        # PROPOSAL SUBMISSION KEYWORDS
        # ========================================
        submission_keywords = [
            "Volume",
            "section",
            "page limit",
            "font size",
            "margin",
            "format",
            "electronic submission",
            "deadline",
            "due date",
        ]
        
        for term in submission_keywords:
            self.keywords.append(RequirementKeyword(
                term=term,
                severity=RequirementSeverity.STRONG,
                search_type=SearchType.EXACT,
                category="Submission Instructions",
                description="Proposal formatting or submission requirement"
            ))
    
    def find_requirements(self, text: str) -> Dict[str, List[Dict]]:
        """
        Find all requirement keywords in text
        
        Returns:
            Dict with severity as key, list of findings as value
            Each finding: {
                'term': keyword,
                'position': char position,
                'matched_text': actual matched text,
                'severity': severity level,
                'description': description
            }
        """
        findings = {
            'mandatory': [],
            'strong': [],
            'possible': []
        }
        
        for keyword in self.keywords:
            matches = keyword.search(text)
            for position, matched_text in matches:
                findings[keyword.severity.value].append({
                    'term': keyword.term,
                    'position': position,
                    'matched_text': matched_text,
                    'severity': keyword.severity.value,
                    'description': keyword.description,
                    'category': keyword.category
                })
        
        return findings
    
    def get_severity_stats(self, text: str) -> Dict[str, int]:
        """Get count of each severity level in text"""
        findings = self.find_requirements(text)
        return {
            'mandatory': len(findings['mandatory']),
            'strong': len(findings['strong']),
            'possible': len(findings['possible']),
            'total': sum(len(v) for v in findings.values())
        }


# Singleton instance
_dictionary = None


def get_requirement_dictionary() -> RequirementDictionary:
    """Get the singleton requirement dictionary"""
    global _dictionary
    if _dictionary is None:
        _dictionary = RequirementDictionary()
    return _dictionary
