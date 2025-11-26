"""
PropelAI Cycle 5: Amendment Processor
Track requirement lifecycle across RFP amendments

Parses amendment documents to:
1. Extract Q&A pairs (clarifications)
2. Identify explicit modifications
3. Match changes to base requirements
4. Track lifecycle: ADDED | MODIFIED | DELETED | UNCHANGED
5. Flag conflicts and superseded requirements

Supports:
- SF30 amendment format (standard federal)
- NIH amendment format
- DoD amendment format
- Q&A response documents
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

from .models import (
    RequirementNode, RequirementType, ConfidenceLevel,
    DocumentType, SourceLocation, ParsedDocument
)
from .parser import MultiFormatParser
from .extractor import RequirementExtractor


class ChangeType(Enum):
    """Type of change in amendment"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    CLARIFIED = "clarified"
    UNCHANGED = "unchanged"


class AmendmentType(Enum):
    """Type of amendment document"""
    SF30 = "sf30"                    # Standard SF30 modification
    QA_RESPONSE = "qa_response"      # Q&A / RFI responses
    MODIFICATION = "modification"    # Direct modification notice
    COMBINED = "combined"            # SF30 with Q&A
    UNKNOWN = "unknown"


@dataclass
class QAPair:
    """Question and Answer pair from amendment"""
    question_number: str
    question_text: str
    answer_text: str
    source_page: int
    affects_requirements: List[str] = field(default_factory=list)
    change_type: ChangeType = ChangeType.CLARIFIED
    
    @property
    def id(self) -> str:
        return f"QA-{self.question_number}"


@dataclass
class Modification:
    """Explicit modification to the RFP"""
    mod_number: str
    section_affected: str
    original_text: Optional[str]
    new_text: str
    change_type: ChangeType
    source_page: int
    effective_date: Optional[str] = None
    affected_requirement_ids: List[str] = field(default_factory=list)
    
    @property
    def id(self) -> str:
        return f"MOD-{self.mod_number}-{self.section_affected}"


@dataclass
class RequirementChange:
    """Tracked change to a requirement"""
    requirement_id: str
    original_text: str
    new_text: Optional[str]
    change_type: ChangeType
    change_source: str  # QA-xxx or MOD-xxx
    amendment_number: int
    amendment_date: Optional[str]
    notes: str = ""
    
    @property
    def is_substantive(self) -> bool:
        """Check if change is substantive (not just clarification)"""
        return self.change_type in [ChangeType.ADDED, ChangeType.MODIFIED, ChangeType.DELETED]


@dataclass
class AmendmentResult:
    """Result of processing an amendment"""
    amendment_number: int
    amendment_date: Optional[str]
    amendment_type: AmendmentType
    
    # Extracted content
    qa_pairs: List[QAPair]
    modifications: List[Modification]
    
    # Tracked changes
    requirement_changes: List[RequirementChange]
    
    # Statistics
    total_questions: int = 0
    total_modifications: int = 0
    requirements_added: int = 0
    requirements_modified: int = 0
    requirements_deleted: int = 0
    requirements_clarified: int = 0
    
    # Conflicts detected
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.total_questions = len(self.qa_pairs)
        self.total_modifications = len(self.modifications)
        
        for change in self.requirement_changes:
            if change.change_type == ChangeType.ADDED:
                self.requirements_added += 1
            elif change.change_type == ChangeType.MODIFIED:
                self.requirements_modified += 1
            elif change.change_type == ChangeType.DELETED:
                self.requirements_deleted += 1
            elif change.change_type == ChangeType.CLARIFIED:
                self.requirements_clarified += 1


class AmendmentProcessor:
    """
    Process RFP amendments and track requirement changes
    
    Usage:
        processor = AmendmentProcessor()
        
        # Load base requirements
        processor.load_base_requirements(requirements_graph)
        
        # Process amendment
        result = processor.process_amendment("/path/to/amendment.pdf", amendment_number=2)
        
        # Get updated requirements
        updated_reqs = processor.get_updated_requirements()
    """
    
    # Patterns for detecting amendment structure
    AMENDMENT_NUMBER_PATTERNS = [
        r"AMENDMENT\s+(?:NO\.?\s*)?(\d+)",
        r"MODIFICATION\s+(?:NO\.?\s*)?(\d+)",
        r"AMENDMENT/MODIFICATION\s+NO\.?\s*(\d+)",
        r"SF\s*30.*AMENDMENT\s+(\d+)",
    ]
    
    # Q&A section patterns
    QA_SECTION_PATTERNS = [
        r"QUESTIONS?\s+AND\s+ANSWERS?",
        r"Q\s*&\s*A",
        r"RFI\s+RESPONSES?",
        r"CLARIFICATIONS?",
        r"OFFEROR\s+QUESTIONS?",
    ]
    
    # Question patterns
    QUESTION_PATTERNS = [
        r"(?:Q|QUESTION)[\s.:]*(\d+)[.\s:]+(.+?)(?=(?:A|ANSWER)[\s.:]*\d*[.\s:])",
        r"(\d+)\.\s*(?:Q|QUESTION)[.\s:]*(.+?)(?=(?:A|ANSWER)[.\s:])",
        r"Q(\d+)[.\s:]+(.+?)(?=A\1[.\s:])",
    ]
    
    # Answer patterns
    ANSWER_PATTERNS = [
        r"(?:A|ANSWER)[\s.:]*(\d+)?[.\s:]+(.+?)(?=(?:Q|QUESTION)[\s.:]*\d+|$)",
        r"A(\d+)[.\s:]+(.+?)(?=Q\d+|$)",
    ]
    
    # Modification patterns
    MOD_PATTERNS = [
        r"(?:Section|Page|Paragraph)\s+([A-Z][\d.]*)\s+is\s+(?:hereby\s+)?(?:amended|modified|revised|changed)\s+(?:to\s+read|as\s+follows)",
        r"DELETE[D]?\s+(?:the\s+following|from\s+Section)\s+([A-Z][\d.]*)",
        r"ADD[ED]?\s+(?:the\s+following|to\s+Section)\s+([A-Z][\d.]*)",
        r"REPLACE[D]?\s+(?:the\s+following\s+in|in\s+Section)\s+([A-Z][\d.]*)",
        r"The\s+following\s+(?:is|are)\s+(?:added|deleted|modified)\s+(?:to|from|in)\s+Section\s+([A-Z][\d.]*)",
    ]
    
    # Section reference patterns  
    SECTION_REF_PATTERN = r"Section\s+([A-Z])(?:\.(\d+))?(?:\.(\d+))?"
    
    def __init__(self):
        self.parser = MultiFormatParser()
        self.extractor = RequirementExtractor(strict_mode=True)
        
        # Base requirements (keyed by ID)
        self.base_requirements: Dict[str, RequirementNode] = {}
        
        # Index for matching (text hash -> requirement ID)
        self.text_index: Dict[str, str] = {}
        
        # Keyword index for fuzzy matching
        self.keyword_index: Dict[str, Set[str]] = {}
        
        # Amendment history
        self.amendments_processed: List[AmendmentResult] = []
        
        # Current state of all requirements
        self.current_requirements: Dict[str, RequirementNode] = {}
        
        # Change history per requirement
        self.change_history: Dict[str, List[RequirementChange]] = {}
    
    def load_base_requirements(self, requirements: Dict[str, RequirementNode]):
        """
        Load base requirements from initial extraction
        
        Args:
            requirements: Dict mapping requirement ID to RequirementNode
        """
        self.base_requirements = requirements.copy()
        self.current_requirements = requirements.copy()
        
        # Build indices for matching
        self._build_indices()
    
    def _build_indices(self):
        """Build text and keyword indices for requirement matching"""
        self.text_index.clear()
        self.keyword_index.clear()
        
        for req_id, req in self.current_requirements.items():
            # Text hash index
            text_hash = self._hash_text(req.text)
            self.text_index[text_hash] = req_id
            
            # Keyword index
            keywords = self._extract_keywords(req.text)
            for keyword in keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = set()
                self.keyword_index[keyword].add(req_id)
    
    def _hash_text(self, text: str) -> str:
        """Create hash of normalized text"""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract significant keywords from text"""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'shall', 'will', 'should', 'would', 'could', 'may', 'might',
            'must', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'this', 'that', 'these', 'those', 'it', 'its',
            'contractor', 'government', 'offeror', 'contract', 'proposal'
        }
        
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        return {w for w in words if w not in stop_words}
    
    def process_amendment(
        self, 
        file_path: str, 
        amendment_number: Optional[int] = None,
        amendment_date: Optional[str] = None
    ) -> AmendmentResult:
        """
        Process an amendment document
        
        Args:
            file_path: Path to amendment PDF/DOCX
            amendment_number: Override amendment number detection
            amendment_date: Override amendment date detection
            
        Returns:
            AmendmentResult with extracted changes
        """
        # Parse document
        parsed = self.parser.parse_file(file_path, DocumentType.AMENDMENT)
        
        # Detect amendment type and number
        amend_type = self._detect_amendment_type(parsed)
        if amendment_number is None:
            amendment_number = self._extract_amendment_number(parsed)
        
        # Extract Q&A pairs
        qa_pairs = self._extract_qa_pairs(parsed)
        
        # Extract modifications
        modifications = self._extract_modifications(parsed)
        
        # Match changes to existing requirements
        requirement_changes = self._match_changes_to_requirements(
            qa_pairs, modifications, amendment_number, amendment_date
        )
        
        # Detect conflicts
        conflicts = self._detect_conflicts(requirement_changes)
        
        # Create result
        result = AmendmentResult(
            amendment_number=amendment_number or len(self.amendments_processed) + 1,
            amendment_date=amendment_date,
            amendment_type=amend_type,
            qa_pairs=qa_pairs,
            modifications=modifications,
            requirement_changes=requirement_changes,
            conflicts=conflicts
        )
        
        # Apply changes to current requirements
        self._apply_changes(result)
        
        # Store in history
        self.amendments_processed.append(result)
        
        return result
    
    def _detect_amendment_type(self, doc: ParsedDocument) -> AmendmentType:
        """Detect the type of amendment document"""
        text_lower = doc.full_text.lower()[:5000]
        
        # Check for SF30
        if 'sf 30' in text_lower or 'sf30' in text_lower or 'standard form 30' in text_lower:
            # Check if it also has Q&A
            for pattern in self.QA_SECTION_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return AmendmentType.COMBINED
            return AmendmentType.SF30
        
        # Check for Q&A document
        for pattern in self.QA_SECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return AmendmentType.QA_RESPONSE
        
        # Check for modification notice
        for pattern in self.MOD_PATTERNS:
            if re.search(pattern, doc.full_text, re.IGNORECASE):
                return AmendmentType.MODIFICATION
        
        return AmendmentType.UNKNOWN
    
    def _extract_amendment_number(self, doc: ParsedDocument) -> Optional[int]:
        """Extract amendment number from document"""
        for pattern in self.AMENDMENT_NUMBER_PATTERNS:
            match = re.search(pattern, doc.full_text[:3000], re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    def _extract_qa_pairs(self, doc: ParsedDocument) -> List[QAPair]:
        """Extract Q&A pairs from document"""
        qa_pairs = []
        text = doc.full_text
        
        # Try to find Q&A section
        qa_section_start = None
        for pattern in self.QA_SECTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                qa_section_start = match.start()
                break
        
        if qa_section_start:
            text = text[qa_section_start:]
        
        # Pattern 1: Q1: ... A1: ... format
        pattern1 = r'Q[uestion]*[\s.:]*(\d+)[\s.:]*(.+?)A[nswer]*[\s.:]*\1?[\s.:]*(.+?)(?=Q[uestion]*[\s.:]*\d+|$)'
        matches = re.findall(pattern1, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            q_num, q_text, a_text = match
            q_text = self._clean_text(q_text)
            a_text = self._clean_text(a_text)
            
            if len(q_text) > 20 and len(a_text) > 10:
                qa_pairs.append(QAPair(
                    question_number=q_num,
                    question_text=q_text,
                    answer_text=a_text,
                    source_page=self._find_page(q_text, doc)
                ))
        
        # Pattern 2: Numbered list format
        if not qa_pairs:
            pattern2 = r'(\d+)\.\s*(?:Question|Q)[:\s]*(.+?)(?:Answer|A)[:\s]*(.+?)(?=\d+\.\s*(?:Question|Q)|$)'
            matches = re.findall(pattern2, text, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                q_num, q_text, a_text = match
                q_text = self._clean_text(q_text)
                a_text = self._clean_text(a_text)
                
                if len(q_text) > 20 and len(a_text) > 10:
                    qa_pairs.append(QAPair(
                        question_number=q_num,
                        question_text=q_text,
                        answer_text=a_text,
                        source_page=self._find_page(q_text, doc)
                    ))
        
        # Pattern 3: Table format (Q | A columns) - simplified
        if not qa_pairs:
            # Look for patterns like "Question Answer" rows
            rows = re.split(r'\n(?=\d+[\.\)]\s)', text)
            q_num = 0
            for row in rows:
                q_num += 1
                parts = re.split(r'\t{2,}|\s{4,}', row, maxsplit=1)
                if len(parts) == 2:
                    q_text = self._clean_text(parts[0])
                    a_text = self._clean_text(parts[1])
                    
                    if len(q_text) > 20 and len(a_text) > 10:
                        qa_pairs.append(QAPair(
                            question_number=str(q_num),
                            question_text=q_text,
                            answer_text=a_text,
                            source_page=self._find_page(q_text, doc)
                        ))
        
        return qa_pairs
    
    def _extract_modifications(self, doc: ParsedDocument) -> List[Modification]:
        """Extract explicit modifications from document"""
        modifications = []
        text = doc.full_text
        mod_counter = 0
        
        for pattern in self.MOD_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                mod_counter += 1
                section = match.group(1)
                
                # Try to extract the new text that follows
                start = match.end()
                end = min(start + 2000, len(text))
                following_text = text[start:end]
                
                # Find the end of this modification (next section or mod marker)
                end_match = re.search(r'\n(?:Section|The following|DELETE|ADD|REPLACE)', following_text)
                if end_match:
                    following_text = following_text[:end_match.start()]
                
                new_text = self._clean_text(following_text)
                
                # Determine change type
                match_text = match.group(0).upper()
                if 'DELETE' in match_text:
                    change_type = ChangeType.DELETED
                elif 'ADD' in match_text:
                    change_type = ChangeType.ADDED
                else:
                    change_type = ChangeType.MODIFIED
                
                modifications.append(Modification(
                    mod_number=str(mod_counter),
                    section_affected=section,
                    original_text=None,  # Would need base doc comparison
                    new_text=new_text[:1000],  # Truncate
                    change_type=change_type,
                    source_page=self._find_page(match.group(0), doc)
                ))
        
        return modifications
    
    def _match_changes_to_requirements(
        self,
        qa_pairs: List[QAPair],
        modifications: List[Modification],
        amendment_number: int,
        amendment_date: Optional[str]
    ) -> List[RequirementChange]:
        """Match extracted changes to existing requirements"""
        changes = []
        
        # Process Q&A pairs
        for qa in qa_pairs:
            # Find requirements mentioned in question or answer
            matched_reqs = self._find_matching_requirements(qa.question_text + " " + qa.answer_text)
            
            for req_id in matched_reqs:
                req = self.current_requirements.get(req_id)
                if req:
                    # Determine if this is a substantive change or just clarification
                    change_type = self._classify_qa_change(qa, req)
                    
                    changes.append(RequirementChange(
                        requirement_id=req_id,
                        original_text=req.text,
                        new_text=None if change_type == ChangeType.CLARIFIED else qa.answer_text,
                        change_type=change_type,
                        change_source=qa.id,
                        amendment_number=amendment_number,
                        amendment_date=amendment_date,
                        notes=f"Q: {qa.question_text[:100]}... A: {qa.answer_text[:100]}..."
                    ))
                    
                    qa.affects_requirements.append(req_id)
                    qa.change_type = change_type
        
        # Process modifications
        for mod in modifications:
            # Find requirements in affected section
            section_reqs = self._find_requirements_in_section(mod.section_affected)
            
            # Also do text matching
            text_matches = self._find_matching_requirements(mod.new_text)
            matched_reqs = set(section_reqs) | set(text_matches)
            
            for req_id in matched_reqs:
                req = self.current_requirements.get(req_id)
                if req:
                    changes.append(RequirementChange(
                        requirement_id=req_id,
                        original_text=req.text,
                        new_text=mod.new_text if mod.change_type != ChangeType.DELETED else None,
                        change_type=mod.change_type,
                        change_source=mod.id,
                        amendment_number=amendment_number,
                        amendment_date=amendment_date,
                        notes=f"Section {mod.section_affected}: {mod.change_type.value}"
                    ))
                    
                    mod.affected_requirement_ids.append(req_id)
        
        # Check for new requirements in modifications (ADDED)
        for mod in modifications:
            if mod.change_type == ChangeType.ADDED and mod.new_text:
                # Extract requirements from new text
                new_reqs = self._extract_requirements_from_text(mod.new_text, mod.section_affected)
                
                for new_req in new_reqs:
                    changes.append(RequirementChange(
                        requirement_id=new_req.id,
                        original_text="",
                        new_text=new_req.text,
                        change_type=ChangeType.ADDED,
                        change_source=mod.id,
                        amendment_number=amendment_number,
                        amendment_date=amendment_date,
                        notes=f"New requirement added in Amendment {amendment_number}"
                    ))
        
        return changes
    
    def _find_matching_requirements(self, text: str) -> List[str]:
        """Find requirements that match the given text"""
        matched = []
        
        # Extract keywords from text
        keywords = self._extract_keywords(text)
        
        # Find requirements with overlapping keywords
        candidate_scores: Dict[str, int] = {}
        for keyword in keywords:
            if keyword in self.keyword_index:
                for req_id in self.keyword_index[keyword]:
                    candidate_scores[req_id] = candidate_scores.get(req_id, 0) + 1
        
        # Require at least 3 keyword matches
        for req_id, score in candidate_scores.items():
            if score >= 3:
                matched.append(req_id)
        
        # Also check for section references
        section_refs = re.findall(self.SECTION_REF_PATTERN, text, re.IGNORECASE)
        for section_match in section_refs:
            section = section_match[0]
            section_reqs = self._find_requirements_in_section(section)
            matched.extend(section_reqs)
        
        return list(set(matched))
    
    def _find_requirements_in_section(self, section: str) -> List[str]:
        """Find requirements in a given section"""
        matched = []
        section_upper = section.upper()
        
        for req_id, req in self.current_requirements.items():
            if req.source and req.source.section_id:
                if req.source.section_id.upper().startswith(section_upper):
                    matched.append(req_id)
        
        return matched
    
    def _classify_qa_change(self, qa: QAPair, req: RequirementNode) -> ChangeType:
        """Classify whether a Q&A represents a substantive change"""
        answer_lower = qa.answer_text.lower()
        
        # Check for modification indicators
        mod_indicators = [
            'is changed to', 'is revised to', 'is modified to',
            'will be changed', 'has been changed', 'is amended',
            'replace', 'delete', 'add the following', 'remove'
        ]
        
        for indicator in mod_indicators:
            if indicator in answer_lower:
                return ChangeType.MODIFIED
        
        # Check for deletion indicators
        delete_indicators = ['is deleted', 'is removed', 'no longer applies', 'is withdrawn']
        for indicator in delete_indicators:
            if indicator in answer_lower:
                return ChangeType.DELETED
        
        # Check for addition indicators
        add_indicators = ['is added', 'new requirement', 'additional requirement']
        for indicator in add_indicators:
            if indicator in answer_lower:
                return ChangeType.ADDED
        
        # Default to clarification
        return ChangeType.CLARIFIED
    
    def _extract_requirements_from_text(self, text: str, section: str) -> List[RequirementNode]:
        """Extract new requirements from modification text"""
        # Create a temporary parsed document
        temp_doc = ParsedDocument(
            filename="amendment_extract",
            document_type=DocumentType.AMENDMENT,
            full_text=text,
            pages={1: text},
            sections={f"section_{section.lower()}": text}
        )
        
        # Extract using existing extractor
        new_reqs = self.extractor.extract_from_document(temp_doc)
        
        # Update IDs to indicate amendment source
        for req in new_reqs:
            req.id = f"REQ-AMD-{req.id.split('-')[-1]}"
        
        return new_reqs
    
    def _detect_conflicts(self, changes: List[RequirementChange]) -> List[Dict[str, Any]]:
        """Detect conflicts between changes"""
        conflicts = []
        
        # Group changes by requirement
        by_req: Dict[str, List[RequirementChange]] = {}
        for change in changes:
            if change.requirement_id not in by_req:
                by_req[change.requirement_id] = []
            by_req[change.requirement_id].append(change)
        
        # Check for conflicting changes to same requirement
        for req_id, req_changes in by_req.items():
            if len(req_changes) > 1:
                # Multiple changes to same requirement
                change_types = set(c.change_type for c in req_changes)
                
                if ChangeType.DELETED in change_types and ChangeType.MODIFIED in change_types:
                    conflicts.append({
                        "requirement_id": req_id,
                        "conflict_type": "delete_vs_modify",
                        "description": f"Requirement {req_id} is both deleted and modified",
                        "changes": [c.change_source for c in req_changes]
                    })
                
                if len([c for c in req_changes if c.change_type == ChangeType.MODIFIED]) > 1:
                    conflicts.append({
                        "requirement_id": req_id,
                        "conflict_type": "multiple_modifications",
                        "description": f"Requirement {req_id} has multiple modifications",
                        "changes": [c.change_source for c in req_changes if c.change_type == ChangeType.MODIFIED]
                    })
        
        return conflicts
    
    def _apply_changes(self, result: AmendmentResult):
        """Apply changes to current requirements state"""
        for change in result.requirement_changes:
            req_id = change.requirement_id
            
            # Track history
            if req_id not in self.change_history:
                self.change_history[req_id] = []
            self.change_history[req_id].append(change)
            
            # Apply change
            if change.change_type == ChangeType.DELETED:
                if req_id in self.current_requirements:
                    # Mark as deleted but keep for reference
                    self.current_requirements[req_id].status = "DELETED"
                    
            elif change.change_type == ChangeType.MODIFIED and change.new_text:
                if req_id in self.current_requirements:
                    # Update text
                    self.current_requirements[req_id].text = change.new_text
                    self.current_requirements[req_id].confidence = ConfidenceLevel.HIGH
                    
            elif change.change_type == ChangeType.ADDED and change.new_text:
                # Add new requirement
                new_req = RequirementNode(
                    id=req_id,
                    text=change.new_text,
                    requirement_type=RequirementType.PERFORMANCE,
                    confidence=ConfidenceLevel.HIGH,
                    source=SourceLocation(
                        document_name=f"Amendment {result.amendment_number}",
                        document_type=DocumentType.AMENDMENT,
                        page_number=0,
                        section_id=""
                    ),
                    extraction_method="amendment"
                )
                self.current_requirements[req_id] = new_req
        
        # Rebuild indices
        self._build_indices()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _find_page(self, text: str, doc: ParsedDocument) -> int:
        """Find which page contains the text"""
        text_snippet = text[:100]
        
        # Handle both dict and list pages
        if isinstance(doc.pages, dict):
            for page_num, page_text in doc.pages.items():
                if text_snippet in page_text:
                    return page_num
        elif isinstance(doc.pages, list):
            for page_num, page_text in enumerate(doc.pages, 1):
                if text_snippet in str(page_text):
                    return page_num
        
        return 1
    
    def get_updated_requirements(self) -> Dict[str, RequirementNode]:
        """Get current state of all requirements after amendments"""
        return self.current_requirements.copy()
    
    def get_change_history(self, requirement_id: str) -> List[RequirementChange]:
        """Get change history for a specific requirement"""
        return self.change_history.get(requirement_id, [])
    
    def get_all_changes(self) -> List[RequirementChange]:
        """Get all changes across all amendments"""
        all_changes = []
        for changes in self.change_history.values():
            all_changes.extend(changes)
        return sorted(all_changes, key=lambda c: c.amendment_number)
    
    def generate_change_report(self) -> str:
        """Generate a markdown report of all changes"""
        report = ["# Amendment Change Report\n"]
        
        for result in self.amendments_processed:
            report.append(f"\n## Amendment {result.amendment_number}")
            if result.amendment_date:
                report.append(f"\n**Date:** {result.amendment_date}")
            report.append(f"\n**Type:** {result.amendment_type.value}")
            
            report.append(f"\n\n### Summary")
            report.append(f"- Questions/Answers: {result.total_questions}")
            report.append(f"- Modifications: {result.total_modifications}")
            report.append(f"- Requirements Added: {result.requirements_added}")
            report.append(f"- Requirements Modified: {result.requirements_modified}")
            report.append(f"- Requirements Deleted: {result.requirements_deleted}")
            report.append(f"- Requirements Clarified: {result.requirements_clarified}")
            
            if result.conflicts:
                report.append(f"\n\n### ⚠️ Conflicts Detected")
                for conflict in result.conflicts:
                    report.append(f"- **{conflict['requirement_id']}**: {conflict['description']}")
            
            if result.requirement_changes:
                report.append(f"\n\n### Requirement Changes")
                for change in result.requirement_changes:
                    emoji = {
                        ChangeType.ADDED: "➕",
                        ChangeType.MODIFIED: "✏️",
                        ChangeType.DELETED: "❌",
                        ChangeType.CLARIFIED: "💬",
                        ChangeType.UNCHANGED: "⚪"
                    }.get(change.change_type, "⚪")
                    
                    report.append(f"\n{emoji} **{change.requirement_id}** ({change.change_type.value})")
                    report.append(f"   - Source: {change.change_source}")
                    if change.notes:
                        report.append(f"   - {change.notes[:200]}")
        
        return "\n".join(report)
