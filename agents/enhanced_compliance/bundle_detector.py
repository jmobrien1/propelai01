"""
RFP Document Bundle Detector
Phase 4.1 - Sprint 1

Detects and classifies multiple documents in an RFP solicitation bundle.
Handles non-standard formats where critical requirements are spread across 10+ files.
"""

import re
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum


class DocumentType(Enum):
    """Document classification types"""
    MAIN_SOLICITATION = "main_solicitation"      # Base RFP/RFQ document
    RFP_LETTER = "rfp_letter"                    # Submission instructions/transmittal
    AMENDMENT = "amendment"                      # Modifications to base
    ATTACHMENT = "attachment"                    # J.1, J.2, J.3 style attachments
    REQUIREMENTS_DOC = "requirements_doc"        # Detailed technical requirements
    PRICING_TEMPLATE = "pricing_template"        # Excel pricing sheets
    QUESTIONNAIRE = "questionnaire"              # Q&A Excel files
    CLAUSE = "clause"                           # Contract clauses
    UNKNOWN = "unknown"


class DocumentPriority(Enum):
    """Priority for processing order"""
    CRITICAL = 1   # Must process first (RFP Letter, Main Solicitation)
    HIGH = 2       # Important context (Requirements, Amendments)
    MEDIUM = 3     # Supporting docs (Attachments)
    LOW = 4        # Reference only (Clauses)


class DocumentBundle:
    """Represents a classified bundle of RFP documents"""
    
    def __init__(self):
        self.documents: List[Dict] = []
        self.main_solicitation: Optional[Dict] = None
        self.rfp_letter: Optional[Dict] = None
        self.amendments: List[Dict] = []
        self.attachments: List[Dict] = []
        self.metadata: Dict = {}
    
    def add_document(self, doc: Dict):
        """Add classified document to bundle"""
        self.documents.append(doc)
        
        # Organize by type
        doc_type = doc.get('type')
        if doc_type == DocumentType.MAIN_SOLICITATION.value:
            self.main_solicitation = doc
        elif doc_type == DocumentType.RFP_LETTER.value:
            self.rfp_letter = doc
        elif doc_type == DocumentType.AMENDMENT.value:
            self.amendments.append(doc)
        elif doc_type == DocumentType.ATTACHMENT.value:
            self.attachments.append(doc)
    
    def get_processing_order(self) -> List[Dict]:
        """Return documents in optimal processing order"""
        return sorted(self.documents, key=lambda d: d.get('priority', 4))
    
    def to_dict(self) -> Dict:
        """Serialize bundle to dict"""
        return {
            'total_documents': len(self.documents),
            'main_solicitation': self.main_solicitation,
            'rfp_letter': self.rfp_letter,
            'amendments_count': len(self.amendments),
            'attachments_count': len(self.attachments),
            'documents': self.documents,
            'metadata': self.metadata
        }


class BundleDetector:
    """
    Detects and classifies RFP document bundles.
    
    Uses:
    - Filename pattern matching
    - Content-based classification
    - Keyword detection
    - Structural analysis
    """
    
    # Filename patterns for classification
    FILENAME_PATTERNS = {
        DocumentType.AMENDMENT: [
            r'amendment',
            r'amnd',
            r'amend',
            r'/\d{4}',  # e.g., "/0004"
            r'mod\s*\d+',
            r'modification'
        ],
        DocumentType.RFP_LETTER: [
            r'rfp.*letter',
            r'rfi.*letter',
            r'rfq.*letter',
            r'transmittal',
            r'cover.*letter',
            r'submission.*instructions',
            r'instructions.*letter'
        ],
        DocumentType.ATTACHMENT: [
            r'attachment.*j[.\s]*\d+',
            r'attach.*\d+',
            r'exhibit.*[a-z]',
            r'appendix.*[a-z]'
        ],
        DocumentType.QUESTIONNAIRE: [
            r'questionnaire',
            r'requirements.*questionnaire',
            r'compliance.*matrix.*xlsx?',
            r'q&a',
            r'questions'
        ],
        DocumentType.PRICING_TEMPLATE: [
            r'pricing.*sheet',
            r'price.*schedule',
            r'cost.*template',
            r'j[.\s]*3.*pricing'
        ]
    }
    
    # Content keywords for classification
    CONTENT_KEYWORDS = {
        DocumentType.RFP_LETTER: [
            'submission instructions',
            'volume i',
            'volume ii',
            'volume iii',
            'page limit',
            'proposal shall be submitted',
            'quote shall be submitted',
            'due date for submission'
        ],
        DocumentType.AMENDMENT: [
            'this amendment',
            'hereby amended',
            'modifies the solicitation',
            'replaces the following',
            'deletes the following',
            'supersedes'
        ],
        DocumentType.MAIN_SOLICITATION: [
            'request for proposal',
            'request for quote',
            'solicitation number',
            'section l',
            'section m',
            'section c',
            'performance work statement',
            'statement of work'
        ]
    }
    
    def __init__(self):
        self.bundle = DocumentBundle()
    
    def classify_document(self, filename: str, file_path: str, content_preview: str = None) -> Dict:
        """
        Classify a single document.
        
        Args:
            filename: Original filename
            file_path: Full path to file
            content_preview: Optional first 500 chars for content analysis
        
        Returns:
            Dict with classification results
        """
        filename_lower = filename.lower()
        
        # Step 1: Filename pattern matching
        doc_type = self._classify_by_filename(filename_lower)
        confidence = 0.7 if doc_type != DocumentType.UNKNOWN else 0.3
        
        # Step 2: Content-based classification (if preview available)
        if content_preview and doc_type == DocumentType.UNKNOWN:
            content_type, content_confidence = self._classify_by_content(content_preview)
            if content_confidence > confidence:
                doc_type = content_type
                confidence = content_confidence
        
        # Step 3: Determine priority
        priority = self._determine_priority(doc_type)
        
        # Step 4: Extract metadata from filename
        metadata = self._extract_filename_metadata(filename_lower)
        
        return {
            'filename': filename,
            'file_path': file_path,
            'type': doc_type.value,
            'priority': priority.value,
            'confidence': confidence,
            'metadata': metadata
        }
    
    def _classify_by_filename(self, filename: str) -> DocumentType:
        """Classify based on filename patterns"""
        for doc_type, patterns in self.FILENAME_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return doc_type
        
        # Default: If contains solicitation number pattern, likely main doc
        if re.search(r'[a-z]{2,4}\d{2}[a-z]?\d{4,6}', filename):
            return DocumentType.MAIN_SOLICITATION
        
        return DocumentType.UNKNOWN
    
    def _classify_by_content(self, content: str) -> tuple:
        """Classify based on content keywords"""
        content_lower = content.lower()
        
        best_match = DocumentType.UNKNOWN
        best_score = 0
        
        for doc_type, keywords in self.CONTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_match = doc_type
        
        confidence = min(0.9, 0.5 + (best_score * 0.1))
        return best_match, confidence
    
    def _determine_priority(self, doc_type: DocumentType) -> DocumentPriority:
        """Determine processing priority based on document type"""
        priority_map = {
            DocumentType.RFP_LETTER: DocumentPriority.CRITICAL,
            DocumentType.MAIN_SOLICITATION: DocumentPriority.CRITICAL,
            DocumentType.AMENDMENT: DocumentPriority.HIGH,
            DocumentType.REQUIREMENTS_DOC: DocumentPriority.HIGH,
            DocumentType.ATTACHMENT: DocumentPriority.MEDIUM,
            DocumentType.QUESTIONNAIRE: DocumentPriority.MEDIUM,
            DocumentType.PRICING_TEMPLATE: DocumentPriority.MEDIUM,
            DocumentType.CLAUSE: DocumentPriority.LOW,
            DocumentType.UNKNOWN: DocumentPriority.LOW
        }
        return priority_map.get(doc_type, DocumentPriority.LOW)
    
    def _extract_filename_metadata(self, filename: str) -> Dict:
        """Extract metadata from filename"""
        metadata = {}
        
        # Extract amendment number
        amend_match = re.search(r'/(\d{4})', filename)
        if amend_match:
            metadata['amendment_number'] = amend_match.group(1)
        
        # Extract attachment identifier
        attach_match = re.search(r'j[.\s]*(\d+)', filename, re.IGNORECASE)
        if attach_match:
            metadata['attachment_id'] = f"J.{attach_match.group(1)}"
        
        # Extract solicitation number
        sol_match = re.search(r'([a-z]{2,4}\d{2}[a-z]?\d{4,6})', filename, re.IGNORECASE)
        if sol_match:
            metadata['solicitation_number'] = sol_match.group(1).upper()
        
        return metadata
    
    def detect_from_files(self, file_paths: List[str]) -> DocumentBundle:
        """
        Convenience method: Detect bundle from file paths.
        
        Args:
            file_paths: List of file path strings
        
        Returns:
            DocumentBundle with all classified documents
        """
        # Convert file paths to file info dicts
        files = []
        for file_path in file_paths:
            filename = Path(file_path).name
            files.append({
                'filename': filename,
                'file_path': file_path,
                'content_preview': None
            })
        
        return self.detect_bundle(files)
    
    def detect_bundle(self, files: List[Dict]) -> DocumentBundle:
        """
        Classify all files in bundle and organize.
        
        Args:
            files: List of dicts with 'filename', 'file_path', and optional 'content_preview'
        
        Returns:
            DocumentBundle with all classified documents
        """
        self.bundle = DocumentBundle()
        
        # Classify each document
        for file_info in files:
            classified = self.classify_document(
                filename=file_info['filename'],
                file_path=file_info['file_path'],
                content_preview=file_info.get('content_preview')
            )
            self.bundle.add_document(classified)
        
        # Extract bundle-level metadata
        self._extract_bundle_metadata()
        
        # Validate bundle completeness
        self._validate_bundle()
        
        return self.bundle
    
    def _extract_bundle_metadata(self):
        """Extract metadata from the entire bundle"""
        # Find common solicitation number
        sol_numbers = []
        for doc in self.bundle.documents:
            sol_num = doc.get('metadata', {}).get('solicitation_number')
            if sol_num:
                sol_numbers.append(sol_num)
        
        if sol_numbers:
            # Most common solicitation number
            self.bundle.metadata['solicitation_number'] = max(set(sol_numbers), key=sol_numbers.count)
        
        # Count amendments
        self.bundle.metadata['amendment_count'] = len(self.bundle.amendments)
        
        # Identify if non-standard format
        if self.bundle.rfp_letter and not self.bundle.main_solicitation:
            self.bundle.metadata['format'] = 'non_standard_rfq'
        elif self.bundle.main_solicitation:
            self.bundle.metadata['format'] = 'standard_ucf'
        else:
            self.bundle.metadata['format'] = 'unknown'
    
    def _validate_bundle(self):
        """Validate bundle completeness and flag issues"""
        issues = []
        warnings = []
        
        # Check for main solicitation or RFP letter
        if not self.bundle.main_solicitation and not self.bundle.rfp_letter:
            issues.append('No main solicitation or RFP letter detected')
        
        # Check for unclassified documents
        unknown_docs = [d for d in self.bundle.documents if d['type'] == DocumentType.UNKNOWN.value]
        if unknown_docs:
            warnings.append(f'{len(unknown_docs)} document(s) could not be classified')
        
        # Check for amendments without base
        if self.bundle.amendments and not self.bundle.main_solicitation:
            warnings.append('Amendments detected but no base solicitation found')
        
        self.bundle.metadata['validation'] = {
            'issues': issues,
            'warnings': warnings,
            'complete': len(issues) == 0
        }


def detect_and_classify(files: List[Dict]) -> DocumentBundle:
    """
    Convenience function for bundle detection.
    
    Args:
        files: List of file info dicts
    
    Returns:
        Classified DocumentBundle
    """
    detector = BundleDetector()
    return detector.detect_bundle(files)
