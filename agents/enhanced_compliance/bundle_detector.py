"""
PropelAI Cycle 5: Bundle Detector
Auto-detect and classify RFP document bundles

Handles NIH, DoD, GSA, and generic federal RFP patterns
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .models import RFPBundle, DocumentType


class BundleDetector:
    """
    Auto-detect RFP bundle structure from files
    
    Classifies documents by filename patterns and content analysis
    """
    
    # Filename patterns for document classification
    FILENAME_PATTERNS = {
        DocumentType.MAIN_SOLICITATION: [
            r"SF[-_]?33",
            r"SF[-_]?1449", 
            r"RFP[-_]?\d+",
            r"solicitation",
            r"^RFP[_\s]",
            r"Request\s*for\s*Proposal",
        ],
        DocumentType.STATEMENT_OF_WORK: [
            r"SOW",
            r"PWS",
            r"Statement\s*of\s*Work",
            r"Performance\s*Work\s*Statement",
            r"Scope\s*of\s*Work",
            r"ATTACHMENT[-_]?2",  # NIH pattern: SOW is often Attachment 2
        ],
        DocumentType.AMENDMENT: [
            r"Amendment[-_]?\d*",
            r"Modification[-_]?\d*",
            r"AMEND[-_]?\d*",
            r"MOD[-_]?\d*",
        ],
        DocumentType.CDRL: [
            r"CDRL",
            r"DD[-_]?1423",
            r"Contract\s*Data\s*Requirements",
            r"Exhibit[-_]?A",
        ],
        DocumentType.RESEARCH_OUTLINE: [
            r"Research[-_]?Outline",
            r"RO[-_]?[IVX]+",
            r"RO[-_]?\d+",
        ],
        DocumentType.BUDGET_TEMPLATE: [
            r"Budget",
            r"Cost[-_]?Template",
            r"Pricing[-_]?Template",
            r"ATTACHMENT[-_]?11",  # NIH pattern
            r"\.xlsx?$",
        ],
        DocumentType.SECURITY: [
            r"DD[-_]?254",
            r"Security\s*Classification",
            r"Clearance",
        ],
        DocumentType.QA_RESPONSE: [
            r"Q[-_]?&[-_]?A",
            r"Questions?\s*and\s*Answers?",
            r"QA[-_]?Response",
        ],
        DocumentType.ATTACHMENT: [
            r"Attachment[-_]?[A-Z0-9]+",
            r"Exhibit[-_]?[A-Z0-9]+",
            r"Appendix[-_]?[A-Z0-9]+",
            r"J[-_]?\d+",  # DoD J-series attachments
        ],
    }
    
    # Content patterns for validation/detection
    CONTENT_PATTERNS = {
        DocumentType.MAIN_SOLICITATION: [
            r"SECTION\s*[ABCLM]",
            r"FAR\s*\d+\.\d+",
            r"PART\s*I.*SCHEDULE",
            r"Contracting\s*Officer",
        ],
        DocumentType.STATEMENT_OF_WORK: [
            r"Statement\s*of\s*Work",
            r"Scope\s*of\s*Work",
            r"(?:Contractor|Vendor)\s*shall",
            r"Performance\s*Requirements",
            r"Deliverables",
        ],
        DocumentType.AMENDMENT: [
            r"Amendment\s*(?:No\.?|Number)",
            r"This\s*amendment\s*modifies",
            r"The\s*following\s*changes",
            r"is\s*hereby\s*(?:amended|modified)",
        ],
        DocumentType.RESEARCH_OUTLINE: [
            r"Research\s*Outline",
            r"Background\s*and\s*Rationale",
            r"Specific\s*Aims",
            r"Phase\s*[I1].*Phase\s*[I2]",
        ],
    }
    
    # NIH-specific patterns
    NIH_PATTERNS = {
        "agency_identifiers": [
            r"NIH",
            r"National\s*Institutes?\s*of\s*Health",
            r"NIEHS",
            r"NCI",
            r"NIAID",
            r"NICHD",
        ],
        "solicitation_number": r"75N\d{11}",  # NIH solicitation format
        "research_outline_refs": r"RO\s*[IVX]+\.?\d*|Research\s*Outline\s*[IVX]+",
    }
    
    # DoD-specific patterns
    DOD_PATTERNS = {
        "agency_identifiers": [
            r"DoD",
            r"Department\s*of\s*Defense",
            r"Navy|Army|Air\s*Force",
            r"NAVSEA|NAVAIR",
        ],
        "solicitation_number": r"N\d{5}[-]?\d{2}[-]?[RQ][-]?\d+",  # Navy format
        "dfars_references": r"DFARS\s*\d+\.\d+",
    }
    
    def __init__(self):
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[DocumentType, List[re.Pattern]]:
        """Pre-compile regex patterns for performance"""
        compiled = {}
        for doc_type, patterns in self.FILENAME_PATTERNS.items():
            compiled[doc_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        return compiled
    
    def detect_from_files(self, filepaths: List[str]) -> RFPBundle:
        """
        Detect bundle structure from a list of file paths
        
        Args:
            filepaths: List of paths to RFP documents
            
        Returns:
            RFPBundle with classified documents
        """
        bundle = RFPBundle(solicitation_number="UNKNOWN")
        
        # Classify each file
        classifications: List[Tuple[str, DocumentType, float]] = []
        
        for filepath in filepaths:
            doc_type, confidence = self._classify_file(filepath)
            classifications.append((filepath, doc_type, confidence))
        
        # Sort by confidence to handle conflicts
        classifications.sort(key=lambda x: x[2], reverse=True)
        
        # Assign to bundle (highest confidence wins for main/SOW)
        main_assigned = False
        sow_assigned = False
        
        for filepath, doc_type, confidence in classifications:
            filename = os.path.basename(filepath)
            
            if doc_type == DocumentType.MAIN_SOLICITATION and not main_assigned:
                bundle.main_document = filepath
                main_assigned = True
                # Try to extract solicitation number from filename first
                sol_num = self._extract_solicitation_number(filename)
                if sol_num:
                    bundle.solicitation_number = sol_num

            elif doc_type == DocumentType.STATEMENT_OF_WORK and not sow_assigned:
                bundle.sow_document = filepath
                sow_assigned = True
                
            elif doc_type == DocumentType.AMENDMENT:
                bundle.amendments.append(filepath)
                
            elif doc_type == DocumentType.RESEARCH_OUTLINE:
                ro_id = self._extract_research_outline_id(filename)
                bundle.research_outlines[ro_id] = filepath
                
            elif doc_type == DocumentType.BUDGET_TEMPLATE:
                bundle.budget_templates.append(filepath)
                
            elif doc_type == DocumentType.ATTACHMENT:
                att_id = self._extract_attachment_id(filename)
                bundle.attachments[att_id] = filepath
                
            else:
                # Default to attachment
                att_id = self._extract_attachment_id(filename)
                bundle.attachments[att_id] = filepath
        
        # Sort amendments by number
        bundle.amendments = self._sort_amendments(bundle.amendments)

        # If solicitation number is still UNKNOWN, try to extract from document content
        if bundle.solicitation_number == "UNKNOWN":
            sol_num = self._extract_solicitation_from_documents(bundle)
            if sol_num:
                bundle.solicitation_number = sol_num

        return bundle

    def _extract_solicitation_from_documents(self, bundle: RFPBundle) -> Optional[str]:
        """
        Try to extract solicitation number from document content.

        Reads the first 500 characters of the main document and any attachments
        to find explicit "Solicitation Number:" patterns.
        """
        # Priority order: main document, then attachments
        documents_to_check = []

        if bundle.main_document:
            documents_to_check.append(bundle.main_document)

        # Also check attachments (solicitation number often in attachment headers)
        for att_path in bundle.attachments.values():
            documents_to_check.append(att_path)

        for doc_path in documents_to_check:
            try:
                content = self._read_document_header(doc_path)
                if content:
                    sol_num = self.extract_solicitation_from_content(content)
                    if sol_num and sol_num != "UNKNOWN":
                        return sol_num
            except Exception:
                continue

        return None

    def _read_document_header(self, filepath: str, max_chars: int = 2000) -> Optional[str]:
        """Read the first portion of a document for metadata extraction."""
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.txt':
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(max_chars)

            elif ext == '.pdf':
                # Try pypdf first
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    if reader.pages:
                        text = reader.pages[0].extract_text() or ""
                        return text[:max_chars]
                except ImportError:
                    pass

            elif ext in ['.docx', '.doc']:
                # Try python-docx
                try:
                    from docx import Document
                    doc = Document(filepath)
                    text_parts = []
                    for para in doc.paragraphs[:10]:  # First 10 paragraphs
                        text_parts.append(para.text)
                        if sum(len(t) for t in text_parts) > max_chars:
                            break
                    return "\n".join(text_parts)[:max_chars]
                except ImportError:
                    pass

        except Exception:
            pass

        return None
    
    def detect_from_folder(self, folder_path: str) -> RFPBundle:
        """
        Detect bundle from all files in a folder
        
        Args:
            folder_path: Path to folder containing RFP documents
            
        Returns:
            RFPBundle with classified documents
        """
        filepaths = []
        
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                # Skip hidden files and non-documents
                if filename.startswith('.'):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.txt']:
                    filepaths.append(os.path.join(root, filename))
        
        return self.detect_from_files(filepaths)
    
    def _classify_file(self, filepath: str) -> Tuple[DocumentType, float]:
        """
        Classify a single file by its filename
        
        Returns:
            (DocumentType, confidence_score)
        """
        filename = os.path.basename(filepath).upper()
        
        best_match = DocumentType.ATTACHMENT
        best_confidence = 0.0
        
        for doc_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(filename):
                    # Weight by specificity of pattern
                    confidence = 0.8
                    
                    # Boost confidence for more specific matches
                    if doc_type == DocumentType.MAIN_SOLICITATION:
                        if "SF33" in filename or "SF1449" in filename:
                            confidence = 0.95
                        elif "RFP" in filename:
                            confidence = 0.9
                    elif doc_type == DocumentType.AMENDMENT:
                        if re.search(r"AMENDMENT[-_]?\d", filename):
                            confidence = 0.95
                    elif doc_type == DocumentType.STATEMENT_OF_WORK:
                        if "SOW" in filename or "PWS" in filename:
                            confidence = 0.95
                    
                    if confidence > best_confidence:
                        best_match = doc_type
                        best_confidence = confidence
                    break
        
        return best_match, best_confidence
    
    def _extract_solicitation_number(self, text: str) -> Optional[str]:
        """Extract solicitation number from filename or text"""
        # VA format: 36C26126Q0281 (36C + 5 digits + Q/R + 4 digits)
        # Matches VA Network Contracting Office formats
        va_match = re.search(r"36C\d{5}[QR]\d{4}", text, re.IGNORECASE)
        if va_match:
            return va_match.group().upper()

        # NIH format: 75N96025R00004
        nih_match = re.search(r"75N\d{11}", text)
        if nih_match:
            return nih_match.group()

        # DoD Navy format: N0017826R30020003
        navy_match = re.search(r"N\d{5}\d{2}[RQ]\d+", text)
        if navy_match:
            return navy_match.group()

        # Air Force formats:
        # - FA880625RB003 (6 digits + 2 letters + 3 digits)
        # - FA8806-25-R-B003 (with hyphens)
        af_patterns = [
            r"FA\d{6}[A-Z]{2}\d{3,4}",  # Modern format: FA880625RB003
            r"FA\d{4}[-]?\d{2}[-]?[A-Z][-]?\d{4}",  # Legacy format: FA8806-25-R-0001
        ]
        for pattern in af_patterns:
            af_match = re.search(pattern, text, re.IGNORECASE)
            if af_match:
                return af_match.group().upper()

        # Army format: W911NF-XX-X-XXXX
        army_match = re.search(r"W\d{3}[A-Z]{2}[-]?\d{2}[-]?[A-Z][-]?\d{4}", text, re.IGNORECASE)
        if army_match:
            return army_match.group().upper()

        # GSA format: GS-XXX-XXXX or 47QXXX-XX-X-XXXX
        gsa_match = re.search(r"(?:GS[-]?[A-Z0-9]{2,5}[-]?\d{4}|47Q[A-Z]{2,4}[-]?\d{2}[-]?[A-Z][-]?\d{4})", text, re.IGNORECASE)
        if gsa_match:
            return gsa_match.group().upper()

        # Generic format
        generic_match = re.search(r"(?:RFP|SOL)[-_]?(\S+)", text, re.IGNORECASE)
        if generic_match:
            return generic_match.group(1)

        return None

    def extract_solicitation_from_content(self, content: str, max_chars: int = 2000) -> Optional[str]:
        """
        Extract solicitation number from document content header.

        Searches the first `max_chars` characters of the document for
        explicit "Solicitation Number:" patterns and common formats.

        Args:
            content: Full document text content
            max_chars: How many characters from the start to search (default 2000)

        Returns:
            Extracted solicitation number or None
        """
        # Only search the header portion
        header_text = content[:max_chars]

        # Look for explicit solicitation number patterns (comprehensive list)
        explicit_patterns = [
            # SF1449 Block 5 patterns (most common in federal solicitations)
            r"5\.\s*SOLICITATION\s*NUMBER\s*[\n\r\s]*([A-Z0-9][-A-Z0-9]+)",
            r"SOLICITATION\s*NUMBER\s*[\n\r\s]*([A-Z0-9][-A-Z0-9]+)",
            # Standard patterns
            r"Solicitation\s*(?:Number|No\.?|#)[:\s]+([A-Z0-9][-A-Z0-9]+)",
            r"Solicitation[:\s]+([A-Z0-9][-A-Z0-9]+)",
            # RFP/RFQ patterns
            r"RFP\s*(?:Number|No\.?|#)?[:\s]+([A-Z0-9][-A-Z0-9]+)",
            r"RFQ\s*(?:Number|No\.?|#)?[:\s]+([A-Z0-9][-A-Z0-9]+)",
            r"RFI\s*(?:Number|No\.?|#)?[:\s]+([A-Z0-9][-A-Z0-9]+)",
            # Contract/Reference patterns
            r"Contract\s*(?:Number|No\.?)[:\s]+([A-Z0-9][-A-Z0-9]+)",
            r"Reference\s*(?:Number|No\.?)[:\s]+([A-Z0-9][-A-Z0-9]+)",
            r"Award\s*(?:Number|No\.?)[:\s]+([A-Z0-9][-A-Z0-9]+)",
            # SF1449 block patterns
            r"(?:Block\s*)?2\.\s*(?:Contract|Solicitation)\s*(?:Number|No\.?)[:\s]*([A-Z0-9][-A-Z0-9]+)",
            # Inline patterns (commonly found in document headers)
            r"(?:Sol|Solicitation)\s*#\s*([A-Z0-9][-A-Z0-9]+)",
            # Page header patterns (VA style: "36C26126Q0281" at top of page)
            r"^([0-9]{2}[A-Z][0-9]+[A-Z][0-9]+)\s*$",
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                # Validate it looks like a real solicitation number (at least 5 chars)
                if len(result) >= 5:
                    return result

        # Fall back to format-based extraction
        return self._extract_solicitation_number(header_text)
    
    def _extract_research_outline_id(self, filename: str) -> str:
        """Extract Research Outline ID (e.g., 'RO-I', 'RO-III')"""
        match = re.search(r"RO[-_]?([IVX]+|\d+)", filename, re.IGNORECASE)
        if match:
            return f"RO-{match.group(1).upper()}"
        return f"RO-{len(filename) % 10}"
    
    def _extract_attachment_id(self, filename: str) -> str:
        """Extract attachment ID from filename"""
        # J-series: J.1, J.2, etc.
        j_match = re.search(r"J[-_.]?(\d+)", filename, re.IGNORECASE)
        if j_match:
            return f"J.{j_match.group(1)}"
        
        # Attachment N
        att_match = re.search(r"Attachment[-_]?(\d+|[A-Z])", filename, re.IGNORECASE)
        if att_match:
            return f"ATT-{att_match.group(1).upper()}"
        
        # Exhibit
        exh_match = re.search(r"Exhibit[-_]?([A-Z]|\d+)", filename, re.IGNORECASE)
        if exh_match:
            return f"EXH-{exh_match.group(1).upper()}"
        
        # Default: use filename hash
        return f"DOC-{hash(filename) % 1000:03d}"
    
    def _sort_amendments(self, amendments: List[str]) -> List[str]:
        """Sort amendments by number"""
        def extract_number(filepath: str) -> int:
            match = re.search(r"(\d+)", os.path.basename(filepath))
            return int(match.group(1)) if match else 999
        
        return sorted(amendments, key=extract_number)
    
    def classify_by_content(self, text: str, current_type: DocumentType) -> DocumentType:
        """
        Refine classification by analyzing document content
        
        Args:
            text: First few pages of document text
            current_type: Current classification from filename
            
        Returns:
            Refined DocumentType
        """
        # Check for strong content indicators
        for doc_type, patterns in self.CONTENT_PATTERNS.items():
            matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
            if matches >= 2:  # At least 2 pattern matches
                return doc_type
        
        return current_type
    
    def detect_agency(self, text: str) -> Optional[str]:
        """Detect issuing agency from document text"""
        for pattern in self.NIH_PATTERNS["agency_identifiers"]:
            if re.search(pattern, text, re.IGNORECASE):
                return "NIH"
        
        for pattern in self.DOD_PATTERNS["agency_identifiers"]:
            if re.search(pattern, text, re.IGNORECASE):
                return "DoD"
        
        # Check for other common agencies
        agency_patterns = {
            "GSA": r"General\s*Services\s*Administration|GSA",
            "VA": r"Department\s*of\s*Veterans\s*Affairs|VA\s",
            "DHS": r"Department\s*of\s*Homeland\s*Security|DHS",
            "HHS": r"Health\s*and\s*Human\s*Services|HHS",
        }
        
        for agency, pattern in agency_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return agency
        
        return None
