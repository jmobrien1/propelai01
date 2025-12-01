"""
RFP Document Correlator
Phase 4.1 - Sprint 1

Correlates requirements across multiple documents in an RFP bundle.
Links base solicitation → amendments → attachments → RFP letter.
"""

from typing import List, Dict, Optional, Set
from datetime import datetime
import re


class RequirementSource:
    """Tracks the source of a requirement across documents"""
    
    def __init__(self, doc_id: str, doc_type: str, page: int = None, section: str = None):
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.page = page
        self.section = section
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'doc_type': self.doc_type,
            'page': self.page,
            'section': self.section
        }


class CorrelatedRequirement:
    """Requirement with sources from multiple documents"""
    
    def __init__(self, req_id: str, text: str, primary_source: RequirementSource):
        self.req_id = req_id
        self.text = text
        self.primary_source = primary_source
        self.additional_sources: List[RequirementSource] = []
        self.amendments: List[Dict] = []  # Tracks if req was modified
        self.related_reqs: Set[str] = set()  # Related requirement IDs
        self.metadata: Dict = {}
    
    def add_source(self, source: RequirementSource):
        """Add additional source document"""
        self.additional_sources.append(source)
    
    def add_amendment(self, amendment_info: Dict):
        """Track amendment that modified this requirement"""
        self.amendments.append(amendment_info)
    
    def to_dict(self) -> Dict:
        return {
            'req_id': self.req_id,
            'text': self.text,
            'primary_source': self.primary_source.to_dict(),
            'additional_sources': [s.to_dict() for s in self.additional_sources],
            'amendments': self.amendments,
            'related_reqs': list(self.related_reqs),
            'metadata': self.metadata
        }


class CorrelatedRFP:
    """Represents a fully correlated RFP with unified requirements"""
    
    def __init__(self, rfp_id: str, bundle_metadata: Dict):
        self.rfp_id = rfp_id
        self.bundle_metadata = bundle_metadata
        self.requirements: List[CorrelatedRequirement] = []
        self.structural_requirements: List[Dict] = []  # Page limits, formatting, etc.
        self.compliance_flags: List[Dict] = []  # Critical compliance rules
        self.document_hierarchy: Dict = {}
        self.processing_summary: Dict = {}
    
    def add_requirement(self, req: CorrelatedRequirement):
        """Add correlated requirement"""
        self.requirements.append(req)
    
    def add_structural_requirement(self, struct_req: Dict):
        """Add structural requirement (page limits, formatting)"""
        self.structural_requirements.append(struct_req)
    
    def add_compliance_flag(self, flag: Dict):
        """Add critical compliance flag"""
        self.compliance_flags.append(flag)
    
    def to_dict(self) -> Dict:
        return {
            'rfp_id': self.rfp_id,
            'bundle_metadata': self.bundle_metadata,
            'requirements': [r.to_dict() for r in self.requirements],
            'structural_requirements': self.structural_requirements,
            'compliance_flags': self.compliance_flags,
            'document_hierarchy': self.document_hierarchy,
            'processing_summary': self.processing_summary
        }


class DocumentCorrelator:
    """
    Correlates requirements from multiple RFP documents.
    
    Workflow:
    1. Process documents in priority order
    2. Extract requirements from each
    3. Link related requirements across docs
    4. Track amendments and modifications
    5. Create unified requirements list
    """
    
    def __init__(self):
        self.correlated_rfp: Optional[CorrelatedRFP] = None
        self.requirement_map: Dict[str, CorrelatedRequirement] = {}  # ID -> Requirement
    
    def correlate_bundle(self, rfp_id: str, bundle: 'DocumentBundle', 
                        extracted_reqs: Dict[str, List[Dict]]) -> CorrelatedRFP:
        """
        Correlate all documents in bundle.
        
        Args:
            rfp_id: RFP identifier
            bundle: Classified DocumentBundle
            extracted_reqs: Dict mapping doc_id -> list of extracted requirements
        
        Returns:
            CorrelatedRFP with unified requirements
        """
        self.correlated_rfp = CorrelatedRFP(rfp_id, bundle.metadata)
        self.requirement_map = {}
        
        # Step 1: Process documents in priority order
        processing_order = bundle.get_processing_order()
        self.correlated_rfp.document_hierarchy = self._build_hierarchy(processing_order)
        
        # Step 2: Process each document's requirements
        for doc in processing_order:
            doc_id = doc['filename']
            doc_type = doc['type']
            reqs = extracted_reqs.get(doc_id, [])
            
            if doc_type == 'rfp_letter':
                # RFP Letter contains structural requirements
                self._process_rfp_letter(doc, reqs)
            elif doc_type == 'amendment':
                # Amendments modify existing requirements
                self._process_amendment(doc, reqs)
            elif doc_type == 'attachment':
                # Attachments add detailed requirements
                self._process_attachment(doc, reqs)
            else:
                # Main solicitation and other docs
                self._process_standard_requirements(doc, reqs)
        
        # Step 3: Link related requirements
        self._link_related_requirements()
        
        # Step 4: Generate processing summary
        self._generate_summary()
        
        return self.correlated_rfp
    
    def _build_hierarchy(self, documents: List[Dict]) -> Dict:
        """Build document hierarchy"""
        hierarchy = {
            'base': None,
            'amendments': [],
            'attachments': [],
            'instructions': None
        }
        
        for doc in documents:
            doc_type = doc['type']
            if doc_type == 'main_solicitation':
                hierarchy['base'] = doc['filename']
            elif doc_type == 'amendment':
                hierarchy['amendments'].append(doc['filename'])
            elif doc_type == 'attachment':
                hierarchy['attachments'].append(doc['filename'])
            elif doc_type == 'rfp_letter':
                hierarchy['instructions'] = doc['filename']
        
        return hierarchy
    
    def _process_rfp_letter(self, doc: Dict, reqs: List[Dict]):
        """Process RFP Letter for structural requirements"""
        # RFP Letters typically contain:
        # - Volume structure
        # - Page limits
        # - Formatting rules
        # - Critical compliance rules
        
        for req in reqs:
            # Check if this is a structural requirement
            if self._is_structural_requirement(req):
                struct_req = {
                    'type': req.get('type', 'structural'),
                    'text': req['text'],
                    'source': doc['filename'],
                    'metadata': req.get('metadata', {})
                }
                self.correlated_rfp.add_structural_requirement(struct_req)
            
            # Check if this is a critical compliance rule
            if self._is_compliance_flag(req):
                flag = {
                    'flag_id': f"CF-{len(self.correlated_rfp.compliance_flags) + 1:03d}",
                    'severity': req.get('severity', 'MEDIUM'),
                    'rule': req['text'],
                    'source': doc['filename'],
                    'metadata': req.get('metadata', {})
                }
                self.correlated_rfp.add_compliance_flag(flag)
    
    def _is_structural_requirement(self, req: Dict) -> bool:
        """Check if requirement is structural (page limits, formatting)"""
        text_lower = req['text'].lower()
        structural_keywords = [
            'page limit',
            'pages',
            'font',
            'margin',
            'format',
            'volume i',
            'volume ii',
            'volume iii',
            'line spacing'
        ]
        return any(kw in text_lower for kw in structural_keywords)
    
    def _is_compliance_flag(self, req: Dict) -> bool:
        """Check if requirement is a critical compliance rule"""
        text_lower = req['text'].lower()
        flag_keywords = [
            'only in volume',
            'shall not exceed',
            'will not be evaluated',
            'may be rejected',
            'must be registered',
            'mandatory'
        ]
        return any(kw in text_lower for kw in flag_keywords)
    
    def _process_amendment(self, doc: Dict, reqs: List[Dict]):
        """Process amendment and track modifications"""
        amendment_info = {
            'amendment_id': doc.get('metadata', {}).get('amendment_number', 'Unknown'),
            'filename': doc['filename'],
            'date': datetime.now().isoformat()
        }
        
        for req in reqs:
            req_id = req.get('id', f"REQ-AMD-{len(self.requirement_map) + 1:03d}")
            
            # Check if this modifies an existing requirement
            if self._matches_existing_requirement(req):
                # Update existing requirement
                existing_req = self._find_matching_requirement(req)
                if existing_req:
                    existing_req.add_amendment({
                        **amendment_info,
                        'change_type': req.get('change_type', 'modified'),
                        'new_text': req['text']
                    })
            else:
                # New requirement added by amendment
                source = RequirementSource(
                    doc_id=doc['filename'],
                    doc_type='amendment',
                    page=req.get('page'),
                    section=req.get('section')
                )
                corr_req = CorrelatedRequirement(req_id, req['text'], source)
                corr_req.metadata['added_by_amendment'] = amendment_info['amendment_id']
                
                self.requirement_map[req_id] = corr_req
                self.correlated_rfp.add_requirement(corr_req)
    
    def _process_attachment(self, doc: Dict, reqs: List[Dict]):
        """Process attachment requirements"""
        attachment_id = doc.get('metadata', {}).get('attachment_id', 'Unknown')
        
        for req in reqs:
            req_id = req.get('id', f"REQ-ATT-{len(self.requirement_map) + 1:03d}")
            
            source = RequirementSource(
                doc_id=doc['filename'],
                doc_type='attachment',
                page=req.get('page'),
                section=req.get('section')
            )
            
            corr_req = CorrelatedRequirement(req_id, req['text'], source)
            corr_req.metadata['attachment_id'] = attachment_id
            corr_req.metadata['source_type'] = req.get('type', 'technical')
            
            self.requirement_map[req_id] = corr_req
            self.correlated_rfp.add_requirement(corr_req)
    
    def _process_standard_requirements(self, doc: Dict, reqs: List[Dict]):
        """Process standard requirements from main solicitation"""
        for req in reqs:
            req_id = req.get('id', f"REQ-{len(self.requirement_map) + 1:03d}")
            
            source = RequirementSource(
                doc_id=doc['filename'],
                doc_type=doc['type'],
                page=req.get('page'),
                section=req.get('section')
            )
            
            corr_req = CorrelatedRequirement(req_id, req['text'], source)
            corr_req.metadata.update(req.get('metadata', {}))
            
            self.requirement_map[req_id] = corr_req
            self.correlated_rfp.add_requirement(corr_req)
    
    def _matches_existing_requirement(self, req: Dict) -> bool:
        """Check if requirement matches an existing one"""
        # Simple similarity check (can be enhanced)
        req_text = req['text'].lower().strip()
        for existing in self.requirement_map.values():
            existing_text = existing.text.lower().strip()
            # Exact match or very similar
            if req_text == existing_text or self._similarity(req_text, existing_text) > 0.9:
                return True
        return False
    
    def _find_matching_requirement(self, req: Dict) -> Optional[CorrelatedRequirement]:
        """Find existing requirement that matches"""
        req_text = req['text'].lower().strip()
        for existing in self.requirement_map.values():
            existing_text = existing.text.lower().strip()
            if req_text == existing_text or self._similarity(req_text, existing_text) > 0.9:
                return existing
        return None
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard)"""
        set1 = set(s1.split())
        set2 = set(s2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0
    
    def _link_related_requirements(self):
        """Link related requirements (e.g., same section, same topic)"""
        # Group by section
        section_groups = {}
        for req in self.requirement_map.values():
            section = req.primary_source.section
            if section:
                if section not in section_groups:
                    section_groups[section] = []
                section_groups[section].append(req.req_id)
        
        # Link requirements in same section
        for section, req_ids in section_groups.items():
            for req_id in req_ids:
                req = self.requirement_map[req_id]
                req.related_reqs.update(r for r in req_ids if r != req_id)
    
    def _generate_summary(self):
        """Generate processing summary"""
        self.correlated_rfp.processing_summary = {
            'total_requirements': len(self.correlated_rfp.requirements),
            'structural_requirements': len(self.correlated_rfp.structural_requirements),
            'compliance_flags': len(self.correlated_rfp.compliance_flags),
            'amendments_processed': len(self.correlated_rfp.document_hierarchy.get('amendments', [])),
            'attachments_processed': len(self.correlated_rfp.document_hierarchy.get('attachments', [])),
            'sources': {
                'main': self.correlated_rfp.document_hierarchy.get('base'),
                'letter': self.correlated_rfp.document_hierarchy.get('instructions'),
                'amendments': len(self.correlated_rfp.document_hierarchy.get('amendments', [])),
                'attachments': len(self.correlated_rfp.document_hierarchy.get('attachments', []))
            }
        }


def correlate_documents(rfp_id: str, bundle: 'DocumentBundle', 
                       extracted_reqs: Dict[str, List[Dict]]) -> CorrelatedRFP:
    """
    Convenience function for document correlation.
    
    Args:
        rfp_id: RFP identifier
        bundle: Classified DocumentBundle
        extracted_reqs: Extracted requirements per document
    
    Returns:
        CorrelatedRFP
    """
    correlator = DocumentCorrelator()
    return correlator.correlate_bundle(rfp_id, bundle, extracted_reqs)
