"""
PropelAI Cycle 5: Enhanced Compliance Agent
Multi-document, graph-based compliance extraction

Replaces naive v1.0 agent with:
- Multi-file bundle ingestion
- Requirements Graph with cross-document edges
- Semantic classification
- Amendment tracking
- 95%+ requirement extraction (vs 40% naive)
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from .models import (
    RequirementNode, RequirementType, RequirementStatus,
    RFPBundle, ParsedDocument, DocumentType,
    ComplianceMatrixRow, ExtractionResult, ConfidenceLevel
)
from .bundle_detector import BundleDetector
from .parser import MultiFormatParser
from .extractor import RequirementExtractor
from .resolver import CrossReferenceResolver
from .strategic_mapper import get_strategic_mapper

# Import base state for compatibility with orchestrator
try:
    from core.state import ProposalState, ProposalPhase, ComplianceStatus
except ImportError:
    # Standalone mode
    ProposalState = dict
    ProposalPhase = type('ProposalPhase', (), {'SHRED': type('SHRED', (), {'value': 'shred'})()})()
    ComplianceStatus = type('ComplianceStatus', (), {'NOT_STARTED': type('NOT_STARTED', (), {'value': 'not_started'})()})()


class EnhancedComplianceAgent:
    """
    Cycle 5: Multi-document, graph-based compliance agent
    
    Upgrades from naive extraction to:
    1. Process entire RFP bundles (not just main document)
    2. Build Requirements Graph with cross-document edges
    3. Semantic classification of requirement types
    4. Track amendments and requirement lifecycle
    5. Generate comprehensive compliance matrix
    
    Target: 95%+ requirement extraction (vs 40% with v1.0)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        use_llm_enhancement: bool = False,
    ):
        """
        Initialize the Enhanced Compliance Agent
        
        Args:
            llm_client: Optional LLM client for semantic enhancement
            use_llm_enhancement: Whether to use LLM for classification
        """
        self.llm_client = llm_client
        self.use_llm_enhancement = use_llm_enhancement and llm_client is not None
        
        # Initialize components
        self.bundle_detector = BundleDetector()
        self.parser = MultiFormatParser()
        self.extractor = RequirementExtractor()
        self.resolver = CrossReferenceResolver()
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "requirements_extracted": 0,
            "cross_references_resolved": 0,
            "processing_time_ms": 0,
        }
    
    def __call__(self, state: ProposalState) -> Dict[str, Any]:
        """
        Main entry point - called by the Orchestrator
        
        Compatible with existing PropelAI orchestrator interface
        """
        start_time = datetime.now()
        
        # Get file paths from state
        file_paths = state.get("rfp_file_paths", [])
        folder_path = state.get("rfp_folder_path", "")
        rfp_raw_text = state.get("rfp_raw_text", "")
        
        # If we have file paths, process as bundle
        if file_paths:
            result = self.process_files(file_paths)
        elif folder_path:
            result = self.process_folder(folder_path)
        elif rfp_raw_text:
            # Fallback: process raw text (like v1.0)
            result = self.process_text(rfp_raw_text)
        else:
            return self._error_result("No RFP content to process", start_time)
        
        # Calculate duration
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Build trace log for orchestrator
        trace_log = {
            "timestamp": start_time.isoformat(),
            "agent_name": "enhanced_compliance_agent",
            "action": "shred_rfp_bundle",
            "input_summary": f"Processed {result.stats.get('documents_processed', 0)} documents",
            "output_summary": f"Extracted {len(result.requirements_graph)} requirements, {result.cross_reference_count} edges",
            "reasoning_trace": f"Used multi-document extraction with semantic classification",
            "duration_ms": duration_ms,
            "tool_calls": [
                {"tool": "bundle_detector", "result": f"{result.stats.get('documents_processed', 0)} files classified"},
                {"tool": "parser", "result": f"{result.stats.get('total_pages', 0)} pages parsed"},
                {"tool": "extractor", "result": f"{len(result.requirements_graph)} requirements"},
                {"tool": "resolver", "result": f"{result.cross_reference_count} cross-references"},
            ]
        }
        
        # Return state update (compatible with orchestrator)
        return {
            "current_phase": ProposalPhase.SHRED.value,
            "requirements": [req.to_dict() for req in result.requirements_graph.values()],
            "requirements_graph": {
                req_id: req.to_dict() for req_id, req in result.requirements_graph.items()
            },
            "compliance_matrix": [self._matrix_row_to_dict(row) for row in result.compliance_matrix],
            "rfp_metadata": {
                "total_requirements": len(result.requirements_graph),
                "by_type": result.stats.get("by_type", {}),
                "by_section": result.stats.get("by_section", {}),
                "documents_processed": result.stats.get("documents_processed", 0),
                "cross_references": result.cross_reference_count,
                "coverage_estimate": result.extraction_coverage,
                "processed_at": start_time.isoformat(),
            },
            "extraction_stats": result.stats,
            "extraction_warnings": result.warnings,
            "agent_trace_log": [trace_log],
            "updated_at": datetime.now().isoformat(),
        }
    
    def process_files(self, file_paths: List[str]) -> ExtractionResult:
        """
        Process a list of RFP files
        
        Args:
            file_paths: List of paths to RFP documents
            
        Returns:
            ExtractionResult with requirements graph and matrix
        """
        start_time = datetime.now()
        
        # Step 1: Detect bundle structure
        bundle = self.bundle_detector.detect_from_files(file_paths)
        
        # Step 2: Parse all documents
        parsed_docs = self.parser.parse_bundle(bundle)
        
        # Step 3: Extract requirements from all documents
        all_requirements = []
        self.extractor.reset_counter()
        
        for doc_key, doc in parsed_docs.items():
            requirements = self.extractor.extract_from_document(doc)
            all_requirements.extend(requirements)
        
        # Step 4: Resolve cross-references (build graph)
        requirements_graph = self.resolver.resolve_references(all_requirements, parsed_docs)
        
        # Step 5: Generate compliance matrix
        compliance_matrix = self._generate_compliance_matrix(requirements_graph)
        
        # Step 6: Calculate statistics
        stats = self.resolver.get_graph_statistics(requirements_graph)
        stats["documents_processed"] = len(parsed_docs)
        stats["total_pages"] = sum(doc.page_count for doc in parsed_docs.values())
        
        # Handle solicitation number from either bundle type
        if hasattr(bundle, 'solicitation_number'):
            stats["solicitation_number"] = bundle.solicitation_number
        elif hasattr(bundle, 'metadata') and isinstance(bundle.metadata, dict):
            stats["solicitation_number"] = bundle.metadata.get('solicitation_number', 'Unknown')
        else:
            stats["solicitation_number"] = "Unknown"
        
        # Calculate estimated coverage
        coverage = self._estimate_coverage(requirements_graph, parsed_docs)
        
        # Find gaps
        gaps = self.resolver.find_gaps(requirements_graph)
        
        return ExtractionResult(
            requirements_graph=requirements_graph,
            compliance_matrix=compliance_matrix,
            stats=stats,
            extraction_coverage=coverage,
            cross_reference_count=stats.get("total_edges", 0),
            started_at=start_time.isoformat(),
            completed_at=datetime.now().isoformat(),
            duration_seconds=(datetime.now() - start_time).total_seconds(),
            warnings=[f"{len(gaps[k])} {k}" for k, v in gaps.items() if v],
        )
    
    def process_folder(self, folder_path: str) -> ExtractionResult:
        """Process all RFP files in a folder"""
        bundle = self.bundle_detector.detect_from_folder(folder_path)
        
        # Get all file paths from bundle
        file_paths = []
        
        # Handle main solicitation (could be dict or string)
        if bundle.main_solicitation:
            main_path = bundle.main_solicitation.get('file_path') if isinstance(bundle.main_solicitation, dict) else bundle.main_solicitation
            if main_path:
                file_paths.append(main_path)
        
        # Handle legacy attributes if they exist
        if hasattr(bundle, 'main_document') and bundle.main_document:
            file_paths.append(bundle.main_document)
        if hasattr(bundle, 'sow_document') and bundle.sow_document:
            file_paths.append(bundle.sow_document)
        
        # Handle amendments (could be list of dicts or list of strings)
        if bundle.amendments:
            for amendment in bundle.amendments:
                amend_path = amendment.get('file_path') if isinstance(amendment, dict) else amendment
                if amend_path:
                    file_paths.append(amend_path)
        
        # Handle attachments (could be list of dicts or dict of strings)
        if bundle.attachments:
            for attachment in bundle.attachments:
                attach_path = attachment.get('file_path') if isinstance(attachment, dict) else attachment
                if attach_path:
                    file_paths.append(attach_path)
        
        # Legacy attributes
        if hasattr(bundle, 'research_outlines'):
            file_paths.extend(bundle.research_outlines.values())
        if hasattr(bundle, 'budget_templates'):
            file_paths.extend(bundle.budget_templates)
        
        return self.process_files(file_paths)
    
    def process_text(self, text: str, filename: str = "rfp.txt") -> ExtractionResult:
        """
        Process raw RFP text (fallback for v1.0 compatibility)
        
        Args:
            text: Raw RFP text
            filename: Optional filename for identification
            
        Returns:
            ExtractionResult
        """
        start_time = datetime.now()
        
        # Create a pseudo-document
        doc = ParsedDocument(
            filepath="",
            filename=filename,
            document_type=DocumentType.MAIN_SOLICITATION,
            full_text=text,
            pages=[text[i:i+3000] for i in range(0, len(text), 3000)],
            page_count=max(1, len(text) // 3000),
        )
        
        # Detect sections
        doc.sections = self.parser._detect_sections(text)
        
        # Extract requirements
        self.extractor.reset_counter()
        requirements = self.extractor.extract_from_document(doc)
        
        # Build graph (limited cross-referencing with single doc)
        requirements_graph = {req.id: req for req in requirements}
        
        # Resolve internal references
        for req in requirements_graph.values():
            self._resolve_internal_references(req, requirements_graph)
        
        # Generate matrix
        compliance_matrix = self._generate_compliance_matrix(requirements_graph)
        
        stats = self.resolver.get_graph_statistics(requirements_graph)
        stats["documents_processed"] = 1
        stats["total_pages"] = doc.page_count
        
        return ExtractionResult(
            requirements_graph=requirements_graph,
            compliance_matrix=compliance_matrix,
            stats=stats,
            extraction_coverage=0.6,  # Lower for single-doc processing
            cross_reference_count=stats.get("total_edges", 0),
            started_at=start_time.isoformat(),
            completed_at=datetime.now().isoformat(),
            duration_seconds=(datetime.now() - start_time).total_seconds(),
            warnings=["Single document processing - coverage may be limited"],
        )
    
    def _generate_compliance_matrix(
        self, 
        requirements_graph: Dict[str, RequirementNode]
    ) -> List[ComplianceMatrixRow]:
        """Generate compliance matrix from requirements graph"""
        matrix = []
        
        # Sort requirements by section and ID
        sorted_reqs = sorted(
            requirements_graph.values(),
            key=lambda r: (r.source.section_id if r.source else "ZZZ", r.id)
        )
        
        for req in sorted_reqs:
            # Determine priority based on requirement type and confidence
            priority = self._determine_priority(req)
            
            # Get evaluation factor if linked
            eval_factor = None
            if req.evaluated_by:
                eval_req = requirements_graph.get(req.evaluated_by[0])
                if eval_req:
                    eval_factor = self._extract_factor_name(eval_req.text)
            
            row = ComplianceMatrixRow(
                requirement_id=req.id,
                requirement_text=req.text[:500],  # Truncate for matrix
                section_reference=req.source.section_id if req.source else "",
                section_type=req.source.document_type.value if req.source else "",
                requirement_type=req.requirement_type.value,
                related_requirements=req.references_to[:5],
                evaluation_factor=eval_factor,
                evidence_required=self._identify_evidence(req),
                priority=priority,
                risk_if_non_compliant=self._assess_risk(req),
            )
            matrix.append(row)
        
        return matrix
    
    def _determine_priority(self, req: RequirementNode) -> str:
        """Determine requirement priority"""
        # High priority: mandatory, performance, with evaluation link
        if req.requirement_type in [RequirementType.PERFORMANCE, RequirementType.QUALIFICATION]:
            if req.evaluated_by:
                return "High"
            if req.confidence == ConfidenceLevel.HIGH:
                return "High"
        
        # Medium priority: proposal instructions, deliverables
        if req.requirement_type in [RequirementType.PROPOSAL_INSTRUCTION, RequirementType.DELIVERABLE]:
            return "Medium"
        
        # Low priority: format, compliance clauses
        if req.requirement_type in [RequirementType.FORMAT, RequirementType.COMPLIANCE]:
            return "Low"
        
        return "Medium"
    
    def _identify_evidence(self, req: RequirementNode) -> List[str]:
        """Identify what evidence is needed to demonstrate compliance"""
        evidence = []
        
        text_lower = req.text.lower()
        
        # Past performance evidence
        if "past performance" in text_lower or "relevant experience" in text_lower:
            evidence.append("Past Performance References")
            evidence.append("CPARS Reports")
        
        # Certification evidence
        if "certified" in text_lower or "certification" in text_lower:
            evidence.append("Certification Documentation")
        
        # Clearance evidence
        if "clearance" in text_lower or "security" in text_lower:
            evidence.append("Security Clearance Verification")
        
        # Resume evidence
        if "key personnel" in text_lower or "qualifications" in text_lower:
            evidence.append("Resumes")
            evidence.append("Personnel Qualifications")
        
        # Plan evidence
        if "plan" in text_lower:
            evidence.append("Written Plan/Approach")
        
        # Default for performance requirements
        if not evidence and req.requirement_type == RequirementType.PERFORMANCE:
            evidence.append("Technical Approach Description")
        
        return evidence
    
    def _assess_risk(self, req: RequirementNode) -> str:
        """Assess risk if requirement is not met"""
        # Prohibitions are high risk
        if req.requirement_type == RequirementType.PROHIBITION:
            return "Disqualification"
        
        # Qualifications are often mandatory
        if req.requirement_type == RequirementType.QUALIFICATION:
            return "Proposal may be rejected as non-responsive"
        
        # Performance requirements with evaluation links
        if req.requirement_type == RequirementType.PERFORMANCE and req.evaluated_by:
            return "Significant point deduction in evaluation"
        
        # Format requirements
        if req.requirement_type == RequirementType.FORMAT:
            return "Page may not be evaluated"
        
        return "May reduce technical score"
    
    def _extract_factor_name(self, text: str) -> str:
        """Extract evaluation factor name from text"""
        text_lower = text.lower()
        
        factors = [
            ("Technical Approach", ["technical approach", "technical capability"]),
            ("Management Approach", ["management approach", "management plan"]),
            ("Past Performance", ["past performance", "relevant experience"]),
            ("Price/Cost", ["price", "cost", "pricing"]),
            ("Key Personnel", ["key personnel", "staffing", "qualifications"]),
        ]
        
        for factor_name, keywords in factors:
            if any(kw in text_lower for kw in keywords):
                return factor_name
        
        return "Other"
    
    def _estimate_coverage(
        self, 
        graph: Dict[str, RequirementNode],
        parsed_docs: Dict[str, ParsedDocument]
    ) -> float:
        """Estimate requirement extraction coverage"""
        # Count "shall" statements in source documents
        total_shall_count = 0
        for doc in parsed_docs.values():
            shall_count = doc.full_text.lower().count("shall")
            must_count = doc.full_text.lower().count("must")
            total_shall_count += shall_count + must_count
        
        # Compare to extracted requirements
        extracted_count = len(graph)
        
        if total_shall_count == 0:
            return 1.0
        
        # Rough estimate (not all "shall" are requirements, but gives baseline)
        coverage = min(1.0, extracted_count / (total_shall_count * 0.7))
        
        return round(coverage, 2)
    
    def _resolve_internal_references(
        self, 
        req: RequirementNode, 
        graph: Dict[str, RequirementNode]
    ):
        """Resolve references within a single document"""
        # Find requirements that share keywords
        for other_id, other_req in graph.items():
            if other_id == req.id:
                continue
            
            # Check for shared keywords
            shared = set(req.keywords) & set(other_req.keywords)
            if len(shared) >= 2:
                if other_id not in req.references_to:
                    req.references_to.append(other_id)
    
    def _matrix_row_to_dict(self, row: ComplianceMatrixRow) -> Dict[str, Any]:
        """Convert ComplianceMatrixRow to dictionary"""
        return {
            "requirement_id": row.requirement_id,
            "requirement_text": row.requirement_text,
            "section_reference": row.section_reference,
            "section_type": row.section_type,
            "requirement_type": row.requirement_type,
            "compliance_status": row.compliance_status,
            "response_text": row.response_text,
            "proposal_section": row.proposal_section,
            "assigned_owner": row.assigned_owner,
            "evidence_required": row.evidence_required,
            "related_requirements": row.related_requirements,
            "evaluation_factor": row.evaluation_factor,
            "priority": row.priority,
            "risk_if_non_compliant": row.risk_if_non_compliant,
            "notes": row.notes,
        }
    
    def _error_result(self, message: str, start_time: datetime) -> Dict[str, Any]:
        """Generate error result"""
        return {
            "error_state": message,
            "agent_trace_log": [{
                "timestamp": start_time.isoformat(),
                "agent_name": "enhanced_compliance_agent",
                "action": "shred_rfp_bundle",
                "input_summary": "Error",
                "output_summary": f"Error: {message}",
                "reasoning_trace": message,
            }]
        }
    
    def export_graph_to_json(
        self, 
        result: ExtractionResult, 
        output_path: str
    ) -> str:
        """Export requirements graph to JSON file"""
        data = {
            "solicitation_number": result.stats.get("solicitation_number", "UNKNOWN"),
            "extraction_date": result.completed_at,
            "statistics": result.stats,
            "coverage_estimate": result.extraction_coverage,
            "requirements": {
                req_id: req.to_dict() 
                for req_id, req in result.requirements_graph.items()
            },
            "compliance_matrix": [
                self._matrix_row_to_dict(row) 
                for row in result.compliance_matrix
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path


# Factory function for compatibility with existing codebase
def create_enhanced_compliance_agent(
    llm_client: Optional[Any] = None
) -> EnhancedComplianceAgent:
    """Factory function to create Enhanced Compliance Agent"""
    return EnhancedComplianceAgent(llm_client=llm_client)
