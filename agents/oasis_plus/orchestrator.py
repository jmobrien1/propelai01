"""
OASIS+ Orchestrator
===================

Main orchestrator that coordinates all OASIS+ components for
end-to-end proposal scoring and artifact generation.

Workflow:
1. Ingest J.P-1 Matrix → Load scoring rules
2. Upload Project Library → Parse documents, generate embeddings
3. Auto-Score Projects → Match projects to criteria
4. Hunt Evidence → Find supporting documentation
5. Optimize Selection → Choose best 5 projects
6. Generate Artifacts → Tagged PDFs + J.P-3 forms
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from decimal import Decimal

from .models import (
    OASISDomain,
    ScoringCriteria,
    Project,
    ProjectClaim,
    DocumentChunk,
    ScorecardResult,
    OptimizationConstraints,
    BusinessSize,
    DomainType,
    VerificationStatus,
    JP3FormData,
)
from .jp1_parser import JP1MatrixParser
from .evidence_hunter import EvidenceHunter, EvidenceSearchResult
from .optimizer import ProjectOptimizer, OptimizationResult
from .pdf_tagger import PDFTagger, TaggedPDF
from .form_generator import JP3FormGenerator, JP3GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class OASISProject:
    """A project in the OASIS+ proposal with all associated data"""
    project: Project
    documents: List[Dict[str, Any]] = field(default_factory=list)  # Original files
    chunks: List[DocumentChunk] = field(default_factory=list)  # Parsed chunks
    claims: List[ProjectClaim] = field(default_factory=list)  # Scoring claims
    score_by_domain: Dict[DomainType, int] = field(default_factory=dict)


@dataclass
class OASISProposal:
    """Complete OASIS+ proposal state"""
    proposal_id: str
    contractor_name: str
    contractor_cage: str = ""
    business_size: BusinessSize = BusinessSize.UNRESTRICTED

    # Target domains
    target_domains: List[DomainType] = field(default_factory=list)

    # J.P-1 Matrix (loaded scoring rules)
    domains: Dict[DomainType, OASISDomain] = field(default_factory=dict)

    # Project library
    projects: Dict[str, OASISProject] = field(default_factory=dict)

    # Optimization results per domain
    optimization_results: Dict[DomainType, OptimizationResult] = field(default_factory=dict)

    # Generated artifacts
    tagged_pdfs: List[TaggedPDF] = field(default_factory=list)
    jp3_forms: List[JP3GenerationResult] = field(default_factory=list)

    # Status tracking
    status: str = "created"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessingProgress:
    """Progress tracking for long-running operations"""
    stage: str
    progress: int  # 0-100
    message: str
    details: Optional[Dict[str, Any]] = None


class OASISOrchestrator:
    """
    Orchestrates the complete OASIS+ proposal workflow.

    Coordinates document ingestion, scoring, optimization,
    and artifact generation.
    """

    def __init__(
        self,
        embedding_function: Optional[Callable] = None,
        llm_function: Optional[Callable] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            embedding_function: Function to generate text embeddings
            llm_function: Function to call LLM for verification
            progress_callback: Callback for progress updates
        """
        self.embedding_function = embedding_function
        self.llm_function = llm_function
        self.progress_callback = progress_callback

        # Initialize components
        self.jp1_parser = JP1MatrixParser()
        self.evidence_hunter = EvidenceHunter(
            embedding_function=embedding_function,
            llm_function=llm_function,
        )
        self.optimizer = ProjectOptimizer()
        self.pdf_tagger = None  # Initialized lazily
        self.form_generator = None  # Initialized lazily

        # Active proposals
        self.proposals: Dict[str, OASISProposal] = {}

    def _report_progress(self, stage: str, progress: int, message: str, details: Dict = None):
        """Report progress to callback if available"""
        if self.progress_callback:
            self.progress_callback(ProcessingProgress(
                stage=stage,
                progress=progress,
                message=message,
                details=details,
            ))
        logger.info(f"[{stage}] {progress}% - {message}")

    # ==================== Proposal Management ====================

    def create_proposal(
        self,
        contractor_name: str,
        contractor_cage: str = "",
        business_size: BusinessSize = BusinessSize.UNRESTRICTED,
        target_domains: List[DomainType] = None,
    ) -> OASISProposal:
        """
        Create a new OASIS+ proposal.

        Args:
            contractor_name: Name of the contracting company
            contractor_cage: CAGE code
            business_size: Business size classification
            target_domains: Domains to pursue (default: all)

        Returns:
            New OASISProposal instance
        """
        proposal_id = f"OASIS-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        proposal = OASISProposal(
            proposal_id=proposal_id,
            contractor_name=contractor_name,
            contractor_cage=contractor_cage,
            business_size=business_size,
            target_domains=target_domains or list(DomainType),
        )

        self.proposals[proposal_id] = proposal
        logger.info(f"Created proposal {proposal_id} for {contractor_name}")

        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[OASISProposal]:
        """Get a proposal by ID"""
        return self.proposals.get(proposal_id)

    # ==================== J.P-1 Matrix Loading ====================

    def load_jp1_matrix(
        self,
        proposal_id: str,
        jp1_filepath: str,
    ) -> Dict[DomainType, OASISDomain]:
        """
        Load the J.P-1 Qualifications Matrix.

        Args:
            proposal_id: Proposal to load matrix for
            jp1_filepath: Path to the J.P-1 Excel file

        Returns:
            Dictionary of parsed domains
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        self._report_progress("jp1_load", 0, "Loading J.P-1 Matrix...")

        domains = self.jp1_parser.parse_file(jp1_filepath)
        proposal.domains = domains
        proposal.updated_at = datetime.now()

        # Report stats
        total_criteria = sum(len(d.criteria) for d in domains.values())
        self._report_progress(
            "jp1_load", 100,
            f"Loaded {len(domains)} domains with {total_criteria} scoring criteria",
            {"domains": [d.value for d in domains.keys()]}
        )

        return domains

    # ==================== Project Library Management ====================

    def add_project(
        self,
        proposal_id: str,
        project: Project,
        documents: List[Dict[str, Any]] = None,
    ) -> OASISProject:
        """
        Add a project to the proposal's library.

        Args:
            proposal_id: Proposal ID
            project: Project data
            documents: Associated document files

        Returns:
            OASISProject wrapper
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        oasis_project = OASISProject(
            project=project,
            documents=documents or [],
        )

        proposal.projects[project.project_id] = oasis_project
        proposal.updated_at = datetime.now()

        logger.info(f"Added project {project.project_id}: {project.title}")
        return oasis_project

    def ingest_project_documents(
        self,
        proposal_id: str,
        project_id: str,
        document_paths: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[DocumentChunk]:
        """
        Ingest and chunk documents for a project.

        Args:
            proposal_id: Proposal ID
            project_id: Project ID
            document_paths: Paths to document files
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of created DocumentChunk objects
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        oasis_project = proposal.projects.get(project_id)
        if not oasis_project:
            raise ValueError(f"Project not found: {project_id}")

        self._report_progress("ingest", 0, f"Ingesting {len(document_paths)} documents...")

        all_chunks = []
        for idx, doc_path in enumerate(document_paths):
            progress = int((idx / len(document_paths)) * 100)
            self._report_progress("ingest", progress, f"Processing {Path(doc_path).name}...")

            chunks = self._parse_and_chunk_document(
                doc_path, project_id, chunk_size, chunk_overlap
            )
            all_chunks.extend(chunks)

        # Generate embeddings if function available
        if self.embedding_function:
            self._report_progress("ingest", 90, "Generating embeddings...")
            for chunk in all_chunks:
                try:
                    chunk.embedding = self.embedding_function(chunk.content)
                except Exception as e:
                    logger.warning(f"Failed to embed chunk {chunk.chunk_id}: {e}")

        oasis_project.chunks = all_chunks
        proposal.updated_at = datetime.now()

        self._report_progress("ingest", 100, f"Ingested {len(all_chunks)} chunks")
        return all_chunks

    def _parse_and_chunk_document(
        self,
        filepath: str,
        project_id: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[DocumentChunk]:
        """Parse a document and split into chunks"""
        chunks = []

        try:
            # Try PyMuPDF for PDFs
            if filepath.lower().endswith('.pdf'):
                import fitz
                doc = fitz.open(filepath)

                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()

                    # Simple chunking by character count
                    start = 0
                    chunk_idx = 0
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        chunk_text = text[start:end]

                        if chunk_text.strip():
                            chunk = DocumentChunk(
                                chunk_id=f"{project_id}-{page_num}-{chunk_idx}",
                                document_id=Path(filepath).name,
                                project_id=project_id,
                                content=chunk_text,
                                page_number=page_num + 1,
                                chunk_index=len(chunks),
                                char_start=start,
                                char_end=end,
                            )
                            chunks.append(chunk)
                            chunk_idx += 1

                        start = end - chunk_overlap

                doc.close()

            # Handle other formats (DOCX, etc.)
            elif filepath.lower().endswith('.docx'):
                from docx import Document
                doc = Document(filepath)

                full_text = "\n".join(p.text for p in doc.paragraphs)

                start = 0
                chunk_idx = 0
                while start < len(full_text):
                    end = min(start + chunk_size, len(full_text))
                    chunk_text = full_text[start:end]

                    if chunk_text.strip():
                        chunk = DocumentChunk(
                            chunk_id=f"{project_id}-doc-{chunk_idx}",
                            document_id=Path(filepath).name,
                            project_id=project_id,
                            content=chunk_text,
                            page_number=1,  # Estimate
                            chunk_index=len(chunks),
                            char_start=start,
                            char_end=end,
                        )
                        chunks.append(chunk)
                        chunk_idx += 1

                    start = end - chunk_overlap

        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")

        return chunks

    # ==================== Scoring & Evidence ====================

    def auto_score_project(
        self,
        proposal_id: str,
        project_id: str,
        domain: DomainType,
    ) -> List[ProjectClaim]:
        """
        Automatically score a project against domain criteria.

        Args:
            proposal_id: Proposal ID
            project_id: Project ID
            domain: Domain to score against

        Returns:
            List of generated claims
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        oasis_project = proposal.projects.get(project_id)
        if not oasis_project:
            raise ValueError(f"Project not found: {project_id}")

        domain_obj = proposal.domains.get(domain)
        if not domain_obj:
            raise ValueError(f"Domain not loaded: {domain}")

        project = oasis_project.project
        chunks = oasis_project.chunks

        self._report_progress(
            "scoring", 0,
            f"Scoring {project.title} for {domain.value}..."
        )

        claims = []
        criteria_list = domain_obj.criteria

        for idx, criteria in enumerate(criteria_list):
            progress = int((idx / len(criteria_list)) * 100)
            self._report_progress(
                "scoring", progress,
                f"Checking {criteria.criteria_id}..."
            )

            # Search for evidence
            evidence_result = self.evidence_hunter.search_evidence(
                criteria, project, chunks
            )

            # Create claim if evidence found
            if evidence_result.best_match:
                claim = self.evidence_hunter.create_claim_from_evidence(
                    evidence_result, criteria
                )
                if claim:
                    claims.append(claim)
                    logger.info(
                        f"Claim created: {criteria.criteria_id} "
                        f"({claim.claimed_points} pts, "
                        f"confidence={evidence_result.confidence_score:.2f})"
                    )

        # Store claims
        oasis_project.claims.extend(claims)
        oasis_project.score_by_domain[domain] = sum(c.claimed_points for c in claims)
        proposal.updated_at = datetime.now()

        self._report_progress(
            "scoring", 100,
            f"Found {len(claims)} claims totaling {oasis_project.score_by_domain[domain]} points"
        )

        return claims

    def score_all_projects(
        self,
        proposal_id: str,
        domain: DomainType,
    ) -> Dict[str, List[ProjectClaim]]:
        """
        Score all projects in library for a domain.

        Args:
            proposal_id: Proposal ID
            domain: Domain to score

        Returns:
            Dictionary mapping project_id to claims
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        all_claims = {}
        for project_id in proposal.projects:
            claims = self.auto_score_project(proposal_id, project_id, domain)
            all_claims[project_id] = claims

        return all_claims

    # ==================== Optimization ====================

    def optimize_domain(
        self,
        proposal_id: str,
        domain: DomainType,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> OptimizationResult:
        """
        Optimize project selection for a domain.

        Args:
            proposal_id: Proposal ID
            domain: Domain to optimize
            constraints: Optional constraints

        Returns:
            OptimizationResult with selected projects
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        domain_obj = proposal.domains.get(domain)
        if not domain_obj:
            raise ValueError(f"Domain not loaded: {domain}")

        self._report_progress("optimize", 0, f"Optimizing for {domain.value}...")

        # Gather projects and claims
        projects = [op.project for op in proposal.projects.values()]
        claims = {
            pid: op.claims for pid, op in proposal.projects.items()
        }

        # Run optimization
        if constraints:
            self.optimizer.constraints = constraints

        result = self.optimizer.optimize(
            projects=projects,
            claims=claims,
            domain=domain_obj,
            business_size=proposal.business_size,
        )

        proposal.optimization_results[domain] = result
        proposal.updated_at = datetime.now()

        self._report_progress(
            "optimize", 100,
            f"Selected {len(result.selected_projects)} projects: "
            f"{result.total_score} points ({result.margin:+d} margin)",
            {
                "score": result.total_score,
                "threshold": result.threshold,
                "margin": result.margin,
                "risk": result.overall_risk,
            }
        )

        return result

    # ==================== Artifact Generation ====================

    def generate_tagged_pdfs(
        self,
        proposal_id: str,
        domain: DomainType,
        output_dir: str,
    ) -> List[TaggedPDF]:
        """
        Generate Symphony-ready tagged PDFs.

        Args:
            proposal_id: Proposal ID
            domain: Domain to generate for
            output_dir: Output directory

        Returns:
            List of TaggedPDF results
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        result = proposal.optimization_results.get(domain)
        if not result:
            raise ValueError(f"No optimization result for domain: {domain}")

        self._report_progress("artifacts", 0, "Generating tagged PDFs...")

        # Initialize tagger if needed
        if not self.pdf_tagger:
            try:
                self.pdf_tagger = PDFTagger()
            except ImportError as e:
                logger.error(f"PDF tagger not available: {e}")
                return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tagged_pdfs = []
        for idx, project in enumerate(result.selected_projects):
            progress = int((idx / len(result.selected_projects)) * 100)
            self._report_progress("artifacts", progress, f"Tagging {project.title}...")

            oasis_project = proposal.projects.get(project.project_id)
            if not oasis_project:
                continue

            # Get claims for this project
            project_claims = [
                c for c in result.all_claims
                if c.project_id == project.project_id
            ]

            # Tag each document
            for doc in oasis_project.documents:
                if not doc.get("path"):
                    continue

                input_path = doc["path"]
                output_file = output_path / f"TAGGED_{Path(input_path).name}"

                tagged = self.pdf_tagger.tag_from_search_text(
                    input_path=input_path,
                    output_path=str(output_file),
                    claims=project_claims,
                )
                tagged_pdfs.append(tagged)

        proposal.tagged_pdfs.extend(tagged_pdfs)
        proposal.updated_at = datetime.now()

        self._report_progress(
            "artifacts", 100,
            f"Generated {len(tagged_pdfs)} tagged PDFs"
        )

        return tagged_pdfs

    def generate_jp3_forms(
        self,
        proposal_id: str,
        domain: DomainType,
        output_dir: str,
    ) -> List[JP3GenerationResult]:
        """
        Generate J.P-3 verification forms.

        Args:
            proposal_id: Proposal ID
            domain: Domain to generate for
            output_dir: Output directory

        Returns:
            List of JP3GenerationResult
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        result = proposal.optimization_results.get(domain)
        if not result:
            raise ValueError(f"No optimization result for domain: {domain}")

        self._report_progress("artifacts", 0, "Generating J.P-3 forms...")

        # Initialize generator if needed
        if not self.form_generator:
            try:
                self.form_generator = JP3FormGenerator()
            except ImportError as e:
                logger.error(f"Form generator not available: {e}")
                return []

        # Find claims that need J.P-3
        projects_needing_jp3 = []
        for project in result.selected_projects:
            project_claims = [
                c for c in result.all_claims
                if c.project_id == project.project_id
                and c.status == VerificationStatus.JP3_REQUIRED
            ]
            if project_claims:
                projects_needing_jp3.append((project, project_claims))

        if not projects_needing_jp3:
            self._report_progress("artifacts", 100, "No J.P-3 forms needed")
            return []

        # Generate forms
        results = self.form_generator.batch_generate(
            projects_and_claims=projects_needing_jp3,
            domain=domain,
            output_dir=output_dir,
            contractor_name=proposal.contractor_name,
            contractor_cage=proposal.contractor_cage,
        )

        proposal.jp3_forms.extend(results)
        proposal.updated_at = datetime.now()

        self._report_progress(
            "artifacts", 100,
            f"Generated {len(results)} J.P-3 forms"
        )

        return results

    # ==================== Full Workflow ====================

    def run_full_workflow(
        self,
        proposal_id: str,
        jp1_filepath: str,
        project_documents: Dict[str, List[str]],  # project_id -> document paths
        output_dir: str,
        target_domain: DomainType = DomainType.TECHNICAL_ENGINEERING,
    ) -> Dict[str, Any]:
        """
        Run the complete OASIS+ workflow.

        Args:
            proposal_id: Proposal ID
            jp1_filepath: Path to J.P-1 Matrix
            project_documents: Mapping of project IDs to document paths
            output_dir: Output directory for artifacts
            target_domain: Domain to process

        Returns:
            Complete results dictionary
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        results = {
            "proposal_id": proposal_id,
            "domain": target_domain.value,
            "stages": {},
        }

        try:
            # Stage 1: Load J.P-1
            self._report_progress("workflow", 10, "Loading J.P-1 Matrix...")
            domains = self.load_jp1_matrix(proposal_id, jp1_filepath)
            results["stages"]["jp1_load"] = {
                "domains": len(domains),
                "criteria": sum(len(d.criteria) for d in domains.values()),
            }

            # Stage 2: Ingest documents
            self._report_progress("workflow", 30, "Ingesting project documents...")
            for project_id, doc_paths in project_documents.items():
                self.ingest_project_documents(proposal_id, project_id, doc_paths)
            results["stages"]["ingest"] = {
                "projects": len(project_documents),
            }

            # Stage 3: Score all projects
            self._report_progress("workflow", 50, "Scoring projects...")
            all_claims = self.score_all_projects(proposal_id, target_domain)
            results["stages"]["scoring"] = {
                "total_claims": sum(len(c) for c in all_claims.values()),
            }

            # Stage 4: Optimize
            self._report_progress("workflow", 70, "Optimizing project selection...")
            opt_result = self.optimize_domain(proposal_id, target_domain)
            results["stages"]["optimization"] = {
                "selected_projects": len(opt_result.selected_projects),
                "total_score": opt_result.total_score,
                "threshold": opt_result.threshold,
                "margin": opt_result.margin,
                "meets_threshold": opt_result.meets_threshold,
                "risk": opt_result.overall_risk,
            }

            # Stage 5: Generate artifacts
            self._report_progress("workflow", 85, "Generating artifacts...")
            tagged = self.generate_tagged_pdfs(proposal_id, target_domain, output_dir)
            jp3s = self.generate_jp3_forms(proposal_id, target_domain, output_dir)
            results["stages"]["artifacts"] = {
                "tagged_pdfs": len(tagged),
                "jp3_forms": len(jp3s),
            }

            results["success"] = True
            self._report_progress("workflow", 100, "Workflow complete!")

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    # ==================== Reporting ====================

    def get_scorecard(
        self,
        proposal_id: str,
        domain: DomainType,
    ) -> Optional[ScorecardResult]:
        """Get the scorecard for a domain"""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return None

        opt_result = proposal.optimization_results.get(domain)
        if not opt_result:
            return None

        return opt_result.to_scorecard()

    def get_summary(self, proposal_id: str) -> Dict[str, Any]:
        """Get a summary of proposal status"""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return {"error": "Proposal not found"}

        return {
            "proposal_id": proposal.proposal_id,
            "contractor": proposal.contractor_name,
            "business_size": proposal.business_size.value,
            "status": proposal.status,
            "domains_loaded": len(proposal.domains),
            "projects_in_library": len(proposal.projects),
            "optimization_results": {
                domain.value: {
                    "score": result.total_score,
                    "threshold": result.threshold,
                    "margin": result.margin,
                    "meets_threshold": result.meets_threshold,
                    "risk": result.overall_risk,
                }
                for domain, result in proposal.optimization_results.items()
            },
            "artifacts": {
                "tagged_pdfs": len(proposal.tagged_pdfs),
                "jp3_forms": len(proposal.jp3_forms),
            },
            "created_at": proposal.created_at.isoformat(),
            "updated_at": proposal.updated_at.isoformat(),
        }
