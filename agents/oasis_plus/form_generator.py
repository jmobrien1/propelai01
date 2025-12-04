"""
J.P-3 Form Generator
====================

Generates pre-filled J.P-3 Project Verification Forms for OASIS+ submissions.

When FPDS data doesn't fully validate a claim (e.g., surge capability,
integration of subsystems), offerors must submit a J.P-3 form signed
by the Contracting Officer.

This module auto-populates the form fields, leaving only signature
fields blank for the government client.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from decimal import Decimal

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import fitz  # PyMuPDF for form filling
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .models import Project, JP3FormData, ProjectClaim, DomainType

logger = logging.getLogger(__name__)


@dataclass
class JP3GenerationResult:
    """Result of J.P-3 form generation"""
    output_path: str
    project_id: str
    project_title: str
    claims_included: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


class JP3FormGenerator:
    """
    Generates J.P-3 Project Verification Forms.

    Can either:
    1. Fill an existing J.P-3 PDF template
    2. Generate a new PDF from scratch matching the official format
    """

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the form generator.

        Args:
            template_path: Path to official J.P-3 PDF template (optional)
        """
        self.template_path = template_path

        if template_path and not PYMUPDF_AVAILABLE:
            logger.warning(
                "PyMuPDF not available for template filling. "
                "Will generate forms from scratch."
            )

        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "ReportLab is required for J.P-3 generation. "
                "Install with: pip install reportlab"
            )

    def generate_from_project(
        self,
        project: Project,
        claims: List[ProjectClaim],
        domain: DomainType,
        output_path: str,
        contractor_name: str = "",
        contractor_cage: str = "",
    ) -> JP3GenerationResult:
        """
        Generate a J.P-3 form for a project.

        Args:
            project: The project to create form for
            claims: Claims that require J.P-3 verification
            domain: The OASIS+ domain
            output_path: Where to save the generated PDF
            contractor_name: Name of contracting company
            contractor_cage: CAGE code

        Returns:
            JP3GenerationResult with generation status
        """
        result = JP3GenerationResult(
            output_path=output_path,
            project_id=project.project_id,
            project_title=project.title,
        )

        try:
            # Create form data from project
            form_data = self._create_form_data(
                project, claims, contractor_name, contractor_cage
            )

            # Generate the PDF
            if self.template_path and PYMUPDF_AVAILABLE:
                self._fill_template(form_data, output_path)
            else:
                self._generate_pdf(form_data, domain, output_path)

            result.claims_included = [c.criteria_id for c in claims]
            logger.info(f"Generated J.P-3 form: {output_path}")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"J.P-3 generation failed: {e}")

        return result

    def _create_form_data(
        self,
        project: Project,
        claims: List[ProjectClaim],
        contractor_name: str,
        contractor_cage: str,
    ) -> JP3FormData:
        """Create form data structure from project and claims"""
        # Calculate AAV
        aav = project.calculate_aav()

        # Build claims description
        claims_desc = "\n".join([
            f"- {c.criteria_id}: {c.evidence_snippet[:100]}..."
            if c.evidence_snippet else f"- {c.criteria_id}"
            for c in claims
        ])

        # Build relevance statement
        relevance = self._generate_relevance_statement(project, claims)

        return JP3FormData(
            project_title=project.title,
            contract_number=project.contract_number,
            task_order_number=project.task_order_number,
            contractor_name=contractor_name,
            contractor_cage_code=contractor_cage,
            start_date=project.start_date,
            end_date=project.end_date,
            total_obligated_value=project.total_obligated_amount,
            average_annual_value=aav,
            naics_code=project.naics_code,
            psc_code=project.psc_code,
            relevance_statement=relevance,
            claims_description=claims_desc,
        )

    def _generate_relevance_statement(
        self,
        project: Project,
        claims: List[ProjectClaim],
    ) -> str:
        """Generate a relevance statement for the form"""
        lines = [
            f"Project '{project.title}' demonstrates relevant experience as follows:",
            "",
            f"Contract Number: {project.contract_number}",
            f"Client Agency: {project.client_agency}",
            f"Period of Performance: {project.start_date} to {project.end_date or 'Present'}",
            "",
            "Scope Summary:",
            project.scope_description[:500] if project.scope_description else "See attached documentation.",
            "",
            "This project supports the following OASIS+ qualification claims:",
        ]

        for claim in claims:
            lines.append(f"  • {claim.criteria_id}")

        return "\n".join(lines)

    def _generate_pdf(
        self,
        form_data: JP3FormData,
        domain: DomainType,
        output_path: str,
    ):
        """Generate a J.P-3 PDF from scratch"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=14,
            alignment=1,  # Center
            spaceAfter=20,
        )
        story.append(Paragraph(
            "ATTACHMENT J.P-3<br/>PROJECT VERIFICATION FORM",
            title_style
        ))

        # Domain
        story.append(Paragraph(
            f"<b>Domain:</b> {domain.value.replace('_', ' ').title()}",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))

        # Section 1: Contractor Information
        story.append(Paragraph("<b>SECTION 1: CONTRACTOR INFORMATION</b>", styles['Heading2']))
        story.append(Spacer(1, 6))

        contractor_data = [
            ["Contractor Name:", form_data.contractor_name or "_" * 40],
            ["CAGE Code:", form_data.contractor_cage_code or "_" * 20],
            ["DUNS Number:", form_data.contractor_duns or "_" * 20],
        ]
        contractor_table = Table(contractor_data, colWidths=[2*inch, 4*inch])
        contractor_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(contractor_table)
        story.append(Spacer(1, 12))

        # Section 2: Project Information
        story.append(Paragraph("<b>SECTION 2: PROJECT INFORMATION</b>", styles['Heading2']))
        story.append(Spacer(1, 6))

        project_data = [
            ["Project Title:", form_data.project_title],
            ["Contract Number:", form_data.contract_number],
            ["Task Order Number:", form_data.task_order_number or "N/A"],
            ["NAICS Code:", form_data.naics_code],
            ["PSC Code:", form_data.psc_code],
        ]
        project_table = Table(project_data, colWidths=[2*inch, 4*inch])
        project_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(project_table)
        story.append(Spacer(1, 12))

        # Section 3: Period of Performance and Value
        story.append(Paragraph("<b>SECTION 3: PERIOD OF PERFORMANCE AND VALUE</b>", styles['Heading2']))
        story.append(Spacer(1, 6))

        start_str = form_data.start_date.strftime("%m/%d/%Y") if form_data.start_date else "_" * 15
        end_str = form_data.end_date.strftime("%m/%d/%Y") if form_data.end_date else "Present"

        pop_data = [
            ["Start Date:", start_str, "End Date:", end_str],
            ["Total Obligated Value:", f"${form_data.total_obligated_value:,.2f}", "", ""],
            ["Average Annual Value (AAV):", f"${form_data.average_annual_value:,.2f}", "", ""],
        ]
        pop_table = Table(pop_data, colWidths=[2*inch, 2*inch, 1*inch, 1.5*inch])
        pop_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(pop_table)

        # AAV Formula note
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "<i>Note: AAV = (Total Obligated Value ÷ Days of Performance) × 366</i>",
            styles['Normal']
        ))
        story.append(Spacer(1, 12))

        # Section 4: Relevance Statement
        story.append(Paragraph("<b>SECTION 4: RELEVANCE STATEMENT</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "Describe how this project is relevant to the OASIS+ domain and "
            "supports the qualification claims listed below:",
            styles['Normal']
        ))
        story.append(Spacer(1, 6))

        # Relevance text box
        relevance_text = form_data.relevance_statement.replace("\n", "<br/>")
        relevance_style = ParagraphStyle(
            'Relevance',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            borderWidth=1,
            borderColor=colors.black,
            borderPadding=6,
        )
        story.append(Paragraph(relevance_text, relevance_style))
        story.append(Spacer(1, 12))

        # Section 5: Claims Being Verified
        story.append(Paragraph("<b>SECTION 5: CLAIMS BEING VERIFIED</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        claims_text = form_data.claims_description.replace("\n", "<br/>")
        story.append(Paragraph(claims_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Section 6: Government Verification
        story.append(Paragraph("<b>SECTION 6: GOVERNMENT VERIFICATION</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "I, the undersigned Contracting Officer, verify that the information "
            "provided above is accurate based on contract records.",
            styles['Normal']
        ))
        story.append(Spacer(1, 20))

        # Signature block
        sig_data = [
            ["Contracting Officer Name:", "_" * 35],
            ["", ""],
            ["Contracting Officer Signature:", "_" * 35],
            ["", ""],
            ["Date:", "_" * 20],
            ["", ""],
            ["Email:", "_" * 35],
            ["", ""],
            ["Phone:", "_" * 25],
        ]
        sig_table = Table(sig_data, colWidths=[2.5*inch, 4*inch])
        sig_table.setStyle(TableStyle([
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(sig_table)

        # Build the PDF
        doc.build(story)

    def _fill_template(
        self,
        form_data: JP3FormData,
        output_path: str,
    ):
        """Fill an existing J.P-3 PDF template"""
        doc = fitz.open(self.template_path)

        # Field mappings (template field name -> form data attribute)
        field_mappings = {
            "contractor_name": form_data.contractor_name,
            "cage_code": form_data.contractor_cage_code,
            "duns": form_data.contractor_duns,
            "project_title": form_data.project_title,
            "contract_number": form_data.contract_number,
            "task_order": form_data.task_order_number or "",
            "naics": form_data.naics_code,
            "psc": form_data.psc_code,
            "start_date": form_data.start_date.strftime("%m/%d/%Y") if form_data.start_date else "",
            "end_date": form_data.end_date.strftime("%m/%d/%Y") if form_data.end_date else "Present",
            "total_value": f"${form_data.total_obligated_value:,.2f}",
            "aav": f"${form_data.average_annual_value:,.2f}",
            "relevance": form_data.relevance_statement,
            "claims": form_data.claims_description,
        }

        # Try to fill form fields
        for page in doc:
            for field_name, value in field_mappings.items():
                # Look for widget with matching name
                for widget in page.widgets():
                    if widget.field_name and field_name.lower() in widget.field_name.lower():
                        widget.field_value = str(value)
                        widget.update()

        doc.save(output_path)
        doc.close()

    def batch_generate(
        self,
        projects_and_claims: List[tuple],  # List of (Project, List[ProjectClaim])
        domain: DomainType,
        output_dir: str,
        contractor_name: str = "",
        contractor_cage: str = "",
    ) -> List[JP3GenerationResult]:
        """
        Generate J.P-3 forms for multiple projects.

        Args:
            projects_and_claims: List of (project, claims) tuples
            domain: The OASIS+ domain
            output_dir: Directory to save generated PDFs
            contractor_name: Name of contracting company
            contractor_cage: CAGE code

        Returns:
            List of JP3GenerationResult for each project
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for project, claims in projects_and_claims:
            # Filter to claims that need J.P-3
            jp3_claims = [c for c in claims if c.status.value == "jp3_required"]

            if not jp3_claims:
                logger.debug(f"No J.P-3 claims for project {project.project_id}")
                continue

            # Generate filename
            safe_title = "".join(
                c if c.isalnum() or c in "- _" else "_"
                for c in project.title
            )[:50]
            filename = f"JP3_{safe_title}_{project.contract_number}.pdf"
            file_path = str(output_path / filename)

            result = self.generate_from_project(
                project=project,
                claims=jp3_claims,
                domain=domain,
                output_path=file_path,
                contractor_name=contractor_name,
                contractor_cage=contractor_cage,
            )
            results.append(result)

        return results


def generate_jp3_form(
    project: Project,
    claims: List[ProjectClaim],
    domain: DomainType,
    output_path: str,
    contractor_name: str = "",
    contractor_cage: str = "",
    template_path: Optional[str] = None,
) -> JP3GenerationResult:
    """
    Convenience function to generate a single J.P-3 form.

    Args:
        project: The project to create form for
        claims: Claims that require J.P-3 verification
        domain: The OASIS+ domain
        output_path: Where to save the generated PDF
        contractor_name: Name of contracting company
        contractor_cage: CAGE code
        template_path: Optional path to J.P-3 PDF template

    Returns:
        JP3GenerationResult with generation status
    """
    generator = JP3FormGenerator(template_path)
    return generator.generate_from_project(
        project=project,
        claims=claims,
        domain=domain,
        output_path=output_path,
        contractor_name=contractor_name,
        contractor_cage=contractor_cage,
    )
