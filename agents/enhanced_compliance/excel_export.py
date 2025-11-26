"""
PropelAI Cycle 5: Excel Exporter
Export compliance matrix to Excel for proposal teams

Creates a usable, formatted Excel workbook with:
- Requirements summary dashboard
- Full compliance matrix
- By-section breakdown
- Priority-sorted view
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import os

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Fill, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

from .models import (
    RequirementNode, RequirementType, ConfidenceLevel,
    ComplianceMatrixRow, ExtractionResult
)


class ExcelExporter:
    """
    Export requirements and compliance matrix to Excel
    
    Creates proposal-team-ready workbooks with:
    - Color-coded priority
    - Dropdown menus for status
    - Auto-sized columns
    - Summary statistics
    """
    
    # Color scheme
    COLORS = {
        "header": "1F4E79",      # Dark blue
        "high_priority": "F4CCCC",    # Light red
        "medium_priority": "FFF2CC",  # Light yellow
        "low_priority": "D9EAD3",     # Light green
        "section_c": "CFE2F3",   # Light blue
        "section_l": "D9D2E9",   # Light purple
        "section_m": "FCE5CD",   # Light orange
    }
    
    # Compliance status options
    STATUS_OPTIONS = [
        "Not Started",
        "In Progress", 
        "Compliant",
        "Partial",
        "Non-Compliant",
        "N/A",
        "Needs Clarification"
    ]
    
    def __init__(self):
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl")
    
    def export_compliance_matrix(
        self,
        result: ExtractionResult,
        output_path: str,
        solicitation_number: str = "",
        title: str = ""
    ) -> str:
        """
        Export extraction result to Excel workbook
        
        Args:
            result: ExtractionResult from EnhancedComplianceAgent
            output_path: Path for output .xlsx file
            solicitation_number: RFP/solicitation number
            title: Contract title
            
        Returns:
            Path to created file
        """
        wb = Workbook()
        
        # Sheet 1: Summary Dashboard
        self._create_summary_sheet(wb, result, solicitation_number, title)
        
        # Sheet 2: Full Compliance Matrix
        self._create_matrix_sheet(wb, result)
        
        # Sheet 3: High Priority Items
        self._create_priority_sheet(wb, result, "High")
        
        # Sheet 4: By Section (C, L, M breakdown)
        self._create_section_sheet(wb, result)
        
        # Sheet 5: Requirements Graph Data (for advanced users)
        self._create_graph_sheet(wb, result)
        
        # Save workbook
        wb.save(output_path)
        
        return output_path
    
    def _create_summary_sheet(
        self, 
        wb: Workbook, 
        result: ExtractionResult,
        solicitation_number: str,
        title: str
    ):
        """Create summary dashboard sheet"""
        ws = wb.active
        ws.title = "Summary"
        
        # Header styling
        header_font = Font(bold=True, color="FFFFFF", size=14)
        header_fill = PatternFill(start_color=self.COLORS["header"], 
                                  end_color=self.COLORS["header"], 
                                  fill_type="solid")
        
        # Title section
        ws.merge_cells('A1:F1')
        ws['A1'] = "PropelAI Compliance Matrix"
        ws['A1'].font = Font(bold=True, size=18)
        
        ws.merge_cells('A2:F2')
        ws['A2'] = f"Solicitation: {solicitation_number}" if solicitation_number else "Solicitation: [Not Specified]"
        ws['A2'].font = Font(size=12)
        
        ws.merge_cells('A3:F3')
        ws['A3'] = f"Title: {title}" if title else ""
        
        ws['A4'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ws['A5'] = ""
        
        # Statistics section
        row = 7
        ws[f'A{row}'] = "EXTRACTION STATISTICS"
        ws[f'A{row}'].font = Font(bold=True, size=14)
        
        # Safely get stats with defaults
        cross_ref_count = getattr(result, 'cross_reference_count', 0) or 0
        extraction_coverage = getattr(result, 'extraction_coverage', 0.0) or 0.0
        duration = getattr(result, 'duration_seconds', 0.0) or 0.0
        stats_dict = getattr(result, 'stats', {}) or {}
        req_graph = getattr(result, 'requirements_graph', {}) or {}
        
        stats = [
            ("Total Requirements", len(req_graph)),
            ("Cross-References", cross_ref_count),
            ("Documents Processed", stats_dict.get('documents_processed', 0)),
            ("Pages Analyzed", stats_dict.get('total_pages', 0)),
            ("Coverage Estimate", f"{extraction_coverage * 100:.0f}%"),
            ("Processing Time", f"{duration:.1f}s"),
        ]
        
        row += 1
        for label, value in stats:
            ws[f'A{row}'] = label
            ws[f'B{row}'] = value
            ws[f'A{row}'].font = Font(bold=True)
            row += 1
        
        # By Type breakdown
        row += 2
        ws[f'A{row}'] = "BY REQUIREMENT TYPE"
        ws[f'A{row}'].font = Font(bold=True, size=14)
        row += 1
        
        for req_type, count in sorted(result.stats.get('by_type', {}).items()):
            ws[f'A{row}'] = req_type.replace('_', ' ').title()
            ws[f'B{row}'] = count
            row += 1
        
        # Priority breakdown
        row += 2
        ws[f'A{row}'] = "BY PRIORITY"
        ws[f'A{row}'].font = Font(bold=True, size=14)
        row += 1
        
        priority_counts = {"High": 0, "Medium": 0, "Low": 0}
        for matrix_row in result.compliance_matrix:
            priority_counts[matrix_row.priority] = priority_counts.get(matrix_row.priority, 0) + 1
        
        for priority, count in priority_counts.items():
            ws[f'A{row}'] = priority
            ws[f'B{row}'] = count
            
            # Color code
            if priority == "High":
                ws[f'A{row}'].fill = PatternFill(start_color=self.COLORS["high_priority"],
                                                  end_color=self.COLORS["high_priority"],
                                                  fill_type="solid")
            elif priority == "Medium":
                ws[f'A{row}'].fill = PatternFill(start_color=self.COLORS["medium_priority"],
                                                  end_color=self.COLORS["medium_priority"],
                                                  fill_type="solid")
            else:
                ws[f'A{row}'].fill = PatternFill(start_color=self.COLORS["low_priority"],
                                                  end_color=self.COLORS["low_priority"],
                                                  fill_type="solid")
            row += 1
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20
    
    def _create_matrix_sheet(self, wb: Workbook, result: ExtractionResult):
        """Create main compliance matrix sheet"""
        ws = wb.create_sheet("Compliance Matrix")
        
        # Headers
        headers = [
            "ID", "Requirement Text", "Section", "Type", "Priority",
            "Status", "Response", "Proposal Section", "Owner",
            "Evidence Required", "Evaluation Factor", "Risk", "Notes"
        ]
        
        header_fill = PatternFill(start_color=self.COLORS["header"],
                                  end_color=self.COLORS["header"],
                                  fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', wrap_text=True)
        
        # Data rows
        for row_idx, matrix_row in enumerate(result.compliance_matrix, 2):
            ws.cell(row=row_idx, column=1, value=matrix_row.requirement_id)
            
            # Truncate long text for readability
            text = matrix_row.requirement_text
            if len(text) > 500:
                text = text[:497] + "..."
            ws.cell(row=row_idx, column=2, value=text)
            
            ws.cell(row=row_idx, column=3, value=matrix_row.section_reference)
            ws.cell(row=row_idx, column=4, value=matrix_row.requirement_type.replace('_', ' ').title())
            ws.cell(row=row_idx, column=5, value=matrix_row.priority)
            ws.cell(row=row_idx, column=6, value=matrix_row.compliance_status)
            ws.cell(row=row_idx, column=7, value=matrix_row.response_text)
            ws.cell(row=row_idx, column=8, value=matrix_row.proposal_section)
            ws.cell(row=row_idx, column=9, value=matrix_row.assigned_owner)
            ws.cell(row=row_idx, column=10, value=", ".join(matrix_row.evidence_required))
            ws.cell(row=row_idx, column=11, value=matrix_row.evaluation_factor or "")
            ws.cell(row=row_idx, column=12, value=matrix_row.risk_if_non_compliant)
            ws.cell(row=row_idx, column=13, value=matrix_row.notes)
            
            # Priority color coding
            priority_cell = ws.cell(row=row_idx, column=5)
            if matrix_row.priority == "High":
                priority_cell.fill = PatternFill(start_color=self.COLORS["high_priority"],
                                                  end_color=self.COLORS["high_priority"],
                                                  fill_type="solid")
            elif matrix_row.priority == "Medium":
                priority_cell.fill = PatternFill(start_color=self.COLORS["medium_priority"],
                                                  end_color=self.COLORS["medium_priority"],
                                                  fill_type="solid")
            else:
                priority_cell.fill = PatternFill(start_color=self.COLORS["low_priority"],
                                                  end_color=self.COLORS["low_priority"],
                                                  fill_type="solid")
            
            # Wrap text in requirement column
            ws.cell(row=row_idx, column=2).alignment = Alignment(wrap_text=True, vertical='top')
        
        # Add dropdown for Status column
        dv = DataValidation(type="list", formula1=f'"{",".join(self.STATUS_OPTIONS)}"', allow_blank=True)
        dv.error = "Please select from the list"
        dv.errorTitle = "Invalid Status"
        ws.add_data_validation(dv)
        dv.add(f'F2:F{len(result.compliance_matrix) + 1}')
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 60
        ws.column_dimensions['C'].width = 12
        ws.column_dimensions['D'].width = 18
        ws.column_dimensions['E'].width = 10
        ws.column_dimensions['F'].width = 15
        ws.column_dimensions['G'].width = 40
        ws.column_dimensions['H'].width = 15
        ws.column_dimensions['I'].width = 15
        ws.column_dimensions['J'].width = 30
        ws.column_dimensions['K'].width = 20
        ws.column_dimensions['L'].width = 30
        ws.column_dimensions['M'].width = 30
        
        # Freeze header row
        ws.freeze_panes = 'A2'
    
    def _create_priority_sheet(self, wb: Workbook, result: ExtractionResult, priority: str):
        """Create sheet with priority-filtered items"""
        ws = wb.create_sheet(f"{priority} Priority")
        
        # Filter to priority
        filtered = [r for r in result.compliance_matrix if r.priority == priority]
        
        # Headers
        headers = ["ID", "Requirement", "Section", "Type", "Status", "Owner", "Evidence", "Risk"]
        
        header_fill = PatternFill(start_color=self.COLORS["high_priority"] if priority == "High" else self.COLORS["header"],
                                  end_color=self.COLORS["high_priority"] if priority == "High" else self.COLORS["header"],
                                  fill_type="solid")
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = Font(bold=True, color="000000" if priority == "High" else "FFFFFF")
        
        # Data
        for row_idx, matrix_row in enumerate(filtered, 2):
            ws.cell(row=row_idx, column=1, value=matrix_row.requirement_id)
            
            text = matrix_row.requirement_text
            if len(text) > 300:
                text = text[:297] + "..."
            ws.cell(row=row_idx, column=2, value=text)
            
            ws.cell(row=row_idx, column=3, value=matrix_row.section_reference)
            ws.cell(row=row_idx, column=4, value=matrix_row.requirement_type.replace('_', ' ').title())
            ws.cell(row=row_idx, column=5, value=matrix_row.compliance_status)
            ws.cell(row=row_idx, column=6, value=matrix_row.assigned_owner)
            ws.cell(row=row_idx, column=7, value=", ".join(matrix_row.evidence_required))
            ws.cell(row=row_idx, column=8, value=matrix_row.risk_if_non_compliant)
        
        # Column widths
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 60
        ws.column_dimensions['C'].width = 12
        ws.column_dimensions['D'].width = 18
        ws.column_dimensions['E'].width = 15
        ws.column_dimensions['F'].width = 15
        ws.column_dimensions['G'].width = 30
        ws.column_dimensions['H'].width = 30
        
        ws.freeze_panes = 'A2'
    
    def _create_section_sheet(self, wb: Workbook, result: ExtractionResult):
        """Create sheet organized by section"""
        ws = wb.create_sheet("By Section")
        
        # Group by section
        by_section: Dict[str, List[ComplianceMatrixRow]] = {}
        for matrix_row in result.compliance_matrix:
            section = matrix_row.section_reference[:1] if matrix_row.section_reference else "Other"
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(matrix_row)
        
        row = 1
        
        for section in sorted(by_section.keys()):
            items = by_section[section]
            
            # Section header
            ws.merge_cells(f'A{row}:E{row}')
            ws[f'A{row}'] = f"Section {section} ({len(items)} requirements)"
            ws[f'A{row}'].font = Font(bold=True, size=14)
            
            # Color by section type
            if section == "C":
                ws[f'A{row}'].fill = PatternFill(start_color=self.COLORS["section_c"],
                                                  end_color=self.COLORS["section_c"],
                                                  fill_type="solid")
            elif section == "L":
                ws[f'A{row}'].fill = PatternFill(start_color=self.COLORS["section_l"],
                                                  end_color=self.COLORS["section_l"],
                                                  fill_type="solid")
            elif section == "M":
                ws[f'A{row}'].fill = PatternFill(start_color=self.COLORS["section_m"],
                                                  end_color=self.COLORS["section_m"],
                                                  fill_type="solid")
            
            row += 1
            
            # Column headers
            for col, header in enumerate(["ID", "Requirement", "Type", "Priority", "Status"], 1):
                ws.cell(row=row, column=col, value=header)
                ws.cell(row=row, column=col).font = Font(bold=True)
            row += 1
            
            # Items
            for item in items:
                ws.cell(row=row, column=1, value=item.requirement_id)
                
                text = item.requirement_text
                if len(text) > 200:
                    text = text[:197] + "..."
                ws.cell(row=row, column=2, value=text)
                
                ws.cell(row=row, column=3, value=item.requirement_type.replace('_', ' ').title())
                ws.cell(row=row, column=4, value=item.priority)
                ws.cell(row=row, column=5, value=item.compliance_status)
                row += 1
            
            row += 1  # Blank row between sections
        
        # Column widths
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 60
        ws.column_dimensions['C'].width = 18
        ws.column_dimensions['D'].width = 10
        ws.column_dimensions['E'].width = 15
    
    def _create_graph_sheet(self, wb: Workbook, result: ExtractionResult):
        """Create sheet with graph relationship data"""
        ws = wb.create_sheet("Graph Data")
        
        # Headers
        headers = ["ID", "Type", "Section", "Confidence", "References To", "Evaluated By", "Instructed By"]
        
        header_fill = PatternFill(start_color=self.COLORS["header"],
                                  end_color=self.COLORS["header"],
                                  fill_type="solid")
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = Font(bold=True, color="FFFFFF")
        
        # Data
        row = 2
        for req_id, req in result.requirements_graph.items():
            ws.cell(row=row, column=1, value=req_id)
            ws.cell(row=row, column=2, value=req.requirement_type.value)
            ws.cell(row=row, column=3, value=req.source.section_id if req.source else "")
            ws.cell(row=row, column=4, value=req.confidence.value)
            ws.cell(row=row, column=5, value=", ".join(req.references_to[:5]))
            ws.cell(row=row, column=6, value=", ".join(req.evaluated_by[:3]))
            ws.cell(row=row, column=7, value=", ".join(req.instructed_by[:3]))
            row += 1
        
        # Column widths
        for col in range(1, 8):
            ws.column_dimensions[get_column_letter(col)].width = 20
        
        ws.freeze_panes = 'A2'


def export_to_excel(
    result: ExtractionResult,
    output_path: str,
    solicitation_number: str = "",
    title: str = ""
) -> str:
    """
    Convenience function to export results to Excel
    
    Args:
        result: ExtractionResult from EnhancedComplianceAgent
        output_path: Path for output .xlsx file
        solicitation_number: RFP/solicitation number
        title: Contract title
        
    Returns:
        Path to created file
    """
    exporter = ExcelExporter()
    return exporter.export_compliance_matrix(result, output_path, solicitation_number, title)
