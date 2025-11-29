"""
PropelAI: Annotated Outline Exporter (Python Wrapper)

This module provides a Python interface to generate annotated proposal outlines.
It wraps the Node.js-based document generator for integration with the FastAPI backend.

Usage:
    from agents.enhanced_compliance.annotated_outline_exporter import AnnotatedOutlineExporter
    
    exporter = AnnotatedOutlineExporter()
    docx_bytes = exporter.export(outline_data, requirements, format_requirements)
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AnnotatedOutlineConfig:
    """Configuration for annotated outline generation"""
    rfp_title: str = "RFP Title"
    solicitation_number: str = "TBD"
    due_date: str = "TBD"
    submission_method: str = "Not Specified"
    total_pages: Optional[int] = None
    company_name: str = "[Company Name]"


class AnnotatedOutlineExporter:
    """
    Generates annotated proposal outlines in Word format.
    
    The annotated outline is the "single most important tool" for government
    proposals - it serves as the architectural blueprint for a winning response.
    
    Features:
    - Structure mirrors Section L exactly (evaluator navigation)
    - Color-coded requirements (Red=L, Blue=M, Purple=C)
    - Page allocations per section
    - Win theme & discriminator placeholders
    - Proof point guidance
    - Graphics placeholders with action caption templates
    """
    
    def __init__(self, node_script_path: Optional[str] = None):
        """
        Initialize the exporter.
        
        Args:
            node_script_path: Path to the Node.js exporter script.
                              If None, uses the default location.
        """
        if node_script_path:
            self.script_path = Path(node_script_path)
        else:
            # Default: same directory as this module
            self.script_path = Path(__file__).parent / "annotated_outline_exporter.js"
        
        # Verify Node.js is available
        self._verify_node()
    
    def _verify_node(self):
        """Verify Node.js and docx package are available"""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Node.js not available")
        except FileNotFoundError:
            raise RuntimeError("Node.js not installed")
    
    def export(
        self,
        outline_data: Dict[str, Any],
        requirements: Optional[List[Dict[str, Any]]] = None,
        format_requirements: Optional[Dict[str, Any]] = None,
        config: Optional[AnnotatedOutlineConfig] = None
    ) -> bytes:
        """
        Generate annotated outline document.
        
        Args:
            outline_data: Proposal outline from SmartOutlineGenerator.to_json()
            requirements: List of requirements from CTM extraction
            format_requirements: Document format requirements (font, margins, etc.)
            config: Additional configuration options
            
        Returns:
            bytes: The generated Word document as bytes
        """
        # Build the input data for the Node.js script
        input_data = self._build_input_data(
            outline_data,
            requirements or [],
            format_requirements or {},
            config or AnnotatedOutlineConfig()
        )
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(input_data, f)
            input_path = f.name
        
        output_path = tempfile.mktemp(suffix='.docx')
        
        try:
            # Run the Node.js exporter
            result = subprocess.run(
                ["node", str(self.script_path), input_path, output_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Log stdout from Node.js (includes our debug console.log statements)
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"[EXPORTER] {line}")
            
            if result.returncode != 0:
                error_msg = f"Document generation failed: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout}"
                raise RuntimeError(error_msg)
            
            # Read the generated document
            with open(output_path, 'rb') as f:
                return f.read()
                
        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def export_to_file(
        self,
        filepath: str,
        outline_data: Dict[str, Any],
        requirements: Optional[List[Dict[str, Any]]] = None,
        format_requirements: Optional[Dict[str, Any]] = None,
        config: Optional[AnnotatedOutlineConfig] = None
    ) -> str:
        """
        Generate annotated outline and save to file.
        
        Args:
            filepath: Output file path
            outline_data: Proposal outline data
            requirements: List of requirements from CTM
            format_requirements: Format requirements
            config: Configuration options
            
        Returns:
            str: Path to the generated file
        """
        doc_bytes = self.export(outline_data, requirements, format_requirements, config)
        
        with open(filepath, 'wb') as f:
            f.write(doc_bytes)
        
        return filepath
    
    def _build_input_data(
        self,
        outline_data: Dict[str, Any],
        requirements: List[Dict[str, Any]],
        format_requirements: Dict[str, Any],
        config: AnnotatedOutlineConfig
    ) -> Dict[str, Any]:
        """Build the complete input data for the Node.js exporter"""
        
        # Start with outline data
        data = {
            "rfpTitle": config.rfp_title,
            "solicitationNumber": config.solicitation_number,
            "dueDate": config.due_date,
            "submissionMethod": config.submission_method,
            "totalPages": config.total_pages,
            "companyName": config.company_name,
            "formatRequirements": format_requirements,
            "requirements": requirements
        }
        
        # Merge outline data
        if "volumes" in outline_data:
            data["volumes"] = outline_data["volumes"]
        
        if "evaluation_factors" in outline_data:
            data["evaluationFactors"] = outline_data["evaluation_factors"]
        elif "eval_factors" in outline_data:
            data["evaluationFactors"] = outline_data["eval_factors"]
        
        if "format_requirements" in outline_data:
            # Merge with any explicit format requirements
            outline_fmt = outline_data["format_requirements"]
            for key, value in outline_fmt.items():
                if value and key not in format_requirements:
                    data["formatRequirements"][key] = value
        
        if "submission" in outline_data:
            sub = outline_data["submission"]
            if sub.get("due_date") and data["dueDate"] == "TBD":
                data["dueDate"] = sub["due_date"]
            if sub.get("method") and data["submissionMethod"] == "Not Specified":
                data["submissionMethod"] = sub["method"]
        
        if "total_pages" in outline_data and outline_data["total_pages"]:
            data["totalPages"] = outline_data["total_pages"]
        
        return data


def generate_annotated_outline(
    outline_data: Dict[str, Any],
    requirements: List[Dict[str, Any]] = None,
    rfp_title: str = "RFP",
    solicitation_number: str = "TBD",
    due_date: str = "TBD",
    company_name: str = "[Company Name]"
) -> bytes:
    """
    Convenience function to generate an annotated outline.
    
    Args:
        outline_data: Outline from SmartOutlineGenerator
        requirements: Requirements from CTM extraction
        rfp_title: Title of the RFP
        solicitation_number: Solicitation number
        due_date: Proposal due date
        company_name: Proposing company name
        
    Returns:
        bytes: Generated Word document
    """
    config = AnnotatedOutlineConfig(
        rfp_title=rfp_title,
        solicitation_number=solicitation_number,
        due_date=due_date,
        company_name=company_name
    )
    
    exporter = AnnotatedOutlineExporter()
    return exporter.export(
        outline_data,
        requirements or [],
        outline_data.get("format_requirements", {}),
        config
    )


# For testing
if __name__ == "__main__":
    # Test with sample data
    test_outline = {
        "volumes": [
            {
                "id": "vol-1",
                "name": "Technical Proposal",
                "type": "technical",
                "page_limit": 25,
                "sections": [
                    {
                        "id": "1.0",
                        "name": "Executive Summary",
                        "page_limit": 2,
                        "requirements": ["Provide overview", "Demonstrate understanding"]
                    }
                ]
            }
        ],
        "evaluation_factors": [
            {"id": "1", "name": "Technical", "weight": "Most Important"}
        ]
    }
    
    try:
        exporter = AnnotatedOutlineExporter()
        doc = exporter.export(
            test_outline,
            config=AnnotatedOutlineConfig(
                rfp_title="Test RFP",
                company_name="Test Company"
            )
        )
        print(f"Generated document: {len(doc)} bytes")
    except Exception as e:
        print(f"Error: {e}")
