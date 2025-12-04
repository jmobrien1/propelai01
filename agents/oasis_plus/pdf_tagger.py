"""
PDF Tagger for Symphony Portal
==============================

Creates Symphony-ready PDF files with evidence annotations.

The Symphony portal requires offerors to "tag" evidence within PDF files
using highlights and comments to facilitate government validation.
Evaluators use these tags to quickly locate evidence supporting claims.

This module uses PyMuPDF (fitz) to programmatically inject:
- Highlight annotations over evidence text
- Sticky note comments with claim references
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .models import ProjectClaim, VerificationStatus

logger = logging.getLogger(__name__)


# Annotation colors (RGB values 0-1)
COLORS = {
    "highlight_yellow": (1.0, 1.0, 0.0),      # Standard yellow highlight
    "highlight_green": (0.6, 1.0, 0.6),       # Verified evidence
    "highlight_orange": (1.0, 0.8, 0.4),      # Pending verification
    "sticky_note": (1.0, 0.85, 0.0),          # Yellow sticky note
}


@dataclass
class AnnotationSpec:
    """Specification for a single annotation to add"""
    page_number: int  # 0-indexed
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    annotation_type: str  # "highlight" or "sticky_note"
    content: str  # Text content for sticky note
    criteria_id: str  # Reference to scoring criteria
    claim_id: str
    color: Tuple[float, float, float] = COLORS["highlight_yellow"]


@dataclass
class TaggedPDF:
    """Result of PDF tagging operation"""
    original_path: str
    output_path: str
    annotations_added: int = 0
    pages_modified: List[int] = field(default_factory=list)
    claims_tagged: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class PDFTagger:
    """
    Tags PDF documents with evidence annotations for Symphony submission.

    Creates highlighted regions and sticky notes pointing evaluators
    to specific evidence supporting OASIS+ scoring claims.
    """

    def __init__(
        self,
        highlight_color: Tuple[float, float, float] = COLORS["highlight_yellow"],
        include_sticky_notes: bool = True,
        sticky_note_prefix: str = "OASIS+ Evidence: ",
    ):
        """
        Initialize the PDF tagger.

        Args:
            highlight_color: RGB color for highlights (0-1 range)
            include_sticky_notes: Whether to add sticky note comments
            sticky_note_prefix: Prefix for sticky note content
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF tagging. "
                "Install with: pip install PyMuPDF"
            )

        self.highlight_color = highlight_color
        self.include_sticky_notes = include_sticky_notes
        self.sticky_note_prefix = sticky_note_prefix

    def tag_pdf(
        self,
        input_path: str,
        output_path: str,
        claims: List[ProjectClaim],
    ) -> TaggedPDF:
        """
        Add evidence annotations to a PDF file.

        Args:
            input_path: Path to the original PDF
            output_path: Path for the annotated output PDF
            claims: List of claims with evidence locations

        Returns:
            TaggedPDF with operation results
        """
        result = TaggedPDF(
            original_path=input_path,
            output_path=output_path,
        )

        try:
            # Open the PDF
            doc = fitz.open(input_path)
            logger.info(f"Opened PDF: {input_path} ({doc.page_count} pages)")

            # Process each claim
            for claim in claims:
                if not claim.evidence_bbox or claim.evidence_page_number is None:
                    logger.debug(f"Skipping claim {claim.claim_id} - no location info")
                    continue

                try:
                    self._add_claim_annotation(doc, claim, result)
                except Exception as e:
                    error_msg = f"Failed to annotate claim {claim.claim_id}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

            # Save the modified PDF
            doc.save(output_path)
            doc.close()

            logger.info(
                f"Tagged PDF saved: {output_path} "
                f"({result.annotations_added} annotations)"
            )

        except Exception as e:
            error_msg = f"PDF tagging failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    def _add_claim_annotation(
        self,
        doc: "fitz.Document",
        claim: ProjectClaim,
        result: TaggedPDF,
    ):
        """Add annotation for a single claim"""
        page_num = claim.evidence_page_number - 1  # Convert to 0-indexed

        if page_num < 0 or page_num >= doc.page_count:
            raise ValueError(f"Page {claim.evidence_page_number} out of range")

        page = doc[page_num]
        bbox = claim.evidence_bbox

        # Create rectangle for annotation
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

        # Add highlight
        highlight = page.add_highlight_annot(rect)
        if highlight:
            # Set color based on verification status
            color = self._get_color_for_status(claim.status)
            highlight.set_colors(stroke=color)
            highlight.set_info(
                title="PropelAI Evidence Tag",
                content=f"Claim: {claim.criteria_id}",
            )
            highlight.update()
            result.annotations_added += 1

        # Add sticky note if enabled
        if self.include_sticky_notes:
            note_content = self._format_sticky_note(claim)
            note_rect = fitz.Point(rect.x1, rect.y0)  # Top-right of highlight

            note = page.add_text_annot(note_rect, note_content)
            if note:
                note.set_colors(stroke=COLORS["sticky_note"])
                note.set_info(
                    title=f"Evidence for {claim.criteria_id}",
                    subject="OASIS+ Scoring Claim",
                )
                note.update()
                result.annotations_added += 1

        # Track modified pages
        if page_num not in result.pages_modified:
            result.pages_modified.append(page_num)

        result.claims_tagged.append(claim.claim_id)

    def _get_color_for_status(
        self,
        status: VerificationStatus,
    ) -> Tuple[float, float, float]:
        """Get highlight color based on verification status"""
        if status == VerificationStatus.VERIFIED:
            return COLORS["highlight_green"]
        elif status == VerificationStatus.PENDING:
            return COLORS["highlight_orange"]
        else:
            return self.highlight_color

    def _format_sticky_note(self, claim: ProjectClaim) -> str:
        """Format sticky note content for a claim"""
        lines = [
            f"{self.sticky_note_prefix}{claim.criteria_id}",
            f"Points: {claim.claimed_points}",
        ]

        if claim.evidence_snippet:
            # Truncate snippet for sticky note
            snippet = claim.evidence_snippet[:200]
            if len(claim.evidence_snippet) > 200:
                snippet += "..."
            lines.append(f"Evidence: \"{snippet}\"")

        if claim.status == VerificationStatus.VERIFIED:
            lines.append("Status: VERIFIED")

        return "\n".join(lines)

    def tag_from_search_text(
        self,
        input_path: str,
        output_path: str,
        claims: List[ProjectClaim],
    ) -> TaggedPDF:
        """
        Tag PDF by searching for evidence text rather than using coordinates.

        Useful when bounding box coordinates are not available.

        Args:
            input_path: Path to the original PDF
            output_path: Path for the annotated output PDF
            claims: List of claims with evidence snippets

        Returns:
            TaggedPDF with operation results
        """
        result = TaggedPDF(
            original_path=input_path,
            output_path=output_path,
        )

        try:
            doc = fitz.open(input_path)

            for claim in claims:
                if not claim.evidence_snippet:
                    continue

                # Search for the text in the document
                search_text = claim.evidence_snippet[:100]  # First 100 chars
                found = False

                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text_instances = page.search_for(search_text)

                    if text_instances:
                        # Use first match
                        rect = text_instances[0]

                        # Add highlight
                        highlight = page.add_highlight_annot(rect)
                        if highlight:
                            color = self._get_color_for_status(claim.status)
                            highlight.set_colors(stroke=color)
                            highlight.update()
                            result.annotations_added += 1

                        # Add sticky note
                        if self.include_sticky_notes:
                            note_content = self._format_sticky_note(claim)
                            note = page.add_text_annot(
                                fitz.Point(rect.x1, rect.y0),
                                note_content
                            )
                            if note:
                                note.update()
                                result.annotations_added += 1

                        if page_num not in result.pages_modified:
                            result.pages_modified.append(page_num)
                        result.claims_tagged.append(claim.claim_id)
                        found = True
                        break

                if not found:
                    result.errors.append(
                        f"Could not find text for claim {claim.claim_id}"
                    )

            doc.save(output_path)
            doc.close()

        except Exception as e:
            result.errors.append(f"PDF tagging failed: {e}")

        return result

    def create_evidence_index(
        self,
        tagged_pdf: TaggedPDF,
        claims: List[ProjectClaim],
    ) -> str:
        """
        Create a text index of all tagged evidence for reference.

        Returns a formatted string listing all claims and their locations.
        """
        lines = [
            "OASIS+ Evidence Index",
            "=" * 50,
            f"Document: {Path(tagged_pdf.original_path).name}",
            f"Generated: {tagged_pdf.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"Total Annotations: {tagged_pdf.annotations_added}",
            "",
            "Claims Tagged:",
            "-" * 50,
        ]

        for claim in claims:
            if claim.claim_id in tagged_pdf.claims_tagged:
                lines.extend([
                    f"",
                    f"Criteria: {claim.criteria_id}",
                    f"Points: {claim.claimed_points}",
                    f"Page: {claim.evidence_page_number}",
                    f"Status: {claim.status.value}",
                    f"Evidence: {claim.evidence_snippet[:100]}..." if claim.evidence_snippet else "",
                ])

        if tagged_pdf.errors:
            lines.extend([
                "",
                "Errors:",
                "-" * 50,
            ])
            for error in tagged_pdf.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)


def tag_evidence_pdf(
    input_path: str,
    output_path: str,
    claims: List[ProjectClaim],
    use_text_search: bool = False,
) -> TaggedPDF:
    """
    Convenience function to tag a PDF with evidence annotations.

    Args:
        input_path: Path to the original PDF
        output_path: Path for the annotated output PDF
        claims: List of claims to tag
        use_text_search: If True, search for text instead of using coordinates

    Returns:
        TaggedPDF with operation results
    """
    tagger = PDFTagger()

    if use_text_search:
        return tagger.tag_from_search_text(input_path, output_path, claims)
    else:
        return tagger.tag_pdf(input_path, output_path, claims)
