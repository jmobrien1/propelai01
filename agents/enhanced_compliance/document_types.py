"""
PropelAI: Document Type Definitions for Guided Upload

This module defines the document types that users can upload, with clear
descriptions and guidance for proposal managers (especially newcomers).

The goal is to help users upload ONLY the documents needed for accurate
extraction, avoiding noise from irrelevant attachments like DD254s, CDRLs, etc.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class DocumentType(Enum):
    """
    Core document types for RFP processing.

    These map to the key sections of government RFPs that contain
    extractable requirements and proposal guidance.
    """
    # Required documents
    SOW = "sow"                      # Statement of Work / PWS - Section C
    INSTRUCTIONS = "instructions"    # Proposal Instructions - Section L
    EVALUATION = "evaluation"        # Evaluation Criteria - Section M

    # Optional but useful
    SOLICITATION = "solicitation"    # Main solicitation document (SF1449, cover)
    AMENDMENT = "amendment"          # Amendments/modifications to RFP

    # Special cases
    COMBINED_LM = "combined_lm"      # Combined Section L and M (common)
    ATTACHMENT = "attachment"        # Other relevant attachments

    # For backwards compatibility / bulk upload
    AUTO_DETECT = "auto_detect"      # System will attempt to classify

    # User explicitly says not applicable
    NOT_APPLICABLE = "not_applicable"


@dataclass
class DocumentSlot:
    """
    Definition of an upload slot in the guided UI.

    Each slot represents a type of document users should look for
    in their RFP package.
    """
    id: str
    doc_type: DocumentType
    title: str
    description: str
    help_text: str
    common_names: List[str]
    required: bool = False
    allows_multiple: bool = False
    show_not_applicable: bool = True
    order: int = 0

    # For UI display
    icon: str = "📄"
    color: str = "#4A5568"  # Default gray


# Define all upload slots with user-friendly descriptions
UPLOAD_SLOTS: List[DocumentSlot] = [
    DocumentSlot(
        id="sow",
        doc_type=DocumentType.SOW,
        title="Statement of Work (SOW/PWS)",
        description="The technical requirements - what the contractor must DO",
        help_text="""This is the most important document for requirements extraction.

It describes ALL the work the contractor must perform, including:
• Technical tasks and deliverables
• Performance standards
• Personnel requirements
• Reporting requirements

Also called: Performance Work Statement (PWS), Scope of Work,
Statement of Objectives (SOO), or Technical Requirements.""",
        common_names=[
            "Statement of Work",
            "SOW",
            "PWS",
            "Performance Work Statement",
            "Attachment 1",
            "Section C",
            "Scope of Work",
            "Technical Requirements",
            "SOO",
            "Statement of Objectives"
        ],
        required=True,
        allows_multiple=True,  # Some RFPs split SOW across attachments
        show_not_applicable=False,
        order=1,
        icon="📋",
        color="#2B6CB0"  # Blue
    ),

    DocumentSlot(
        id="instructions",
        doc_type=DocumentType.INSTRUCTIONS,
        title="Proposal Instructions (Section L)",
        description="HOW to write and format your proposal",
        help_text="""This document tells you how to structure your proposal response.

Look for information about:
• Volume structure (Technical, Management, Cost volumes)
• Page limits for each section
• Font, margin, and formatting requirements
• What to include in each volume
• Submission method and deadline

Also called: Instructions to Offerors, Proposal Preparation Instructions,
Placement Procedures, or Section L.""",
        common_names=[
            "Section L",
            "Instructions to Offerors",
            "Proposal Instructions",
            "Proposal Preparation",
            "Placement Procedures",
            "Attachment 2",
            "Instructions",
            "Submission Requirements"
        ],
        required=True,
        allows_multiple=False,
        show_not_applicable=True,
        order=2,
        icon="📝",
        color="#D69E2E"  # Yellow/Gold
    ),

    DocumentSlot(
        id="evaluation",
        doc_type=DocumentType.EVALUATION,
        title="Evaluation Criteria (Section M)",
        description="HOW your proposal will be SCORED",
        help_text="""This document explains how the government will evaluate proposals.

Look for:
• Evaluation factors (Technical, Management, Past Performance, Price)
• Relative importance of each factor
• Subfactors and their weights
• Rating scales (Outstanding, Good, Acceptable, etc.)
• What makes a proposal "win"

Also called: Evaluation Factors, Basis for Award, Section M,
or Award Criteria.

NOTE: Sometimes combined with Section L in one document.""",
        common_names=[
            "Section M",
            "Evaluation Factors",
            "Evaluation Criteria",
            "Basis for Award",
            "Award Criteria",
            "Evaluation",
            "Selection Criteria"
        ],
        required=False,  # Often combined with Section L
        allows_multiple=False,
        show_not_applicable=True,
        order=3,
        icon="⚖️",
        color="#38A169"  # Green
    ),

    DocumentSlot(
        id="combined_lm",
        doc_type=DocumentType.COMBINED_LM,
        title="Combined Instructions & Evaluation",
        description="Single document with BOTH Section L and M content",
        help_text="""Many RFPs combine proposal instructions and evaluation criteria
in a single document.

If your RFP has ONE document that covers both:
• How to structure your proposal (volumes, pages, format)
• How proposals will be evaluated (factors, weights, ratings)

Upload it here instead of using separate L and M slots.

Common in: GSA task orders, smaller procurements, some DoD RFPs.""",
        common_names=[
            "Sections L and M",
            "Section L-M",
            "Instructions and Evaluation",
            "Proposal Requirements",
            "RFP Instructions"
        ],
        required=False,
        allows_multiple=False,
        show_not_applicable=True,
        order=4,
        icon="📑",
        color="#805AD5"  # Purple
    ),

    DocumentSlot(
        id="solicitation",
        doc_type=DocumentType.SOLICITATION,
        title="Main Solicitation Document",
        description="The cover/overview document with key dates and contacts",
        help_text="""The main RFP document that ties everything together.

Typically includes:
• SF1449 or SF33 (Standard Forms)
• Solicitation number
• Due date and time
• Contracting Officer contact info
• List of attachments
• Special instructions

This helps PropelAI extract key dates and organize the RFP context.""",
        common_names=[
            "Solicitation",
            "RFP",
            "SF1449",
            "SF33",
            "Request for Proposal",
            "Cover Page",
            "Main Document"
        ],
        required=False,
        allows_multiple=False,
        show_not_applicable=True,
        order=5,
        icon="📰",
        color="#4A5568"  # Gray
    ),

    DocumentSlot(
        id="amendments",
        doc_type=DocumentType.AMENDMENT,
        title="Amendments / Modifications",
        description="Changes or updates to the original RFP",
        help_text="""Amendments modify the original RFP requirements.

IMPORTANT: Amendments can:
• Change requirements in the SOW
• Modify page limits or formatting
• Extend deadlines
• Answer questions that clarify requirements

If you have amendments, upload them here so PropelAI can
incorporate the changes.

Tip: Upload in order (Amendment 1, then 2, etc.)""",
        common_names=[
            "Amendment",
            "Modification",
            "SF30",
            "Change Notice",
            "RFP Update",
            "Addendum"
        ],
        required=False,
        allows_multiple=True,  # Can have multiple amendments
        show_not_applicable=True,
        order=6,
        icon="📌",
        color="#E53E3E"  # Red
    ),
]


# Documents that should be SKIPPED (shown to user as "don't upload these")
SKIP_DOCUMENTS = [
    {
        "name": "DD254 - Security Classification",
        "description": "Security forms don't contain proposal requirements",
        "patterns": ["dd254", "dd 254", "security classification"]
    },
    {
        "name": "DD1423 - CDRL",
        "description": "Contract Data Requirements Lists are for post-award",
        "patterns": ["dd1423", "cdrl", "contract data requirements"]
    },
    {
        "name": "Pricing Templates",
        "description": "Cost tables are for your pricing team, not requirements",
        "patterns": ["pricing template", "cost template", "rate card", "labor rates"]
    },
    {
        "name": "Resume Templates",
        "description": "Personnel forms are for your HR team",
        "patterns": ["resume template", "personnel form", "key personnel form"]
    },
    {
        "name": "Past Performance Forms",
        "description": "Reference questionnaires are for your references",
        "patterns": ["past performance questionnaire", "ppq", "reference form"]
    },
    {
        "name": "OCI Plan Templates",
        "description": "Conflict of interest templates don't contain requirements",
        "patterns": ["oci", "organizational conflict", "conflict of interest"]
    },
    {
        "name": "Q&A Documents",
        "description": "Questions & Answers can confuse the extraction - review manually",
        "patterns": ["q&a", "questions and answers", "industry day"]
    },
]


def get_slot_by_id(slot_id: str) -> Optional[DocumentSlot]:
    """Get a document slot by its ID"""
    for slot in UPLOAD_SLOTS:
        if slot.id == slot_id:
            return slot
    return None


def get_slot_by_doc_type(doc_type: DocumentType) -> Optional[DocumentSlot]:
    """Get a document slot by its document type"""
    for slot in UPLOAD_SLOTS:
        if slot.doc_type == doc_type:
            return slot
    return None


def classify_document_by_filename(filename: str) -> DocumentType:
    """
    Attempt to classify a document based on its filename.
    Used for backwards compatibility with bulk uploads.
    """
    filename_lower = filename.lower()

    # Check each slot's common names
    for slot in UPLOAD_SLOTS:
        for name in slot.common_names:
            if name.lower() in filename_lower:
                return slot.doc_type

    # Check skip patterns - if it matches, still return AUTO_DETECT
    # but the extraction pipeline will know to skip it
    for skip_doc in SKIP_DOCUMENTS:
        for pattern in skip_doc["patterns"]:
            if pattern in filename_lower:
                return DocumentType.AUTO_DETECT  # Will be filtered by extractor

    return DocumentType.AUTO_DETECT


def get_ui_config() -> Dict[str, Any]:
    """
    Get the complete UI configuration for the guided upload interface.
    Returns a JSON-serializable dictionary.
    """
    return {
        "upload_slots": [
            {
                "id": slot.id,
                "doc_type": slot.doc_type.value,
                "title": slot.title,
                "description": slot.description,
                "help_text": slot.help_text,
                "common_names": slot.common_names,
                "required": slot.required,
                "allows_multiple": slot.allows_multiple,
                "show_not_applicable": slot.show_not_applicable,
                "order": slot.order,
                "icon": slot.icon,
                "color": slot.color,
            }
            for slot in sorted(UPLOAD_SLOTS, key=lambda s: s.order)
        ],
        "skip_documents": SKIP_DOCUMENTS,
        "tips": [
            "Upload only the documents that contain requirements or proposal instructions",
            "If Section L and M are in one document, use the 'Combined' slot",
            "Amendments should be uploaded separately so changes are tracked",
            "When in doubt about a document, check if it contains 'shall' statements",
        ]
    }
