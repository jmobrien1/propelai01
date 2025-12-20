"""
PropelAI: Document Type Definitions for Guided Upload (v2.0)

This module defines the document types that users can upload, with clear
descriptions and guidance for proposal managers (especially newcomers).

The goal is to help users upload ONLY the documents needed for accurate
extraction, avoiding noise from irrelevant attachments like DD254s, CDRLs, etc.

Design Philosophy (v2.0):
- Only SOW/PWS is truly required (it contains the actual requirements)
- Many RFPs combine everything in one document - make this easy
- Avoid government jargon (Section L/M) in primary UI - use plain English
- Provide expert-level help text for those who want to learn
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
    # The one truly required document
    SOW = "sow"                      # Statement of Work / PWS - Section C

    # Combined/All-in-one (very common)
    COMBINED_RFP = "combined_rfp"    # Full RFP with everything in one document

    # Separate instruction documents (when not combined)
    INSTRUCTIONS = "instructions"    # How to write proposal (Section L)
    EVALUATION = "evaluation"        # How proposal scored (Section M)
    COMBINED_LM = "combined_lm"      # Instructions + Evaluation together

    # Supporting documents
    SOLICITATION = "solicitation"    # Cover/admin document (SF1449, etc.)
    AMENDMENT = "amendment"          # Changes to original RFP
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
# v2.0: Simplified structure based on real-world RFP analysis across agencies
UPLOAD_SLOTS: List[DocumentSlot] = [
    # ==== PRIMARY SLOT: Technical Requirements (the one truly required doc) ====
    DocumentSlot(
        id="sow",
        doc_type=DocumentType.SOW,
        title="Technical Requirements (SOW/PWS)",
        description="What the contractor must DO - the actual work requirements",
        help_text="""This is the MOST IMPORTANT document for requirements extraction.

It describes ALL the work the contractor must perform:
• Technical tasks and deliverables
• Performance standards and metrics
• Personnel qualifications
• Reporting and delivery requirements

WHAT TO LOOK FOR in your RFP package:
• "Statement of Work" or "SOW"
• "Performance Work Statement" or "PWS"
• "Statement of Objectives" or "SOO"
• "Technical Requirements"
• Often labeled as "Attachment 1" or "Section C"

TIP: If requirements are spread across multiple attachments,
upload all of them here.""",
        common_names=[
            "Statement of Work", "SOW", "PWS", "Performance Work Statement",
            "Attachment 1", "Section C", "Scope of Work", "Technical Requirements",
            "SOO", "Statement of Objectives", "Work Statement", "Requirements"
        ],
        required=True,
        allows_multiple=True,  # SOW can span multiple attachments
        show_not_applicable=False,
        order=1,
        icon="📋",
        color="#2B6CB0"  # Blue
    ),

    # ==== COMBINED/ALL-IN-ONE OPTION (prominently displayed) ====
    DocumentSlot(
        id="combined_rfp",
        doc_type=DocumentType.COMBINED_RFP,
        title="Complete RFP (All-in-One Document)",
        description="Single document containing requirements, instructions, AND evaluation criteria",
        help_text="""Many RFPs - especially from civilian agencies, GSA, and smaller
procurements - combine EVERYTHING in one document.

USE THIS SLOT IF your RFP is a single document that includes:
• Technical requirements (what to do)
• Proposal instructions (how to write it)
• Evaluation criteria (how it will be scored)

COMMON EXAMPLES:
• GSA Schedule RFQs and BPA task orders
• NIH and HHS solicitations
• State and local government RFPs
• Small Business set-asides
• Most RFPs under $10M

If you upload here, you can skip the other slots below.""",
        common_names=[
            "RFP", "Request for Proposal", "Solicitation", "Complete RFP",
            "Full Solicitation", "Task Order Request", "RFQ", "Request for Quote"
        ],
        required=False,
        allows_multiple=False,
        show_not_applicable=True,
        order=2,
        icon="📚",
        color="#805AD5"  # Purple - stands out
    ),

    # ==== INSTRUCTIONS & EVALUATION (simplified from separate L/M) ====
    DocumentSlot(
        id="instructions_evaluation",
        doc_type=DocumentType.COMBINED_LM,
        title="Proposal Instructions & Evaluation Criteria",
        description="How to write your proposal and how it will be scored",
        help_text="""This covers the "rules of the game" for your proposal.

PROPOSAL INSTRUCTIONS tell you:
• Volume structure (Technical, Management, Past Performance, Cost)
• Page limits for each section
• Font, margins, and formatting requirements
• What to address in each volume
• Submission method and deadline

EVALUATION CRITERIA tell you:
• Evaluation factors (Technical Approach, Management, Price, etc.)
• Relative importance ("Technical is more important than Price")
• Subfactors and their weights
• Rating scales (Outstanding, Good, Acceptable, etc.)

WHERE TO FIND THIS:
• "Section L" and/or "Section M" (DoD UCF format)
• "Instructions to Offerors"
• "Evaluation Factors" or "Basis for Award"
• "Placement Procedures" (GSA)
• Often combined in one attachment

NOTE: If your RFP has separate Section L and Section M documents,
you can upload them both here - or use the advanced slots below.""",
        common_names=[
            "Section L", "Section M", "Instructions to Offerors",
            "Proposal Instructions", "Evaluation Factors", "Evaluation Criteria",
            "Basis for Award", "Placement Procedures", "Award Criteria",
            "Selection Criteria", "Instructions and Evaluation"
        ],
        required=False,  # Not all RFPs have separate instruction docs
        allows_multiple=True,  # Can upload both L and M
        show_not_applicable=True,
        order=3,
        icon="📝",
        color="#D69E2E"  # Yellow/Gold
    ),

    # ==== AMENDMENTS (important for accuracy) ====
    DocumentSlot(
        id="amendments",
        doc_type=DocumentType.AMENDMENT,
        title="Amendments / Modifications",
        description="Any changes or updates to the original RFP",
        help_text="""Amendments modify the original RFP and are CRITICAL for accuracy.

AMENDMENTS CAN:
• Change technical requirements in the SOW
• Modify page limits or volume structure
• Extend or change deadlines
• Add, remove, or clarify requirements
• Answer industry questions that affect requirements

IMPORTANT: Upload ALL amendments you have received.
PropelAI will incorporate the changes into extraction.

TIP: If you have multiple amendments (Amendment 1, 2, 3, etc.),
upload them in order for best results.""",
        common_names=[
            "Amendment", "Modification", "SF30", "Change Notice",
            "RFP Update", "Addendum", "Amendment 1", "Amendment 2",
            "Mod", "Questions and Answers"
        ],
        required=False,
        allows_multiple=True,  # Often multiple amendments
        show_not_applicable=True,
        order=4,
        icon="📌",
        color="#E53E3E"  # Red - attention
    ),

    # ==== COVER/ADMIN DOCUMENT (helpful but optional) ====
    DocumentSlot(
        id="solicitation",
        doc_type=DocumentType.SOLICITATION,
        title="Cover Document / SF1449",
        description="Administrative info: solicitation number, dates, contacts",
        help_text="""The administrative cover document for the RFP.

TYPICALLY INCLUDES:
• SF1449 or SF33 (Standard Forms)
• Solicitation number
• Proposal due date and time
• Contracting Officer contact information
• List of attachments
• NAICS code and size standard

This helps PropelAI extract key dates and metadata, but is
NOT required if this info is already in your other documents.""",
        common_names=[
            "SF1449", "SF33", "Cover Page", "Solicitation",
            "Standard Form", "Cover Document", "Admin"
        ],
        required=False,
        allows_multiple=False,
        show_not_applicable=True,
        order=5,
        icon="📰",
        color="#4A5568"  # Gray - less prominent
    ),
]

# ==== ADVANCED SLOTS (for users who want granular control) ====
ADVANCED_UPLOAD_SLOTS: List[DocumentSlot] = [
    DocumentSlot(
        id="instructions_only",
        doc_type=DocumentType.INSTRUCTIONS,
        title="Section L Only (Proposal Instructions)",
        description="Just the instructions - no evaluation criteria",
        help_text="""Use this if you have a SEPARATE Section L document.

Most users should use the combined "Proposal Instructions & Evaluation"
slot above instead.""",
        common_names=["Section L", "Instructions to Offerors", "Proposal Instructions"],
        required=False,
        allows_multiple=False,
        show_not_applicable=True,
        order=10,
        icon="📝",
        color="#D69E2E"
    ),
    DocumentSlot(
        id="evaluation_only",
        doc_type=DocumentType.EVALUATION,
        title="Section M Only (Evaluation Criteria)",
        description="Just the evaluation factors - no instructions",
        help_text="""Use this if you have a SEPARATE Section M document.

Most users should use the combined "Proposal Instructions & Evaluation"
slot above instead.""",
        common_names=["Section M", "Evaluation Factors", "Evaluation Criteria", "Basis for Award"],
        required=False,
        allows_multiple=False,
        show_not_applicable=True,
        order=11,
        icon="⚖️",
        color="#38A169"
    ),
    DocumentSlot(
        id="other_attachment",
        doc_type=DocumentType.ATTACHMENT,
        title="Other Relevant Attachment",
        description="Additional document that contains requirements",
        help_text="""Use this for attachments that contain requirements but don't
fit the other categories.

Examples: Technical exhibits, CDRLs with requirements, labor category
descriptions, security requirements documents.""",
        common_names=["Attachment", "Exhibit", "Appendix"],
        required=False,
        allows_multiple=True,
        show_not_applicable=True,
        order=12,
        icon="📎",
        color="#718096"
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


def get_ui_config(include_advanced: bool = False) -> Dict[str, Any]:
    """
    Get the complete UI configuration for the guided upload interface.
    Returns a JSON-serializable dictionary.

    Args:
        include_advanced: If True, include advanced granular slots (Section L only, etc.)
    """
    def slot_to_dict(slot: DocumentSlot) -> Dict[str, Any]:
        return {
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

    slots = sorted(UPLOAD_SLOTS, key=lambda s: s.order)
    advanced_slots = sorted(ADVANCED_UPLOAD_SLOTS, key=lambda s: s.order) if include_advanced else []

    return {
        "upload_slots": [slot_to_dict(s) for s in slots],
        "advanced_slots": [slot_to_dict(s) for s in advanced_slots] if include_advanced else [],
        "skip_documents": SKIP_DOCUMENTS,
        "tips": [
            "Only the SOW/PWS is required - all other slots are optional",
            "If your RFP is ONE document with everything, use 'Complete RFP (All-in-One)'",
            "Upload amendments separately so PropelAI can track changes accurately",
            "When in doubt, check if a document contains 'shall' or 'must' statements",
            "Skip DD254s, CDRLs, pricing templates - they don't help extraction",
        ],
        "quick_start_tips": [
            {
                "scenario": "Single document RFP",
                "action": "Upload in 'Complete RFP (All-in-One)' slot",
                "examples": "GSA RFQs, NIH solicitations, State RFPs"
            },
            {
                "scenario": "DoD RFP with multiple attachments",
                "action": "Upload SOW in first slot, Section L/M in second",
                "examples": "Most DoD contracts, UCF format RFPs"
            },
            {
                "scenario": "Not sure what documents you have",
                "action": "Start with just the SOW/PWS - you can add more later",
                "examples": "Any RFP where you're unsure"
            },
        ]
    }
