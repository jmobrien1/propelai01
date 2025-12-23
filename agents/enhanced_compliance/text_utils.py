"""
Text utilities for PropelAI Enhanced Compliance.

Provides common text transformations including typo correction for
government RFP document processing.
"""

from typing import Dict, Optional
import re


# Common typos found in RFP documents and their corrections
# Keys are lowercase for case-insensitive matching
COMMON_TYPOS: Dict[str, str] = {
    # Statement of Work typos
    "stament": "Statement",
    "statment": "Statement",
    "statemnt": "Statement",
    "satatement": "Statement",
    # Performance Work Statement typos
    "performace": "Performance",
    "perfomance": "Performance",
    "preformance": "Performance",
    # Requirement typos
    "requirment": "Requirement",
    "requirments": "Requirements",
    "requiremnt": "Requirement",
    # Solicitation typos
    "solicitaion": "Solicitation",
    "solictation": "Solicitation",
    # Evaluation typos
    "evalution": "Evaluation",
    "evluation": "Evaluation",
    # Amendment typos
    "ammendment": "Amendment",
    "amendmnet": "Amendment",
    # Attachment typos
    "attachemnt": "Attachment",
    "attchment": "Attachment",
    # Proposal typos
    "porposal": "Proposal",
    "proposl": "Proposal",
    # Technical typos
    "techical": "Technical",
    "tecnical": "Technical",
    # Specification typos
    "specfication": "Specification",
    "specificaiton": "Specification",
    # Contract typos
    "contarct": "Contract",
    "contrct": "Contract",
    # Government typos
    "governement": "Government",
    "goverment": "Government",
    # Instructions typos
    "instrutions": "Instructions",
    "instuctions": "Instructions",
    # Compliance typos
    "complance": "Compliance",
    "compliace": "Compliance",
    # Experience typos
    "experiance": "Experience",
    "experince": "Experience",
    # Management typos
    "managment": "Management",
    "mangement": "Management",
    # Description typos
    "discription": "Description",
    "desciption": "Description",
}


def correct_text_typos(text: str, preserve_case: bool = True) -> str:
    """
    Correct common typos found in RFP documents.

    Args:
        text: Input text that may contain typos
        preserve_case: If True, attempts to match the original case pattern

    Returns:
        Text with typos corrected
    """
    if not text:
        return text

    result = text

    for typo, correction in COMMON_TYPOS.items():
        # Build a case-insensitive pattern that captures the typo
        pattern = re.compile(re.escape(typo), re.IGNORECASE)

        def replace_match(match):
            original = match.group(0)
            if not preserve_case:
                return correction

            # Try to match the case pattern of the original
            if original.isupper():
                return correction.upper()
            elif original.islower():
                return correction.lower()
            elif original[0].isupper():
                return correction  # Default is Title Case
            else:
                return correction.lower()

        result = pattern.sub(replace_match, result)

    return result


def correct_filename(filename: str) -> str:
    """
    Correct common typos in document filenames.

    Preserves file extension and corrects known typos in the
    filename stem.

    Args:
        filename: Original filename (with or without path)

    Returns:
        Corrected filename
    """
    if not filename:
        return filename

    # Split filename from extension
    import os
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)

    # Correct typos in the filename stem
    corrected_name = correct_text_typos(name, preserve_case=True)

    return corrected_name + ext


def normalize_document_name(name: str) -> str:
    """
    Normalize a document name for display.

    - Corrects typos
    - Replaces underscores with spaces
    - Title cases words

    Args:
        name: Original document name

    Returns:
        Normalized display name
    """
    if not name:
        return name

    # First correct typos
    corrected = correct_text_typos(name)

    # Replace underscores and hyphens with spaces
    normalized = corrected.replace('_', ' ').replace('-', ' ')

    # Clean up multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


# For backwards compatibility, expose as module-level function
def fix_common_typos(text: str) -> str:
    """Alias for correct_text_typos for backward compatibility."""
    return correct_text_typos(text)
