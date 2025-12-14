"""
PropelAI Validation Framework - Requirement Matching Algorithms

Provides algorithms for matching extracted requirements against ground truth
to determine true positives, false positives, and false negatives.

Matching strategies:
1. Exact match: Text hash equality
2. Fuzzy match: Text similarity (Jaccard, Levenshtein)
3. Semantic match: Embedding-based similarity (future)
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .schemas import GroundTruthRequirement, FalsePositiveType, FalseNegativeType


@dataclass
class MatchResult:
    """Result of matching a single extracted requirement to ground truth"""
    extracted_id: str
    extracted_text: str
    matched: bool
    match_type: str = ""                    # exact, fuzzy, semantic
    similarity_score: float = 0.0

    # If matched
    matched_gt_id: Optional[str] = None
    matched_gt_text: Optional[str] = None

    # Classification match (only if requirement matched)
    section_correct: bool = False
    binding_correct: bool = False
    category_correct: bool = False

    # If not matched (false positive)
    fp_type: Optional[str] = None
    fp_reason: str = ""


@dataclass
class GroundTruthMatchResult:
    """Result of checking if a ground truth requirement was found"""
    gt_id: str
    gt_text: str
    found: bool
    match_type: str = ""

    # If found
    matched_extracted_id: Optional[str] = None
    matched_extracted_text: Optional[str] = None
    similarity_score: float = 0.0

    # Classification accuracy
    section_correct: bool = False
    binding_correct: bool = False
    category_correct: bool = False

    # If not found (false negative)
    fn_type: Optional[str] = None
    fn_reason: str = ""


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove punctuation for matching (but keep for display)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Jaccard = |A ∩ B| / |A ∪ B|
    """
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def sequence_similarity(text1: str, text2: str) -> float:
    """
    Calculate sequence similarity using SequenceMatcher (Ratcliff/Obershelp).

    Better for detecting partial matches and text fragments.
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


def combined_similarity(text1: str, text2: str) -> float:
    """
    Combined similarity score using both Jaccard and sequence matching.

    Returns weighted average favoring sequence match for longer texts.
    """
    jaccard = jaccard_similarity(text1, text2)
    sequence = sequence_similarity(text1, text2)

    # Weight sequence matching more for longer texts
    min_len = min(len(text1), len(text2))
    if min_len > 200:
        weight_sequence = 0.7
    elif min_len > 100:
        weight_sequence = 0.6
    else:
        weight_sequence = 0.5

    return (1 - weight_sequence) * jaccard + weight_sequence * sequence


def find_best_match(
    extracted_text: str,
    ground_truth_list: List[GroundTruthRequirement],
    threshold: float = 0.7,
    already_matched: Optional[Set[str]] = None
) -> Tuple[Optional[GroundTruthRequirement], float, str]:
    """
    Find the best matching ground truth requirement for an extracted requirement.

    Args:
        extracted_text: The extracted requirement text
        ground_truth_list: List of ground truth requirements to match against
        threshold: Minimum similarity score to consider a match
        already_matched: Set of gt_ids already matched (to avoid double-matching)

    Returns:
        Tuple of (matched_requirement, similarity_score, match_type)
        Returns (None, 0.0, "") if no match found
    """
    if already_matched is None:
        already_matched = set()

    best_match = None
    best_score = 0.0
    best_type = ""

    extracted_hash = normalize_text(extracted_text)

    for gt_req in ground_truth_list:
        if gt_req.gt_id in already_matched:
            continue

        # Try exact hash match first
        if gt_req.text_hash and extracted_hash == normalize_text(gt_req.text):
            return (gt_req, 1.0, "exact")

        # Try fuzzy matching
        similarity = combined_similarity(extracted_text, gt_req.text)

        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = gt_req
            best_type = "fuzzy" if similarity < 0.95 else "near_exact"

    return (best_match, best_score, best_type)


def classify_false_positive(
    extracted_text: str,
    extracted_section: str = "",
    extracted_binding: str = ""
) -> Tuple[FalsePositiveType, str]:
    """
    Classify a false positive extraction to understand why it was incorrectly extracted.

    Returns:
        Tuple of (FalsePositiveType, explanation)
    """
    text_lower = extracted_text.lower()

    # Check for TOC entries
    if re.search(r'\.\s*\.\s*\.', extracted_text) or re.search(r'\d+\s*$', extracted_text.strip()):
        if len(extracted_text) < 150:
            return (FalsePositiveType.TOC_ENTRY, "Contains TOC patterns (dots, trailing page numbers)")

    # Check for headers
    if extracted_text.isupper() and len(extracted_text) < 100:
        return (FalsePositiveType.HEADER, "All uppercase, likely a header")

    if re.match(r'^(SECTION|ARTICLE|PART)\s+[A-Z\d]', extracted_text, re.IGNORECASE):
        if len(extracted_text) < 80:
            return (FalsePositiveType.HEADER, "Matches section header pattern")

    # Check for government obligations (not contractor requirements)
    gov_patterns = [
        r'government\s+(shall|will|must)',
        r'the\s+government\s+is\s+responsible',
        r'government\s+will\s+provide',
    ]
    for pattern in gov_patterns:
        if re.search(pattern, text_lower):
            return (FalsePositiveType.GOVERNMENT_OBLIGATION, "Government action, not contractor requirement")

    # Check for boilerplate
    boilerplate_patterns = [
        r'this\s+page\s+intentionally',
        r'continued\s+on\s+next\s+page',
        r'see\s+attached',
        r'end\s+of\s+section',
        r'reserved',
    ]
    for pattern in boilerplate_patterns:
        if re.search(pattern, text_lower):
            return (FalsePositiveType.BOILERPLATE, "Matches boilerplate pattern")

    # Check for pure informational content
    if not re.search(r'\b(shall|must|should|will|may|required)\b', text_lower):
        return (FalsePositiveType.INFORMATIONAL, "No binding language found")

    # Check for cross-references without standalone requirement
    if re.match(r'^(see|refer\s+to|per|in\s+accordance\s+with)\s+', text_lower):
        if len(extracted_text) < 100:
            return (FalsePositiveType.CROSS_REFERENCE, "Cross-reference only, not standalone requirement")

    # Check for incomplete fragments
    if len(extracted_text) < 50:
        return (FalsePositiveType.INCOMPLETE, "Too short to be a complete requirement")

    # Default: unknown false positive
    return (FalsePositiveType.INFORMATIONAL, "Does not match any ground truth requirement")


def classify_false_negative(
    gt_requirement: GroundTruthRequirement,
    extracted_requirements: List[Any],
    partial_threshold: float = 0.4
) -> Tuple[FalseNegativeType, str]:
    """
    Classify why a ground truth requirement was not found in extraction.

    Args:
        gt_requirement: The missed ground truth requirement
        extracted_requirements: All extracted requirements
        partial_threshold: Threshold for detecting partial matches

    Returns:
        Tuple of (FalseNegativeType, explanation)
    """
    gt_text = gt_requirement.text

    # Check for partial matches
    best_partial_score = 0.0
    best_partial_text = ""
    best_partial_section = ""

    for ext_req in extracted_requirements:
        ext_text = ext_req.full_text if hasattr(ext_req, 'full_text') else str(ext_req)
        ext_section = ext_req.source_section.value if hasattr(ext_req, 'source_section') else ""

        similarity = combined_similarity(gt_text, ext_text)

        if similarity > best_partial_score:
            best_partial_score = similarity
            best_partial_text = ext_text
            best_partial_section = ext_section

    # Partial match detected but below threshold
    if best_partial_score >= partial_threshold and best_partial_score < 0.7:
        return (FalseNegativeType.PARTIAL_MATCH,
                f"Partial match found (similarity={best_partial_score:.2f}) but below threshold")

    # Check if found but in wrong section
    if best_partial_score >= 0.7:
        if best_partial_section and gt_requirement.rfp_section:
            if best_partial_section != gt_requirement.rfp_section:
                return (FalseNegativeType.WRONG_SECTION,
                        f"Found in section {best_partial_section}, expected {gt_requirement.rfp_section}")

    # Check if this was likely filtered
    if len(gt_text) < 50:
        return (FalseNegativeType.FILTERED, "Short requirement likely filtered by length threshold")

    if gt_requirement.binding_level == "Informational":
        return (FalseNegativeType.FILTERED, "Informational requirement likely filtered")

    # Check for merged requirements (part of a compound)
    if gt_requirement.parent_compound_id:
        return (FalseNegativeType.MERGED, "Part of compound requirement that was extracted as single item")

    # Default: not extracted at all
    return (FalseNegativeType.NOT_EXTRACTED, "Requirement not found in extraction results")


def match_requirements(
    extracted_requirements: List[Any],
    ground_truth: List[GroundTruthRequirement],
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Match all extracted requirements against ground truth.

    Args:
        extracted_requirements: List of extracted requirements (StructuredRequirement objects)
        ground_truth: List of ground truth requirements
        threshold: Minimum similarity for a match

    Returns:
        Dictionary with matching results:
        - extracted_matches: List[MatchResult] for each extracted requirement
        - gt_matches: List[GroundTruthMatchResult] for each ground truth
        - true_positives: Count of correctly extracted requirements
        - false_positives: Count of incorrect extractions
        - false_negatives: Count of missed requirements
    """
    extracted_matches: List[MatchResult] = []
    gt_matches: List[GroundTruthMatchResult] = []
    matched_gt_ids: Set[str] = set()
    matched_extracted_ids: Set[str] = set()

    # First pass: Match extracted → ground truth
    for ext_req in extracted_requirements:
        ext_id = ext_req.generated_id if hasattr(ext_req, 'generated_id') else str(id(ext_req))
        ext_text = ext_req.full_text if hasattr(ext_req, 'full_text') else str(ext_req)
        ext_section = ext_req.source_section.value if hasattr(ext_req, 'source_section') else ""
        ext_binding = ext_req.binding_level.value if hasattr(ext_req, 'binding_level') else ""
        ext_category = ext_req.category.value if hasattr(ext_req, 'category') else ""

        matched_gt, score, match_type = find_best_match(
            ext_text, ground_truth, threshold, matched_gt_ids
        )

        if matched_gt:
            matched_gt_ids.add(matched_gt.gt_id)
            matched_extracted_ids.add(ext_id)

            # Check classification accuracy
            section_correct = (ext_section == matched_gt.rfp_section) if ext_section else False
            binding_correct = (ext_binding == matched_gt.binding_level) if ext_binding else False
            category_correct = (ext_category == matched_gt.category) if ext_category else False

            extracted_matches.append(MatchResult(
                extracted_id=ext_id,
                extracted_text=ext_text,
                matched=True,
                match_type=match_type,
                similarity_score=score,
                matched_gt_id=matched_gt.gt_id,
                matched_gt_text=matched_gt.text,
                section_correct=section_correct,
                binding_correct=binding_correct,
                category_correct=category_correct,
            ))
        else:
            # False positive
            fp_type, fp_reason = classify_false_positive(ext_text, ext_section, ext_binding)

            extracted_matches.append(MatchResult(
                extracted_id=ext_id,
                extracted_text=ext_text,
                matched=False,
                match_type="",
                similarity_score=0.0,
                fp_type=fp_type.value,
                fp_reason=fp_reason,
            ))

    # Second pass: Check ground truth → extracted (for false negatives)
    for gt_req in ground_truth:
        if gt_req.gt_id in matched_gt_ids:
            # Already matched in first pass
            # Find the corresponding extracted match
            for em in extracted_matches:
                if em.matched and em.matched_gt_id == gt_req.gt_id:
                    gt_matches.append(GroundTruthMatchResult(
                        gt_id=gt_req.gt_id,
                        gt_text=gt_req.text,
                        found=True,
                        match_type=em.match_type,
                        matched_extracted_id=em.extracted_id,
                        matched_extracted_text=em.extracted_text,
                        similarity_score=em.similarity_score,
                        section_correct=em.section_correct,
                        binding_correct=em.binding_correct,
                        category_correct=em.category_correct,
                    ))
                    break
        else:
            # False negative
            fn_type, fn_reason = classify_false_negative(gt_req, extracted_requirements)

            gt_matches.append(GroundTruthMatchResult(
                gt_id=gt_req.gt_id,
                gt_text=gt_req.text,
                found=False,
                fn_type=fn_type.value,
                fn_reason=fn_reason,
            ))

    # Calculate summary stats
    true_positives = sum(1 for em in extracted_matches if em.matched)
    false_positives = sum(1 for em in extracted_matches if not em.matched)
    false_negatives = sum(1 for gm in gt_matches if not gm.found)

    return {
        "extracted_matches": extracted_matches,
        "gt_matches": gt_matches,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_extracted": len(extracted_requirements),
        "total_ground_truth": len(ground_truth),
    }
