"""
TF-IDF Deduplication Engine for Requirements

Identifies and merges semantically similar requirements using:
1. TF-IDF vectorization for text similarity
2. Cosine similarity thresholds
3. Intelligent merge strategies that preserve source information

From Long-Form Generation Strategy:
"Cross-referenced requirements create duplicates. Without deduplication,
compliance matrices become bloated and confusing. Use TF-IDF similarity
to detect near-duplicates and merge them while preserving all source citations."
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import RequirementNode, ConfidenceLevel, RequirementType

logger = logging.getLogger(__name__)


class DuplicateType(str, Enum):
    """Classification of duplicate relationship"""
    EXACT = "exact"              # Identical text
    NEAR_EXACT = "near_exact"    # Minor differences (whitespace, punctuation)
    SEMANTIC = "semantic"        # Same meaning, different wording
    PARTIAL = "partial"          # Overlapping content
    RELATED = "related"          # Similar topic, different requirement


@dataclass
class DuplicatePair:
    """Represents a pair of potentially duplicate requirements"""
    req_a_id: str
    req_b_id: str
    similarity_score: float
    duplicate_type: DuplicateType
    merge_recommendation: str  # "merge_a_into_b", "merge_b_into_a", "keep_both", "manual_review"
    reason: str


@dataclass
class DeduplicationResult:
    """Result of deduplication process"""
    original_count: int
    deduplicated_count: int
    duplicates_found: int
    merged_requirements: List[RequirementNode]
    duplicate_pairs: List[DuplicatePair]
    merge_log: List[str]

    # Quality metrics
    reduction_percentage: float = 0.0
    avg_similarity_of_merged: float = 0.0


class TFIDFDeduplicator:
    """
    TF-IDF based requirement deduplication engine.

    Uses TF-IDF vectorization and cosine similarity to identify
    semantically similar requirements, then merges them intelligently.

    Usage:
        deduplicator = TFIDFDeduplicator(similarity_threshold=0.85)
        result = deduplicator.deduplicate(requirements)
    """

    # Default configuration
    DEFAULT_SIMILARITY_THRESHOLD = 0.85
    EXACT_MATCH_THRESHOLD = 0.99
    NEAR_EXACT_THRESHOLD = 0.95
    SEMANTIC_THRESHOLD = 0.85
    PARTIAL_THRESHOLD = 0.70
    RELATED_THRESHOLD = 0.60

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_df: int = 1,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        stop_words: str = "english"
    ):
        """
        Initialize the deduplicator.

        Args:
            similarity_threshold: Minimum similarity to consider duplicate
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            ngram_range: N-gram range for vectorization
            stop_words: Stop words setting for vectorizer
        """
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words,
            lowercase=True,
            strip_accents="unicode"
        )
        self._seen_hashes: Set[str] = set()

    def deduplicate(
        self,
        requirements: List[RequirementNode],
        preserve_all_sources: bool = True
    ) -> DeduplicationResult:
        """
        Deduplicate a list of requirements.

        Args:
            requirements: List of RequirementNode objects to deduplicate
            preserve_all_sources: Whether to merge source info from duplicates

        Returns:
            DeduplicationResult with merged requirements and statistics
        """
        if not requirements:
            return DeduplicationResult(
                original_count=0,
                deduplicated_count=0,
                duplicates_found=0,
                merged_requirements=[],
                duplicate_pairs=[],
                merge_log=[]
            )

        original_count = len(requirements)
        merge_log = []

        # Step 1: Find exact duplicates using hash
        requirements, exact_pairs, exact_log = self._remove_exact_duplicates(requirements)
        merge_log.extend(exact_log)

        if len(requirements) < 2:
            return DeduplicationResult(
                original_count=original_count,
                deduplicated_count=len(requirements),
                duplicates_found=original_count - len(requirements),
                merged_requirements=requirements,
                duplicate_pairs=exact_pairs,
                merge_log=merge_log
            )

        # Step 2: Build TF-IDF matrix
        texts = [self._preprocess_text(req.text) for req in requirements]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except ValueError as e:
            logger.warning(f"TF-IDF vectorization failed: {e}")
            return DeduplicationResult(
                original_count=original_count,
                deduplicated_count=len(requirements),
                duplicates_found=len(exact_pairs),
                merged_requirements=requirements,
                duplicate_pairs=exact_pairs,
                merge_log=merge_log
            )

        # Step 3: Calculate pairwise similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Step 4: Find duplicate pairs above threshold
        semantic_pairs = self._find_duplicate_pairs(
            requirements, similarity_matrix
        )

        # Step 5: Merge duplicates
        merged_requirements, merge_actions = self._merge_duplicates(
            requirements,
            semantic_pairs,
            preserve_all_sources
        )
        merge_log.extend(merge_actions)

        # Calculate statistics
        all_pairs = exact_pairs + semantic_pairs
        avg_similarity = np.mean([p.similarity_score for p in all_pairs]) if all_pairs else 0.0

        result = DeduplicationResult(
            original_count=original_count,
            deduplicated_count=len(merged_requirements),
            duplicates_found=original_count - len(merged_requirements),
            merged_requirements=merged_requirements,
            duplicate_pairs=all_pairs,
            merge_log=merge_log,
            reduction_percentage=(1 - len(merged_requirements) / original_count) * 100 if original_count > 0 else 0,
            avg_similarity_of_merged=avg_similarity
        )

        logger.info(
            f"Deduplication complete: {original_count} -> {len(merged_requirements)} "
            f"({result.reduction_percentage:.1f}% reduction)"
        )

        return result

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF vectorization"""
        # Lowercase
        text = text.lower()

        # Remove section references (they cause false matches)
        text = re.sub(r"section\s+[a-z]\.\d+(?:\.\d+)*", "", text)
        text = re.sub(r"\b[a-z]\.\d+(?:\.\d+)*\b", "", text)

        # Remove FAR/DFARS references
        text = re.sub(r"(?:far|dfars)\s*\d+\.\d+[-\d]*", "", text)

        # Remove numbers (dates, amounts)
        text = re.sub(r"\b\d+(?:\.\d+)?(?:\s*%|\s*days?|\s*hours?)?\b", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _text_hash(self, text: str) -> str:
        """Create hash of normalized text"""
        normalized = self._preprocess_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _remove_exact_duplicates(
        self,
        requirements: List[RequirementNode]
    ) -> Tuple[List[RequirementNode], List[DuplicatePair], List[str]]:
        """Remove exact duplicates using text hash"""
        seen_hashes: Dict[str, RequirementNode] = {}
        unique_requirements = []
        duplicate_pairs = []
        merge_log = []

        for req in requirements:
            text_hash = self._text_hash(req.text)

            if text_hash in seen_hashes:
                # Found exact duplicate
                original = seen_hashes[text_hash]
                pair = DuplicatePair(
                    req_a_id=original.id,
                    req_b_id=req.id,
                    similarity_score=1.0,
                    duplicate_type=DuplicateType.EXACT,
                    merge_recommendation="merge_b_into_a",
                    reason="Identical text content"
                )
                duplicate_pairs.append(pair)

                # Merge source information
                if req.source and original.source:
                    original.referenced_by.append(req.id)
                    merge_log.append(
                        f"EXACT: Merged {req.id} into {original.id} "
                        f"(source: {req.source.document_name})"
                    )
            else:
                seen_hashes[text_hash] = req
                unique_requirements.append(req)

        return unique_requirements, duplicate_pairs, merge_log

    def _find_duplicate_pairs(
        self,
        requirements: List[RequirementNode],
        similarity_matrix: np.ndarray
    ) -> List[DuplicatePair]:
        """Find duplicate pairs based on similarity matrix"""
        pairs = []
        n = len(requirements)

        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i, j]

                if sim >= self.RELATED_THRESHOLD:
                    dup_type = self._classify_duplicate_type(sim)
                    recommendation = self._get_merge_recommendation(
                        requirements[i], requirements[j], sim, dup_type
                    )

                    pair = DuplicatePair(
                        req_a_id=requirements[i].id,
                        req_b_id=requirements[j].id,
                        similarity_score=float(sim),
                        duplicate_type=dup_type,
                        merge_recommendation=recommendation,
                        reason=self._get_duplicate_reason(
                            requirements[i], requirements[j], sim, dup_type
                        )
                    )
                    pairs.append(pair)

        return pairs

    def _classify_duplicate_type(self, similarity: float) -> DuplicateType:
        """Classify duplicate type based on similarity score"""
        if similarity >= self.EXACT_MATCH_THRESHOLD:
            return DuplicateType.EXACT
        elif similarity >= self.NEAR_EXACT_THRESHOLD:
            return DuplicateType.NEAR_EXACT
        elif similarity >= self.SEMANTIC_THRESHOLD:
            return DuplicateType.SEMANTIC
        elif similarity >= self.PARTIAL_THRESHOLD:
            return DuplicateType.PARTIAL
        else:
            return DuplicateType.RELATED

    def _get_merge_recommendation(
        self,
        req_a: RequirementNode,
        req_b: RequirementNode,
        similarity: float,
        dup_type: DuplicateType
    ) -> str:
        """Determine merge recommendation based on requirements"""
        # Don't merge if below threshold
        if similarity < self.similarity_threshold:
            return "keep_both"

        # Don't merge if different types (proposal instruction vs performance)
        if req_a.requirement_type != req_b.requirement_type:
            # Exception: PERFORMANCE and DELIVERABLE can be related
            compatible_types = {
                (RequirementType.PERFORMANCE, RequirementType.DELIVERABLE),
                (RequirementType.DELIVERABLE, RequirementType.PERFORMANCE),
            }
            if (req_a.requirement_type, req_b.requirement_type) not in compatible_types:
                return "keep_both"

        # Prefer higher confidence requirement
        if req_a.confidence == ConfidenceLevel.HIGH and req_b.confidence != ConfidenceLevel.HIGH:
            return "merge_b_into_a"
        elif req_b.confidence == ConfidenceLevel.HIGH and req_a.confidence != ConfidenceLevel.HIGH:
            return "merge_a_into_b"

        # Prefer requirement from more authoritative source
        authoritative_docs = ["MAIN_SOLICITATION", "STATEMENT_OF_WORK"]
        a_authoritative = req_a.source and req_a.source.document_type.value.upper() in authoritative_docs
        b_authoritative = req_b.source and req_b.source.document_type.value.upper() in authoritative_docs

        if a_authoritative and not b_authoritative:
            return "merge_b_into_a"
        elif b_authoritative and not a_authoritative:
            return "merge_a_into_b"

        # Prefer longer, more detailed requirement
        if len(req_a.text) > len(req_b.text) * 1.2:
            return "merge_b_into_a"
        elif len(req_b.text) > len(req_a.text) * 1.2:
            return "merge_a_into_b"

        # Default: merge newer into older (by ID)
        return "merge_b_into_a"

    def _get_duplicate_reason(
        self,
        req_a: RequirementNode,
        req_b: RequirementNode,
        similarity: float,
        dup_type: DuplicateType
    ) -> str:
        """Generate human-readable reason for duplicate detection"""
        reasons = []

        if dup_type == DuplicateType.EXACT:
            reasons.append("Identical text content")
        elif dup_type == DuplicateType.NEAR_EXACT:
            reasons.append(f"Near-identical content (similarity: {similarity:.2%})")
        elif dup_type == DuplicateType.SEMANTIC:
            reasons.append(f"Semantically equivalent (similarity: {similarity:.2%})")
        elif dup_type == DuplicateType.PARTIAL:
            reasons.append(f"Overlapping content (similarity: {similarity:.2%})")
        else:
            reasons.append(f"Related topic (similarity: {similarity:.2%})")

        # Add source context
        if req_a.source and req_b.source:
            if req_a.source.document_name == req_b.source.document_name:
                reasons.append(f"Same document: {req_a.source.document_name}")
            else:
                reasons.append(
                    f"Cross-document: {req_a.source.document_name} vs {req_b.source.document_name}"
                )

        return "; ".join(reasons)

    def _merge_duplicates(
        self,
        requirements: List[RequirementNode],
        pairs: List[DuplicatePair],
        preserve_all_sources: bool
    ) -> Tuple[List[RequirementNode], List[str]]:
        """Merge duplicate requirements based on pairs"""
        merge_log = []

        # Build merge map
        merge_into: Dict[str, str] = {}  # req_id -> target_id
        req_map = {req.id: req for req in requirements}

        # Process pairs that should be merged
        for pair in pairs:
            if pair.similarity_score < self.similarity_threshold:
                continue

            if pair.merge_recommendation == "merge_b_into_a":
                source_id, target_id = pair.req_b_id, pair.req_a_id
            elif pair.merge_recommendation == "merge_a_into_b":
                source_id, target_id = pair.req_a_id, pair.req_b_id
            elif pair.merge_recommendation == "manual_review":
                merge_log.append(
                    f"REVIEW: {pair.req_a_id} and {pair.req_b_id} may be duplicates "
                    f"({pair.duplicate_type.value}, {pair.similarity_score:.2%})"
                )
                continue
            else:
                continue

            # Handle transitive merges (A->B, B->C means A->C)
            while target_id in merge_into:
                target_id = merge_into[target_id]

            merge_into[source_id] = target_id

        # Apply merges
        merged_ids: Set[str] = set(merge_into.keys())

        for source_id, target_id in merge_into.items():
            if source_id in req_map and target_id in req_map:
                source_req = req_map[source_id]
                target_req = req_map[target_id]

                # Merge information
                if preserve_all_sources:
                    self._merge_requirement_info(source_req, target_req)

                merge_log.append(
                    f"MERGED: {source_id} -> {target_id} "
                    f"({pairs[0].duplicate_type.value if pairs else 'unknown'})"
                )

        # Return non-merged requirements
        result = [req for req in requirements if req.id not in merged_ids]

        return result, merge_log

    def _merge_requirement_info(
        self,
        source: RequirementNode,
        target: RequirementNode
    ) -> None:
        """Merge information from source into target requirement"""
        # Merge references
        target.referenced_by.append(source.id)
        target.references_to.extend(source.references_to)

        # Merge keywords (unique)
        target.keywords = list(set(target.keywords + source.keywords))

        # Merge entities
        target.entities = list(set(target.entities + source.entities))

        # Merge cross-document links
        if source.evaluated_by:
            target.evaluated_by.extend(source.evaluated_by)
        if source.instructed_by:
            target.instructed_by.extend(source.instructed_by)

        # Add source citation to context
        if source.source:
            citation = f"Also referenced in: {source.source.document_name}, {source.source.section_id}"
            if target.context_after:
                target.context_after += f" [{citation}]"
            else:
                target.context_after = f"[{citation}]"

    def find_related_requirements(
        self,
        requirements: List[RequirementNode],
        threshold: float = 0.60
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find related (but not duplicate) requirements.

        Useful for building requirement clusters and detecting
        cross-references that may need attention.

        Args:
            requirements: List of requirements to analyze
            threshold: Minimum similarity to consider related

        Returns:
            Dict mapping requirement ID to list of (related_id, similarity) tuples
        """
        if len(requirements) < 2:
            return {}

        # Build TF-IDF matrix
        texts = [self._preprocess_text(req.text) for req in requirements]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            return {}

        # Calculate similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Build relationship map
        related: Dict[str, List[Tuple[str, float]]] = {}

        for i, req in enumerate(requirements):
            related[req.id] = []
            for j, other in enumerate(requirements):
                if i != j:
                    sim = similarity_matrix[i, j]
                    if sim >= threshold:
                        related[req.id].append((other.id, float(sim)))

            # Sort by similarity
            related[req.id].sort(key=lambda x: x[1], reverse=True)

        return related


def deduplicate_requirements(
    requirements: List[RequirementNode],
    similarity_threshold: float = 0.85,
    preserve_sources: bool = True
) -> DeduplicationResult:
    """
    Convenience function to deduplicate requirements.

    Args:
        requirements: List of RequirementNode objects
        similarity_threshold: Minimum similarity to consider duplicate
        preserve_sources: Whether to merge source information

    Returns:
        DeduplicationResult with merged requirements
    """
    deduplicator = TFIDFDeduplicator(similarity_threshold=similarity_threshold)
    return deduplicator.deduplicate(requirements, preserve_all_sources=preserve_sources)
