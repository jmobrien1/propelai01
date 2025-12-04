"""
Evidence Hunter Agent
=====================

RAG-based agent that searches project documents to find evidence
supporting OASIS+ scoring claims.

Uses semantic search over document embeddings to locate specific
paragraphs that demonstrate required capabilities.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from .models import (
    Project,
    ProjectClaim,
    ScoringCriteria,
    DocumentChunk,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# Evidence search prompts for different claim types
EVIDENCE_PROMPTS = {
    "surge": {
        "keywords": ["surge", "rapid expansion", "ramp up", "urgent", "accelerat"],
        "prompt": "Find text demonstrating surge capability or rapid staffing increase",
    },
    "oconus": {
        "keywords": ["oconus", "overseas", "international", "foreign", "outside continental"],
        "prompt": "Find text proving work performed outside the continental United States",
    },
    "clearance": {
        "keywords": ["clearance", "secret", "top secret", "ts/sci", "classified", "security"],
        "prompt": "Find text demonstrating security clearance requirements or classified work",
    },
    "cost_reimbursement": {
        "keywords": ["cost-plus", "cost reimbursement", "cpff", "cpaf", "cpif"],
        "prompt": "Find text indicating cost-reimbursement contract type",
    },
    "cmmi": {
        "keywords": ["cmmi", "capability maturity", "level 3", "level 5", "sei"],
        "prompt": "Find text demonstrating CMMI certification or process maturity",
    },
    "iso": {
        "keywords": ["iso 9001", "iso 27001", "iso 20000", "quality management system"],
        "prompt": "Find text demonstrating ISO certification",
    },
    "agile": {
        "keywords": ["agile", "scrum", "devops", "sprint", "continuous integration", "ci/cd"],
        "prompt": "Find text demonstrating agile development methodology",
    },
    "systems_engineering": {
        "keywords": ["systems engineering", "requirements analysis", "system design", "integration"],
        "prompt": "Find text demonstrating systems engineering work",
    },
    "program_management": {
        "keywords": ["program management", "pmp", "project manager", "milestone", "schedule"],
        "prompt": "Find text demonstrating program/project management",
    },
    "cybersecurity": {
        "keywords": ["cybersecurity", "information security", "fisma", "nist", "rmf", "ato"],
        "prompt": "Find text demonstrating cybersecurity or information security work",
    },
}


@dataclass
class EvidenceMatch:
    """A potential evidence match from document search"""
    chunk_id: str
    document_id: str
    project_id: str
    content: str
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None

    # Scoring
    relevance_score: float = 0.0
    keyword_matches: List[str] = field(default_factory=list)

    # LLM verification
    llm_verified: bool = False
    llm_confidence: float = 0.0
    llm_explanation: str = ""

    # Extracted quote
    exact_quote: str = ""
    quote_start: int = 0
    quote_end: int = 0


@dataclass
class EvidenceSearchResult:
    """Results of an evidence search for a claim"""
    criteria_id: str
    project_id: str
    search_keywords: List[str]

    # Matches found
    matches: List[EvidenceMatch] = field(default_factory=list)
    best_match: Optional[EvidenceMatch] = None

    # Search metadata
    total_chunks_searched: int = 0
    search_duration_ms: int = 0

    # Recommendation
    recommended_status: VerificationStatus = VerificationStatus.UNVERIFIED
    confidence_score: float = 0.0


class EvidenceHunter:
    """
    Searches project documents for evidence supporting scoring claims.

    Uses a combination of:
    1. Keyword matching for initial filtering
    2. Semantic similarity via vector embeddings
    3. LLM verification for final confirmation
    """

    def __init__(
        self,
        embedding_function: Optional[callable] = None,
        llm_function: Optional[callable] = None,
        similarity_threshold: float = 0.7,
        top_k: int = 5,
    ):
        """
        Initialize the Evidence Hunter.

        Args:
            embedding_function: Function to generate embeddings for queries
            llm_function: Function to call LLM for verification
            similarity_threshold: Minimum similarity score to consider
            top_k: Number of top matches to consider
        """
        self.embedding_function = embedding_function
        self.llm_function = llm_function
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def search_evidence(
        self,
        criteria: ScoringCriteria,
        project: Project,
        chunks: List[DocumentChunk],
    ) -> EvidenceSearchResult:
        """
        Search for evidence supporting a scoring criteria claim.

        Args:
            criteria: The scoring criteria to find evidence for
            project: The project being claimed
            chunks: Document chunks to search through

        Returns:
            EvidenceSearchResult with matches and recommendations
        """
        start_time = datetime.now()

        # Get search keywords for this criteria type
        keywords = self._get_search_keywords(criteria)

        result = EvidenceSearchResult(
            criteria_id=criteria.criteria_id,
            project_id=project.project_id,
            search_keywords=keywords,
            total_chunks_searched=len(chunks),
        )

        if not chunks:
            logger.warning(f"No chunks to search for project {project.project_id}")
            return result

        # Phase 1: Keyword filtering
        keyword_matches = self._keyword_search(chunks, keywords)
        logger.info(f"Keyword search found {len(keyword_matches)} potential matches")

        # Phase 2: Semantic search (if embedding function available)
        if self.embedding_function and len(keyword_matches) < self.top_k:
            semantic_matches = self._semantic_search(chunks, criteria.description)
            # Merge with keyword matches
            seen_ids = {m.chunk_id for m in keyword_matches}
            for match in semantic_matches:
                if match.chunk_id not in seen_ids:
                    keyword_matches.append(match)

        # Sort by relevance score
        keyword_matches.sort(key=lambda m: m.relevance_score, reverse=True)
        matches = keyword_matches[:self.top_k]

        # Phase 3: LLM verification (if available)
        if self.llm_function and matches:
            matches = self._llm_verify(matches, criteria)

        result.matches = matches

        # Determine best match and recommendation
        if matches:
            result.best_match = max(matches, key=lambda m: m.relevance_score)
            result.confidence_score = result.best_match.relevance_score

            if result.best_match.llm_verified:
                result.recommended_status = VerificationStatus.PENDING
            elif result.confidence_score >= 0.8:
                result.recommended_status = VerificationStatus.PENDING
            else:
                result.recommended_status = VerificationStatus.UNVERIFIED

        # Calculate duration
        duration = datetime.now() - start_time
        result.search_duration_ms = int(duration.total_seconds() * 1000)

        return result

    def _get_search_keywords(self, criteria: ScoringCriteria) -> List[str]:
        """Get search keywords for a criteria"""
        # Use criteria's own keywords if available
        if criteria.evidence_keywords:
            return criteria.evidence_keywords

        # Try to match to known evidence types
        desc_lower = criteria.description.lower()
        for evidence_type, config in EVIDENCE_PROMPTS.items():
            if evidence_type in desc_lower or any(kw in desc_lower for kw in config["keywords"][:2]):
                return config["keywords"]

        # Extract keywords from description
        return self._extract_keywords_from_text(criteria.description)

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "this",
            "that", "these", "those", "it", "its", "as", "if", "than", "so",
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]

        # Get unique keywords, preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:10]  # Limit to 10 keywords

    def _keyword_search(
        self,
        chunks: List[DocumentChunk],
        keywords: List[str]
    ) -> List[EvidenceMatch]:
        """Search chunks using keyword matching"""
        matches = []

        for chunk in chunks:
            content_lower = chunk.content.lower()
            matched_keywords = []

            for keyword in keywords:
                if keyword.lower() in content_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                # Calculate relevance score based on keyword coverage
                score = len(matched_keywords) / len(keywords) if keywords else 0

                # Boost score for multiple mentions
                mention_count = sum(
                    content_lower.count(kw.lower()) for kw in matched_keywords
                )
                if mention_count > len(matched_keywords):
                    score *= min(1.5, 1 + (mention_count - len(matched_keywords)) * 0.1)

                match = EvidenceMatch(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    project_id=chunk.project_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    bbox=chunk.bbox,
                    relevance_score=min(1.0, score),
                    keyword_matches=matched_keywords,
                )
                matches.append(match)

        return matches

    def _semantic_search(
        self,
        chunks: List[DocumentChunk],
        query: str
    ) -> List[EvidenceMatch]:
        """Search chunks using semantic similarity"""
        if not self.embedding_function:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_function(query)

            # Calculate similarity with each chunk
            matches = []
            for chunk in chunks:
                if chunk.embedding:
                    similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                    if similarity >= self.similarity_threshold:
                        match = EvidenceMatch(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            project_id=chunk.project_id,
                            content=chunk.content,
                            page_number=chunk.page_number,
                            bbox=chunk.bbox,
                            relevance_score=similarity,
                        )
                        matches.append(match)

            matches.sort(key=lambda m: m.relevance_score, reverse=True)
            return matches[:self.top_k]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _llm_verify(
        self,
        matches: List[EvidenceMatch],
        criteria: ScoringCriteria
    ) -> List[EvidenceMatch]:
        """Use LLM to verify evidence matches"""
        if not self.llm_function:
            return matches

        for match in matches:
            try:
                prompt = f"""Analyze this text and determine if it provides evidence for the following OASIS+ scoring criterion:

CRITERION: {criteria.description}

TEXT TO ANALYZE:
{match.content}

Instructions:
1. Does this text explicitly demonstrate the required capability?
2. If YES, extract the exact quote (verbatim) that proves the criterion
3. Rate your confidence from 0.0 to 1.0

Respond in this exact format:
VERIFIED: YES or NO
CONFIDENCE: 0.X
QUOTE: "exact quote here" or "N/A"
EXPLANATION: brief explanation
"""

                response = self.llm_function(prompt)
                parsed = self._parse_llm_response(response)

                match.llm_verified = parsed.get("verified", False)
                match.llm_confidence = parsed.get("confidence", 0.0)
                match.exact_quote = parsed.get("quote", "")
                match.llm_explanation = parsed.get("explanation", "")

                # Update relevance score with LLM confidence
                if match.llm_verified:
                    match.relevance_score = max(
                        match.relevance_score,
                        match.llm_confidence
                    )

            except Exception as e:
                logger.error(f"LLM verification failed: {e}")
                continue

        return matches

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse structured LLM response"""
        result = {
            "verified": False,
            "confidence": 0.0,
            "quote": "",
            "explanation": "",
        }

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("VERIFIED:"):
                result["verified"] = "YES" in line.upper()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = re.search(r'[\d.]+', line)
                    if conf_str:
                        result["confidence"] = float(conf_str.group())
                except ValueError:
                    pass
            elif line.startswith("QUOTE:"):
                quote = line.replace("QUOTE:", "").strip().strip('"')
                if quote != "N/A":
                    result["quote"] = quote
            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line.replace("EXPLANATION:", "").strip()

        return result

    def create_claim_from_evidence(
        self,
        evidence: EvidenceSearchResult,
        criteria: ScoringCriteria,
    ) -> Optional[ProjectClaim]:
        """Create a ProjectClaim from evidence search results"""
        if not evidence.best_match:
            return None

        match = evidence.best_match

        claim = ProjectClaim(
            claim_id=f"CLAIM-{criteria.criteria_id}-{evidence.project_id}",
            project_id=evidence.project_id,
            criteria_id=criteria.criteria_id,
            claimed_points=criteria.max_points,
            verified_points=criteria.max_points if match.llm_verified else 0,
            evidence_snippet=match.exact_quote or match.content[:500],
            evidence_page_number=match.page_number,
            evidence_document_id=match.document_id,
            evidence_bbox=match.bbox,
            status=evidence.recommended_status,
            ai_confidence_score=evidence.confidence_score,
        )

        return claim

    def batch_search(
        self,
        criteria_list: List[ScoringCriteria],
        project: Project,
        chunks: List[DocumentChunk],
    ) -> Dict[str, EvidenceSearchResult]:
        """
        Search for evidence for multiple criteria at once.

        Args:
            criteria_list: List of scoring criteria to search for
            project: The project being claimed
            chunks: Document chunks to search through

        Returns:
            Dictionary mapping criteria_id to search results
        """
        results = {}

        for criteria in criteria_list:
            logger.info(f"Searching evidence for {criteria.criteria_id}: {criteria.description[:50]}...")
            result = self.search_evidence(criteria, project, chunks)
            results[criteria.criteria_id] = result

            if result.best_match:
                logger.info(
                    f"  Found evidence with confidence {result.confidence_score:.2f} "
                    f"on page {result.best_match.page_number}"
                )
            else:
                logger.info("  No evidence found")

        return results
