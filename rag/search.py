"""
PropelAI RAG Search
Vector similarity search using pgvector for semantic retrieval
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np

from rag.embeddings import EmbeddingService, get_embedding_service
from rag.chunker import Chunk


class SearchMode(str, Enum):
    """Search modes for different use cases."""
    SIMILARITY = "similarity"      # Pure vector similarity
    HYBRID = "hybrid"              # Vector + keyword
    KEYWORD = "keyword"            # Traditional keyword search
    RERANKED = "reranked"          # Vector search with reranking


@dataclass
class SearchResult:
    """A single search result with relevance scoring."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source traceability
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None

    # Entity info (for library search)
    entity_type: Optional[str] = None  # resume, past_performance, capability
    entity_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section": self.section,
            "bbox": self.bbox,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "metadata": self.metadata,
        }


@dataclass
class SearchQuery:
    """Search query with configuration."""
    query: str
    mode: SearchMode = SearchMode.SIMILARITY
    top_k: int = 10
    min_score: float = 0.5

    # Filters
    document_ids: Optional[List[str]] = None
    entity_types: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None

    # Boost specific entity types
    entity_boost: Optional[Dict[str, float]] = None


class RAGSearch:
    """
    RAG Search engine with pgvector backend.
    Supports similarity, hybrid, and reranked search modes.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        db_pool: Optional[Any] = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.db_pool = db_pool
        self._connection = None

    async def _get_connection(self):
        """Get database connection (lazy initialization)."""
        if self._connection is None and self.db_pool:
            self._connection = await self.db_pool.acquire()
        return self._connection

    async def search(
        self,
        query: SearchQuery,
        tenant_id: str,
    ) -> List[SearchResult]:
        """
        Perform semantic search across the library.

        Args:
            query: SearchQuery with parameters
            tenant_id: Tenant ID for isolation

        Returns:
            List of SearchResults ranked by relevance
        """
        if query.mode == SearchMode.SIMILARITY:
            return await self._similarity_search(query, tenant_id)
        elif query.mode == SearchMode.HYBRID:
            return await self._hybrid_search(query, tenant_id)
        elif query.mode == SearchMode.KEYWORD:
            return await self._keyword_search(query, tenant_id)
        elif query.mode == SearchMode.RERANKED:
            return await self._reranked_search(query, tenant_id)
        else:
            return await self._similarity_search(query, tenant_id)

    async def _similarity_search(
        self,
        query: SearchQuery,
        tenant_id: str,
    ) -> List[SearchResult]:
        """Pure vector similarity search using pgvector."""
        # Generate query embedding
        query_result = await self.embedding_service.embed(query.query)
        query_embedding = query_result.embedding

        # Build SQL query with pgvector
        sql = """
            SELECT
                le.id,
                le.chunk_text,
                le.embedding <=> $1::vector AS distance,
                le.chunk_index,
                le.page_number,
                le.section,
                le.bbox,
                le.metadata,
                ld.id AS document_id,
                ld.filename AS document_name,
                ld.entity_type,
                ld.entity_name
            FROM library_embeddings le
            JOIN library_documents ld ON le.document_id = ld.id
            WHERE ld.tenant_id = $2
        """
        params = [query_embedding, tenant_id]
        param_idx = 3

        # Apply filters
        if query.document_ids:
            sql += f" AND ld.id = ANY(${param_idx}::uuid[])"
            params.append(query.document_ids)
            param_idx += 1

        if query.entity_types:
            sql += f" AND ld.entity_type = ANY(${param_idx}::text[])"
            params.append(query.entity_types)
            param_idx += 1

        # Order by distance and limit
        sql += f"""
            ORDER BY distance
            LIMIT ${param_idx}
        """
        params.append(query.top_k * 2)  # Fetch extra for filtering

        # Execute query
        conn = await self._get_connection()
        if conn is None:
            # Return mock results for testing without DB
            return self._mock_search_results(query)

        rows = await conn.fetch(sql, *params)

        # Convert to SearchResults
        results = []
        for row in rows:
            # Convert distance to similarity score (1 - distance for cosine)
            score = 1.0 - float(row["distance"])

            if score < query.min_score:
                continue

            # Apply entity boost if specified
            if query.entity_boost and row["entity_type"] in query.entity_boost:
                score *= query.entity_boost[row["entity_type"]]

            results.append(SearchResult(
                chunk_id=str(row["id"]),
                text=row["chunk_text"],
                score=score,
                document_id=str(row["document_id"]),
                document_name=row["document_name"],
                page_number=row["page_number"],
                section=row["section"],
                bbox=row["bbox"],
                entity_type=row["entity_type"],
                entity_name=row["entity_name"],
                metadata=row["metadata"] or {},
            ))

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.top_k]

    async def _hybrid_search(
        self,
        query: SearchQuery,
        tenant_id: str,
    ) -> List[SearchResult]:
        """Hybrid search combining vector similarity and keyword matching."""
        # Get vector results
        vector_results = await self._similarity_search(query, tenant_id)

        # Get keyword results
        keyword_query = SearchQuery(
            query=query.query,
            mode=SearchMode.KEYWORD,
            top_k=query.top_k,
            min_score=0.3,  # Lower threshold for keywords
            document_ids=query.document_ids,
            entity_types=query.entity_types,
        )
        keyword_results = await self._keyword_search(keyword_query, tenant_id)

        # Merge results using Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}

        k = 60  # RRF constant

        # Score vector results
        for rank, result in enumerate(vector_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1.0 / (k + rank + 1)
            result_map[result.chunk_id] = result

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + 1.0 / (k + rank + 1)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Update scores and sort
        results = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = result_map[chunk_id]
            result.score = rrf_score
            results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:query.top_k]

    async def _keyword_search(
        self,
        query: SearchQuery,
        tenant_id: str,
    ) -> List[SearchResult]:
        """Traditional full-text keyword search using PostgreSQL ts_vector."""
        # Build PostgreSQL full-text search query
        sql = """
            SELECT
                le.id,
                le.chunk_text,
                ts_rank(
                    to_tsvector('english', le.chunk_text),
                    plainto_tsquery('english', $1)
                ) AS rank,
                le.page_number,
                le.section,
                le.bbox,
                le.metadata,
                ld.id AS document_id,
                ld.filename AS document_name,
                ld.entity_type,
                ld.entity_name
            FROM library_embeddings le
            JOIN library_documents ld ON le.document_id = ld.id
            WHERE ld.tenant_id = $2
              AND to_tsvector('english', le.chunk_text) @@ plainto_tsquery('english', $1)
        """
        params = [query.query, tenant_id]
        param_idx = 3

        if query.document_ids:
            sql += f" AND ld.id = ANY(${param_idx}::uuid[])"
            params.append(query.document_ids)
            param_idx += 1

        if query.entity_types:
            sql += f" AND ld.entity_type = ANY(${param_idx}::text[])"
            params.append(query.entity_types)
            param_idx += 1

        sql += f" ORDER BY rank DESC LIMIT ${param_idx}"
        params.append(query.top_k)

        conn = await self._get_connection()
        if conn is None:
            return []

        rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            score = float(row["rank"])
            if score < query.min_score:
                continue

            results.append(SearchResult(
                chunk_id=str(row["id"]),
                text=row["chunk_text"],
                score=score,
                document_id=str(row["document_id"]),
                document_name=row["document_name"],
                page_number=row["page_number"],
                section=row["section"],
                bbox=row["bbox"],
                entity_type=row["entity_type"],
                entity_name=row["entity_name"],
                metadata=row["metadata"] or {},
            ))

        return results

    async def _reranked_search(
        self,
        query: SearchQuery,
        tenant_id: str,
    ) -> List[SearchResult]:
        """
        Vector search with cross-encoder reranking.
        Fetches more candidates and reranks with a more accurate model.
        """
        # Fetch more candidates for reranking
        expanded_query = SearchQuery(
            query=query.query,
            mode=SearchMode.SIMILARITY,
            top_k=query.top_k * 3,  # Fetch 3x candidates
            min_score=query.min_score * 0.8,  # Lower initial threshold
            document_ids=query.document_ids,
            entity_types=query.entity_types,
        )

        candidates = await self._similarity_search(expanded_query, tenant_id)

        if not candidates:
            return []

        # Rerank using cross-encoder (if available)
        try:
            reranked = await self._cross_encoder_rerank(query.query, candidates)
        except ImportError:
            # Fall back to original ranking if cross-encoder not available
            reranked = candidates

        return reranked[:query.top_k]

    async def _cross_encoder_rerank(
        self,
        query: str,
        candidates: List[SearchResult],
    ) -> List[SearchResult]:
        """Rerank using a cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return candidates

        # Load cross-encoder (cached)
        if not hasattr(self, "_cross_encoder"):
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Prepare pairs
        pairs = [(query, result.text) for result in candidates]

        # Score in executor (cross-encoder is sync)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._cross_encoder.predict(pairs)
        )

        # Update scores
        for result, score in zip(candidates, scores):
            result.score = float(score)

        # Sort by new scores
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _mock_search_results(self, query: SearchQuery) -> List[SearchResult]:
        """Return mock results for testing without database."""
        return [
            SearchResult(
                chunk_id="mock-1",
                text=f"Mock result for: {query.query}",
                score=0.95,
                document_id="mock-doc-1",
                document_name="sample_resume.pdf",
                page_number=1,
                entity_type="resume",
                entity_name="John Smith",
            )
        ]

    async def find_similar_entities(
        self,
        entity_id: str,
        entity_type: str,
        tenant_id: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Find entities similar to a given entity."""
        conn = await self._get_connection()
        if conn is None:
            return []

        # Get the entity's average embedding
        sql = """
            SELECT AVG(embedding) as avg_embedding
            FROM library_embeddings le
            JOIN library_documents ld ON le.document_id = ld.id
            WHERE ld.id = $1 AND ld.tenant_id = $2
        """
        row = await conn.fetchrow(sql, entity_id, tenant_id)

        if not row or not row["avg_embedding"]:
            return []

        # Search for similar entities
        sql = """
            SELECT DISTINCT ON (ld.id)
                ld.id AS document_id,
                ld.filename AS document_name,
                ld.entity_type,
                ld.entity_name,
                le.embedding <=> $1::vector AS distance
            FROM library_embeddings le
            JOIN library_documents ld ON le.document_id = ld.id
            WHERE ld.tenant_id = $2
              AND ld.id != $3
              AND ld.entity_type = $4
            ORDER BY ld.id, distance
            LIMIT $5
        """

        rows = await conn.fetch(
            sql,
            row["avg_embedding"],
            tenant_id,
            entity_id,
            entity_type,
            top_k,
        )

        return [
            SearchResult(
                chunk_id=str(r["document_id"]),
                text="",
                score=1.0 - float(r["distance"]),
                document_id=str(r["document_id"]),
                document_name=r["document_name"],
                entity_type=r["entity_type"],
                entity_name=r["entity_name"],
            )
            for r in rows
        ]

    async def search_for_requirement(
        self,
        requirement_text: str,
        tenant_id: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search library for content matching a requirement.
        Used by the proposal generation engine.

        Args:
            requirement_text: The requirement to match
            tenant_id: Tenant ID
            entity_types: Filter to specific entity types

        Returns:
            Relevant library content for the requirement
        """
        query = SearchQuery(
            query=requirement_text,
            mode=SearchMode.HYBRID,
            top_k=10,
            min_score=0.6,
            entity_types=entity_types,
            entity_boost={
                "past_performance": 1.2,  # Boost past performance
                "capability": 1.1,
            },
        )

        return await self.search(query, tenant_id)


# Convenience function
def get_rag_search(
    embedding_service: Optional[EmbeddingService] = None,
    db_pool: Optional[Any] = None,
) -> RAGSearch:
    """Get a configured RAG search instance."""
    return RAGSearch(
        embedding_service=embedding_service,
        db_pool=db_pool,
    )
