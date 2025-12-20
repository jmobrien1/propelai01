"""
PropelAI v4.0: Vector Store for Company Library Semantic Search

This module provides embedding generation and vector similarity search
using PostgreSQL with pgvector extension.

Features:
- Embedding generation with multiple providers (Voyage AI, OpenAI, or simple TF-IDF fallback)
- Semantic search across capabilities, past performances, key personnel
- Hybrid search combining vector similarity with keyword filtering
- Batch embedding for bulk imports

Usage:
    from api.vector_store import VectorStore, get_vector_store

    store = await get_vector_store()

    # Add capability with embedding
    await store.add_capability(company_id, "Cloud Migration", "Expert AWS/Azure migration...")

    # Search for relevant capabilities
    results = await store.search_capabilities("need cloud infrastructure expertise", top_k=5)

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (use Render's Internal Database URL)
    VOYAGE_API_KEY: Voyage AI API key (Anthropic's recommended embedding provider)
    OPENAI_API_KEY: OpenAI API key (alternative)

    If no API key is set, uses simple TF-IDF based similarity (no external calls).
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Async database
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Voyage AI for embeddings (Anthropic's recommended provider)
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False

# OpenAI for embeddings (alternative)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Fallback to httpx for API calls
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Configuration
EMBEDDING_DIMENSION = 1536  # Works with both Voyage and OpenAI
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Convert postgres:// to postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


@dataclass
class SearchResult:
    """Result from vector similarity search"""
    id: str
    content_type: str  # capability, past_performance, key_personnel, differentiator
    name: str
    description: str
    similarity_score: float
    metadata: Dict[str, Any]


class EmbeddingGenerator:
    """
    Generates embeddings using available providers.

    Priority order:
    1. Voyage AI (VOYAGE_API_KEY) - Anthropic's recommended provider
    2. OpenAI (OPENAI_API_KEY) - Alternative
    3. Simple hash-based fallback - No external API needed
    """

    def __init__(self):
        self.voyage_key = os.environ.get("VOYAGE_API_KEY", "")
        self.openai_key = os.environ.get("OPENAI_API_KEY", "")

        self.provider = None
        self.client = None

        # Try Voyage AI first (Anthropic's partner)
        if self.voyage_key and VOYAGE_AVAILABLE:
            self.provider = "voyage"
            self.client = voyageai.Client(api_key=self.voyage_key)
            self.model = "voyage-2"  # or voyage-large-2 for better quality
            print("=== Using Voyage AI for embeddings ===")
        # Fall back to OpenAI
        elif self.openai_key and OPENAI_AVAILABLE:
            self.provider = "openai"
            self.client = openai.OpenAI(api_key=self.openai_key)
            self.model = "text-embedding-3-small"
            print("=== Using OpenAI for embeddings ===")
        # Use simple fallback (no API needed)
        else:
            self.provider = "simple"
            print("=== Using simple hash-based embeddings (no API key configured) ===")

    async def generate(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        try:
            if self.provider == "voyage":
                result = self.client.embed([text], model=self.model)
                return result.embeddings[0]

            elif self.provider == "openai":
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding

            elif self.provider == "simple":
                return self._simple_embedding(text)

            return None

        except Exception as e:
            print(f"Embedding generation failed: {e}")
            # Fall back to simple embedding on error
            return self._simple_embedding(text)

    async def generate_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        try:
            if self.provider == "voyage":
                result = self.client.embed(texts, model=self.model)
                return result.embeddings

            elif self.provider == "openai":
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                return [item.embedding for item in response.data]

            elif self.provider == "simple":
                return [self._simple_embedding(t) for t in texts]

            return [None] * len(texts)

        except Exception as e:
            print(f"Batch embedding generation failed: {e}")
            return [self._simple_embedding(t) for t in texts]

    def _simple_embedding(self, text: str) -> List[float]:
        """
        Simple hash-based embedding for when no API is available.

        This creates a deterministic embedding based on the text content.
        Not as semantically meaningful as real embeddings, but enables
        basic similarity search based on word overlap.
        """
        import math

        # Normalize text
        text = text.lower().strip()
        words = text.split()

        # Create a simple bag-of-words style embedding
        embedding = [0.0] * EMBEDDING_DIMENSION

        for i, word in enumerate(words):
            # Hash each word to get consistent indices
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)

            # Distribute word influence across multiple dimensions
            for j in range(min(10, EMBEDDING_DIMENSION // 100)):
                idx = (word_hash + j * 7919) % EMBEDDING_DIMENSION  # 7919 is prime
                value = ((word_hash >> (j * 4)) & 0xF) / 15.0 - 0.5  # Normalize to [-0.5, 0.5]
                embedding[idx] += value / (1 + math.log(1 + i))  # Weight by position

        # Normalize to unit vector
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


class VectorStore:
    """
    PostgreSQL + pgvector based vector store for Company Library.

    Provides semantic search across:
    - Capabilities (technical competencies)
    - Past Performances (relevant project experience)
    - Key Personnel (team expertise)
    - Differentiators (competitive advantages)
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.session_factory = None
        self.embedding_generator = EmbeddingGenerator()
        self._initialized = False

    async def initialize(self):
        """Initialize database connection"""
        if self._initialized:
            return

        if not self.database_url or not SQLALCHEMY_AVAILABLE:
            print("Warning: Vector store not available (no database URL or SQLAlchemy)")
            return

        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
            )
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            self._initialized = True
            print("=== PropelAI Vector Store initialized ===")
        except Exception as e:
            print(f"Vector store initialization failed: {e}")

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False

    # =========================================================================
    # Capability Operations
    # =========================================================================

    async def add_capability(
        self,
        company_id: str,
        name: str,
        description: str,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Add a capability with auto-generated embedding"""
        if not self._initialized:
            await self.initialize()

        if not self._initialized:
            return None

        # Generate embedding from name + description
        embed_text = f"{name}. {description}"
        embedding = await self.embedding_generator.generate(embed_text)

        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO capabilities (company_id, name, description, category, keywords, embedding)
                    VALUES (:company_id, :name, :description, :category, :keywords, :embedding)
                    RETURNING id
                """),
                {
                    "company_id": company_id,
                    "name": name,
                    "description": description,
                    "category": category,
                    "keywords": keywords or [],
                    "embedding": embedding,
                }
            )
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else None

    async def search_capabilities(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search capabilities by semantic similarity"""
        return await self._vector_search(
            table="capabilities",
            query=query,
            top_k=top_k,
            filters={
                "category": category_filter,
                "company_id": company_id,
            },
            content_type="capability",
        )

    # =========================================================================
    # Past Performance Operations
    # =========================================================================

    async def add_past_performance(
        self,
        company_id: str,
        project_name: str,
        description: str,
        client_name: Optional[str] = None,
        client_agency: Optional[str] = None,
        contract_number: Optional[str] = None,
        contract_value: Optional[float] = None,
        period_of_performance: Optional[str] = None,
        relevance_keywords: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Add past performance with auto-generated embedding"""
        if not self._initialized:
            await self.initialize()

        if not self._initialized:
            return None

        # Generate embedding
        embed_text = f"{project_name}. {description}"
        if client_agency:
            embed_text += f" Client: {client_agency}."
        embedding = await self.embedding_generator.generate(embed_text)

        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO past_performances
                    (company_id, project_name, client_name, client_agency, contract_number,
                     contract_value, period_of_performance, description, relevance_keywords,
                     metrics, embedding)
                    VALUES (:company_id, :project_name, :client_name, :client_agency,
                            :contract_number, :contract_value, :period_of_performance,
                            :description, :relevance_keywords, :metrics, :embedding)
                    RETURNING id
                """),
                {
                    "company_id": company_id,
                    "project_name": project_name,
                    "client_name": client_name,
                    "client_agency": client_agency,
                    "contract_number": contract_number,
                    "contract_value": contract_value,
                    "period_of_performance": period_of_performance,
                    "description": description,
                    "relevance_keywords": relevance_keywords or [],
                    "metrics": json.dumps(metrics) if metrics else None,
                    "embedding": embedding,
                }
            )
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else None

    async def search_past_performances(
        self,
        query: str,
        top_k: int = 5,
        agency_filter: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search past performances by semantic similarity"""
        return await self._vector_search(
            table="past_performances",
            query=query,
            top_k=top_k,
            filters={
                "client_agency": agency_filter,
                "company_id": company_id,
            },
            content_type="past_performance",
            name_column="project_name",
        )

    # =========================================================================
    # Key Personnel Operations
    # =========================================================================

    async def add_key_personnel(
        self,
        company_id: str,
        name: str,
        bio: str,
        title: Optional[str] = None,
        role: Optional[str] = None,
        years_experience: Optional[int] = None,
        clearance_level: Optional[str] = None,
        certifications: Optional[List[str]] = None,
        expertise_areas: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Add key personnel with auto-generated embedding"""
        if not self._initialized:
            await self.initialize()

        if not self._initialized:
            return None

        # Generate embedding
        embed_text = f"{name}, {title or role or 'Professional'}. {bio}"
        if expertise_areas:
            embed_text += f" Expertise: {', '.join(expertise_areas)}."
        embedding = await self.embedding_generator.generate(embed_text)

        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO key_personnel
                    (company_id, name, title, role, years_experience, clearance_level,
                     certifications, bio, expertise_areas, embedding)
                    VALUES (:company_id, :name, :title, :role, :years_experience,
                            :clearance_level, :certifications, :bio, :expertise_areas, :embedding)
                    RETURNING id
                """),
                {
                    "company_id": company_id,
                    "name": name,
                    "title": title,
                    "role": role,
                    "years_experience": years_experience,
                    "clearance_level": clearance_level,
                    "certifications": certifications or [],
                    "bio": bio,
                    "expertise_areas": expertise_areas or [],
                    "embedding": embedding,
                }
            )
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else None

    async def search_key_personnel(
        self,
        query: str,
        top_k: int = 5,
        role_filter: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search key personnel by semantic similarity"""
        return await self._vector_search(
            table="key_personnel",
            query=query,
            top_k=top_k,
            filters={
                "role": role_filter,
                "company_id": company_id,
            },
            content_type="key_personnel",
            description_column="bio",
        )

    # =========================================================================
    # Differentiator Operations
    # =========================================================================

    async def add_differentiator(
        self,
        company_id: str,
        title: str,
        description: str,
        category: Optional[str] = None,
        proof_points: Optional[List[str]] = None,
        competitor_comparison: Optional[str] = None,
    ) -> Optional[str]:
        """Add differentiator with auto-generated embedding"""
        if not self._initialized:
            await self.initialize()

        if not self._initialized:
            return None

        # Generate embedding
        embed_text = f"{title}. {description}"
        if proof_points:
            embed_text += f" Proof: {'; '.join(proof_points)}."
        embedding = await self.embedding_generator.generate(embed_text)

        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO differentiators
                    (company_id, title, description, category, proof_points,
                     competitor_comparison, embedding)
                    VALUES (:company_id, :title, :description, :category,
                            :proof_points, :competitor_comparison, :embedding)
                    RETURNING id
                """),
                {
                    "company_id": company_id,
                    "title": title,
                    "description": description,
                    "category": category,
                    "proof_points": proof_points or [],
                    "competitor_comparison": competitor_comparison,
                    "embedding": embedding,
                }
            )
            await session.commit()
            row = result.fetchone()
            return str(row[0]) if row else None

    async def search_differentiators(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search differentiators by semantic similarity"""
        return await self._vector_search(
            table="differentiators",
            query=query,
            top_k=top_k,
            filters={
                "category": category_filter,
                "company_id": company_id,
            },
            content_type="differentiator",
            name_column="title",
        )

    # =========================================================================
    # Unified Search
    # =========================================================================

    async def search_all(
        self,
        query: str,
        top_k: int = 10,
        company_id: Optional[str] = None,
        include_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search across all content types and return unified results.

        Args:
            query: Search query
            top_k: Total number of results (distributed across types)
            company_id: Optional company filter
            include_types: List of types to include (capability, past_performance,
                          key_personnel, differentiator). None = all.

        Returns:
            Combined results sorted by similarity score
        """
        types = include_types or ["capability", "past_performance", "key_personnel", "differentiator"]
        per_type_k = max(1, top_k // len(types))

        all_results = []

        if "capability" in types:
            results = await self.search_capabilities(query, per_type_k, company_id=company_id)
            all_results.extend(results)

        if "past_performance" in types:
            results = await self.search_past_performances(query, per_type_k, company_id=company_id)
            all_results.extend(results)

        if "key_personnel" in types:
            results = await self.search_key_personnel(query, per_type_k, company_id=company_id)
            all_results.extend(results)

        if "differentiator" in types:
            results = await self.search_differentiators(query, per_type_k, company_id=company_id)
            all_results.extend(results)

        # Sort by similarity and return top_k
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:top_k]

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _vector_search(
        self,
        table: str,
        query: str,
        top_k: int,
        filters: Dict[str, Any],
        content_type: str,
        name_column: str = "name",
        description_column: str = "description",
    ) -> List[SearchResult]:
        """Internal method for vector similarity search"""
        if not self._initialized:
            await self.initialize()

        if not self._initialized:
            return []

        # Generate query embedding
        query_embedding = await self.embedding_generator.generate(query)
        if not query_embedding:
            return []

        # Build filter conditions
        where_clauses = []
        params = {
            "query_embedding": query_embedding,
            "top_k": top_k,
        }

        for key, value in filters.items():
            if value is not None:
                where_clauses.append(f"{key} = :{key}")
                params[key] = value

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Execute similarity search
        sql = f"""
            SELECT
                id,
                {name_column} as name,
                {description_column} as description,
                1 - (embedding <=> :query_embedding::vector) as similarity
            FROM {table}
            {where_sql}
            ORDER BY embedding <=> :query_embedding::vector
            LIMIT :top_k
        """

        try:
            async with self.session_factory() as session:
                result = await session.execute(text(sql), params)
                rows = result.fetchall()

                return [
                    SearchResult(
                        id=str(row[0]),
                        content_type=content_type,
                        name=row[1],
                        description=row[2][:500] if row[2] else "",
                        similarity_score=float(row[3]) if row[3] else 0.0,
                        metadata={},
                    )
                    for row in rows
                ]
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []


# Global instance
_vector_store: Optional[VectorStore] = None


async def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        await _vector_store.initialize()
    return _vector_store


# Check availability
VECTOR_STORE_AVAILABLE = SQLALCHEMY_AVAILABLE and bool(DATABASE_URL)
