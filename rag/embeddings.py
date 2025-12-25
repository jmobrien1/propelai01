"""
PropelAI Embedding Service
Generates vector embeddings using OpenAI, Google, or local models
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dataclasses import dataclass
import hashlib


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    token_count: Optional[int] = None

    @property
    def cache_key(self) -> str:
        """Generate cache key for this embedding."""
        return hashlib.md5(f"{self.model}:{self.text}".encode()).hexdigest()


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimension of embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._dimensions = 1536 if "small" in model else 3072
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched)."""
        client = self._get_client()

        # OpenAI supports up to 2048 texts per batch
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])

        return all_embeddings

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self.model


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google Vertex AI embedding provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-004"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self._dimensions = 768
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
        return self._client

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding using Google's embedding model."""
        client = self._get_client()

        # Run in executor since google-generativeai is sync
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.embed_content(
                model=f"models/{self.model}",
                content=text,
                task_type="retrieval_document",
            )
        )
        return result["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Google doesn't have native batch, so we parallelize
        tasks = [self.embed_text(text) for text in texts]
        return await asyncio.gather(*tasks)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self.model


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.
    Useful for development or air-gapped environments.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = model
        self._dimensions = 384  # MiniLM default
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model)
            self._dimensions = self._encoder.get_sentence_embedding_dimension()
        return self._encoder

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding locally."""
        encoder = self._get_encoder()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: encoder.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for batch."""
        encoder = self._get_encoder()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: encoder.encode(texts, convert_to_numpy=True)
        )
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self.model


class EmbeddingService:
    """
    Unified embedding service with caching and fallback support.
    """

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        self.cache_enabled = cache_enabled
        self._cache: dict = {}
        self._provider = self._init_provider(primary_provider)

    def _init_provider(self, provider_name: Optional[str] = None) -> BaseEmbeddingProvider:
        """Initialize the embedding provider based on available API keys."""
        provider = provider_name or os.getenv("EMBEDDING_PROVIDER", "auto")

        if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
            return OpenAIEmbeddingProvider()
        elif provider == "google" or (provider == "auto" and os.getenv("GOOGLE_API_KEY")):
            return GoogleEmbeddingProvider()
        else:
            # Fall back to local embeddings
            try:
                return LocalEmbeddingProvider()
            except ImportError:
                raise RuntimeError(
                    "No embedding provider available. Set OPENAI_API_KEY, "
                    "GOOGLE_API_KEY, or install sentence-transformers."
                )

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._provider.dimensions

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._provider.model_name

    async def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text with caching.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        # Check cache
        cache_key = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Generate embedding
        embedding = await self._provider.embed_text(text)

        result = EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.model_name,
            dimensions=len(embedding),
        )

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = result

        return result

    async def embed_many(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResults
        """
        # Check cache for already embedded texts
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
            if self.cache_enabled and cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Embed uncached texts
        if texts_to_embed:
            embeddings = await self._provider.embed_batch(texts_to_embed)

            for idx, (text, embedding) in zip(indices_to_embed, zip(texts_to_embed, embeddings)):
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self.model_name,
                    dimensions=len(embedding),
                )
                results[idx] = result

                if self.cache_enabled:
                    cache_key = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
                    self._cache[cache_key] = result

        return results

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
