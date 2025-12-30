# PropelAI RAG (Retrieval Augmented Generation) Layer
# Vector embeddings and semantic search for Company Library

from rag.embeddings import (
    EmbeddingService,
    EmbeddingResult,
    BaseEmbeddingProvider,
    OpenAIEmbeddingProvider,
    GoogleEmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_service,
)
from rag.chunker import (
    TextChunker,
    Chunk,
    ChunkStrategy,
    get_chunker,
)
from rag.search import (
    RAGSearch,
    SearchResult,
    SearchQuery,
    SearchMode,
    get_rag_search,
)
from rag.ingestion import (
    LibraryIngestionPipeline,
    IngestionResult,
    IngestionStatus,
    EntityType,
    ExtractedEntity,
    get_ingestion_pipeline,
)

__all__ = [
    # Embeddings
    "EmbeddingService",
    "EmbeddingResult",
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GoogleEmbeddingProvider",
    "LocalEmbeddingProvider",
    "get_embedding_service",
    # Chunking
    "TextChunker",
    "Chunk",
    "ChunkStrategy",
    "get_chunker",
    # Search
    "RAGSearch",
    "SearchResult",
    "SearchQuery",
    "SearchMode",
    "get_rag_search",
    # Ingestion
    "LibraryIngestionPipeline",
    "IngestionResult",
    "IngestionStatus",
    "EntityType",
    "ExtractedEntity",
    "get_ingestion_pipeline",
]
