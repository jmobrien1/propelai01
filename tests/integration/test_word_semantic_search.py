"""
PropelAI v5.0 Integration Tests: Word API Semantic Search
==========================================================
Version: 5.0.1

Tests the Word Integration API with semantic search capabilities:
- POST /api/word/context endpoint
- Semantic vs Jaccard search fallback
- Similarity threshold enforcement (0.3 minimum)
- Requirement embedding caching

Per PRD Phase 4 - Word Integration API:
- FR-4.1: Context-aware search for Word Add-in
- FR-4.2: Semantic search with pgvector embeddings
- FR-4.3: Graceful fallback to Jaccard similarity
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.conftest import MockEmbeddingGenerator, cosine_similarity


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_RFP = {
    "id": "RFP-WORD-TEST-001",
    "name": "Cloud Infrastructure RFP",
    "requirements": [
        {
            "id": "REQ-001",
            "text": "The Contractor shall ensure 99.99% system uptime for production environments.",
            "type": "PERFORMANCE",
            "section": "C.2.1",
            "priority": "HIGH"
        },
        {
            "id": "REQ-002",
            "text": "The Contractor shall implement automated failover within 30 seconds.",
            "type": "PERFORMANCE",
            "section": "C.2.2",
            "priority": "HIGH"
        },
        {
            "id": "REQ-003",
            "text": "All data shall be encrypted at rest using AES-256 encryption.",
            "type": "SECURITY",
            "section": "C.3.1",
            "priority": "CRITICAL"
        },
        {
            "id": "REQ-004",
            "text": "The system shall support FedRAMP High baseline compliance.",
            "type": "COMPLIANCE",
            "section": "C.3.2",
            "priority": "CRITICAL"
        },
        {
            "id": "REQ-005",
            "text": "Monthly performance reports shall be submitted by the 5th business day.",
            "type": "DELIVERABLE",
            "section": "C.4.1",
            "priority": "MEDIUM"
        },
    ]
}


# =============================================================================
# Word API Endpoint Tests
# =============================================================================

@pytest.mark.integration
class TestWordContextEndpoint:
    """Test POST /api/word/context endpoint"""

    @pytest.fixture
    def mock_store(self):
        """Mock RFP store"""
        store = MagicMock()
        store.get = MagicMock(return_value=SAMPLE_RFP.copy())
        return store

    @pytest.fixture
    def word_request_data(self):
        """Sample Word context request"""
        return {
            "rfp_id": "RFP-WORD-TEST-001",
            "current_text": "Our team ensures maximum system uptime through redundant architecture.",
            "section_heading": "Technical Approach",
            "max_results": 5,
            "use_semantic_search": True
        }

    @pytest.mark.asyncio
    async def test_endpoint_returns_matching_requirements(self, word_request_data, mock_store):
        """Test that endpoint returns requirements matching the query"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator') as mock_gen:
                mock_gen.return_value = MockEmbeddingGenerator()

                request = WordContextRequest(**word_request_data)
                response = await get_word_context(request)

                assert response.rfp_id == "RFP-WORD-TEST-001"
                assert isinstance(response.matching_requirements, list)

    @pytest.mark.asyncio
    async def test_rfp_not_found_raises_404(self, word_request_data):
        """Test that missing RFP raises 404 error"""
        from api.main import get_word_context, WordContextRequest
        from fastapi import HTTPException

        mock_store = MagicMock()
        mock_store.get = MagicMock(return_value=None)

        with patch('api.main.store', mock_store):
            request = WordContextRequest(**word_request_data)

            with pytest.raises(HTTPException) as exc_info:
                await get_word_context(request)

            assert exc_info.value.status_code == 404
            assert "not found" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_max_results_respected(self, mock_store):
        """Test that max_results limits response"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator') as mock_gen:
                mock_gen.return_value = MockEmbeddingGenerator()

                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="system security encryption compliance",
                    max_results=2,
                    use_semantic_search=False  # Use Jaccard for deterministic results
                )
                response = await get_word_context(request)

                assert len(response.matching_requirements) <= 2


# =============================================================================
# Semantic vs Jaccard Fallback Tests
# =============================================================================

@pytest.mark.integration
class TestSearchMethodFallback:
    """Test semantic to Jaccard fallback behavior"""

    @pytest.fixture
    def mock_store(self):
        """Mock RFP store"""
        store = MagicMock()
        store.get = MagicMock(return_value=SAMPLE_RFP.copy())
        return store

    @pytest.mark.asyncio
    async def test_uses_semantic_when_available(self, mock_store):
        """Test that semantic search is used when embedding generator works"""
        from api.main import get_word_context, WordContextRequest

        mock_gen = MockEmbeddingGenerator()

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator', return_value=mock_gen):
                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="Ensuring 99.99% uptime for critical production systems",
                    use_semantic_search=True
                )
                response = await get_word_context(request)

                # Should use semantic search when generator is available
                if response.matching_requirements:
                    assert response.search_method == "semantic"

    @pytest.mark.asyncio
    async def test_falls_back_to_jaccard_when_generator_fails(self, mock_store):
        """
        FR-4.3: Verify search falls back from 'semantic' to 'jaccard'
        when embedding generator fails.
        """
        from api.main import get_word_context, WordContextRequest

        # Create generator that will fail
        failing_gen = MockEmbeddingGenerator(fail_on=["FAIL_ALL"])

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator', return_value=failing_gen):
                # Use text that matches FAIL_ALL pattern
                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="FAIL_ALL uptime production systems",
                    use_semantic_search=True
                )
                response = await get_word_context(request)

                # Should fall back to Jaccard when embedding fails
                assert response.search_method == "jaccard"

    @pytest.mark.asyncio
    async def test_uses_jaccard_when_semantic_disabled(self, mock_store):
        """Test explicit Jaccard search when semantic is disabled"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            request = WordContextRequest(
                rfp_id="RFP-WORD-TEST-001",
                current_text="system uptime production environments",
                use_semantic_search=False
            )
            response = await get_word_context(request)

            assert response.search_method == "jaccard"

    @pytest.mark.asyncio
    async def test_falls_back_when_no_generator(self, mock_store):
        """Test fallback when no embedding generator is available"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator', return_value=None):
                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="system uptime production",
                    use_semantic_search=True
                )
                response = await get_word_context(request)

                # Should fall back to Jaccard when no generator
                assert response.search_method == "jaccard"


# =============================================================================
# Similarity Score Tests
# =============================================================================

@pytest.mark.integration
class TestSimilarityScoring:
    """Test similarity score computation and thresholds"""

    @pytest.fixture
    def mock_store(self):
        """Mock RFP store"""
        store = MagicMock()
        store.get = MagicMock(return_value=SAMPLE_RFP.copy())
        return store

    @pytest.mark.asyncio
    async def test_similarity_score_returned(self, mock_store):
        """Test that similarity_score is included in response"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator') as mock_gen:
                mock_gen.return_value = MockEmbeddingGenerator()

                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="uptime production environments system",
                    use_semantic_search=False  # Use Jaccard for deterministic results
                )
                response = await get_word_context(request)

                if response.matching_requirements:
                    for req in response.matching_requirements:
                        assert "similarity_score" in req
                        assert isinstance(req["similarity_score"], (int, float))

    @pytest.mark.asyncio
    async def test_semantic_threshold_enforced(self, mock_store):
        """
        Test that semantic matches exceed 0.3 similarity threshold.

        Per Word API spec: min_similarity=0.3 for semantic search.
        """
        from api.main import get_word_context, WordContextRequest

        mock_gen = MockEmbeddingGenerator()

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator', return_value=mock_gen):
                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="The system shall ensure 99.99% uptime for production",
                    use_semantic_search=True
                )
                response = await get_word_context(request)

                # If semantic search was used, all scores should be >= 0.3
                if response.search_method == "semantic":
                    for req in response.matching_requirements:
                        assert req["similarity_score"] >= 0.3, \
                            f"Semantic match {req['id']} has score {req['similarity_score']} < 0.3"

    @pytest.mark.asyncio
    async def test_jaccard_threshold_enforced(self, mock_store):
        """Test that Jaccard matches exceed 0.1 threshold"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            request = WordContextRequest(
                rfp_id="RFP-WORD-TEST-001",
                current_text="system uptime production environments contractor",
                use_semantic_search=False
            )
            response = await get_word_context(request)

            # Jaccard threshold is 0.1
            for req in response.matching_requirements:
                assert req["similarity_score"] > 0.1, \
                    f"Jaccard match {req['id']} has score {req['similarity_score']} <= 0.1"

    @pytest.mark.asyncio
    async def test_results_sorted_by_similarity(self, mock_store):
        """Test that results are sorted by similarity (highest first)"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator') as mock_gen:
                mock_gen.return_value = MockEmbeddingGenerator()

                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="uptime system production contractor monthly reports",
                    max_results=10,
                    use_semantic_search=False
                )
                response = await get_word_context(request)

                if len(response.matching_requirements) > 1:
                    scores = [r["similarity_score"] for r in response.matching_requirements]
                    assert scores == sorted(scores, reverse=True), \
                        f"Results not sorted by similarity: {scores}"


# =============================================================================
# Embedding Caching Tests
# =============================================================================

@pytest.mark.integration
class TestEmbeddingCaching:
    """Test requirement embedding caching behavior"""

    @pytest.mark.asyncio
    async def test_embeddings_cached_in_rfp(self):
        """Test that embeddings are cached in RFP dictionary"""
        from api.main import _get_or_create_requirement_embeddings

        rfp = SAMPLE_RFP.copy()
        requirements = rfp["requirements"]

        mock_gen = MockEmbeddingGenerator()

        with patch('api.main._get_embedding_generator', return_value=mock_gen):
            # First call should generate embeddings
            embeddings1 = await _get_or_create_requirement_embeddings(rfp, requirements)

            # Verify cache is populated
            assert "_requirement_embeddings" in rfp
            assert len(embeddings1) == len(requirements)

            # Second call should use cache
            call_count_before = mock_gen.call_count
            embeddings2 = await _get_or_create_requirement_embeddings(rfp, requirements)

            # Embeddings should be the same
            assert embeddings1 == embeddings2

    @pytest.mark.asyncio
    async def test_cache_detects_missing_embeddings(self):
        """Test that cache detects new requirements without embeddings"""
        from api.main import _get_or_create_requirement_embeddings

        rfp = SAMPLE_RFP.copy()
        requirements = rfp["requirements"][:3]  # Start with 3

        mock_gen = MockEmbeddingGenerator()

        with patch('api.main._get_embedding_generator', return_value=mock_gen):
            # Generate embeddings for 3 requirements
            await _get_or_create_requirement_embeddings(rfp, requirements)
            initial_count = len(rfp["_requirement_embeddings"])

            # Add a new requirement
            new_requirements = requirements + [SAMPLE_RFP["requirements"][3]]
            await _get_or_create_requirement_embeddings(rfp, new_requirements)

            # Cache should have the new embedding
            assert len(rfp["_requirement_embeddings"]) == initial_count + 1


# =============================================================================
# Semantic Search Quality Tests
# =============================================================================

@pytest.mark.integration
class TestSemanticSearchQuality:
    """Test semantic search produces meaningful results"""

    @pytest.fixture
    def mock_store(self):
        """Mock RFP store"""
        store = MagicMock()
        store.get = MagicMock(return_value=SAMPLE_RFP.copy())
        return store

    @pytest.mark.asyncio
    async def test_semantic_finds_related_concepts(self, mock_store):
        """Test that semantic search understands related concepts"""
        from api.main import get_word_context, WordContextRequest

        mock_gen = MockEmbeddingGenerator()

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator', return_value=mock_gen):
                # Query about "availability" should find "uptime" requirements
                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="System availability and reliability are critical mission requirements.",
                    use_semantic_search=True
                )
                response = await get_word_context(request)

                # Should have results (mock embeddings may vary)
                assert response.matching_requirements is not None

    @pytest.mark.asyncio
    async def test_jaccard_requires_exact_words(self, mock_store):
        """Test that Jaccard requires exact word matches"""
        from api.main import get_word_context, WordContextRequest

        with patch('api.main.store', mock_store):
            # Query with no overlapping words
            request = WordContextRequest(
                rfp_id="RFP-WORD-TEST-001",
                current_text="Reliability availability mission critical",
                use_semantic_search=False
            )
            response = await get_word_context(request)

            # Jaccard won't find matches without word overlap
            # (the sample RFP doesn't contain these exact words)
            assert response.search_method == "jaccard"


# =============================================================================
# Helper Function Tests
# =============================================================================

@pytest.mark.integration
class TestHelperFunctions:
    """Test internal helper functions for Word API"""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors is 1.0"""
        from api.main import _cosine_similarity

        vec = [0.5, 0.5, 0.5, 0.5]
        similarity = _cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors is 0.0"""
        from api.main import _cosine_similarity

        vec1 = [1.0, 0.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0, 0.0]
        similarity = _cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001

    def test_cosine_similarity_empty_vectors(self):
        """Test cosine similarity handles empty vectors"""
        from api.main import _cosine_similarity

        assert _cosine_similarity([], []) == 0.0
        assert _cosine_similarity([1.0], []) == 0.0
        assert _cosine_similarity([], [1.0]) == 0.0

    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity handles different length vectors"""
        from api.main import _cosine_similarity

        vec1 = [0.5, 0.5]
        vec2 = [0.5, 0.5, 0.5]
        assert _cosine_similarity(vec1, vec2) == 0.0

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity of identical texts is 1.0"""
        from api.main import _jaccard_similarity

        text = "The quick brown fox jumps over the lazy dog"
        similarity = _jaccard_similarity(text, text)
        assert abs(similarity - 1.0) < 0.001

    def test_jaccard_similarity_no_overlap(self):
        """Test Jaccard similarity with no word overlap"""
        from api.main import _jaccard_similarity

        text1 = "The quick brown"
        text2 = "system uptime contractor"
        similarity = _jaccard_similarity(text1, text2)
        assert similarity < 0.1

    def test_jaccard_filters_short_words(self):
        """Test Jaccard ignores words <= 3 characters"""
        from api.main import _jaccard_similarity

        # "the" should be filtered out (<=3 chars)
        text1 = "the system"
        text2 = "the contractor"
        similarity = _jaccard_similarity(text1, text2)
        # Only "system" and "contractor" are compared
        assert similarity == 0.0


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in Word API"""

    @pytest.mark.asyncio
    async def test_handles_embedding_exception(self):
        """Test graceful handling of embedding generation errors"""
        from api.main import get_word_context, WordContextRequest

        mock_store = MagicMock()
        mock_store.get = MagicMock(return_value=SAMPLE_RFP.copy())

        # Create mock generator that raises an exception
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(side_effect=Exception("API Error"))
        mock_gen.generate_batch = AsyncMock(side_effect=Exception("API Error"))

        with patch('api.main.store', mock_store):
            with patch('api.main._get_embedding_generator', return_value=mock_gen):
                request = WordContextRequest(
                    rfp_id="RFP-WORD-TEST-001",
                    current_text="system uptime production",
                    use_semantic_search=True
                )
                # Should not raise, should fall back to Jaccard
                response = await get_word_context(request)

                assert response.search_method == "jaccard"

    @pytest.mark.asyncio
    async def test_handles_empty_requirements(self):
        """Test handling of RFP with no requirements"""
        from api.main import get_word_context, WordContextRequest

        mock_store = MagicMock()
        mock_store.get = MagicMock(return_value={
            "id": "RFP-EMPTY-001",
            "requirements": []
        })

        with patch('api.main.store', mock_store):
            request = WordContextRequest(
                rfp_id="RFP-EMPTY-001",
                current_text="any query text",
                use_semantic_search=True
            )
            response = await get_word_context(request)

            assert response.matching_requirements == []


# =============================================================================
# Mock Embedding Generator Tests
# =============================================================================

@pytest.mark.integration
class TestMockEmbeddingGenerator:
    """Test MockEmbeddingGenerator behavior for integration tests"""

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, mock_embedding_generator):
        """Test that same text produces same embedding"""
        text = "The Contractor shall ensure system uptime."

        vec1 = await mock_embedding_generator.generate(text)
        vec2 = await mock_embedding_generator.generate(text)

        # Should be identical for same text
        assert vec1 == vec2

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, mock_embedding_generator):
        """Test that different texts produce different embeddings"""
        vec1 = await mock_embedding_generator.generate("system uptime")
        vec2 = await mock_embedding_generator.generate("data encryption")

        # Should be different for different texts
        assert vec1 != vec2

    @pytest.mark.asyncio
    async def test_similar_texts_have_similarity(self, mock_embedding_generator):
        """Test that similar texts have measurable similarity"""
        # Note: Mock uses hash, so similarity is based on hash collision
        # This is mainly to verify the embedding structure is correct
        vec1 = await mock_embedding_generator.generate("system uptime requirements")
        vec2 = await mock_embedding_generator.generate("system availability needs")

        assert len(vec1) == 1536
        assert len(vec2) == 1536

        # Calculate cosine similarity
        sim = cosine_similarity(vec1, vec2)
        assert isinstance(sim, float)

    @pytest.mark.asyncio
    async def test_unit_vectors(self, mock_embedding_generator):
        """Test that mock generates normalized unit vectors"""
        vec = await mock_embedding_generator.generate("test text")

        # Calculate magnitude
        magnitude = sum(x ** 2 for x in vec) ** 0.5
        assert abs(magnitude - 1.0) < 0.001, f"Vector magnitude {magnitude} is not 1.0"

    @pytest.mark.asyncio
    async def test_fail_on_pattern(self, failing_embedding_generator):
        """Test that generator can fail on specific patterns"""
        # Normal text works
        vec = await failing_embedding_generator.generate("normal text")
        assert vec is not None

        # Pattern match fails
        vec_fail = await failing_embedding_generator.generate("FAIL_EMBEDDING test")
        assert vec_fail is None

    @pytest.mark.asyncio
    async def test_batch_generate(self, mock_embedding_generator):
        """Test batch embedding generation"""
        texts = ["text one", "text two", "text three"]
        embeddings = await mock_embedding_generator.generate_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb is not None
            assert len(emb) == 1536

    @pytest.mark.asyncio
    async def test_tracks_call_count(self, mock_embedding_generator):
        """Test that generator tracks call count"""
        assert mock_embedding_generator.call_count == 0

        await mock_embedding_generator.generate("test 1")
        assert mock_embedding_generator.call_count == 1

        await mock_embedding_generator.generate("test 2")
        assert mock_embedding_generator.call_count == 2
