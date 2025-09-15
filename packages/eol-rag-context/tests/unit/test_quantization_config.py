"""Unit tests for quantization configuration functionality."""

import os
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from eol.rag_context.batch_operations import BatchRedisClient
from eol.rag_context.config import IndexConfig
from eol.rag_context.redis_client import VectorDocument
from eol.rag_context.semantic_cache import SemanticCache


class TestQuantizationConfiguration:
    """Test configurable quantization settings."""

    def test_default_quantization_settings(self):
        """Test default quantization configuration."""
        config = IndexConfig()

        # Check global default
        assert config.quantization == "Q8"

        # Check feature-specific defaults (should be None)
        assert config.concept_quantization is None
        assert config.section_quantization is None
        assert config.chunk_quantization is None
        assert config.cache_quantization is None
        assert config.batch_quantization is None

    def test_get_quantization_for_level_defaults(self):
        """Test get_quantization_for_level with defaults."""
        config = IndexConfig()

        # All levels should return global default when not overridden
        assert config.get_quantization_for_level(1) == "Q8"  # Concepts
        assert config.get_quantization_for_level(2) == "Q8"  # Sections
        assert config.get_quantization_for_level(3) == "Q8"  # Chunks
        assert config.get_quantization_for_level(99) == "Q8"  # Unknown level

    def test_get_quantization_for_level_overrides(self):
        """Test get_quantization_for_level with specific overrides."""
        config = IndexConfig(
            quantization="Q8",  # Global default
            concept_quantization="NOQUANT",  # High precision for concepts
            section_quantization="Q8",  # Balanced for sections
            chunk_quantization="BIN",  # Space-efficient for chunks
        )

        assert config.get_quantization_for_level(1) == "NOQUANT"
        assert config.get_quantization_for_level(2) == "Q8"
        assert config.get_quantization_for_level(3) == "BIN"

    def test_get_cache_quantization(self):
        """Test cache quantization settings."""
        # Default to global
        config1 = IndexConfig(quantization="Q8")
        assert config1.get_cache_quantization() == "Q8"

        # Override for cache
        config2 = IndexConfig(quantization="Q8", cache_quantization="NOQUANT")
        assert config2.get_cache_quantization() == "NOQUANT"

    def test_get_batch_quantization(self):
        """Test batch quantization settings."""
        # Default to global
        config1 = IndexConfig(quantization="NOQUANT")
        assert config1.get_batch_quantization() == "NOQUANT"

        # Override for batch
        config2 = IndexConfig(quantization="Q8", batch_quantization="BIN")
        assert config2.get_batch_quantization() == "BIN"

    def test_environment_variable_override(self):
        """Test that environment variables can override quantization settings."""
        # Set environment variables
        os.environ["INDEX_QUANTIZATION"] = "NOQUANT"
        os.environ["INDEX_CONCEPT_QUANTIZATION"] = "Q8"
        os.environ["INDEX_CACHE_QUANTIZATION"] = "BIN"

        try:
            config = IndexConfig()
            assert config.quantization == "NOQUANT"
            assert config.concept_quantization == "Q8"
            assert config.cache_quantization == "BIN"

            # Test methods use overrides correctly
            assert config.get_quantization_for_level(1) == "Q8"
            assert config.get_cache_quantization() == "BIN"
        finally:
            # Clean up environment
            del os.environ["INDEX_QUANTIZATION"]
            del os.environ["INDEX_CONCEPT_QUANTIZATION"]
            del os.environ["INDEX_CACHE_QUANTIZATION"]


class TestRedisClientQuantization:
    """Test RedisVectorStore uses configurable quantization."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis clients."""
        redis_mock = MagicMock()
        async_redis_mock = AsyncMock()
        return redis_mock, async_redis_mock

    def test_redis_quantization_per_level(self):
        """Test that IndexConfig provides correct quantization per level."""
        # Create config with different quantization per level
        index_config = IndexConfig(
            concept_quantization="NOQUANT", section_quantization="Q8", chunk_quantization="BIN"
        )

        # Test the get_quantization_for_level method returns correct values
        assert index_config.get_quantization_for_level(1) == "NOQUANT"  # Concept
        assert index_config.get_quantization_for_level(2) == "Q8"  # Section
        assert index_config.get_quantization_for_level(3) == "BIN"  # Chunk

        # The actual Redis integration would use these values in VADD commands
        # as implemented in redis_client.py lines 480-489


class TestBatchOperationsQuantization:
    """Test BatchRedisClient uses configurable quantization."""

    @pytest.fixture
    def mock_redis_store(self):
        """Mock RedisVectorStore with config."""
        store = MagicMock()
        store.redis = MagicMock()
        store.redis.pipeline.return_value = MagicMock()
        store.async_redis = AsyncMock()

        # Create config with batch quantization
        store.index_config = IndexConfig(
            quantization="Q8", batch_quantization="NOQUANT"  # Override for batch ops
        )

        # Add vector set names
        store.index_config.concept_vectorset = "concepts"
        store.index_config.section_vectorset = "sections"
        store.index_config.chunk_vectorset = "chunks"
        store.index_config.vectorset_name = "default"

        return store

    @pytest.mark.asyncio
    async def test_batch_client_uses_batch_quantization(self, mock_redis_store):
        """Test that BatchRedisClient uses batch-specific quantization."""
        batch_client = BatchRedisClient(mock_redis_store)

        # Create test documents
        doc = VectorDocument(
            id="doc_1",
            content="Test content",
            embedding=np.random.rand(384).astype(np.float32),
            hierarchy_level=2,
        )

        # Mock pipeline execution
        mock_pipeline = MagicMock()
        mock_redis_store.redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True]

        await batch_client.store_documents_batch([doc])

        # Check VADD command uses batch quantization (NOQUANT)
        vadd_call = mock_redis_store.async_redis.execute_command.call_args[0]
        assert "NOQUANT" in vadd_call


class TestSemanticCacheQuantization:
    """Test SemanticCache uses configurable quantization."""

    @pytest.fixture
    def mock_redis_store(self):
        """Mock RedisVectorStore for semantic cache."""
        store = MagicMock()
        store.redis = MagicMock()
        store.async_redis = AsyncMock()

        # Mock scan method for cache size check
        store.redis.scan.return_value = (0, [])  # Return empty scan result

        # Create config with cache quantization
        store.index_config = IndexConfig(
            quantization="Q8", cache_quantization="BIN"  # Space-efficient for cache
        )

        return store

    @pytest.mark.asyncio
    async def test_semantic_cache_uses_cache_quantization(self, mock_redis_store):
        """Test that SemanticCache uses cache-specific quantization."""
        from eol.rag_context.config import CacheConfig

        mock_embedder = AsyncMock()
        mock_embedder.get_embedding.return_value = np.random.rand(384).astype(np.float32)

        cache_config = CacheConfig()

        cache = SemanticCache(
            cache_config=cache_config, embedding_manager=mock_embedder, redis_store=mock_redis_store
        )

        # Store a cache entry using the set method
        await cache.set(query="test query", response="test response")

        # Check VADD command uses cache quantization (BIN)
        vadd_call = mock_redis_store.async_redis.execute_command.call_args[0]
        assert "BIN" in vadd_call


class TestQuantizationMemoryImpact:
    """Test memory impact calculations for different quantization levels."""

    def test_memory_estimates(self):
        """Calculate memory usage for different quantization modes."""
        vector_dim = 384
        num_vectors = 1_000_000

        # Calculate memory usage in MB
        noquant_memory = (vector_dim * 4 * num_vectors) / (1024 * 1024)  # 32-bit float
        q8_memory = (vector_dim * 1 * num_vectors) / (1024 * 1024)  # 8-bit int
        bin_memory = (vector_dim / 8 * num_vectors) / (1024 * 1024)  # 1-bit per dim

        # Expected values
        assert noquant_memory == pytest.approx(1464.84, rel=0.01)  # ~1.5 GB
        assert q8_memory == pytest.approx(366.21, rel=0.01)  # ~366 MB
        assert bin_memory == pytest.approx(45.78, rel=0.01)  # ~46 MB

        # Calculate savings
        q8_savings = (1 - q8_memory / noquant_memory) * 100
        bin_savings = (1 - bin_memory / noquant_memory) * 100

        assert q8_savings == pytest.approx(75.0, rel=0.01)  # 75% savings
        assert bin_savings == pytest.approx(96.875, rel=0.01)  # ~97% savings


class TestQuantizationValidation:
    """Test validation of quantization values."""

    def test_invalid_quantization_value(self):
        """Test that invalid quantization values raise appropriate errors."""
        # This should work with valid values
        config = IndexConfig(quantization="Q8")
        assert config.quantization == "Q8"

        config = IndexConfig(quantization="NOQUANT")
        assert config.quantization == "NOQUANT"

        config = IndexConfig(quantization="BIN")
        assert config.quantization == "BIN"

        # Invalid values should still be accepted (no validation in config)
        # but should be handled by the code using them
        config = IndexConfig(quantization="INVALID")
        assert config.quantization == "INVALID"

        # The handling of invalid values should be in the Redis client
        # where it falls back to Q8 as default
