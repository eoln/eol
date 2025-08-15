"""Focused tests for semantic_cache.py module.

This test file contains meaningful tests for the SemanticCache class, extracted from
coverage booster files and enhanced with real functionality testing.

"""

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from eol.rag_context.config import CacheConfig
from eol.rag_context.semantic_cache import CachedQuery, SemanticCache


class TestCachedQuery:
    """Test the CachedQuery dataclass."""

    def test_cached_query_creation(self):
        """Test CachedQuery creation with all fields."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        timestamp = time.time()

        cached_query = CachedQuery(
            query="What is Python?",
            response="Python is a programming language",
            embedding=embedding,
            timestamp=timestamp,
            hit_count=5,
            metadata={"source": "test", "tokens": 10},
        )

        assert cached_query.query == "What is Python?"
        assert cached_query.response == "Python is a programming language"
        assert np.array_equal(cached_query.embedding, embedding)
        assert cached_query.timestamp == timestamp
        assert cached_query.hit_count == 5
        assert cached_query.metadata["source"] == "test"

    def test_cached_query_defaults(self):
        """Test CachedQuery with default values."""
        embedding = np.array([0.1, 0.2], dtype=np.float32)

        cached_query = CachedQuery(
            query="test query", response="test response", embedding=embedding
        )

        assert cached_query.hit_count == 0
        assert isinstance(cached_query.timestamp, float)
        assert cached_query.metadata == {}


class TestSemanticCache:
    """Test the SemanticCache class with real functionality."""

    @pytest.fixture
    def cache_config(self):
        """Create a test cache configuration."""
        return CacheConfig(
            enabled=True,
            ttl_seconds=3600,
            similarity_threshold=0.9,
            target_hit_rate=0.31,
            adaptive_threshold=True,
            max_cache_size=100,
        )

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager."""
        mock = MagicMock()
        mock.get_embedding = AsyncMock(
            return_value=np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        mock.config = MagicMock()
        mock.config.dimension = 384
        return mock

    @pytest.fixture
    def mock_redis_store(self):
        """Create a mock Redis store."""
        mock = MagicMock()
        mock.async_redis = MagicMock()
        mock.redis = MagicMock()
        return mock

    @pytest.fixture
    def semantic_cache(self, cache_config, mock_embedding_manager, mock_redis_store):
        """Create a SemanticCache instance for testing."""
        return SemanticCache(cache_config, mock_embedding_manager, mock_redis_store)

    def test_cache_initialization(self, semantic_cache, cache_config):
        """Test cache initialization sets up correct attributes."""
        assert semantic_cache.config == cache_config
        assert semantic_cache.stats["queries"] == 0
        assert semantic_cache.stats["hits"] == 0
        assert semantic_cache.stats["misses"] == 0
        assert semantic_cache.adaptive_threshold == cache_config.similarity_threshold
        assert semantic_cache.similarity_scores == []

    @pytest.mark.asyncio
    async def test_cache_disabled_behavior(
        self, mock_embedding_manager, mock_redis_store
    ):
        """Test cache behavior when disabled."""
        disabled_config = CacheConfig(enabled=False)
        cache = SemanticCache(disabled_config, mock_embedding_manager, mock_redis_store)

        # get() should return None and increment misses
        result = await cache.get("test query")
        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["queries"] == 1

        # set() should do nothing when disabled
        await cache.set("test query", "test response")
        # No assertion needed - just ensure it doesn't raise an exception

    @pytest.mark.asyncio
    async def test_cache_set_basic(
        self, semantic_cache, mock_embedding_manager, mock_redis_store
    ):
        """Test basic cache set operation."""
        mock_redis_store.redis.hset = MagicMock()
        mock_redis_store.redis.expire = MagicMock()

        # Mock _get_cache_size to return small value (no eviction needed)
        semantic_cache._get_cache_size = AsyncMock(return_value=5)

        await semantic_cache.set("test query", "test response", {"key": "value"})

        # Verify embedding was requested
        mock_embedding_manager.get_embedding.assert_called_once_with("test query")

        # Verify Redis operations
        mock_redis_store.redis.hset.assert_called_once()
        mock_redis_store.redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, semantic_cache):
        """Test cache get operation with no similar queries."""
        semantic_cache._search_similar = AsyncMock(return_value=[])

        result = await semantic_cache.get("test query")

        assert result is None
        assert semantic_cache.stats["queries"] == 1
        assert semantic_cache.stats["misses"] == 1
        assert semantic_cache.stats["hits"] == 0

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, semantic_cache):
        """Test cache get operation with similar query found."""
        # Mock search result with high similarity
        mock_result = (
            "cache_id_123",
            0.95,  # High similarity
            {"response": "cached response", "hit_count": 2},
        )
        semantic_cache._search_similar = AsyncMock(return_value=[mock_result])
        semantic_cache.redis.redis.hincrby = MagicMock()

        result = await semantic_cache.get("similar query")

        assert result == "cached response"
        assert semantic_cache.stats["queries"] == 1
        assert semantic_cache.stats["hits"] == 1
        assert semantic_cache.stats["misses"] == 0

        # Verify hit count was incremented
        semantic_cache.redis.redis.hincrby.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_below_threshold(self, semantic_cache):
        """Test cache get when similarity is below threshold."""
        # Mock search result with low similarity
        mock_result = (
            "cache_id_123",
            0.85,  # Below threshold (0.9)
            {"response": "cached response"},
        )
        semantic_cache._search_similar = AsyncMock(return_value=[mock_result])

        result = await semantic_cache.get("dissimilar query")

        assert result is None
        assert semantic_cache.stats["misses"] == 1

    def test_get_stats(self, semantic_cache):
        """Test statistics retrieval."""
        semantic_cache.stats["queries"] = 100
        semantic_cache.stats["hits"] = 31
        semantic_cache.adaptive_threshold = 0.87

        # Update stats to calculate hit_rate
        semantic_cache._update_stats()
        stats = semantic_cache.get_stats()

        assert stats["queries"] == 100
        assert stats["hits"] == 31
        assert stats["hit_rate"] == 0.31
        assert stats["adaptive_threshold"] == 0.87
        assert stats["cache_enabled"] is True

    def test_update_stats_hit_rate(self, semantic_cache):
        """Test hit rate calculation in _update_stats."""
        semantic_cache.stats["queries"] = 100
        semantic_cache.stats["hits"] = 25

        semantic_cache._update_stats()

        assert semantic_cache.stats["hit_rate"] == 0.25

    def test_update_stats_avg_similarity(self, semantic_cache):
        """Test average similarity calculation."""
        # Add similarity scores
        semantic_cache.similarity_scores = [0.8, 0.85, 0.9, 0.95]

        semantic_cache._update_stats()

        expected_avg = np.mean([0.8, 0.85, 0.9, 0.95])
        assert semantic_cache.stats["avg_similarity"] == expected_avg

    def test_adaptive_threshold_adjustment_low_hit_rate(self, semantic_cache):
        """Test adaptive threshold lowering when hit rate is too low."""
        semantic_cache.config.adaptive_threshold = True
        semantic_cache.config.target_hit_rate = 0.31
        semantic_cache.adaptive_threshold = 0.90
        semantic_cache.stats["queries"] = 200
        semantic_cache.stats["hits"] = 40  # 20% hit rate, below target
        semantic_cache.similarity_scores = [0.8] * 100  # Enough for adjustment

        semantic_cache._update_stats()

        # Threshold should be lowered (multiplied by 0.98)
        assert semantic_cache.adaptive_threshold < 0.90

    def test_adaptive_threshold_adjustment_high_hit_rate(self, semantic_cache):
        """Test adaptive threshold raising when hit rate is too high."""
        semantic_cache.config.adaptive_threshold = True
        semantic_cache.config.target_hit_rate = 0.31
        semantic_cache.adaptive_threshold = 0.85
        semantic_cache.stats["queries"] = 200
        semantic_cache.stats["hits"] = 80  # 40% hit rate, above target
        semantic_cache.similarity_scores = [0.9] * 100  # Enough for adjustment

        semantic_cache._update_stats()

        # Threshold should be raised (multiplied by 1.02)
        assert semantic_cache.adaptive_threshold > 0.85

    def test_adaptive_threshold_bounds(self, semantic_cache):
        """Test adaptive threshold stays within bounds."""
        semantic_cache.config.adaptive_threshold = True
        semantic_cache.adaptive_threshold = 0.83  # Below minimum

        # Set up conditions for clamping: need >= 100 similarity scores
        # and significant hit rate difference
        semantic_cache.similarity_scores = [0.8] * 100
        semantic_cache.stats["queries"] = 200
        semantic_cache.stats["hits"] = 40  # 20% hit rate, far from target 31%

        semantic_cache._update_stats()

        # Should be clamped to minimum of 0.85
        assert semantic_cache.adaptive_threshold >= 0.85

        # Test upper bound
        semantic_cache.adaptive_threshold = 1.01  # Above maximum
        semantic_cache.stats["hits"] = 80  # 40% hit rate, above target

        semantic_cache._update_stats()

        # Should be clamped to maximum of 0.99
        assert semantic_cache.adaptive_threshold <= 0.99

    @pytest.mark.asyncio
    async def test_clear_cache(self, semantic_cache):
        """Test cache clearing functionality."""
        # Set up some initial state
        semantic_cache.stats["queries"] = 50
        semantic_cache.stats["hits"] = 15
        semantic_cache.similarity_scores = [0.8, 0.9, 0.95]
        semantic_cache.adaptive_threshold = 0.87

        # Mock Redis scan operations
        semantic_cache.redis.redis.scan = MagicMock(
            side_effect=[
                (0, [b"cache:123", b"cache:456"]),  # First scan iteration
            ]
        )
        semantic_cache.redis.redis.delete = MagicMock()

        await semantic_cache.clear()

        # Verify stats reset
        assert semantic_cache.stats["queries"] == 0
        assert semantic_cache.stats["hits"] == 0
        assert semantic_cache.stats["misses"] == 0
        assert semantic_cache.similarity_scores == []
        assert (
            semantic_cache.adaptive_threshold
            == semantic_cache.config.similarity_threshold
        )

        # Verify Redis delete was called
        semantic_cache.redis.redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_size(self, semantic_cache):
        """Test cache size calculation."""
        # Mock Redis scan to return cache keys
        semantic_cache.redis.redis.scan = MagicMock(
            side_effect=[
                (123, [b"cache:1", b"cache:2", b"cache:3"]),  # First batch
                (0, [b"cache:4", b"cache:5"]),  # Second batch, cursor = 0 ends
            ]
        )

        size = await semantic_cache._get_cache_size()

        assert size == 5  # Total keys across both batches

    @pytest.mark.asyncio
    async def test_evict_oldest(self, semantic_cache):
        """Test eviction of oldest cache entries."""
        # Mock Redis scan to return entries with timestamps
        semantic_cache.redis.redis.scan = MagicMock(
            side_effect=[
                (0, [b"cache:1", b"cache:2", b"cache:3"]),
            ]
        )

        # Mock Redis hget to return different timestamps
        semantic_cache.redis.redis.hget = MagicMock(
            side_effect=["1000.0", "2000.0", "3000.0"]  # Different timestamps
        )
        semantic_cache.redis.redis.delete = MagicMock()

        await semantic_cache._evict_oldest()

        # Should delete oldest entry (cache:1 with timestamp 1000.0)
        semantic_cache.redis.redis.delete.assert_called_once_with(b"cache:1")

    @pytest.mark.asyncio
    async def test_cache_size_eviction_trigger(
        self, semantic_cache, mock_embedding_manager, mock_redis_store
    ):
        """Test that cache eviction is triggered when size limit is reached."""
        # Mock cache size to be at limit
        semantic_cache._get_cache_size = AsyncMock(
            return_value=100
        )  # At max_cache_size
        semantic_cache._evict_oldest = AsyncMock()

        mock_redis_store.redis.hset = MagicMock()
        mock_redis_store.redis.expire = MagicMock()

        await semantic_cache.set("test query", "test response")

        # Verify eviction was called
        semantic_cache._evict_oldest.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_error_handling(self, semantic_cache):
        """Test error handling in _search_similar method."""
        # Mock Redis search to raise an exception
        semantic_cache.redis.redis.ft = MagicMock()
        semantic_cache.redis.redis.ft.return_value.search = MagicMock(
            side_effect=Exception("Redis error")
        )

        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Should return empty list on error, not raise exception
        result = await semantic_cache._search_similar(query_embedding)

        assert result == []
