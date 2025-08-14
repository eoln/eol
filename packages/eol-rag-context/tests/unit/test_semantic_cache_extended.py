"""
Extended tests for semantic_cache.py to improve coverage.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from eol.rag_context import config, semantic_cache


class TestSemanticCacheExtended:
    """Extended tests for SemanticCache to improve coverage."""

    @pytest.fixture
    def cache_config(self):
        """Create a mock cache config."""
        return config.CacheConfig(
            enabled=True,
            ttl_seconds=3600,
            similarity_threshold=0.95,
            max_cache_size=100,
            target_hit_rate=0.31,
            adaptive_threshold=True,
        )

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager."""
        manager = MagicMock()
        manager.config = MagicMock()
        manager.config.dimension = 384
        manager.get_embedding = AsyncMock(return_value=np.random.rand(384).astype(np.float32))
        return manager

    @pytest.fixture
    def mock_redis_store(self):
        """Create a mock Redis store."""
        store = MagicMock()
        store.redis = MagicMock()
        store.async_redis = AsyncMock()
        return store

    @pytest.fixture
    def cache(self, cache_config, mock_embedding_manager, mock_redis_store):
        """Create a SemanticCache instance."""
        return semantic_cache.SemanticCache(cache_config, mock_embedding_manager, mock_redis_store)

    @pytest.mark.asyncio
    async def test_initialize_create_index(self, cache):
        """Test initialize when index doesn't exist."""
        # Mock index doesn't exist
        cache.redis.async_redis.ft = MagicMock()
        cache.redis.async_redis.ft.return_value.info = AsyncMock(
            side_effect=Exception("Index not found")
        )
        cache.redis.async_redis.ft.return_value.create_index = AsyncMock()

        await cache.initialize()

        # Should have attempted to create index
        cache.redis.async_redis.ft.return_value.create_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_index_exists(self, cache):
        """Test initialize when index already exists."""
        # Mock index exists
        cache.redis.async_redis.ft = MagicMock()
        cache.redis.async_redis.ft.return_value.info = AsyncMock(
            return_value={"index_name": "cache_index"}
        )

        await cache.initialize()

        # Should have checked for existing index
        cache.redis.async_redis.ft.return_value.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_disabled(self, cache):
        """Test get when cache is disabled."""
        cache.config.enabled = False

        result = await cache.get("test query")

        assert result is None
        assert cache.stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache):
        """Test successful cache hit."""
        cache.config.enabled = True

        # Mock similar query found
        mock_similar = [
            ("cache_123", 0.98, {"response": "cached response", "query": "similar query"})
        ]
        cache._search_similar = AsyncMock(return_value=mock_similar)

        # Mock Redis operations
        cache.redis.redis.hincrby = MagicMock()

        result = await cache.get("test query")

        assert result == "cached response"
        assert cache.stats["hits"] == 1
        cache.redis.redis.hincrby.assert_called_once_with("cache:cache_123", "hit_count", 1)

    @pytest.mark.asyncio
    async def test_get_cache_miss_low_similarity(self, cache):
        """Test cache miss due to low similarity."""
        cache.config.enabled = True
        cache.adaptive_threshold = 0.95

        # Mock similar query with low similarity
        mock_similar = [
            ("cache_123", 0.80, {"response": "cached response", "query": "different query"})
        ]
        cache._search_similar = AsyncMock(return_value=mock_similar)

        result = await cache.get("test query")

        assert result is None
        assert cache.stats["misses"] == 1
        # Similarity score should be tracked
        assert len(cache.similarity_scores) == 1
        assert cache.similarity_scores[0] == 0.80

    @pytest.mark.asyncio
    async def test_set_cache_disabled(self, cache):
        """Test set when cache is disabled."""
        cache.config.enabled = False

        await cache.set("query", "response")

        # Should not store anything
        cache.redis.redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_with_eviction(self, cache):
        """Test set with cache eviction when at capacity."""
        cache.config.enabled = True
        cache.config.max_cache_size = 10

        # Mock cache at capacity
        cache._get_cache_size = AsyncMock(return_value=10)
        cache._evict_oldest = AsyncMock()

        # Mock Redis operations
        cache.redis.redis.hset = MagicMock()
        cache.redis.redis.expire = MagicMock()

        await cache.set("query", "response", {"key": "value"})

        # Should have checked size and evicted
        cache._get_cache_size.assert_called_once()
        cache._evict_oldest.assert_called_once()

        # Should have stored the new entry
        cache.redis.redis.hset.assert_called_once()
        cache.redis.redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_error_handling(self, cache):
        """Test _search_similar with Redis error."""
        cache.redis.redis.ft = MagicMock()
        cache.redis.redis.ft.return_value.search = MagicMock(side_effect=Exception("Redis error"))

        query_embedding = np.random.rand(384).astype(np.float32)
        result = await cache._search_similar(query_embedding)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_cache_size(self, cache):
        """Test _get_cache_size method."""
        # Mock scan results
        cache.redis.redis.scan = MagicMock()
        cache.redis.redis.scan.side_effect = [
            (100, ["cache:1", "cache:2", "cache:3"]),
            (200, ["cache:4", "cache:5"]),
            (0, ["cache:6"]),
        ]

        size = await cache._get_cache_size()

        assert size == 6

    @pytest.mark.asyncio
    async def test_evict_oldest(self, cache):
        """Test _evict_oldest method."""
        # Mock scan and hget
        cache.redis.redis.scan = MagicMock(return_value=(0, [b"cache:1", b"cache:2", b"cache:3"]))
        cache.redis.redis.hget = MagicMock()
        cache.redis.redis.hget.side_effect = [
            b"100.0",  # oldest
            b"300.0",  # newest
            b"200.0",  # middle
        ]
        cache.redis.redis.delete = MagicMock()

        await cache._evict_oldest()

        # Should delete the oldest entry (10% of 3 = 1)
        cache.redis.redis.delete.assert_called_once()

    def test_update_stats_basic(self, cache):
        """Test _update_stats basic functionality."""
        cache.stats["queries"] = 100
        cache.stats["hits"] = 31
        cache.similarity_scores = [0.9] * 100

        cache._update_stats()

        assert cache.stats["hit_rate"] == 0.31
        assert abs(cache.stats["avg_similarity"] - 0.9) < 0.001

    def test_update_stats_adaptive_threshold_increase(self, cache):
        """Test _update_stats with adaptive threshold increase."""
        cache.config.adaptive_threshold = True
        cache.stats["queries"] = 100
        cache.stats["hits"] = 20  # Below target
        cache.similarity_scores = [0.9] * 100
        cache.adaptive_threshold = 0.95

        with patch("eol.rag_context.semantic_cache.logger") as mock_logger:
            cache._update_stats()

        # Should have lowered threshold
        assert cache.adaptive_threshold < 0.95
        mock_logger.info.assert_called_once()

    def test_update_stats_adaptive_threshold_decrease(self, cache):
        """Test _update_stats with adaptive threshold decrease."""
        cache.config.adaptive_threshold = True
        cache.stats["queries"] = 100
        cache.stats["hits"] = 50  # Above target
        cache.similarity_scores = [0.9] * 100
        cache.adaptive_threshold = 0.90

        with patch("eol.rag_context.semantic_cache.logger") as mock_logger:
            cache._update_stats()

        # Should have raised threshold
        assert cache.adaptive_threshold > 0.90
        mock_logger.info.assert_called_once()

    def test_get_stats(self, cache):
        """Test get_stats method."""
        cache.stats = {
            "queries": 100,
            "hits": 31,
            "misses": 69,
            "hit_rate": 0.31,
            "avg_similarity": 0.92,
            "threshold_adjustments": 5,
        }
        cache.adaptive_threshold = 0.94

        stats = cache.get_stats()

        assert stats["queries"] == 100
        assert stats["hits"] == 31
        assert stats["hit_rate"] == 0.31
        assert stats["adaptive_threshold"] == 0.94
        assert stats["cache_enabled"] is True

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clear method."""
        # Mock scan and delete
        cache.redis.redis.scan = MagicMock()
        cache.redis.redis.scan.side_effect = [(100, ["cache:1", "cache:2"]), (0, ["cache:3"])]
        cache.redis.redis.delete = MagicMock()

        await cache.clear()

        # Should have deleted all cache entries
        assert cache.redis.redis.delete.call_count == 2

        # Stats should be reset
        assert cache.stats["queries"] == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["hit_rate"] == 0.0
        assert len(cache.similarity_scores) == 0

    @pytest.mark.asyncio
    async def test_optimize_with_data(self, cache):
        """Test optimize method with sufficient data."""
        cache.stats["hit_rate"] = 0.25
        cache.similarity_scores = np.random.uniform(0.7, 1.0, 200).tolist()
        cache.adaptive_threshold = 0.95
        cache._get_cache_size = AsyncMock(return_value=50)

        report = await cache.optimize()

        assert "current_hit_rate" in report
        assert "target_hit_rate" in report
        assert "similarity_percentiles" in report
        assert "recommended_threshold" in report
        assert "recommendations" in report
        assert report["cache_size"] == 50

    @pytest.mark.asyncio
    async def test_optimize_near_capacity(self, cache):
        """Test optimize when cache is near capacity."""
        cache.stats["hit_rate"] = 0.31
        cache.similarity_scores = []  # No data
        cache.config.max_cache_size = 100
        cache._get_cache_size = AsyncMock(return_value=95)

        report = await cache.optimize()

        # Should recommend increasing cache size
        assert any("near capacity" in rec for rec in report["recommendations"])

    @pytest.mark.asyncio
    async def test_optimize_low_hit_rate(self, cache):
        """Test optimize with very low hit rate."""
        cache.stats["queries"] = 1500
        cache.stats["hit_rate"] = 0.05
        cache.similarity_scores = []
        cache._get_cache_size = AsyncMock(return_value=50)

        report = await cache.optimize()

        # Should recommend adjustments for low hit rate
        assert any("low hit rate" in rec for rec in report["recommendations"])

    @pytest.mark.asyncio
    async def test_optimize_high_hit_rate(self, cache):
        """Test optimize with very high hit rate."""
        cache.stats["queries"] = 1500
        cache.stats["hit_rate"] = 0.65
        cache.similarity_scores = []
        cache._get_cache_size = AsyncMock(return_value=50)

        report = await cache.optimize()

        # Should recommend adjustments for high hit rate
        assert any("high hit rate" in rec for rec in report["recommendations"])

    def test_cached_query_dataclass(self):
        """Test CachedQuery dataclass creation."""
        embedding = np.random.rand(384).astype(np.float32)
        cached = semantic_cache.CachedQuery(
            query="test query",
            response="test response",
            embedding=embedding,
            timestamp=1234567890.0,
            hit_count=5,
            metadata={"source": "llm"},
        )

        assert cached.query == "test query"
        assert cached.response == "test response"
        assert cached.hit_count == 5
        assert cached.metadata["source"] == "llm"
        assert np.array_equal(cached.embedding, embedding)

    def test_cached_query_defaults(self):
        """Test CachedQuery with default values."""
        embedding = np.random.rand(384).astype(np.float32)
        cached = semantic_cache.CachedQuery(query="test", response="response", embedding=embedding)

        assert cached.hit_count == 0
        assert isinstance(cached.metadata, dict)
        assert len(cached.metadata) == 0
        assert cached.timestamp > 0  # Should be set to current time

    @pytest.mark.asyncio
    async def test_search_similar_with_results(self, cache):
        """Test _search_similar with valid results."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock search results
        mock_doc = MagicMock()
        mock_doc.id = "cache:test_123"
        mock_doc.similarity = 0.05  # Redis returns distance, not similarity
        mock_doc.query = "cached query"
        mock_doc.response = "cached response"
        mock_doc.hit_count = "3"
        mock_doc.timestamp = "1234567890.0"

        mock_results = MagicMock()
        mock_results.docs = [mock_doc]

        cache.redis.redis.ft = MagicMock()
        cache.redis.redis.ft.return_value.search = MagicMock(return_value=mock_results)

        results = await cache._search_similar(query_embedding, k=5)

        assert len(results) == 1
        assert results[0][0] == "test_123"  # ID without prefix
        assert results[0][1] == 0.95  # Converted to similarity
        assert results[0][2]["response"] == "cached response"
