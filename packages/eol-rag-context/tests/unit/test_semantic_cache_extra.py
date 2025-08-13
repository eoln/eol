"""
Extra tests for semantic_cache to achieve 80% coverage.
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# Mock dependencies
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()

from eol.rag_context import config
from eol.rag_context.semantic_cache import SemanticCache


class TestSemanticCacheExtra:
    """Extra tests to achieve 80% coverage."""
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    @patch('eol.rag_context.semantic_cache.EmbeddingManager')
    async def test_cache_set_operation(self, mock_embedding_class, mock_redis_class):
        """Test cache set operation."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        mock_embedding = AsyncMock()
        mock_embedding.embed = AsyncMock(return_value=np.random.rand(384).astype(np.float32))
        mock_embedding_class.return_value = mock_embedding
        
        # Create cache
        redis_config = config.RedisConfig()
        cache_config = config.CacheConfig(enabled=True, ttl_seconds=3600)
        cache = SemanticCache(redis_config, cache_config)
        
        # Connect and set embedding manager
        await cache.connect_async()
        cache.embedding_manager = mock_embedding
        
        # Test set operation
        await cache.set("test query", "test response")
        
        # Verify Redis operations were called
        mock_embedding.embed.assert_called_once_with("test query")
        mock_redis.hset.assert_called()
        mock_redis.expire.assert_called()
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    @patch('eol.rag_context.semantic_cache.EmbeddingManager')
    async def test_cache_get_operation_miss(self, mock_embedding_class, mock_redis_class):
        """Test cache get operation with cache miss."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[])  # No keys found
        mock_redis_class.return_value = mock_redis
        
        mock_embedding = AsyncMock()
        mock_embedding.embed = AsyncMock(return_value=np.random.rand(384).astype(np.float32))
        mock_embedding_class.return_value = mock_embedding
        
        # Create cache
        redis_config = config.RedisConfig()
        cache_config = config.CacheConfig(enabled=True, similarity_threshold=0.9)
        cache = SemanticCache(redis_config, cache_config)
        
        # Connect and set embedding manager
        await cache.connect_async()
        cache.embedding_manager = mock_embedding
        
        # Test get operation with cache miss
        result = await cache.get("test query")
        
        assert result is None
        mock_embedding.embed.assert_called_once_with("test query")
        mock_redis.keys.assert_called()
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    @patch('eol.rag_context.semantic_cache.EmbeddingManager')
    async def test_cache_get_operation_hit(self, mock_embedding_class, mock_redis_class):
        """Test cache get operation with cache hit."""
        # Setup mocks
        mock_redis = AsyncMock()
        
        # Mock finding cache keys
        mock_redis.keys = AsyncMock(return_value=[b'cache:entry1', b'cache:entry2'])
        
        # Mock getting cache entries
        cache_entry = {
            b'query_embedding': json.dumps([0.1, 0.2, 0.3]).encode(),
            b'response': b'cached response',
            b'metadata': json.dumps({'hits': 1}).encode()
        }
        mock_redis.hgetall = AsyncMock(return_value=cache_entry)
        mock_redis.hset = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        mock_embedding = AsyncMock()
        # Return embedding similar to cached one
        mock_embedding.embed = AsyncMock(return_value=np.array([0.1, 0.2, 0.3]).astype(np.float32))
        mock_embedding_class.return_value = mock_embedding
        
        # Create cache
        redis_config = config.RedisConfig()
        cache_config = config.CacheConfig(enabled=True, similarity_threshold=0.5)
        cache = SemanticCache(redis_config, cache_config)
        
        # Connect and set embedding manager
        await cache.connect_async()
        cache.embedding_manager = mock_embedding
        
        # Test get operation with cache hit
        result = await cache.get("test query")
        
        assert result == "cached response"
        mock_embedding.embed.assert_called_once_with("test query")
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    async def test_cache_clear_operation(self, mock_redis_class):
        """Test cache clear operation."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[b'cache:1', b'cache:2'])
        mock_redis.delete = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        cache = SemanticCache(config.RedisConfig(), config.CacheConfig())
        await cache.connect_async()
        
        # Test clear operation
        cleared = await cache.clear()
        
        assert cleared == 2
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_called_once()
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    async def test_cache_get_stats(self, mock_redis_class):
        """Test getting cache statistics."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(return_value=[b'cache:1', b'cache:2', b'cache:3'])
        mock_redis.info = AsyncMock(return_value={'used_memory_human': '10MB'})
        mock_redis_class.return_value = mock_redis
        
        cache = SemanticCache(config.RedisConfig(), config.CacheConfig())
        await cache.connect_async()
        cache._stats = {'hits': 5, 'misses': 3, 'sets': 8}
        
        # Test get_stats
        stats = await cache.get_stats()
        
        assert stats['total_entries'] == 3
        assert stats['hits'] == 5
        assert stats['misses'] == 3
        assert stats['hit_rate'] == 0.625
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    async def test_cache_eviction(self, mock_redis_class):
        """Test cache eviction when max size reached."""
        mock_redis = AsyncMock()
        # Return many cache entries to trigger eviction
        mock_redis.keys = AsyncMock(return_value=[f'cache:{i}'.encode() for i in range(1000)])
        
        # Mock cache entries with access times
        old_entry = {
            b'query_embedding': json.dumps([0.1, 0.2]).encode(),
            b'response': b'old response',
            b'metadata': json.dumps({'last_accessed': 1000}).encode()
        }
        mock_redis.hgetall = AsyncMock(return_value=old_entry)
        mock_redis.delete = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        cache_config = config.CacheConfig(max_cache_size=500)  # Small max size
        cache = SemanticCache(config.RedisConfig(), cache_config)
        await cache.connect_async()
        
        # Trigger eviction check
        await cache._check_cache_size()
        
        # Should have deleted old entries
        mock_redis.delete.assert_called()
    
    async def test_cache_disabled(self):
        """Test cache operations when disabled."""
        cache_config = config.CacheConfig(enabled=False)
        cache = SemanticCache(config.RedisConfig(), cache_config)
        
        # All operations should return None/0 when disabled
        result = await cache.get("query")
        assert result is None
        
        await cache.set("query", "response")  # Should not error
        
        cleared = await cache.clear()
        assert cleared == 0
        
        stats = await cache.get_stats()
        assert stats['total_entries'] == 0
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    async def test_cache_connection_error_handling(self, mock_redis_class):
        """Test error handling during cache operations."""
        mock_redis = AsyncMock()
        mock_redis.keys = AsyncMock(side_effect=Exception("Connection error"))
        mock_redis_class.return_value = mock_redis
        
        cache = SemanticCache(config.RedisConfig(), config.CacheConfig())
        await cache.connect_async()
        
        # Operations should handle errors gracefully
        result = await cache.get("query")
        assert result is None  # Returns None on error
        
        cleared = await cache.clear()
        assert cleared == 0  # Returns 0 on error
    
    def test_cache_initialization_variations(self):
        """Test various cache initialization scenarios."""
        # Test with different configurations
        configs = [
            config.CacheConfig(enabled=True, ttl_seconds=7200),
            config.CacheConfig(enabled=False),
            config.CacheConfig(similarity_threshold=0.95),
            config.CacheConfig(max_cache_size=10000)
        ]
        
        for cache_config in configs:
            cache = SemanticCache(config.RedisConfig(), cache_config)
            assert cache.cache_config == cache_config
            assert cache.redis_config is not None
    
    @patch('eol.rag_context.semantic_cache.AsyncRedis')
    async def test_cache_close_operation(self, mock_redis_class):
        """Test cache close/cleanup."""
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        cache = SemanticCache(config.RedisConfig(), config.CacheConfig())
        await cache.connect_async()
        
        # Test close
        await cache.close()
        
        mock_redis.close.assert_called_once()
        assert cache.redis is None


if __name__ == "__main__":
    print("✅ Semantic cache extra tests ready!")