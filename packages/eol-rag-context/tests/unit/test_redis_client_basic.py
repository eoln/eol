"""
Basic tests for redis_client to improve coverage.
"""

# Removed unused imports

from eol.rag_context import config
from eol.rag_context.redis_client import RedisVectorStore


class TestRedisVectorStore:
    """Test RedisVectorStore basic functionality."""

    def test_redis_vector_store_init(self):
        """Test RedisVectorStore initialization."""
        redis_config = config.RedisConfig()
        index_config = config.IndexConfig()

        store = RedisVectorStore(redis_config, index_config)

        assert store.redis_config == redis_config
        assert store.index_config == index_config
        assert store.redis is None  # Not connected yet
        assert store.async_redis is None  # Not connected yet

    def test_redis_store_attributes(self):
        """Test RedisVectorStore attributes."""
        redis_config = config.RedisConfig()
        index_config = config.IndexConfig()

        store = RedisVectorStore(redis_config, index_config)

        # Check attributes exist
        assert hasattr(store, "redis_config")
        assert hasattr(store, "index_config")
        assert hasattr(store, "redis")
        assert hasattr(store, "async_redis")

        # Check initial state
        assert store.redis_config == redis_config
        assert store.index_config == index_config

    def test_redis_config_properties(self):
        """Test RedisConfig properties."""
        redis_config = config.RedisConfig()

        # Check default values
        assert redis_config.host == "localhost"
        assert redis_config.port == 6379
        assert redis_config.db == 0
        assert redis_config.password is None

        # Check URL generation
        url = redis_config.url
        assert "redis://" in url
        assert "localhost" in url
        assert "6379" in url

    def test_index_config_properties(self):
        """Test IndexConfig properties."""
        index_config = config.IndexConfig()

        # Check Vector Set properties (updated for Redis 8.2+)
        assert index_config.vectorset_name == "eol_context"
        assert index_config.prefix == "doc:"
        assert index_config.algorithm == "SVS-VAMANA"
        assert index_config.distance_metric == "COSINE"  # Legacy compatibility

    def test_redis_vector_store_methods(self):
        """Test that RedisVectorStore has expected methods."""
        redis_config = config.RedisConfig()
        index_config = config.IndexConfig()

        store = RedisVectorStore(redis_config, index_config)

        # Check methods exist
        assert hasattr(store, "connect")
        assert hasattr(store, "store_document")
        assert hasattr(store, "create_hierarchical_indexes")
        assert hasattr(store, "vector_search")
        assert hasattr(store, "hierarchical_search")
        assert callable(store.connect)


if __name__ == "__main__":
    print("✅ Redis client basic tests!")
