"""Test configuration module."""

from pathlib import Path

import pytest

from eol.rag_context.config import (
    CacheConfig,
    ChunkingConfig,
    ContextConfig,
    DocumentConfig,
    EmbeddingConfig,
    IndexConfig,
    RAGConfig,
    RedisConfig,
)


class TestRedisConfig:
    """Test Redis configuration."""

    def test_default_config(self):
        """Test default Redis configuration."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.decode_responses is False

    def test_url_generation(self):
        """Test Redis URL generation."""
        config = RedisConfig(host="redis.example.com", port=6380, db=1)
        assert config.url == "redis://redis.example.com:6380/1"

        # With password
        config = RedisConfig(password="secret")
        assert config.url == "redis://:secret@localhost:6379/0"


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_default_config(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        assert config.provider == "sentence-transformers"
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.normalize is True

    def test_dimension_validation(self):
        """Test embedding dimension validation."""
        # Should auto-correct dimension for known models
        config = EmbeddingConfig(model_name="all-MiniLM-L6-v2", dimension=100)
        # Validator should correct it to 384
        assert config.dimension == 384


class TestIndexConfig:
    """Test index configuration."""

    def test_default_config(self):
        """Test default index configuration."""
        config = IndexConfig()
        assert config.index_name == "eol_context"
        assert config.prefix == "doc:"
        assert config.algorithm == "HNSW"
        assert config.distance_metric == "COSINE"
        assert config.hierarchy_levels == 3


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()
        assert config.min_chunk_size == 100
        assert config.max_chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.use_semantic_chunking is True


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.similarity_threshold == 0.97
        assert config.target_hit_rate == 0.31


class TestContextConfig:
    """Test context configuration."""

    def test_default_config(self):
        """Test default context configuration."""
        config = ContextConfig()
        assert config.max_context_tokens == 32000
        assert config.default_top_k == 10
        assert config.min_relevance_score == 0.7
        assert config.use_hierarchical_retrieval is True


class TestDocumentConfig:
    """Test document configuration."""

    def test_default_config(self):
        """Test default document configuration."""
        config = DocumentConfig()
        assert len(config.file_patterns) > 0
        assert "*.md" in config.file_patterns
        assert "*.py" in config.file_patterns
        assert config.extract_metadata is True
        assert config.max_file_size_mb == 100


class TestRAGConfig:
    """Test main RAG configuration."""

    def test_default_config(self, tmp_path):
        """Test default RAG configuration."""
        config = RAGConfig(data_dir=tmp_path / "data", index_dir=tmp_path / "index")

        assert config.server_name == "eol-rag-context"
        assert config.server_version == "0.1.0"
        assert config.debug is False

        # Sub-configs should be initialized
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.index, IndexConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.context, ContextConfig)
        assert isinstance(config.document, DocumentConfig)

    def test_directory_creation(self, tmp_path):
        """Test that directories are created."""
        data_dir = tmp_path / "test_data"
        index_dir = tmp_path / "test_index"

        config = RAGConfig(data_dir=data_dir, index_dir=index_dir)

        assert data_dir.exists()
        assert index_dir.exists()
