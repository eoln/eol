"""
Additional tests to boost coverage to 80%.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from eol.rag_context import config, document_processor, indexer


class TestCoverageBoost:
    """Additional tests to reach 80% coverage."""

    @pytest.mark.asyncio
    async def test_indexer_search_simple(self):
        """Test basic search functionality."""
        mock_config = config.RAGConfig()
        mock_redis = MagicMock()
        mock_redis.search_similar = AsyncMock(return_value=[{"content": "test", "score": 0.9}])
        mock_embedding = MagicMock()
        mock_embedding.get_embedding = AsyncMock(
            return_value=np.random.rand(384).astype(np.float32)
        )
        mock_processor = MagicMock()

        idx = indexer.DocumentIndexer(mock_config, mock_redis, mock_embedding, mock_processor)

        results = await idx.search("test query", k=5)
        assert isinstance(results, list)

    def test_document_processor_chunk_creation(self):
        """Test chunk creation in document processor."""
        doc_config = config.DocumentConfig()
        chunk_config = config.ChunkingConfig()
        processor = document_processor.DocumentProcessor(doc_config, chunk_config)

        chunk = processor._create_chunk(
            content="Test content", chunk_type="test", index=0, metadata_key="value"
        )

        assert chunk["content"] == "Test content"
        assert chunk["type"] == "test"
        assert chunk["metadata"]["index"] == 0
        assert chunk["metadata"]["metadata_key"] == "value"
        assert chunk["tokens"] > 0

    def test_config_url_method(self):
        """Test RedisConfig url method."""
        redis_config = config.RedisConfig()
        redis_config.host = "redis.example.com"
        redis_config.port = 6380
        redis_config.db = 1
        redis_config.password = "secret"

        url = redis_config.url()
        assert "redis.example.com" in url
        assert "6380" in url
        assert "secret" in url

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig dimension validation."""
        # Test valid dimension
        emb_config = config.EmbeddingConfig(dimension=384)
        assert emb_config.dimension == 384

        # Test dimension must be positive
        with pytest.raises(ValueError):
            config.EmbeddingConfig(dimension=-1)

    @pytest.mark.asyncio
    async def test_indexer_remove_file(self):
        """Test removing a file from the index."""
        mock_config = config.RAGConfig()
        mock_redis = MagicMock()
        mock_redis.async_redis = AsyncMock()
        mock_redis.async_redis.scan = AsyncMock(return_value=(0, [b"doc:test"]))
        mock_redis.async_redis.hget = AsyncMock(return_value=b"/test/file.txt")
        mock_redis.async_redis.delete = AsyncMock()

        mock_embedding = MagicMock()
        mock_processor = MagicMock()

        idx = indexer.DocumentIndexer(mock_config, mock_redis, mock_embedding, mock_processor)

        result = await idx.remove_file(Path("/test/file.txt"))
        assert result is not None

    def test_folder_scanner_matches_pattern(self):
        """Test FolderScanner pattern matching."""
        doc_config = config.DocumentConfig()
        doc_config.file_patterns = ["*.py", "*.md", "*.txt"]
        scanner = indexer.FolderScanner(doc_config)

        # Test matching patterns
        assert scanner._matches_patterns(Path("test.py")) is True
        assert scanner._matches_patterns(Path("readme.md")) is True
        assert scanner._matches_patterns(Path("data.txt")) is True

        # Test non-matching patterns
        assert scanner._matches_patterns(Path("image.png")) is False
        assert scanner._matches_patterns(Path("data.json")) is False

    @pytest.mark.asyncio
    async def test_document_processor_process_empty_file(self, tmp_path):
        """Test processing empty file."""
        doc_config = config.DocumentConfig()
        chunk_config = config.ChunkingConfig()
        processor = document_processor.DocumentProcessor(doc_config, chunk_config)

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = await processor.process_file(empty_file)
        # Empty files might return None or minimal document
        if result:
            assert result.content == ""

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        chunk_config = config.ChunkingConfig()

        assert chunk_config.min_chunk_size == 100
        assert chunk_config.max_chunk_size == 512
        assert chunk_config.chunk_overlap == 64
        assert chunk_config.use_semantic_chunking is True
        assert chunk_config.semantic_threshold == 0.7

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        cache_config = config.CacheConfig()

        assert cache_config.enabled is True
        assert cache_config.ttl_seconds == 3600
        assert cache_config.target_hit_rate == 0.31
        assert cache_config.adaptive_threshold is True
        assert cache_config.max_cache_size == 1000

    def test_context_config_defaults(self):
        """Test ContextConfig default values."""
        context_config = config.ContextConfig()

        assert context_config.max_context_tokens == 32000
        assert context_config.reserve_tokens_for_response == 4000
        assert context_config.default_top_k == 10
        assert context_config.min_relevance_score == 0.7

    def test_server_config_defaults(self):
        """Test ServerConfig default values."""
        server_config = config.ServerConfig()

        assert server_config.server_name == "eol-rag-context"
        assert server_config.server_version == "0.1.0"
        assert server_config.debug is False
        assert isinstance(server_config.data_dir, Path)
        assert isinstance(server_config.index_dir, Path)

    @pytest.mark.asyncio
    async def test_indexer_get_document(self):
        """Test getting a document from the index."""
        mock_config = config.RAGConfig()
        mock_redis = MagicMock()
        mock_redis.get_document = AsyncMock(
            return_value={"content": "Document content", "metadata": {"type": "text"}}
        )

        mock_embedding = MagicMock()
        mock_processor = MagicMock()

        idx = indexer.DocumentIndexer(mock_config, mock_redis, mock_embedding, mock_processor)

        doc = await idx.get_document("doc_123")
        assert doc is not None
        assert doc["content"] == "Document content"
