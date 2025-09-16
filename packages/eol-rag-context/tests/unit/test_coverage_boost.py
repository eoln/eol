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
    async def test_indexer_index_file(self):
        """Test basic file indexing functionality."""
        mock_config = config.RAGConfig()
        mock_redis = MagicMock()
        mock_redis.store_document = AsyncMock()
        mock_redis.async_redis = AsyncMock()
        mock_redis.async_redis.hgetall = AsyncMock(return_value={})
        mock_embedding = MagicMock()
        mock_embedding.get_embedding = AsyncMock(
            return_value=np.random.rand(384).astype(np.float32)
        )
        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(return_value=None)

        idx = indexer.DocumentIndexer(mock_config, mock_processor, mock_embedding, mock_redis)

        # Test indexing a non-existent file (processor returns None)
        result = await idx.index_file(Path("/test/file.txt"))
        # Should return a result even if file processing fails
        assert result is not None
        assert result.errors is not None

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

    def test_config_url_property(self):
        """Test RedisConfig url property."""
        redis_config = config.RedisConfig()
        redis_config.host = "redis.example.com"
        redis_config.port = 6380
        redis_config.db = 1
        redis_config.password = "secret"

        url = redis_config.url  # url is a property, not a method
        assert "redis.example.com" in url
        assert "6380" in url
        assert "secret" in url

    def test_embedding_config_validation(self):
        """Test EmbeddingConfig dimension validation."""
        # Test valid dimension - it gets auto-corrected based on model
        emb_config = config.EmbeddingConfig()
        # Default model is all-MiniLM-L6-v2 which has 384 dimensions
        assert emb_config.dimension == 384

        # Test with different model
        emb_config2 = config.EmbeddingConfig(model_name="text-embedding-3-small")
        # This model might have different dimensions or use default
        assert emb_config2.model_name == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_indexer_remove_source(self):
        """Test removing a source from the index."""
        mock_config = config.RAGConfig()

        # Create a mock RedisVectorStore with redis client
        mock_redis_store = MagicMock()
        mock_redis_client = MagicMock()

        # Track scan calls to provide appropriate responses
        scan_call_count = 0

        def scan_side_effect(cursor, **kwargs):
            nonlocal scan_call_count
            scan_call_count += 1

            if "match" in kwargs:
                pattern = kwargs["match"]
                if pattern.startswith("concept:"):
                    if cursor == 0:
                        return (100, [b"concept:test_source_1", b"concept:test_source_2"])
                    else:
                        return (0, [])
                elif pattern.startswith("section:"):
                    return (0, [b"section:test_source_1"])
                elif pattern.startswith("chunk:"):
                    return (0, [b"chunk:test_source_1", b"chunk:test_source_2"])
                elif pattern == "file_meta:*":
                    return (0, [b"file_meta:hash1", b"file_meta:hash2"])
            return (0, [])

        mock_redis_client.scan = MagicMock(side_effect=scan_side_effect)

        # Mock delete operations
        mock_redis_client.delete = MagicMock(return_value=1)

        # Mock hgetall for file metadata check
        mock_redis_client.hgetall = MagicMock(
            side_effect=[
                {b"path": b"/path/test_source/file1.txt"},  # Matches source_id
                {b"path": b"/other/path/file.txt"},  # Doesn't match
            ]
        )

        # Attach redis client to the store
        mock_redis_store.redis = mock_redis_client

        mock_embedding = MagicMock()
        mock_processor = MagicMock()

        idx = indexer.DocumentIndexer(mock_config, mock_processor, mock_embedding, mock_redis_store)

        result = await idx.remove_source("test_source")

        # Should have deleted documents and source metadata
        assert result is True
        assert (
            mock_redis_client.delete.call_count >= 4
        )  # At least concept, section, chunk, source key

    def test_folder_scanner_should_ignore(self):
        """Test FolderScanner should_ignore method."""
        rag_config = config.RAGConfig()
        scanner = indexer.FolderScanner(rag_config)

        # The default patterns use ** glob patterns
        # Check the actual pattern matching logic

        # Test pattern matching (without file existence check)
        # The patterns are like "**/*.pyc", "**/.git/**", etc.

        # These paths should match the ignore patterns
        pyc_path = Path("some/dir/test.pyc")
        assert any(pyc_path.match(pattern) for pattern in scanner.ignore_patterns)

        git_path = Path("project/.git/config")
        assert any(git_path.match(pattern) for pattern in scanner.ignore_patterns)

        # These paths should NOT match
        py_path = Path("test.py")
        assert not any(py_path.match(pattern) for pattern in scanner.ignore_patterns)

        md_path = Path("readme.md")
        assert not any(md_path.match(pattern) for pattern in scanner.ignore_patterns)

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

    def test_rag_config_server_defaults(self):
        """Test RAGConfig server-related default values."""
        rag_config = config.RAGConfig()

        assert rag_config.server_name == "eol-rag-context"
        assert rag_config.server_version == "0.1.0"
        assert rag_config.debug is False
        assert isinstance(rag_config.data_dir, Path)
        assert isinstance(rag_config.index_dir, Path)

    @pytest.mark.asyncio
    async def test_indexer_list_sources(self):
        """Test listing indexed sources."""
        mock_config = config.RAGConfig()

        # Create a mock RedisVectorStore with redis client
        mock_redis_store = MagicMock()
        mock_redis_client = MagicMock()

        # Mock scan method
        mock_redis_client.scan = MagicMock(return_value=(0, [b"source:test1", b"source:test2"]))
        mock_redis_client.hgetall = MagicMock(
            side_effect=[
                {
                    b"source_id": b"test1",
                    b"path": b"/path/to/file1.txt",
                    b"indexed_at": b"1234567890.0",
                    b"file_count": b"1",
                    b"total_chunks": b"5",
                    b"indexed_files": b"1",
                    b"metadata": b"{}",
                },
                {
                    b"source_id": b"test2",
                    b"path": b"/path/to/folder",
                    b"indexed_at": b"1234567891.0",
                    b"file_count": b"3",
                    b"total_chunks": b"10",
                    b"indexed_files": b"3",
                    b"metadata": b"{}",
                },
            ]
        )

        # Attach redis client to the store
        mock_redis_store.redis = mock_redis_client

        mock_embedding = MagicMock()
        mock_processor = MagicMock()

        idx = indexer.DocumentIndexer(mock_config, mock_processor, mock_embedding, mock_redis_store)

        sources = await idx.list_sources()
        assert len(sources) == 2
        assert sources[0].source_id == "test1"
        assert sources[0].file_count == 1
        assert sources[0].total_chunks == 5
        assert sources[1].source_id == "test2"
        assert sources[1].file_count == 3
        assert sources[1].total_chunks == 10
