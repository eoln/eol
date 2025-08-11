"""
Comprehensive unit tests for all modules.
"""

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import pytest

# Mock all external dependencies
mock_modules = [
    "magic",
    "pypdf",
    "docx",
    "redis",
    "redis.asyncio",
    "redis.commands",
    "redis.commands.search",
    "redis.commands.search.field",
    "redis.commands.search.indexDefinition",
    "redis.commands.search.query",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "networkx",
    "fastmcp",
    "sentence_transformers",
]
for module in mock_modules:
    sys.modules[module] = MagicMock()

# Now import our modules
from eol.rag_context.config import EmbeddingConfig, RAGConfig, RedisConfig
from eol.rag_context.document_processor import DocumentProcessor, ProcessedDocument
from eol.rag_context.embeddings import EmbeddingManager, SentenceTransformerProvider
from eol.rag_context.file_watcher import ChangeType, FileChange, FileWatcher, WatchedSource
from eol.rag_context.indexer import DocumentIndexer, DocumentMetadata, FolderScanner, IndexedSource
from eol.rag_context.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraphBuilder,
    Relationship,
    RelationType,
)
from eol.rag_context.redis_client import RedisVectorStore, VectorDocument
from eol.rag_context.semantic_cache import CachedQuery, SemanticCache


class TestRedisVectorStore:
    """Test Redis vector store."""

    def test_init(self):
        """Test initialization."""
        redis_config = RedisConfig()
        index_config = EmbeddingConfig()

        store = RedisVectorStore(redis_config, index_config)
        assert store.redis_config == redis_config
        assert store.redis is None
        assert store.async_redis is None

    @pytest.mark.asyncio
    async def test_connect_async(self):
        """Test async connection."""
        redis_config = RedisConfig()
        index_config = EmbeddingConfig()
        store = RedisVectorStore(redis_config, index_config)

        # Mock Redis - AsyncRedis needs to be awaitable
        with patch("eol.rag_context.redis_client.AsyncRedis") as MockRedis:
            mock_redis = MagicMock()
            MockRedis.return_value = mock_redis

            # Mock the coroutine
            async def mock_connect(*args, **kwargs):
                return mock_redis

            MockRedis.side_effect = mock_connect
            mock_redis.ping = AsyncMock(return_value=True)

            await store.connect_async()

            assert store.async_redis is not None

    def test_vector_document(self):
        """Test VectorDocument dataclass."""
        doc = VectorDocument(
            id="test_id",
            content="test content",
            embedding=np.array([1.0, 2.0, 3.0]),
            metadata={"key": "value"},
            hierarchy_level=2,
            parent_id="parent",
            children_ids=["child1", "child2"],
        )

        assert doc.id == "test_id"
        assert doc.content == "test content"
        assert doc.hierarchy_level == 2
        assert len(doc.children_ids) == 2


class TestSemanticCache:
    """Test semantic cache."""

    def test_init(self):
        """Test cache initialization."""
        cache_config = MagicMock()
        cache_config.similarity_threshold = 0.9
        embedding_manager = MagicMock()
        redis_store = MagicMock()

        cache = SemanticCache(cache_config, embedding_manager, redis_store)

        assert cache.config == cache_config
        assert cache.embeddings == embedding_manager
        assert cache.redis == redis_store
        assert cache.adaptive_threshold == 0.9

    def test_cached_query(self):
        """Test CachedQuery dataclass."""
        query = CachedQuery(
            query="test query",
            response="test response",
            embedding=np.array([1.0, 2.0]),
            hit_count=5,
        )

        assert query.query == "test query"
        assert query.response == "test response"
        assert query.hit_count == 5
        assert query.timestamp > 0

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache_config = MagicMock()
        cache_config.similarity_threshold = 0.9
        cache = SemanticCache(cache_config, MagicMock(), MagicMock())

        cache.stats = {"queries": 100, "hits": 31, "misses": 69}
        stats = cache.get_stats()

        assert stats["queries"] == 100
        assert stats["hits"] == 31
        assert stats["adaptive_threshold"] == 0.9


class TestKnowledgeGraph:
    """Test knowledge graph builder."""

    def test_entity(self):
        """Test Entity dataclass."""
        entity = Entity(
            id="entity_1",
            name="Test Entity",
            type=EntityType.CONCEPT,
            content="Entity content",
            properties={"prop": "value"},
        )

        assert entity.id == "entity_1"
        assert entity.name == "Test Entity"
        assert entity.type == EntityType.CONCEPT

    def test_relationship(self):
        """Test Relationship dataclass."""
        rel = Relationship(
            source_id="entity_1", target_id="entity_2", type=RelationType.RELATES_TO, weight=0.8
        )

        assert rel.source_id == "entity_1"
        assert rel.target_id == "entity_2"
        assert rel.type == RelationType.RELATES_TO
        assert rel.weight == 0.8

    def test_graph_builder_init(self):
        """Test knowledge graph builder initialization."""
        redis_store = MagicMock()
        embedding_manager = MagicMock()

        builder = KnowledgeGraphBuilder(redis_store, embedding_manager)

        assert builder.redis == redis_store
        assert builder.embeddings == embedding_manager
        assert len(builder.entities) == 0
        assert len(builder.relationships) == 0

    def test_get_graph_stats(self):
        """Test getting graph statistics."""
        builder = KnowledgeGraphBuilder(MagicMock(), MagicMock())

        # Add some entities
        builder.entities = {
            "1": Entity("1", "E1", EntityType.CONCEPT),
            "2": Entity("2", "E2", EntityType.FUNCTION),
        }
        builder.relationships = [Relationship("1", "2", RelationType.CONTAINS)]

        stats = builder.get_graph_stats()

        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1


class TestFileWatcher:
    """Test file watcher."""

    def test_file_change(self):
        """Test FileChange dataclass."""
        change = FileChange(
            path=Path("/test/file.py"),
            change_type=ChangeType.MODIFIED,
            old_path=Path("/test/old.py"),
        )

        assert change.path == Path("/test/file.py")
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_path == Path("/test/old.py")
        assert change.timestamp > 0

    def test_watched_source(self):
        """Test WatchedSource dataclass."""
        source = WatchedSource(
            path=Path("/test"), source_id="test_id", recursive=True, file_patterns=["*.py", "*.md"]
        )

        assert source.path == Path("/test")
        assert source.source_id == "test_id"
        assert source.recursive is True
        assert len(source.file_patterns) == 2

    def test_file_watcher_init(self):
        """Test file watcher initialization."""
        indexer = MagicMock()
        watcher = FileWatcher(indexer, debounce_seconds=1.0, batch_size=5)

        assert watcher.indexer == indexer
        assert watcher.debounce_seconds == 1.0
        assert watcher.batch_size == 5
        assert not watcher.is_running

    def test_get_stats(self):
        """Test getting watcher statistics."""
        watcher = FileWatcher(MagicMock())
        watcher.stats = {"changes_detected": 10, "changes_processed": 8, "errors": 1}

        stats = watcher.get_stats()

        assert stats["changes_detected"] == 10
        assert stats["changes_processed"] == 8
        assert stats["errors"] == 1
        assert stats["is_running"] is False


class TestDocumentIndexer:
    """Test document indexer."""

    def test_indexed_source(self):
        """Test IndexedSource dataclass."""
        source = IndexedSource(
            source_id="source_123",
            path=Path("/test"),
            indexed_at=1234567890.0,
            file_count=10,
            total_chunks=50,
            metadata={"key": "value"},
        )

        assert source.source_id == "source_123"
        assert source.path == Path("/test")
        assert source.file_count == 10
        assert source.total_chunks == 50

    def test_document_metadata(self):
        """Test DocumentMetadata dataclass."""
        metadata = DocumentMetadata(
            source_path="/test/file.py",
            source_id="test_id",
            relative_path="file.py",
            file_type="code",
            file_size=1024,
            file_hash="abc123",
            modified_time=1234567890.0,
            indexed_at=1234567891.0,
            chunk_index=0,
            total_chunks=5,
            hierarchy_level=3,
        )

        assert metadata.source_path == "/test/file.py"
        assert metadata.file_type == "code"
        assert metadata.hierarchy_level == 3

        # Test as dict
        metadata_dict = asdict(metadata)
        assert metadata_dict["source_id"] == "test_id"
        assert metadata_dict["file_size"] == 1024

    def test_indexer_init(self):
        """Test indexer initialization."""
        config = RAGConfig(data_dir=Path("/tmp/data"), index_dir=Path("/tmp/index"))
        processor = MagicMock()
        embedding_manager = MagicMock()
        redis_store = MagicMock()

        indexer = DocumentIndexer(config, processor, embedding_manager, redis_store)

        assert indexer.config == config
        assert indexer.processor == processor
        assert indexer.embeddings == embedding_manager
        assert indexer.redis == redis_store

    def test_get_stats(self):
        """Test getting indexer statistics."""
        indexer = DocumentIndexer(MagicMock(), MagicMock(), MagicMock(), MagicMock())

        indexer.stats = {"documents_indexed": 10, "chunks_created": 100, "errors": 2}

        stats = indexer.get_stats()

        assert stats["documents_indexed"] == 10
        assert stats["chunks_created"] == 100
        assert stats["errors"] == 2


class TestProcessedDocument:
    """Test ProcessedDocument dataclass."""

    def test_processed_document(self):
        """Test ProcessedDocument creation."""
        doc = ProcessedDocument(
            file_path=Path("/test/file.md"),
            content="# Test Document",
            doc_type="markdown",
            metadata={"headers": ["Test Document"]},
            chunks=[{"content": "chunk1"}, {"content": "chunk2"}],
            language=None,
        )

        assert doc.file_path == Path("/test/file.md")
        assert doc.doc_type == "markdown"
        assert len(doc.chunks) == 2
        assert doc.language is None
