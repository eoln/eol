"""
Simple unit tests focused on increasing coverage without complex mocks.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
import json
import tempfile
from dataclasses import asdict, fields

# Mock external dependencies
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'fastmcp', 'sentence_transformers']:
    sys.modules[module] = MagicMock()

from eol.rag_context.config import *
from eol.rag_context.embeddings import *
from eol.rag_context.document_processor import *
from eol.rag_context.indexer import *
from eol.rag_context.redis_client import *
from eol.rag_context.semantic_cache import *
from eol.rag_context.knowledge_graph import *
from eol.rag_context.file_watcher import *


class TestDataClasses:
    """Test all dataclass functionality."""
    
    def test_vector_document(self):
        """Test VectorDocument dataclass."""
        doc = VectorDocument(
            id="test",
            content="content",
            embedding=np.array([1, 2, 3]),
            metadata={"key": "val"},
            hierarchy_level=1
        )
        assert doc.id == "test"
        assert doc.parent_id is None
        assert doc.children_ids == []
        
        # Test with all fields
        doc2 = VectorDocument(
            id="test2",
            content="content2",
            embedding=np.array([4, 5, 6]),
            metadata={},
            hierarchy_level=2,
            parent_id="parent",
            children_ids=["child1", "child2"]
        )
        assert doc2.parent_id == "parent"
        assert len(doc2.children_ids) == 2
    
    def test_cached_query(self):
        """Test CachedQuery dataclass."""
        query = CachedQuery(
            query="test query",
            response="test response",
            embedding=np.array([1, 2]),
            hit_count=0
        )
        assert query.query == "test query"
        assert query.hit_count == 0
        assert query.timestamp > 0
        
        # With metadata
        query2 = CachedQuery(
            query="q2",
            response="r2",
            embedding=np.array([3, 4]),
            metadata={"key": "val"},
            hit_count=5
        )
        assert query2.metadata == {"key": "val"}
    
    def test_entity(self):
        """Test Entity dataclass."""
        entity = Entity(
            id="e1",
            name="Entity One",
            type=EntityType.CONCEPT
        )
        assert entity.id == "e1"
        assert entity.type == EntityType.CONCEPT
        assert entity.content is None
        assert entity.properties == {}
        
        # With all fields
        entity2 = Entity(
            id="e2",
            name="Entity Two",
            type=EntityType.FUNCTION,
            content="function content",
            properties={"lang": "python"}
        )
        assert entity2.content == "function content"
        assert entity2.properties["lang"] == "python"
    
    def test_relationship(self):
        """Test Relationship dataclass."""
        rel = Relationship(
            source_id="s1",
            target_id="t1",
            type=RelationType.DEPENDS_ON
        )
        assert rel.source_id == "s1"
        assert rel.type == RelationType.DEPENDS_ON
        assert rel.weight == 1.0
        assert rel.properties == {}
        
        # With weight and properties
        rel2 = Relationship(
            source_id="s2",
            target_id="t2",
            type=RelationType.SIMILAR_TO,
            weight=0.8,
            properties={"reason": "test"}
        )
        assert rel2.weight == 0.8
        assert rel2.properties["reason"] == "test"
    
    def test_file_change(self):
        """Test FileChange dataclass."""
        change = FileChange(
            path=Path("/test.py"),
            change_type=ChangeType.MODIFIED
        )
        assert change.path == Path("/test.py")
        assert change.change_type == ChangeType.MODIFIED
        assert change.old_path is None
        assert change.timestamp > 0
        
        # With old path (rename)
        change2 = FileChange(
            path=Path("/new.py"),
            change_type=ChangeType.MOVED,
            old_path=Path("/old.py")
        )
        assert change2.old_path == Path("/old.py")
    
    def test_watched_source(self):
        """Test WatchedSource dataclass."""
        source = WatchedSource(
            path=Path("/src"),
            source_id="src123",
            recursive=True
        )
        assert source.path == Path("/src")
        assert source.recursive is True
        assert source.file_patterns == []
        assert source.ignore_patterns == []
        
        # With patterns
        source2 = WatchedSource(
            path=Path("/src2"),
            source_id="src456",
            recursive=False,
            file_patterns=["*.py", "*.md"],
            ignore_patterns=["test_*"]
        )
        assert len(source2.file_patterns) == 2
        assert "test_*" in source2.ignore_patterns
    
    def test_indexed_source(self):
        """Test IndexedSource dataclass."""
        source = IndexedSource(
            source_id="idx123",
            path=Path("/project"),
            indexed_at=1234567890.0,
            file_count=10,
            total_chunks=50
        )
        assert source.source_id == "idx123"
        assert source.file_count == 10
        assert source.metadata == {}
        
        # With metadata
        source2 = IndexedSource(
            source_id="idx456",
            path=Path("/other"),
            indexed_at=1234567891.0,
            file_count=5,
            total_chunks=20,
            metadata={"version": "1.0"}
        )
        assert source2.metadata["version"] == "1.0"
    
    def test_processed_document(self):
        """Test ProcessedDocument dataclass."""
        doc = ProcessedDocument(
            file_path=Path("/test.md"),
            content="# Test",
            doc_type="markdown"
        )
        assert doc.file_path == Path("/test.md")
        assert doc.metadata == {}
        assert doc.chunks == []
        assert doc.language is None
        
        # With all fields
        doc2 = ProcessedDocument(
            file_path=Path("/code.py"),
            content="def test(): pass",
            doc_type="code",
            metadata={"lines": 1},
            chunks=[{"content": "def test(): pass"}],
            language="python"
        )
        assert doc2.language == "python"
        assert len(doc2.chunks) == 1
    
    def test_document_metadata(self):
        """Test DocumentMetadata dataclass."""
        meta = DocumentMetadata(
            source_path="/test.py",
            source_id="src123",
            relative_path="test.py",
            file_type="code",
            file_size=1024,
            file_hash="abc123",
            modified_time=123.0,
            indexed_at=124.0,
            chunk_index=0,
            total_chunks=5,
            hierarchy_level=3
        )
        assert meta.source_path == "/test.py"
        assert meta.hierarchy_level == 3
        
        # Test optional fields
        assert meta.language is None
        assert meta.line_start is None
        assert meta.parent_chunk_id is None
        
        # With optional fields
        meta2 = DocumentMetadata(
            source_path="/test2.py",
            source_id="src456",
            relative_path="test2.py",
            file_type="code",
            file_size=2048,
            file_hash="def456",
            modified_time=125.0,
            indexed_at=126.0,
            chunk_index=1,
            total_chunks=10,
            hierarchy_level=2,
            language="python",
            line_start=10,
            line_end=20,
            parent_chunk_id="parent123",
            chunk_type="function",
            semantic_density=0.8,
            git_commit="abc",
            git_branch="main",
            tags=["test", "code"]
        )
        assert meta2.language == "python"
        assert meta2.line_start == 10
        assert meta2.semantic_density == 0.8
        assert "test" in meta2.tags


class TestEnums:
    """Test all enum values."""
    
    def test_entity_type(self):
        """Test EntityType enum."""
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.SECTION.value == "section"
        assert EntityType.TOPIC.value == "topic"
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.CLASS.value == "class"
        assert EntityType.MODULE.value == "module"
        assert EntityType.API.value == "api"
        assert EntityType.FEATURE.value == "feature"
        assert EntityType.DEPENDENCY.value == "dependency"
    
    def test_relation_type(self):
        """Test RelationType enum."""
        assert RelationType.CONTAINS.value == "contains"
        assert RelationType.DEPENDS_ON.value == "depends_on"
        assert RelationType.RELATES_TO.value == "relates_to"
        assert RelationType.SIMILAR_TO.value == "similar_to"
        assert RelationType.CALLS.value == "calls"
        assert RelationType.IMPLEMENTS.value == "implements"
        assert RelationType.EXTENDS.value == "extends"
        assert RelationType.USES.value == "uses"
    
    def test_change_type(self):
        """Test ChangeType enum."""
        assert ChangeType.CREATED.value == "created"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
        assert ChangeType.MOVED.value == "moved"


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_redis_config_defaults(self):
        """Test Redis config defaults."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.socket_keepalive is True
        assert config.max_connections == 50
    
    def test_embedding_config_defaults(self):
        """Test embedding config defaults."""
        config = EmbeddingConfig()
        assert config.provider == "sentence-transformers"
        assert config.model_name == "all-mpnet-base-v2"
        assert config.dimension == 768  # Should be auto-set for known model
        assert config.batch_size == 32
        assert config.cache_embeddings is True
    
    def test_index_config_defaults(self):
        """Test index config defaults."""
        config = IndexConfig()
        assert config.name == "rag_context_index"
        assert config.prefix == "doc"
        assert config.distance_metric == "COSINE"
        assert config.initial_cap == 1000
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construction == 200
        assert config.hnsw_ef_runtime == 10
    
    def test_chunking_config_defaults(self):
        """Test chunking config defaults."""
        config = ChunkingConfig()
        assert config.max_chunk_size == 1024
        assert config.chunk_overlap == 128
        assert config.min_chunk_size == 100
        assert config.use_semantic_chunking is True
        assert config.code_max_lines == 50
        assert config.preserve_formatting is True
    
    def test_cache_config_defaults(self):
        """Test cache config defaults."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_cache_size == 1000
        assert config.similarity_threshold == 0.95
        assert config.target_hit_rate == 0.31
        assert config.adaptive_threshold is False
    
    def test_context_config_defaults(self):
        """Test context config defaults."""
        config = ContextConfig()
        assert config.max_context_length == 128000
        assert config.compression_threshold == 0.8
        assert config.quality_threshold == 0.7
        assert config.progressive_loading is True
        assert config.remove_redundancy is False
        assert config.redundancy_threshold == 0.9
    
    def test_document_config_defaults(self):
        """Test document config defaults."""
        config = DocumentConfig()
        assert config.max_file_size_mb == 100
        assert config.skip_binary_files is True
        assert config.detect_language is True
        assert "*.py" in config.file_patterns
        assert "*.md" in config.file_patterns
    
    def test_rag_config_paths(self):
        """Test RAG config path handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data"
            index_dir = tmpdir / "index"
            
            config = RAGConfig(
                data_dir=data_dir,
                index_dir=index_dir
            )
            
            assert config.data_dir == data_dir
            assert config.index_dir == index_dir
            assert data_dir.exists()
            assert index_dir.exists()


class TestSimpleMethods:
    """Test simple methods that don't require complex mocking."""
    
    def test_document_processor_language_detection(self):
        """Test language detection from file extension."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        assert processor._detect_language(".py") == "python"
        assert processor._detect_language(".js") == "javascript"
        assert processor._detect_language(".ts") == "typescript"
        assert processor._detect_language(".java") == "java"
        assert processor._detect_language(".go") == "go"
        assert processor._detect_language(".rs") == "rust"
        assert processor._detect_language(".cpp") == "cpp"
        assert processor._detect_language(".c") == "c"
        assert processor._detect_language(".cs") == "csharp"
        assert processor._detect_language(".rb") == "ruby"
        assert processor._detect_language(".php") == "php"
        assert processor._detect_language(".swift") == "swift"
        assert processor._detect_language(".kt") == "kotlin"
        assert processor._detect_language(".scala") == "scala"
        assert processor._detect_language(".r") == "r"
        assert processor._detect_language(".m") == "matlab"
        assert processor._detect_language(".jl") == "julia"
        assert processor._detect_language(".sh") == "bash"
        assert processor._detect_language(".unknown") == "unknown"
    
    def test_folder_scanner_source_id(self):
        """Test source ID generation."""
        scanner = FolderScanner(RAGConfig())
        
        id1 = scanner.generate_source_id(Path("/test/path"))
        id2 = scanner.generate_source_id(Path("/test/path"))
        id3 = scanner.generate_source_id(Path("/other/path"))
        
        assert id1 == id2  # Same path = same ID
        assert id1 != id3  # Different path = different ID
        assert len(id1) == 16
        assert all(c in "0123456789abcdef" for c in id1)
    
    def test_indexer_summary_generation(self):
        """Test summary generation."""
        indexer = DocumentIndexer(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        
        # Short content
        summary = indexer._generate_summary("Short content")
        assert summary == "Short content"
        
        # Long content
        long_content = "word " * 200
        summary = indexer._generate_summary(long_content)
        assert len(summary) <= 500
        
        # Empty content
        summary = indexer._generate_summary("")
        assert summary == ""
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = SemanticCache(MagicMock(), MagicMock(), MagicMock())
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["queries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Update stats
        cache.stats["queries"] = 100
        cache.stats["hits"] = 31
        cache.stats["misses"] = 69
        
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.31
    
    def test_file_watcher_stats(self):
        """Test file watcher statistics."""
        watcher = FileWatcher(MagicMock())
        
        # Initial stats
        stats = watcher.get_stats()
        assert stats["is_running"] is False
        assert stats["watched_sources"] == 0
        assert stats["changes_detected"] == 0
        
        # Update stats
        watcher.stats["changes_detected"] = 10
        watcher.stats["changes_processed"] = 8
        watcher.is_running = True
        watcher.watched_sources["src1"] = MagicMock()
        
        stats = watcher.get_stats()
        assert stats["is_running"] is True
        assert stats["watched_sources"] == 1
        assert stats["changes_detected"] == 10
        assert stats["changes_processed"] == 8
    
    def test_knowledge_graph_stats(self):
        """Test knowledge graph statistics."""
        builder = KnowledgeGraphBuilder(MagicMock(), MagicMock())
        
        # Initial stats
        stats = builder.get_graph_stats()
        assert stats["entity_count"] == 0
        assert stats["relationship_count"] == 0
        
        # Add entities and relationships
        builder.entities["1"] = Entity("1", "E1", EntityType.CONCEPT)
        builder.entities["2"] = Entity("2", "E2", EntityType.FUNCTION)
        builder.relationships.append(Relationship("1", "2", RelationType.CONTAINS))
        
        stats = builder.get_graph_stats()
        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1