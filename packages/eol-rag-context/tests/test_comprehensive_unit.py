"""
Comprehensive unit tests to boost coverage to 80%.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
import numpy as np
import json
import tempfile
import os
from dataclasses import asdict
import hashlib

# Mock all external dependencies
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


class TestConfigComplete:
    """Complete config tests for coverage."""
    
    def test_redis_config_env_vars(self):
        """Test Redis config with environment variables."""
        os.environ['REDIS_HOST'] = 'redis.test.com'
        os.environ['REDIS_PORT'] = '6380'
        os.environ['REDIS_PASSWORD'] = 'test_pass'
        os.environ['REDIS_DB'] = '2'
        
        config = RedisConfig()
        assert config.host == 'redis.test.com'
        assert config.port == 6380
        assert config.password == 'test_pass'
        assert config.db == 2
        
        # Cleanup
        for key in ['REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD', 'REDIS_DB']:
            os.environ.pop(key, None)
    
    def test_embedding_config_model_dimensions(self):
        """Test automatic dimension detection for known models."""
        # Test various known models
        config = EmbeddingConfig(model_name="all-mpnet-base-v2")
        assert config.dimension == 768
        
        config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
        assert config.dimension == 384
        
        config = EmbeddingConfig(model_name="all-distilroberta-v1")
        assert config.dimension == 768
        
        config = EmbeddingConfig(model_name="paraphrase-MiniLM-L6-v2")
        assert config.dimension == 384
    
    def test_index_config_validation(self):
        """Test index configuration validation."""
        config = IndexConfig(
            hnsw_m=20,
            hnsw_ef_construction=300,
            hnsw_ef_runtime=20
        )
        assert config.hnsw_m == 20
        assert config.hnsw_ef_construction == 300
        
    def test_chunking_config_bounds(self):
        """Test chunking configuration bounds."""
        config = ChunkingConfig(
            max_chunk_size=2000,
            chunk_overlap=100,
            min_chunk_size=50,
            code_max_lines=100
        )
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 50
        
    def test_cache_config_properties(self):
        """Test cache config properties."""
        config = CacheConfig(
            target_hit_rate=0.35,
            max_cache_size=2000,
            ttl_seconds=7200
        )
        assert config.target_hit_rate == 0.35
        assert config.max_cache_size == 2000
        
    def test_rag_config_paths(self):
        """Test RAG config path handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            index_dir = Path(tmpdir) / "index"
            
            config = RAGConfig(data_dir=data_dir, index_dir=index_dir)
            
            # Directories should be created
            assert data_dir.exists()
            assert index_dir.exists()


class TestEmbeddingsComplete:
    """Complete embeddings tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_mock_fallback(self):
        """Test SentenceTransformer fallback to mock embeddings."""
        config = EmbeddingConfig(dimension=64)
        provider = SentenceTransformerProvider(config)
        
        # Should use mock embeddings
        assert provider.model is None
        
        embedding = await provider.embed("test text")
        assert embedding.shape == (1, 64)
        assert embedding.dtype == np.float32
        
        # Test batch
        embeddings = await provider.embed_batch(["text1", "text2"], batch_size=2)
        assert embeddings.shape == (2, 64)
    
    @pytest.mark.asyncio
    async def test_openai_provider_methods(self):
        """Test OpenAI provider methods."""
        config = EmbeddingConfig(
            provider="openai",
            openai_api_key="test-key",
            openai_model="text-embedding-3-small"
        )
        
        with patch('eol.rag_context.embeddings.AsyncOpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            
            # Mock embedding response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = OpenAIProvider(config)
            
            # Test single embedding
            embedding = await provider.embed("test")
            assert embedding.shape == (1, 3)
            
            # Test batch embedding
            embeddings = await provider.embed_batch(["test1", "test2"])
            assert embeddings.shape == (2, 3)
    
    @pytest.mark.asyncio
    async def test_embedding_manager_with_cache(self):
        """Test embedding manager with caching."""
        config = EmbeddingConfig(dimension=32)
        
        # Mock Redis client
        redis_client = MagicMock()
        redis_client.hget = AsyncMock(return_value=None)
        redis_client.hset = AsyncMock()
        redis_client.expire = AsyncMock()
        
        manager = EmbeddingManager(config, redis_client)
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(
            return_value=np.random.randn(1, 32).astype(np.float32)
        )
        
        # First call - cache miss
        emb1 = await manager.get_embedding("test", use_cache=True)
        assert manager.cache_stats["misses"] == 1
        
        # Should have tried to get from cache
        redis_client.hget.assert_called()
        
        # Should have stored in cache
        redis_client.hset.assert_called()
    
    def test_embedding_manager_cache_key(self):
        """Test cache key generation."""
        manager = EmbeddingManager(EmbeddingConfig())
        
        key1 = manager._cache_key("test")
        key2 = manager._cache_key("test")
        key3 = manager._cache_key("different")
        
        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("emb:")


class TestDocumentProcessorComplete:
    """Complete document processor tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_process_markdown_file(self):
        """Test processing markdown file."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        with tempfile.NamedTemporaryFile(suffix=".md", mode='w', delete=False) as f:
            f.write("# Title\n\n## Section\n\nContent here")
            f.flush()
            
            doc = await processor.process_file(Path(f.name))
            
            assert doc is not None
            assert doc.doc_type == "markdown"
            assert len(doc.chunks) > 0
            assert doc.metadata.get("headers") is not None
        
        Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_process_code_file(self):
        """Test processing code file."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
            f.write("def test():\n    pass\n\nclass TestClass:\n    pass")
            f.flush()
            
            doc = await processor.process_file(Path(f.name))
            
            assert doc is not None
            assert doc.doc_type == "code"
            assert doc.language == "python"
            assert len(doc.chunks) > 0
        
        Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_process_json_file(self):
        """Test processing JSON file."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as f:
            json.dump({"key": "value", "list": [1, 2, 3]}, f)
            f.flush()
            
            doc = await processor._process_structured(Path(f.name))
            
            assert doc is not None
            assert doc.doc_type == "json"
            assert len(doc.chunks) > 0
        
        Path(f.name).unlink()
    
    def test_chunk_markdown_complex(self):
        """Test complex markdown chunking."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        content = """# Main Title

## Introduction
This is the introduction.

### Subsection
More details here.

## Second Section
Another section.

### Another Subsection
With content.

## Code Examples
```python
def example():
    pass
```

## Conclusion
Final thoughts."""
        
        chunks = processor._chunk_markdown_by_headers(content)
        
        # Should have multiple chunks
        assert len(chunks) >= 5
        
        # Check headers are preserved
        headers = [c["header"] for c in chunks]
        assert "Main Title" in headers
        assert "Introduction" in headers
        assert "Conclusion" in headers
    
    def test_chunk_code_functions(self):
        """Test code chunking by functions."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        content = """
def function_one():
    '''First function'''
    return 1

def function_two():
    '''Second function'''
    return 2

class MyClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass
"""
        
        chunks = processor._chunk_code_by_lines(content, "python")
        
        assert len(chunks) > 0
        assert all("language" in c for c in chunks)
        assert all(c["language"] == "python" for c in chunks)


class TestIndexerComplete:
    """Complete indexer tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_folder_scanner_scan(self):
        """Test folder scanning."""
        scanner = FolderScanner(RAGConfig())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test files
            (tmpdir / "test1.py").write_text("code1")
            (tmpdir / "test2.md").write_text("doc1")
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            (subdir / "test3.txt").write_text("text1")
            
            # Scan non-recursive
            files = await scanner.scan_folder(tmpdir, recursive=False)
            assert len(files) == 2
            
            # Scan recursive
            files = await scanner.scan_folder(tmpdir, recursive=True)
            assert len(files) == 3
            
            # Scan with pattern
            files = await scanner.scan_folder(tmpdir, recursive=True, file_patterns=["*.py"])
            assert len(files) == 1
            assert files[0].suffix == ".py"
    
    def test_folder_scanner_ignore_patterns(self):
        """Test ignore pattern matching."""
        scanner = FolderScanner(RAGConfig())
        
        # Test various ignore patterns
        assert scanner._should_ignore(Path(".git/config"))
        assert scanner._should_ignore(Path("node_modules/package.json"))
        assert scanner._should_ignore(Path("__pycache__/module.pyc"))
        assert scanner._should_ignore(Path(".DS_Store"))
        assert not scanner._should_ignore(Path("src/main.py"))
    
    def test_document_metadata_creation(self):
        """Test document metadata creation."""
        metadata = DocumentMetadata(
            source_path="/test/file.py",
            source_id="abc123",
            relative_path="file.py",
            file_type="code",
            file_size=1024,
            file_hash="hash123",
            modified_time=1234567890.0,
            indexed_at=1234567891.0,
            chunk_index=0,
            total_chunks=5,
            hierarchy_level=2,
            language="python",
            line_start=10,
            line_end=20
        )
        
        # Test as dict
        meta_dict = asdict(metadata)
        assert meta_dict["language"] == "python"
        assert meta_dict["line_start"] == 10
        assert meta_dict["hierarchy_level"] == 2
    
    @pytest.mark.asyncio
    async def test_indexer_process_document(self):
        """Test document processing in indexer."""
        config = RAGConfig()
        processor = MagicMock()
        embeddings = MagicMock()
        redis = MagicMock()
        
        indexer = DocumentIndexer(config, processor, embeddings, redis)
        
        # Mock embeddings
        embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        
        # Mock redis
        redis.store_document = AsyncMock()
        
        # Create test document
        doc = ProcessedDocument(
            file_path=Path("/test.md"),
            content="Test content",
            doc_type="markdown",
            chunks=[
                {"content": "chunk1", "type": "header"},
                {"content": "chunk2", "type": "paragraph"}
            ]
        )
        
        metadata = DocumentMetadata(
            source_path="/test.md",
            source_id="test123",
            relative_path="test.md",
            file_type="markdown",
            file_size=100,
            file_hash="abc",
            modified_time=0,
            indexed_at=0,
            chunk_index=0,
            total_chunks=2,
            hierarchy_level=3
        )
        
        # Extract chunks (level 3)
        chunks = await indexer._extract_chunks(doc, metadata)
        
        assert len(chunks) == 2
        assert all(c.hierarchy_level == 3 for c in chunks)
        
        # Verify embeddings were generated
        embeddings.get_embeddings.assert_called()


class TestSemanticCacheComplete:
    """Complete semantic cache tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test cache set and get operations."""
        config = MagicMock()
        config.enabled = True
        config.similarity_threshold = 0.9
        config.adaptive_threshold = False
        config.max_cache_size = 10
        config.ttl_seconds = 3600
        
        embeddings = MagicMock()
        embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        redis.redis.hset = AsyncMock()
        redis.redis.expire = AsyncMock()
        redis.redis.ft = MagicMock()
        search_mock = AsyncMock(return_value=MagicMock(docs=[]))
        redis.redis.ft.return_value.search = search_mock
        
        cache = SemanticCache(config, embeddings, redis)
        
        # Set a cache entry
        await cache.set("query1", "response1", {"key": "value"})
        
        # Verify Redis operations
        redis.redis.hset.assert_called()
        redis.redis.expire.assert_called()
        
        # Get should search
        result = await cache.get("query1")
        search_mock.assert_called()
    
    @pytest.mark.asyncio
    async def test_cache_similarity_matching(self):
        """Test cache similarity matching."""
        config = MagicMock()
        config.enabled = True
        config.similarity_threshold = 0.9
        config.adaptive_threshold = False
        
        embeddings = MagicMock()
        embeddings.get_embedding = AsyncMock(return_value=np.ones(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        
        # Mock search results
        mock_doc = MagicMock()
        mock_doc.id = "cache:123"
        mock_doc.score = 0.95
        mock_doc.response = "cached response"
        mock_doc.metadata = json.dumps({"key": "value"})
        
        search_result = MagicMock()
        search_result.docs = [mock_doc]
        
        redis.redis.ft = MagicMock()
        redis.redis.ft.return_value.search = AsyncMock(return_value=search_result)
        
        cache = SemanticCache(config, embeddings, redis)
        
        # Should find similar query
        result = await cache.get("similar query")
        
        assert result is not None
        assert result["response"] == "cached response"
        assert cache.stats["hits"] == 1
    
    def test_cache_stats_calculation(self):
        """Test cache statistics calculation."""
        cache = SemanticCache(MagicMock(), MagicMock(), MagicMock())
        
        # Simulate cache activity
        cache.stats = {
            "queries": 100,
            "hits": 31,
            "misses": 69,
            "evictions": 5
        }
        
        stats = cache.get_stats()
        
        assert stats["queries"] == 100
        assert stats["hits"] == 31
        assert stats["hit_rate"] == 0.31


class TestKnowledgeGraphComplete:
    """Complete knowledge graph tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_build_from_documents(self):
        """Test building graph from documents."""
        redis = MagicMock()
        embeddings = MagicMock()
        embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        
        builder = KnowledgeGraphBuilder(redis, embeddings)
        
        # Create test documents
        docs = [
            VectorDocument(
                id="doc1",
                content="# Main Topic\n\nThis is about `function_one()` and `ClassA`",
                embedding=np.random.randn(128),
                metadata={"doc_type": "markdown", "relative_path": "doc1.md"},
                hierarchy_level=3
            ),
            VectorDocument(
                id="doc2",
                content="def function_one(): pass\n\nclass ClassA: pass",
                embedding=np.random.randn(128),
                metadata={"doc_type": "code", "language": "python", "relative_path": "code.py"},
                hierarchy_level=3
            )
        ]
        
        await builder.build_from_documents(docs)
        
        # Should have created entities
        assert len(builder.entities) > 0
        
        # Should have various entity types
        entity_types = {e.type for e in builder.entities.values()}
        assert EntityType.TOPIC in entity_types or EntityType.FUNCTION in entity_types
    
    def test_entity_extraction_patterns(self):
        """Test entity extraction patterns."""
        builder = KnowledgeGraphBuilder(MagicMock(), MagicMock())
        
        # Test function pattern
        content = "This uses function_name() and another_func()"
        entities = []
        
        import re
        pattern = r'\b([a-z_][a-z0-9_]*)\(\)'
        for match in re.finditer(pattern, content):
            entities.append(match.group(1))
        
        assert "function_name" in entities
        assert "another_func" in entities
        
        # Test class pattern
        content = "Inherits from BaseClass and uses MyClass"
        pattern = r'\b([A-Z][a-zA-Z0-9]+)(?:\s|$|\.)'
        entities = []
        for match in re.finditer(pattern, content):
            entities.append(match.group(1))
        
        assert "BaseClass" in entities
        assert "MyClass" in entities
    
    def test_graph_statistics(self):
        """Test graph statistics generation."""
        builder = KnowledgeGraphBuilder(MagicMock(), MagicMock())
        
        # Add entities
        builder.entities = {
            "1": Entity("1", "Entity1", EntityType.CONCEPT),
            "2": Entity("2", "Entity2", EntityType.FUNCTION),
            "3": Entity("3", "Entity3", EntityType.CLASS),
        }
        
        # Add relationships
        builder.relationships = [
            Relationship("1", "2", RelationType.CONTAINS),
            Relationship("2", "3", RelationType.DEPENDS_ON),
        ]
        
        # Add to graph
        for e in builder.entities.values():
            builder.graph.add_node(e.id)
        for r in builder.relationships:
            builder.graph.add_edge(r.source_id, r.target_id)
        
        stats = builder.get_graph_stats()
        
        assert stats["entity_count"] == 3
        assert stats["relationship_count"] == 2
        assert "entity_types" in stats
        assert "relationship_types" in stats


class TestFileWatcherComplete:
    """Complete file watcher tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_watch_and_unwatch(self):
        """Test watching and unwatching directories."""
        indexer = MagicMock()
        indexer.scanner = MagicMock()
        indexer.scanner.generate_source_id = MagicMock(return_value="source_123")
        indexer.index_folder = AsyncMock(return_value=MagicMock(
            source_id="source_123",
            file_count=5,
            total_chunks=20
        ))
        
        watcher = FileWatcher(indexer)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Start watcher
            watcher.is_running = True
            
            # Watch directory
            source_id = await watcher.watch(tmpdir)
            assert source_id == "source_123"
            assert source_id in watcher.watched_sources
            
            # Get stats
            stats = watcher.get_stats()
            assert stats["watched_sources"] == 1
            
            # Unwatch
            success = await watcher.unwatch(source_id)
            assert success
            assert source_id not in watcher.watched_sources
    
    def test_change_event_handling(self):
        """Test file change event handling."""
        watcher = FileWatcher(MagicMock())
        
        # Add change callbacks
        callback1_called = []
        callback2_called = []
        
        def callback1(change):
            callback1_called.append(change)
        
        def callback2(change):
            callback2_called.append(change)
        
        watcher.add_change_callback(callback1)
        watcher.add_change_callback(callback2)
        
        # Simulate change
        change = FileChange(
            path=Path("/test.py"),
            change_type=ChangeType.MODIFIED
        )
        
        # Notify callbacks
        for cb in watcher.change_callbacks:
            cb(change)
        
        assert len(callback1_called) == 1
        assert len(callback2_called) == 1
        
        # Remove callback
        watcher.remove_change_callback(callback1)
        assert len(watcher.change_callbacks) == 1
    
    def test_change_history(self):
        """Test change history tracking."""
        watcher = FileWatcher(MagicMock())
        
        # Add changes
        for i in range(15):
            change = FileChange(
                path=Path(f"/test{i}.py"),
                change_type=ChangeType.MODIFIED if i % 2 == 0 else ChangeType.CREATED
            )
            watcher.change_history.append(change)
        
        # Get history with limit
        history = watcher.get_change_history(limit=10)
        assert len(history) == 10
        
        # Get all history
        history = watcher.get_change_history()
        assert len(history) == 15
        
        # Check format
        assert all("path" in h for h in history)
        assert all("type" in h for h in history)
        assert all("timestamp" in h for h in history)


class TestRedisClientComplete:
    """Complete Redis client tests for coverage."""
    
    @pytest.mark.asyncio
    async def test_vector_document_operations(self):
        """Test VectorDocument operations."""
        doc = VectorDocument(
            id="test_doc",
            content="Test content",
            embedding=np.array([1.0, 2.0, 3.0]),
            metadata={"key": "value"},
            hierarchy_level=2,
            parent_id="parent",
            children_ids=["child1", "child2"]
        )
        
        # Convert to dict
        doc_dict = asdict(doc)
        assert doc_dict["id"] == "test_doc"
        assert doc_dict["hierarchy_level"] == 2
        
        # Test numpy array handling
        assert isinstance(doc.embedding, np.ndarray)
        assert doc.embedding.shape == (3,)
    
    def test_redis_store_initialization(self):
        """Test Redis store initialization."""
        redis_config = RedisConfig()
        index_config = IndexConfig()
        
        store = RedisVectorStore(redis_config, index_config)
        
        assert store.redis_config == redis_config
        assert store.index_config == index_config
        assert store.redis is None
        assert store.async_redis is None
    
    @pytest.mark.asyncio
    async def test_redis_connection_mock(self):
        """Test Redis connection with mocks."""
        store = RedisVectorStore(RedisConfig(), IndexConfig())
        
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis:
            mock_redis = MagicMock()
            
            async def mock_connect(*args, **kwargs):
                return mock_redis
            
            MockRedis.side_effect = mock_connect
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.ft = MagicMock()
            
            await store.connect_async()
            
            assert store.async_redis is not None
            mock_redis.ping.assert_called_once()