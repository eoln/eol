"""
Maximum coverage tests - directly test all accessible code paths.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, PropertyMock
import numpy as np
import json
import tempfile
import os
from dataclasses import asdict
import asyncio
import hashlib

# Mock all external dependencies before imports
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'fastmcp', 'fastmcp.server', 'sentence_transformers', 'openai',
               'tree_sitter', 'tree_sitter_python', 'pypdf.PdfReader']:
    sys.modules[module] = MagicMock()

# Now import our modules
from eol.rag_context.config import *
from eol.rag_context.embeddings import *
from eol.rag_context.document_processor import *
from eol.rag_context.indexer import *
from eol.rag_context.redis_client import *
from eol.rag_context.semantic_cache import *
from eol.rag_context.knowledge_graph import *
from eol.rag_context.file_watcher import *


class TestConfigCoverage:
    """Maximize config module coverage."""
    
    def test_all_config_classes(self):
        """Test all configuration classes."""
        # RedisConfig
        redis_cfg = RedisConfig(host="test", port=1234, password="pass", db=1)
        assert redis_cfg.url == "redis://:pass@test:1234/1"
        
        redis_cfg2 = RedisConfig(password=None)
        assert "redis://localhost:6379/0" in redis_cfg2.url
        
        # EmbeddingConfig with dimension validation
        emb_cfg = EmbeddingConfig(model_name="all-mpnet-base-v2", dimension=100)
        assert emb_cfg.dimension == 768  # Should be corrected
        
        emb_cfg2 = EmbeddingConfig(model_name="unknown-model", dimension=256)
        assert emb_cfg2.dimension == 256  # Should keep custom dimension
        
        # IndexConfig
        idx_cfg = IndexConfig(hnsw_m=20, hnsw_ef_construction=300)
        assert idx_cfg.hnsw_m == 20
        
        # ChunkingConfig
        chunk_cfg = ChunkingConfig(max_chunk_size=2000, min_chunk_size=50)
        assert chunk_cfg.max_chunk_size == 2000
        
        # CacheConfig
        cache_cfg = CacheConfig(target_hit_rate=0.35, max_cache_size=2000)
        assert cache_cfg.target_hit_rate == 0.35
        
        # ContextConfig
        ctx_cfg = ContextConfig(max_context_length=100000, compression_threshold=0.7)
        assert ctx_cfg.max_context_length == 100000
        
        # DocumentConfig
        doc_cfg = DocumentConfig(max_file_size_mb=200, file_patterns=["*.py", "*.md"])
        assert doc_cfg.max_file_size_mb == 200
        assert "*.py" in doc_cfg.file_patterns
        
        # RAGConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            rag_cfg = RAGConfig(
                data_dir=Path(tmpdir) / "data",
                index_dir=Path(tmpdir) / "index"
            )
            assert rag_cfg.data_dir.exists()
            assert rag_cfg.index_dir.exists()


class TestEmbeddingsCoverage:
    """Maximize embeddings module coverage."""
    
    @pytest.mark.asyncio
    async def test_mock_provider(self):
        """Test MockEmbeddingsProvider."""
        cfg = EmbeddingConfig(dimension=64)
        provider = MockEmbeddingsProvider(cfg)
        
        # Single embed
        emb = await provider.embed("test")
        assert emb.shape == (1, 64)
        assert emb.dtype == np.float32
        
        # Batch embed
        embs = await provider.embed_batch(["a", "b", "c"], batch_size=2)
        assert embs.shape == (3, 64)
    
    @pytest.mark.asyncio
    async def test_sentence_transformer_provider(self):
        """Test SentenceTransformerProvider."""
        cfg = EmbeddingConfig(model_name="test-model", dimension=32)
        provider = SentenceTransformerProvider(cfg)
        
        # Should fall back to mock when model not available
        assert provider.model is None
        
        emb = await provider.embed("test")
        assert emb.shape == (1, 32)
        
        embs = await provider.embed_batch(["a", "b"], batch_size=1)
        assert embs.shape == (2, 32)
    
    @pytest.mark.asyncio
    async def test_openai_provider(self):
        """Test OpenAIProvider."""
        # Without API key
        cfg = EmbeddingConfig(provider="openai", openai_api_key=None)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(cfg)
        
        # With API key
        cfg = EmbeddingConfig(provider="openai", openai_api_key="test-key")
        with patch('eol.rag_context.embeddings.AsyncOpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            
            provider = OpenAIProvider(cfg)
            assert provider.client is not None
            
            # Mock embedding response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            emb = await provider.embed("test")
            assert emb.shape == (1, 3)
            
            embs = await provider.embed_batch(["a", "b"])
            assert embs.shape == (2, 3)
    
    @pytest.mark.asyncio
    async def test_embedding_manager(self):
        """Test EmbeddingManager."""
        cfg = EmbeddingConfig(dimension=32)
        manager = EmbeddingManager(cfg)
        
        # Mock provider
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(return_value=np.random.randn(1, 32))
        manager.provider.embed_batch = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 32)
        )
        
        # Test single embedding
        emb = await manager.get_embedding("test", use_cache=False)
        assert emb.shape == (32,) or emb.shape == (1, 32)
        assert manager.cache_stats["misses"] == 1
        
        # Test batch embeddings
        embs = await manager.get_embeddings(["a", "b"], use_cache=False)
        assert embs.shape == (2, 32)
        
        # Test with cache (mock Redis)
        redis_client = MagicMock()
        redis_client.hget = AsyncMock(return_value=None)
        redis_client.hset = AsyncMock()
        redis_client.expire = AsyncMock()
        
        manager2 = EmbeddingManager(cfg, redis_client)
        manager2.provider = manager.provider
        
        emb = await manager2.get_embedding("cached", use_cache=True)
        redis_client.hget.assert_called()
        
        # Test cache stats
        stats = manager.get_cache_stats()
        assert "hit_rate" in stats
    
    def test_embedding_manager_initialization(self):
        """Test EmbeddingManager initialization with different providers."""
        # Sentence transformers (default)
        cfg = EmbeddingConfig(provider="sentence-transformers")
        manager = EmbeddingManager(cfg)
        assert isinstance(manager.provider, SentenceTransformerProvider)
        
        # OpenAI
        cfg = EmbeddingConfig(provider="openai", openai_api_key="test")
        with patch('eol.rag_context.embeddings.AsyncOpenAI'):
            manager = EmbeddingManager(cfg)
            assert isinstance(manager.provider, OpenAIProvider)
        
        # Unknown provider
        cfg = EmbeddingConfig(provider="unknown")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingManager(cfg)


class TestDocumentProcessorCoverage:
    """Maximize document processor module coverage."""
    
    @pytest.mark.asyncio
    async def test_process_files(self):
        """Test file processing."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        # Test text file
        with tempfile.NamedTemporaryFile(suffix=".txt", mode='w', delete=False) as f:
            f.write("Test content")
            f.flush()
            
            doc = await processor.process_file(Path(f.name))
            assert doc.doc_type == "text"
            assert doc.content == "Test content"
            assert len(doc.chunks) > 0
        
        Path(f.name).unlink()
        
        # Test markdown file
        with tempfile.NamedTemporaryFile(suffix=".md", mode='w', delete=False) as f:
            f.write("# Title\n\nContent")
            f.flush()
            
            doc = await processor.process_file(Path(f.name))
            assert doc.doc_type == "markdown"
            assert len(doc.chunks) > 0
        
        Path(f.name).unlink()
        
        # Test code file
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
            f.write("def test():\n    pass")
            f.flush()
            
            doc = await processor.process_file(Path(f.name))
            assert doc.doc_type == "code"
            assert doc.language == "python"
        
        Path(f.name).unlink()
    
    @pytest.mark.asyncio
    async def test_process_structured(self):
        """Test structured file processing."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        # JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as f:
            json.dump({"key": "value"}, f)
            f.flush()
            
            doc = await processor._process_structured(Path(f.name))
            assert doc.doc_type == "json"
        
        Path(f.name).unlink()
        
        # YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
            f.write("key: value")
            f.flush()
            
            doc = await processor._process_structured(Path(f.name))
            assert doc.doc_type == "yaml"
        
        Path(f.name).unlink()
    
    def test_chunking_methods(self):
        """Test various chunking methods."""
        processor = DocumentProcessor(DocumentConfig(), ChunkingConfig())
        
        # Chunk by AST (fallback to lines)
        code = "def test():\n    pass\n\nclass Test:\n    pass"
        chunks = processor._chunk_code_by_ast(code, None, "python")
        assert len(chunks) > 0
        
        # Chunk PDF content
        pages = ["Page 1 content", "Page 2 content"]
        chunks = processor._chunk_pdf_content(pages)
        assert len(chunks) == 2
        
        # Chunk markdown
        md = "# Title\n\n## Section 1\n\nContent\n\n## Section 2\n\nMore content"
        chunks = processor._chunk_markdown_by_headers(md)
        assert len(chunks) >= 2


class TestIndexerCoverage:
    """Maximize indexer module coverage."""
    
    @pytest.mark.asyncio
    async def test_folder_scanner(self):
        """Test FolderScanner."""
        scanner = FolderScanner(RAGConfig())
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test files
            (tmpdir / "test.py").write_text("code")
            (tmpdir / ".git").mkdir()
            (tmpdir / ".git" / "config").write_text("git")
            
            # Scan folder
            files = await scanner.scan_folder(tmpdir, recursive=True)
            
            # Should exclude .git
            assert all(".git" not in str(f) for f in files)
            
            # Test with patterns
            files = await scanner.scan_folder(tmpdir, file_patterns=["*.py"])
            assert len(files) == 1
    
    @pytest.mark.asyncio
    async def test_document_indexer(self):
        """Test DocumentIndexer."""
        cfg = RAGConfig()
        processor = MagicMock()
        embeddings = MagicMock()
        redis = MagicMock()
        
        indexer = DocumentIndexer(cfg, processor, embeddings, redis)
        
        # Mock methods
        embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        redis.store_document = AsyncMock()
        
        # Create test document
        doc = ProcessedDocument(
            file_path=Path("/test.md"),
            content="Test content " * 100,
            doc_type="markdown",
            chunks=[{"content": "chunk1"}, {"content": "chunk2"}]
        )
        
        metadata = DocumentMetadata(
            source_path="/test.md",
            source_id="test123",
            relative_path="test.md",
            file_type="markdown",
            file_size=1024,
            file_hash="abc",
            modified_time=0,
            indexed_at=0,
            chunk_index=0,
            total_chunks=2,
            hierarchy_level=3
        )
        
        # Extract concepts
        concepts = await indexer._extract_concepts(doc, metadata)
        assert len(concepts) > 0
        
        # Extract sections
        sections = await indexer._extract_sections(doc, metadata, "concept_id")
        assert len(sections) > 0
        
        # Extract chunks
        chunks = await indexer._extract_chunks(doc, metadata)
        assert len(chunks) == 2


class TestSemanticCacheCoverage:
    """Maximize semantic cache module coverage."""
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache operations."""
        cfg = MagicMock()
        cfg.enabled = True
        cfg.similarity_threshold = 0.9
        cfg.adaptive_threshold = False
        cfg.max_cache_size = 10
        cfg.ttl_seconds = 3600
        cfg.target_hit_rate = 0.31
        
        embeddings = MagicMock()
        embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        redis.redis.hset = AsyncMock()
        redis.redis.expire = AsyncMock()
        redis.redis.hlen = AsyncMock(return_value=5)
        redis.redis.ft = MagicMock()
        redis.redis.ft.return_value.search = AsyncMock(return_value=MagicMock(docs=[]))
        
        cache = SemanticCache(cfg, embeddings, redis)
        
        # Set entry
        await cache.set("query", "response", {"meta": "data"})
        
        # Get entry (miss)
        result = await cache.get("query")
        
        # Clear cache
        redis.redis.delete = AsyncMock()
        redis.redis.keys = AsyncMock(return_value=["cache:1", "cache:2"])
        await cache.clear()
        
        # Get optimization report
        async def mock_cache_size():
            return 10
        cache._get_cache_size = mock_cache_size
        cache.similarity_scores = [0.8, 0.9, 0.95] * 10
        
        report = await cache.get_optimization_report()
        assert "recommendations" in report


class TestKnowledgeGraphCoverage:
    """Maximize knowledge graph module coverage."""
    
    @pytest.mark.asyncio
    async def test_graph_building(self):
        """Test graph building."""
        redis = MagicMock()
        embeddings = MagicMock()
        embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        redis.store_entities = AsyncMock()
        redis.store_relationships = AsyncMock()
        
        builder = KnowledgeGraphBuilder(redis, embeddings)
        
        # Build from documents
        docs = [
            VectorDocument(
                id="doc1",
                content="# Topic\n\nUses `function()` and ClassA",
                embedding=np.random.randn(128),
                metadata={"doc_type": "markdown"},
                hierarchy_level=3
            ),
            VectorDocument(
                id="doc2",
                content="def function(): pass\nclass ClassA: pass",
                embedding=np.random.randn(128),
                metadata={"doc_type": "code", "language": "python"},
                hierarchy_level=3
            )
        ]
        
        await builder.build_from_documents(docs)
        assert len(builder.entities) > 0
        
        # Query subgraph
        subgraph = await builder.query_subgraph("test", max_depth=2)
        assert "entities" in subgraph
        
        # Get stats
        stats = builder.get_graph_stats()
        assert "entity_count" in stats


class TestFileWatcherCoverage:
    """Maximize file watcher module coverage."""
    
    @pytest.mark.asyncio
    async def test_watcher_operations(self):
        """Test watcher operations."""
        indexer = MagicMock()
        indexer.scanner = MagicMock()
        indexer.scanner.generate_source_id = MagicMock(return_value="src123")
        indexer.index_folder = AsyncMock(return_value=MagicMock(
            source_id="src123",
            file_count=5,
            total_chunks=20
        ))
        indexer.remove_source = AsyncMock()
        
        watcher = FileWatcher(indexer)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Start watcher
            with patch.object(watcher, '_start_observer'):
                await watcher.start()
                assert watcher.is_running
            
            # Watch directory
            source_id = await watcher.watch(tmpdir)
            assert source_id == "src123"
            
            # Unwatch
            success = await watcher.unwatch(source_id)
            assert success
            
            # Stop watcher
            with patch.object(watcher, 'observer', create=True) as mock_observer:
                mock_observer.stop = MagicMock()
                mock_observer.join = MagicMock()
                await watcher.stop()
                assert not watcher.is_running
    
    def test_change_callbacks(self):
        """Test change callbacks."""
        watcher = FileWatcher(MagicMock())
        
        called = []
        def callback(change):
            called.append(change)
        
        watcher.add_change_callback(callback)
        
        # Simulate change
        change = FileChange(Path("/test.py"), ChangeType.MODIFIED)
        for cb in watcher.change_callbacks:
            cb(change)
        
        assert len(called) == 1


class TestRedisClientCoverage:
    """Maximize Redis client module coverage."""
    
    @pytest.mark.asyncio
    async def test_redis_operations(self):
        """Test Redis operations."""
        redis_cfg = RedisConfig()
        index_cfg = IndexConfig()
        
        store = RedisVectorStore(redis_cfg, index_cfg)
        
        # Mock Redis connection
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis:
            mock_redis = MagicMock()
            
            async def mock_connect(*args, **kwargs):
                return mock_redis
            
            MockRedis.side_effect = mock_connect
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.ft = MagicMock()
            mock_redis.ft.return_value.create_index = AsyncMock()
            mock_redis.ft.return_value.info = AsyncMock(side_effect=Exception("No index"))
            
            await store.connect_async()
            assert store.async_redis is not None
            
            # Create indexes
            await store.create_indexes()
            
            # Store document
            doc = VectorDocument(
                id="test",
                content="content",
                embedding=np.array([1, 2, 3]),
                metadata={},
                hierarchy_level=1
            )
            
            mock_redis.hset = AsyncMock()
            await store.store_document(doc)
            mock_redis.hset.assert_called()
    
    def test_vector_document_dataclass(self):
        """Test VectorDocument dataclass."""
        doc = VectorDocument(
            id="test",
            content="content",
            embedding=np.array([1, 2, 3]),
            metadata={"key": "val"},
            hierarchy_level=2,
            parent_id="parent",
            children_ids=["child1"]
        )
        
        doc_dict = asdict(doc)
        assert doc_dict["id"] == "test"
        assert doc_dict["parent_id"] == "parent"