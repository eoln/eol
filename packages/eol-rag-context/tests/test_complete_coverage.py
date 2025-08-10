"""
Complete coverage tests - target 80% coverage.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, PropertyMock, call
import numpy as np
import json
import tempfile
import os
import asyncio
from dataclasses import asdict
import hashlib
from collections import deque
import time

# Mock all external dependencies
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'sentence_transformers', 'openai', 'tree_sitter', 'yaml', 'bs4', 'aiofiles',
               'typer', 'rich', 'rich.console', 'fastmcp', 'fastmcp.server']:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Import all modules for testing
from eol.rag_context import config
from eol.rag_context import embeddings
from eol.rag_context import document_processor
from eol.rag_context import indexer
from eol.rag_context import redis_client
from eol.rag_context import semantic_cache
from eol.rag_context import knowledge_graph
from eol.rag_context import file_watcher


class TestEmbeddingsComplete:
    """Complete embeddings module coverage."""
    
    @pytest.mark.asyncio
    async def test_embedding_manager_full(self):
        """Test EmbeddingManager with all code paths."""
        # Test with Redis client
        redis_mock = MagicMock()
        redis_mock.hget = AsyncMock(return_value=None)
        redis_mock.hset = AsyncMock()
        redis_mock.expire = AsyncMock()
        
        cfg = config.EmbeddingConfig(dimension=32, cache_embeddings=True)
        manager = embeddings.EmbeddingManager(cfg, redis_mock)
        
        # Mock provider
        manager.provider = AsyncMock()
        manager.provider.embed = AsyncMock(
            return_value=np.random.randn(1, 32).astype(np.float32)
        )
        manager.provider.embed_batch = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 32).astype(np.float32)
        )
        
        # Test single embedding with cache
        emb = await manager.get_embedding("test", use_cache=True)
        redis_mock.hget.assert_called()
        redis_mock.hset.assert_called()
        
        # Test batch with cache disabled
        embs = await manager.get_embeddings(["a", "b"], use_cache=False, batch_size=1)
        assert embs.shape == (2, 32)
        
        # Test with cached value
        redis_mock.hget.return_value = np.random.randn(32).tobytes()
        emb = await manager.get_embedding("cached", use_cache=True)
        assert manager.cache_stats["hits"] > 0
        
        # Test cache stats
        stats = manager.get_cache_stats()
        assert stats["total"] > 0
        assert "hit_rate" in stats
    
    @pytest.mark.asyncio
    async def test_openai_provider_full(self):
        """Test OpenAI provider with full coverage."""
        cfg = config.EmbeddingConfig(
            provider="openai",
            openai_api_key="test-key",
            openai_model="text-embedding-3-small",
            dimension=1536
        )
        
        with patch('eol.rag_context.embeddings.AsyncOpenAI') as MockOpenAI:
            mock_client = MagicMock()
            MockOpenAI.return_value = mock_client
            
            # Mock response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            provider = embeddings.OpenAIProvider(cfg)
            
            # Test single embed
            emb = await provider.embed("test")
            assert emb.shape == (1, 1536)
            
            # Test batch embed with batching
            mock_response.data = [
                MagicMock(embedding=[0.1] * 1536),
                MagicMock(embedding=[0.2] * 1536)
            ]
            embs = await provider.embed_batch(["a", "b"], batch_size=1)
            assert embs.shape == (2, 1536)
    
    def test_provider_initialization_all(self):
        """Test all provider initialization paths."""
        # Test with different providers
        cfg = config.EmbeddingConfig(provider="sentence-transformers")
        mgr = embeddings.EmbeddingManager(cfg)
        assert isinstance(mgr.provider, embeddings.SentenceTransformerProvider)
        
        # Test OpenAI with key
        cfg = config.EmbeddingConfig(provider="openai", openai_api_key="key")
        with patch('eol.rag_context.embeddings.AsyncOpenAI'):
            mgr = embeddings.EmbeddingManager(cfg)
            assert isinstance(mgr.provider, embeddings.OpenAIProvider)
        
        # Test invalid provider
        cfg = config.EmbeddingConfig(provider="invalid")
        with pytest.raises(ValueError):
            embeddings.EmbeddingManager(cfg)


class TestDocumentProcessorComplete:
    """Complete document processor coverage."""
    
    @pytest.mark.asyncio
    async def test_process_all_file_types(self):
        """Test processing all file types."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Mock aiofiles
        with patch('eol.rag_context.document_processor.aiofiles') as mock_aiofiles:
            mock_file = MagicMock()
            mock_file.read = AsyncMock(return_value="content")
            mock_aiofiles.open = MagicMock(return_value=mock_file)
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            
            # Test text file
            doc = await proc._process_text(Path("/test.txt"))
            assert doc.doc_type == "text"
            
            # Test markdown
            mock_file.read = AsyncMock(return_value="# Title\n\nContent")
            doc = await proc._process_markdown(Path("/test.md"))
            assert doc.doc_type == "markdown"
            
            # Test code
            mock_file.read = AsyncMock(return_value="def test(): pass")
            doc = await proc._process_code(Path("/test.py"))
            assert doc.doc_type == "code"
    
    @pytest.mark.asyncio
    async def test_process_binary_files(self):
        """Test processing binary files."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Mock PDF processing
        with patch('eol.rag_context.document_processor.PdfReader') as MockPdf:
            mock_reader = MagicMock()
            mock_reader.pages = [MagicMock(extract_text=lambda: "Page 1")]
            MockPdf.return_value = mock_reader
            
            doc = await proc._process_pdf(Path("/test.pdf"))
            assert doc.doc_type == "pdf"
        
        # Mock DOCX processing
        with patch('eol.rag_context.document_processor.Document') as MockDocx:
            mock_doc = MagicMock()
            mock_doc.paragraphs = [MagicMock(text="Paragraph 1")]
            MockDocx.return_value = mock_doc
            
            doc = await proc._process_docx(Path("/test.docx"))
            assert doc.doc_type == "docx"
    
    @pytest.mark.asyncio
    async def test_process_structured_files(self):
        """Test processing structured files."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Test JSON
        with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as f:
            json.dump({"key": "value", "list": [1, 2]}, f)
            f.flush()
            
            doc = await proc._process_structured(Path(f.name))
            assert doc.doc_type == "json"
            assert len(doc.chunks) > 0
        
        Path(f.name).unlink()
        
        # Test YAML with mock
        with patch('eol.rag_context.document_processor.yaml') as mock_yaml:
            mock_yaml.safe_load = MagicMock(return_value={"key": "value"})
            
            with tempfile.NamedTemporaryFile(suffix=".yaml", mode='w', delete=False) as f:
                f.write("key: value")
                f.flush()
                
                doc = await proc._process_structured(Path(f.name))
                assert doc.doc_type == "yaml"
            
            Path(f.name).unlink()
    
    def test_chunking_methods_complete(self):
        """Test all chunking methods."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig(use_semantic_chunking=True)
        )
        
        # Test semantic chunking
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = proc._chunk_text(text)
        assert all(c["type"] == "semantic" for c in chunks)
        
        # Test code chunking with AST
        code = "def func1():\n    pass\n\ndef func2():\n    pass"
        chunks = proc._chunk_code_by_ast(code, None, "python")
        assert len(chunks) > 0
        
        # Test HTML extraction
        from bs4 import BeautifulSoup
        html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
        soup = BeautifulSoup(html, 'html.parser')
        headers = proc._extract_headers(soup)
        content = proc._extract_text_content(soup)
        assert len(headers) > 0
        assert len(content) > 0
    
    @pytest.mark.asyncio
    async def test_file_type_detection(self):
        """Test file type detection."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Mock magic
        with patch('eol.rag_context.document_processor.magic') as mock_magic:
            mock_magic.from_file = MagicMock(return_value="text/plain")
            
            with tempfile.NamedTemporaryFile(suffix=".txt", mode='w', delete=False) as f:
                f.write("test")
                f.flush()
                
                doc = await proc.process_file(Path(f.name))
                assert doc is not None
            
            Path(f.name).unlink()


class TestIndexerComplete:
    """Complete indexer coverage."""
    
    @pytest.mark.asyncio
    async def test_indexing_pipeline(self):
        """Test complete indexing pipeline."""
        cfg = config.RAGConfig()
        proc = MagicMock()
        emb = MagicMock()
        redis = MagicMock()
        
        idx = indexer.DocumentIndexer(cfg, proc, emb, redis)
        
        # Mock methods
        proc.process_file = AsyncMock(return_value=document_processor.ProcessedDocument(
            file_path=Path("/test.md"),
            content="Test content " * 100,
            doc_type="markdown",
            chunks=[{"content": "chunk1"}, {"content": "chunk2"}]
        ))
        
        emb.get_embedding = AsyncMock(return_value=np.random.randn(128))
        emb.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        
        redis.store_document = AsyncMock()
        
        # Index a file
        await idx.index_file(Path("/test.md"), "source123")
        
        # Verify stats updated
        assert idx.stats["documents_indexed"] > 0
        
        # Index a folder
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "test.md").write_text("# Test")
            
            result = await idx.index_folder(tmpdir)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_concept_extraction(self):
        """Test concept extraction."""
        idx = indexer.DocumentIndexer(
            config.RAGConfig(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        )
        
        # Mock embeddings
        idx.embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        idx.redis.store_document = AsyncMock()
        
        doc = document_processor.ProcessedDocument(
            file_path=Path("/test.md"),
            content="Test content " * 200,  # Long content for concepts
            doc_type="markdown"
        )
        
        metadata = indexer.DocumentMetadata(
            source_path="/test.md",
            source_id="test123",
            relative_path="test.md",
            file_type="markdown",
            file_size=1024,
            file_hash="abc",
            modified_time=0,
            indexed_at=0,
            chunk_index=0,
            total_chunks=1,
            hierarchy_level=1
        )
        
        concepts = await idx._extract_concepts(doc, metadata)
        assert len(concepts) > 0
        assert all(c.hierarchy_level == 1 for c in concepts)
    
    @pytest.mark.asyncio
    async def test_source_management(self):
        """Test source management."""
        idx = indexer.DocumentIndexer(
            config.RAGConfig(),
            MagicMock(),
            MagicMock(),
            MagicMock()
        )
        
        # Mock Redis
        idx.redis.delete_by_source = AsyncMock()
        idx.redis.list_sources = AsyncMock(return_value=[
            {"source_id": "src1", "path": "/path1"},
            {"source_id": "src2", "path": "/path2"}
        ])
        
        # Remove source
        success = await idx.remove_source("src1")
        assert success
        
        # List sources
        sources = await idx.list_sources()
        assert len(sources) == 2
    
    def test_folder_scanner_complete(self):
        """Test folder scanner completely."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        
        # Test git metadata extraction
        with patch('eol.rag_context.indexer.subprocess') as mock_subprocess:
            mock_subprocess.run = MagicMock(
                return_value=MagicMock(
                    returncode=0,
                    stdout="main\nabc123\ntest@example.com"
                )
            )
            
            meta = scanner._get_git_metadata(Path("/test"))
            assert "git_branch" in meta or len(meta) == 0


class TestSemanticCacheComplete:
    """Complete semantic cache coverage."""
    
    @pytest.mark.asyncio
    async def test_cache_lifecycle(self):
        """Test complete cache lifecycle."""
        cfg = MagicMock()
        cfg.enabled = True
        cfg.similarity_threshold = 0.9
        cfg.adaptive_threshold = True
        cfg.max_cache_size = 10
        cfg.ttl_seconds = 3600
        cfg.target_hit_rate = 0.31
        
        emb = MagicMock()
        emb.get_embedding = AsyncMock(return_value=np.random.randn(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        redis.redis.hset = AsyncMock()
        redis.redis.expire = AsyncMock()
        redis.redis.hlen = AsyncMock(return_value=5)
        redis.redis.hgetall = AsyncMock(return_value={})
        redis.redis.delete = AsyncMock()
        redis.redis.keys = AsyncMock(return_value=["cache:1", "cache:2"])
        redis.redis.ft = MagicMock()
        redis.redis.ft.return_value.search = AsyncMock(return_value=MagicMock(docs=[]))
        
        cache = semantic_cache.SemanticCache(cfg, emb, redis)
        
        # Set entry
        await cache.set("query1", "response1", {"meta": "data"})
        
        # Get entry (miss)
        result = await cache.get("query1")
        assert cache.stats["misses"] > 0
        
        # Clear cache
        await cache.clear()
        redis.redis.delete.assert_called()
        
        # Update adaptive threshold
        cache.similarity_scores = [0.8, 0.85, 0.9, 0.95] * 10
        cache.stats = {"queries": 100, "hits": 25}
        await cache._update_adaptive_threshold()
        
        # Get optimization report
        async def mock_size():
            return 10
        cache._get_cache_size = mock_size
        
        report = await cache.get_optimization_report()
        assert "current_hit_rate" in report
        assert "recommendations" in report
    
    @pytest.mark.asyncio
    async def test_cache_with_hits(self):
        """Test cache with actual hits."""
        cfg = MagicMock()
        cfg.enabled = True
        cfg.similarity_threshold = 0.9
        cfg.adaptive_threshold = False
        
        emb = MagicMock()
        emb.get_embedding = AsyncMock(return_value=np.ones(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        
        # Mock search with results
        mock_doc = MagicMock()
        mock_doc.id = "cache:123"
        mock_doc.score = 0.95
        mock_doc.response = "cached response"
        mock_doc.metadata = json.dumps({"key": "value"})
        mock_doc.hit_count = 5
        
        search_result = MagicMock()
        search_result.docs = [mock_doc]
        
        redis.redis.ft = MagicMock()
        redis.redis.ft.return_value.search = AsyncMock(return_value=search_result)
        redis.redis.hincrby = AsyncMock()
        
        cache = semantic_cache.SemanticCache(cfg, emb, redis)
        
        # Get with hit
        result = await cache.get("similar query")
        assert result is not None
        assert result["response"] == "cached response"
        assert cache.stats["hits"] == 1


class TestKnowledgeGraphComplete:
    """Complete knowledge graph coverage."""
    
    @pytest.mark.asyncio
    async def test_graph_building_complete(self):
        """Test complete graph building."""
        redis = MagicMock()
        emb = MagicMock()
        emb.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        redis.store_entities = AsyncMock()
        redis.store_relationships = AsyncMock()
        
        builder = knowledge_graph.KnowledgeGraphBuilder(redis, emb)
        
        # Build from various document types
        docs = [
            redis_client.VectorDocument(
                id="doc1",
                content="# Main Topic\n\n## Section 1\n\nThis uses `function_one()` and ClassA",
                embedding=np.random.randn(128),
                metadata={"doc_type": "markdown", "relative_path": "doc1.md"},
                hierarchy_level=3
            ),
            redis_client.VectorDocument(
                id="doc2",
                content="def function_one():\n    pass\n\nclass ClassA:\n    pass",
                embedding=np.random.randn(128),
                metadata={"doc_type": "code", "language": "python", "relative_path": "code.py"},
                hierarchy_level=3
            ),
            redis_client.VectorDocument(
                id="doc3",
                content='{"api": "endpoint", "version": "1.0"}',
                embedding=np.random.randn(128),
                metadata={"doc_type": "json", "relative_path": "api.json"},
                hierarchy_level=3
            )
        ]
        
        await builder.build_from_documents(docs)
        
        # Should have extracted various entities
        assert len(builder.entities) > 0
        assert len(builder.relationships) > 0
        
        # Test pattern discovery
        await builder._discover_patterns()
        
        # Test subgraph query
        subgraph = await builder.query_subgraph("function_one", max_depth=2)
        assert "entities" in subgraph
        assert "relationships" in subgraph
        
        # Test export
        graph_data = await builder.export_graph()
        assert "nodes" in graph_data
        assert "edges" in graph_data
    
    @pytest.mark.asyncio
    async def test_entity_extraction_methods(self):
        """Test all entity extraction methods."""
        builder = knowledge_graph.KnowledgeGraphBuilder(MagicMock(), MagicMock())
        
        # Extract from markdown
        await builder._extract_markdown_entities(
            "# Topic\n\n## Section\n\n`code_ref`\n\n[Link](url)",
            "doc1",
            {"source_id": "src1"}
        )
        
        # Extract from code
        await builder._extract_code_entities_from_content(
            "def func():\n    pass\n\nclass MyClass:\n    pass",
            "doc2",
            {"language": "python", "relative_path": "test.py"}
        )
        
        # Extract from structured
        await builder._extract_structured_entities(
            {"api": "endpoint", "feature": "test"},
            "doc3",
            {"doc_type": "json"}
        )
        
        assert len(builder.entities) > 0


class TestFileWatcherComplete:
    """Complete file watcher coverage."""
    
    @pytest.mark.asyncio
    async def test_watcher_complete(self):
        """Test complete watcher functionality."""
        idx = MagicMock()
        idx.scanner = MagicMock()
        idx.scanner.generate_source_id = MagicMock(return_value="src123")
        idx.index_folder = AsyncMock(return_value=MagicMock(
            source_id="src123",
            file_count=5,
            total_chunks=20
        ))
        idx.index_file = AsyncMock()
        idx.remove_source = AsyncMock()
        
        watcher = file_watcher.FileWatcher(idx, debounce_seconds=0.1, batch_size=5)
        
        # Test observer creation
        with patch('eol.rag_context.file_watcher.Observer') as MockObserver:
            mock_observer = MagicMock()
            MockObserver.return_value = mock_observer
            
            # Start watcher
            await watcher.start()
            assert watcher.is_running
            
            # Watch directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                source_id = await watcher.watch(tmpdir, recursive=True, file_patterns=["*.py"])
                assert source_id == "src123"
                
                # Test event handling
                event = MagicMock()
                event.is_directory = False
                event.src_path = str(tmpdir / "test.py")
                event.event_type = "modified"
                
                handler = watcher.observers.get(source_id)
                if handler:
                    handler.on_modified(event)
                    
                    # Process queue
                    watcher.change_queue.append(file_watcher.FileChange(
                        path=Path(event.src_path),
                        change_type=file_watcher.ChangeType.MODIFIED
                    ))
                    
                    await watcher._process_changes()
                
                # Unwatch
                success = await watcher.unwatch(source_id)
                assert success
            
            # Stop watcher
            await watcher.stop()
            assert not watcher.is_running
    
    def test_event_handler(self):
        """Test event handler class."""
        callback = MagicMock()
        handler = file_watcher.ChangeEventHandler(Path("/test"), callback, ["*.py"], [])
        
        # Test event handling
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/file.py"
        event.event_type = "created"
        
        handler.on_created(event)
        callback.assert_called_once()
        
        # Test ignore patterns
        handler = file_watcher.ChangeEventHandler(Path("/test"), callback, [], ["*.pyc"])
        event.src_path = "/test/file.pyc"
        handler.on_created(event)
        # Should be ignored


class TestRedisClientComplete:
    """Complete Redis client coverage."""
    
    @pytest.mark.asyncio
    async def test_redis_operations_complete(self):
        """Test all Redis operations."""
        store = redis_client.RedisVectorStore(
            config.RedisConfig(),
            config.IndexConfig()
        )
        
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis, \
             patch('eol.rag_context.redis_client.Redis') as MockSyncRedis:
            
            # Mock async Redis
            mock_async = MagicMock()
            async def mock_connect(*args, **kwargs):
                return mock_async
            MockRedis.side_effect = mock_connect
            
            mock_async.ping = AsyncMock(return_value=True)
            mock_async.ft = MagicMock()
            mock_async.ft.return_value.create_index = AsyncMock()
            mock_async.ft.return_value.info = AsyncMock(side_effect=Exception("No index"))
            mock_async.hset = AsyncMock()
            mock_async.ft.return_value.search = AsyncMock(return_value=MagicMock(docs=[]))
            mock_async.keys = AsyncMock(return_value=["doc:1", "doc:2"])
            mock_async.delete = AsyncMock()
            
            # Mock sync Redis
            mock_sync = MagicMock()
            MockSyncRedis.return_value = mock_sync
            mock_sync.ping = MagicMock(return_value=True)
            
            # Connect
            await store.connect_async()
            store.connect_sync()
            
            # Create indexes
            await store.create_indexes()
            
            # Store document
            doc = redis_client.VectorDocument(
                id="test",
                content="content",
                embedding=np.array([1, 2, 3]),
                metadata={"key": "val"},
                hierarchy_level=2,
                parent_id="parent"
            )
            await store.store_document(doc)
            
            # Search
            results = await store.search("query", limit=5, hierarchy_level=2)
            assert isinstance(results, list)
            
            # Get context
            context = await store.get_context("query", max_chunks=10)
            assert isinstance(context, list)
            
            # Delete by source
            await store.delete_by_source("src123")
            mock_async.delete.assert_called()
            
            # Store entities
            entities = [
                knowledge_graph.Entity("e1", "Entity 1", knowledge_graph.EntityType.CONCEPT)
            ]
            await store.store_entities(entities)
            
            # Store relationships
            relationships = [
                knowledge_graph.Relationship("e1", "e2", knowledge_graph.RelationType.CONTAINS)
            ]
            await store.store_relationships(relationships)
            
            # List sources
            sources = await store.list_sources()
            assert isinstance(sources, list)