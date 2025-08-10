"""
Boost test coverage to 80% with focused unit tests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import numpy as np
import json
import tempfile
import os
import hashlib
from dataclasses import asdict

# Mock all external dependencies
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'fastmcp', 'sentence_transformers', 'openai']:
    sys.modules[module] = MagicMock()

# Import after mocking
from eol.rag_context import config
from eol.rag_context import embeddings
from eol.rag_context import document_processor
from eol.rag_context import indexer
from eol.rag_context import redis_client
from eol.rag_context import semantic_cache
from eol.rag_context import knowledge_graph
from eol.rag_context import file_watcher


class TestEmbeddingsModule:
    """Test embeddings module for coverage."""
    
    def test_embedding_provider_interface(self):
        """Test base embedding provider."""
        provider = embeddings.EmbeddingProvider()
        
        # Should have abstract methods
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.embed("test"))
        
        with pytest.raises(NotImplementedError):
            import asyncio
            asyncio.run(provider.embed_batch(["test"]))
    
    @pytest.mark.asyncio
    async def test_mock_embeddings_provider(self):
        """Test mock embeddings provider."""
        cfg = config.EmbeddingConfig(dimension=64)
        provider = embeddings.MockEmbeddingsProvider(cfg)
        
        # Single embedding
        emb = await provider.embed("test")
        assert emb.shape == (1, 64)
        assert emb.dtype == np.float32
        
        # Batch embeddings
        embs = await provider.embed_batch(["a", "b", "c"])
        assert embs.shape == (3, 64)
        assert embs.dtype == np.float32
    
    def test_embedding_manager_init(self):
        """Test embedding manager initialization."""
        cfg = config.EmbeddingConfig()
        manager = embeddings.EmbeddingManager(cfg)
        
        assert manager.config == cfg
        assert manager.cache_stats["hits"] == 0
        assert manager.cache_stats["misses"] == 0
        assert manager.cache_stats["total"] == 0
    
    def test_embedding_manager_cache_key(self):
        """Test cache key generation."""
        manager = embeddings.EmbeddingManager(config.EmbeddingConfig())
        
        key1 = manager._cache_key("test")
        key2 = manager._cache_key("test")
        key3 = manager._cache_key("other")
        
        assert key1 == key2
        assert key1 != key3
        assert key1.startswith("emb:")
        
        # Hash consistency
        h = hashlib.sha256("test".encode()).hexdigest()[:16]
        assert key1 == f"emb:{h}"
    
    def test_embedding_manager_stats(self):
        """Test embedding manager statistics."""
        manager = embeddings.EmbeddingManager(config.EmbeddingConfig())
        
        # Initial stats
        stats = manager.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Update stats
        manager.cache_stats["hits"] = 31
        manager.cache_stats["misses"] = 69
        manager.cache_stats["total"] = 100
        
        stats = manager.get_cache_stats()
        assert stats["hit_rate"] == 0.31


class TestDocumentProcessorModule:
    """Test document processor module for coverage."""
    
    def test_processor_init(self):
        """Test processor initialization."""
        doc_cfg = config.DocumentConfig()
        chunk_cfg = config.ChunkingConfig()
        
        processor = document_processor.DocumentProcessor(doc_cfg, chunk_cfg)
        
        assert processor.doc_config == doc_cfg
        assert processor.chunk_config == chunk_cfg
    
    def test_detect_language(self):
        """Test language detection."""
        processor = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Test various extensions
        assert processor._detect_language(".py") == "python"
        assert processor._detect_language(".js") == "javascript"
        assert processor._detect_language(".java") == "java"
        assert processor._detect_language(".unknown") == "unknown"
    
    def test_chunk_text_simple(self):
        """Test simple text chunking."""
        chunk_cfg = config.ChunkingConfig()
        chunk_cfg.use_semantic_chunking = False
        chunk_cfg.max_chunk_size = 20
        chunk_cfg.chunk_overlap = 5
        
        processor = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            chunk_cfg
        )
        
        text = "This is a test. " * 10
        chunks = processor._chunk_text(text)
        
        assert len(chunks) > 0
        assert all("content" in c for c in chunks)
        assert all("type" in c for c in chunks)
    
    def test_chunk_markdown_headers(self):
        """Test markdown chunking by headers."""
        processor = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        md = "# H1\nContent 1\n## H2\nContent 2"
        chunks = processor._chunk_markdown_by_headers(md)
        
        assert len(chunks) == 2
        assert chunks[0]["header"] == "H1"
        assert chunks[1]["header"] == "H2"
    
    def test_chunk_code_lines(self):
        """Test code chunking by lines."""
        chunk_cfg = config.ChunkingConfig()
        chunk_cfg.code_max_lines = 5
        
        processor = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            chunk_cfg
        )
        
        code = "\n".join([f"line {i}" for i in range(20)])
        chunks = processor._chunk_code_by_lines(code, "python")
        
        assert len(chunks) > 0
        assert all(c["language"] == "python" for c in chunks)
        assert all("start_line" in c for c in chunks)
    
    def test_extract_headers(self):
        """Test HTML header extraction."""
        from bs4 import BeautifulSoup
        
        processor = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        soup = BeautifulSoup(html, 'html.parser')
        headers = processor._extract_headers(soup)
        
        assert len(headers) == 2
        assert headers[0]["level"] == 1
        assert headers[0]["text"] == "Title"
    
    def test_chunk_structured_data(self):
        """Test structured data chunking."""
        processor = document_processor.DocumentProcessor(
            config.DocumentConfig(),
            config.ChunkingConfig()
        )
        
        # Dict chunking
        data = {"key1": "val1", "key2": "val2"}
        chunks = processor._chunk_structured_data(data, "json")
        assert len(chunks) == 2
        assert all(c["type"] == "object_field" for c in chunks)
        
        # List chunking
        data = ["item1", "item2"]
        chunks = processor._chunk_structured_data(data, "json")
        assert len(chunks) == 2
        assert all(c["type"] == "array_item" for c in chunks)


class TestIndexerModule:
    """Test indexer module for coverage."""
    
    def test_folder_scanner_init(self):
        """Test folder scanner initialization."""
        cfg = config.RAGConfig()
        scanner = indexer.FolderScanner(cfg)
        
        assert scanner.config == cfg
    
    def test_folder_scanner_source_id(self):
        """Test source ID generation."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        
        id1 = scanner.generate_source_id(Path("/test"))
        id2 = scanner.generate_source_id(Path("/test"))
        id3 = scanner.generate_source_id(Path("/other"))
        
        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16
    
    def test_folder_scanner_ignore_patterns(self):
        """Test default ignore patterns."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        patterns = scanner._default_ignore_patterns()
        
        assert "**/.git/**" in patterns
        assert "**/node_modules/**" in patterns
        assert "**/__pycache__/**" in patterns
    
    def test_folder_scanner_should_ignore(self):
        """Test file ignore logic."""
        scanner = indexer.FolderScanner(config.RAGConfig())
        
        assert scanner._should_ignore(Path(".git/config"))
        assert scanner._should_ignore(Path("node_modules/pkg.json"))
        assert not scanner._should_ignore(Path("src/main.py"))
    
    def test_document_indexer_init(self):
        """Test document indexer initialization."""
        cfg = config.RAGConfig()
        proc = MagicMock()
        emb = MagicMock()
        redis = MagicMock()
        
        idx = indexer.DocumentIndexer(cfg, proc, emb, redis)
        
        assert idx.config == cfg
        assert idx.processor == proc
        assert idx.embeddings == emb
        assert idx.redis == redis
    
    def test_indexer_generate_summary(self):
        """Test summary generation."""
        idx = indexer.DocumentIndexer(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        
        # Short content
        assert idx._generate_summary("short") == "short"
        
        # Long content
        long = "word " * 200
        summary = idx._generate_summary(long)
        assert len(summary) <= 500
    
    def test_indexer_stats(self):
        """Test indexer statistics."""
        idx = indexer.DocumentIndexer(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        
        idx.stats = {
            "documents_indexed": 10,
            "chunks_created": 50,
            "errors": 2
        }
        
        stats = idx.get_stats()
        assert stats["documents_indexed"] == 10
        assert stats["chunks_created"] == 50


class TestSemanticCacheModule:
    """Test semantic cache module for coverage."""
    
    def test_cache_init(self):
        """Test cache initialization."""
        cfg = MagicMock()
        cfg.similarity_threshold = 0.9
        
        cache = semantic_cache.SemanticCache(cfg, MagicMock(), MagicMock())
        
        assert cache.config == cfg
        assert cache.adaptive_threshold == 0.9
        assert len(cache.similarity_scores) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = semantic_cache.SemanticCache(
            MagicMock(), MagicMock(), MagicMock()
        )
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["queries"] == 0
        assert stats["hits"] == 0
        assert stats["hit_rate"] == 0.0
        
        # Update stats
        cache.stats = {
            "queries": 100,
            "hits": 31,
            "misses": 69
        }
        
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.31
    
    @pytest.mark.asyncio
    async def test_cache_update_threshold(self):
        """Test adaptive threshold update."""
        cfg = MagicMock()
        cfg.target_hit_rate = 0.31
        cfg.adaptive_threshold = True
        cfg.similarity_threshold = 0.9
        
        cache = semantic_cache.SemanticCache(cfg, MagicMock(), MagicMock())
        
        # Add similarity scores
        cache.similarity_scores = [0.8, 0.85, 0.9, 0.95] * 25
        cache.stats = {"queries": 100, "hits": 25}
        
        await cache._update_adaptive_threshold()
        
        # Should have adjusted threshold
        assert cache.adaptive_threshold != 0.9


class TestKnowledgeGraphModule:
    """Test knowledge graph module for coverage."""
    
    def test_graph_builder_init(self):
        """Test graph builder initialization."""
        builder = knowledge_graph.KnowledgeGraphBuilder(
            MagicMock(), MagicMock()
        )
        
        assert len(builder.entities) == 0
        assert len(builder.relationships) == 0
        assert builder.graph is not None
    
    def test_graph_stats(self):
        """Test graph statistics."""
        builder = knowledge_graph.KnowledgeGraphBuilder(
            MagicMock(), MagicMock()
        )
        
        # Add entities
        builder.entities = {
            "1": knowledge_graph.Entity("1", "E1", knowledge_graph.EntityType.CONCEPT),
            "2": knowledge_graph.Entity("2", "E2", knowledge_graph.EntityType.FUNCTION),
        }
        
        # Add relationships
        builder.relationships = [
            knowledge_graph.Relationship("1", "2", knowledge_graph.RelationType.CONTAINS)
        ]
        
        stats = builder.get_graph_stats()
        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1


class TestFileWatcherModule:
    """Test file watcher module for coverage."""
    
    def test_watcher_init(self):
        """Test watcher initialization."""
        watcher = file_watcher.FileWatcher(MagicMock())
        
        assert not watcher.is_running
        assert len(watcher.watched_sources) == 0
        assert len(watcher.change_callbacks) == 0
    
    def test_watcher_callbacks(self):
        """Test callback management."""
        watcher = file_watcher.FileWatcher(MagicMock())
        
        def cb1(change): pass
        def cb2(change): pass
        
        watcher.add_change_callback(cb1)
        watcher.add_change_callback(cb2)
        assert len(watcher.change_callbacks) == 2
        
        watcher.remove_change_callback(cb1)
        assert len(watcher.change_callbacks) == 1
    
    def test_watcher_change_history(self):
        """Test change history."""
        watcher = file_watcher.FileWatcher(MagicMock())
        
        # Add changes
        for i in range(10):
            change = file_watcher.FileChange(
                path=Path(f"/test{i}.py"),
                change_type=file_watcher.ChangeType.MODIFIED
            )
            watcher.change_history.append(change)
        
        # Get history
        history = watcher.get_change_history(limit=5)
        assert len(history) == 5
        
        history = watcher.get_change_history()
        assert len(history) == 10
    
    def test_watcher_stats(self):
        """Test watcher statistics."""
        watcher = file_watcher.FileWatcher(MagicMock())
        
        watcher.stats = {
            "changes_detected": 10,
            "changes_processed": 8,
            "errors": 1
        }
        
        stats = watcher.get_stats()
        assert stats["changes_detected"] == 10
        assert stats["changes_processed"] == 8


class TestRedisClientModule:
    """Test Redis client module for coverage."""
    
    def test_vector_store_init(self):
        """Test vector store initialization."""
        redis_cfg = config.RedisConfig()
        index_cfg = config.IndexConfig()
        
        store = redis_client.RedisVectorStore(redis_cfg, index_cfg)
        
        assert store.redis_config == redis_cfg
        assert store.index_config == index_cfg
        assert store.redis is None
        assert store.async_redis is None
    
    @pytest.mark.asyncio
    async def test_vector_store_connect(self):
        """Test async connection."""
        store = redis_client.RedisVectorStore(
            config.RedisConfig(),
            config.IndexConfig()
        )
        
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis:
            mock_redis = MagicMock()
            
            async def mock_connect(*args, **kwargs):
                return mock_redis
            
            MockRedis.side_effect = mock_connect
            mock_redis.ping = AsyncMock(return_value=True)
            
            await store.connect_async()
            
            assert store.async_redis is not None