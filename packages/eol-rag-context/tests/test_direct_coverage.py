"""
Direct coverage tests - directly exercise code paths without complex mocking.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, ANY
import numpy as np
import json
import tempfile
import os
import asyncio
from dataclasses import asdict, fields
import hashlib

# Mock external dependencies
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'fastmcp', 'fastmcp.server', 'sentence_transformers', 'openai',
               'tree_sitter', 'tree_sitter_python', 'pypdf.PdfReader', 'yaml',
               'bs4', 'aiofiles']:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Import modules
import eol.rag_context.config as config
import eol.rag_context.embeddings as embeddings
import eol.rag_context.document_processor as document_processor
import eol.rag_context.indexer as indexer
import eol.rag_context.redis_client as redis_client
import eol.rag_context.semantic_cache as semantic_cache
import eol.rag_context.knowledge_graph as knowledge_graph
import eol.rag_context.file_watcher as file_watcher


# Direct config tests
def test_config_coverage():
    """Direct config module coverage."""
    # Test all config classes
    r = config.RedisConfig()
    assert r.host == "localhost"
    
    e = config.EmbeddingConfig()
    assert e.provider == "sentence-transformers"
    
    i = config.IndexConfig()
    assert i.name == "rag_context_index"
    
    ch = config.ChunkingConfig()
    assert ch.max_chunk_size == 1024
    
    ca = config.CacheConfig()
    assert ca.enabled is True
    
    ct = config.ContextConfig()
    assert ct.max_context_length == 128000
    
    d = config.DocumentConfig()
    assert d.max_file_size_mb == 100
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ra = config.RAGConfig(
            data_dir=Path(tmpdir) / "data",
            index_dir=Path(tmpdir) / "index"
        )
        assert ra.data_dir.exists()


# Direct embeddings tests
@pytest.mark.asyncio
async def test_embeddings_coverage():
    """Direct embeddings module coverage."""
    # Base provider
    base = embeddings.EmbeddingProvider()
    with pytest.raises(NotImplementedError):
        await base.embed("test")
    with pytest.raises(NotImplementedError):
        await base.embed_batch(["test"])
    
    # Mock provider
    mock_cfg = config.EmbeddingConfig(dimension=32)
    mock_prov = embeddings.MockEmbeddingsProvider(mock_cfg)
    emb = await mock_prov.embed("test")
    assert emb.shape == (1, 32)
    embs = await mock_prov.embed_batch(["a", "b"])
    assert embs.shape == (2, 32)
    
    # Sentence transformer provider
    st_cfg = config.EmbeddingConfig(model_name="test", dimension=64)
    st_prov = embeddings.SentenceTransformerProvider(st_cfg)
    assert st_prov.model is None  # Not installed
    emb = await st_prov.embed("test")
    assert emb.shape == (1, 64)
    
    # Manager
    mgr = embeddings.EmbeddingManager(mock_cfg)
    assert mgr.config == mock_cfg
    
    # Cache key
    key = mgr._cache_key("test")
    assert key.startswith("emb:")
    
    # Stats
    stats = mgr.get_cache_stats()
    assert stats["hit_rate"] == 0.0


# Direct document processor tests
@pytest.mark.asyncio
async def test_document_processor_coverage():
    """Direct document processor module coverage."""
    proc = document_processor.DocumentProcessor(
        config.DocumentConfig(),
        config.ChunkingConfig()
    )
    
    # Language detection
    assert proc._detect_language(".py") == "python"
    assert proc._detect_language(".js") == "javascript"
    assert proc._detect_language(".unknown") == "unknown"
    
    # Text chunking
    chunks = proc._chunk_text("short text")
    assert len(chunks) > 0
    
    # Markdown chunking
    md_chunks = proc._chunk_markdown_by_headers("# Title\n\nContent")
    assert len(md_chunks) > 0
    
    # Code chunking
    code_chunks = proc._chunk_code_by_lines("def test():\n    pass", "python")
    assert len(code_chunks) > 0
    
    # Structured data chunking
    struct_chunks = proc._chunk_structured_data({"key": "value"}, "json")
    assert len(struct_chunks) > 0
    
    # Process file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode='w', delete=False) as f:
        f.write("test content")
        f.flush()
        doc = await proc.process_file(Path(f.name))
        assert doc is not None
    Path(f.name).unlink()


# Direct indexer tests
@pytest.mark.asyncio
async def test_indexer_coverage():
    """Direct indexer module coverage."""
    # Folder scanner
    scanner = indexer.FolderScanner(config.RAGConfig())
    
    # Source ID
    sid = scanner.generate_source_id(Path("/test"))
    assert len(sid) == 16
    
    # Ignore patterns
    patterns = scanner._default_ignore_patterns()
    assert "**/.git/**" in patterns
    
    # Should ignore
    assert scanner._should_ignore(Path(".git/config"))
    assert not scanner._should_ignore(Path("main.py"))
    
    # Git metadata
    meta = scanner._get_git_metadata(Path("/nonexistent"))
    assert meta == {}
    
    # Document indexer
    idx = indexer.DocumentIndexer(
        config.RAGConfig(),
        MagicMock(),
        MagicMock(),
        MagicMock()
    )
    
    # Summary
    assert idx._generate_summary("short") == "short"
    long_text = "word " * 200
    summary = idx._generate_summary(long_text)
    assert len(summary) <= 500
    
    # Stats
    stats = idx.get_stats()
    assert stats["documents_indexed"] == 0
    
    # Scan folder
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.py").write_text("code")
        files = await scanner.scan_folder(tmpdir)
        assert len(files) > 0


# Direct semantic cache tests
@pytest.mark.asyncio
async def test_semantic_cache_coverage():
    """Direct semantic cache module coverage."""
    cfg = MagicMock()
    cfg.similarity_threshold = 0.9
    cfg.enabled = True
    cfg.adaptive_threshold = False
    cfg.max_cache_size = 100
    cfg.ttl_seconds = 3600
    cfg.target_hit_rate = 0.31
    
    cache = semantic_cache.SemanticCache(cfg, MagicMock(), MagicMock())
    
    # Stats
    stats = cache.get_stats()
    assert stats["queries"] == 0
    
    # Update stats
    cache.stats = {"queries": 100, "hits": 31, "misses": 69}
    stats = cache.get_stats()
    assert stats["hit_rate"] == 0.31
    
    # Cached query dataclass
    query = semantic_cache.CachedQuery(
        query="test",
        response="response",
        embedding=np.array([1, 2]),
        hit_count=0
    )
    assert query.query == "test"


# Direct knowledge graph tests
@pytest.mark.asyncio
async def test_knowledge_graph_coverage():
    """Direct knowledge graph module coverage."""
    builder = knowledge_graph.KnowledgeGraphBuilder(MagicMock(), MagicMock())
    
    # Entity
    entity = knowledge_graph.Entity(
        id="e1",
        name="Entity",
        type=knowledge_graph.EntityType.CONCEPT
    )
    assert entity.id == "e1"
    
    # Relationship
    rel = knowledge_graph.Relationship(
        source_id="s1",
        target_id="t1",
        type=knowledge_graph.RelationType.CONTAINS
    )
    assert rel.source_id == "s1"
    
    # Stats
    stats = builder.get_graph_stats()
    assert stats["entity_count"] == 0
    
    # Add entities
    builder.entities["e1"] = entity
    stats = builder.get_graph_stats()
    assert stats["entity_count"] == 1


# Direct file watcher tests
@pytest.mark.asyncio
async def test_file_watcher_coverage():
    """Direct file watcher module coverage."""
    watcher = file_watcher.FileWatcher(MagicMock())
    
    # File change
    change = file_watcher.FileChange(
        path=Path("/test.py"),
        change_type=file_watcher.ChangeType.MODIFIED
    )
    assert change.path == Path("/test.py")
    
    # Watched source
    source = file_watcher.WatchedSource(
        path=Path("/src"),
        source_id="src123",
        recursive=True
    )
    assert source.source_id == "src123"
    
    # Stats
    stats = watcher.get_stats()
    assert stats["is_running"] is False
    
    # Callbacks
    def cb(change): pass
    watcher.add_change_callback(cb)
    assert len(watcher.change_callbacks) == 1
    watcher.remove_change_callback(cb)
    assert len(watcher.change_callbacks) == 0
    
    # Change history
    watcher.change_history.append(change)
    history = watcher.get_change_history()
    assert len(history) == 1


# Direct Redis client tests
@pytest.mark.asyncio
async def test_redis_client_coverage():
    """Direct Redis client module coverage."""
    store = redis_client.RedisVectorStore(
        config.RedisConfig(),
        config.IndexConfig()
    )
    
    assert store.redis is None
    assert store.async_redis is None
    
    # Vector document
    doc = redis_client.VectorDocument(
        id="doc1",
        content="content",
        embedding=np.array([1, 2, 3]),
        metadata={},
        hierarchy_level=1
    )
    assert doc.id == "doc1"
    
    # Connect with mock
    with patch('eol.rag_context.redis_client.AsyncRedis') as MockRedis:
        mock_redis = MagicMock()
        
        async def mock_connect(*args, **kwargs):
            return mock_redis
        
        MockRedis.side_effect = mock_connect
        mock_redis.ping = AsyncMock(return_value=True)
        
        await store.connect_async()
        assert store.async_redis is not None


# Run all direct tests
@pytest.mark.asyncio
async def test_all_direct_coverage():
    """Run all direct coverage tests."""
    test_config_coverage()
    await test_embeddings_coverage()
    await test_document_processor_coverage()
    await test_indexer_coverage()
    await test_semantic_cache_coverage()
    await test_knowledge_graph_coverage()
    await test_file_watcher_coverage()
    await test_redis_client_coverage()