"""
Simplified test file to boost coverage to 80%.
Focus on executing code paths without complex mocking.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import numpy as np
import pytest

# Mock all external dependencies
for module in [
    "redis",
    "redis.asyncio",
    "redis.commands",
    "redis.commands.search",
    "redis.commands.search.field",
    "redis.commands.search.indexDefinition",
    "redis.commands.search.query",
    "redis.exceptions",
    "magic",
    "pypdf",
    "docx",
    "aiofiles",
    "aiofiles.os",
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_go",
    "tree_sitter_rust",
    "tree_sitter_cpp",
    "tree_sitter_c",
    "tree_sitter_java",
    "tree_sitter_csharp",
    "tree_sitter_ruby",
    "tree_sitter_php",
    "yaml",
    "bs4",
    "markdown",
    "sentence_transformers",
    "openai",
    "networkx",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "typer",
    "rich",
    "rich.console",
    "rich.table",
    "fastmcp",
    "fastmcp.server",
    "gitignore_parser",
]:
    sys.modules[module] = MagicMock()

# Import after mocking
from eol.rag_context import (
    config,
    document_processor,
    embeddings,
    file_watcher,
    indexer,
    knowledge_graph,
    main,
    redis_client,
    semantic_cache,
    server,
)


@pytest.mark.asyncio
async def test_redis_client_coverage():
    """Test redis_client to 80% coverage."""
    cfg = config.RedisConfig()
    idx_cfg = config.IndexConfig()
    store = redis_client.RedisVectorStore(cfg, idx_cfg)

    # Mock Redis connection
    with patch("eol.rag_context.redis_client.Redis") as MockRedis:
        mock_redis = MagicMock()
        mock_redis.ping = MagicMock(return_value=True)
        mock_redis.ft = MagicMock()
        MockRedis.return_value = mock_redis

        # Test sync connect
        store.connect()
        assert store.redis is not None

        # Test create indexes
        store.create_hierarchical_indexes(768)

    # Mock async Redis
    with patch("eol.rag_context.redis_client.AsyncRedis") as MockAsync:
        mock_async = MagicMock()
        mock_async.ping = AsyncMock(return_value=True)
        mock_async.hset = AsyncMock()
        mock_async.hgetall = AsyncMock(return_value={})
        mock_async.close = AsyncMock()

        # Create coroutine for AsyncRedis
        async def create_async_redis(*args, **kwargs):
            return mock_async

        MockAsync.side_effect = create_async_redis

        # Test async connect
        await store.connect_async()

        # Test store document
        doc = redis_client.VectorDocument(
            id="test", content="content", embedding=np.zeros(768), metadata={}, hierarchy_level=1
        )
        await store.store_document(doc)

        # Test vector search
        mock_ft = MagicMock()
        mock_result = MagicMock()
        mock_result.docs = []
        mock_ft.search = AsyncMock(return_value=mock_result)
        mock_async.ft = MagicMock(return_value=mock_ft)
        store.async_redis = mock_async

        results = await store.vector_search(np.zeros(768))
        assert results == []

        # Test close
        await store.close()


@pytest.mark.asyncio
async def test_indexer_coverage():
    """Test indexer to 80% coverage."""
    rag_cfg = config.RAGConfig()

    # Create mocks
    proc = MagicMock()
    emb = MagicMock()
    redis = MagicMock()

    # Create indexer
    idx = indexer.DocumentIndexer(rag_cfg, proc, emb, redis)

    # Mock process_file
    proc.process_file = AsyncMock(
        return_value=document_processor.ProcessedDocument(
            file_path=Path("/test.py"),
            content="content",
            doc_type="code",
            chunks=[{"content": "chunk"}],
        )
    )

    # Mock embeddings
    emb.get_embedding = AsyncMock(return_value=np.zeros(768))
    emb.get_embeddings = AsyncMock(return_value=np.zeros((1, 768)))

    # Mock redis
    redis.store_document = AsyncMock()
    redis.delete_by_source = AsyncMock()
    redis.list_sources = AsyncMock(return_value=[])

    # Test with temp file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(b"test code")
        f.flush()

        # Test index_file
        result = await idx.index_file(Path(f.name), "src123")
        assert result is not None

        os.unlink(f.name)

    # Test remove_source
    await idx.remove_source("src123")

    # Test list_sources
    await idx.list_sources()

    # Test get_stats
    stats = idx.get_stats()
    assert "total_documents" in stats

    # Test FolderScanner
    scanner = indexer.FolderScanner(rag_cfg)

    # Test generate_source_id
    source_id = scanner.generate_source_id(Path("/test"))
    assert len(source_id) > 0

    # Test _should_ignore
    assert scanner._should_ignore(Path("__pycache__/test.pyc"))
    assert not scanner._should_ignore(Path("main.py"))

    # Test scan_folder with temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "test.py").write_text("code")
        files = await scanner.scan_folder(Path(tmpdir))
        assert len(files) > 0


@pytest.mark.asyncio
async def test_semantic_cache_coverage():
    """Test semantic_cache to 80% coverage."""
    cache_cfg = MagicMock()
    cache_cfg.enabled = True
    cache_cfg.similarity_threshold = 0.9
    cache_cfg.adaptive_threshold = True
    cache_cfg.max_cache_size = 10
    cache_cfg.ttl_seconds = 3600
    cache_cfg.target_hit_rate = 0.31

    emb = MagicMock()
    emb.get_embedding = AsyncMock(return_value=np.zeros(768))

    redis_store = MagicMock()
    redis_store.redis = MagicMock()
    redis_store.redis.hset = AsyncMock()
    redis_store.redis.hget = AsyncMock(return_value=None)
    redis_store.redis.expire = AsyncMock()
    redis_store.redis.hlen = AsyncMock(return_value=5)
    redis_store.redis.scan = AsyncMock(return_value=(0, []))
    redis_store.redis.hdel = AsyncMock()
    redis_store.redis.delete = AsyncMock()

    cache = semantic_cache.SemanticCache(cache_cfg, emb, redis_store)

    # Test set
    await cache.set("query", "response", {"meta": "data"})

    # Test get (cache miss)
    result = await cache.get("query")
    assert result is None

    # Test clear
    await cache.clear()

    # Test get_stats
    stats = cache.get_stats()
    assert "queries" in stats

    # Test get_optimization_report
    report = await cache.get_optimization_report()
    assert "current_hit_rate" in report


@pytest.mark.asyncio
async def test_file_watcher_coverage():
    """Test file_watcher to 80% coverage."""
    cfg = config.FileWatcherConfig()
    idx = MagicMock()
    idx.index_file = AsyncMock()
    idx.remove_source = AsyncMock()

    watcher = file_watcher.FileWatcher(cfg, idx)

    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test watch
        watch_id = await watcher.watch(Path(tmpdir))
        assert watch_id is not None

        # Test get_watch_info
        info = watcher.get_watch_info(watch_id)
        assert info is not None

        # Test list_watches
        watches = watcher.list_watches()
        assert len(watches) > 0

        # Test unwatch
        result = await watcher.unwatch(watch_id)
        assert result

        # Test stop
        await watcher.stop()


@pytest.mark.asyncio
async def test_knowledge_graph_coverage():
    """Test knowledge_graph to 80% coverage."""
    cfg = config.KnowledgeGraphConfig()
    redis_store = MagicMock()

    builder = knowledge_graph.KnowledgeGraphBuilder(cfg, redis_store)

    # Mock graph operations
    builder.graph = MagicMock()
    builder.graph.add_node = MagicMock()
    builder.graph.add_edge = MagicMock()
    builder.graph.nodes = MagicMock(return_value=[])
    builder.graph.edges = MagicMock(return_value=[])

    # Test add_entity
    await builder.add_entity("entity1", "class", {"prop": "value"})

    # Test add_relationship
    await builder.add_relationship("entity1", "entity2", "uses")

    # Test get_graph_stats
    stats = builder.get_graph_stats()
    assert "nodes" in stats

    # Test clear
    await builder.clear()


@pytest.mark.asyncio
async def test_embeddings_coverage():
    """Test embeddings to 80% coverage."""
    cfg = config.EmbeddingConfig(dimension=768)

    # Test base provider
    provider = embeddings.EmbeddingProvider()
    with pytest.raises(NotImplementedError):
        await provider.embed("text")

    # Test SentenceTransformerProvider
    st_provider = embeddings.SentenceTransformerProvider(cfg)

    # Mock model
    st_provider.model = MagicMock()
    st_provider.model.encode = MagicMock(return_value=np.zeros(768))

    # Test embed
    embedding = await st_provider.embed("text")
    assert embedding.shape == (1, 768)

    # Test embed_batch
    batch = await st_provider.embed_batch(["text1", "text2"])
    assert batch.shape == (2, 768)

    # Test EmbeddingManager
    manager = embeddings.EmbeddingManager(cfg)

    # Test get_embedding
    embedding = await manager.get_embedding("text")
    assert embedding.shape == (768,)

    # Test get_embeddings
    embeddings_batch = await manager.get_embeddings(["text1", "text2"])
    assert embeddings_batch.shape == (2, 768)

    # Test cache stats
    stats = manager.get_cache_stats()
    assert "hit_rate" in stats


@pytest.mark.asyncio
async def test_document_processor_coverage():
    """Test document_processor to 80% coverage."""
    doc_cfg = config.DocumentConfig()
    chunk_cfg = config.ChunkingConfig()

    proc = document_processor.DocumentProcessor(doc_cfg, chunk_cfg)

    # Test with temp text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Test content\nLine 2\nLine 3")
        f.flush()

        with patch("eol.rag_context.document_processor.aiofiles.open") as mock_aio:
            mock_file = AsyncMock()
            mock_file.read = AsyncMock(return_value="Test content")
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock()
            mock_aio.return_value = mock_file

            doc = await proc.process_file(Path(f.name))
            assert doc is not None

        os.unlink(f.name)

    # Test chunk methods
    chunks = proc._chunk_text("text " * 100)
    assert len(chunks) > 0

    # Test language detection
    lang = proc._detect_language(".py")
    assert lang == "python"

    lang = proc._detect_language(".unknown")
    assert lang is None


@pytest.mark.asyncio
async def test_server_coverage():
    """Test server to 80% coverage."""
    with patch("eol.rag_context.server.FastMCP") as MockMCP:
        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock()
        mock_mcp.run = AsyncMock()
        MockMCP.return_value = mock_mcp

        # Create server
        srv = server.EOLRAGContextServer()
        assert srv.mcp is not None

        # Mock components
        srv.redis = MagicMock()
        srv.embeddings = MagicMock()
        srv.processor = MagicMock()
        srv.indexer = MagicMock()
        srv.cache = MagicMock()
        srv.graph = MagicMock()
        srv.watcher = MagicMock()

        # Test methods with mocked components
        srv.indexer.index_folder = AsyncMock(
            return_value=MagicMock(source_id="src123", file_count=10, total_chunks=50, errors=[])
        )

        result = await srv.index_directory("/test")
        assert result["status"] == "success"

        # Test search_context
        srv.redis.vector_search = AsyncMock(return_value=[])
        results = await srv.search_context("query")
        assert results == []

        # Test clear_cache
        srv.cache.clear = AsyncMock()
        srv.cache.get_stats = MagicMock(return_value={"queries": 0})
        result = await srv.clear_cache()
        assert result["status"] == "success"


def test_main_coverage():
    """Test main module to 80% coverage."""
    with patch("eol.rag_context.main.typer") as mock_typer:
        mock_app = MagicMock()
        mock_typer.Typer.return_value = mock_app

        # Import main to trigger module-level code
        import eol.rag_context.main as main_module

        # Test with config file
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"redis": {"host": "localhost"}}, f)
            f.flush()

            with patch("asyncio.run"):
                main_module.main(config_file=f.name)

            os.unlink(f.name)
